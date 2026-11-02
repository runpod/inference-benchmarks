"""CPU-hosted sweep wrapper. The originating database remains the import authority."""

import io
import json
import os
import signal
import sqlite3
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from . import sweeps
from .controller import SSH
from .db import Database, encode

ROOT = "/workspace/sls-controller"
PYTHON = f"{ROOT}/.venv/bin/python"
REMOTE_DB = f"{ROOT}/data/benchmarks.sqlite3"
SCHEMA = """
CREATE TABLE IF NOT EXISTS remote_controllers (
 sweep_id TEXT PRIMARY KEY REFERENCES sweeps(id), pod_name TEXT NOT NULL UNIQUE,
 pod_id TEXT, ssh_key TEXT NOT NULL, state TEXT NOT NULL, created_at REAL NOT NULL,
 pulled_at REAL, error TEXT
);
"""


def setup(db):
    db.conn.executescript(SCHEMA)


def record(db, sweep_id):
    setup(db)
    row = db.conn.execute("SELECT * FROM remote_controllers WHERE sweep_id=?", (sweep_id,)).fetchone()
    if row is None:
        job = db.conn.execute("SELECT options FROM sweeps WHERE id=?", (sweep_id,)).fetchone()
        owner = json.loads(job[0]).get("fleet_owner") if job else None
        if owner and owner != sweep_id:
            row = db.conn.execute("SELECT * FROM remote_controllers WHERE sweep_id=?", (owner,)).fetchone()
    return dict(row) if row else None


def update(db, sweep_id, **values):
    with db.conn:
        db.conn.execute(
            "UPDATE remote_controllers SET " + ",".join(f"{k}=?" for k in values) + " WHERE sweep_id=?",
            [*values.values(), sweep_id],
        )


def connection(db, api, sweep_id):
    row = record(db, sweep_id)
    if not row or row["state"] == "deleted":
        raise ValueError("No retained remote controller for this sweep")
    if not row["pod_id"]:
        matches = [p for p in api.pods(row["pod_name"]) if p.get("name") == row["pod_name"]]
        if len(matches) != 1:
            raise ValueError("CPU creation unresolved; inspect provider by recorded name; do not relaunch")
        update(db, sweep_id, pod_id=matches[0]["id"])
        row = record(db, sweep_id)
    return SSH(api.pod(row["pod_id"]), row["ssh_key"], db.path.parent / "known_hosts")


def bundle(db, sweep_id, budget, parallel, frozen_sources=False):
    """Stage only code, configs and this sweep, never unrelated local runs/secrets."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "benchmarks.sqlite3"
        seed = Database(path)
        with seed.conn:
            for table in ("configs", "sweeps", "sweep_items"):
                query = f"SELECT * FROM {table}"
                args = ()
                if table == "sweeps":
                    query += " WHERE id=?"
                    args = (sweep_id,)
                elif table == "sweep_items":
                    query += " WHERE sweep_id=?"
                    args = (sweep_id,)
                for row in db.conn.execute(query, args):
                    values = dict(row)
                    if table == "sweeps":
                        options = json.loads(values["options"])
                        options.update(ssh_key=f"{ROOT}/data/controller_key", parallel=parallel)
                        values.update(
                            options=encode(options), approved_budget=budget, status="queued", pid=None
                        )
                    seed.conn.execute(
                        f"INSERT INTO {table} ({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
                        list(values.values()),
                    )
        seed.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        seed.conn.close()
        output = io.BytesIO()
        project = Path(__file__).resolve().parent.parent
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name in ("pyproject.toml", "uv.lock"):
                archive.add(project / name, arcname=name)
            if frozen_sources:
                from .resolve import source_hash

                specs = [json.loads(row[0]) for row in db.conn.execute(
                    "SELECT spec FROM sweep_items WHERE sweep_id=?", (sweep_id,)
                )]
                if not specs or any(spec.get("sources") != specs[0].get("sources") for spec in specs):
                    raise ValueError("Frozen staging requires one consistent source manifest")
                sources = specs[0]["sources"]
                if any(source_hash(sources) != spec.get("code_sha256") for spec in specs):
                    raise ValueError("Frozen source hash mismatch")
                for name, content in sources.items():
                    if Path(name).name != name or not name.endswith(".py"):
                        raise ValueError("Invalid frozen source path")
                    data = content.encode()
                    entry = tarfile.TarInfo(f"sls_bench/{name}")
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
            else:
                for source in sorted((project / "sls_bench").glob("*.py")):
                    archive.add(source, arcname=f"sls_bench/{source.name}")
            archive.add(path, arcname="data/benchmarks.sqlite3")
        return output.getvalue()


def launch(
    db,
    api,
    sweep_id,
    budget,
    cpu_hourly_limit,
    ssh_key,
    parallel=1,
    cpu_flavor="cpu3g",
    vcpus=2,
    volume_gb=30,
    *,
    stage_only=False,
    frozen_sources=False,
    data_centers=None,
    ready_timeout=600,
):
    job = sweeps.job(db, sweep_id)
    sweeps.validate_limits(db, sweep_id)
    previous = record(db, sweep_id)
    if previous and previous["state"] != "rejected":
        raise ValueError("Remote controller already recorded; pull/reconcile it, never relaunch blindly")
    if (
        job["status"] != "prepared"
        or db.conn.execute(
            "SELECT 1 FROM sweep_items WHERE sweep_id=? AND (run_id IS NOT NULL OR attempts!=0)", (sweep_id,)
        ).fetchone()
    ):
        raise ValueError("Remote launch requires a fresh prepared sweep")
    if json.loads(job["options"]).get("after_sweep"):
        raise ValueError("Cross-host sweep dependencies are not supported")
    if not 1 <= parallel <= 8 or budget < job["estimated_ceiling"] or cpu_hourly_limit <= 0 or ready_timeout <= 0:
        raise ValueError("Invalid parallelism or insufficient GPU/CPU budget approval")
    key = Path(ssh_key).expanduser().resolve()
    public = Path(str(key) + ".pub").read_text().strip()
    key.read_bytes()  # Validate before allocating.
    credential = os.environ.get("RUNPOD_ORGS_API_KEY") or os.environ.get("runpod_orgs_api_key")
    if not credential:
        raise ValueError("RUNPOD_ORGS_API_KEY is required")
    payload = bundle(db, sweep_id, budget, parallel, frozen_sources=frozen_sources)
    name = f"sls-controller-{sweep_id}"
    with db.conn:
        db.conn.execute(
            "INSERT INTO remote_controllers VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(sweep_id) DO UPDATE SET state='creating',error=NULL "
            "WHERE remote_controllers.state='rejected'",
            (sweep_id, name, None, str(key), "creating", time.time(), None, None),
        )
        db.conn.execute("UPDATE sweeps SET status='remote',pid=NULL WHERE id=?", (sweep_id,))
    bootstrap = """set -eu
apt-get update
apt-get install -y --no-install-recommends openssh-server ca-certificates git
mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password
"""
    try:
        pod = api.rest(
            "POST",
            "/pods",
            json={
                "name": name,
                "computeType": "CPU",
                "cloudType": "SECURE",
                "cpuFlavorIds": cpu_flavor.split(","),
                **({"dataCenterIds": list(data_centers), "dataCenterPriority": "custom"} if data_centers else {}),
                "vcpuCount": vcpus,
                "imageName": "python:3.12-bookworm",
                "containerDiskInGb": 10,
                "volumeInGb": volume_gb,
                "volumeMountPath": "/workspace",
                "ports": ["22/tcp"],
                "supportPublicIp": True,
                "dockerEntrypoint": ["/bin/bash"],
                "dockerStartCmd": ["-lc", bootstrap],
                "env": {"PUBLIC_KEY": public},
            },
        )
        update(db, sweep_id, pod_id=pod["id"], state="staging")
        deadline = time.monotonic() + ready_timeout
        while True:
            pod = api.pod(pod["id"])
            price = pod.get("adjustedCostPerHr", pod.get("costPerHr"))
            if price is not None and float(price) > cpu_hourly_limit:
                api.delete(pod["id"])
                update(db, sweep_id, state="deleted")
                raise ValueError("CPU hourly price exceeds approval; CPU deleted")
            try:
                ssh = connection(db, api, sweep_id)
                ssh.call("true")
                if price is None:
                    raise ValueError("CPU price unavailable")
                break
            except (ValueError, RuntimeError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("CPU not SSH-ready/priced; retained for reconciliation")
                time.sleep(5)
        ssh.call(f"umask 077; mkdir -p {ROOT}; tar xzf - -C {ROOT}", payload, timeout=120)
        # A dedicated worker key is generated remotely; the user's private key is never uploaded.
        ssh.call(f"ssh-keygen -q -t ed25519 -N '' -f {ROOT}/data/controller_key")
        ssh.call(f"umask 077; cat > {ROOT}/data/api_key", credential.encode())
        token = db.path.parent / "private" / "hf_token"
        if token.exists():
            ssh.call(
                f"umask 077; mkdir -p {ROOT}/data/private; cat > {ROOT}/data/private/hf_token",
                token.read_bytes(),
            )
        ssh.call(
            f"cd {ROOT} && python -m venv .venv && "
            ".venv/bin/pip install uv && .venv/bin/uv sync --frozen --no-dev",
            timeout=600,
        )
        # Sweep ID is a generated UUID, but quote it as untrusted shell input regardless.
        import shlex

        if not stage_only:
            ssh.call(
                f"cd {ROOT}; nohup {PYTHON} -m sls_bench.remote {shlex.quote(sweep_id)} "
                "> data/controller.log 2>&1 < /dev/null &"
            )
        update(db, sweep_id, state="staged" if stage_only else "running", error=None)
    except BaseException as error:
        update(db, sweep_id, error=str(error))
        # This specific provider response is a rejected allocation, not a timeout.
        # Still reconcile by exact name before allowing an explicit retry.
        current = record(db, sweep_id)
        if not current["pod_id"] and "no longer any instances available" in str(error):
            matches = [p for p in api.pods(name) if p.get("name") == name]
            if not matches:
                with db.conn:
                    db.conn.execute(
                        "UPDATE remote_controllers SET state='rejected' WHERE sweep_id=?", (sweep_id,)
                    )
                    db.conn.execute("UPDATE sweeps SET status='prepared' WHERE id=?", (sweep_id,))
        raise


def merge(db, source, sweep_id):
    """Atomic, scoped, repeatable import into the original DB; remap trial IDs."""
    remote = sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True)
    remote.row_factory = sqlite3.Row
    try:
        job = remote.execute("SELECT * FROM sweeps WHERE id=?", (sweep_id,)).fetchone()
        if not job:
            raise ValueError("Snapshot does not contain expected sweep")
        items = remote.execute("SELECT * FROM sweep_items WHERE sweep_id=?", (sweep_id,)).fetchall()
        with db.conn:
            for config in remote.execute("SELECT * FROM configs"):
                local = db.conn.execute("SELECT * FROM configs WHERE id=?", (config["id"],)).fetchone()
                if not local or dict(local) != dict(config):
                    raise ValueError("Snapshot belongs to a different config lineage")
            for item in items:
                local = db.conn.execute("SELECT * FROM sweep_items WHERE id=?", (item["id"],)).fetchone()
                if not local or any(
                    local[k] != item[k] for k in ("sweep_id", "spec", "target_config_id", "ordinal")
                ):
                    raise ValueError("Sweep item identity mismatch")

            def upsert(table, row, keys):
                values = dict(row)
                columns = list(values)
                changes = [c for c in columns if c not in keys]
                db.conn.execute(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
                    f"ON CONFLICT({','.join(keys)}) DO UPDATE SET "
                    + ",".join(f"{c}=excluded.{c}" for c in changes),
                    list(values.values()),
                )

            runs = remote.execute(
                "SELECT * FROM runs WHERE json_extract(spec,'$.sweep_id')=?", (sweep_id,)
            ).fetchall()
            for run in runs:
                existing = db.conn.execute("SELECT spec FROM runs WHERE id=?", (run["id"],)).fetchone()
                if existing and existing[0] != run["spec"]:
                    raise ValueError("Run identity mismatch")
                # Immutable run snapshot columns cannot be included in UPDATE clauses.
                if not existing:
                    cols = list(run.keys())
                    db.conn.execute(
                        f"INSERT INTO runs ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
                        tuple(run),
                    )
                else:
                    db.conn.execute(
                        "UPDATE runs SET status=?,finished_at=?,error=?,hourly_price=?,estimated_cost=? WHERE id=?",
                        tuple(
                            run[k]
                            for k in (
                                "status",
                                "finished_at",
                                "error",
                                "hourly_price",
                                "estimated_cost",
                                "id",
                            )
                        ),
                    )
                for table, keys in (
                    ("resources", ["run_id"]),
                    ("hardware", ["run_id"]),
                    ("artifacts", ["run_id", "path"]),
                ):
                    for row in remote.execute(f"SELECT * FROM {table} WHERE run_id=?", (run["id"],)):
                        upsert(table, row, keys)
                for row in remote.execute("SELECT * FROM run_configs WHERE run_id=?", (run["id"],)):
                    db.conn.execute("INSERT OR IGNORE INTO run_configs VALUES(?,?)", tuple(row))
                for trial in remote.execute("SELECT * FROM trials WHERE run_id=?", (run["id"],)):
                    values = dict(trial)
                    old_id = values.pop("id")
                    upsert("trials", values, ["run_id", "ordinal"])
                    new_id = db.conn.execute(
                        "SELECT id FROM trials WHERE run_id=? AND ordinal=?", (run["id"], trial["ordinal"])
                    ).fetchone()[0]
                    for table, keys in (
                        ("metrics", ["trial_id", "name"]),
                        ("requests", ["trial_id", "ordinal"]),
                    ):
                        for row in remote.execute(f"SELECT * FROM {table} WHERE trial_id=?", (old_id,)):
                            upsert(table, {**dict(row), "trial_id": new_id}, keys)
                for event in remote.execute("SELECT * FROM events WHERE run_id=?", (run["id"],)):
                    values = tuple(event[k] for k in ("run_id", "at", "kind", "details"))
                    db.conn.execute(
                        "INSERT INTO events(run_id,at,kind,details) SELECT ?,?,?,? WHERE NOT EXISTS "
                        "(SELECT 1 FROM events WHERE run_id=? AND at=? AND kind=? AND details=?)",
                        values + values,
                    )
            for item in items:
                db.conn.execute(
                    "UPDATE sweep_items SET status=?,reason=?,run_id=?,attempts=? WHERE id=?",
                    tuple(item[k] for k in ("status", "reason", "run_id", "attempts", "id")),
                )
            db.conn.execute(
                "UPDATE sweeps SET status=?,approved_budget=?,heartbeat_at=?,finished_at=?,error=?,pid=NULL WHERE id=?",
                tuple(
                    job[k]
                    for k in ("status", "approved_budget", "heartbeat_at", "finished_at", "error", "id")
                ),
            )
        return job["status"]
    finally:
        remote.close()


def pull(db, api, sweep_id):
    ssh = connection(db, api, sweep_id)
    snapshot = f"{ROOT}/snapshot-{uuid.uuid4().hex}.sqlite3"
    ssh.call(f"cd {ROOT} && {PYTHON} -m sls_bench.cli --db {REMOTE_DB} backup {snapshot}", timeout=120)
    data = ssh.call(f"cat {snapshot}", timeout=300)
    ssh.call(f"rm -f {snapshot}")
    folder = db.path.parent / "remote" / sweep_id
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "snapshot.tmp"
    temporary.write_bytes(data)
    status = merge(db, temporary, sweep_id)
    temporary.replace(folder / "snapshot.sqlite3")
    (folder / "controller.log").write_bytes(ssh.call(f"tail -c 10000000 {ROOT}/data/controller.log"))
    update(
        db,
        sweep_id,
        pulled_at=time.time(),
        state="synced" if status in ("completed", "partial") else "running",
    )
    return status


def release(db, api, sweep_id):
    owner = record(db, sweep_id)
    if owner and owner["sweep_id"] != sweep_id:
        raise ValueError("Release the fleet owner, not an individual diagnostic sweep")
    status = pull(db, api, sweep_id)
    unresolved = db.conn.execute(
        "SELECT 1 FROM resources p JOIN runs r ON p.run_id=r.id WHERE json_extract(r.spec,'$.sweep_id')=? "
        "AND p.state NOT IN ('deleted','detached')",
        (sweep_id,),
    ).fetchone()
    if status not in ("completed", "partial") or unresolved:
        raise ValueError("CPU retained: sweep is not terminal or GPU cleanup is unresolved")
    api.delete(record(db, sweep_id)["pod_id"])
    update(db, sweep_id, state="deleted")


def resume(db, api, sweep_id):
    """Invoke the same dispatcher again; its existing flock/recovery prevents duplicates."""
    import shlex

    owner = record(db, sweep_id)
    if owner and owner["sweep_id"] != sweep_id:
        raise ValueError("Resume the fleet owner, not an individual diagnostic sweep")
    options = json.loads(sweeps.job(db, sweep_id)["options"])
    entrypoint = options.get("controller_entrypoint")
    if entrypoint not in (None, "data/image-fleet-controller.py"):
        raise ValueError("Unsupported controller entrypoint")
    command = entrypoint or f"-m sls_bench.remote {shlex.quote(sweep_id)}"
    ssh = connection(db, api, sweep_id)
    ssh.call(
        f"cd {ROOT}; nohup {PYTHON} {command} "
        ">> data/controller.log 2>&1 < /dev/null &"
    )
    update(db, sweep_id, state="running", error=None)


def worker(sweep_id):
    # Background shells/nohup can inherit SIGINT=SIG_IGN. Restore interruption
    # so the unchanged sweep dispatcher can stop its threads and clean GPU Pods.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    os.environ["RUNPOD_ORGS_API_KEY"] = Path(f"{ROOT}/data/api_key").read_text().strip()
    sweeps.run(REMOTE_DB, sweep_id)


if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
