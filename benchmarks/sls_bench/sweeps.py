"""Durable, budgeted sweep plans. Preparing a plan never creates a Pod."""

import fcntl
import json
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .config import MAX_CONCURRENCY, validate
from .controller import SSH, cleanup, collect, execute
from .db import Database, digest, encode
from .matrix import materialize as materialize_matrix
from .registry import TYPES, sync_types, type_for
from .resolve import resolve
from .runpod import Runpod
from .targets import hydrate_target, materialize_target

TERMINAL = {"completed", "partial", "failed", "unavailable", "timed_out", "interrupted", "blocked"}


def config_row(db, selector, kind=None):
    name, separator, revision = selector.partition("@")
    return db.config(name, kind, int(revision) if separator else None)


def workloads_for(
    db, suite=None, workload_names=None, benchmark_type=None, model=None, parameters=None, matrix=None
):
    db.seed()
    sync_types(db)
    if matrix:
        if suite or workload_names or model or parameters:
            raise ValueError(
                "--matrix is authoritative; edit its SQLite revision instead of combining overrides"
            )
        row, workloads = materialize_matrix(db, matrix)
        if benchmark_type and json.loads(row["spec"])["benchmark_type"] != benchmark_type:
            raise ValueError("Matrix has the wrong benchmark type")
        return row, workloads
    if benchmark_type and benchmark_type not in TYPES:
        raise ValueError(f"Unknown benchmark type {benchmark_type}; use `bench types`")
    suite_row = None
    if not workload_names:
        suite = suite or (TYPES[benchmark_type].default_suite if benchmark_type else "text")
        if not suite:
            raise ValueError("This type has no default suite/adapter; save explicit workload configs first")
        suite_row = config_row(db, suite, "suite")
        workload_names = json.loads(suite_row["spec"])["workloads"]
    rows = []
    for name in workload_names:
        row = config_row(db, name)
        value = validate(json.loads(row["spec"]))
        category = type_for(value)
        if benchmark_type and category.key != benchmark_type:
            raise ValueError(f"{name} is {category.key}, not {benchmark_type}")
        if model or parameters:
            overrides = dict(parameters or {})
            if set(overrides) & {"kind", "name", "benchmark_type"}:
                raise ValueError("Parameters cannot change workload identity or benchmark type")
            if model:
                overrides["model"] = model
            suffix = digest(encode(overrides).encode())[:8]
            value.update(overrides)
            value["name"] = value["name"][:65] + "-" + suffix
            row = db.save_config(value)
        rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Suite contains duplicate workloads")
    return suite_row, rows


def assess(gpu, workload, target):
    value = validate(json.loads(workload["spec"]))
    category = type_for(value)
    if value.get("backend") == "diffusers":
        if value.get("readiness") != "ready":
            return "blocked", f"Explicit Diffusers readiness gate: {value['readiness']}"
        if value.get("precision") == "fp8" and gpu.get("gpuGeneration") == "AMPERE":
            return "blocked", "Native FP8 unsupported on Ampere; no weight-only fallback"
    if not category.adapter:
        return "blocked", f"No adapter implemented for {category.key}"
    if not gpu.get("memoryInGb") or gpu["memoryInGb"] <= 0:
        return "blocked", "GPU catalog has missing/invalid VRAM metadata"
    # Runtime compatibility comes from vendor metadata, not an allowlist of GPU SKUs.
    if str(gpu.get("manufacturer", "")).casefold() != "nvidia":
        return "blocked", "Default runtime is CUDA; this vendor needs a separately validated runtime image"
    placement = workload.get("matrix_placement")
    if placement:
        if placement["gpu_count"] != target["gpu_count"]:
            return (
                "blocked",
                f"Matrix row requires {placement['gpu_count']} device(s) / TP{placement['gpu_count']}",
            )
        if gpu["memoryInGb"] < placement["min_device_vram_gb"]:
            return "blocked", f"Row requires at least {placement['min_device_vram_gb']} GB per device"
        if placement["require_full_gpu"] and target.get("require_mig", "MIG" in gpu.get("id", "").upper()):
            return "blocked", "This TP row requires full GPUs, not MIG slices"
        generations = placement["allowed_gpu_generations"]
        if generations and gpu.get("gpuGeneration") not in generations:
            return (
                "blocked",
                f"Requires catalog generation in {generations}; unknown generations remain unvalidated",
            )
    total_vram = gpu["memoryInGb"] * (target["gpu_count"] if category.key == "text_gen" else 1)
    if value.get("min_vram_gb", 1) > total_vram:
        return (
            "blocked",
            f"Profile estimates {value['min_vram_gb']} GB minimum total VRAM; allocation has {total_vram} GB",
        )
    if category.key == "image_gen" and target["gpu_count"] != 1:
        return "blocked", "Image adapter supports one GPU per replica"
    return "eligible_unvalidated", "Catalog preflight only; actual device/runtime fit is verified on Pod"


def prepare(
    db,
    api,
    gpu_types=None,
    all_gpus=False,
    targets=None,
    suite=None,
    workload_names=None,
    benchmark_type=None,
    model=None,
    parameters=None,
    matrix=None,
    max_minutes=45,
    retries=1,
    validate_image_diagnostics=False,
    ssh_key="~/.ssh/id_ed25519",
    **target_options,
):
    if not all_gpus and not gpu_types and not targets:
        raise ValueError("Specify --gpu, --target, or --all-gpus")
    if all_gpus and (gpu_types or targets):
        raise ValueError("--all-gpus cannot be combined with --gpu/--target")
    suite_row, workloads = workloads_for(db, suite, workload_names, benchmark_type, model, parameters, matrix)
    catalog = {g["id"]: g for g in api.gpus()}
    with db.conn:
        db.conn.executemany(
            "INSERT INTO gpu_catalog VALUES(?,?,?) ON CONFLICT(gpu_type) DO UPDATE SET "
            "discovered_at=excluded.discovered_at,metadata=excluded.metadata",
            [(g["id"], time.time(), encode(g)) for g in catalog.values()],
        )
    chosen = list(catalog) if all_gpus else list(gpu_types or [])
    if matrix and target_options.get("gpu_count", 1) != 1:
        raise ValueError("Matrix rows determine GPU/TP counts; edit the matrix rather than --gpu-count")
    counts = (
        sorted({w["matrix_placement"]["gpu_count"] for w in workloads})
        if matrix
        else [target_options.get("gpu_count", 1)]
    )
    rows = [
        materialize_target(db, name, catalog, **{**target_options, "gpu_count": count})
        for name in chosen
        for count in counts
    ]
    rows.extend(hydrate_target(config_row(db, name, "target"), catalog) for name in targets or [])
    rows = list({r["id"]: r for r in rows}.values())
    # Resolve common model/container/code revisions once, not once per GPU.
    resolvable = [w for w in workloads if type_for(json.loads(w["spec"])).adapter]
    baseline = resolve(rows[0], resolvable) if resolvable else None
    job_id = uuid.uuid4().hex
    options = {
        "max_minutes": max_minutes,
        "capacity_retries": retries,
        "ssh_key": str(Path(ssh_key).expanduser().resolve()),
        "host": platform.node(),
        "gpu_count": len(rows),
        "all_gpus": all_gpus,
        "matrix_config_id": suite_row["id"] if matrix else None,
        "validate_image_diagnostics": validate_image_diagnostics,
    }
    items, estimate = [], 0
    for ordinal, row in enumerate(rows):
        target = json.loads(row["spec"])
        gpu = catalog[target["gpu_type"]]
        assessments = []
        eligible = []
        for workload in workloads:
            status, reason = assess(gpu, workload, target)
            assessments.append(
                {"config_id": workload["id"], "name": workload["name"], "status": status, "reason": reason}
            )
            if status == "eligible_unvalidated":
                eligible.append(workload["id"])
        price = gpu.get("securePrice" if target["cloud"] == "SECURE" else "communityPrice")
        reason = None
        if price is None or price <= 0:
            reason = "No positive catalog price in selected cloud"
        if not eligible:
            reason = "No eligible workload profiles; inspect per-profile assessments"
        ceiling = (
            (float(price) * target["gpu_count"] * 1.2 + target["disk_gb"] * 0.0002)
            if price is not None
            else 0
        )
        spec = dict(baseline or {})
        if baseline:
            target["image"] = baseline["target"]["image"]
        spec.update(
            target=target,
            workloads=[w for w in (baseline or {}).get("workloads", []) if w["config_id"] in eligible],
            catalog_gpu=gpu,
            max_hourly_price=ceiling,
            max_minutes=max_minutes,
            catalog_hourly_price=price,
            gpus_per_replica=target["gpu_count"],
            replica_count=1,
        )
        if not reason:
            estimate += ceiling * max_minutes / 60
        items.append(
            (
                job_id,
                ordinal,
                row["id"],
                "blocked" if reason else "planned",
                reason,
                encode(assessments),
                encode(spec),
            )
        )
    with db.conn:
        db.conn.execute(
            "INSERT INTO sweeps(id,created_at,status,suite_config_id,options,estimated_ceiling) VALUES(?,?,?,?,?,?)",
            (
                job_id,
                time.time(),
                "prepared",
                suite_row["id"] if suite_row else None,
                encode(options),
                estimate,
            ),
        )
        db.conn.executemany(
            "INSERT INTO sweep_items(sweep_id,ordinal,target_config_id,status,reason,assessments,spec) "
            "VALUES(?,?,?,?,?,?,?)",
            items,
        )
    return job_id


def job(db, job_id):
    row = db.conn.execute("SELECT * FROM sweeps WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown sweep {job_id}")
    return dict(row)


def validate_limits(db, job_id):
    for row in db.conn.execute(
        "SELECT spec FROM sweep_items WHERE sweep_id=? AND status IN ('planned','running')", (job_id,)
    ):
        for workload in json.loads(row[0]).get("workloads", []):
            if workload.get("kind") == "text" and (
                (workload.get("saturation_max_concurrency") or 0) > MAX_CONCURRENCY
                or max(workload.get("concurrency", [1])) > MAX_CONCURRENCY
                or workload.get("max_num_seqs", 1) > MAX_CONCURRENCY
            ):
                raise ValueError(
                    f"Frozen sweep exceeds global concurrency cap {MAX_CONCURRENCY}; prepare a revised matrix"
                )


def start(db, job_id, budget, detached=True, parallel=None, after_sweep=None):
    from .remote import record

    remote = record(db, job_id)
    if remote:
        raise ValueError("Sweep is owned by a remote controller; use `bench remote pull ID`")
    row = job(db, job_id)
    validate_limits(db, job_id)
    options = json.loads(row["options"])
    if parallel is not None:
        if not 1 <= parallel <= 8:
            raise ValueError("Parallelism must be between 1 and 8")
        options["parallel"] = parallel
    if after_sweep is not None:
        if after_sweep == job_id:
            raise ValueError("Sweep cannot depend on itself")
        predecessor = job(db, after_sweep)
        # Only permit already-created predecessors; reject manually configured cycles.
        visited = {job_id}
        while predecessor:
            if predecessor["id"] in visited:
                raise ValueError("Sweep dependency cycle")
            visited.add(predecessor["id"])
            parent = json.loads(predecessor["options"]).get("after_sweep")
            predecessor = job(db, parent) if parent else None
        options["after_sweep"] = after_sweep
    if budget < row["estimated_ceiling"]:
        raise ValueError(f"Prepared estimate ${row['estimated_ceiling']:.2f} exceeds budget ${budget:.2f}")
    key = Path(options["ssh_key"])
    if not key.exists() or not Path(str(key) + ".pub").exists():
        raise ValueError("SSH private key and matching .pub file must exist")
    if row["status"] not in ("prepared", "interrupted", "queued", "running", "waiting"):
        raise ValueError("Sweep already finished or superseded; prepare a new sweep to rerun it")
    lock_path = db.path.parent / "jobs" / f"{job_id}.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Sweep already has a live dispatcher") from None
    if not db.conn.execute(
        "SELECT 1 FROM sweep_items WHERE sweep_id=? AND status IN ('planned','running')", (job_id,)
    ).fetchone():
        raise ValueError(
            "No launchable items; inspect compatibility blocks or explicitly retry unavailable items"
        )
    with db.conn:
        db.conn.execute(
            "UPDATE sweeps SET approved_budget=?,options=?,status='queued',error=NULL WHERE id=?",
            (budget, encode(options), job_id),
        )
    if not detached:
        run(db.path, job_id)
        return None
    folder = db.path.parent / "jobs"
    folder.mkdir(exist_ok=True)
    with (folder / f"{job_id}.log").open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "sls_bench.sweeps", str(db.path), job_id],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    return process.pid


def recover(db, api, run_id, ssh_key, progress):
    """Recover collection, never allocate a second Pod for an existing run."""
    row = db.run(run_id)
    if row["resource_state"] == "deleted":
        return row["status"]
    remote = f"/workspace/bench/{run_id}"
    try:
        if not row["pod_id"]:
            raise RuntimeError(
                "Interrupted before Pod identity was persisted; reconciling ownership, not reallocating"
            )
        ssh = SSH(api.pod(row["pod_id"]), ssh_key, db.path.parent / "known_hosts")
        while time.time() < row["lease_expires_at"] - 30:
            progress(f"Recovering {run_id}")
            report = collect(db, run_id, ssh, remote)
            if report["status"] in ("completed", "partial", "failed"):
                collect(db, run_id, ssh, remote, artifacts=True)
                db.update("runs", run_id, status=report["status"], finished_at=time.time())
                return report["status"]
            time.sleep(10)
        raise TimeoutError("Existing run lease expired")
    except Exception as error:
        db.update("runs", run_id, status="interrupted", error=str(error), finished_at=time.time())
        return "interrupted"
    finally:
        cleanup(db, api, run_id)


def run_item(path, job_id, item, options, value, stop):
    """One queue slot; SQLite connections and API clients are never shared across threads."""
    db = Database(path)
    api = Runpod()

    def heartbeat():
        if stop.is_set():
            raise RuntimeError("Sweep interrupted; cleaning up active Pod")

    def progress(message):
        print(message, flush=True)
        heartbeat()

    progress.heartbeat = heartbeat
    try:
        heartbeat()
        spec = json.loads(item["spec"])
        if item["run_id"]:
            status = recover(db, api, item["run_id"], options["ssh_key"], progress)
            run_id = item["run_id"]
        else:
            for attempt in range(options["capacity_retries"] + 1):
                heartbeat()
                target = dict(
                    db.conn.execute(
                        "SELECT * FROM configs WHERE id=?", (item["target_config_id"],)
                    ).fetchone()
                )
                workloads = [
                    dict(db.conn.execute("SELECT * FROM configs WHERE id=?", (w["config_id"],)).fetchone())
                    for w in spec["workloads"]
                ]
                spec.update(
                    budget_dollars=value["approved_budget"], sweep_id=job_id, sweep_item_id=item["id"]
                )
                run_id = db.new_run(target, workloads, spec, options["max_minutes"] * 60)
                with db.conn:
                    db.conn.execute(
                        "UPDATE sweep_items SET run_id=?,status='running',attempts=attempts+1 WHERE id=?",
                        (run_id, item["id"]),
                    )
                print(
                    f"{job_id}: {spec['target']['gpu_type']}; run {run_id}; attempt {attempt + 1}", flush=True
                )
                try:
                    report = execute(db, api, run_id, options["ssh_key"], progress)
                    status = report["status"]
                except Exception as error:
                    print(f"{run_id}: {error}", flush=True)
                    status = db.run(run_id)["status"]
                if db.run(run_id)["resource_state"] not in ("deleted", "detached"):
                    raise RuntimeError(
                        f"Cleanup unresolved for {run_id}; stopping all subsequent allocations"
                    )
                if status != "unavailable" or attempt == options["capacity_retries"]:
                    break
                time.sleep(min(30 * 2**attempt, 120))
        if db.run(run_id)["resource_state"] not in ("deleted", "detached"):
            raise RuntimeError(f"Cleanup unresolved for {run_id}; stopping all subsequent allocations")
        if options.get("validate_image_diagnostics") and status != "unavailable":
            # Validate retained CUDA evidence BEFORE admitting the next GPU.
            from .image_gate import validate_run

            validate_run(db, run_id, expected_sweep_id=job_id, expected_item_id=item["id"])
        with db.conn:
            db.conn.execute("UPDATE sweep_items SET status=? WHERE id=?", (status, item["id"]))
    except BaseException:
        stop.set()
        raise
    finally:
        db.conn.close()
        api.client.close()


def run(path, job_id):
    db = Database(path)
    folder = db.path.parent / "jobs"
    folder.mkdir(exist_ok=True)
    with (folder / f"{job_id}.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # An existing dispatcher owns this sweep; never duplicate allocations.
        value = job(db, job_id)
        validate_limits(db, job_id)
        if value["approved_budget"] is None or value["approved_budget"] < value["estimated_ceiling"]:
            raise ValueError("Sweep has no sufficient approved budget")
        options = json.loads(value["options"])
        with db.conn:
            db.conn.execute(
                "UPDATE sweeps SET status='running',pid=?,heartbeat_at=?,error=NULL WHERE id=?",
                (os.getpid(), time.time(), job_id),
            )

        def touch():
            with db.conn:
                db.conn.execute("UPDATE sweeps SET heartbeat_at=? WHERE id=?", (time.time(), job_id))

        def progress(message):
            print(message, flush=True)
            touch()

        progress.heartbeat = touch

        try:
            predecessor = options.get("after_sweep")
            if predecessor:
                with db.conn:
                    db.conn.execute("UPDATE sweeps SET status='waiting' WHERE id=?", (job_id,))
                progress(f"Waiting for sweep {predecessor}; no Pods allocated for {job_id}")
                while job(db, predecessor)["status"] not in ("completed", "partial"):
                    touch()
                    time.sleep(10)
                unresolved = db.conn.execute(
                    "SELECT 1 FROM resources res JOIN runs r ON res.run_id=r.id "
                    "WHERE json_extract(r.spec,'$.sweep_id')=? AND res.owned=1 "
                    "AND res.state NOT IN ('deleted','detached') LIMIT 1",
                    (predecessor,),
                ).fetchone()
                if unresolved:
                    raise RuntimeError("Predecessor cleanup unresolved; refusing follow-up allocations")
                with db.conn:
                    db.conn.execute("UPDATE sweeps SET status='running' WHERE id=?", (job_id,))
            items = db.conn.execute(
                "SELECT * FROM sweep_items WHERE sweep_id=? ORDER BY ordinal", (job_id,)
            ).fetchall()
            # Ambiguous allocations must be reconciled before admitting ANY new Pods.
            for item in items:
                if item["run_id"] and item["status"] not in TERMINAL:
                    previous = db.run(item["run_id"])
                    if previous["resource_state"] in ("creation_ambiguous", "cleanup_failed"):
                        cleanup(db, Runpod(), item["run_id"])
            stop = threading.Event()
            pending = iter(dict(i) for i in items if i["status"] not in TERMINAL)
            pool = ThreadPoolExecutor(max_workers=options.get("parallel", 1))
            active = set()
            try:

                def submit_next():
                    item = next(pending, None)
                    if item is not None and not stop.is_set():
                        active.add(pool.submit(run_item, db.path, job_id, item, options, value, stop))

                for _ in range(options.get("parallel", 1)):
                    submit_next()
                while active:
                    done, active = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    touch()
                    # Check the whole completed batch before refilling any slot.
                    for future in done:
                        future.result()
                    for _ in done:
                        submit_next()
            except BaseException:
                stop.set()
                raise
            finally:
                # Workers observe stop through controller heartbeats and clean their own Pods.
                pool.shutdown(wait=True, cancel_futures=True)
            states = [
                r[0] for r in db.conn.execute("SELECT status FROM sweep_items WHERE sweep_id=?", (job_id,))
            ]
            has_blocks = any(
                a["status"] == "blocked"
                for r in db.conn.execute("SELECT assessments FROM sweep_items WHERE sweep_id=?", (job_id,))
                for a in json.loads(r[0])
            )
            status = (
                "completed"
                if states and all(s == "completed" for s in states) and not has_blocks
                else "partial"
            )
            with db.conn:
                db.conn.execute(
                    "UPDATE sweeps SET status=?,finished_at=?,heartbeat_at=? WHERE id=?",
                    (status, time.time(), time.time(), job_id),
                )
        except BaseException as error:
            with db.conn:
                db.conn.execute(
                    "UPDATE sweeps SET status='interrupted',error=?,heartbeat_at=? WHERE id=?",
                    (str(error), time.time(), job_id),
                )
            raise


if __name__ == "__main__":
    run(*sys.argv[1:])
