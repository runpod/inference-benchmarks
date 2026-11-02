import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .db import encode
from .runpod import ProvisionRejected, SupplyUnavailable, safe_metadata


class SSH:
    def __init__(self, pod, key, known_hosts):
        port = (pod.get("portMappings") or {}).get("22")
        if not pod.get("publicIp") or not port:
            raise ValueError("Pod public SSH mapping is not ready")
        self.args = [
            "ssh",
            "-T",
            "-i",
            str(Path(key).expanduser()),
            "-p",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            f"root@{pod['publicIp']}",
        ]

    def call(self, command, data=None, timeout=30):
        result = subprocess.run([*self.args, command], input=data, capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(
                f"SSH exit {result.returncode}: {result.stderr.decode(errors='replace')[-1500:]}"
            )
        return result.stdout


def cleanup(db, api, run_id, expired_only=False):
    from .remote import record

    row = db.run(run_id)
    owner = record(db, json.loads(row["spec"]).get("sweep_id"))
    if owner and owner["state"] != "deleted":
        raise ValueError("GPU cleanup belongs to the remote controller; do not race its dispatcher")
    if not row["owned"] or row["resource_state"] == "deleted":
        return
    if expired_only and row["lease_expires_at"] > time.time():
        return
    try:
        pod_id = row["pod_id"]
        if not pod_id:
            # Recover the accepted-create/lost-response crash window using the unique persisted name.
            matches = [p for p in api.pods(row["pod_name"]) if p.get("name") == row["pod_name"]]
            for pod in matches:
                api.delete(pod["id"])
            if (
                not matches
                and row["resource_state"] == "creation_ambiguous"
                and time.time() < row["lease_expires_at"] + 120
            ):
                raise RuntimeError(
                    "Create outcome is ambiguous; keeping the lease and reconciling by exact name"
                )
        else:
            api.delete(pod_id)
        db.update("resources", run_id, state="deleted", cleanup_error=None)
        db.event(run_id, "pod_deleted", {"pod_id": pod_id})
    except Exception as error:
        db.update(
            "resources",
            run_id,
            state="creation_ambiguous" if row["resource_state"] == "creation_ambiguous" else "cleanup_failed",
            cleanup_error=str(error),
        )
        raise


def collect(db, run_id, ssh, remote, artifacts=False):
    report = json.loads(ssh.call(f"test -f {remote}/report.json && head -c 100000000 {remote}/report.json"))
    db.ingest(run_id, report)
    db.artifact(run_id, "report.json", encode(report).encode())
    if artifacts:
        # No model files or environment dumps; only files emitted by the benchmark worker.
        data = ssh.call(
            f"cd {remote} && find . -maxdepth 1 -type f ! -name spec.json "
            "! -name worker.py ! -name report.tmp -print0 | tar --null -T - -czf -",
            timeout=120,
        )
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for entry in archive:
                if entry.isfile() and entry.size <= 100_000_000:
                    db.artifact(run_id, Path(entry.name).name, archive.extractfile(entry).read())
    return report


def watchdog(db, run_id):
    # Separate process survives controller termination. No credentials in command-line arguments.
    log = db.path.parent / "watchdog.log"
    with log.open("ab") as out:
        subprocess.Popen(
            [sys.executable, "-m", "sls_bench.watchdog", str(db.path), run_id],
            stdout=out,
            stderr=out,
            start_new_session=True,
            close_fds=True,
        )


def execute(db, api, run_id, ssh_key, progress=print, provision_timeout=1200):
    row = db.run(run_id)
    spec = json.loads(row["spec"])
    remote = f"/workspace/bench/{run_id}"
    ssh = None
    report = None
    started = time.monotonic()
    heartbeat = getattr(progress, "heartbeat", None)
    db.update("runs", run_id, status="provisioning")
    if row["owned"]:
        watchdog(db, run_id)
    try:
        if row["owned"]:
            public_key = Path(str(Path(ssh_key).expanduser()) + ".pub").read_text().strip()
            remaining = max(1, int(row["lease_expires_at"] - time.time()))
            try:
                pod = api.create(row["pod_name"], spec["target"], public_key, remaining)
            except (SupplyUnavailable, ProvisionRejected):
                raise
            except BaseException:
                # A lost response or interrupt is not proof the billable resource was never created.
                db.update("resources", run_id, state="creation_ambiguous")
                raise
            db.update("resources", run_id, pod_id=pod["id"], state="created")
            db.event(run_id, "pod_created", safe_metadata(pod))
        else:
            pod = api.pod(row["pod_id"])
        pod_id = pod["id"]
        progress(f"{run_id}: Pod {pod_id}; waiting for SSH")
        deadline = min(time.time() + provision_timeout, row["lease_expires_at"])
        last_ready_error = None
        while time.time() < deadline:
            if heartbeat:
                heartbeat()
            pod = api.pod(pod_id)
            try:
                candidate = SSH(pod, ssh_key, db.path.parent / "known_hosts")
                requirements = "command -v python3"
                if any(w["kind"] == "text" for w in spec["workloads"]):
                    requirements += " && command -v vllm"
                candidate.call(requirements, timeout=15)
                ssh = candidate
                break
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                last_ready_error = str(error)
                time.sleep(5)
        if ssh is None:
            db.event(run_id, "provision_timeout", {"last_readiness_error": last_ready_error})
            raise TimeoutError(
                f"Pod did not become SSH-ready before provisioning deadline: {last_ready_error}"
            )
        metadata = safe_metadata(pod)
        # Exact SKU allocation check in addition to CUDA-level checks inside the worker.
        actual = (pod.get("gpu") or {}).get("id") or (pod.get("machine") or {}).get("gpuTypeId")
        if actual and actual != spec["target"]["gpu_type"]:
            raise RuntimeError(f"Wrong GPU allocation: {actual}")
        price = pod.get("adjustedCostPerHr")
        if price is None:
            price = pod.get("costPerHr")
        if price is None:
            raise RuntimeError("Allocated Pod has no reported hourly price; refusing unpriced execution")
        if float(price) > spec["max_hourly_price"]:
            raise RuntimeError(
                f"Actual hourly price {price} exceeds approved ceiling {spec['max_hourly_price']}"
            )
        db.update("resources", run_id, state="running", metadata=encode(metadata))
        db.update("runs", run_id, status="running", hourly_price=float(price) if price is not None else None)
        db.event(run_id, "ssh_ready", {"seconds": time.monotonic() - started})
        ssh.call(f"mkdir -p {remote}")
        worker = (
            spec["sources"]["worker.py"].encode()
            if spec.get("sources")
            else Path(__file__).with_name("worker.py").read_bytes()
        )
        db.artifact(run_id, "worker.py", worker)
        if any(w.get("backend") == "diffusers" for w in spec["workloads"]):
            module = spec["sources"]["image_diffusers.py"].encode()
            ssh.call(f"cat > {remote}/image_diffusers.py", module)
            db.artifact(run_id, "image_diffusers.py", module)
            if any(w.get("gated_model") for w in spec["workloads"]):
                # Model credential only, outside the collected artifact directory.
                # Never add it to spec.json, Pod env, logs, events or source snapshots.
                token_path = db.path.parent / "private" / "hf_token"
                token = token_path.read_text().strip()
                if not token:
                    raise RuntimeError("Missing authorized model credential")
                ssh.call(
                    "umask 077; mkdir -p /root/.cache/huggingface; cat > /root/.cache/huggingface/token; chmod 600 /root/.cache/huggingface/token",
                    token.encode(),
                )
        ssh.call(f"cat > {remote}/worker.py", worker)
        ssh.call(f"cat > {remote}/spec.json", encode(spec).encode())
        ssh.call(
            f"nohup python3 -u {remote}/worker.py > {remote}/worker.log 2>&1 < /dev/null & "
            f"echo $! > {remote}/worker.pid",
            timeout=30,
        )
        last_count = -1
        while time.time() < row["lease_expires_at"] - 30:
            if heartbeat:
                heartbeat()
            time.sleep(10)
            try:
                report = collect(db, run_id, ssh, remote)
            except (ValueError, RuntimeError, subprocess.TimeoutExpired):
                continue
            count = len(report.get("trials", []))
            if count != last_count:
                progress(f"{run_id}: {report['status']}, {count} trials collected")
                last_count = count
            if report["status"] in ("completed", "partial", "failed"):
                break
        else:
            raise TimeoutError("Run reached its wall-clock limit")
        report = collect(db, run_id, ssh, remote, artifacts=True)
        db.update("runs", run_id, status=report["status"], error=report.get("error"))
        return report
    except BaseException as error:
        db.update(
            "runs",
            run_id,
            status=(
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "unavailable"
                if isinstance(error, SupplyUnavailable)
                else "timed_out"
                if isinstance(error, TimeoutError)
                else "failed"
            ),
            error=str(error),
        )
        if ssh:
            try:
                collect(db, run_id, ssh, remote, artifacts=True)
            except Exception as collection_error:
                db.event(run_id, "collection_failed", {"error": str(collection_error)})
        raise
    finally:
        if row["owned"]:
            cleanup(db, api, run_id)
        elif ssh:
            # Signal only this run's PID, checked against its absolute worker path.
            if report is None or report.get("status") not in ("completed", "partial", "failed"):
                try:
                    ssh.call(
                        f"pid=$(cat {remote}/worker.pid); "
                        f"if tr '\\0' ' ' < /proc/$pid/cmdline | grep -Fq '{remote}/worker.py'; "
                        "then kill -TERM $pid; fi",
                        timeout=15,
                    )
                    db.event(run_id, "attached_worker_cancelled", {"remote": remote})
                except Exception as error:
                    db.event(run_id, "attached_worker_cancel_failed", {"remote": remote, "error": str(error)})
            db.update("resources", run_id, state="detached")
        finished = time.time()
        current = db.run(run_id)
        price = current["hourly_price"]
        db.update(
            "runs",
            run_id,
            finished_at=finished,
            estimated_cost=(finished - row["created_at"]) / 3600 * price if price is not None else None,
        )
