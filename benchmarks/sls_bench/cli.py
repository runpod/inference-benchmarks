import json
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import sweeps as sweep_jobs
from .commands import register
from .controller import SSH, execute
from .controller import cleanup as cleanup_run
from .db import Database
from .registry import sync_types
from .remote_commands import app as remote_app
from .resolve import resolve
from .runpod import Runpod
from .targets import hydrate_target, materialize_target

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
configs = typer.Typer(no_args_is_help=True)
app.add_typer(configs, name="config")
app.add_typer(remote_app, name="remote")
console = Console()


@app.callback()
def root(
    ctx: typer.Context,
    db: Annotated[Path, typer.Option(help="SQLite source of truth")] = Path("data/benchmarks.sqlite3"),
):
    ctx.obj = Database(db)


@app.command()
def init(ctx: typer.Context):
    """Initialize SQLite and insert missing built-ins (never overwrite user revisions)."""
    ctx.obj.seed()
    sync_types(ctx.obj)
    console.print(f"Initialized {ctx.obj.path}")


@configs.command("list")
def config_list(ctx: typer.Context):
    rows = ctx.obj.conn.execute(
        "SELECT c.* FROM configs c JOIN (SELECT kind,name,MAX(revision) revision FROM configs GROUP BY kind,name) "
        "x USING(kind,name,revision) ORDER BY c.kind,c.name"
    ).fetchall()
    table = Table("ID", "Kind", "Name", "Revision")
    for row in rows:
        table.add_row(str(row["id"]), row["kind"], row["name"], str(row["revision"]))
    console.print(table)


@configs.command("show")
def config_show(ctx: typer.Context, name: str, revision: int | None = None):
    """Print JSON suitable for editing and piping into `config put`."""
    print(json.dumps(json.loads(ctx.obj.config(name, revision=revision)["spec"]), indent=2))


@configs.command("put")
def config_put(ctx: typer.Context):
    """Read a validated config from stdin and save a new immutable SQLite revision."""
    row = ctx.obj.save_config(json.load(sys.stdin))
    console.print(f"Saved {row['name']} revision {row['revision']} (ID {row['id']})")


@app.command()
def gpus(match: str = ""):
    """Discover exact catalog SKU names and advertised prices; does not allocate anything."""
    table = Table("GPU type", "GB", "Secure $/hr", "Community $/hr")
    for gpu in Runpod().gpus():
        if match.lower() in gpu["id"].lower():
            table.add_row(
                gpu["id"], str(gpu["memoryInGb"]), str(gpu.get("securePrice")), str(gpu.get("communityPrice"))
            )
    console.print(table)


def selection(db, targets, workloads):
    def selected(selector, kind=None):
        name, separator, revision = selector.partition("@")
        return db.config(name, kind, revision=int(revision) if separator else None)

    ts = [selected(name, "target") for name in targets]
    ws = [selected(name) for name in workloads]
    if not ts or not ws or any(w["kind"] not in ("text", "image") for w in ws):
        raise typer.BadParameter("Specify at least one target and one text/image workload")
    if len({w["id"] for w in ws}) != len(ws):
        raise typer.BadParameter("Duplicate workload")
    return ts, ws


def choose_configs(db, targets, workloads, gpu_types, catalog, **options):
    names = list(targets or [])
    for gpu_type in gpu_types or []:
        try:
            names.append(materialize_target(db, gpu_type, catalog, **options)["name"])
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
    return selection(db, names, workloads)


@app.command()
def plan(
    ctx: typer.Context,
    workload: Annotated[list[str], typer.Option("--workload", "-w")],
    gpu: Annotated[
        list[str] | None, typer.Option("--gpu", help="Exact live Runpod GPU SKU; repeat for a sweep")
    ] = None,
    target: Annotated[
        list[str] | None, typer.Option("--target", "-t", help="Optional saved SQLite target profile")
    ] = None,
    max_minutes: Annotated[int, typer.Option(min=5, max=240)] = 45,
    cloud: str = "SECURE",
    gpu_count: Annotated[int, typer.Option(min=1, max=8)] = 1,
    min_vcpu: Annotated[int, typer.Option(min=2)] = 4,
    min_ram_gb: Annotated[int, typer.Option(min=16)] = 32,
    image: str = "vllm/vllm-openai:v0.11.0",
):
    """Preview workloads; --gpu resolves the live catalog into SQLite. Never rents Pods."""
    catalog = {g["id"]: g for g in Runpod().gpus()} if gpu else {}
    targets, workloads = choose_configs(
        ctx.obj,
        target,
        workload,
        gpu,
        catalog,
        cloud=cloud,
        gpu_count=gpu_count,
        min_vcpu=min_vcpu,
        min_ram_gb=min_ram_gb,
        image=image,
    )
    for t in targets:
        spec = json.loads(t["spec"])
        console.print(f"[bold]{t['name']} r{t['revision']}[/bold]: {spec['gpu_type']} ({spec['cloud']})")
        for w in workloads:
            value = json.loads(w["spec"])
            if w["kind"] == "text":
                n = len(value["concurrency"]) * value["repeats"]
                console.print(
                    f"  {w['name']} r{w['revision']}: {value['input_tokens']} input + "
                    f"{value['output_tokens']} output, c={value['concurrency']}, "
                    f"{n} trials × {value['requests']} requests, {value['dtype']}"
                )
            else:
                console.print(
                    f"  {w['name']} r{w['revision']}: {value['width']}×{value['height']}, "
                    f"{value['steps']} steps, batch={value['batch_size']}"
                )
    console.print(
        f"Sequential, one Pod per target, wall-clock ceiling {max_minutes} minutes/Pod. "
        "Long-context OOMs remain failures; no automatic precision/context fallback."
    )


@app.command()
def sweep(
    ctx: typer.Context,
    workload: Annotated[list[str] | None, typer.Option("--workload", "-w")] = None,
    gpu: Annotated[
        list[str] | None, typer.Option("--gpu", help="Exact live Runpod GPU SKU; repeat for a sweep")
    ] = None,
    target: Annotated[
        list[str] | None, typer.Option("--target", "-t", help="Optional saved SQLite target profile")
    ] = None,
    max_minutes: Annotated[int, typer.Option(min=5, max=240)] = 45,
    budget: Annotated[
        float, typer.Option(min=0.01, help="Maximum estimated total dollars incl. 20% buffer")
    ] = 3,
    ssh_key: Path = Path("~/.ssh/id_ed25519"),
    yes: Annotated[bool, typer.Option("--yes", help="Approve paid Pod creation")] = False,
    pod_id: Annotated[str | None, typer.Option(help="Attach to an existing Pod; never delete it")] = None,
    cloud: str = "SECURE",
    gpu_count: Annotated[int, typer.Option(min=1, max=8)] = 1,
    min_vcpu: Annotated[int, typer.Option(min=2)] = 4,
    min_ram_gb: Annotated[int, typer.Option(min=16)] = 32,
    image: str = "vllm/vllm-openai:v0.11.0",
    suite: str | None = None,
    benchmark_type: Annotated[str | None, typer.Option("--type")] = None,
    model: str | None = None,
    parameters: Annotated[
        str | None, typer.Option("--params", help="JSON workload parameter overrides")
    ] = None,
    detach: bool = False,
):
    """Run a GPU sweep. --suite/--type selects a stored matrix; --detach runs it in the background."""
    db = ctx.obj
    if suite or benchmark_type or model or parameters or detach or not workload:
        if pod_id:
            raise typer.BadParameter(
                "Durable suites manage their own Pods; use explicit foreground workloads to attach"
            )
        if not yes:
            raise typer.BadParameter(
                "Paid execution requires --yes; use `prepare` to save a plan without renting Pods"
            )
        try:
            overrides = json.loads(parameters) if parameters else None
            if overrides is not None and not isinstance(overrides, dict):
                raise ValueError("--params must be a JSON object")
            job_id = sweep_jobs.prepare(
                db,
                Runpod(),
                gpu_types=gpu,
                targets=target,
                suite=suite,
                workload_names=workload,
                benchmark_type=benchmark_type,
                model=model,
                parameters=overrides,
                max_minutes=max_minutes,
                ssh_key=ssh_key,
                cloud=cloud,
                gpu_count=gpu_count,
                min_vcpu=min_vcpu,
                min_ram_gb=min_ram_gb,
                image=image,
            )
            pid = sweep_jobs.start(db, job_id, budget, detached=detach)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        console.print(f"Sweep {job_id}; {'background PID ' + str(pid) if pid else 'finished'}")
        console.print(f"Inspect: bench --db {db.path} job {job_id}")
        return
    ssh_key = ssh_key.expanduser().resolve()
    if not ssh_key.exists() or not Path(str(ssh_key) + ".pub").exists():
        raise typer.BadParameter("SSH private key and matching .pub file must exist")
    if not yes:
        raise typer.BadParameter(
            "This rents paid Pods. Inspect `plan`, then pass --yes with an explicit budget."
        )
    api = Runpod()
    catalog = {g["id"]: g for g in api.gpus()}
    targets, workloads = choose_configs(
        db,
        target,
        workload,
        gpu,
        catalog,
        cloud=cloud,
        gpu_count=gpu_count,
        min_vcpu=min_vcpu,
        min_ram_gb=min_ram_gb,
        image=image,
    )
    if pod_id and len(targets) != 1:
        raise typer.BadParameter("--pod-id requires exactly one GPU/target")
    specs = []
    estimate = 0
    for original in targets:
        row = hydrate_target(original, catalog)
        value = json.loads(row["spec"])
        if value["gpu_count"] != 1 and any(w["kind"] == "image" for w in workloads):
            raise typer.BadParameter(
                "The ComfyUI adapter is single-GPU; use --gpu-count 1 for image workloads"
            )
        gpu = catalog.get(value["gpu_type"])
        if not gpu:
            raise typer.BadParameter(f"Exact SKU not found: {value['gpu_type']}; refusing substitution")
        price = gpu.get("securePrice" if value["cloud"] == "SECURE" else "communityPrice")
        if price is None:
            raise typer.BadParameter(f"No catalog price for {row['name']} in {value['cloud']}")
        # Explicit margin accounts for storage and teardown time, not an exact billing guarantee.
        ceiling = float(price) * value["gpu_count"] * 1.2 + value["disk_gb"] * 0.0002
        estimate += ceiling * max_minutes / 60
        spec = resolve(row, workloads)
        spec.update(
            max_hourly_price=ceiling,
            catalog_hourly_price=price,
            max_minutes=max_minutes,
            budget_dollars=budget,
            controller_python=sys.version,
        )
        specs.append((row, spec))
    if estimate > budget:
        raise typer.BadParameter(f"Estimated ceiling ${estimate:.2f} exceeds budget ${budget:.2f}")
    console.print(f"Approved estimate ≤ ${estimate:.2f}; {len(specs)} sequential Pod runs.")
    failures = 0
    for target_row, spec in specs:
        run_id = db.new_run(
            target_row, workloads, spec, max_minutes * 60, owned=pod_id is None, pod_id=pod_id
        )
        console.print(f"[bold]Run {run_id}[/bold] ({target_row['name']})")
        try:
            result = execute(db, api, run_id, ssh_key, progress=console.print)
            failures += result["status"] != "completed"
        except KeyboardInterrupt:
            raise
        except Exception as error:
            failures += 1
            console.print(f"{run_id}: {error}", style="red", markup=False)
            if db.run(run_id)["resource_state"] in ("cleanup_failed", "creation_ambiguous"):
                console.print("Stopping sweep because Pod deletion failed. Run cleanup immediately.")
                break
    if failures:
        raise typer.Exit(1)


@app.command()
def status(ctx: typer.Context, run_id: Annotated[str | None, typer.Argument()] = None):
    """Inspect local state without calling Runpod."""
    if run_id:
        row = ctx.obj.run(run_id)
        row.pop("spec")
        print(json.dumps(row, indent=2))
        return
    table = Table("Run", "Target", "Status", "Pod", "Resource", "Trials", "Est. $")
    rows = ctx.obj.conn.execute(
        "SELECT r.id,c.name,r.status,p.pod_id,p.state,r.estimated_cost,COUNT(t.id) trials "
        "FROM runs r JOIN configs c ON r.target_config_id=c.id JOIN resources p ON r.id=p.run_id "
        "LEFT JOIN trials t ON t.run_id=r.id GROUP BY r.id ORDER BY r.created_at DESC LIMIT 50"
    ).fetchall()
    for r in rows:
        table.add_row(
            r["id"],
            r["name"],
            r["status"],
            r["pod_id"] or "—",
            r["state"],
            str(r["trials"]),
            f"{r['estimated_cost']:.3f}" if r["estimated_cost"] is not None else "—",
        )
    console.print(table)


@app.command()
def results(
    ctx: typer.Context,
    run_id: Annotated[str | None, typer.Argument()] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
):
    """Show per-trial metrics. Compare only matching workload revisions and settings."""
    query = (
        "SELECT t.*,COALESCE(json_extract(r.spec,'$.replica_count'),1) replica_count,"
        "COALESCE(json_extract(r.spec,'$.gpus_per_replica'),1) gpus_per_replica,"
        "c.name workload,tc.name target FROM trials t JOIN configs c "
        "ON t.workload_config_id=c.id JOIN runs r ON t.run_id=r.id "
        "JOIN configs tc ON r.target_config_id=tc.id"
    )
    rows = ctx.obj.conn.execute(
        query + (" WHERE t.run_id=?" if run_id else "") + " ORDER BY t.id", (run_id,) if run_id else ()
    ).fetchall()
    if as_json:
        print(
            json.dumps(
                [
                    {**{k: r[k] for k in r.keys() if k != "result"}, "result": json.loads(r["result"])}
                    for r in rows
                ],
                indent=2,
            )
        )
        return
    table = Table(
        "Target",
        "Workload",
        "Replicas",
        "GPU/replica",
        "Load (concurrency or req/s)",
        "Repeat",
        "Status",
        "Output tok/s",
        "TTFT p95 ms",
        "E2E p95 ms",
    )
    for row in rows:
        value = json.loads(row["result"])
        m = value.get("metrics", {})

        def fmt(k):
            return f"{m[k]:.2f}" if m.get(k) is not None else "—"

        table.add_row(
            row["target"],
            row["workload"],
            str(row["replica_count"]),
            str(row["gpus_per_replica"]),
            f"{value['request_rate']} req/s"
            if value.get("load_mode") == "open_loop_request_rate"
            else f"c={row['concurrency']}",
            str(row["repeat"]),
            row["status"],
            fmt("output_throughput"),
            fmt("p95_ttft_ms"),
            fmt("p95_e2el_ms"),
        )
    console.print(table)


@app.command()
def cleanup(
    ctx: typer.Context, run_id: Annotated[str | None, typer.Argument()] = None, expired: bool = False
):
    """Delete only CLI-owned Pods, including ambiguous-create orphans; attached Pods are never deleted."""
    if not run_id and not expired:
        raise typer.BadParameter("Specify a run ID or --expired")
    db = ctx.obj
    api = Runpod()
    rows = (
        [db.run(run_id)]
        if run_id
        else db.conn.execute(
            "SELECT r.id FROM runs r JOIN resources p ON r.id=p.run_id "
            "WHERE p.owned=1 AND p.state!='deleted' AND r.lease_expires_at<=?",
            (time.time(),),
        ).fetchall()
    )
    for row in rows:
        cleanup_run(db, api, row["id"], expired_only=expired)
        console.print(f"Cleaned {row['id']}")


@app.command()
def logs(
    ctx: typer.Context,
    run_id: str,
    lines: Annotated[int, typer.Option(min=1, max=1000)] = 60,
    ssh_key: Path = Path("~/.ssh/id_ed25519"),
    name: str = "worker.log",
):
    """Tail a log on a live Pod or read its collected SQLite artifact."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.log", name):
        raise typer.BadParameter("Log name must be a simple .log filename")
    db = ctx.obj
    row = db.run(run_id)
    if row["resource_state"] in ("running", "created") and row["pod_id"]:
        pod = Runpod().pod(row["pod_id"])
        ssh = SSH(pod, ssh_key, db.path.parent / "known_hosts")
        value = ssh.call(f"tail -n {lines} /workspace/bench/{run_id}/{name}")
    else:
        artifact_row = db.conn.execute(
            "SELECT content FROM artifacts WHERE run_id=? AND path=?", (run_id, name)
        ).fetchone()
        if not artifact_row:
            raise typer.BadParameter("No collected worker log; execution may not have started")
        value = b"\n".join(zlib.decompress(artifact_row[0]).splitlines()[-lines:])
    print(value.decode(errors="replace"))


@app.command()
def artifact(ctx: typer.Context, run_id: str, name: str, output: Annotated[Path, typer.Option()]):
    """Export a log/image/raw result stored as a compressed SQLite BLOB."""
    row = ctx.obj.conn.execute(
        "SELECT content FROM artifacts WHERE run_id=? AND path=?", (run_id, name)
    ).fetchone()
    if not row:
        raise typer.BadParameter("Artifact not found")
    if output.exists():
        raise typer.BadParameter("Refusing to overwrite existing file")
    output.write_bytes(zlib.decompress(row["content"]))


@app.command()
def backup(ctx: typer.Context, output: Path):
    """Online SQLite backup including WAL contents."""
    if output.exists():
        raise typer.BadParameter("Refusing to overwrite existing file")
    ctx.obj.backup(output)
    console.print(f"Backed up to {output}")


register(app)

if __name__ == "__main__":
    app()
