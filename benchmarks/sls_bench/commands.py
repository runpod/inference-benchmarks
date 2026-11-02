"""First-class durable sweep commands, separate from low-level run inspection."""

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import matrix as matrix_configs
from . import sweeps
from .registry import TYPES, sync_types
from .runpod import Runpod

console = Console()


def register(app):
    @app.command("types")
    def types(ctx: typer.Context):
        """List benchmark categories, their adapters, and their default SQLite suites."""
        sync_types(ctx.obj)
        table = Table("Type", "Adapter", "Default suite", "Implemented")
        for value in TYPES.values():
            table.add_row(
                value.key, value.adapter or "—", value.default_suite or "—", str(value.adapter is not None)
            )
        console.print(table)

    @app.command("matrix")
    def matrix(
        ctx: typer.Context,
        name: Annotated[str, typer.Argument()] = "text_gen",
        gpu: Annotated[list[str] | None, typer.Option("--gpu")] = None,
        live: bool = False,
        as_json: Annotated[bool, typer.Option("--json")] = False,
    ):
        """Inspect the SQLite matrix and catalog coverage. Never resolves downloads or rents Pods."""
        db = ctx.obj
        db.seed()
        try:
            row, expanded = matrix_configs.expand(db, name)
            catalog = (
                Runpod().gpus()
                if live
                else [json.loads(r[0]) for r in db.conn.execute("SELECT metadata FROM gpu_catalog")]
            )
            if gpu:
                missing = set(gpu) - {g["id"] for g in catalog}
                if missing:
                    raise ValueError(f"GPU not in selected catalog: {sorted(missing)}; use --live to refresh")
                catalog = [g for g in catalog if g["id"] in gpu]
            coverage = []
            for g in catalog:
                for w in expanded:
                    status, reason = sweeps.assess(
                        g,
                        {"spec": json.dumps(w["value"]), "matrix_placement": w["placement"]},
                        {"gpu_count": w["placement"]["gpu_count"]},
                    )
                    if not g.get("securePrice") or g["securePrice"] <= 0:
                        status, reason = "blocked", "No positive Secure Cloud price"
                    coverage.append(
                        {
                            "gpu": g["id"],
                            "gpu_count": w["placement"]["gpu_count"],
                            "workload": w["value"]["name"],
                            "status": status,
                            "reason": reason,
                        }
                    )
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        if as_json:
            print(
                json.dumps(
                    {
                        "matrix_id": row["id"],
                        "revision": row["revision"],
                        "matrix": json.loads(row["spec"]),
                        "workloads": expanded,
                        "catalog_source": "live" if live else "cached",
                        "coverage": coverage,
                    },
                    indent=2,
                )
            )
            return
        console.print(f"Matrix {row['name']} r{row['revision']} (ID {row['id']}); edit with config show/put")
        table = Table(
            "Row",
            "Model / precision",
            "GPU / TP",
            "Input → output",
            "Concurrency",
            "Requests × repeats",
            "Min GB",
            "Eligible SKUs",
        )
        for w in expanded:
            v = w["value"]
            policy = str(v.get("concurrency", "sequential"))
            if v.get("saturation_max_concurrency"):
                policy = f"{v['concurrency'][0]} → ×2 → {v['saturation_max_concurrency']} (plateau stop)"
            mixed = v.get("load_mode") == "open_loop_request_rate"
            if mixed:
                policy = f"{v['request_rates']} req/s; burst={v['burstiness']}"
            eligible = sum(
                c["workload"] == v["name"] and c["status"] == "eligible_unvalidated" for c in coverage
            )
            table.add_row(
                w["row"],
                f"{v['model']} / {v.get('quantization') or v.get('dtype', 'fp16')}",
                f"{w['placement']['gpu_count']}× / TP{w['placement']['gpu_count']}",
                "varies → varies"
                if mixed
                else f"{v.get('input_tokens', v.get('width'))} → {v.get('output_tokens', v.get('height'))}",
                policy,
                (
                    (
                        ", ".join(f"C{c}:{n}" for c, n in v.get("requests_by_concurrency", {}).items())
                        + f"; otherwise {v['requests']}"
                    )
                    if v.get("requests_by_concurrency")
                    else str(v["requests"])
                )
                + f" × {v['repeats']}",
                str(v["min_vram_gb"]),
                str(eligible),
            )
        console.print(table)
        console.print(
            "Coverage uses live catalog only with --live; eligibility is estimated, not proof of fit. --json includes per-SKU exclusions."
        )

    @app.command("prepare")
    def prepare(
        ctx: typer.Context,
        gpu: Annotated[list[str] | None, typer.Option("--gpu")] = None,
        all_gpus: bool = False,
        target: Annotated[list[str] | None, typer.Option("--target")] = None,
        suite: str | None = None,
        matrix: str | None = None,
        benchmark_type: Annotated[str | None, typer.Option("--type")] = None,
        workload: Annotated[list[str] | None, typer.Option("--workload", "-w")] = None,
        model: str | None = None,
        parameters: Annotated[str | None, typer.Option("--params")] = None,
        max_minutes: Annotated[int, typer.Option(min=5, max=20160)] = 45,
        capacity_retries: Annotated[int, typer.Option(min=0, max=10)] = 1,
        ssh_key: Path = Path("~/.ssh/id_ed25519"),
        cloud: str = "SECURE",
        gpu_count: Annotated[int, typer.Option(min=1, max=8)] = 1,
        image: str = "vllm/vllm-openai:v0.11.0",
        disk_gb: Annotated[int, typer.Option(min=40, max=500)] = 100,
    ):
        """Save a resolved GPU × model/profile sweep in SQLite. NEVER creates paid Pods."""
        try:
            overrides = json.loads(parameters) if parameters else None
            if overrides is not None and not isinstance(overrides, dict):
                raise ValueError("--params must be a JSON object")
            job_id = sweeps.prepare(
                ctx.obj,
                Runpod(),
                gpu_types=gpu,
                all_gpus=all_gpus,
                targets=target,
                suite=suite,
                matrix=matrix,
                workload_names=workload,
                benchmark_type=benchmark_type,
                model=model,
                parameters=overrides,
                max_minutes=max_minutes,
                retries=capacity_retries,
                ssh_key=ssh_key,
                cloud=cloud,
                gpu_count=gpu_count,
                image=image,
                disk_gb=disk_gb,
            )
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        value = sweeps.job(ctx.obj, job_id)
        counts = dict(
            ctx.obj.conn.execute(
                "SELECT status,COUNT(*) FROM sweep_items WHERE sweep_id=? GROUP BY status", (job_id,)
            ).fetchall()
        )
        console.print(
            f"Prepared sweep {job_id}; {counts}; estimated eligible ceiling ${value['estimated_ceiling']:.2f}"
        )
        console.print(
            "No Pods launched. Inspect with `bench job ID`; launch explicitly with `bench start ID --budget DOLLARS --yes`."
        )

    @app.command("start")
    def start(
        ctx: typer.Context,
        sweep_id: str,
        budget: Annotated[float, typer.Option(min=0.01)],
        yes: bool = False,
        foreground: bool = False,
        parallel: Annotated[int | None, typer.Option(min=1, max=8, help="Maximum concurrent Pods")] = None,
        retry_unavailable: bool = False,
        after: Annotated[
            str | None, typer.Option(help="Wait for this sweep to finish and clean up before allocating")
        ] = None,
    ):
        """Start/resume a prepared sweep. Detached by default; completed runs are never repeated."""
        if not yes:
            raise typer.BadParameter("Paid execution requires --yes")
        if retry_unavailable:
            if sweeps.job(ctx.obj, sweep_id)["status"] not in ("partial", "interrupted", "prepared"):
                raise typer.BadParameter("Do not reset unavailable items while a dispatcher is running")
            with ctx.obj.conn:
                ctx.obj.conn.execute(
                    "UPDATE sweep_items SET status='planned',run_id=NULL,reason=NULL "
                    "WHERE sweep_id=? AND status='unavailable'",
                    (sweep_id,),
                )
                ctx.obj.conn.execute(
                    "UPDATE sweeps SET status='prepared' WHERE id=? AND status='partial'", (sweep_id,)
                )
        try:
            pid = sweeps.start(
                ctx.obj, sweep_id, budget, detached=not foreground, parallel=parallel, after_sweep=after
            )
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        console.print(f"Sweep {sweep_id}; {'background PID ' + str(pid) if pid else 'finished'}")

    @app.command("jobs")
    def jobs(ctx: typer.Context):
        """Inspect persistent sweep queue/status without contacting Runpod."""
        table = Table("Sweep", "Status", "Items", "Finished items", "Est. ceiling $")
        for row in ctx.obj.conn.execute(
            "SELECT s.*,COUNT(i.id) items,SUM(i.status NOT IN ('planned','running')) done "
            "FROM sweeps s LEFT JOIN sweep_items i ON s.id=i.sweep_id "
            "GROUP BY s.id ORDER BY s.created_at DESC LIMIT 30"
        ):
            table.add_row(
                row["id"],
                row["status"],
                str(row["items"]),
                str(row["done"]),
                f"{row['estimated_ceiling']:.2f}",
            )
        console.print(table)

    @app.command("job")
    def job(ctx: typer.Context, sweep_id: str, as_json: Annotated[bool, typer.Option("--json")] = False):
        """Inspect per-GPU progress, compatibility decisions, retries and linked run IDs."""
        value = sweeps.job(ctx.obj, sweep_id)
        value["options"] = json.loads(value["options"])
        rows = [
            dict(r)
            for r in ctx.obj.conn.execute(
                "SELECT i.id,i.ordinal,i.status,i.reason,i.assessments,i.run_id,i.attempts,"
                "json_extract(c.spec,'$.gpu_type') gpu_type FROM sweep_items i JOIN configs c ON i.target_config_id=c.id "
                "WHERE i.sweep_id=? ORDER BY i.ordinal",
                (sweep_id,),
            )
        ]
        for row in rows:
            row["assessments"] = json.loads(row["assessments"])
        if as_json:
            print(json.dumps({**value, "items": rows}, indent=2))
            return
        console.print(
            f"{sweep_id}: {value['status']}; estimate ${value['estimated_ceiling']:.2f}; error={value['error']}"
        )
        table = Table("GPU", "Status", "Attempts", "Run", "Reason")
        for row in rows:
            table.add_row(
                row["gpu_type"],
                row["status"],
                str(row["attempts"]),
                row["run_id"] or "—",
                row["reason"] or "—",
            )
        console.print(table)

    @app.command("cancel")
    def cancel(ctx: typer.Context, sweep_id: str):
        """Interrupt this local sweep dispatcher; its controller cleans up its owned Pod."""
        value = sweeps.job(ctx.obj, sweep_id)
        if not value["pid"] or value["status"] not in ("running", "waiting"):
            raise typer.BadParameter("Sweep has no running local dispatcher")
        try:
            args = subprocess.check_output(["ps", "-p", str(value["pid"]), "-o", "args="], timeout=5).decode()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            raise typer.BadParameter(
                "Dispatcher is not alive; use `start` for recovery or `cleanup --expired`"
            ) from None
        if sweep_id not in args or "sls_bench.sweeps" not in args:
            raise typer.BadParameter("PID identity mismatch; refusing to signal an unrelated process")
        os.kill(value["pid"], signal.SIGINT)
        console.print(f"Cancellation requested for {sweep_id}; inspect job status and resource cleanup")
