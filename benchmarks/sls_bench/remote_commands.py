"""Explicit paid launch and reconnect commands for remote controllers."""

from pathlib import Path
from typing import Annotated

import typer

from . import remote
from .runpod import Runpod

app = typer.Typer(no_args_is_help=True)


@app.command()
def launch(
    ctx: typer.Context,
    sweep_id: str,
    budget: Annotated[float, typer.Option(min=0.01)],
    cpu_hourly_limit: Annotated[float, typer.Option(min=0.001)],
    yes: bool = False,
    ssh_key: Path = Path("~/.ssh/id_ed25519"),
    parallel: Annotated[int, typer.Option(min=1, max=8)] = 1,
    cpu_flavor: str = "cpu3g",
    vcpus: Annotated[int, typer.Option(min=2)] = 2,
    volume_gb: Annotated[int, typer.Option(min=10)] = 30,
):
    """Rent a CPU controller for a fresh prepared sweep. CPU billed until release."""
    if not yes:
        raise typer.BadParameter("Paid CPU/GPU execution requires --yes")
    remote.launch(
        ctx.obj, Runpod(), sweep_id, budget, cpu_hourly_limit, ssh_key, parallel, cpu_flavor, vcpus, volume_gb
    )
    typer.echo(f"Remote controller launched for {sweep_id}; reconnect with `bench remote pull {sweep_id}`")


@app.command()
def pull(ctx: typer.Context, sweep_id: Annotated[str | None, typer.Argument()] = None):
    """Reconnect and merge results; without ID, pull all retained controllers."""
    remote.setup(ctx.obj)
    ids = (
        [sweep_id]
        if sweep_id
        else [
            r[0]
            for r in ctx.obj.conn.execute(
                "SELECT sweep_id FROM remote_controllers WHERE state NOT IN ('deleted','rejected')"
            )
        ]
    )
    errors = []
    for identity in ids:
        try:
            typer.echo(f"{identity}: {remote.pull(ctx.obj, Runpod(), identity)}")
        except Exception as error:
            errors.append(identity)
            typer.echo(f"{identity}: {error}", err=True)
    if errors:
        raise typer.Exit(1)


@app.command()
def resume(ctx: typer.Context, sweep_id: str, yes: bool = False):
    """Resume the existing remote dispatcher with its existing approved budget."""
    if not yes:
        raise typer.BadParameter("Resuming paid execution requires --yes")
    remote.resume(ctx.obj, Runpod(), sweep_id)
    typer.echo("Existing remote dispatcher invoked; pull to inspect progress")


@app.command()
def status(ctx: typer.Context):
    """Show local ownership records without contacting the provider."""
    remote.setup(ctx.obj)
    for row in ctx.obj.conn.execute("SELECT * FROM remote_controllers"):
        typer.echo(dict(row))


@app.command()
def release(ctx: typer.Context, sweep_id: str, yes: bool = False):
    """Pull once more, verify GPU cleanup, then delete the CPU Pod."""
    if not yes:
        raise typer.BadParameter("CPU deletion requires --yes")
    remote.release(ctx.obj, Runpod(), sweep_id)
    typer.echo("Results imported; CPU Pod deleted")
