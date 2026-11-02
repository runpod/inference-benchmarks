import hashlib
import json
import subprocess

import pytest

from sls_bench import image_diffusers, worker


@pytest.mark.parametrize("installed", [False, True])
def test_hard_row_timeout_reserves_cleanup_and_retains_points(tmp_path, monkeypatch, installed):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "REPORT", {"trials": [{"config_id": 1}], "phases": []})
    clock = [0.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    if installed:
        requirements = "\n".join(image_diffusers.REQUIREMENTS) + "\n"
        (tmp_path / "diffusers-environment.sha256").write_text(
            hashlib.sha256(requirements.encode()).hexdigest()
        )
    calls = []

    def command(args, log, timeout, **kwargs):
        calls.append((log, timeout))
        if log == "diffusers-install.log":
            clock[0] += 100
            return
        config = json.loads((tmp_path / "diffusers-225-config.json").read_text())
        assert config["row_budget_seconds"] == timeout
        directory = tmp_path / "diffusers-225"
        (directory / "progress.json").write_text(json.dumps({
            "trials": [{"config_id": 225, "status": "completed", "concurrency": 1}],
        }))
        # Simulate the child deadline followed by command()'s full stop grace.
        clock[0] += timeout + 20
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr(worker, "command", command)
    worker.direct_image_workload({"config_id": 225, "row_budget_seconds": 1800})
    assert clock[0] <= 1800
    assert calls[-1][1] == (1780 if installed else 1680)
    if not installed:
        assert calls[0][1] == 600
    assert [t.get("status") for t in worker.REPORT["trials"]] == [
        None, "completed", "budget_exhausted",
    ]
    assert (tmp_path / "diffusers-225-details.json").exists()


def test_absolute_row_timer_covers_finalization_and_restores_lease(monkeypatch):
    timers = []
    handlers = []
    clock = iter([10.0, 1810.0])
    original_handler = object()
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(worker.signal, "getsignal", lambda _: original_handler)
    monkeypatch.setattr(worker.signal, "getitimer", lambda _: (3000.0, 0.0))
    monkeypatch.setattr(worker.signal, "signal", lambda _, handler: handlers.append(handler))
    monkeypatch.setattr(worker.signal, "setitimer", lambda *args: timers.append(args))

    def finalize(config):
        handlers[0](None, None)

    monkeypatch.setattr(worker, "_direct_image_workload", finalize)
    monkeypatch.setattr(worker, "REPORT", {"trials": [{"status": "completed"}]})
    worker.direct_image_workload({"config_id": 225, "row_budget_seconds": 1800})
    assert [t["status"] for t in worker.REPORT["trials"]] == ["completed", "budget_exhausted"]
    assert timers[0][1] == 1800
    assert timers[-1][1] == 1200
    assert handlers[-1] is original_handler
