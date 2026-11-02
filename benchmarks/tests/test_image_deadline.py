import json
from types import SimpleNamespace

from sls_bench import worker


def test_absolute_deadline_kills_child_without_grace(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    process = SimpleNamespace(pid=123, wait=lambda **kw: (_ for _ in ()).throw(
        worker.ImageRowDeadline("deadline")
    ))
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: process)
    stops = []
    monkeypatch.setattr(worker, "stop", lambda p, grace=20: stops.append((p, grace)))
    try:
        worker.command(["child"], "child.log", timeout=100)
    except worker.ImageRowDeadline:
        pass
    else:
        raise AssertionError("deadline must propagate")
    assert stops == [(process, 0)]


def test_real_stop_at_absolute_deadline_never_waits(monkeypatch):
    kills = []
    monkeypatch.setattr(worker.os, "killpg", lambda pid, sig: kills.append((pid, sig)))

    def wait(**kwargs):
        raise AssertionError("absolute deadline must never synchronously reap")

    worker.stop(SimpleNamespace(pid=123, wait=wait), grace=0)
    assert kills == [(123, worker.signal.SIGKILL)]


def test_absolute_deadline_skips_final_sync_and_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "REPORT", {"trials": [{"status": "completed"}]})

    def command(args, log, timeout, **kwargs):
        if log == "diffusers-install.log":
            return
        directory = tmp_path / "diffusers-225"
        (directory / "progress.json").write_text(json.dumps({"trials": []}))
        (directory / "sample.png").write_bytes(b"png")
        raise worker.ImageRowDeadline("deadline")

    monkeypatch.setattr(worker, "command", command)
    monkeypatch.setattr(worker.shutil, "copyfile", lambda *a: (_ for _ in ()).throw(
        AssertionError("must not copy after absolute deadline")
    ))
    worker.direct_image_workload({"config_id": 225, "row_budget_seconds": 1800})
    assert [t["status"] for t in worker.REPORT["trials"]] == ["completed", "budget_exhausted"]
