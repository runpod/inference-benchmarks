import json
import threading
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from sls_bench import sweeps
from sls_bench.cli import app
from sls_bench.config import validate
from sls_bench.db import Database
from sls_bench.registry import TYPES
from sls_bench.runpod import SupplyUnavailable

CATALOG = [
    {"id": "Example CUDA GPU", "manufacturer": "Nvidia", "memoryInGb": 24, "securePrice": 1},
    {"id": "Example Small CUDA GPU", "manufacturer": "NVIDIA", "memoryInGb": 6, "securePrice": 0.2},
    {"id": "Example Other Vendor", "manufacturer": "AMD", "memoryInGb": 192, "securePrice": 2},
]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    db = Database(tmp_path / "bench.sqlite3")
    api = Mock()
    api.gpus.return_value = CATALOG
    monkeypatch.setattr(sweeps, "Runpod", Mock(return_value=api))

    def resolved(target, workloads):
        value = json.loads(target["spec"])
        value["image"] = "example/runtime@sha256:fixed"
        return {
            "code_sha256": "code",
            "sources": {"worker.py": "captured"},
            "target": value,
            "workloads": [
                {**validate(json.loads(w["spec"])), "config_id": w["id"], "model_revision": "fixed-model-sha"}
                for w in workloads
            ],
        }

    resolver = Mock(side_effect=resolved)
    monkeypatch.setattr(sweeps, "resolve", resolver)
    monkeypatch.setattr(sweeps.time, "sleep", lambda _: None)
    key = tmp_path / "id_ed25519"
    key.write_text("private")
    key.with_suffix(".pub").write_text("public")
    yield db, api, key, resolver
    db.conn.close()


def test_prepare_all_catalog_gpus_without_allocating_and_resolve_models_once(setup):
    db, api, key, resolver = setup
    job_id = sweeps.prepare(db, api, all_gpus=True, suite="image", ssh_key=key, max_minutes=30)
    api.create.assert_not_called()
    resolver.assert_called_once()
    assert db.conn.execute("SELECT COUNT(*) FROM gpu_catalog").fetchone()[0] == 3
    items = db.conn.execute(
        "SELECT * FROM sweep_items WHERE sweep_id=? ORDER BY ordinal", (job_id,)
    ).fetchall()
    assert len(items) == 3
    assert [i["status"] for i in items] == ["planned", "planned", "blocked"]
    assert len(json.loads(items[0]["spec"])["workloads"]) == 2
    assert len(json.loads(items[1]["spec"])["workloads"]) == 1
    assert json.loads(items[1]["assessments"])[1]["status"] == "blocked"
    assert sweeps.job(db, job_id)["status"] == "prepared"
    assert sweeps.job(db, job_id)["approved_budget"] is None


def test_type_and_model_are_independent_of_gpu_and_saved_to_sqlite(setup):
    db, api, key, _ = setup
    job_id = sweeps.prepare(
        db,
        api,
        gpu_types=[CATALOG[0]["id"]],
        benchmark_type="image_gen",
        model="example/new-model",
        parameters={"steps": 17, "checkpoint": "weights.safetensors"},
        ssh_key=key,
    )
    item = db.conn.execute("SELECT spec FROM sweep_items WHERE sweep_id=?", (job_id,)).fetchone()
    values = json.loads(item[0])["workloads"]
    assert all(w["model"] == "example/new-model" and w["steps"] == 17 for w in values)
    assert all(w["benchmark_type"] == "image_gen" for w in values)
    assert not TYPES["image_processing"].adapter
    row = db.conn.execute("SELECT adapter FROM benchmark_types WHERE type_id='image_processing'").fetchone()
    assert row[0] is None


def test_large_sample_text_sweep_freezes_requests_repeats_and_long_timeouts(setup):
    db, api, key, _ = setup
    job_id = sweeps.prepare(
        db,
        api,
        gpu_types=[CATALOG[0]["id"]],
        benchmark_type="text_gen",
        ssh_key=key,
        parameters={"requests": 1024, "repeats": 2, "trial_timeout_seconds": 86400},
        max_minutes=2880,
    )
    spec = json.loads(
        db.conn.execute("SELECT spec FROM sweep_items WHERE sweep_id=?", (job_id,)).fetchone()[0]
    )
    assert len(spec["workloads"]) == 4
    for workload in spec["workloads"]:
        assert workload["requests"] == 1024
        assert workload["repeats"] == 2
        assert workload["trial_timeout_seconds"] == 86400
        assert workload["concurrency"] == [1, 4, 16]
    assert spec["max_minutes"] == 2880
    assert sweeps.job(db, job_id)["estimated_ceiling"] > 48
    api.create.assert_not_called()


def test_budget_required_before_start_and_no_execution_for_underfunded_plan(setup, monkeypatch):
    db, api, key, _ = setup
    job_id = sweeps.prepare(db, api, gpu_types=[CATALOG[0]["id"]], suite="image-smoke", ssh_key=key)
    runner = Mock()
    monkeypatch.setattr(sweeps, "run", runner)
    with pytest.raises(ValueError, match="exceeds budget"):
        sweeps.start(db, job_id, 0.01, detached=False)
    runner.assert_not_called()
    sweeps.start(db, job_id, 2, detached=False)
    runner.assert_called_once()


def test_runner_executes_once_and_never_repeats_completed_items(setup, monkeypatch):
    db, api, key, _ = setup
    job_id = sweeps.prepare(db, api, gpu_types=[CATALOG[0]["id"]], suite="image-smoke", ssh_key=key)
    with db.conn:
        db.conn.execute("UPDATE sweeps SET approved_budget=2 WHERE id=?", (job_id,))

    def execute(db, api, run_id, key, progress):
        db.update("resources", run_id, state="deleted")
        db.update("runs", run_id, status="completed")
        return {"status": "completed"}

    executor = Mock(side_effect=execute)
    monkeypatch.setattr(sweeps, "execute", executor)
    sweeps.run(db.path, job_id)
    assert sweeps.job(db, job_id)["status"] == "completed"
    sweeps.run(db.path, job_id)
    assert executor.call_count == 1
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_only_capacity_failures_retry_and_old_attempts_are_preserved(setup, monkeypatch):
    db, api, key, _ = setup
    job_id = sweeps.prepare(
        db, api, gpu_types=[CATALOG[0]["id"]], suite="image-smoke", ssh_key=key, retries=1
    )
    with db.conn:
        db.conn.execute("UPDATE sweeps SET approved_budget=2 WHERE id=?", (job_id,))
    count = 0

    def execute(db, api, run_id, key, progress):
        nonlocal count
        count += 1
        db.update("resources", run_id, state="deleted")
        if count == 1:
            db.update("runs", run_id, status="unavailable")
            raise SupplyUnavailable("No stock")
        db.update("runs", run_id, status="completed")
        return {"status": "completed"}

    monkeypatch.setattr(sweeps, "execute", execute)
    sweeps.run(db.path, job_id)
    assert count == 2
    assert db.conn.execute("SELECT attempts FROM sweep_items WHERE sweep_id=?", (job_id,)).fetchone()[0] == 2
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2


def test_parallel_bound_and_recovery_never_duplicates_existing_run(setup, monkeypatch):
    db, api, key, _ = setup
    api.gpus.return_value = [{**CATALOG[0], "id": f"GPU {i}"} for i in range(8)]
    job_id = sweeps.prepare(db, api, all_gpus=True, suite="image-smoke", ssh_key=key)
    with db.conn:
        db.conn.execute(
            "UPDATE sweeps SET approved_budget=20,options=json_set(options,'$.parallel',4) WHERE id=?",
            (job_id,),
        )
    item = dict(
        db.conn.execute("SELECT * FROM sweep_items WHERE sweep_id=? ORDER BY ordinal", (job_id,)).fetchone()
    )
    spec = json.loads(item["spec"])
    target = dict(db.conn.execute("SELECT * FROM configs WHERE id=?", (item["target_config_id"],)).fetchone())
    workloads = [
        dict(db.conn.execute("SELECT * FROM configs WHERE id=?", (w["config_id"],)).fetchone())
        for w in spec["workloads"]
    ]
    existing = db.new_run(target, workloads, spec, 1000)
    db.update("resources", existing, state="running", pod_id="existing-pod")
    with db.conn:
        db.conn.execute("UPDATE sweep_items SET run_id=?,status='running' WHERE id=?", (existing, item["id"]))
    barrier = threading.Barrier(4, timeout=10)
    seen = []
    mutex = threading.Lock()

    def finish(db, run_id):
        with mutex:
            seen.append(run_id)
        barrier.wait()
        db.update("resources", run_id, state="deleted")
        db.update("runs", run_id, status="completed")
        return "completed"

    def execute(db, api, run_id, key, progress):
        assert run_id != existing
        return {"status": finish(db, run_id)}

    recovered = Mock(side_effect=lambda db, api, run_id, key, progress: finish(db, run_id))
    monkeypatch.setattr(sweeps, "recover", recovered)
    monkeypatch.setattr(sweeps, "execute", execute)
    sweeps.run(db.path, job_id)
    assert len(seen) == len(set(seen)) == 8
    assert recovered.call_count == 1
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 8
    assert sweeps.job(db, job_id)["status"] == "completed"


def test_cleanup_failure_stops_further_admission(setup, monkeypatch):
    db, api, key, _ = setup
    api.gpus.return_value = [{**CATALOG[0], "id": f"GPU {i}"} for i in range(8)]
    job_id = sweeps.prepare(db, api, all_gpus=True, suite="image-smoke", ssh_key=key)
    with db.conn:
        db.conn.execute("UPDATE sweeps SET approved_budget=20 WHERE id=?", (job_id,))
    executor = Mock(return_value={"status": "failed"})
    monkeypatch.setattr(sweeps, "execute", executor)
    with pytest.raises(RuntimeError, match="Cleanup unresolved"):
        sweeps.run(db.path, job_id)
    assert executor.call_count == 1
    assert sweeps.job(db, job_id)["status"] == "interrupted"


def test_followup_waits_for_predecessor_before_allocating(setup, monkeypatch):
    db, api, key, _ = setup
    parent = sweeps.prepare(db, api, gpu_types=[CATALOG[0]["id"]], suite="image-smoke", ssh_key=key)
    child = sweeps.prepare(db, api, gpu_types=[CATALOG[0]["id"]], suite="image-smoke", ssh_key=key)
    executions = []
    waited = []

    def sleep(seconds):
        assert not executions
        waited.append(seconds)
        assert sweeps.job(db, child)["status"] == "waiting"
        with db.conn:
            db.conn.execute("UPDATE sweeps SET status='completed' WHERE id=?", (parent,))

    def execute(db, api, run_id, key, progress):
        assert sweeps.job(db, parent)["status"] == "completed"
        executions.append(run_id)
        db.update("resources", run_id, state="deleted")
        db.update("runs", run_id, status="completed")
        return {"status": "completed"}

    monkeypatch.setattr(sweeps.time, "sleep", sleep)
    monkeypatch.setattr(sweeps, "execute", execute)
    sweeps.start(db, child, 2, detached=False, after_sweep=parent)
    assert waited == [10]
    assert len(executions) == 1
    assert sweeps.job(db, child)["status"] == "completed"
    with pytest.raises(ValueError, match="cycle"):
        sweeps.start(db, parent, 2, after_sweep=child)


def test_followup_refuses_to_overlap_unresolved_parent_resources(setup, monkeypatch):
    db, api, key, _ = setup
    parent = sweeps.prepare(db, api, gpu_types=[CATALOG[0]["id"]], suite="image-smoke", ssh_key=key)
    child = sweeps.prepare(db, api, gpu_types=[CATALOG[0]["id"]], suite="image-smoke", ssh_key=key)
    item = db.conn.execute("SELECT * FROM sweep_items WHERE sweep_id=?", (parent,)).fetchone()
    target = db.conn.execute("SELECT * FROM configs WHERE id=?", (item["target_config_id"],)).fetchone()
    db.new_run(target, [], {"code_sha256": "test", "sweep_id": parent}, 60)
    with db.conn:
        db.conn.execute("UPDATE sweeps SET status='completed' WHERE id=?", (parent,))
    executor = Mock()
    monkeypatch.setattr(sweeps, "execute", executor)
    with pytest.raises(RuntimeError, match="Predecessor cleanup unresolved"):
        sweeps.start(db, child, 2, detached=False, after_sweep=parent)
    executor.assert_not_called()


def test_matrix_preparation_freezes_source_matrix_and_models(setup):
    db, api, key, _ = setup
    jid = sweeps.prepare(db, api, gpu_types=[CATALOG[0]["id"]], matrix="text_gen", ssh_key=key)
    value = sweeps.job(db, jid)
    assert json.loads(value["options"])["matrix_config_id"] == db.config("text_gen", "matrix")["id"]
    items = db.conn.execute("SELECT * FROM sweep_items WHERE sweep_id=? ORDER BY ordinal", (jid,)).fetchall()
    assert len(items) == 2
    assert items[1]["status"] == "blocked"  # TP2 32B is blocked on this 24GB device.
    item = items[0]
    spec = json.loads(item["spec"])
    assert len(spec["workloads"]) == 6  # Five fixed 8B profiles plus mixed ShareGPT.
    assert len(json.loads(item["assessments"])) == 15
    assert all(w["model"] == "Qwen/Qwen3-8B" for w in spec["workloads"])
    api.create.assert_not_called()


def test_frozen_legacy_sweep_above_global_cap_cannot_launch(setup):
    db, api, key, _ = setup
    jid = sweeps.prepare(db, api, gpu_types=[CATALOG[0]["id"]], benchmark_type="text_gen", ssh_key=key)
    item = db.conn.execute("SELECT ordinal,spec FROM sweep_items WHERE sweep_id=? LIMIT 1", (jid,)).fetchone()
    spec = json.loads(item["spec"])
    spec["workloads"][0]["saturation_max_concurrency"] = 1024
    with db.conn:
        db.conn.execute(
            "UPDATE sweep_items SET spec=? WHERE sweep_id=? AND ordinal=?",
            (json.dumps(spec), jid, item["ordinal"]),
        )
    with pytest.raises(ValueError, match="global concurrency cap 512"):
        sweeps.start(db, jid, budget=10000)
    api.create.assert_not_called()


def test_matrix_tp_axis_freezes_actual_devices_and_prices(setup):
    db, api, key, _ = setup
    api.gpus.return_value = [{**CATALOG[0], "memoryInGb": 80, "gpuGeneration": "HOPPER"}]
    jid = sweeps.prepare(
        db, api, gpu_types=[CATALOG[0]["id"]], matrix="text_gen", ssh_key=key, max_minutes=60, disk_gb=200
    )
    specs = [
        json.loads(r[0])
        for r in db.conn.execute("SELECT spec FROM sweep_items WHERE sweep_id=? ORDER BY ordinal", (jid,))
    ]
    assert [s["target"]["gpu_count"] for s in specs] == [1, 2]
    assert [s["gpus_per_replica"] for s in specs] == [1, 2]
    assert len(specs[0]["workloads"]) == 10
    assert len(specs[1]["workloads"]) == 5
    assert all(w["model"] == "Qwen/Qwen3-32B" for w in specs[1]["workloads"])
    assert specs[0]["max_hourly_price"] == pytest.approx(1.24)
    assert specs[1]["max_hourly_price"] == pytest.approx(2.44)
    assert sweeps.job(db, jid)["estimated_ceiling"] == pytest.approx(3.68)
    api.create.assert_not_called()


def test_one_command_detached_sweep_does_not_require_enumerating_workloads(setup, monkeypatch):
    db, api, key, _ = setup
    monkeypatch.setattr("sls_bench.cli.Runpod", Mock(return_value=api))
    starter = Mock(return_value=1234)
    monkeypatch.setattr(sweeps, "start", starter)
    result = CliRunner().invoke(
        app,
        [
            "--db",
            str(db.path),
            "sweep",
            "--gpu",
            CATALOG[0]["id"],
            "--type",
            "image_gen",
            "--suite",
            "image-smoke",
            "--detach",
            "--yes",
            "--budget",
            "2",
            "--ssh-key",
            str(key),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "background PID 1234" in result.output
    assert starter.call_args.kwargs["detached"] is True
    api.create.assert_not_called()
