import json
from unittest.mock import Mock

import httpx
import pytest

from sls_bench import controller
from sls_bench.db import Database
from sls_bench.runpod import SupplyUnavailable

from .helpers import seed_test_targets


@pytest.fixture
def setup(tmp_path, monkeypatch):
    db = Database(tmp_path / "bench.sqlite3")
    db.seed()
    seed_test_targets(db)
    target = db.config("4090")
    workload = db.config("qwen8b-chat")
    spec = {
        "code_sha256": "test",
        "target": json.loads(target["spec"]),
        "workloads": [],
        "max_hourly_price": 1,
        "sources": {"worker.py": "captured-source"},
        "replica_count": 1,
        "gpus_per_replica": 1,
    }
    key = tmp_path / "id_ed25519"
    key.write_text("private")
    key.with_suffix(".pub").write_text("public")
    api = Mock()
    pod = {
        "id": "pod",
        "publicIp": "127.0.0.1",
        "portMappings": {"22": 22},
        "costPerHr": 0.89,
        "machine": {"gpuTypeId": "NVIDIA GeForce RTX 4090"},
    }
    api.create.return_value = pod
    api.pod.return_value = pod
    api.pods.return_value = []
    ssh = Mock()
    ssh.call.return_value = b"python3 vllm"
    monkeypatch.setattr(controller, "SSH", Mock(return_value=ssh))
    monkeypatch.setattr(controller, "watchdog", Mock())
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    yield db, target, workload, spec, api, key, ssh
    db.conn.close()


def test_success_collects_and_deletes_owned_pod(setup, monkeypatch):
    db, target, workload, spec, api, key, ssh = setup
    run_id = db.new_run(target, [workload], spec, 60)
    report = {"status": "completed", "trials": []}
    collector = Mock(return_value=report)
    monkeypatch.setattr(controller, "collect", collector)
    assert controller.execute(db, api, run_id, key) == report
    api.delete.assert_called_once_with("pod")
    assert db.run(run_id)["status"] == "completed"
    assert db.run(run_id)["resource_state"] == "deleted"
    assert db.run(run_id)["finished_at"] is not None
    assert collector.call_args.kwargs["artifacts"] is True
    assert any(call.args[1:] == (b"captured-source",) for call in ssh.call.call_args_list)


def test_wrong_gpu_still_deleted(setup):
    db, target, workload, spec, api, key, ssh = setup
    run_id = db.new_run(target, [workload], spec, 60)
    api.pod.return_value["machine"]["gpuTypeId"] = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    with pytest.raises(RuntimeError, match="Wrong GPU"):
        controller.execute(db, api, run_id, key)
    api.delete.assert_called_once_with("pod")
    assert db.run(run_id)["status"] == "failed"


def test_supply_unavailable_is_not_a_benchmark_failure(setup):
    db, target, workload, spec, api, key, ssh = setup
    run_id = db.new_run(target, [workload], spec, 60)
    api.create.side_effect = SupplyUnavailable("No stock")
    with pytest.raises(SupplyUnavailable):
        controller.execute(db, api, run_id, key)
    assert db.run(run_id)["status"] == "unavailable"
    api.delete.assert_not_called()
    assert db.run(run_id)["resource_state"] == "deleted"


def test_ambiguous_create_is_retained_until_a_late_pod_can_be_reconciled(setup):
    db, target, workload, spec, api, key, ssh = setup
    run_id = db.new_run(target, [workload], spec, 60)
    api.create.side_effect = httpx.ReadTimeout("lost response")
    with pytest.raises(RuntimeError, match="ambiguous"):
        controller.execute(db, api, run_id, key)
    assert api.create.call_count == 1
    assert db.run(run_id)["resource_state"] == "creation_ambiguous"
    api.delete.assert_not_called()
    api.pods.return_value = [{"name": db.run(run_id)["pod_name"], "id": "late-pod"}]
    controller.cleanup(db, api, run_id)
    api.delete.assert_called_once_with("late-pod")
    assert db.run(run_id)["resource_state"] == "deleted"


def test_ctrl_c_attached_cancels_worker_not_pod(setup, monkeypatch):
    db, target, workload, spec, api, key, ssh = setup
    run_id = db.new_run(target, [workload], spec, 60, owned=False, pod_id="pod")
    monkeypatch.setattr(
        controller, "collect", Mock(side_effect=[KeyboardInterrupt(), RuntimeError("offline")])
    )
    with pytest.raises(KeyboardInterrupt):
        controller.execute(db, api, run_id, key)
    api.create.assert_not_called()
    api.delete.assert_not_called()
    assert db.run(run_id)["status"] == "interrupted"
    assert db.run(run_id)["resource_state"] == "detached"
    assert any("kill -TERM" in call.args[0] for call in ssh.call.call_args_list)


def test_actual_price_over_budget_deletes_pod(setup):
    db, target, workload, spec, api, key, ssh = setup
    run_id = db.new_run(target, [workload], spec, 60)
    api.pod.return_value["costPerHr"] = 99
    with pytest.raises(RuntimeError, match="hourly price"):
        controller.execute(db, api, run_id, key)
    api.delete.assert_called_once_with("pod")
