import copy
import json
import subprocess
from unittest.mock import Mock

import pytest

from sls_bench import worker
from sls_bench.config import Text

from .test_text_time_budget import run
from .test_text_time_budget import runtime as runtime


def test_plan_shares_largest_context_and_explicit_capacity_without_mutating_profiles():
    profiles = [
        {**Text(name="chat").model_dump(), "config_id": 1},
        {**Text(name="long", max_model_len=32768, max_num_seqs=64).model_dump(), "config_id": 2},
    ]
    original = copy.deepcopy(profiles)
    plan = worker.text_engine_plan(profiles)
    assert len(plan) == 1
    assert plan[0]["config_ids"] == [1, 2]
    assert plan[0]["engine_config"]["max_model_len"] == 32768
    assert plan[0]["engine_config"]["max_num_seqs"] == 64
    assert plan[0]["engine_config"]["max_num_batched_tokens"] == 8192
    assert profiles == original


@pytest.mark.parametrize(
    "change",
    [
        {"model": "another/model"},
        {"model_revision": "other"},
        {"dtype": "half"},
        {"quantization": "fp8"},
        {"gpu_memory_utilization": 0.85},
        {"kv_cache_dtype": "other"},
        {"prefix_caching": True},
        {"seed": 24},
        {"max_num_batched_tokens": 4096},
        {"enable_chunked_prefill": False},
    ],
)
def test_incompatible_engine_settings_are_separate_groups(change):
    config = {**Text(name="base").model_dump(), "config_id": 1}
    assert len(worker.text_engine_plan([config, {**config, **change, "config_id": 2}])) == 2


def test_fixed_engine_schedule_preserves_capacity_and_frozen_legacy_specs_keep_old_policy():
    config = {
        **Text(
            name="schedule",
            requests=512,
            concurrency=[1],
            saturation_max_concurrency=512,
            coverage_first=True,
            measurement_sample_policy="full_or_smoke",
            measurement_budget_seconds=900,
        ).model_dump(),
        "config_id": 1,
    }
    points = worker.coverage_schedule([config])
    assert [c["concurrency"][0] for c in points] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    assert all(c["max_num_seqs"] == 16 for c in points)
    del config["engine_capacity_policy"]
    assert all(c["max_num_seqs"] == c["concurrency"][0] for c in worker.coverage_schedule([config]))
    assert worker.text_engine_plan([config]) == []


def spies(monkeypatch):
    start = Mock(wraps=worker.start)
    stop = Mock(wraps=worker.stop)
    monkeypatch.setattr(worker, "start", start)
    monkeypatch.setattr(worker, "stop", stop)
    return start, stop


def test_standalone_search_reuses_one_engine_above_its_scheduling_capacity(runtime, monkeypatch):
    config, calls, _ = runtime
    config.update(
        requests=8,
        saturation_max_concurrency=8,
        max_num_seqs=2,
        coverage_first=True,
        measurement_sample_policy="full_or_smoke",
    )
    start, stop = spies(monkeypatch)
    worker.text_workload(config, {"gpu_count": 1}, {(config["model"], "fixed"): "/model"})
    assert start.call_count == stop.call_count == 1
    assert len(calls) == 12  # A fresh pilot and both repeats at every level.
    trials = worker.REPORT["trials"]
    assert {t["concurrency"] for t in trials} == {1, 2, 4, 8}
    assert all(t["engine_max_num_seqs"] == 2 for t in trials)
    assert all(t["engine_instance_id"] == 1 for t in trials)
    assert [t["engine_reused"] for t in trials] == [False, False] + [True] * 6
    command = start.call_args.args[0]
    assert command[command.index("--max-num-batched-tokens") + 1] == "8192"
    assert "--enable-chunked-prefill" in command
    assert all(t["status"] == "completed" for t in trials)


def main_setup(config, monkeypatch):
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(worker.signal, "alarm", lambda *a: None)
    monkeypatch.setattr(worker, "hardware", lambda: {})
    monkeypatch.setattr(worker, "verify_hardware", lambda *a: None)
    monkeypatch.setattr(worker, "download", lambda *a: "/model")
    second = {**config, "config_id": 2, "max_model_len": 32768}
    (worker.ROOT / "spec.json").write_text(
        json.dumps(
            {
                "workloads": [config, second],
                "target": {"gpu_count": 1},
                "text_engine_plan": worker.text_engine_plan([config, second]),
            }
        )
    )


def test_worker_reuses_largest_context_engine_across_profiles(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(
        requests=8,
        saturation_max_concurrency=4,
        max_num_seqs=2,
        coverage_first=True,
        measurement_sample_policy="full_or_smoke",
    )
    main_setup(config, monkeypatch)
    start, stop = spies(monkeypatch)
    worker.main()
    assert start.call_count == stop.call_count == 1
    assert config["max_model_len"] == 4096
    trials = worker.REPORT["trials"]
    assert len(trials) == 12
    assert all(t["engine_config"]["max_model_len"] == 32768 for t in trials)
    assert all(t["engine_config"]["max_num_seqs"] == 2 for t in trials)
    phases = worker.REPORT["phases"]
    assert len([p for p in phases if p["name"] == "server_load_compile"]) == 1
    assert len([p for p in phases if p["name"] == "server_reuse"]) == 5
    assert all(p["status"] == "completed" for p in worker.REPORT["coverage_plan"])


def test_timeout_discards_possibly_queued_work_before_next_point(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(requests=8, coverage_first=True, measurement_sample_policy="full_or_smoke")
    original = worker.command

    def timeout(args, *a, **kw):
        filename = args[args.index("--result-filename") + 1]
        if "warmup" not in filename:
            raise subprocess.TimeoutExpired(args, 300)
        return original(args, *a, **kw)

    start, stop = spies(monkeypatch)
    with worker.TextEngineSession() as session:
        config["_engine_session"] = session
        monkeypatch.setattr(worker, "command", timeout)
        with pytest.raises(worker.TextTimeBudget):
            run(config)
        assert session.process is None
        assert stop.call_count == 1
        monkeypatch.setattr(worker, "command", original)
        run(config)
        assert start.call_count == 2
        assert worker.REPORT["trials"][-1]["engine_instance_id"] == 2
    assert stop.call_count == 2


@pytest.mark.parametrize("dead", [False, True])
def test_changed_command_or_dead_engine_requires_restart(runtime, monkeypatch, dead):
    start, stop = spies(monkeypatch)
    with worker.TextEngineSession() as session:
        process, reused, instance = session.acquire(["vllm", "serve", "one"], "server.log", 10)
        assert not reused and instance == 1
        if dead:
            process.poll = lambda: 1
        args = ["vllm", "serve", "one" if dead else "two"]
        _, reused, instance = session.acquire(args, "next.log", 10)
        assert not reused and instance == 2
    assert start.call_count == stop.call_count == 2


def test_failed_engine_startup_is_not_retried_across_profiles(runtime, monkeypatch):
    start, stop = spies(monkeypatch)
    monkeypatch.setattr(worker, "ready", Mock(side_effect=RuntimeError("Insufficient KV cache")))
    with worker.TextEngineSession() as session:
        with pytest.raises(RuntimeError, match="Insufficient KV cache"):
            session.acquire(["vllm", "serve", "one"], "first.log", 10)
        with pytest.raises(RuntimeError, match="circuit open"):
            session.acquire(["vllm", "serve", "one"], "other-profile.log", 10)
        assert start.call_count == stop.call_count == 1
        monkeypatch.setattr(worker, "ready", Mock())
        session.acquire(["vllm", "serve", "different"], "different.log", 10)
        assert start.call_count == 2
    assert stop.call_count == 2


def test_cancellation_closes_the_shared_engine(runtime, monkeypatch):
    config, _, _ = runtime

    def cancelled(*a, **kw):
        raise worker.WorkerCancelled("cancelled")

    monkeypatch.setattr(worker, "command", cancelled)
    start, stop = spies(monkeypatch)
    with pytest.raises(worker.WorkerCancelled):
        with worker.TextEngineSession() as session:
            config["_engine_session"] = session
            run(config)
    assert start.call_count == stop.call_count == 1


def test_intentionally_tiny_planned_counts_are_labeled_smoke(runtime):
    config, _, _ = runtime
    config.update(requests=2, measurement_sample_policy="full_or_smoke", coverage_first=True)
    run(config)
    assert all(t["sample_class"] == "smoke" for t in worker.REPORT["trials"])
    assert all(not t["reduced_sample"] for t in worker.REPORT["trials"])
    assert all("tail latency not established" in t["sample_size_warning"] for t in worker.REPORT["trials"])


def test_non_chunked_prefill_requires_sufficient_token_budget():
    with pytest.raises(ValueError, match="batched-token budget"):
        Text(name="invalid", max_model_len=32768, enable_chunked_prefill=False)


def test_cache_reset_occurs_after_completed_point_before_next_warmup(runtime, monkeypatch):
    config, calls, _ = runtime
    config.update(
        requests=8,
        saturation_max_concurrency=2,
        prefix_caching=True,
        coverage_first=True,
        measurement_sample_policy="full_or_smoke",
    )
    resets = []
    monkeypatch.setattr(
        worker, "reset_text_prefix_cache", lambda: resets.append((len(calls), len(worker.REPORT["trials"])))
    )
    worker.text_workload(config, {"gpu_count": 1}, {(config["model"], "fixed"): "/model"})
    assert resets == [(3, 2)]  # Pilot + both measured repeats have drained.
    assert len(calls) == 6


def test_schedule_finishes_compatible_group_before_switching_model(runtime):
    config, _, _ = runtime
    config.update(
        saturation_max_concurrency=2, coverage_first=True, measurement_sample_policy="full_or_smoke"
    )
    other = {**config, "config_id": 2, "model": "another/model"}
    compatible = {**config, "config_id": 3, "max_model_len": 32768}
    points = worker.coverage_schedule([config, other, compatible])
    assert [(p["config_id"], p["concurrency"][0]) for p in points] == [
        (1, 1),
        (3, 1),
        (1, 2),
        (3, 2),
        (2, 1),
        (2, 2),
    ]


def test_shared_text_engine_released_before_another_gpu_adapter(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(requests=8, coverage_first=True, measurement_sample_policy="full_or_smoke")
    main_setup(config, monkeypatch)
    image = {"kind": "image", "benchmark_type": "image_gen", "config_id": 2, "concurrency": [1], "repeats": 1}
    (worker.ROOT / "spec.json").write_text(
        json.dumps({"workloads": [config, image], "target": {"gpu_count": 1}})
    )
    start, stop = spies(monkeypatch)

    def image_workload(c):
        assert stop.call_count == 1
        worker.REPORT["trials"].append({"status": "completed", "config_id": c["config_id"]})

    monkeypatch.setattr(worker, "image_workload", image_workload)
    worker.main()
    assert start.call_count == stop.call_count == 1
