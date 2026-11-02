import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sls_bench import worker
from sls_bench.config import Text


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    clock = [0.0]
    calls = []
    config = Text(
        name="budget", concurrency=[1], requests=512, repeats=2, measurement_budget_seconds=300
    ).model_dump()
    config.update(config_id=1, model_revision="fixed")
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "REPORT", {"trials": [], "phases": []})
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    monkeypatch.setattr(worker, "start", lambda *a: Mock(poll=lambda: None))
    monkeypatch.setattr(worker, "stop", lambda *a: None)
    monkeypatch.setattr(worker, "ready", lambda *a: None)
    monkeypatch.setattr(worker, "corpus", lambda *a: tmp_path / "dataset")
    monkeypatch.setattr(worker, "lengths_match", lambda *a: True)
    monkeypatch.setattr(worker, "Telemetry", lambda *a: nullcontext())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: object())),
    )

    def command(args, log, timeout, **kwargs):
        calls.append((args, timeout))
        count = int(args[args.index("--num-prompts") + 1])
        clock[0] += 10
        filename = args[args.index("--result-filename") + 1]
        (tmp_path / filename).write_text(
            json.dumps(
                {
                    "completed": count,
                    "input_lens": [512] * count,
                    "output_lens": [256] * count,
                    "output_throughput": 100,
                    "duration": 5.0,
                }
            )
        )

    monkeypatch.setattr(worker, "command", command)
    return config, calls, clock


def run(config):
    worker.fixed_text_workload(config, {"gpu_count": 1}, {(config["model"], "fixed"): "/model"})


def test_slow_pilot_skips_without_running_full_samples(runtime):
    config, calls, _ = runtime
    with pytest.raises(worker.TextTimeBudget):
        run(config)
    assert len(calls) == 1
    skipped = worker.REPORT["trials"][0]
    assert skipped["status"] == "skipped_time_budget"
    assert "metrics" not in skipped
    assert skipped["requested_requests"] == 512
    assert worker.REPORT["throughput_probes"][0]["estimated_measurement_seconds"] == 2954
    assert calls[0][1] <= 120


def test_fast_point_keeps_configured_counts_and_bounded_timeouts(runtime):
    config, calls, _ = runtime
    config["requests"] = 8
    run(config)
    assert len(calls) == 3
    assert all(t["status"] == "completed" for t in worker.REPORT["trials"])
    assert all(t["requested_requests"] == 8 for t in worker.REPORT["trials"])
    assert calls[1][1] <= 290
    assert calls[2][1] <= 280


def test_pilot_is_matched_to_actual_concurrency(runtime):
    config, calls, _ = runtime
    config.update(concurrency=[8], max_num_seqs=8, requests=512)
    with pytest.raises(worker.TextTimeBudget):
        run(config)
    args = calls[0][0]
    assert args[args.index("--max-concurrency") + 1] == "8"
    assert args[args.index("--num-prompts") + 1] == "8"


def test_fixed_overhead_is_not_multiplied_by_the_full_request_count():
    # Replay the actual 5090/C128 evidence that the first estimator rejected.
    estimate = worker.text_measurement_estimate(
        {"requests": 1024, "repeats": 2}, {"duration": 24.6007149819}, 256, 35.836045321
    )
    assert estimate["estimated_measurement_seconds"] == pytest.approx(248.797238512)
    assert estimate["estimated_measurement_seconds"] < 300 - 35.836045321
    assert 35.836045321 * 1024 / 256 * 2 * 1.25 > 300 - 35.836045321


@pytest.mark.parametrize("duration", [None, 0, -1, float("nan"), float("inf")])
def test_missing_or_invalid_serving_duration_is_not_used_as_a_prediction(duration):
    with pytest.raises(RuntimeError, match="duration"):
        worker.text_measurement_estimate({"requests": 1024, "repeats": 2}, {"duration": duration}, 128, 20)


def test_single_wave_above_old_60s_cutoff_can_admit_and_finish(runtime, monkeypatch):
    config, calls, clock = runtime
    config.update(concurrency=[512], max_num_seqs=512, requests=512)
    original = worker.command

    def command(args, *a, **kw):
        assert kw.get("timeout", a[1] if len(a) > 1 else 0) >= 75
        original(args, *a, **kw)
        clock[0] += 65  # 75 seconds wall, including 10 seconds fixed overhead.
        filename = args[args.index("--result-filename") + 1]
        path = worker.ROOT / filename
        raw = json.loads(path.read_text())
        raw["duration"] = 65
        path.write_text(json.dumps(raw))

    monkeypatch.setattr(worker, "command", command)
    run(config)
    assert len(calls) == 3
    assert calls[0][1] == 120
    assert all(t["status"] == "completed" for t in worker.REPORT["trials"])


def test_pilot_timeout_is_a_skip_not_a_failed_or_completed_measurement(runtime, monkeypatch):
    config, _, _ = runtime

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("pilot", 60)

    monkeypatch.setattr(worker, "command", timeout)
    with pytest.raises(worker.TextTimeBudget):
        run(config)
    assert worker.REPORT["trials"][0]["status"] == "skipped_time_budget"


def test_shrinking_retains_c1_measurements_and_records_actual_counts(runtime):
    config, calls, _ = runtime
    config.update(measurement_sample_policy="shrink_to_budget", coverage_first=True)
    run(config)
    assert len(calls) == 3  # Pilot plus two actual measured repeats, not a skip.
    trials = worker.REPORT["trials"]
    assert len(trials) == 2
    assert all(t["status"] == "completed" and t["concurrency"] == 1 for t in trials)
    assert all(t["planned_requests"] == 512 and t["requested_requests"] == 48 for t in trials)
    assert all(t["reduced_sample"] and t["sample_size_warning"] for t in trials)
    assert all(t["metrics"]["expected_requests"] == len(t["requests"]) == 48 for t in trials)
    assert worker.REPORT["throughput_probes"][0]["decision"] == "shrink_sample"


def test_subsequent_repeat_refits_using_actual_speed_and_remaining_budget(runtime, monkeypatch):
    config, calls, clock = runtime
    config.update(measurement_sample_policy="shrink_to_budget", coverage_first=True)
    original = worker.command

    def command(args, *a, **kw):
        original(args, *a, **kw)
        if len(calls) == 2:
            clock[0] += 160  # First measured repeat is slower than the pilot predicted.
            path = worker.ROOT / args[args.index("--result-filename") + 1]
            raw = json.loads(path.read_text())
            raw["duration"] = 165
            path.write_text(json.dumps(raw))

    monkeypatch.setattr(worker, "command", command)
    run(config)
    trials = worker.REPORT["trials"]
    assert all(t["status"] == "completed" for t in trials)
    assert trials[0]["requested_requests"] == 48
    assert 1 <= trials[1]["requested_requests"] < 48
    assert all(t["planned_requests"] == 512 for t in trials)


def test_shrinking_never_lowers_target_concurrency_or_repeats(runtime):
    config, calls, _ = runtime
    config.update(
        concurrency=[128], max_num_seqs=128, measurement_sample_policy="shrink_to_budget", coverage_first=True
    )
    run(config)
    assert len(calls) == 3
    for trial in worker.REPORT["trials"]:
        assert trial["status"] == "completed"
        assert trial["concurrency"] == trial["engine_max_num_seqs"] == 128
        assert trial["requested_requests"] >= 128
    prediction = worker.text_measurement_estimate(config, {"duration": 1000}, 128, 1001)
    assert worker.text_budget_requests(config, prediction, 128, 100, 128) == 128


def test_pilot_timeout_falls_back_to_real_minimum_measurement(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(measurement_sample_policy="shrink_to_budget", coverage_first=True)
    original = worker.command
    calls = [0]

    def command(*a, **kw):
        calls[0] += 1
        if calls[0] == 1:
            raise subprocess.TimeoutExpired("pilot", 120)
        original(*a, **kw)

    monkeypatch.setattr(worker, "command", command)
    run(config)
    assert calls[0] == 3
    assert all(t["status"] == "completed" and t["requested_requests"] == 1 for t in worker.REPORT["trials"])
    assert worker.REPORT["throughput_probes"][0]["decision"] == "pilot_timeout_minimum_sample"


def test_real_measurement_timeout_is_explicitly_incomplete_not_a_skip(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(measurement_sample_policy="shrink_to_budget", coverage_first=True)
    original = worker.command
    calls = [0]

    def command(*a, **kw):
        calls[0] += 1
        if calls[0] == 2:
            raise subprocess.TimeoutExpired("measurement", 280)
        original(*a, **kw)

    monkeypatch.setattr(worker, "command", command)
    with pytest.raises(worker.TextTimeBudget):
        run(config)
    trial = worker.REPORT["trials"][0]
    assert trial["status"] == "incomplete_time_budget"
    assert "metrics" not in trial


def test_coverage_first_ignores_throughput_floor_and_plateau(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(
        measurement_sample_policy="shrink_to_budget",
        coverage_first=True,
        saturation_max_concurrency=8,
        saturation_min_output_tok_s=25,
    )
    visited = []

    def point(level, *a):
        c = level["concurrency"][0]
        visited.append(c)
        for repeat in range(2):
            worker.REPORT["trials"].append(
                {
                    "status": "completed",
                    "repeat": repeat,
                    "requested_requests": 8,
                    "reduced_sample": True,
                    "metrics": {"output_throughput": 1},
                }
            )

    monkeypatch.setattr(worker, "fixed_text_workload", point)
    worker.text_workload(config, {}, {})
    assert visited == [1, 2, 4, 8]
    assert len(worker.REPORT["saturation"][0]["history"]) == 4


def test_mixed_arrival_rate_shrinks_samples_without_changing_rate(runtime):
    config, calls, _ = runtime
    config.update(
        measurement_sample_policy="shrink_to_budget",
        coverage_first=True,
        load_mode="open_loop_request_rate",
        dataset_generator="sharegpt_v1",
        request_rates=[0.01],
    )
    run(config)
    assert len(calls) == 3
    for trial in worker.REPORT["trials"]:
        assert trial["status"] == "completed" and trial["requested_requests"] == 1
        assert trial["request_rate"] == 0.01 and trial["planned_requests"] == 512


def test_each_rate_is_attempted_even_after_a_failed_point(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(
        measurement_sample_policy="shrink_to_budget",
        coverage_first=True,
        load_mode="open_loop_request_rate",
        request_rates=[0.5, 1.0],
    )
    visited = []

    def point(level, *a):
        visited.append(level["request_rates"])
        if len(visited) == 1:
            raise RuntimeError("setup failed")

    monkeypatch.setattr(worker, "fixed_text_workload", point)
    worker.text_workload(config, {}, {})
    assert visited == [[0.5], [1.0]]
    assert worker.REPORT["trials"][0]["status"] == "setup_failed"
    assert worker.REPORT["trials"][0]["request_rate"] == 0.5


def test_coverage_schedule_visits_all_profiles_before_higher_concurrency(runtime):
    config, _, _ = runtime
    config.update(
        measurement_sample_policy="shrink_to_budget",
        coverage_first=True,
        saturation_max_concurrency=4,
        requests_by_concurrency={1: 128},
    )
    other = {**config, "config_id": 2}
    scheduled = worker.coverage_schedule([config, other])
    assert [(c["config_id"], c["concurrency"][0]) for c in scheduled] == [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
        (1, 4),
        (2, 4),
    ]
    assert scheduled[0]["requests"] == scheduled[1]["requests"] == 128
    assert scheduled[2]["requests"] == 512
    assert config["concurrency"] == [1] and config["requests_by_concurrency"] == {1: 128}


def test_explicit_four_point_grid_preserves_counts_and_repeats(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(
        concurrency=[1, 8, 48, 128],
        saturation_max_concurrency=None,
        requests_by_concurrency={1: 2, 8: 8, 48: 48, 128: 128},
        coverage_first=True,
        measurement_sample_policy="full_or_smoke",
    )
    scheduled = worker.coverage_schedule([config])
    assert [c["concurrency"][0] for c in scheduled] == [1, 8, 48, 128]
    assert [c["requests"] for c in scheduled] == [2, 8, 48, 128]
    assert all(c["repeats"] == config["repeats"] for c in scheduled)
    assert all(c["saturation_max_concurrency"] is None for c in scheduled)
    visited = []
    monkeypatch.setattr(worker, "fixed_text_workload", lambda c, *_: visited.append(c))
    worker.text_workload(config, {}, {})
    assert [c["concurrency"][0] for c in visited] == [1, 8, 48, 128]
    assert [c["requests"] for c in visited] == [2, 8, 48, 128]


def test_worker_persists_an_explicit_breadth_first_coverage_plan(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(
        measurement_sample_policy="shrink_to_budget", coverage_first=True, saturation_max_concurrency=2
    )
    second = {**config, "config_id": 2}
    spec = {"workloads": [config, second], "target": {}, "max_minutes": 480}
    (worker.ROOT / "spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(worker.signal, "alarm", lambda *a: None)
    monkeypatch.setattr(worker, "hardware", lambda: {})
    monkeypatch.setattr(worker, "verify_hardware", lambda *a: None)
    visited = []

    def point(c, *a):
        visited.append((c["config_id"], c["concurrency"][0]))
        for repeat in range(2):
            worker.REPORT["trials"].append({"status": "completed", "repeat": repeat})

    monkeypatch.setattr(worker, "text_workload", point)
    worker.main()
    assert visited == [(1, 1), (2, 1), (1, 2), (2, 2)]
    assert all(p["status"] == "completed" for p in worker.REPORT["coverage_plan"])
    assert worker.REPORT["status"] == "completed"


@pytest.mark.parametrize("minimum,expected", [(1, 2), (2, 2), (4, 4), (128, 128), (512, 512)])
def test_full_or_smoke_uses_tiny_fallback_not_the_largest_fitting_count(minimum, expected):
    config = {"requests": 1024, "measurement_sample_policy": "full_or_smoke"}
    assert worker.text_sample_requests(config, 600, minimum) == expected
    assert worker.text_sample_requests(config, 1024, minimum) == 1024
    assert worker.text_sample_requests(config, 0, minimum) == expected
    assert worker.text_sample_requests({**config, "requests": 1}, 0, 1) == 1


def test_full_or_smoke_c1_measures_two_requests_per_repeat(runtime):
    config, calls, clock = runtime
    config.update(measurement_sample_policy="full_or_smoke", coverage_first=True)
    run(config)
    assert len(calls) == 3
    assert clock[0] == 30  # Does not attempt to fill the 300-second ceiling.
    for t in worker.REPORT["trials"]:
        assert t["status"] == "completed"
        assert t["concurrency"] == 1 and t["requested_requests"] == 2
        assert t["planned_requests"] == 512 and t["sample_class"] == "smoke"
        assert t["reduced_sample"] and "tail latency" in t["sample_size_warning"]
        assert len(t["requests"]) == t["metrics"]["expected_requests"] == 2
    assert worker.REPORT["throughput_probes"][0]["decision"] == "smoke_sample"


def test_full_or_smoke_keeps_full_sample_if_it_fits(runtime):
    config, _, _ = runtime
    config.update(measurement_sample_policy="full_or_smoke", coverage_first=True, requests=8)
    run(config)
    assert all(
        t["requested_requests"] == 8 and t["sample_class"] == "full" and not t["reduced_sample"]
        for t in worker.REPORT["trials"]
    )


def test_full_or_smoke_pilot_timeout_still_attempts_two_measured_requests(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(measurement_sample_policy="full_or_smoke", coverage_first=True)
    original = worker.command
    calls = [0]

    def command(*a, **kw):
        calls[0] += 1
        if calls[0] == 1:
            raise subprocess.TimeoutExpired("pilot", 120)
        original(*a, **kw)

    monkeypatch.setattr(worker, "command", command)
    run(config)
    assert calls[0] == 3
    assert all(t["status"] == "completed" and t["requested_requests"] == 2 for t in worker.REPORT["trials"])


def test_full_or_smoke_mixed_rate_uses_two_not_the_arrival_budget_maximum(runtime):
    config, calls, _ = runtime
    config.update(
        measurement_sample_policy="full_or_smoke",
        coverage_first=True,
        requests=8,
        load_mode="open_loop_request_rate",
        dataset_generator="sharegpt_v1",
        request_rates=[0.01],
    )
    run(config)
    assert len(calls) == 3
    assert all(
        t["requested_requests"] == 2 and t["request_rate"] == 0.01 and t["sample_class"] == "smoke"
        for t in worker.REPORT["trials"]
    )


def test_full_or_smoke_refits_to_two_after_a_slow_first_repeat(runtime, monkeypatch):
    config, calls, clock = runtime
    config.update(measurement_sample_policy="full_or_smoke", coverage_first=True, requests=8)
    original = worker.command

    def command(args, *a, **kw):
        original(args, *a, **kw)
        if len(calls) == 2:
            clock[0] += 260
            path = worker.ROOT / args[args.index("--result-filename") + 1]
            raw = json.loads(path.read_text())
            raw["duration"] = 265
            path.write_text(json.dumps(raw))

    monkeypatch.setattr(worker, "command", command)
    run(config)
    assert [t["requested_requests"] for t in worker.REPORT["trials"]] == [8, 2]
    assert [t["sample_class"] for t in worker.REPORT["trials"]] == ["full", "smoke"]


def test_full_or_smoke_config_requires_budget():
    with pytest.raises(ValueError, match="budget"):
        Text(name="bad", measurement_sample_policy="full_or_smoke", coverage_first=True)
    assert (
        Text(
            name="good",
            measurement_sample_policy="full_or_smoke",
            coverage_first=True,
            measurement_budget_seconds=900,
        ).measurement_sample_policy
        == "full_or_smoke"
    )


def test_shrinking_config_requires_explicit_budget_and_policy():
    with pytest.raises(ValueError, match="budget"):
        Text(name="bad", measurement_sample_policy="shrink_to_budget")
    with pytest.raises(ValueError, match="shrink_to_budget"):
        Text(name="bad", coverage_first=True)


def test_budget_skips_do_not_stop_higher_concurrency_search(runtime, monkeypatch):
    config, _, _ = runtime
    config.update(saturation_max_concurrency=4, requests_by_concurrency={})
    levels = []

    def point(level, *a):
        levels.append(level["concurrency"][0])
        raise worker.TextTimeBudget("too long")

    monkeypatch.setattr(worker, "fixed_text_workload", point)
    worker.text_workload(config, {}, {})
    assert levels == [1, 2, 4]
    assert worker.REPORT["saturation"][0]["reason"] == "time_budget_limited"


@pytest.mark.parametrize("concurrency", [1, 8])
def test_cliff_pilot_forces_minimum_even_when_full_sample_fits(runtime, concurrency):
    config, calls, _ = runtime
    config.update(
        requests=64,
        concurrency=[concurrency],
        max_num_seqs=concurrency,
        measurement_sample_policy="full_or_smoke",
        coverage_first=True,
        cliff_aware=True,
        _boundary_state={"best_output_tok_s": 1000},
    )
    run(config)
    assert len(calls) == 3
    assert all(t["requested_requests"] == max(2, concurrency) for t in worker.REPORT["trials"])
    assert all(t["candidate_stopping_reason"] == "throughput_collapse" for t in worker.REPORT["trials"])
    assert all(t["sample_class"] == "smoke" for t in worker.REPORT["trials"])
    assert all(t["status"] == "completed" for t in worker.REPORT["trials"])


def test_plateau_requires_two_measured_levels_and_pilot_alone_cannot_stop():
    config = Text(name="cliff").model_dump()
    state = {"best_output_tok_s": 100}
    assert worker.text_boundary_reason(config, 102, state) is None

    def trials(rate):
        return [{"status": "completed", "metrics": {"output_throughput": rate}} for _ in range(3)]

    assert worker.record_text_boundary(config, trials(102), state) is None
    assert worker.text_boundary_reason(config, 103, state) == "throughput_plateau"
    # A healthy real measurement contradicts the pilot: continue and reset plateau.
    assert worker.record_text_boundary(config, trials(120), state) is None
    assert state["plateau_steps"] == 0
    assert worker.record_text_boundary(config, trials(121), state) is None
    measured = trials(122)
    assert worker.record_text_boundary(config, measured, state) == "throughput_plateau"
    assert all(t["stopping_reason"] == "throughput_plateau" for t in measured)


@pytest.mark.parametrize("failed", [False, True])
def test_cliff_truncates_only_its_profile_and_retains_evidence(runtime, monkeypatch, failed):
    config, _, _ = runtime
    config.update(
        measurement_sample_policy="full_or_smoke",
        coverage_first=True,
        cliff_aware=True,
        saturation_max_concurrency=8,
        saturation_min_output_tok_s=5,
    )
    second = {**config, "config_id": 2}
    (worker.ROOT / "spec.json").write_text(json.dumps({"workloads": [config, second], "target": {}}))
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(worker.signal, "alarm", lambda *a: None)
    monkeypatch.setattr(worker, "hardware", lambda: {})
    monkeypatch.setattr(worker, "verify_hardware", lambda *a: None)
    visited = []

    def point(c, *a):
        cid, level = c["config_id"], c["concurrency"][0]
        visited.append((cid, level))
        if cid == 1 and level == 2 and failed:
            worker.REPORT["trials"].append({"status": "incomplete_time_budget"})
            return
        rate = 20 if cid == 1 and level == 2 else 100 * level
        for repeat in range(2):
            worker.REPORT["trials"].append(
                {
                    "config_id": cid,
                    "status": "completed",
                    "repeat": repeat,
                    "metrics": {"output_throughput": rate},
                    "requested_requests": max(2, level),
                }
            )

    monkeypatch.setattr(worker, "text_workload", point)
    worker.main()
    assert visited == [(1, 1), (2, 1), (1, 2), (2, 2), (2, 4), (2, 8)]
    truncated = [p for p in worker.REPORT["coverage_plan"] if p["status"] == "truncated"]
    assert [p["concurrency"] for p in truncated] == [4, 8]
    expected = "incomplete_time_budget" if failed else "throughput_collapse"
    assert all(p["stopping_reason"] == expected for p in truncated)
    assert not any(t.get("config_id") == 1 and t.get("concurrency", 0) > 2 for t in worker.REPORT["trials"])
    assert worker.REPORT["text_boundaries"][0]["reason"] == expected


def test_floor_and_nonfinite_boundary_evidence():
    config = Text(name="floor", saturation_min_output_tok_s=5).model_dump()
    assert worker.text_boundary_reason(config, 4, {}) == "throughput_floor"
    for rate in [None, 0, -1, float("nan"), float("inf")]:
        assert worker.text_boundary_reason(config, rate, {}) is None
    with pytest.raises(ValueError, match="Cliff-aware"):
        Text(name="invalid", cliff_aware=True)
