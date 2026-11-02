import copy

import pytest
from pydantic import ValidationError

from sls_bench import worker
from sls_bench.config import Text


@pytest.fixture
def setup(monkeypatch):
    report = {"trials": [], "phases": []}
    monkeypatch.setattr(worker, "REPORT", report)
    monkeypatch.setattr(worker, "checkpoint", lambda: None)
    config = Text(
        name="adaptive",
        requests=1024,
        repeats=2,
        concurrency=[16],
        saturation_max_concurrency=512,
        engine_capacity_policy="match_concurrency",
    ).model_dump()
    config["config_id"] = 1
    return config, report


def executor(monkeypatch, rates, calls):
    def fixed(config, target, models):
        c = config["concurrency"][0]
        calls.append((c, config["max_num_seqs"], config["requests"], config["repeats"]))
        for repeat in range(config["repeats"]):
            worker.REPORT["trials"].append(
                {"status": "completed", "repeat": repeat, "metrics": {"output_throughput": rates[c]}}
            )

    monkeypatch.setattr(worker, "fixed_text_workload", fixed)


def test_two_repeat_medians_and_two_plateau_doublings_stop(setup, monkeypatch):
    config, report = setup
    calls = []
    executor(monkeypatch, {16: 100, 32: 103, 64: 104}, calls)
    original = copy.deepcopy(config)
    worker.text_workload(config, {}, {})
    assert config == original
    assert calls == [(c, c, 1024, 2) for c in (16, 32, 64)]
    result = report["saturation"][0]
    assert result["reason"] == "throughput_plateau"
    assert result["best_concurrency"] == 64
    assert len(report["trials"]) == 6


@pytest.mark.parametrize("rates,floor,expected", [({16: 24}, 25, [16]), ({16: 100, 32: 40}, 25, [16, 32])])
def test_low_absolute_or_collapsed_throughput_stops_search(setup, monkeypatch, rates, floor, expected):
    config, report = setup
    config["saturation_min_output_tok_s"] = floor
    calls = []
    executor(monkeypatch, rates, calls)
    worker.text_workload(config, {}, {})
    assert [c[0] for c in calls] == expected
    assert report["saturation"][0]["reason"] == "low_throughput_cutoff"


@pytest.mark.parametrize("schedule", [{1: 512, 2: 512}, {"1": 512, "2": 512}])
def test_request_counts_follow_concurrency_schedule_without_changing_repeats(setup, monkeypatch, schedule):
    config, report = setup
    config.update(concurrency=[1], saturation_max_concurrency=4, requests_by_concurrency=schedule)
    calls = []
    executor(monkeypatch, {1: 100, 2: 180, 4: 300}, calls)
    worker.text_workload(config, {}, {})
    assert calls == [(1, 1, 512, 2), (2, 2, 512, 2), (4, 4, 1024, 2)]
    assert [h["requests_per_repeat"] for h in report["saturation"][0]["history"]] == [512, 512, 1024]
    assert config["requests"] == 1024


@pytest.mark.parametrize("schedule", [{0: 512}, {3: 512}, {1024: 1024}, {2: 1}, {1: 2048}])
def test_invalid_request_schedules_are_rejected(schedule):
    with pytest.raises(ValidationError):
        Text(
            name="invalid",
            requests=1024,
            concurrency=[1],
            saturation_max_concurrency=512,
            requests_by_concurrency=schedule,
        )


def test_material_gain_resets_plateau_and_cap_is_not_saturation(setup, monkeypatch):
    config, report = setup
    config["saturation_max_concurrency"] = 128
    calls = []
    executor(monkeypatch, {16: 100, 32: 101, 64: 200, 128: 201}, calls)
    worker.text_workload(config, {}, {})
    assert [c[0] for c in calls] == [16, 32, 64, 128]
    assert report["saturation"][0]["reason"] == "concurrency_cap"


def test_trial_failure_stops_without_claiming_saturation(setup, monkeypatch):
    config, report = setup

    def failed(config, target, models):
        report["trials"].append({"status": "failed"})

    monkeypatch.setattr(worker, "fixed_text_workload", failed)
    worker.text_workload(config, {}, {})
    assert report["saturation"][0]["reason"] == "trial_failure"


def test_setup_failure_retains_failing_concurrency(setup, monkeypatch):
    config, report = setup

    def fail(*args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(worker, "fixed_text_workload", fail)
    worker.text_workload(config, {}, {})
    assert report["saturation"][0]["reason"] == "runtime_failure"
    assert report["trials"][0]["status"] == "setup_failed"
    assert report["trials"][0]["concurrency"] == 16


def test_failure_after_one_completed_repeat_does_not_look_successful(setup, monkeypatch):
    config, report = setup

    def fail(*args):
        report["trials"].append({"status": "completed", "metrics": {"output_throughput": 100}})
        raise RuntimeError("server died")

    monkeypatch.setattr(worker, "fixed_text_workload", fail)
    worker.text_workload(config, {}, {})
    assert [t["status"] for t in report["trials"]] == ["completed", "setup_failed"]
    assert report["trials"][-1]["repeat"] == 1
    assert report["saturation"][0]["reason"] == "runtime_failure"


def test_legacy_fixed_sweep_is_unchanged(setup, monkeypatch):
    config, report = setup
    config["saturation_max_concurrency"] = None
    calls = []
    executor(monkeypatch, {16: 100}, calls)
    worker.text_workload(config, {}, {})
    assert len(calls) == 1
    assert "saturation" not in report


@pytest.mark.parametrize(
    "kwargs",
    [
        {"concurrency": [1, 16]},
        {"concurrency": [3]},
        {"saturation_max_concurrency": 24},
        {"saturation_max_concurrency": 8},
        {"requests": 32},
    ],
)
def test_invalid_search_bounds(kwargs):
    values = dict(name="invalid", requests=1024, concurrency=[16], saturation_max_concurrency=512)
    values.update(kwargs)
    with pytest.raises(ValidationError):
        Text(**values)
