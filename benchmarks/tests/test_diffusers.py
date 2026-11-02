import base64
import hashlib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from sls_bench.config import Image
from sls_bench.db import Database
from sls_bench.image_diffusers import (
    Timers,
    classify_failure,
    measurement_modes,
    percentile,
    point_decision,
    sample_count,
    steady_state_seconds,
)
from sls_bench.image_matrix import definitions, tail_profile
from sls_bench.matrix import expand
from sls_bench.sweeps import assess


@pytest.fixture(scope="module")
def definitions_data():
    return definitions()


def test_saved_matrix_has_all_requested_logical_rows_and_tiers(tmp_path, definitions_data):
    profiles, matrix = definitions_data
    db = Database(tmp_path / "bench.sqlite3")
    for profile in profiles:
        db.save_config(profile)
    db.save_config(matrix)
    row, expanded = expand(db, "image_gen_diffusers")
    assert len(expanded) == 19
    assert len({p["pipeline"] for p in profiles}) == 7
    assert len({p["model"] for p in profiles}) == 7
    assert all(
        p["backend"] == "diffusers" and p["warmup_requests"] >= 3 and p["row_budget_seconds"] < 3600
        for p in profiles
    )
    assert all(
        p["offload_policy"] == "none" and p["execution_modes"] == ["batch"] for p in profiles
    )
    flux = [p for p in profiles if p["model"] == "black-forest-labs/FLUX.1-dev"]
    assert {(p["precision"], p["width"]) for p in flux} == {("bf16", 1024), ("bf16", 2048), ("fp8", 1024)}
    qwen = [p for p in profiles if p["pipeline"] == "QwenImagePipeline"]
    assert {p["width"] for p in qwen} == {1328, 2048}
    edit = next(p for p in profiles if p["pipeline"] == "QwenImageEditPipeline")
    reference = base64.b64decode(edit["reference_image_base64"])
    assert (
        reference.startswith(b"\x89PNG")
        and hashlib.sha256(reference).hexdigest() == edit["reference_image_sha256"]
    )
    turbo = next(p for p in profiles if p["pipeline"] == "ZImagePipeline")
    assert turbo["steps"] == 8 and turbo["requested_denoiser_steps"] == 8 and turbo["batch_cap"] == 32
    assert all(p["readiness"] == "blocked_access" for p in profiles if p["pipeline"] == "Flux2Pipeline")
    assert all(p["tier"] == "optimized" for p in profiles if p["precision"] == "fp8")
    db.conn.close()


@pytest.mark.parametrize("requested,expected", [
    (["latency", "batch"], ["batch"]),
    (["batch", "latency"], ["batch"]),
    (["batch"], ["batch"]),
    (["latency"], ["latency"]),
    (["tail"], ["tail"]),
])
def test_b1_is_not_repeated_for_legacy_dual_track_configs(requested, expected):
    original = requested.copy()
    assert measurement_modes(requested) == expected
    assert requested == original


def test_curve_profiles_have_exactly_40_calls_without_duplicate_b1(definitions_data):
    for profile in definitions_data[0]:
        assert profile["requests"] == 40
        assert profile["warmup_requests"] == 3
        assert profile["minimum_requests"] == 40
        assert measurement_modes(profile["execution_modes"]) == ["batch"]


def test_tail_profile_is_separate_b1_exactly_400_and_preserves_model_settings(definitions_data):
    curve = definitions_data[0][0]
    tail = tail_profile(curve, point_budget_seconds=6000, row_budget_seconds=6300)
    assert tail["execution_modes"] == ["tail"]
    assert tail["batch_cap"] == 1
    assert tail["requests"] == tail["minimum_requests"] == 400
    for key in ("model", "model_revision", "pipeline", "precision", "steps", "prompts", "seed", "vae_dtype"):
        assert tail[key] == curve[key]
    assert curve["requests"] == 40
    assert sample_count(400, 400, 13, 600) == 0
    assert sample_count(400, 400, 13, 6000) == 400
    assert percentile(list(range(400)), 0.99) == 395
    for change in ({"minimum_requests": 40}, {"execution_modes": ["batch", "tail"]}, {"batch_cap": 2}):
        with pytest.raises(ValidationError):
            Image(**{**tail, **change})
    with pytest.raises(ValidationError):
        Image(**{**curve, "row_budget_seconds": 6000})


def test_only_typed_cuda_oom_is_a_gpu_capacity_limit():
    class OOM(RuntimeError):
        pass

    torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM))
    assert classify_failure(OOM("CUDA out of memory"), torch) == ("does_not_fit", "cuda_oom")
    assert classify_failure(OOM("CPU out of memory"), torch)[1] == "runtime_failure"
    assert classify_failure(RuntimeError("CUDA out of memory"), torch)[1] == "runtime_failure"


def test_compile_warmup_is_charged_but_not_used_as_steady_state_cost():
    warmups = [120, 3, 3]
    remaining = 726 - sum(warmups)
    assert remaining == 600  # Compilation was NOT refunded.
    assert sample_count(40, 40, max(warmups), remaining) == 0
    pilot = steady_state_seconds(warmups)
    assert pilot == 3
    assert sample_count(40, 40, pilot, remaining) == 40
    assert steady_state_seconds([120, 3, 4]) == 4
    assert steady_state_seconds([3, 120, 3]) == 120
    assert steady_state_seconds([3, 3, 120]) == 120
    assert sample_count(40, 40, pilot, 100) == 0
    assert sample_count(400, 400, pilot, remaining) == 0
    assert steady_state_seconds([4]) == 4
    for invalid in ([], [120, 3, float("nan")], [120, -3, 3], [120, 0, 3], [float("inf"), 3, 3]):
        with pytest.raises(ValueError):
            steady_state_seconds(invalid)


def test_stopping_rules_and_real_sample_budget():
    base = {"images_per_second": 1, "p95_e2e_seconds": 10}
    assert (
        point_decision(base, {"images_per_second": 1.09, "p95_e2e_seconds": 12}, 10, 2, 16)[0]
        == "throughput_plateau"
    )
    assert (
        point_decision(base, {"images_per_second": 2, "p95_e2e_seconds": 41}, 10, 2, 16)[0] == "latency_limit"
    )
    assert (
        point_decision(base, {"images_per_second": 2, "p95_e2e_seconds": 20}, 10, 16, 16)[0]
        == "still_scaling_at_cap"
    )
    assert point_decision(None, base, 10, 1, 1)[0] == "batch_cap"
    assert sample_count(100, 8, 40, 1000) == 25
    assert sample_count(100, 8, 400, 1000) == 0


def test_no_implicit_offload_or_fp8_fallback(definitions_data):
    profile = definitions_data[0][0]
    for change in [
        {"warmup_requests": 2},
        {"gpu_only": False},
        {"precision": "fp8", "tier": "stock"},
        {"batch_cap": 3},
    ]:
        with pytest.raises(ValidationError):
            Image(**{**profile, **change})
    fp8 = next(p for p in definitions_data[0] if p["precision"] == "fp8")
    import json

    status, reason = assess(
        {"manufacturer": "Nvidia", "memoryInGb": 24, "gpuGeneration": "AMPERE"},
        {"spec": json.dumps(fp8)},
        {"gpu_count": 1},
    )
    assert status == "blocked" and "no weight-only fallback" in reason


def test_cuda_events_measure_forward_only_steps_and_true_batch_dimensions():
    class Event:
        counter = 0

        def __init__(self, **kwargs):
            pass

        def record(self):
            Event.counter += 1
            self.time = Event.counter

        def elapsed_time(self, other):
            return other.time - self.time

    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: None))

    def forward(*args, **kwargs):
        return "latent"

    pipe = SimpleNamespace(
        encode_prompt=lambda *a: "embedding",
        transformer=SimpleNamespace(forward=forward),
        vae=SimpleNamespace(decode=lambda *a: "decoded"),
        image_processor=SimpleNamespace(postprocess=lambda *a: "image"),
    )
    timers = Timers(pipe, torch)
    pipe.encode_prompt("text")
    pipe.transformer.forward(hidden_states=SimpleNamespace(shape=(2, 128, 64)))
    pipe.transformer.forward(hidden_states=SimpleNamespace(shape=(2, 128, 64)))
    timers.callback(pipe, 0, 0, {})
    pipe.vae.decode("latent")
    pipe.image_processor.postprocess("decoded")
    result = timers.results()
    assert result["denoiser_step_ms"] == [2]
    assert result["denoise_iteration_ms"][0] > 2
    assert result["actual_denoiser_batch_dimensions"] == [2, 2]
    assert result["denoiser_forwards_per_step"] == [2]


def test_fp32_vae_receives_cast_latents_without_changing_denoiser():
    from unittest.mock import Mock

    events = SimpleNamespace(record=lambda: None)
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw: events))
    decoded = []
    vae = SimpleNamespace(dtype="fp32", decode=lambda z: decoded.append(z))
    pipe = SimpleNamespace(
        encode_prompt=lambda *a: None,
        transformer=SimpleNamespace(forward=lambda *a: None),
        vae=vae,
        image_processor=SimpleNamespace(postprocess=lambda *a: None),
    )
    Timers(pipe, torch)
    latent = Mock()
    latent.to.return_value = "fp32-latent"
    pipe.vae.decode(latent)
    pipe.vae.decode(z=latent)
    assert decoded == ["fp32-latent", "fp32-latent"]
    assert latent.to.call_count == 2
    assert all(call.kwargs == {"dtype": "fp32"} for call in latent.to.call_args_list)
