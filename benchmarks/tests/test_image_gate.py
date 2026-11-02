import binascii
import copy
import hashlib
import json
import struct
import zlib
from unittest.mock import Mock

import pytest

from sls_bench import image_gate, sweeps
from sls_bench.config import validate
from sls_bench.db import Database
from sls_bench.resolve import source_hash, source_snapshot

TARGET = {"gpu_type": "test GPU", "gpu_count": 1, "expected_memory_gb": 24, "require_mig": False}
REQUIREMENTS = {"torch": "2.9.1", "diffusers": "0.42.1"}


def png(blank=False):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    raw = b"".join(b"\0" + (bytes([10, 80, 200]) * 4 if blank else bytes(range(i * 12, i * 12 + 12))) for i in range(4))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def fixture():
    config = {"config_id": 1, "requests": 8, "warmup_requests": 3, "steps": 30, "pipeline": "StableDiffusionPipeline", "guidance_scale": 7.5, "guidance_mode": "true_cfg", "true_cfg_scale": 4, "vae_dtype": "fp32", "width": 4, "height": 4, "model": "test"}
    trials = []
    for batch in (1, 2):
        call = {"images": batch, "denoiser_step_ms": [1] * 30, "denoise_iteration_ms": [1.01] * 30, "denoiser_forward_count": 30, "denoiser_forwards_per_step": [1] * 30, "actual_denoiser_batch_dimensions": [batch * 2] * 30, "denoiser_seconds": 0.03, "denoise_total_seconds": 0.031, "core_seconds": 1.2, "e2e_seconds": 1.3, "vae_decode_input_dtypes": ["torch.float32"], "vae_decode_input_devices": ["cuda:0"]}
        trials.append({"config_id": 1, "status": "completed", "execution_mode": "batch", "batch_size": batch, "warmup_calls": 3, "requests": [copy.deepcopy(call) for _ in range(8)], "metrics": {"completed": 8, "images": 8 * batch, "requested_calls": 8, "images_per_second": batch / 1.2}})
    components = {"vae": {"devices": ["cuda:0"], "dtypes": ["torch.float32"]}, "unet": {"devices": ["cuda:0"], "dtypes": ["torch.bfloat16"]}, "text_encoder": {"devices": ["cuda:0"], "dtypes": ["torch.bfloat16"]}}
    meta = {"offload_policy": "none", "gpu_count": 1, "tp": 1, "precision": "bf16", "tier": "stock", "components": components, "backend": "diffusers", "attention_backend": "native_pytorch_sdpa", "compile": False, "gpu_model": "test GPU", "compute_capability": [8, 9], "cuda": "12.8", "scheduler": "EulerDiscreteScheduler", "guidance_mode": "true_cfg", "guidance_scale": 7.5, "true_cfg_scale": 4, "libraries": {"torch": "2.9.1+cu128", "diffusers": "0.42.1"}}
    report = {"status": "completed", "trials": trials, "image_runs": {"1": {"metadata": meta}}, "hardware": {"torch": "2.9.1+cu128", "cuda": "12.8", "torch_devices": [{"name": "test GPU", "memory_bytes": 24 * 1024**3, "compute_capability": [8, 9]}]}}
    return report, config


def check(report, config, sample=None):
    return image_gate.validate_report(report, [config], sample or (lambda *a: png()), TARGET, REQUIREMENTS)


def test_retained_cuda_gate_accepts_correct_cfg_and_structured_capacity_limit():
    report, config = fixture()
    assert check(report, config)[0]["b2"] == "completed"
    report["status"] = "partial"
    report["trials"][1] = {"config_id": 1, "execution_mode": "batch", "batch_size": 2, "status": "does_not_fit", "failure_kind": "cuda_oom", "failure_stage": "warmup", "completed_calls_before_failure": 0, "partial_requests": [], "error": "CUDA out of memory"}
    assert check(report, config)[0]["b2"] == "does_not_fit"
    report["trials"][1]["metrics"] = {"completed": 8}
    with pytest.raises(image_gate.ValidationFailed, match="contradicts"):
        check(report, config)
    report["trials"][1].pop("metrics")
    report["trials"][1]["failure_kind"] = "cpu_oom"
    with pytest.raises(image_gate.ValidationFailed, match="Unproven CUDA"):
        check(report, config)


@pytest.mark.parametrize("fault", ["batch", "batch-cardinality", "forward-cardinality", "steps", "step-nan", "samples", "warmups", "timing", "dtype", "vae-input", "offload", "runtime", "png", "components", "gpu", "memory", "backend", "attention", "compile", "scheduler", "guidance", "library", "aggregate", "forward-aggregate", "iteration-aggregate"])
def test_gate_rejects_invalid_or_incomplete_evidence(fault):
    report, config = fixture()
    call = report["trials"][0]["requests"][0]
    meta = report["image_runs"]["1"]["metadata"]
    if fault == "batch":
        call["actual_denoiser_batch_dimensions"] = [1] * 30
    elif fault == "batch-cardinality":
        call["actual_denoiser_batch_dimensions"] = [2]
    elif fault == "forward-cardinality":
        call["denoiser_forwards_per_step"] = [1]
    elif fault == "steps":
        call["denoiser_forward_count"] = 29
    elif fault == "step-nan":
        call["denoiser_step_ms"][3] = float("nan")
    elif fault == "samples":
        report["trials"][0]["requests"].pop()
    elif fault == "warmups":
        report["trials"][0]["warmup_calls"] = 2
    elif fault == "timing":
        call["e2e_seconds"] = float("nan")
    elif fault == "dtype":
        meta["components"]["vae"]["dtypes"] = ["torch.bfloat16"]
    elif fault == "vae-input":
        call["vae_decode_input_dtypes"] = ["torch.bfloat16"]
    elif fault == "offload":
        meta["components"]["unet"]["devices"] = ["cpu"]
    elif fault == "runtime":
        report["trials"][1].update(status="unsupported_or_runtime_failure", error="kernel unavailable")
    elif fault == "components":
        meta["components"] = {"vae": meta["components"]["vae"]}
    elif fault == "gpu":
        report["hardware"]["torch_devices"][0]["name"] = "different GPU"
    elif fault == "memory":
        report["hardware"]["torch_devices"][0]["memory_bytes"] = 96 * 1024**3
    elif fault == "backend":
        meta["backend"] = "comfyui"
    elif fault == "attention":
        meta["attention_backend"] = "xformers"
    elif fault == "compile":
        meta["compile"] = True
    elif fault == "scheduler":
        meta["scheduler"] = "OtherScheduler"
    elif fault == "guidance":
        meta["guidance_scale"] = 0
    elif fault == "library":
        meta["libraries"]["torch"] = "2.10.0"
    elif fault == "aggregate":
        report["trials"][0]["metrics"]["images_per_second"] *= 2
    elif fault == "forward-aggregate":
        call["denoiser_seconds"] = 0.5
        call["denoise_total_seconds"] = 0.6
    elif fault == "iteration-aggregate":
        call["denoise_iteration_ms"] = [100] * 30
    with pytest.raises(image_gate.ValidationFailed):
        check(report, config, lambda *a: b"broken" if fault == "png" else png())


def test_partial_cuda_oom_cannot_hide_a_blank_measured_sample():
    report, config = fixture()
    partial = [report["trials"][1]["requests"][0]]
    report["status"] = "partial"
    report["trials"][1] = {"config_id": 1, "execution_mode": "batch", "batch_size": 2, "status": "does_not_fit", "failure_kind": "cuda_oom", "failure_stage": "measurement", "completed_calls_before_failure": 1, "partial_requests": partial, "error": "CUDA out of memory"}
    with pytest.raises(image_gate.ValidationFailed, match="Blank decoded"):
        check(report, config, lambda cid, batch: png(blank=batch == 2))


def test_png_requires_complete_stream_iend_and_decoded_nonblank_pixels():
    image_gate.validate_png(png(), 4, 4)
    for data in (png()[:-12], png() + b"junk", png(blank=True)):
        with pytest.raises(image_gate.ValidationFailed):
            image_gate.validate_png(data, 4, 4)


def make_jobs(tmp_path, monkeypatch):
    db = Database(tmp_path / "bench.sqlite3")
    (tmp_path / "jobs").mkdir()
    api = Mock()
    api.gpus.return_value = [{"id": n, "manufacturer": "Nvidia", "memoryInGb": 24, "securePrice": 1} for n in ("one", "two")]
    sources = source_snapshot()
    monkeypatch.setattr(sweeps, "resolve", lambda target, workloads: {
        "code_sha256": source_hash(sources), "worker_sha256": hashlib.sha256(sources["worker.py"].encode()).hexdigest(), "sources": sources,
        "target": json.loads(target["spec"]), "workloads": [{**validate(json.loads(w["spec"])), "config_id": w["id"]} for w in workloads],
    })
    diag = sweeps.prepare(db, api, all_gpus=True, suite="image-smoke", validate_image_diagnostics=True)
    full = sweeps.prepare(db, api, all_gpus=True, suite="image-smoke")
    return db, api, diag, full


def test_validation_failure_stops_before_allocating_next_gpu(tmp_path, monkeypatch):
    db, api, diag, _ = make_jobs(tmp_path, monkeypatch)
    monkeypatch.setattr(sweeps, "Runpod", Mock(return_value=api))
    with db.conn:
        db.conn.execute("UPDATE sweeps SET approved_budget=10 WHERE id=?", (diag,))

    def execute(db, api, rid, key, progress):
        db.update("runs", rid, status="partial")
        db.update("resources", rid, state="deleted")
        return {"status": "partial"}

    runner = Mock(side_effect=execute)
    monkeypatch.setattr(sweeps, "execute", runner)
    monkeypatch.setattr(image_gate, "validate_run", Mock(side_effect=image_gate.ValidationFailed("wrong batches")))
    with pytest.raises(image_gate.ValidationFailed, match="wrong batches"):
        sweeps.run(db.path, diag)
    assert runner.call_count == 1
    assert sweeps.job(db, diag)["status"] == "interrupted"


@pytest.mark.parametrize("fault", [None, "validation", "cleanup", "unavailable", "source", "image", "budget", "controller-drift", "wrong-run", "wrong-target", "wrong-workloads"])
def test_queue_end_to_end_transitions_and_fail_closed(tmp_path, monkeypatch, fault):
    db, _, diag, full = make_jobs(tmp_path, monkeypatch)
    launches, drift = [], [False]
    monkeypatch.setattr(image_gate, "Database", lambda path: db)
    original_hash = image_gate.source_hash
    monkeypatch.setattr(image_gate, "source_hash", lambda sources=None: "changed" if sources is None and drift[0] else original_hash(sources))
    if fault in ("source", "image"):
        row = db.conn.execute("SELECT id,spec FROM sweep_items WHERE sweep_id=? LIMIT 1", (diag,)).fetchone()
        spec = json.loads(row["spec"])
        if fault == "source":
            spec["sources"]["worker.py"] += "\n# changed"
        else:
            spec["target"]["image"] = "different"
        with db.conn:
            db.conn.execute("UPDATE sweep_items SET spec=? WHERE id=?", (json.dumps(spec), row["id"]))

    def launch(db, sid, budget, **kwargs):
        launches.append(sid)
        if sid == diag:
            for item in db.conn.execute("SELECT * FROM sweep_items WHERE sweep_id=?", (diag,)).fetchall():
                spec = json.loads(item["spec"])
                target = dict(db.conn.execute("SELECT * FROM configs WHERE id=?", (item["target_config_id"],)).fetchone())
                workloads = [dict(db.conn.execute("SELECT * FROM configs WHERE id=?", (w["config_id"],)).fetchone()) for w in spec["workloads"]]
                spec.update(sweep_id=sid, sweep_item_id=item["id"])
                rid = db.new_run(target, workloads, spec, 60)
                db.update("resources", rid, state="running" if fault == "cleanup" else "deleted")
                state = "unavailable" if fault == "unavailable" and spec["target"]["gpu_type"] == "two" else "completed"
                with db.conn:
                    db.conn.execute("UPDATE sweep_items SET run_id=?,status=? WHERE id=?", (rid, state, item["id"]))
            drift[0] = fault == "controller-drift"
        with db.conn:
            db.conn.execute("UPDATE sweeps SET status='completed' WHERE id=?", (sid,))

    def gate(db, rid, expected_sweep_id=None, expected_item_id=None):
        run = db.run(rid)
        spec = json.loads(run["spec"])
        item = db.conn.execute("SELECT * FROM sweep_items WHERE id=?", (expected_item_id,)).fetchone()
        if fault == "wrong-run":
            spec["sweep_item_id"] = -1
        if fault == "wrong-target":
            spec["target"]["gpu_type"] = "another GPU"
        if fault == "wrong-workloads":
            spec["workloads"] = []
        image_gate.validate_binding(run, spec, item, expected_sweep_id)
        image_gate.require(run["resource_state"] == "deleted", "cleanup unresolved")
        image_gate.require(fault != "validation", "bad evidence")
        return [], spec

    monkeypatch.setattr(image_gate, "start", launch)
    monkeypatch.setattr(image_gate, "validate_run", gate)
    if fault in (None, "unavailable"):
        image_gate.run_queue(db.path, diag, full, 10, 10, 20)
        assert launches == [diag, full]
        if fault == "unavailable":
            rows = db.conn.execute("SELECT status FROM sweep_items WHERE sweep_id=? ORDER BY ordinal", (full,)).fetchall()
            assert [r[0] for r in rows] == ["planned", "blocked"]
    else:
        with pytest.raises(image_gate.ValidationFailed):
            image_gate.run_queue(db.path, diag, full, 10, 10, 15 if fault == "budget" else 20)
        assert full not in launches
        if fault in ("source", "image", "budget"):
            assert not launches


def test_retained_executable_must_match_frozen_source_even_with_valid_artifact_hash(tmp_path, monkeypatch):
    db, _, diag, _ = make_jobs(tmp_path, monkeypatch)
    item = db.conn.execute("SELECT * FROM sweep_items WHERE sweep_id=? LIMIT 1", (diag,)).fetchone()
    spec = json.loads(item["spec"])
    target = dict(db.conn.execute("SELECT * FROM configs WHERE id=?", (item["target_config_id"],)).fetchone())
    workloads = [dict(db.conn.execute("SELECT * FROM configs WHERE id=?", (w["config_id"],)).fetchone()) for w in spec["workloads"]]
    rid = db.new_run(target, workloads, spec, 60)
    db.update("resources", rid, state="deleted")
    db.artifact(rid, "worker.py", b"different executable with a freshly valid artifact hash")
    with pytest.raises(image_gate.ValidationFailed, match="differs from frozen"):
        image_gate.validate_run(db, rid)
