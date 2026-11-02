"""Fail-closed CUDA artifact gate between immutable diagnostic and measurement sweeps."""

import argparse
import ast
import binascii
import fcntl
import hashlib
import io
import json
import math
import struct
import time
import zlib

from PIL import Image, ImageStat

from .db import Database
from .resolve import source_hash
from .sweeps import TERMINAL, job, start


class ValidationFailed(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationFailed(message)


def validate_png(data, width, height, channels=3):
    require(data.startswith(b"\x89PNG\r\n\x1a\n"), "Invalid PNG signature")
    offset, compressed, header, ended = 8, bytearray(), None, False
    while offset < len(data):
        require(offset + 12 <= len(data), "Truncated PNG chunk")
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        require(offset + length + 12 <= len(data), "Truncated PNG payload")
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
        require(binascii.crc32(kind + payload) & 0xFFFFFFFF == crc, "PNG CRC mismatch")
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        if kind == b"IDAT":
            compressed.extend(payload)
        offset += length + 12
        if kind == b"IEND":
            require(length == 0, "Invalid IEND")
            ended = True
            break
    require(ended and offset == len(data), "Missing IEND/trailing PNG data")
    require(header is not None and header[:2] == (width, height), "Wrong PNG dimensions")
    require(channels in (3, 4), "Unsupported image channels")
    require(header[2:] == (8, 2 if channels == 3 else 6, 0, 0, 0), "Expected RGB8/RGBA8 noninterlaced PNG")
    expected = height * (width * channels + 1)
    stream = zlib.decompressobj()
    raw = stream.decompress(bytes(compressed), expected + 1)
    require(len(raw) == expected and stream.eof and not stream.unused_data and not stream.unconsumed_tail, "Incomplete PNG zlib stream")
    # Scanline filters can conceal constant pixels: inspect fully decoded pixels, not compressed bytes.
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        require(image.size == (width, height), "Decoded PNG dimensions mismatch")
        require(max(ImageStat.Stat(image.convert("RGB")).stddev) > 1, "Blank decoded PNG")


def validate_frozen(spec):
    require(source_hash(spec["sources"]) == spec["code_sha256"], "Frozen source manifest hash mismatch")
    require(hashlib.sha256(spec["sources"]["worker.py"].encode()).hexdigest() == spec["worker_sha256"], "Frozen worker hash mismatch")


def requirements_from_source(source):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REQUIREMENTS" for t in node.targets):
            return dict(pin.split("==") for pin in ast.literal_eval(node.value))
    raise ValidationFailed("Missing frozen runtime requirements")


def validate_hardware(report, target):
    hardware = report.get("hardware", {})
    devices = hardware.get("torch_devices", [])
    require(len(devices) == target["gpu_count"] == 1, "Wrong visible GPU count")
    device = devices[0]
    def normalize(text):
        return " ".join(text.split()).casefold()
    require(normalize(device["name"]) == normalize(target["gpu_type"]), "Actual GPU does not match frozen SKU")
    require(("mig" in device["name"].lower()) == bool(target["require_mig"]), "Wrong MIG/full-device variant")
    gib = device["memory_bytes"] / 1024**3
    require(target["expected_memory_gb"] * 0.85 <= gib <= target["expected_memory_gb"] * 1.12, "Wrong GPU memory slice")
    require(hardware.get("cuda") == "12.8", "Wrong executed CUDA runtime")
    return hardware, device


def validate_report(report, workloads, sample, target, requirements):
    """Clean structured CUDA OOM at B2 is valid; incomplete/corrupt evidence fails closed."""
    require(report.get("status") in ("completed", "partial"), "Diagnostic report not finished")
    hardware, device = validate_hardware(report, target)
    expected_components = {
        "StableDiffusionPipeline": {"vae", "text_encoder", "unet"},
        "StableDiffusionXLPipeline": {"vae", "text_encoder", "text_encoder_2", "unet"},
        "ZImagePipeline": {"vae", "text_encoder", "transformer"},
        "Flux2Pipeline": {"vae", "text_encoder", "transformer"},
        "Flux2KleinPipeline": {"vae", "text_encoder", "transformer"},
        "QwenImage21Pipeline": {"vae", "text_encoder", "transformer"},
    }
    validated = []
    require({t["config_id"] for t in report.get("trials", [])} == {c["config_id"] for c in workloads}, "Unexpected/missing workload outcomes")
    for config in workloads:
        cid, pipeline = config["config_id"], config["pipeline"]
        require(pipeline in expected_components, "Unvalidated diagnostic pipeline")
        meta = report.get("image_runs", {}).get(str(cid), {}).get("metadata", {})
        require(meta.get("backend") == "diffusers" and meta.get("attention_backend") == "native_pytorch_sdpa", "Wrong executed backend/attention")
        require(meta.get("compile") is False and meta.get("tier") == "stock", "Unexpected compilation/tier")
        require(meta.get("precision") == "bf16" and meta.get("offload_policy") == "none", "Precision/offload mismatch")
        require(meta.get("gpu_count") == meta.get("tp") == 1, "Placement mismatch")
        require(meta.get("gpu_model") == device["name"] and meta.get("compute_capability") == device["compute_capability"], "Metadata/hardware GPU mismatch")
        require(meta.get("cuda") == hardware["cuda"], "Metadata CUDA mismatch")
        expected_scheduler = "EulerDiscreteScheduler" if pipeline.startswith("StableDiffusion") else "FlowMatchEulerDiscreteScheduler"
        require(meta.get("scheduler") == expected_scheduler, "Wrong executed scheduler")
        for key in ("guidance_mode", "guidance_scale", "true_cfg_scale", "use_kv_cache"):
            require(meta.get(key) == config.get(key), f"Wrong executed {key}")
        libraries = meta.get("libraries", {})
        for package, version in requirements.items():
            require(libraries.get(package, "").split("+")[0] == version, f"Runtime package mismatch: {package}")
        require(libraries.get("torch") == hardware.get("torch"), "Executed Torch mismatch")
        components = meta.get("components", {})
        require(set(components) == expected_components[pipeline], "Missing/unexpected pipeline components")
        vae_dtype = "torch.float32" if config["vae_dtype"] == "fp32" else "torch.bfloat16"
        for name, component in components.items():
            dtype = vae_dtype if name == "vae" else "torch.bfloat16"
            require(component.get("devices") == ["cuda:0"], "Offloaded/missing CUDA component")
            require(component.get("dtypes") == [dtype], f"Wrong component dtype: {name}")

        def validate_call(call, batch):
            steps = config["steps"]
            cfg = pipeline.startswith("StableDiffusion") and config["guidance_scale"] > 1
            forward_batch = batch * (2 if cfg else 1)
            # Qwen uses separate conditional/unconditional forwards, unlike SD's
            # concatenated batch. Qwen 2.1's canonical no-CFG recipe has one.
            forwards = 2 if pipeline.startswith("Qwen") and config["guidance_mode"] == "true_cfg" and config["true_cfg_scale"] > 1 else 1
            require(call["images"] == batch, "Wrong delivered image count")
            require(len(call["denoiser_step_ms"]) == steps and call["denoiser_forward_count"] == steps * forwards, "Wrong callbacks/forwards")
            require(call["denoiser_forwards_per_step"] == [forwards] * steps, "Incomplete/unexpected forward pattern")
            require(call["actual_denoiser_batch_dimensions"] == [forward_batch] * (steps * forwards), "Incomplete/wrong true batch dimensions")
            iterations = call["denoise_iteration_ms"]
            require(len(iterations) == steps, "Incomplete iteration timings")
            for forward_ms, iteration_ms in zip(call["denoiser_step_ms"], iterations):
                require(math.isfinite(forward_ms) and math.isfinite(iteration_ms) and 0 < forward_ms <= iteration_ms + 0.001, "Invalid per-step timing")
            require(call.get("vae_decode_input_dtypes") == [vae_dtype], "Wrong/missing VAE decode input dtype")
            require(call.get("vae_decode_input_devices") == ["cuda:0"], "Wrong/missing VAE decode input device")
            times = [call[k] for k in ("denoiser_seconds", "denoise_total_seconds", "core_seconds", "e2e_seconds")]
            require(all(math.isfinite(t) and t > 0 for t in times), "Nonfinite/negative timing")
            require(all(a <= b + 0.001 for a, b in zip(times, times[1:])), "Inconsistent timing separation")
            forward_seconds = sum(call["denoiser_step_ms"]) / 1000
            require(math.isclose(forward_seconds, call["denoiser_seconds"], rel_tol=1e-4, abs_tol=0.001), "Forward timing aggregate mismatch")
            iteration_seconds = sum(iterations) / 1000
            require(forward_seconds <= iteration_seconds + 0.001 <= call["denoise_total_seconds"] + 0.002, "Iteration timing aggregate mismatch")

        def validate_sample(batch):
            # Qwen 2.1's VAE natively delivers RGBA. Preserve alpha rather than
            # silently dropping a delivered channel to satisfy the older RGB gate.
            channels = 4 if pipeline == "QwenImage21Pipeline" else 3
            validate_png(sample(cid, batch), config["width"], config["height"], channels)

        trials = [t for t in report["trials"] if t["config_id"] == cid]
        require(len(trials) == 2, "Expected diagnostic B1 and B2 outcomes")
        for batch, trial in enumerate(trials, 1):
            require(trial.get("execution_mode") == "batch" and trial.get("batch_size") == batch, "Wrong diagnostic point")
            if trial["status"] == "does_not_fit" and batch == 2:
                require(trial.get("failure_kind") == "cuda_oom" and "cuda out of memory" in trial.get("error", "").lower(), "Unproven CUDA capacity failure")
                require("metrics" not in trial and "requests" not in trial, "OOM contradicts completed measurements")
                partial = trial.get("partial_requests", [])
                require(trial.get("completed_calls_before_failure") == len(partial) < config["requests"], "Inconsistent partial OOM observations")
                require(trial.get("failure_stage") in ("warmup", "measurement"), "Missing OOM stage")
                require(trial["failure_stage"] != "warmup" or not partial, "Warmup OOM contradicts measured calls")
                for call in partial:
                    validate_call(call, batch)
                if partial:
                    validate_sample(batch)
                continue
            require(trial["status"] == "completed", f"Diagnostic failed: {trial.get('error', trial['status'])}")
            require(trial.get("warmup_calls") == config["warmup_requests"] == 3, "Wrong warmup count")
            calls = trial.get("requests", [])
            require(len(calls) == config["requests"] == 8, "Diagnostic sample count mismatch")
            for call in calls:
                validate_call(call, batch)
            metrics = trial["metrics"]
            require(metrics["completed"] == len(calls) and metrics["images"] == len(calls) * batch and metrics["requested_calls"] == 8, "Aggregate count mismatch")
            expected_rate = len(calls) * batch / sum(call["core_seconds"] for call in calls)
            require(math.isfinite(metrics["images_per_second"]) and math.isclose(metrics["images_per_second"], expected_rate, rel_tol=1e-6), "Aggregate throughput mismatch")
            validate_sample(batch)
        validated.append({"config_id": cid, "model": config["model"], "b1": "validated", "b2": trials[-1]["status"]})
    return validated


def validate_binding(run, spec, item, sweep_id):
    require(item["run_id"] == run["id"], "Queue item references another run")
    require(spec.get("sweep_id") == sweep_id and spec.get("sweep_item_id") == item["id"], "Run is associated with another queue item")
    expected = json.loads(item["spec"])
    require(run["target_config_id"] == item["target_config_id"] and spec["target"] == expected["target"], "Executed target differs from queue item")
    require(spec["workloads"] == expected["workloads"], "Executed workloads differ from queue item")
    require(spec["sources"] == expected["sources"], "Executed sources differ from queue item")


def validate_run(db, run_id, expected_sweep_id=None, expected_item_id=None):
    run = db.run(run_id)
    require(run["resource_state"] == "deleted", "Diagnostic Pod cleanup unresolved")
    spec = json.loads(run["spec"])
    if expected_sweep_id is not None or expected_item_id is not None:
        item = db.conn.execute("SELECT * FROM sweep_items WHERE id=? AND sweep_id=?", (expected_item_id, expected_sweep_id)).fetchone()
        require(item is not None, "Missing expected diagnostic item")
        validate_binding(run, spec, item, expected_sweep_id)
    validate_frozen(spec)

    def artifact(name):
        row = db.conn.execute("SELECT content,sha256 FROM artifacts WHERE run_id=? AND path=?", (run_id, name)).fetchone()
        require(row is not None, f"Missing retained artifact: {name}")
        data = zlib.decompress(row["content"])
        require(hashlib.sha256(data).hexdigest() == row["sha256"], "Artifact hash mismatch")
        return data

    for name in ("worker.py", "image_diffusers.py"):
        require(artifact(name) == spec["sources"][name].encode(), f"Executed artifact differs from frozen source: {name}")
    report = json.loads(artifact("report.json"))
    for name in ("worker.py", "image_diffusers.py"):
        require(report.get("source_hashes", {}).get(name) == hashlib.sha256(spec["sources"][name].encode()).hexdigest(), f"Reported executable hash mismatch: {name}")
    requirements = requirements_from_source(spec["sources"]["image_diffusers.py"])
    result = validate_report(report, spec["workloads"], lambda cid, batch: artifact(f"diffusers-{cid}-batch{batch}-sample.png"), spec["target"], requirements)
    db.artifact(run_id, "image-gate-validation.json", json.dumps(result).encode())
    return result, spec


def wait_for_sweep(db, sweep_id):
    while job(db, sweep_id)["status"] not in TERMINAL:
        time.sleep(10)
    return job(db, sweep_id)


def identity(config):
    keys = ("model", "model_revision", "pipeline", "precision", "tier", "width", "height", "steps", "guidance_mode", "guidance_scale", "true_cfg_scale", "vae_dtype", "offload_policy", "attention_backend", "seed", "prompts", "use_kv_cache")
    return json.dumps({k: config.get(k) for k in keys}, sort_keys=True)


def run_queue(db_path, diagnostic_id, followup_id, diagnostic_budget, followup_budget, total_budget, parallel=4):
    db = Database(db_path)
    with (db.path.parent / "jobs" / f"image-gate-{diagnostic_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        full_items = [dict(r) for r in db.conn.execute("SELECT * FROM sweep_items WHERE sweep_id=? ORDER BY ordinal", (followup_id,))]
        require(full_items and job(db, followup_id)["status"] == "prepared", "Followup is not an unstarted immutable sweep")
        require(1 <= parallel <= 8, "Parallelism must be within 1..8")
        require(all(math.isfinite(v) and v > 0 for v in (diagnostic_budget, followup_budget, total_budget)), "Invalid budget")
        require(diagnostic_budget + followup_budget <= total_budget + 1e-9, "Queue exceeds total budget")
        require(diagnostic_budget >= job(db, diagnostic_id)["estimated_ceiling"], "Insufficient diagnostic budget")
        require(followup_budget >= job(db, followup_id)["estimated_ceiling"], "Insufficient followup budget")
        require(json.loads(job(db, diagnostic_id)["options"]).get("validate_image_diagnostics"), "Diagnostic dispatcher must enforce validation before refill")
        full_specs = {json.loads(i["spec"])["target"]["gpu_type"]: json.loads(i["spec"]) for i in full_items}
        diagnostic_specs = [json.loads(r[0]) for r in db.conn.execute("SELECT spec FROM sweep_items WHERE sweep_id=?", (diagnostic_id,))]
        require(set(full_specs) == {s["target"]["gpu_type"] for s in diagnostic_specs}, "Missing GPU diagnostic coverage")
        baseline = next(iter(full_specs.values()))
        require(source_hash() == baseline["code_sha256"], "Controller source changed after preparation")
        for spec in [*full_specs.values(), *diagnostic_specs]:
            validate_frozen(spec)
            require(spec["sources"] == baseline["sources"], "Complete frozen source manifest drift")
        for spec in diagnostic_specs:
            full = full_specs[spec["target"]["gpu_type"]]
            require(spec["target"] == full["target"], "Target/image drift")
            require({identity(w) for w in spec["workloads"]} == {identity(w) for w in full["workloads"]}, "Diagnostic workload drift")
        if job(db, diagnostic_id)["status"] == "prepared":
            print(f"Starting diagnostic sweep {diagnostic_id}", flush=True)
            start(db, diagnostic_id, diagnostic_budget, parallel=parallel)
        require(wait_for_sweep(db, diagnostic_id)["status"] in ("completed", "partial"), "Diagnostic dispatch failed; no followup")
        for item in db.conn.execute("SELECT * FROM sweep_items WHERE sweep_id=? ORDER BY ordinal", (diagnostic_id,)).fetchall():
            spec = json.loads(item["spec"])
            gpu = spec["target"]["gpu_type"]
            if item["status"] == "unavailable":
                require(item["run_id"] and db.run(item["run_id"])["resource_state"] == "deleted", "Unavailable allocation cleanup unresolved")
                with db.conn:
                    for full in full_items:
                        if json.loads(full["spec"])["target"]["gpu_type"] == gpu:
                            db.conn.execute("UPDATE sweep_items SET status='blocked',reason=? WHERE id=?", ("Diagnostic capacity unavailable; no runtime proof", full["id"]))
                print(f"Blocked full measurements for unavailable {gpu}", flush=True)
                continue
            result, actual = validate_run(db, item["run_id"], expected_sweep_id=diagnostic_id, expected_item_id=item["id"])
            require(actual["sources"] == full_specs[gpu]["sources"], "Executed source drift")
            print(f"Validated {gpu}: {result}", flush=True)
        require(source_hash() == baseline["code_sha256"], "Controller source changed while diagnostics ran")
        if not db.conn.execute("SELECT 1 FROM sweep_items WHERE sweep_id=? AND status='planned'", (followup_id,)).fetchone():
            with db.conn:
                db.conn.execute("UPDATE sweeps SET status='partial',finished_at=? WHERE id=?", (time.time(), followup_id))
            print("All followup GPUs blocked; no measurements launched", flush=True)
            return
        print(f"Starting validated stock followup {followup_id}", flush=True)
        start(db, followup_id, followup_budget, parallel=parallel)
        wait_for_sweep(db, followup_id)
        print(f"Stock followup finished: {job(db, followup_id)['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db")
    parser.add_argument("diagnostic_sweep")
    parser.add_argument("followup_sweep")
    parser.add_argument("--diagnostic-budget", type=float, required=True)
    parser.add_argument("--followup-budget", type=float, required=True)
    parser.add_argument("--total-budget", type=float, required=True)
    parser.add_argument("--parallel", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    run_queue(args.db, args.diagnostic_sweep, args.followup_sweep, args.diagnostic_budget, args.followup_budget, args.total_budget, args.parallel)


if __name__ == "__main__":
    main()
