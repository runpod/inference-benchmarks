"""Isolated, no-offload Diffusers worker. No serving/concurrent-request claims."""

import argparse
import base64
import functools
import hashlib
import importlib.metadata
import io
import json
import math
import statistics
import time
from pathlib import Path

# Linux/Python3.11+ compatible pins. Torch itself is supplied by the pinned image.
REQUIREMENTS = [
    "torch==2.9.1",
    "diffusers==0.42.1",
    "transformers==5.22.1",
    "accelerate==1.15.0",
    "torchao==0.15.0",
    "huggingface-hub==2.4.0",
    "safetensors==0.8.0",
    "sentencepiece==0.2.2",
    "protobuf==7.36.2",
    "Pillow==13.0.0",
    "tokenizers==0.23.3",
    "numpy==2.3.5",
]


def percentile(values, q):
    return sorted(values)[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def measurement_modes(requested):
    # B1 in a batch sweep already records latency, throughput, and all stage spans.
    # Older saved configs request both tracks; do not execute the identical B1 twice.
    return [mode for mode in requested if mode != "latency" or "batch" not in requested]


def point_decision(previous, current, baseline_p95, batch, cap):
    gain = current["images_per_second"] / previous["images_per_second"] - 1 if previous else None
    if current["p95_e2e_seconds"] > 4 * baseline_p95:
        return "latency_limit", gain
    if gain is not None and gain < 0.10:
        return "throughput_plateau", gain
    if batch == cap:
        return ("still_scaling_at_cap" if gain is not None and gain > 0.10 else "batch_cap"), gain
    return "continue", gain


def steady_state_seconds(warmups):
    """Use the trailing warm calls, not an earlier one-time compile/load cost.

    The slower of the final two remains conservative for late recompilation or
    unsettled timings. All warmups have already consumed the unchanged deadline.
    This is a budget estimate, not a claim that thermal/timing stability is proven.
    """
    if not warmups or any(not math.isfinite(t) or t <= 0 for t in warmups):
        raise ValueError("Warmup durations must be finite and positive")
    return max(warmups[-2:])


def sample_count(requested, minimum, pilot_seconds, remaining_seconds):
    # Keep all completed calls. Never pretend a small sample is 100 observations.
    affordable = int(max(0, remaining_seconds) / max(pilot_seconds, 0.001))
    return min(requested, affordable) if affordable >= minimum else 0


def guidance_arguments(config):
    """Qwen 2.1 has true_cfg_scale, not guidance_scale; its native recipe uses no CFG."""
    if config["pipeline"].startswith("Qwen") and config["guidance_mode"] == "true_cfg":
        return {"true_cfg_scale": config["true_cfg_scale"], "negative_prompt": " "}
    if config["pipeline"] == "QwenImage21Pipeline":
        return {"true_cfg_scale": 1.0}
    # Preserve the existing argument contract for every other pipeline.
    return {"guidance_scale": config["guidance_scale"]}


def classify_failure(error, torch):
    # CPU allocator exhaustion is not evidence of a GPU VRAM capacity limit.
    if isinstance(error, torch.cuda.OutOfMemoryError) and "cuda out of memory" in str(error).lower():
        return "does_not_fit", "cuda_oom"
    return "unsupported_or_runtime_failure", "runtime_failure"


class Timers:
    """Events do not synchronize each denoiser step; synchronize once at postprocess.

    Core wall time ends before postprocessing. PNG/host transfer is timed separately.
    Denoise span includes CFG/scheduler work between first forward and last callback.
    """

    def __init__(self, pipe, torch):
        self.pipe, self.torch = pipe, torch
        self.originals = []
        self.reset()
        self.wrap(pipe, "encode_prompt", "text_encode")
        self.denoiser = getattr(pipe, "transformer", None) or pipe.unet
        self.wrap(self.denoiser, "forward", "denoiser")
        self.wrap(pipe.vae, "decode", "vae_decode")
        if hasattr(pipe.vae, "encode"):
            self.wrap(pipe.vae, "encode", "reference_encode")
        original = pipe.image_processor.postprocess
        self.originals.append((pipe.image_processor, "postprocess", original))

        @functools.wraps(original)
        def postprocess(*args, **kwargs):
            torch.cuda.synchronize()
            self.core_end = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            self.post_seconds += time.perf_counter() - self.core_end
            return result

        pipe.image_processor.postprocess = postprocess

    def reset(self):
        self.events, self.steps, self.batches = {}, [], []
        self.vae_input_dtypes, self.vae_input_devices = [], []
        self.step_forward_offset = 0
        self.core_end, self.post_seconds = None, 0.0

    def event(self):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def wrap(self, obj, name, stage):
        original = getattr(obj, name)
        self.originals.append((obj, name, original))

        @functools.wraps(original)
        def timed(*args, **kwargs):
            start = self.event()
            if stage == "vae_decode" and hasattr(obj, "dtype"):
                # Legacy pipelines do not consistently cast BF16 denoiser latents
                # for an explicitly FP32 VAE. Include the cast in VAE stage time.
                if args and hasattr(args[0], "to"):
                    args = (args[0].to(dtype=obj.dtype), *args[1:])
                elif "z" in kwargs:
                    kwargs["z"] = kwargs["z"].to(dtype=obj.dtype)
                value = args[0] if args else kwargs.get("z")
                self.vae_input_dtypes.append(str(getattr(value, "dtype", "unknown")))
                self.vae_input_devices.append(str(getattr(value, "device", "unknown")))
            if stage == "denoiser":
                value = kwargs.get("hidden_states", kwargs.get("sample", kwargs.get("x")))
                if value is None and args:
                    value = args[0]
                batch = (
                    value.shape[0]
                    if hasattr(value, "shape")
                    else len(value)
                    if isinstance(value, list)
                    else None
                )
                self.batches.append(batch)
            result = original(*args, **kwargs)
            self.events.setdefault(stage, []).append((start, self.event()))
            return result

        setattr(obj, name, timed)

    def callback(self, pipe, step, timestep, callback_kwargs):
        forwards = self.events.get("denoiser", [])
        if len(forwards) <= self.step_forward_offset:
            raise RuntimeError("Denoising callback without an observed model forward")
        self.steps.append(
            (forwards[self.step_forward_offset][0], self.event(), len(forwards) - self.step_forward_offset)
        )
        self.step_forward_offset = len(forwards)
        return callback_kwargs

    def results(self):
        if not self.steps or self.core_end is None:
            raise RuntimeError(
                "Pipeline did not expose denoising callbacks/postprocess; timing is unvalidated"
            )
        totals = {
            k + "_seconds": sum(a.elapsed_time(b) for a, b in pairs) / 1000
            for k, pairs in self.events.items()
        }
        forward_ms = [a.elapsed_time(b) for a, b in self.events["denoiser"]]
        offset, denoiser_steps = 0, []
        for _, _, count in self.steps:
            denoiser_steps.append(sum(forward_ms[offset : offset + count]))
            offset += count
        totals.update(
            denoise_total_seconds=self.steps[0][0].elapsed_time(self.steps[-1][1]) / 1000,
            denoiser_step_ms=denoiser_steps,
            denoise_iteration_ms=[a.elapsed_time(b) for a, b, _ in self.steps],
            denoiser_forwards_per_step=[n for _, _, n in self.steps],
            denoiser_forward_count=len(self.events["denoiser"]),
            actual_denoiser_batch_dimensions=self.batches,
            postprocessing_seconds=self.post_seconds,
            vae_decode_input_dtypes=self.vae_input_dtypes,
            vae_decode_input_devices=self.vae_input_devices,
        )
        return totals


def run(config, directory):
    import diffusers
    import torch
    from PIL import Image

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "trials": [], "metadata": {}, "searches": []}
    started = time.monotonic()
    deadline = started + config["row_budget_seconds"]

    def save():
        temporary = root / "progress.tmp"
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(root / "progress.json")

    def available():
        return deadline - time.monotonic()

    def failure(error):
        return classify_failure(error, torch)[0]

    def row_error(error):
        result["trials"].append(
            {
                "config_id": config["config_id"],
                "concurrency": 1,
                "repeat": -1,
                "status": failure(error),
                "failure_kind": classify_failure(error, torch)[1],
                "error": str(error),
                "backend": "diffusers",
            }
        )
        result.update(status="partial", error=str(error))
        save()

    try:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Validated direct adapter is single-GPU only; no implied TP2 or sharding")
        if config["readiness"] != "ready":
            raise RuntimeError(f"Row not executable: {config['readiness']}")
        capability = torch.cuda.get_device_capability()
        if config["precision"] == "fp8" and capability < (8, 9):
            raise RuntimeError("Native FP8 kernels require compute capability >=8.9; no weight-only fallback")
        kwargs = {
            "revision": config["model_revision"],
            "torch_dtype": torch.bfloat16,
            "use_safetensors": True,
        }
        if config["pipeline"] == "StableDiffusionXLPipeline":
            kwargs["add_watermarker"] = False
        if config["pipeline"] == "StableDiffusionPipeline":
            kwargs.update(safety_checker=None, requires_safety_checker=False)
        if config["precision"] == "fp8":
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

            kwargs["quantization_config"] = diffusers.PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": diffusers.TorchAoConfig(Float8DynamicActivationFloat8WeightConfig())
                }
            )
        pipe = getattr(diffusers, config["pipeline"]).from_pretrained(config["model"], **kwargs)
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if config["pipeline"] in ("StableDiffusionPipeline", "StableDiffusionXLPipeline"):
            pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        vae_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            config["vae_dtype"]
        ]
        pipe.vae.to(dtype=vae_dtype)
        denoiser = getattr(pipe, "transformer", None) or pipe.unet
        if hasattr(denoiser, "set_attention_backend"):
            denoiser.set_attention_backend("native")
        elif hasattr(denoiser, "set_attn_processor"):
            from diffusers.models.attention_processor import AttnProcessor2_0

            denoiser.set_attn_processor(AttnProcessor2_0())
        else:
            raise RuntimeError("Cannot enforce stock native PyTorch SDPA for this model")
        component_metadata = {}
        for name, component in pipe.components.items():
            if isinstance(component, torch.nn.Module):
                parameters = list(component.parameters())
                devices = sorted({str(p.device) for p in parameters})
                if any(not d.startswith("cuda") for d in devices):
                    raise RuntimeError(f"Unexpected offload in component {name}: {devices}")
                component_metadata[name] = {
                    "devices": devices,
                    "dtypes": sorted({str(p.dtype) for p in parameters}),
                    "tensor_types": sorted({type(p).__name__ for p in parameters}),
                }
        if config["precision"] == "fp8":
            weights = [m.weight for m in denoiser.modules() if hasattr(m, "weight")]
            count = sum("Float8" in type(w).__name__ or "float8" in str(w.dtype) for w in weights)
            if not count:
                raise RuntimeError("No quantized FP8 weights detected; refusing mislabeled FP8 results")
            result["metadata"]["fp8_weight_tensors"] = count
        if config["tier"] == "optimized":
            denoiser.forward = torch.compile(denoiser.forward, fullgraph=True, mode="default")
        result["metadata"].update(
            backend="diffusers",
            precision=config["precision"],
            tier=config["tier"],
            offload_policy="none",
            components=component_metadata,
            attention_backend="native_pytorch_sdpa",
            compile=config["tier"] == "optimized",
            compile_fullgraph=True if config["tier"] == "optimized" else None,
            gpu_count=1,
            tp=1,
            gpu_model=torch.cuda.get_device_name(),
            compute_capability=list(capability),
            cuda=torch.version.cuda,
            libraries={
                p: importlib.metadata.version(p)
                for p in (requirement.split("==")[0] for requirement in REQUIREMENTS)
            },
            scheduler=type(pipe.scheduler).__name__,
            setup_seconds=time.monotonic() - started,
            guidance_mode=config["guidance_mode"],
            guidance_scale=config["guidance_scale"],
            true_cfg_scale=config["true_cfg_scale"],
            use_kv_cache=config.get("use_kv_cache"),
            safety_checker="disabled_or_absent",
            timing_basis="core wall excludes postprocess/PNG/transfer; E2E includes them",
        )
        reference = None
        if config.get("reference_image_base64"):
            data = base64.b64decode(config["reference_image_base64"])
            if hashlib.sha256(data).hexdigest() != config["reference_image_sha256"]:
                raise RuntimeError("Reference image SHA256 mismatch")
            reference = Image.open(io.BytesIO(data)).convert("RGB")
        timers = Timers(pipe, torch)
        save()

        def call(batch, index, retain=False):
            timers.reset()
            args = dict(
                prompt=config["prompts"][index % len(config["prompts"])],
                width=config["width"],
                height=config["height"],
                num_inference_steps=config["steps"],
                num_images_per_prompt=batch,
                output_type="pt",
                callback_on_step_end=timers.callback,
                generator=[
                    torch.Generator("cuda").manual_seed(config["seed"] + index * 32 + j) for j in range(batch)
                ],
            )
            args.update(guidance_arguments(config))
            if config["pipeline"] == "QwenImage21Pipeline" and config.get("use_kv_cache") is not None:
                args["use_kv_cache"] = config["use_kv_cache"]
            if reference is not None:
                args["image"] = reference
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode():
                output = pipe(**args).images
            torch.cuda.synchronize()
            stages = timers.results()
            if len(output) != batch or any(b not in (batch, batch * 2) for b in timers.batches):
                raise RuntimeError("True batch dimension/output count validation failed")
            encode_start = time.perf_counter()
            encoded_bytes = 0
            for j, image in enumerate(output):
                if not hasattr(image, "permute"):
                    raise RuntimeError("Expected tensor output for separated postprocessing")
                pixels = (
                    (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
                    .round()
                    .astype("uint8")
                )
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer, format="PNG")
                encoded_bytes += buffer.tell()
                if retain and j == 0:
                    (root / f"batch{batch}-sample.png").write_bytes(buffer.getvalue())
            record = {
                **stages,
                "images": batch,
                "core_seconds": timers.core_end - begin,
                "image_encoding_seconds": time.perf_counter() - encode_start,
                "e2e_seconds": time.perf_counter() - begin,
                "encoded_bytes": encoded_bytes,
                "seed_base": config["seed"] + index * 32,
                "prompt_index": index % len(config["prompts"]),
            }
            expected_steps = config.get("requested_denoiser_steps")
            if expected_steps and len(stages["denoiser_step_ms"]) != expected_steps:
                raise RuntimeError(
                    f"Expected {expected_steps} denoising steps, observed {len(stages['denoiser_step_ms'])}"
                )
            return record

        for mode in measurement_modes(config["execution_modes"]):
            search = {"mode": mode, "history": [], "reason": "running"}
            result["searches"].append(search)
            previous, baseline = None, None
            batch = 1
            cap = 1 if mode in ("latency", "tail") else config["batch_cap"]
            while batch <= cap:
                point_start = time.monotonic()
                point_deadline = min(deadline, point_start + config["point_budget_seconds"])
                trial = {
                    "config_id": config["config_id"],
                    "concurrency": 1,
                    "repeat": 0,
                    "backend": "diffusers",
                    "execution_mode": mode,
                    "batch_size": batch,
                    "precision": config["precision"],
                    "tier": config["tier"],
                    "status": "failed",
                }
                measured = []
                failure_stage = "warmup"
                try:
                    warmups = []
                    for index in range(config["warmup_requests"]):
                        if time.monotonic() >= point_deadline:
                            raise TimeoutError("Point budget exhausted during warmup")
                        warmups.append(call(batch, index)["e2e_seconds"])
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    pilot_seconds = steady_state_seconds(warmups)
                    trial.update(
                        warmup_seconds=warmups,
                        affordability_pilot_seconds=pilot_seconds,
                        affordability_basis="max_final_two_warmups",
                    )
                    count = sample_count(
                        config["requests"],
                        config["minimum_requests"],
                        pilot_seconds,
                        point_deadline - time.monotonic(),
                    )
                    if not count:
                        raise TimeoutError("Insufficient budget for minimum measured samples")
                    failure_stage = "measurement"
                    for index in range(count):
                        if point_deadline - time.monotonic() < pilot_seconds:
                            break
                        measured.append(call(batch, index, retain=index == 0))
                    if len(measured) < config["minimum_requests"]:
                        raise TimeoutError("Too few completed samples within point budget")
                    metrics = {
                        "images": len(measured) * batch,
                        "completed": len(measured),
                        "requested_calls": config["requests"],
                        "images_per_second": len(measured) * batch / sum(r["core_seconds"] for r in measured),
                        "p50_e2e_seconds": statistics.median(r["e2e_seconds"] for r in measured),
                        "p95_e2e_seconds": percentile([r["e2e_seconds"] for r in measured], 0.95),
                        "p99_e2e_seconds": percentile([r["e2e_seconds"] for r in measured], 0.99),
                        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "median_denoiser_step_ms": statistics.median(
                            [ms for r in measured for ms in r["denoiser_step_ms"]]
                        ),
                    }
                    for stage in (
                        "text_encode_seconds",
                        "denoise_total_seconds",
                        "vae_decode_seconds",
                        "postprocessing_seconds",
                        "image_encoding_seconds",
                    ):
                        metrics["median_" + stage] = statistics.median(r.get(stage, 0) for r in measured)
                    trial.update(
                        status="completed",
                        metrics=metrics,
                        requests=measured,
                        warmup_calls=len(warmups),
                        sample_reduction_reason="budget" if len(measured) < config["requests"] else None,
                    )
                    baseline = baseline if baseline is not None else metrics["p95_e2e_seconds"]
                    decision, gain = point_decision(previous, metrics, baseline, batch, cap)
                    search["history"].append(
                        {
                            "batch": batch,
                            "images_per_second": metrics["images_per_second"],
                            "gain": gain,
                            "decision": decision,
                        }
                    )
                    search["reason"] = decision
                    previous = metrics
                except Exception as error:
                    trial.update(
                        status=failure(error) if not isinstance(error, TimeoutError) else "budget_exhausted",
                        failure_kind=classify_failure(error, torch)[1] if not isinstance(error, TimeoutError) else "budget_exhausted",
                        failure_stage=failure_stage,
                        completed_calls_before_failure=len(measured),
                        partial_requests=measured,
                        error=str(error),
                    )
                    search["reason"] = trial["status"]
                result["trials"].append(trial)
                save()
                if trial["status"] != "completed" or search["reason"] != "continue":
                    break
                batch *= 2
            if available() <= 0 or search["reason"] in ("does_not_fit", "budget_exhausted"):
                break
        result["status"] = (
            "completed"
            if result["trials"] and all(t["status"] == "completed" for t in result["trials"])
            else "partial"
        )
        save()
    except Exception as error:
        row_error(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("directory")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.directory)


if __name__ == "__main__":
    main()
