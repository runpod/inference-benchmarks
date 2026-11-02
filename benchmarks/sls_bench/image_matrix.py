"""Canonical executable Diffusers matrix construction; no paid execution here."""

import base64
import hashlib
import struct
import zlib

from .config import Image, Matrix, MatrixRow


def reference_png():
    """Fixed synthetic editing reference: colored geometric objects, not a quality test."""
    size = 1024
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            if (x - 400) ** 2 + (y - 420) ** 2 < 180**2:
                color = (190, 35, 35)
            elif 180 < x < 850 and 640 < y < 760:
                color = (110, 75, 45)
            else:
                color = (230, 235, 240)
            row.extend(color)
        rows.append(b"\0" + row)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
        + chunk(b"IEND", b"")
    )


def tail_profile(profile, *, point_budget_seconds, row_budget_seconds):
    """Separate, explicitly budgeted B1 tail study; never launch as part of a curve."""
    return Image(**{
        **profile,
        "name": profile["name"][:60] + "-tails-b1-400",
        "requests": 400,
        "minimum_requests": 400,
        "execution_modes": ["tail"],
        "batch_cap": 1,
        "point_budget_seconds": point_budget_seconds,
        "row_budget_seconds": row_budget_seconds,
    }).model_dump(mode="json")


def klein_profile():
    """Klein 4B distilled, 4 steps, no CFG/offload/quantization or cross-call cache.

    Published component weights total ~14.87 GiB in BF16. Reserve runtime and
    activation headroom: 20 GB+ catalog devices are candidates, not proven fits.
    The model card's ~13 GB example uses CPU offload; we do not use that recipe.
    """
    return Image(name="df-flux2-klein-4b-stock", backend="diffusers",
                 model="black-forest-labs/FLUX.2-klein-4B", pipeline="Flux2KleinPipeline",
                 width=1024, height=1024, steps=4, requested_denoiser_steps=4,
                 guidance_mode="none", guidance_scale=1.0, true_cfg_scale=1.0,
                 min_vram_gb=20, precision="bf16", vae_dtype="bf16", tier="stock",
                 batch_cap=16, requests=40, minimum_requests=40, warmup_requests=3,
                 repeats=1, execution_modes=["batch"], row_budget_seconds=1500,
                 point_budget_seconds=900).model_dump(mode="json")


def current_generation_profiles():
    """Separate stock BF16 recipes; revisions are resolved/frozen before allocation.

    Qwen 2.1's published recipe is 2048 square / 40 steps / no CFG. Its
    safetensors alone total ~30.84 GiB, so 32 GB devices have no runtime headroom.
    FLUX.2-dev keeps its full transformer and Mistral encoder resident (~112 GB).
    Neither profile implies quantization, model offload, editing, or TP.
    """
    recipes = [
        dict(name="df-flux2-dev-current-stock", model="black-forest-labs/FLUX.2-dev",
             pipeline="Flux2Pipeline", width=1024, height=1024, steps=50,
             min_vram_gb=128, guidance_mode="distilled", guidance_scale=4.0,
             true_cfg_scale=1.0, gated_model=True),
        dict(name="df-qwen-image-21-current-stock", model="Qwen/Qwen-Image-2.1",
             pipeline="QwenImage21Pipeline", width=2048, height=2048, steps=40,
             min_vram_gb=48, guidance_mode="none", guidance_scale=0.0,
             true_cfg_scale=1.0, use_kv_cache=True),
    ]
    return [Image(**recipe, backend="diffusers", precision="bf16", tier="stock",
                  vae_dtype="bf16", batch_cap=8, requests=40, minimum_requests=40,
                  warmup_requests=3, repeats=1, requested_denoiser_steps=recipe["steps"],
                  execution_modes=["batch"], extended_curve_budget=True,
                  point_budget_seconds=3600, row_budget_seconds=6000).model_dump(mode="json")
            for recipe in recipes]


def definitions():
    reference = reference_png()
    cases = [
        (
            "flux1-bf16-standard",
            "black-forest-labs/FLUX.1-dev",
            "FluxPipeline",
            1024,
            28,
            16,
            48,
            "distilled",
            3.5,
            {},
        ),
        (
            "flux1-bf16-highres",
            "black-forest-labs/FLUX.1-dev",
            "FluxPipeline",
            2048,
            28,
            16,
            64,
            "distilled",
            3.5,
            {},
        ),
        (
            "flux1-fp8-standard",
            "black-forest-labs/FLUX.1-dev",
            "FluxPipeline",
            1024,
            28,
            16,
            24,
            "distilled",
            3.5,
            {"precision": "fp8"},
        ),
        (
            "qwen-image-standard",
            "Qwen/Qwen-Image",
            "QwenImagePipeline",
            1328,
            50,
            16,
            64,
            "true_cfg",
            4.0,
            {},
        ),
        ("qwen-image-highres", "Qwen/Qwen-Image", "QwenImagePipeline", 2048, 50, 16, 96, "true_cfg", 4.0, {}),
        (
            "qwen-image-edit",
            "Qwen/Qwen-Image-Edit",
            "QwenImageEditPipeline",
            1024,
            50,
            16,
            64,
            "true_cfg",
            4.0,
            {
                "reference_image_base64": base64.b64encode(reference).decode(),
                "reference_image_sha256": hashlib.sha256(reference).hexdigest(),
                "prompts": [
                    "Change the red circle to blue, preserving the background and rectangular table."
                ],
            },
        ),
        (
            "z-image-turbo",
            "Tongyi-MAI/Z-Image-Turbo",
            "ZImagePipeline",
            1024,
            8,
            32,
            18,
            "none",
            0.0,
            {"requested_denoiser_steps": 8},
        ),
        (
            "flux2-capacity",
            "black-forest-labs/FLUX.2-dev",
            "Flux2Pipeline",
            1024,
            28,
            8,
            128,
            "distilled",
            4.0,
            {"readiness": "blocked_access"},
        ),
        (
            "sd15-legacy",
            "sd-legacy/stable-diffusion-v1-5",
            "StableDiffusionPipeline",
            512,
            30,
            16,
            6,
            "true_cfg",
            7.0,
            {"vae_dtype": "fp32"},
        ),
        (
            "sdxl-legacy",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "StableDiffusionXLPipeline",
            1024,
            30,
            16,
            12,
            "true_cfg",
            7.0,
            {"vae_dtype": "fp32"},
        ),
    ]
    profiles, rows = [], []
    for name, model, pipeline, size, steps, cap, vram, guidance, scale, extra in cases:
        tiers = ["optimized"] if extra.get("precision") == "fp8" else ["stock", "optimized"]
        for tier in tiers:
            profile_name = f"df-{name}-{tier}"
            profile = Image(
                name=profile_name,
                backend="diffusers",
                pipeline=pipeline,
                model=model,
                tier=tier,
                width=size,
                height=size,
                steps=steps,
                batch_cap=cap,
                min_vram_gb=vram,
                guidance_mode=guidance,
                guidance_scale=scale,
                vae_dtype=extra.get("vae_dtype", "bf16"),
                requests=40,
                repeats=1,
                minimum_requests=40,
                warmup_requests=3,
                gated_model=model.startswith("black-forest-labs/"),
                **{k: v for k, v in extra.items() if k != "vae_dtype"},
            ).model_dump(mode="json")
            profiles.append(profile)
            rows.append(MatrixRow(name=f"{name}-{tier}", profiles=[profile_name]))
    matrix = Matrix(
        name="image_gen_diffusers",
        benchmark_type="image_gen",
        description="Direct Diffusers true BATCH (40 calls per point, B1 measured once), stock/optimized, no offload. Eight modern logical rows plus two legacy baselines expanded to 19 tier variants. FP8 transformer only, native kernels required. Pinned Diffusers 0.42.1 Z-Image uses 8 scheduler steps/8 expected forwards (older model-card 9-step recipe is not used). FLUX2 access blocked; multi-GPU unsupported, never implied TP2. Hard 45-minute worker-row budget including dependency/model setup, plus at most 10 minutes provisioning; 15-minute point target with reduced sample counts. Full sweep requires a passing FLUX1 BF16 smoke and review.",
        rows=rows,
    ).model_dump(mode="json")
    return profiles, matrix
