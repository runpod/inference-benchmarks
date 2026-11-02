import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_CONCURRENCY = 512


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["text", "image", "image_processing", "target", "suite", "matrix"]
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$", max_length=80)


class Text(Config):
    kind: Literal["text"] = "text"
    benchmark_type: Literal["text_gen"] = "text_gen"
    min_vram_gb: int = Field(default=18, ge=1)
    model: str = "Qwen/Qwen3-8B"
    model_revision: str = "main"  # resolved to a commit before allocating a Pod
    dataset_generator: Literal["exact_synthetic_v1", "sharegpt_v1"] = "exact_synthetic_v1"
    load_mode: Literal["closed_loop_concurrency", "open_loop_request_rate"] = "closed_loop_concurrency"
    prompt_mode: Literal["raw_completion", "chat_template"] = "raw_completion"
    enable_thinking: bool | None = None
    dataset_repo: str | None = None
    dataset_revision: str | None = None
    dataset_filename: str | None = None
    request_rates: list[float] = Field(default_factory=list)
    burstiness: float = Field(default=1.0, gt=0)
    min_compute_capability: int = Field(default=0, ge=0)
    kv_cache_dtype: Literal["auto"] = "auto"
    dtype: Literal["bfloat16", "half"] = "bfloat16"
    quantization: str | None = None
    input_tokens: int = Field(default=512, ge=1)
    output_tokens: int = Field(default=256, ge=2)
    max_model_len: int = Field(default=4096, ge=128, le=131072)
    concurrency: list[int] = Field(default_factory=lambda: [1, 4, 16], min_length=1)
    requests: int = Field(default=32, ge=1, le=100000)
    requests_by_concurrency: dict[int, int] = Field(default_factory=dict)
    repeats: int = Field(default=3, ge=1, le=100)
    warmup_requests: int = Field(default=2, ge=1, le=100)
    seed: int = 42
    gpu_memory_utilization: float = Field(default=0.92, gt=0.1, lt=1)
    max_num_seqs: int = Field(default=16, ge=1, le=MAX_CONCURRENCY)
    engine_capacity_policy: Literal["fixed", "match_concurrency"] = "fixed"
    max_num_batched_tokens: int = Field(default=8192, ge=1, le=131072)
    enable_chunked_prefill: bool = True
    saturation_max_concurrency: int | None = Field(default=None, ge=1, le=MAX_CONCURRENCY)
    saturation_min_gain: float = Field(default=0.05, gt=0, lt=1)
    saturation_plateau_steps: int = Field(default=2, ge=1, le=5)
    saturation_min_output_tok_s: float = Field(default=0, ge=0, allow_inf_nan=False)
    saturation_collapse_ratio: float = Field(default=0.5, gt=0, lt=1)
    measurement_budget_seconds: int | None = Field(default=None, ge=30, le=3600)
    measurement_sample_policy: Literal["fixed", "shrink_to_budget", "full_or_smoke"] = "fixed"
    coverage_first: bool = False
    cliff_aware: bool = False
    trial_timeout_seconds: int = Field(default=900, ge=10, le=259200)
    server_timeout_seconds: int = Field(default=600, ge=10, le=3600)
    prefix_caching: bool = False
    ttft_slo_ms: float | None = Field(default=None, gt=0)
    tpot_slo_ms: float | None = Field(default=None, gt=0)
    description: str = "Fixed-length synthetic completions; not a model quality evaluation."

    @model_validator(mode="after")
    def lengths(self):
        if self.measurement_sample_policy != "fixed" and self.measurement_budget_seconds is None:
            raise ValueError("Adaptive sampling requires an explicit measurement budget")
        if self.coverage_first and self.measurement_sample_policy == "fixed":
            raise ValueError("Coverage-first execution requires shrink_to_budget or full_or_smoke sampling")
        if self.cliff_aware and (
            not self.coverage_first
            or self.measurement_sample_policy == "fixed"
            or self.load_mode != "closed_loop_concurrency"
        ):
            raise ValueError(
                "Cliff-aware execution requires adaptive coverage scheduling of closed-loop points"
            )
        if (
            self.engine_capacity_policy == "fixed"
            and not self.enable_chunked_prefill
            and self.max_num_batched_tokens < self.max_model_len
        ):
            raise ValueError("Without chunked prefill, batched-token budget must cover max_model_len")
        if self.input_tokens + self.output_tokens > self.max_model_len:
            raise ValueError("input_tokens + output_tokens must be <= max_model_len")
        if any(c < 1 or c > min(self.requests, MAX_CONCURRENCY) for c in self.concurrency):
            raise ValueError(f"concurrency must be between 1 and min(requests, {MAX_CONCURRENCY})")
        if self.prompt_mode == "raw_completion" and self.enable_thinking is not None:
            raise ValueError("Thinking control requires an explicit chat template")
        if self.prompt_mode == "chat_template" and self.enable_thinking is None:
            raise ValueError("Chat-template profiles must explicitly select thinking mode")
        if self.load_mode == "open_loop_request_rate":
            if self.dataset_generator != "sharegpt_v1" or not all(
                (self.dataset_repo, self.dataset_revision, self.dataset_filename)
            ):
                raise ValueError("Open-loop mixed profiles require a pinned ShareGPT dataset")
            if not self.request_rates or any(r <= 0 or not math.isfinite(r) for r in self.request_rates):
                raise ValueError("Request rates must be positive finite values")
            if self.request_rates != sorted(set(self.request_rates)):
                raise ValueError("Request rates must be unique and increasing")
            if self.saturation_max_concurrency is not None:
                raise ValueError("Request-rate and concurrency searches are separate experiments")
        elif self.dataset_generator != "exact_synthetic_v1" or self.request_rates:
            raise ValueError("Fixed profiles require the exact synthetic generator and no request rates")
        if self.requests_by_concurrency:
            if self.load_mode != "closed_loop_concurrency":
                raise ValueError("Per-concurrency request counts require a closed-loop workload")
            for concurrency, requests in self.requests_by_concurrency.items():
                if self.saturation_max_concurrency is None:
                    if concurrency not in self.concurrency:
                        raise ValueError("Request-count schedule keys must be configured concurrency levels")
                elif (
                    concurrency < 1
                    or concurrency > self.saturation_max_concurrency
                    or concurrency & (concurrency - 1)
                ):
                    raise ValueError(
                        "Request-count schedule keys must be power-of-two search levels within the cap"
                    )
                if requests < concurrency or requests > self.requests:
                    raise ValueError(
                        "Scheduled requests must be between concurrency and the default request count"
                    )
        if len(set(self.concurrency)) != len(self.concurrency):
            raise ValueError("duplicate concurrency values")
        if self.saturation_max_concurrency is not None:
            if len(self.concurrency) != 1:
                raise ValueError("Saturation search requires one starting concurrency")
            start, cap = self.concurrency[0], self.saturation_max_concurrency
            if cap < start or cap > self.requests:
                raise ValueError("Saturation cap must be between starting concurrency and requests")
            if start & (start - 1) or cap & (cap - 1):
                raise ValueError("Saturation start and cap must be powers of two")
        return self


class Image(Config):
    kind: Literal["image"] = "image"
    benchmark_type: Literal["image_gen"] = "image_gen"
    min_vram_gb: int = Field(default=12, ge=1)
    backend: Literal["comfyui", "diffusers"] = "comfyui"
    pipeline: str | None = None
    precision: Literal["bf16", "fp8"] = "bf16"
    tier: Literal["stock", "optimized"] = "stock"
    execution_modes: list[Literal["latency", "batch", "tail"]] = Field(default_factory=lambda: ["batch"])
    batch_cap: int = Field(default=16, ge=1, le=32)
    minimum_requests: int = Field(default=8, ge=1, le=10000)
    extended_curve_budget: bool = False  # Explicit 40-call large-model curves; charged against Pod lease.
    point_budget_seconds: int = Field(default=900, ge=30, le=21600)
    row_budget_seconds: int = Field(default=2700, ge=60, le=21600)
    guidance_mode: Literal["distilled", "true_cfg", "none"] = "true_cfg"
    guidance_scale: float = Field(default=7.0, ge=0, le=30)
    true_cfg_scale: float = Field(default=4.0, ge=1, le=30)
    requested_denoiser_steps: int | None = None
    use_kv_cache: bool | None = None  # Qwen 2.1 prefix cache, pinned explicitly per recipe.
    offload_policy: Literal["none"] = "none"
    readiness: Literal["ready", "blocked_access", "blocked_multi_gpu"] = "ready"
    gated_model: bool = False
    reference_image_base64: str | None = None
    reference_image_sha256: str | None = None
    attention_backend: Literal["pytorch"] = "pytorch"
    vae_dtype: Literal["fp32", "fp16", "bf16"] = "fp32"
    gpu_only: bool = True
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    model_revision: str = "main"
    checkpoint: str = "sd_xl_base_1.0.safetensors"
    comfy_revision: str | None = None  # resolved only for the separate ComfyUI track
    width: int = Field(default=1024, ge=64, le=4096, multiple_of=8)
    height: int = Field(default=1024, ge=64, le=4096, multiple_of=8)
    steps: int = Field(default=30, ge=1, le=200)
    cfg: float = Field(default=7, ge=0, le=30)
    sampler: str = "euler"
    scheduler: str = "normal"
    batch_size: int = Field(default=1, ge=1, le=16)
    requests: int = Field(default=8, ge=1, le=10000)
    repeats: int = Field(default=3, ge=1, le=100)
    warmup_requests: int = Field(default=2, ge=1, le=100)
    seed: int = 42
    prompts: list[str] = Field(
        default_factory=lambda: [
            "A studio photograph of a red ceramic teapot on a wooden table, soft window light",
            "A watercolor landscape of mountains and a lake at sunrise",
            "An isometric illustration of a small greenhouse filled with plants",
        ],
        min_length=1,
    )
    trial_timeout_seconds: int = Field(default=900, ge=10, le=7200)
    server_timeout_seconds: int = Field(default=600, ge=10, le=3600)
    workflow: dict | None = None  # ComfyUI graph; never interpreted as a Diffusers pipeline

    @model_validator(mode="after")
    def image_backend(self):
        if self.backend == "comfyui":
            if "tail" in self.execution_modes:
                raise ValueError("Tail study is only implemented for direct Diffusers")
            if not self.workflow or not self.comfy_revision:
                raise ValueError("ComfyUI requires an explicit workflow and revision")
        else:
            allowed = {
                "FluxPipeline",
                "Flux2Pipeline",
                "Flux2KleinPipeline",
                "QwenImagePipeline",
                "QwenImage21Pipeline",
                "QwenImageEditPipeline",
                "ZImagePipeline",
                "StableDiffusionPipeline",
                "StableDiffusionXLPipeline",
            }
            if self.pipeline not in allowed:
                raise ValueError("Unsupported Diffusers pipeline")
            if self.use_kv_cache is not None and self.pipeline != "QwenImage21Pipeline":
                raise ValueError("Prefix KV-cache control is validated only for QwenImage21Pipeline")
            if self.warmup_requests < 3 or self.repeats != 1:
                raise ValueError(
                    "Diffusers requires at least three warmups and one independently recorded point"
                )
            if self.batch_cap & (self.batch_cap - 1):
                raise ValueError("Batch cap must be a power of two")
            if self.precision == "fp8" and self.tier != "optimized":
                raise ValueError("Native FP8 is a separately labeled optimized-tier row")
            if not self.gpu_only:
                raise ValueError("CPU offload is not implemented; never enable it implicitly")
            if not self.execution_modes or len(set(self.execution_modes)) != len(self.execution_modes):
                raise ValueError("Execution modes must be nonempty and unique")
            if self.minimum_requests > self.requests:
                raise ValueError("Minimum samples must not exceed requested samples")
            if "tail" in self.execution_modes:
                if self.execution_modes != ["tail"] or self.batch_cap != 1:
                    raise ValueError("Tail study must be a separate B1-only profile")
                if self.minimum_requests != self.requests:
                    raise ValueError("Tail study requires all requested samples; no budget downsampling")
            elif self.extended_curve_budget:
                if (
                    self.execution_modes != ["batch"]
                    or self.requests != self.minimum_requests
                    or self.requests != 40
                ):
                    raise ValueError(
                        "Extended curves require batch-only execution and exactly 40 measured calls"
                    )
            elif self.point_budget_seconds > 2700 or self.row_budget_seconds > 3300:
                raise ValueError(
                    "Long budgets require a separate tail study or explicit extended 40-call curve"
                )
        return self


class Target(Config):
    kind: Literal["target"] = "target"
    gpu_type: str
    cloud: Literal["SECURE", "COMMUNITY"] = "SECURE"
    gpu_count: int = Field(default=1, ge=1, le=8)
    image: str = "vllm/vllm-openai:v0.11.0"
    container_registry_auth_id: str | None = None
    disk_gb: int = Field(default=100, ge=40, le=500)
    min_cuda_version: str | None = "12.8"
    min_vcpu: int = Field(default=4, ge=2)
    min_ram_gb: int = Field(default=32, ge=16)
    data_centers: list[str] = Field(default_factory=list)
    expected_memory_gb: int | None = Field(default=None, ge=1)
    require_mig: bool | None = None


class ImageProcessing(Config):
    kind: Literal["image_processing"] = "image_processing"
    benchmark_type: Literal["image_processing"] = "image_processing"
    model: str
    model_revision: str = "main"
    parameters: dict = Field(default_factory=dict)
    min_vram_gb: int = Field(default=1, ge=1)


class Suite(Config):
    kind: Literal["suite"] = "suite"
    workloads: list[str] = Field(min_length=1)
    description: str = ""


class MatrixRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$", max_length=40)
    profiles: list[str] = Field(min_length=1)
    gpu_count: int = Field(default=1, ge=1, le=8)
    min_device_vram_gb: int = Field(default=1, ge=1)
    allowed_gpu_generations: list[str] = Field(default_factory=list)
    require_full_gpu: bool = False
    parameters: dict = Field(default_factory=dict)


class Matrix(Config):
    kind: Literal["matrix"] = "matrix"
    benchmark_type: Literal["text_gen", "image_gen", "image_processing"]
    parameters: dict = Field(default_factory=dict)
    rows: list[MatrixRow] = Field(min_length=1)
    description: str = ""

    @model_validator(mode="after")
    def identities(self):
        if len({r.name for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate matrix row names")
        for parameters in [self.parameters, *(r.parameters for r in self.rows)]:
            if set(parameters) & {"kind", "name", "benchmark_type"}:
                raise ValueError("Matrix parameters cannot change workload identity/type")
        return self


MODELS = {
    "text": Text,
    "image": Image,
    "image_processing": ImageProcessing,
    "target": Target,
    "suite": Suite,
    "matrix": Matrix,
}


def validate(value: dict) -> dict:
    return MODELS[value["kind"]].model_validate(value).model_dump(mode="json")


def sdxl_workflow() -> dict:
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 30,
                "cfg": 7,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "bench", "images": ["8", 0]}},
    }


def defaults() -> list[dict]:
    values = [
        Text(name="qwen8b-chat"),
        Text(name="qwen8b-prefill8k", input_tokens=8192, output_tokens=256, max_model_len=8448),
        Text(name="qwen8b-context16k", input_tokens=15360, output_tokens=1024, max_model_len=16384),
        Text(name="qwen8b-context32k", input_tokens=31744, output_tokens=1024, max_model_len=32768),
        Text(name="qwen32b-chat", model="Qwen/Qwen3-32B", min_vram_gb=72),
        Text(name="qwen8b-decode4k", input_tokens=512, output_tokens=4096, max_model_len=4608),
        Text(
            name="qwen8b-mixed",
            load_mode="open_loop_request_rate",
            dataset_generator="sharegpt_v1",
            dataset_repo="anon8231489123/ShareGPT_Vicuna_unfiltered",
            dataset_revision="main",
            dataset_filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            request_rates=[0.5, 1, 2, 4, 8, 16, 32],
            burstiness=0.5,
            max_num_seqs=64,
        ),
        # Only used for image runs; resolved to an immutable Git commit before provisioning.
        Image(name="sdxl-1024", comfy_revision="master", workflow=sdxl_workflow()),
        Image(
            name="sd15-512",
            model="sd-legacy/stable-diffusion-v1-5",
            checkpoint="v1-5-pruned-emaonly.safetensors",
            width=512,
            height=512,
            min_vram_gb=6,
            comfy_revision="master",
            workflow=sdxl_workflow(),
        ),
        Image(
            name="sd15-smoke",
            model="sd-legacy/stable-diffusion-v1-5",
            checkpoint="v1-5-pruned-emaonly.safetensors",
            width=512,
            height=512,
            min_vram_gb=6,
            comfy_revision="master",
            workflow=sdxl_workflow(),
            requests=1,
            repeats=1,
            warmup_requests=1,
        ),
        Suite(
            name="text",
            workloads=["qwen8b-chat", "qwen8b-prefill8k", "qwen8b-context16k", "qwen8b-context32k"],
        ),
        Suite(
            name="image",
            workloads=["sd15-512", "sdxl-1024"],
            description="GPU-only fixed-backend SD1.5/512 and SDXL/1024; profile VRAM gates are estimates.",
        ),
        Suite(name="image-smoke", workloads=["sd15-smoke"]),
        Matrix(
            name="text_gen",
            benchmark_type="text_gen",
            description="Four-point C1/8/48/128 text matrix: 8B BF16 TP1, 32B BF16 TP2, 32B FP8 TP1, MoE TP1, and separate mixed-arrival ShareGPT; VRAM floors are estimates.",
            parameters={
                "requests": 1024,
                "requests_by_concurrency": {1: 512},
                "repeats": 2,
                "trial_timeout_seconds": 259200,
                "concurrency": [1, 8, 48, 128],
                "saturation_max_concurrency": None,
                "prompt_mode": "chat_template",
                "enable_thinking": False,
                "prefix_caching": False,
                "kv_cache_dtype": "auto",
                "saturation_min_gain": 0.05,
                "saturation_plateau_steps": 2,
                "saturation_min_output_tok_s": 25,
                "saturation_collapse_ratio": 0.5,
            },
            rows=[
                MatrixRow(
                    name="qwen8b-bf16",
                    profiles=[
                        "qwen8b-chat",
                        "qwen8b-prefill8k",
                        "qwen8b-decode4k",
                        "qwen8b-context16k",
                        "qwen8b-context32k",
                    ],
                    parameters={
                        "model": "Qwen/Qwen3-8B",
                        "model_revision": "main",
                        "dtype": "bfloat16",
                        "quantization": None,
                        "min_vram_gb": 18,
                    },
                ),
                MatrixRow(
                    name="qwen32b-bf16-tp2",
                    profiles=[
                        "qwen8b-chat",
                        "qwen8b-prefill8k",
                        "qwen8b-decode4k",
                        "qwen8b-context16k",
                        "qwen8b-context32k",
                    ],
                    gpu_count=2,
                    min_device_vram_gb=48,
                    require_full_gpu=True,
                    parameters={
                        "model": "Qwen/Qwen3-32B",
                        "model_revision": "main",
                        "dtype": "bfloat16",
                        "quantization": None,
                        "min_vram_gb": 72,
                    },
                ),
                MatrixRow(
                    name="qwen32b-fp8",
                    profiles=["qwen8b-chat", "qwen8b-context32k"],
                    allowed_gpu_generations=["HOPPER", "BLACKWELL"],
                    parameters={
                        "model": "Qwen/Qwen3-32B-FP8",
                        "model_revision": "main",
                        "dtype": "bfloat16",
                        "quantization": "fp8",
                        "min_vram_gb": 48,
                        "min_compute_capability": 90,
                    },
                ),
                MatrixRow(
                    name="qwen30b-a3b-bf16",
                    profiles=["qwen8b-chat", "qwen8b-decode4k"],
                    parameters={
                        "model": "Qwen/Qwen3-30B-A3B",
                        "model_revision": "main",
                        "dtype": "bfloat16",
                        "quantization": None,
                        "min_vram_gb": 72,
                    },
                ),
                MatrixRow(
                    name="qwen8b-mixed",
                    profiles=["qwen8b-mixed"],
                    parameters={
                        "concurrency": [1],
                        "saturation_max_concurrency": None,
                        "requests_by_concurrency": {},
                    },
                ),
            ],
        ),
        Matrix(
            name="image_gen",
            benchmark_type="image_gen",
            rows=[
                MatrixRow(name="sd15", profiles=["sd15-512"]),
                MatrixRow(name="sdxl", profiles=["sdxl-1024"]),
            ],
        ),
    ]
    return [v.model_dump(mode="json") for v in values]
