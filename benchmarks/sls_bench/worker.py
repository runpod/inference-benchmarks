"""Uploaded to Pods verbatim. No controller dependencies or org credentials required.

Transient spec/report files are transport/checkpoints, not configuration storage.
The controller imports their contents and logs/images into SQLite during execution.
"""

import copy
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = {"status": "starting", "hardware": {}, "trials": [], "phases": []}


class ImageRowDeadline(TimeoutError):
    """Absolute image deadline: no graceful shutdown or artifact copying afterward."""


class TextTimeBudget(TimeoutError):
    """Unmeasured point, not a hardware failure or throughput measurement."""


class WorkerCancelled(BaseException):
    """Cancellation must escape workload failure handling and stop the whole worker."""


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def checkpoint():
    tmp = ROOT / "report.tmp"
    tmp.write_text(json.dumps(clean(REPORT)))
    tmp.replace(ROOT / "report.json")


def phase(name, started, **extra):
    REPORT["phases"].append({"name": name, "seconds": time.monotonic() - started, **extra})
    checkpoint()


def command(args, log, timeout=900, env=None, progress=None):
    with (ROOT / log).open("ab") as output:
        process = subprocess.Popen(
            args, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    code = process.wait(timeout=min(10, remaining) if progress else remaining)
                    break
                except subprocess.TimeoutExpired:
                    if progress:
                        progress()
                    else:
                        raise
        except ImageRowDeadline:
            stop(process, grace=0)
            raise
        except BaseException:
            stop(process)
            raise
    if code:
        raise RuntimeError(f"Command exited {code}; see {log}")


def stop(process, grace=20):
    if process is None:
        return
    if not grace:
        # An absolute deadline cannot synchronously wait for even a SIGKILLed child
        # (e.g. uninterruptible kernel I/O). Pod lease remains the independent backstop.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def api(path, body=None, port=8000, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as result:
        return json.load(result)


def ready(process, port, path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited {process.returncode} before readiness")
        try:
            api(path, port=port, timeout=3)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("Server readiness timed out")


def start(args, filename):
    handle = (ROOT / filename).open("ab")
    process = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    return process


def capture(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=30).decode(errors="replace")
    except Exception as error:
        return str(error)


def hardware():
    # Probe in a short-lived process: the controller worker must not retain a CUDA context/VRAM.
    probe = """import json, torch
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'torch_devices': [
 {'name': torch.cuda.get_device_name(i), 'memory_bytes': torch.cuda.get_device_properties(i).total_memory,
  'compute_capability': list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]}))
"""
    cuda = json.loads(subprocess.check_output([sys.executable, "-c", probe], timeout=60))
    return {
        "nvidia_smi": capture(["nvidia-smi", "-q"]),
        "devices": capture(["nvidia-smi", "-L"]),
        "topology": capture(["nvidia-smi", "topo", "-m"]),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **cuda,
        "python": sys.version,
        "cpu": capture(["lscpu"]),
        "ram": Path("/proc/meminfo").read_text(),
        "packages": capture([sys.executable, "-m", "pip", "freeze"]),
    }


def verify_hardware(target, details):
    devices = details["torch_devices"]
    if len(devices) != target["gpu_count"]:
        raise RuntimeError(f"Expected {target['gpu_count']} visible CUDA devices, found {len(devices)}")
    if target["require_mig"] and not any("MIG" in d["name"].upper() for d in devices):
        raise RuntimeError("Requested MIG but CUDA device is not a MIG slice")
    for device in devices:
        # NVIDIA marketing GB and CUDA GiB differ slightly. Reject a full parent card or wrong slice.
        gib = device["memory_bytes"] / 1024**3
        if not target["expected_memory_gb"] * 0.85 <= gib <= target["expected_memory_gb"] * 1.12:
            raise RuntimeError(f"Unexpected visible VRAM: {gib:.2f} GiB")


class Telemetry:
    """NVML/nvidia-smi is not always available per MIG instance; preserve raw samples, don't invent VRAM."""

    def __init__(self, label):
        self.label = label
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        with (ROOT / f"{self.label}-telemetry.jsonl").open("w") as output:
            while not self.done.is_set():
                value = capture(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                        "--format=csv,noheader,nounits",
                    ]
                )
                output.write(json.dumps({"at": time.time(), "process_memory": value}) + "\n")
                output.flush()
                self.done.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join(timeout=35)


def bench_command(
    config, model_path, concurrency, filename, requests, seed, dataset_path=None, request_rate=None
):
    args = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        "http://127.0.0.1:8000",
        "--endpoint",
        "/v1/completions",
        "--model",
        model_path,
        "--tokenizer",
        model_path,
        "--served-model-name",
        "benchmark",
        "--dataset-name",
        "random",
        "--seed",
        str(seed),
        "--random-input-len",
        str(config["input_tokens"]),
        "--random-output-len",
        str(config["output_tokens"]),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(requests),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--temperature",
        "0",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(ROOT),
        "--result-filename",
        filename,
    ]
    if dataset_path is not None:
        for flag in ("--random-input-len", "--random-output-len", "--random-range-ratio"):
            index = args.index(flag)
            del args[index : index + 2]
        mixed = config.get("dataset_generator") == "sharegpt_v1"
        args[args.index("--dataset-name") + 1] = "sharegpt" if mixed else "custom"
        args.extend(["--dataset-path", str(dataset_path), "--no-oversample"])
        if not mixed:
            args.extend(["--custom-skip-chat-template", "--custom-output-len", str(config["output_tokens"])])
    if request_rate is not None:
        args[args.index("--request-rate") + 1] = str(request_rate)
        args.extend(["--burstiness", str(config["burstiness"])])
        # No client semaphore: all scheduled arrivals reach the engine queue.
        # Otherwise an overloaded test silently becomes closed-loop throttling.
        index = args.index("--max-concurrency")
        del args[index : index + 2]
    slos = [
        f"{metric}:{config[field]}"
        for field, metric in (("ttft_slo_ms", "ttft"), ("tpot_slo_ms", "tpot"))
        if config.get(field) is not None
    ]
    if slos:
        args.extend(["--goodput", *slos])
    return args


def normalize_text(raw, expected):
    requests = []
    count = len(raw.get("input_lens", []))
    for i in range(count):
        itls = raw.get("itls", [[]] * count)[i]
        ttft = raw.get("ttfts", [0] * count)[i]
        requests.append(
            {
                "input_tokens": raw["input_lens"][i],
                "output_tokens": raw.get("output_lens", [0] * count)[i],
                "ttft_seconds": ttft,
                # Full token-level timings live in the compressed raw result artifact.
                "itl_count": len(itls),
                # Exact aggregate E2E is available; vLLM 0.11 does not expose per-request final latencies.
                "stream_token_span_seconds": ttft + sum(itls),
                "error": raw.get("errors", [""] * count)[i],
            }
        )
    metrics = {k: v for k, v in raw.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    metrics["expected_requests"] = expected
    metrics["failed_requests"] = expected - raw.get("completed", 0)
    return metrics, requests


def failure_kind(message):
    text = message.lower()
    if "out of memory" in text or "cuda oom" in text:
        return "out_of_memory"
    if "kv cache" in text and any(
        term in text for term in ("larger", "maximum", "not enough", "insufficient")
    ):
        return "context_capacity"
    if any(term in text for term in ("no kernel image", "unsupported gpu", "unsupported compute capability")):
        return "unsupported_hardware"
    return "execution_error"


def download(config):
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    path = snapshot_download(
        config["model"],
        revision=config["model_revision"],
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "*.h5", "original/*"],
    )
    # Hash every file actually used, including weight shards (once per model revision).
    files = []
    for file in sorted(Path(path).rglob("*")):
        if file.is_file():
            sha = hashlib.sha256()
            with file.open("rb") as data:
                for block in iter(lambda: data.read(8 * 1024 * 1024), b""):
                    sha.update(block)
            files.append(
                {"path": str(file.relative_to(path)), "sha256": sha.hexdigest(), "bytes": file.stat().st_size}
            )
    phase(
        "model_download_and_hash",
        started,
        model=config["model"],
        revision=config["model_revision"],
        files=files,
    )
    return path


def exact_prompt(tokenizer, length, rng, render=None):
    words = (
        "apple river cloud stone green summer light table paper mountain window garden music planet".split()
    )
    text = " ".join(rng.choices(words, k=length))
    for _ in range(32):
        prompt = render(text) if render else text
        actual = len(tokenizer(prompt).input_ids)
        if actual == length:
            return prompt
        if actual > length:
            ids = tokenizer.encode(text, add_special_tokens=False)
            text = tokenizer.decode(ids[: max(0, len(ids) - (actual - length))])
        else:
            text += " x" * (length - actual)
    raise ValueError(f"Tokenizer cannot produce the exact requested input length {length}")


def render_chat(config, tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=config["enable_thinking"],
    )


def mixed_corpus(config, tokenizer, count, seed, filename):
    from huggingface_hub import hf_hub_download

    source = Path(
        hf_hub_download(
            repo_id=config["dataset_repo"],
            repo_type="dataset",
            revision=config["dataset_revision"],
            filename=config["dataset_filename"],
        )
    )
    sha = hashlib.sha256()
    with source.open("rb") as data:
        for block in iter(lambda: data.read(8 * 1024 * 1024), b""):
            sha.update(block)
    data = json.loads(source.read_text())
    random.Random(seed).shuffle(data)
    selected = []
    for entry in data:
        turns = entry.get("conversations", [])
        if (
            len(turns) < 2
            or turns[0].get("from") not in ("human", "user")
            or turns[1].get("from") not in ("gpt", "assistant")
        ):
            continue
        prompt, completion = turns[0].get("value"), turns[1].get("value")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            continue
        if config.get("prompt_mode") == "chat_template":
            prompt = render_chat(config, tokenizer, prompt)
        inputs, outputs = len(tokenizer(prompt).input_ids), len(tokenizer(completion).input_ids)
        # Explicitly match vLLM 0.11 ShareGPT filtering, not an unbounded claim
        # about the entire public ShareGPT distribution.
        if not (
            4 <= inputs <= 1024
            and outputs >= 4
            and inputs + outputs <= min(2048, config.get("_profile_max_model_len", config["max_model_len"]))
        ):
            continue
        selected.append(
            {
                "source_id": entry.get("id"),
                "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": completion}],
            }
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible ShareGPT pairs; need {count}; never oversample")
    path = ROOT / filename
    path.write_text(json.dumps(selected))
    REPORT.setdefault("datasets", []).append(
        {
            "path": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": seed,
            "requests": count,
            "generator": "sharegpt_v1",
            "repo": config["dataset_repo"],
            "revision": config["dataset_revision"],
            "source_sha256": sha.hexdigest(),
            "prompt_mode": config.get("prompt_mode"),
            "enable_thinking": config.get("enable_thinking"),
            "filter": {"min_tokens": 4, "max_input_tokens": 1024, "max_total_tokens": 2048},
        }
    )
    checkpoint()
    return path


def lengths_match(config, tokenizer, dataset, requests):
    actual = [(r["input_tokens"], r["output_tokens"]) for r in requests if not r["error"]]
    if config.get("dataset_generator") == "sharegpt_v1":
        expected = [
            (
                len(tokenizer(e["conversations"][0]["value"]).input_ids),
                len(tokenizer(e["conversations"][1]["value"]).input_ids),
            )
            for e in json.loads(dataset.read_text())
        ]
        return Counter(actual) == Counter(expected)
    return all(i == config["input_tokens"] and o == config["output_tokens"] for i, o in actual)


def corpus(config, tokenizer, count, seed, filename):
    if config.get("dataset_generator") == "sharegpt_v1":
        return mixed_corpus(config, tokenizer, count, seed, filename)
    rng = random.Random(seed)
    path = ROOT / filename
    with path.open("w") as output:
        for _ in range(count):
            render = (
                (lambda text: render_chat(config, tokenizer, text))
                if config.get("prompt_mode") == "chat_template"
                else None
            )
            prompt = exact_prompt(tokenizer, config["input_tokens"], rng, render=render)
            output.write(json.dumps({"prompt": prompt}) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    REPORT.setdefault("datasets", []).append(
        {
            "path": filename,
            "sha256": sha,
            "seed": seed,
            "input_tokens": config["input_tokens"],
            "requests": count,
            "generator": "exact_synthetic_v1",
            "prompt_mode": config.get("prompt_mode", "raw_completion"),
            "enable_thinking": config.get("enable_thinking"),
        }
    )
    checkpoint()
    return path


class LowThroughputStop(RuntimeError):
    """Operational cutoff, not a completed trial or proof of hardware saturation."""


class ThroughputWatch:
    """Abort sustained slow generation without waiting for a 1,024-request batch.

    Uses the server's actual generated-token counter, never offered request rate.
    Three-minute grace, then a two-minute rolling window including queue/prefill time.
    """

    def __init__(self, config, prior_best=0):
        self.config, self.prior_best = config, prior_best
        self.started = time.monotonic()
        self.samples = []
        self.failures = 0

    def observe(self, now, total):
        self.samples.append((now, total))
        while len(self.samples) > 2 and self.samples[1][0] <= now - 120:
            self.samples.pop(0)
        elapsed = now - self.samples[0][0]
        if now - self.started < 180 or elapsed < 120:
            return
        rate = (total - self.samples[0][1]) / elapsed
        try:
            throughput_cutoff(self.config, rate, self.prior_best)
        except LowThroughputStop:
            REPORT.setdefault("throughput_probes", []).append(
                {
                    "config_id": self.config["config_id"],
                    "phase": "live_generation",
                    "output_tok_s": rate,
                    "window_seconds": elapsed,
                    "decision": "low_throughput_cutoff",
                }
            )
            checkpoint()
            raise

    def __call__(self):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=3) as response:
                lines = response.read().decode().splitlines()
            values = [
                float(line.rsplit(None, 1)[1])
                for line in lines
                if line.startswith(("vllm:generation_tokens_total{", "vllm:generation_tokens_total "))
            ]
            if not values:
                raise RuntimeError("Missing vLLM generated-token counter")
        except Exception as error:
            self.failures += 1
            if self.failures >= 9:
                raise RuntimeError(
                    "Live throughput guard unavailable; refusing an unguarded long trial"
                ) from error
            return
        self.failures = 0
        self.observe(time.monotonic(), sum(values))


def throughput_cutoff(config, throughput, prior_best=0):
    floor = config.get("saturation_min_output_tok_s", 0)
    if not math.isfinite(throughput) or throughput <= 0:
        raise LowThroughputStop(f"Invalid or zero output throughput: {throughput}")
    if floor and throughput < floor:
        raise LowThroughputStop(f"Output throughput {throughput:.2f} tok/s below floor {floor:.2f}")
    if prior_best and throughput < prior_best * config.get("saturation_collapse_ratio", 0.5):
        raise LowThroughputStop(
            f"Output throughput {throughput:.2f} tok/s collapsed from best {prior_best:.2f}"
        )


ENGINE_KEYS = (
    "model",
    "model_revision",
    "dtype",
    "quantization",
    "gpu_memory_utilization",
    "kv_cache_dtype",
    "prefix_caching",
    "seed",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
)


def fixed_engine(config):
    # Frozen specs without the new field retain their historical experiment.
    return config.get("engine_capacity_policy", "match_concurrency") == "fixed"


def text_engine_key(config):
    return tuple(config.get(k) for k in ENGINE_KEYS)


def text_engine_plan(workloads):
    """One fixed engine per compatible model/config group; no speculative KV autotuning."""
    groups = {}
    for c in workloads:
        if c.get("kind") != "text" or not fixed_engine(c):
            continue
        key = text_engine_key(c)
        group = groups.setdefault(key, {"config_ids": [], "engine_config": {k: c[k] for k in ENGINE_KEYS}})
        engine = group["engine_config"]
        group["config_ids"].append(c["config_id"])
        engine["max_model_len"] = max(engine.get("max_model_len", 0), c["max_model_len"])
        engine["max_num_seqs"] = max(engine.get("max_num_seqs", 0), c["max_num_seqs"])
        if (
            not engine["enable_chunked_prefill"]
            and engine["max_num_batched_tokens"] < engine["max_model_len"]
        ):
            raise ValueError("Shared engine context exceeds non-chunked batched-token budget")
    return list(groups.values())


def executed_text_engine(config, target):
    if config.get("kind") != "text" or not fixed_engine(config):
        return None
    return {
        **{k: config[k] for k in (*ENGINE_KEYS, "max_model_len", "max_num_seqs")},
        "tensor_parallel_size": target.get("gpu_count"),
    }


class TextEngineSession:
    """Own at most one GPU engine. Reuse only an exactly matching launch command."""

    def __init__(self):
        self.process = None
        self.signature = None
        self.instance_id = 0
        self.log = None
        self.startup_failures = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def acquire(self, args, log, timeout):
        signature = tuple(args)
        if signature in self.startup_failures:
            raise RuntimeError(
                "Engine startup circuit open; not retrying an identical failed launch in this run: "
                + self.startup_failures[signature]
            )
        reused = self.process is not None and self.process.poll() is None and self.signature == signature
        if not reused:
            self.close()
            self.instance_id += 1
            self.signature = signature
            self.log = log
            try:
                self.process = start(args, log)
                ready(self.process, 8000, "/v1/models", timeout)
            except Exception as error:
                self.startup_failures[signature] = str(error)
                self.close()
                raise
        else:
            ready(self.process, 8000, "/v1/models", timeout)
        return self.process, reused, self.instance_id

    def close(self):
        if self.process is not None:
            try:
                stop(self.process)
            finally:
                self.process = None
                self.signature = None
            time.sleep(5)


def reset_text_prefix_cache():
    # The successful vLLM response may have an empty body; do not JSON-decode it.
    request = urllib.request.Request("http://127.0.0.1:8000/reset_prefix_cache", method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def text_workload(config, target, models):
    if fixed_engine(config) and "_engine_session" not in config:
        with TextEngineSession() as session:
            return _text_workload({**config, "_engine_session": session}, target, models)
    return _text_workload(config, target, models)


def _text_workload(config, target, models):
    # Standalone uploaded worker: enforce the global limit even if handed an old spec.
    if (
        max(config["concurrency"]) > 512
        or (config.get("saturation_max_concurrency") or 0) > 512
        or config["max_num_seqs"] > 512
    ):
        raise ValueError("Global concurrency/engine cap is 512; prepare a revised config")
    required = config.get("min_compute_capability", 0)
    if required:
        devices = REPORT.get("hardware", {}).get("torch_devices", [])
        if not devices or any(
            d["compute_capability"][0] * 10 + d["compute_capability"][1] < required for d in devices
        ):
            raise RuntimeError(f"Unsupported GPU compute capability; workload requires >= {required}")
    if config.get("saturation_max_concurrency") is None:
        if config.get("measurement_sample_policy") in ("shrink_to_budget", "full_or_smoke") or config.get(
            "requests_by_concurrency"
        ):
            # Each arrival rate/concurrency is a separate point with its own budget.
            mixed = config.get("load_mode") == "open_loop_request_rate"
            field = "request_rates" if mixed else "concurrency"
            for point in config[field]:
                schedule = config.get("requests_by_concurrency", {})
                level = {
                    **config,
                    field: [point],
                    "_point_label": str(point),
                    "requests": schedule.get(str(point), schedule.get(point, config["requests"])),
                    "requests_by_concurrency": {},
                }
                if not mixed and not fixed_engine(config):
                    level["max_num_seqs"] = point
                before = len(REPORT["trials"])
                try:
                    fixed_text_workload(level, target, models)
                except TextTimeBudget:
                    continue  # Incomplete attempted point; still visit remaining points.
                except Exception as error:
                    if not config.get("coverage_first"):
                        raise
                    if len(REPORT["trials"]) == before:
                        REPORT["trials"].append(
                            {
                                "config_id": config["config_id"],
                                "concurrency": 0 if mixed else point,
                                "request_rate": point if mixed else None,
                                "repeat": -1,
                                "status": "setup_failed",
                                "error": str(error),
                            }
                        )
                    checkpoint()
            return
        before = len(REPORT["trials"])
        try:
            return fixed_text_workload(config, target, models)
        except TextTimeBudget:
            return
        except LowThroughputStop as error:
            if len(REPORT["trials"]) == before:
                REPORT["trials"].append(
                    {
                        "config_id": config["config_id"],
                        "concurrency": 0
                        if config.get("load_mode") == "open_loop_request_rate"
                        else config["concurrency"][0],
                        "repeat": 0,
                        "status": "skipped_low_throughput",
                        "error": str(error),
                    }
                )
            REPORT.setdefault("saturation", []).append(
                {"config_id": config["config_id"], "reason": "low_throughput_cutoff", "error": str(error)}
            )
            checkpoint()
            return
    search = {"config_id": config["config_id"], "history": [], "reason": "running"}
    REPORT.setdefault("saturation", []).append(search)
    concurrency = config["concurrency"][0]
    best = 0.0
    plateau = 0
    while concurrency <= config["saturation_max_concurrency"]:
        # Fixed mode sweeps client load (including queueing), not engine capacity.
        # Historical match_concurrency mode remains a separate experiment.
        request_schedule = config.get("requests_by_concurrency", {})
        level_requests = request_schedule.get(
            str(concurrency), request_schedule.get(concurrency, config["requests"])
        )
        level = {
            **config,
            "requests": level_requests,
            "concurrency": [concurrency],
            "max_num_seqs": config["max_num_seqs"] if fixed_engine(config) else concurrency,
            "_prior_best_output_tok_s": best,
        }
        before = len(REPORT["trials"])
        try:
            fixed_text_workload(level, target, models)
        except TextTimeBudget:
            key = "time_budget_incomplete" if config.get("coverage_first") else "time_budget_skips"
            search.setdefault(key, []).append(concurrency)
            plateau = 0  # Never infer a contiguous plateau across unmeasured levels.
            checkpoint()
            concurrency *= 2
            continue
        except LowThroughputStop as error:
            search.update(reason="low_throughput_cutoff", error=str(error), last_concurrency=concurrency)
            if len(REPORT["trials"]) == before:
                REPORT["trials"].append(
                    {
                        "config_id": config["config_id"],
                        "concurrency": concurrency,
                        "repeat": 0,
                        "status": "skipped_low_throughput",
                        "error": str(error),
                    }
                )
            checkpoint()
            return
        except Exception as error:
            search.setdefault("failed_points", []).append(concurrency)
            search.update(reason="runtime_failure", error=str(error), last_concurrency=concurrency)
            level_trials = REPORT["trials"][before:]
            if all(t["status"] == "completed" for t in level_trials):
                REPORT["trials"].append(
                    {
                        "config_id": config["config_id"],
                        "concurrency": concurrency,
                        "repeat": len(level_trials),
                        "status": "setup_failed",
                        "error": str(error),
                        "engine_max_num_seqs": level["max_num_seqs"],
                    }
                )
            checkpoint()
            if config.get("coverage_first"):
                concurrency *= 2
                continue
            return
        trials = REPORT["trials"][before:]
        rates = [t.get("metrics", {}).get("output_throughput") for t in trials]
        if (
            len(trials) != config["repeats"]
            or any(t["status"] != "completed" for t in trials)
            or any(r is None or not math.isfinite(r) or r <= 0 for r in rates)
        ):
            search.setdefault("failed_points", []).append(concurrency)
            search.update(reason="trial_failure", last_concurrency=concurrency)
            checkpoint()
            if config.get("coverage_first"):
                concurrency *= 2
                continue
            return
        throughput = statistics.median(rates)
        try:
            if not config.get("coverage_first"):
                throughput_cutoff(config, throughput, best)
        except LowThroughputStop as error:
            search.update(reason="low_throughput_cutoff", error=str(error), last_concurrency=concurrency)
            checkpoint()
            return
        gain = throughput / best - 1 if best else None
        plateau = plateau + 1 if gain is not None and gain < config["saturation_min_gain"] else 0
        search["history"].append(
            {
                "concurrency": concurrency,
                "median_output_tok_s": throughput,
                "requests_per_repeat": trials[0].get("requested_requests", level_requests),
                "planned_requests_per_repeat": level_requests,
                "actual_requests_by_repeat": [t.get("requested_requests", level_requests) for t in trials],
                "reduced_sample": any(t.get("reduced_sample", False) for t in trials),
                "gain_over_prior_best": gain,
                "engine_max_num_seqs": level["max_num_seqs"],
            }
        )
        if throughput > best:
            best = throughput
            search.update(best_concurrency=concurrency, best_output_tok_s=best)
        search["last_concurrency"] = concurrency
        if plateau >= config["saturation_plateau_steps"] and not config.get("coverage_first"):
            search["reason"] = (
                "time_budget_limited" if search.get("time_budget_skips") else "throughput_plateau"
            )
            checkpoint()
            return
        checkpoint()
        concurrency *= 2
    # Reaching the cap is NOT proof of saturation.
    search["reason"] = (
        "incomplete_coverage"
        if config.get("coverage_first")
        and (search.get("time_budget_incomplete") or search.get("failed_points"))
        else "time_budget_limited"
        if search.get("time_budget_skips")
        else "concurrency_cap"
    )
    checkpoint()


def text_measurement_estimate(config, raw, pilot_count, pilot_seconds):
    """Scale only measured serving time; charge client/setup overhead once per repeat.

    This is admission evidence, not a throughput measurement or a guarantee.
    The actual measurement still has an independent absolute deadline.
    """
    duration = raw.get("duration")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Time-budget pilot has no valid measured benchmark duration")
    overhead = max(0.0, pilot_seconds - duration)
    sampling = duration * config["requests"] / pilot_count * config["repeats"]
    return {
        "estimator": "serving_duration_plus_fixed_overhead_v2",
        "benchmark_seconds": duration,
        "fixed_overhead_seconds_per_repeat": overhead,
        "sampling_safety_factor": 1.15,
        "estimated_measurement_seconds": sampling * 1.15 + overhead * config["repeats"],
    }


def text_boundary_reason(config, throughput, state):
    """A pilot is advisory; only completed measurements confirm the boundary."""
    if not isinstance(throughput, (int, float)) or not math.isfinite(throughput) or throughput <= 0:
        return None
    best = state.get("best_output_tok_s", 0)
    if throughput < config.get("saturation_min_output_tok_s", 0):
        return "throughput_floor"
    if best and throughput < best * config.get("saturation_collapse_ratio", 0.5):
        return "throughput_collapse"
    if (
        best
        and throughput / best - 1 < config.get("saturation_min_gain", 0.05)
        and state.get("plateau_steps", 0) + 1 >= config.get("saturation_plateau_steps", 2)
    ):
        return "throughput_plateau"
    return None


def record_text_boundary(config, trials, state):
    completed = [t for t in trials if t["status"] == "completed"]
    rates = [t.get("metrics", {}).get("output_throughput") for t in completed]
    valid = [r for r in rates if isinstance(r, (int, float)) and math.isfinite(r) and r > 0]
    throughput = statistics.median(valid) if valid else None
    reason = text_boundary_reason(config, throughput, state)
    best = state.get("best_output_tok_s", 0)
    if throughput is not None:
        state["plateau_steps"] = (
            state.get("plateau_steps", 0) + 1
            if best and throughput / best - 1 < config["saturation_min_gain"]
            else 0
        )
        state["best_output_tok_s"] = max(best, throughput)
    else:
        state["plateau_steps"] = 0
    if not reason and (len(completed) != config["repeats"] or len(valid) != len(completed)):
        reason = (
            "incomplete_time_budget"
            if any(t["status"] == "incomplete_time_budget" for t in trials)
            else "point_failure"
        )
    state.setdefault("history", []).append(
        {
            "concurrency": config["concurrency"][0],
            "median_output_tok_s": throughput,
            "completed_repeats": len(completed),
            "actual_requests_by_repeat": [t.get("requested_requests") for t in completed],
            "reason": reason,
        }
    )
    if reason:
        state.update(stopped=True, reason=reason, last_concurrency=config["concurrency"][0])
        for trial in trials:
            trial["stopping_reason"] = reason
    return reason


def text_sample_requests(config, fitted, minimum):
    """Select a count without changing concurrency or exceeding planned requests."""
    if config.get("measurement_sample_policy") == "full_or_smoke" and fitted < config["requests"]:
        return min(config["requests"], max(minimum, 2))
    return min(config["requests"], max(minimum, fitted))


def text_budget_requests(config, prediction, pilot_count, available, minimum):
    """Apply the explicit sampling policy to the predicted per-repeat capacity."""
    repeats = config["repeats"]
    usable = max(0.0, available - prediction["fixed_overhead_seconds_per_repeat"] * repeats)
    per_request = prediction["benchmark_seconds"] / pilot_count * prediction["sampling_safety_factor"]
    fitted = math.floor(usable / (per_request * repeats))
    return text_sample_requests(config, fitted, minimum)


def fixed_text_workload(config, target, models):
    key = (config["model"], config["model_revision"])
    model = models.setdefault(key, None)
    if model is None:
        model = models[key] = download(config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    command_args = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        "benchmark",
        "--host",
        "127.0.0.1",
        "--dtype",
        config["dtype"],
        "--max-model-len",
        str(config["max_model_len"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--kv-cache-dtype",
        config.get("kv_cache_dtype", "auto"),
        "--max-num-seqs",
        str(config["max_num_seqs"]),
        "--tensor-parallel-size",
        str(target["gpu_count"]),
        "--seed",
        str(config["seed"]),
        "--generation-config",
        "vllm",
    ]
    if fixed_engine(config):
        command_args.extend(
            [
                "--max-num-batched-tokens",
                str(config["max_num_batched_tokens"]),
                "--enable-chunked-prefill"
                if config["enable_chunked_prefill"]
                else "--no-enable-chunked-prefill",
            ]
        )
    if not config["prefix_caching"]:
        command_args.append("--no-enable-prefix-caching")
    if config["quantization"]:
        command_args.extend(["--quantization", config["quantization"]])
    label = f"config-{config['config_id']}"
    if config.get("_point_label"):
        label += f"-point{config['_point_label']}"
    if config.get("saturation_max_concurrency") is not None:
        label += f"-limit{config['max_num_seqs']}"
    if fixed_engine(config) and config.get("load_mode") != "open_loop_request_rate":
        label += f"-client{config['concurrency'][0]}"
    server = None
    session = config.get("_engine_session") if fixed_engine(config) else None
    successful = False
    engine_reused = False
    engine_instance_id = None
    first_trial = len(REPORT["trials"])
    mixed = config.get("load_mode") == "open_loop_request_rate"
    suffix = "json" if mixed else "jsonl"
    levels = config["request_rates"] if mixed else config["concurrency"]
    budget = config.get("measurement_budget_seconds")
    shrink = config.get("measurement_sample_policy") in ("shrink_to_budget", "full_or_smoke")
    planned_requests = config["requests"]
    minimum_requests = 1 if mixed else max(config["concurrency"])
    pilot_timed_out = False
    boundary_reason = None
    prediction = None
    prediction_count = None
    deadline = None
    pilot_count = config["warmup_requests"]
    pilot_concurrency = 1
    pilot_seconds = None
    if budget and not mixed:
        pilot_concurrency = config["concurrency"][0]
        # One occupancy wave, not two completed waves/full benchmark repeats.
        pilot_count = min(config["requests"], max(pilot_count, pilot_concurrency))

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TextTimeBudget("Measurement budget exhausted")
        return seconds

    try:
        begun = time.monotonic()
        if session is not None:
            server, engine_reused, engine_instance_id = session.acquire(
                command_args, f"{label}-server.log", config["server_timeout_seconds"]
            )
        else:
            server = start(command_args, f"{label}-server.log")
            ready(server, 8000, "/v1/models", config["server_timeout_seconds"])
        if engine_reused and config["prefix_caching"]:
            reset_text_prefix_cache()
        phase(
            "server_reuse" if engine_reused else "server_load_compile",
            begun,
            config_id=config["config_id"],
            command=command_args,
            engine_instance_id=engine_instance_id,
            server_log=session.log if session is not None else f"{label}-server.log",
        )
        begun = time.monotonic()
        if budget:
            deadline = begun + budget
        warmup_data = corpus(
            config,
            tokenizer,
            pilot_count,
            config["seed"] + 100000,
            f"{label}-warmup-dataset.{suffix}",
        )
        try:
            command(
                bench_command(
                    config,
                    model,
                    pilot_concurrency,
                    f"{label}-warmup.json",
                    pilot_count,
                    config["seed"] + 100000,
                    dataset_path=warmup_data,
                ),
                f"{label}-warmup.log",
                min(config["trial_timeout_seconds"], budget * 0.4, remaining())
                if budget
                else config["trial_timeout_seconds"],
                progress=ThroughputWatch(config)
                if config.get("saturation_min_output_tok_s", 0) and not config.get("coverage_first")
                else None,
            )
        except subprocess.TimeoutExpired:
            if not shrink:
                raise
            # A timed-out pilot is not an admission veto: attempt the minimum
            # real measurement with the remaining budget, without guessing a rate.
            pilot_timed_out = True
            fallback = text_sample_requests(config, 0, minimum_requests)
            config = {**config, "requests": fallback}
            REPORT.setdefault("throughput_probes", []).append(
                {
                    "config_id": config["config_id"],
                    "phase": "time_budget_pilot",
                    "concurrency": pilot_concurrency,
                    "decision": "pilot_timeout_minimum_sample",
                    "planned_requests": planned_requests,
                    "selected_requests": fallback,
                    "measurement_sample_policy": config.get("measurement_sample_policy", "fixed"),
                    "budget_seconds": budget,
                }
            )
            checkpoint()
        pilot_seconds = time.monotonic() - begun
        phase("warmup", begun, config_id=config["config_id"])
        if budget and not pilot_timed_out and (not mixed or shrink):
            raw = json.loads((ROOT / f"{label}-warmup.json").read_text())
            metrics, requests = normalize_text(raw, pilot_count)
            if metrics["failed_requests"] or not lengths_match(config, tokenizer, warmup_data, requests):
                raise RuntimeError("Time-budget pilot failed sample/length validation")
            prediction = text_measurement_estimate(config, raw, pilot_count, pilot_seconds)
            prediction_count = pilot_count
            estimate = prediction["estimated_measurement_seconds"]
            available = remaining()
            if shrink:
                selected = text_budget_requests(config, prediction, pilot_count, available, minimum_requests)
                if mixed:
                    # Include the offered arrival schedule for every repeat, not
                    # just the pilot's unconstrained serving throughput.
                    rate = min(config["request_rates"])
                    selected = min(
                        selected,
                        max(
                            1,
                            math.floor(
                                max(
                                    0.0,
                                    available / config["repeats"]
                                    - prediction["fixed_overhead_seconds_per_repeat"],
                                )
                                * rate
                                / prediction["sampling_safety_factor"]
                            ),
                        ),
                    )
                selected = text_sample_requests(config, selected, minimum_requests)
                config = {**config, "requests": selected}
            probe = {
                "config_id": config["config_id"],
                "phase": "time_budget_pilot",
                **prediction,
                "concurrency": pilot_concurrency,
                "requests": pilot_count,
                "seconds": pilot_seconds,
                "estimated_measurement_seconds": estimate,
                "budget_seconds": budget,
                "remaining_seconds": available,
                "decision": (
                    "smoke_sample"
                    if config.get("measurement_sample_policy") == "full_or_smoke"
                    and config["requests"] < planned_requests
                    else "shrink_sample"
                    if config["requests"] < planned_requests
                    else "proceed"
                )
                if shrink
                else ("skip_time_budget" if estimate > available else "proceed"),
                "planned_requests": planned_requests,
                "selected_requests": config["requests"],
                "measurement_sample_policy": config.get("measurement_sample_policy", "fixed"),
            }
            REPORT.setdefault("throughput_probes", []).append(probe)
            checkpoint()
            if estimate > available and not shrink:
                raise TextTimeBudget(
                    f"Configured samples/repeats estimated at {estimate:.1f}s exceed remaining budget"
                )
        if config.get("cliff_aware") and not pilot_timed_out:
            raw = json.loads((ROOT / f"{label}-warmup.json").read_text())
            metrics, requests = normalize_text(raw, pilot_count)
            if metrics["failed_requests"] or not lengths_match(config, tokenizer, warmup_data, requests):
                raise RuntimeError("Boundary pilot failed request-count or length validation")
            boundary_reason = text_boundary_reason(
                config, metrics["output_throughput"], config.get("_boundary_state", {})
            )
            if boundary_reason:
                config = {**config, "requests": min(planned_requests, max(2, minimum_requests))}
            REPORT.setdefault("throughput_probes", []).append(
                {
                    "config_id": config["config_id"],
                    "phase": "boundary_pilot",
                    "concurrency": pilot_concurrency,
                    "output_tok_s": metrics["output_throughput"],
                    "decision": "boundary_smoke" if boundary_reason else "proceed",
                    "candidate_stopping_reason": boundary_reason,
                    "planned_requests": planned_requests,
                    "selected_requests": config["requests"],
                }
            )
            checkpoint()
        if (
            not pilot_timed_out
            and not config.get("coverage_first")
            and config.get("saturation_min_output_tok_s", 0)
            and (mixed or config["concurrency"] == [1])
        ):
            raw = json.loads((ROOT / f"{label}-warmup.json").read_text())
            metrics, requests = normalize_text(raw, pilot_count)
            if metrics["failed_requests"] or not lengths_match(config, tokenizer, warmup_data, requests):
                raise RuntimeError("Throughput probe failed request-count or forced-length validation")
            probe = {
                "config_id": config["config_id"],
                "phase": "warmup",
                "output_tok_s": metrics["output_throughput"],
                "requests": pilot_count,
            }
            REPORT.setdefault("throughput_probes", []).append(probe)
            try:
                throughput_cutoff(config, metrics["output_throughput"])
            except LowThroughputStop as error:
                probe["decision"] = "low_throughput_cutoff"
                REPORT["trials"].append(
                    {
                        "config_id": config["config_id"],
                        "concurrency": 0 if mixed else 1,
                        "repeat": 0,
                        "status": "skipped_low_throughput",
                        "error": str(error),
                        "engine_max_num_seqs": config["max_num_seqs"],
                        "load_mode": config.get("load_mode"),
                        "probe_output_tok_s": metrics["output_throughput"],
                    }
                )
                checkpoint()
                raise
            probe["decision"] = "proceed"
            checkpoint()
        for repeat in range(config["repeats"]):
            if shrink and prediction is not None and not boundary_reason:
                # Refit subsequent repeats using observed measurement speed and
                # the remaining deadline, not just the original tiny pilot.
                fitting = {**config, "repeats": config["repeats"] - repeat}
                count = text_budget_requests(
                    fitting, prediction, prediction_count, remaining(), minimum_requests
                )
                config = {**config, "requests": count}
            dataset = corpus(
                config,
                tokenizer,
                config["requests"],
                config["seed"] + repeat,
                f"{label}-r{repeat}-dataset.{suffix}",
            )
            for level in levels:
                if budget:
                    remaining()
                    if mixed and not shrink and config["requests"] / level > remaining():
                        raise TextTimeBudget(
                            "Offered arrival schedule alone exceeds remaining measurement budget"
                        )
                concurrency = 0 if mixed else level
                smoke_sample = (
                    bool(boundary_reason)
                    or (
                        config.get("measurement_sample_policy") == "full_or_smoke"
                        and config["requests"] < planned_requests
                    )
                    or (fixed_engine(config) and config["requests"] <= max(2, minimum_requests))
                )
                trial = {
                    "config_id": config["config_id"],
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "engine_max_num_seqs": config["max_num_seqs"],
                    "engine_config": executed_text_engine(config, target),
                    "engine_instance_id": engine_instance_id,
                    "engine_reused": engine_reused,
                    "requested_requests": config["requests"],
                    "planned_requests": planned_requests,
                    "candidate_stopping_reason": boundary_reason,
                    "reduced_sample": config["requests"] < planned_requests,
                    "measurement_sample_policy": config.get("measurement_sample_policy", "fixed"),
                    "sample_class": "smoke"
                    if smoke_sample
                    else "reduced"
                    if config["requests"] < planned_requests
                    else "full",
                    "sample_size_warning": (
                        "Smoke sample; capability/coarse timing only; tail latency not established"
                        if smoke_sample
                        else "Reduced sample; do not assume original tail-latency precision"
                    )
                    if smoke_sample or config["requests"] < planned_requests
                    else None,
                    "load_mode": config.get("load_mode", "closed_loop_concurrency"),
                    "request_rate": level if mixed else None,
                    "burstiness": config.get("burstiness") if mixed else None,
                    "client_concurrency_limit": None if mixed else concurrency,
                    "prompt_mode": config.get("prompt_mode", "raw_completion"),
                    "enable_thinking": config.get("enable_thinking"),
                    "status": "failed",
                }
                tag = f"{label}-r{repeat}-" + (f"qps{level}" if mixed else f"c{concurrency}")
                try:
                    trial_started = time.monotonic()
                    with Telemetry(tag):
                        command(
                            bench_command(
                                config,
                                model,
                                config["max_num_seqs"] if mixed else concurrency,
                                f"{tag}.json",
                                config["requests"],
                                config["seed"] + repeat,
                                dataset_path=dataset,
                                request_rate=level if mixed else None,
                            ),
                            f"{tag}.log",
                            min(config["trial_timeout_seconds"], remaining())
                            if budget
                            else config["trial_timeout_seconds"],
                            progress=ThroughputWatch(config, config.get("_prior_best_output_tok_s", 0))
                            if not mixed
                            and not config.get("coverage_first")
                            and config.get("saturation_min_output_tok_s", 0)
                            else None,
                        )
                    raw = json.loads((ROOT / f"{tag}.json").read_text())
                    metrics, requests = normalize_text(raw, config["requests"])
                    metrics["engine_max_num_seqs"] = config["max_num_seqs"]
                    if mixed:
                        metrics["offered_request_rate"] = level
                        metrics["burstiness"] = config["burstiness"]
                    lengths_ok = lengths_match(config, tokenizer, dataset, requests)
                    if shrink and metrics["failed_requests"] == 0 and lengths_ok:
                        prediction = text_measurement_estimate(
                            config, raw, config["requests"], time.monotonic() - trial_started
                        )
                        prediction_count = config["requests"]
                    trial.update(
                        metrics=metrics,
                        requests=requests,
                        lengths_match=lengths_ok,
                        status="completed" if metrics["failed_requests"] == 0 and lengths_ok else "failed",
                    )
                except subprocess.TimeoutExpired as error:
                    if budget:
                        raise TextTimeBudget(
                            "Measurement command exceeded its time budget; incomplete samples discarded"
                        ) from error
                    trial["error"] = str(error)
                except TextTimeBudget:
                    raise
                except LowThroughputStop as error:
                    trial.update(status="aborted_low_throughput", error=str(error))
                    REPORT["trials"].append(trial)
                    checkpoint()
                    raise
                except Exception as error:
                    trial["error"] = str(error)
                raw_path = ROOT / f"{tag}.json"
                if raw_path.exists():
                    packed = raw_path.with_suffix(".json.gz")
                    with raw_path.open("rb") as source, gzip.open(packed, "wb") as output:
                        shutil.copyfileobj(source, output)
                    trial["raw_artifact"] = packed.name
                    raw_path.unlink()
                REPORT["trials"].append(trial)
                checkpoint()
                if not mixed and not config.get("coverage_first") and trial["status"] == "completed":
                    throughput_cutoff(
                        config,
                        trial["metrics"]["output_throughput"],
                        config.get("_prior_best_output_tok_s", 0),
                    )
                if server.poll() is not None:
                    raise RuntimeError("vLLM server died during trials")
                if (mixed or config.get("saturation_max_concurrency") is not None) and trial[
                    "status"
                ] != "completed":
                    return
        # Every benchmark command has returned and all its requests completed:
        # the queue is drained before the next point may acquire this session.
        successful = bool(REPORT["trials"][first_trial:]) and all(
            t["status"] == "completed" for t in REPORT["trials"][first_trial:]
        )
    except (TextTimeBudget, subprocess.TimeoutExpired) as error:
        if not budget:
            raise
        REPORT["trials"].append(
            {
                "config_id": config["config_id"],
                "concurrency": 0 if mixed else config["concurrency"][0],
                "request_rate": config["request_rates"][0] if mixed else None,
                "repeat": -1,
                "status": "incomplete_time_budget" if shrink else "skipped_time_budget",
                "error": str(error),
                "planned_requests": planned_requests,
                "reduced_sample": config["requests"] < planned_requests,
                "measurement_sample_policy": config.get("measurement_sample_policy", "fixed"),
                "requested_requests": config["requests"],
                "requested_repeats": config["repeats"],
                "budget_seconds": budget,
                "pilot_requests": pilot_count,
                "pilot_seconds": pilot_seconds,
                "engine_max_num_seqs": config["max_num_seqs"],
                "engine_config": executed_text_engine(config, target),
                "engine_instance_id": engine_instance_id,
                "engine_reused": engine_reused,
            }
        )
        checkpoint()
        raise TextTimeBudget(str(error)) from error
    finally:
        if session is not None:
            if not successful:
                # A timeout/failed client can leave GPU requests in flight.
                # Discard that engine rather than contaminate the next point.
                session.close()
        else:
            stop(server)
            time.sleep(5)


def image_graph(config, seed, prompt):
    graph = copy.deepcopy(config["workflow"])
    graph["3"]["inputs"].update(
        seed=seed,
        steps=config["steps"],
        cfg=config["cfg"],
        sampler_name=config["sampler"],
        scheduler=config["scheduler"],
    )
    graph["4"]["inputs"]["ckpt_name"] = config["checkpoint"]
    graph["5"]["inputs"].update(
        width=config["width"], height=config["height"], batch_size=config["batch_size"]
    )
    graph["6"]["inputs"]["text"] = prompt
    return graph


def direct_image_workload(config):
    # Bound the entire row, including setup, stop grace and final artifact copying.
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    budget = config["row_budget_seconds"]

    def row_expired(_signum, _frame):
        raise ImageRowDeadline("Diffusers absolute row deadline reached")

    row_first = not previous_timer[0] or budget < previous_timer[0]
    if row_first:
        signal.signal(signal.SIGALRM, row_expired)
        signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        return _direct_image_workload(config)
    except ImageRowDeadline:
        # Completed points were synchronized periodically; do not do unbounded I/O here.
        REPORT["trials"].append(
            {
                "config_id": config["config_id"],
                "concurrency": 1,
                "repeat": -1,
                "status": "budget_exhausted",
                "backend": "diffusers",
                "error": "Absolute row deadline reached; active point may be incomplete",
            }
        )
    finally:
        if row_first:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0]:
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(0.001, previous_timer[0] - (time.monotonic() - started)),
                    previous_timer[1],
                )


def preinstalled_diffusers(requirements, manifest_path=Path("/opt/sls-bench/diffusers-runtime.json")):
    """Trust neither a stale marker nor a tag: check requested pins and actual versions."""
    try:
        manifest = json.loads(manifest_path.read_text())
        expected_hash = hashlib.sha256(("\n".join(requirements) + "\n").encode()).hexdigest()
        if not isinstance(manifest, dict) or manifest.get("requirements_sha256") != expected_hash:
            return False
        if not isinstance(manifest.get("versions"), dict):
            return False
        for requirement in requirements:
            name, expected = requirement.split("==")
            actual = importlib.metadata.version(name)
            if actual.split("+")[0] != expected or manifest["versions"].get(name) != actual:
                return False
        return True
    except (OSError, ValueError, KeyError, importlib.metadata.PackageNotFoundError):
        return False


def _direct_image_workload(config):
    try:
        from . import image_diffusers
    except ImportError:
        import image_diffusers

    begun = time.monotonic()
    # command() may spend 20s terminating a child. Keep that grace inside the row ceiling.
    cleanup_reserve = 20
    label = f"diffusers-{config['config_id']}"
    directory = ROOT / label
    directory.mkdir(exist_ok=True)
    requirements = ROOT / "diffusers-requirements.txt"
    requirements.write_text("\n".join(image_diffusers.REQUIREMENTS) + "\n")
    marker = ROOT / "diffusers-environment.sha256"
    required_hash = hashlib.sha256(requirements.read_bytes()).hexdigest()
    prebaked = preinstalled_diffusers(image_diffusers.REQUIREMENTS)
    if prebaked:
        marker.write_text(required_hash)
    elif Path("/opt/sls-bench/diffusers-runtime.json").exists():
        # A baked manifest with drift must not be masked by a prior row marker.
        marker.unlink(missing_ok=True)
    REPORT.setdefault("image_environments", {})[str(config["config_id"])] = {
        "preinstalled_verified": prebaked,
        "requirements_sha256": required_hash,
    }
    if not marker.exists() or marker.read_text() != required_hash:
        command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
            "diffusers-install.log",
            min(600, max(1, config["row_budget_seconds"] - cleanup_reserve)),
        )
        marker.write_text(required_hash)
    remaining = int(config["row_budget_seconds"] - (time.monotonic() - begun) - cleanup_reserve)
    if remaining < 30:
        raise TimeoutError("Diffusers row setup exhausted the sub-hour row budget")
    config_path = ROOT / f"{label}-config.json"
    config_path.write_text(json.dumps({**config, "row_budget_seconds": remaining}))
    before = len(REPORT["trials"])

    def synchronize():
        path = directory / "progress.json"
        if not path.exists():
            return
        value = json.loads(path.read_text())
        REPORT["trials"] = REPORT["trials"][:before] + value.get("trials", [])
        REPORT.setdefault("image_runs", {})[str(config["config_id"])] = {
            k: v for k, v in value.items() if k != "trials"
        }
        checkpoint()

    timed_out = False
    try:
        command(
            [sys.executable, str(ROOT / "image_diffusers.py"), str(config_path), str(directory)],
            f"{label}.log",
            remaining,
            progress=synchronize,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        if isinstance(sys.exc_info()[1], ImageRowDeadline):
            # The alarm has fired: skip all final synchronization/copies. The wrapper
            # records budget_exhausted using already-checkpointed completed points.
            raise
        synchronize()
        if timed_out:
            REPORT["trials"].append(
                {
                    "config_id": config["config_id"],
                    "concurrency": 1,
                    "repeat": -1,
                    "status": "budget_exhausted",
                    "error": "Hard sub-hour row deadline reached",
                    "backend": "diffusers",
                }
            )
            checkpoint()
        for path in directory.glob("*.png"):
            shutil.copyfile(path, ROOT / f"{label}-{path.name}")
        path = directory / "progress.json"
        if path.exists():
            shutil.copyfile(path, ROOT / f"{label}-details.json")


def image_workload(config):
    if config.get("backend") == "diffusers":
        return direct_image_workload(config)
    from huggingface_hub import hf_hub_download

    comfy = ROOT / "ComfyUI"
    begun = time.monotonic()
    if not comfy.exists():
        command(
            ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(comfy)],
            "comfy-install.log",
            300,
        )
    command(
        ["git", "-C", str(comfy), "checkout", "--detach", config["comfy_revision"]], "comfy-install.log", 60
    )
    # Freeze installed versions into the report. For strict repeatability use a prebuilt image digest.
    command(
        [sys.executable, "-m", "pip", "install", "-r", str(comfy / "requirements.txt")],
        "comfy-install.log",
        600,
    )
    checkpoint_path = hf_hub_download(
        config["model"], config["checkpoint"], revision=config["model_revision"]
    )
    dest = (
        comfy
        / "models"
        / "checkpoints"
        / f"benchmark-{config['config_id']}-{Path(config['checkpoint']).name}"
    )
    if dest.is_symlink():
        dest.unlink()
    elif dest.exists():
        raise FileExistsError(f"Refusing to replace existing checkpoint {dest}")
    dest.symlink_to(checkpoint_path)
    sha = hashlib.sha256()
    with open(checkpoint_path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha.update(block)
    config = {**config, "checkpoint": dest.name}
    phase(
        "comfy_install_model_download",
        begun,
        packages=capture([sys.executable, "-m", "pip", "freeze"]),
        checkpoint_sha256=sha.hexdigest(),
        checkpoint_filename=dest.name,
    )
    server = None
    try:
        begun = time.monotonic()
        server = start(
            [
                sys.executable,
                str(comfy / "main.py"),
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
                "--disable-auto-launch",
                "--force-fp16",
                "--gpu-only" if config.get("gpu_only", True) else "--highvram",
                "--fp32-vae" if config.get("vae_dtype", "fp32") == "fp32" else "--fp16-vae",
                "--use-pytorch-cross-attention",
                "--disable-xformers",
                "--disable-async-offload",
                "--disable-all-custom-nodes",
                "--cache-none",
            ],
            "comfy-server.log",
        )
        ready(server, 8188, "/system_stats", config["server_timeout_seconds"])
        phase("comfy_load", begun, config_id=config["config_id"])
        ordinal = 0
        for repeat in range(-1, config["repeats"]):
            requests = []
            begun = time.monotonic()
            count = config["warmup_requests"] if repeat == -1 else config["requests"]
            for i in range(count):
                seed = config["seed"] + ordinal
                ordinal += 1
                graph = image_graph(config, seed, config["prompts"][i % len(config["prompts"])])
                sent = time.monotonic()
                value = api("/prompt", {"prompt": graph}, port=8188)
                prompt_id = value["prompt_id"]
                deadline = sent + config["trial_timeout_seconds"]
                while True:
                    history = api(f"/history/{prompt_id}", port=8188).get(prompt_id)
                    if history:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Image generation timed out")
                    time.sleep(0.05)
                if history.get("status", {}).get("status_str") != "success":
                    raise RuntimeError(f"ComfyUI execution failed: {history.get('status')}")
                images = [im for out in history.get("outputs", {}).values() for im in out.get("images", [])]
                if len(images) != config["batch_size"]:
                    raise RuntimeError(f"Expected {config['batch_size']} images, got {len(images)}")
                elapsed = time.monotonic() - sent
                # ComfyUI's execution events carry epoch-ms timestamps; cached/no-op runs are invalid.
                messages = history.get("status", {}).get("messages", [])
                requests.append(
                    {
                        "seed": seed,
                        "e2e_seconds": elapsed,
                        "images": len(images),
                        "events": messages,
                        "workflow_sha256": hashlib.sha256(
                            json.dumps(graph, sort_keys=True).encode()
                        ).hexdigest(),
                    }
                )
                if repeat >= 0 and i == 0:
                    for j, image in enumerate(images):
                        query = urllib.parse.urlencode(image)
                        with urllib.request.urlopen(f"http://127.0.0.1:8188/view?{query}") as data:
                            (ROOT / f"image-{config['config_id']}-{repeat}-{j}.png").write_bytes(data.read())
            if repeat >= 0:
                duration = time.monotonic() - begun
                latencies = sorted(r["e2e_seconds"] for r in requests)

                def percentile(q):
                    return latencies[min(len(latencies) - 1, math.ceil(q * len(latencies)) - 1)]

                REPORT["trials"].append(
                    {
                        "config_id": config["config_id"],
                        "concurrency": 1,
                        "repeat": repeat,
                        "status": "completed",
                        "requests": requests,
                        "metrics": {
                            "duration": duration,
                            "completed": count,
                            "images": count * config["batch_size"],
                            "images_per_second": count * config["batch_size"] / duration,
                            "p50_e2e_seconds": percentile(0.5),
                            "p95_e2e_seconds": percentile(0.95),
                            "p99_e2e_seconds": percentile(0.99),
                        },
                    }
                )
                checkpoint()
            else:
                phase("image_warmup", begun, config_id=config["config_id"])
    finally:
        stop(server)


def coverage_schedule(workloads):
    """Round-robin points across profiles so long searches cannot starve later rows."""
    queues = []
    for config in workloads:
        if config.get(
            "benchmark_type", "text_gen" if config.get("kind") == "text" else None
        ) != "text_gen" or not config.get("coverage_first"):
            queues.append([config])
            continue
        if config.get("load_mode") == "open_loop_request_rate":
            queues.append([{**config, "request_rates": [rate]} for rate in config["request_rates"]])
            continue
        cap = config.get("saturation_max_concurrency")
        levels = config["concurrency"]
        if cap is not None:
            levels = []
            level = config["concurrency"][0]
            while level <= cap:
                levels.append(level)
                level *= 2
        schedule = config.get("requests_by_concurrency", {})
        queues.append(
            [
                {
                    **config,
                    "concurrency": [level],
                    "max_num_seqs": config["max_num_seqs"] if fixed_engine(config) else level,
                    "saturation_max_concurrency": level if cap is not None else None,
                    "requests": schedule.get(str(level), schedule.get(level, config["requests"])),
                    "requests_by_concurrency": {},
                }
                for level in levels
            ]
        )
    points = [
        queue[depth]
        for depth in range(max((len(q) for q in queues), default=0))
        for queue in queues
        if depth < len(queue)
    ]
    # Keep breadth-first ordering within compatible engines, but finish each
    # engine group before switching model/precision/scheduler settings.
    ordered, seen = [], set()
    for point in points:
        if point.get("kind") != "text" or not fixed_engine(point):
            ordered.append(point)
            continue
        key = text_engine_key(point)
        if key not in seen:
            seen.add(key)
            ordered.extend(
                p for p in points if p.get("kind") == "text" and fixed_engine(p) and text_engine_key(p) == key
            )
    return ordered


def main():
    with TextEngineSession() as engine_session:
        return run_worker(engine_session)


def run_worker(engine_session):
    spec = json.loads((ROOT / "spec.json").read_text())

    def interrupted(signum, _frame):
        raise WorkerCancelled(f"Worker received signal {signum}; cancelling its engine processes")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    remaining = int(
        spec.get("lease_expires_at", time.time() + spec.get("max_minutes", 45) * 60) - time.time()
    )
    signal.alarm(max(1, remaining))
    REPORT["status"] = "running"
    REPORT["source_hashes"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("worker.py", "image_diffusers.py")
        if (ROOT / name).exists()
    }
    REPORT["hardware"] = hardware()
    checkpoint()
    verify_hardware(spec["target"], REPORT["hardware"])
    models = {}
    adapters = {"text_gen": lambda c: text_workload(c, spec["target"], models), "image_gen": image_workload}
    scheduled = coverage_schedule(spec["workloads"])
    coverage = any(c.get("coverage_first") for c in spec["workloads"])
    if coverage:
        REPORT["coverage_plan"] = [
            {
                "config_id": c["config_id"],
                "concurrency": c["concurrency"][0] if c.get("load_mode") != "open_loop_request_rate" else 0,
                "request_rate": c["request_rates"][0]
                if c.get("load_mode") == "open_loop_request_rate"
                else None,
                "status": "pending",
            }
            for c in scheduled
        ]
        checkpoint()
    engine_plan = spec.get("text_engine_plan", text_engine_plan(spec["workloads"]))
    REPORT["text_engine_plan"] = engine_plan
    engines = {cid: group["engine_config"] for group in engine_plan for cid in group["config_ids"]}
    boundaries = {}
    for point_index, config in enumerate(scheduled):
        if config.get("kind") == "text" and fixed_engine(config):
            config = {
                **config,
                **engines[config["config_id"]],
                "_engine_session": engine_session,
                "_profile_max_model_len": config["max_model_len"],
            }
        else:
            engine_session.close()
        if config.get("cliff_aware"):
            state = boundaries.setdefault(config["config_id"], {"config_id": config["config_id"]})
            if state.get("stopped"):
                REPORT["coverage_plan"][point_index].update(
                    status="truncated",
                    stopping_reason=state["reason"],
                    stopped_at_concurrency=state["last_concurrency"],
                )
                checkpoint()
                continue
            config = {**config, "_boundary_state": state}
        before = len(REPORT["trials"])
        if coverage:
            REPORT["coverage_plan"][point_index]["status"] = "running"
            checkpoint()
        try:
            benchmark_type = config.get("benchmark_type") or {"text": "text_gen", "image": "image_gen"}.get(
                config["kind"]
            )
            if benchmark_type not in adapters:
                raise ValueError(f"No worker adapter implemented for benchmark type {benchmark_type}")
            adapters[benchmark_type](config)
        except Exception as error:
            evidence = str(error)
            server_log = ROOT / (
                engine_session.log
                if config.get("kind") == "text" and fixed_engine(config) and engine_session.log
                else f"config-{config['config_id']}-server.log"
            )
            if server_log.exists():
                with server_log.open("rb") as log:
                    log.seek(max(0, server_log.stat().st_size - 16000))
                    evidence += log.read().decode(errors="replace")
            REPORT["trials"].append(
                {
                    "config_id": config["config_id"],
                    "concurrency": 0,
                    "repeat": -1,
                    "status": "setup_failed",
                    "failure_kind": failure_kind(evidence),
                    "engine_config": executed_text_engine(config, spec["target"]),
                    "engine_instance_id": engine_session.instance_id if fixed_engine(config) else None,
                    "error": str(error),
                }
            )
            (ROOT / f"config-{config['config_id']}-error.log").write_text(traceback.format_exc())
            checkpoint()
        if config.get("cliff_aware"):
            reason = record_text_boundary(config, REPORT["trials"][before:], config["_boundary_state"])
            REPORT["text_boundaries"] = list(boundaries.values())
            if reason:
                REPORT["coverage_plan"][point_index]["stopping_reason"] = reason
        if coverage:
            trials = REPORT["trials"][before:]
            REPORT["coverage_plan"][point_index]["status"] = (
                "completed"
                if len(trials) == config["repeats"] and all(t["status"] == "completed" for t in trials)
                else "incomplete"
            )
            checkpoint()
    REPORT["status"] = "completed" if all(t["status"] == "completed" for t in REPORT["trials"]) else "partial"
    checkpoint()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        REPORT.update(status="failed", error=str(error))
        (ROOT / "worker-error.log").write_text(traceback.format_exc())
        checkpoint()
        raise
