#!/usr/bin/env python3
"""One-way, allowlisted export. Never imports or executes the live benchmark runner."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_FIELDS = set(
    """kind name model model_revision backend benchmark_type dtype precision
input_tokens output_tokens concurrency repeats requests warmup_requests seed load_mode
request_rates burstiness dataset_repo dataset_revision dataset_filename dataset_generator
max_model_len max_num_seqs max_num_batched_tokens gpu_memory_utilization kv_cache_dtype
quantization enable_chunked_prefill prefix_caching enable_thinking engine_capacity_policy
measurement_budget_seconds measurement_sample_policy minimum_requests coverage_first cliff_aware
saturation_max_concurrency saturation_min_gain saturation_min_output_tok_s saturation_plateau_steps
saturation_collapse_ratio requests_by_concurrency server_timeout_seconds trial_timeout_seconds
batch_size batch_cap width height steps cfg guidance_mode guidance_scale true_cfg_scale
sampler scheduler pipeline checkpoint comfy_revision execution_modes offload_policy gpu_only
attention_backend vae_dtype use_kv_cache requested_denoiser_steps point_budget_seconds
row_budget_seconds extended_curve_budget prompt_mode gated_model min_compute_capability
min_vram_gb ttft_slo_ms tpot_slo_ms reference_image_sha256 tier readiness""".split()
)
RESULT_FIELDS = set(
    """backend batch_size concurrency repeat status failure_kind failure_stage
execution_mode precision load_mode request_rate burstiness client_concurrency_limit
engine_max_num_seqs engine_instance_id engine_reused enable_thinking lengths_match
measurement_sample_policy planned_requests requested_requests requested_repeats reduced_sample
sample_class sample_reduction_reason sample_size_warning stopping_reason candidate_stopping_reason
pilot_requests pilot_seconds affordability_basis affordability_pilot_seconds budget_seconds
probe_output_tok_s completed_calls_before_failure prompt_mode tier warmup_calls""".split()
)
TARGET_FIELDS = set(
    "gpu_type gpu_count expected_memory_gb require_mig image min_cuda_version min_ram_gb min_vcpu cloud".split()
)
ENGINE_FIELDS = set(
    "model model_revision dtype quantization kv_cache_dtype max_model_len max_num_seqs max_num_batched_tokens gpu_memory_utilization enable_chunked_prefill prefix_caching seed tensor_parallel_size".split()
)
METRIC_FIELDS = (
    set(
        """burstiness completed duration engine_max_num_seqs expected_requests failed_requests
images images_per_second max_concurrency max_concurrent_requests max_output_tokens_per_s
median_denoise_total_seconds median_denoiser_step_ms median_image_encoding_seconds
median_postprocessing_seconds median_text_encode_seconds median_vae_decode_seconds num_prompts
offered_request_rate output_throughput peak_memory_allocated_bytes peak_memory_reserved_bytes
request_rate request_throughput requested_calls total_input_tokens total_output_tokens
total_token_throughput estimated_dollars_per_million_output_tokens
estimated_dollars_per_thousand_images""".split()
    )
    | {
        f"{stat}_{metric}_ms"
        for stat in ("mean", "median", "p50", "p95", "p99", "std")
        for metric in ("e2el", "itl", "tpot", "ttft")
    }
    | {f"{stat}_e2e_seconds" for stat in ("p50", "p95", "p99")}
)


def pick(value, fields):
    return {k: v for k, v in value.items() if k in fields}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def copy_source(source, destination):
    """Only these files can cross the boundary; no directories copied recursively."""
    bench = source / "benchmarks"
    paths = [
        bench / n
        for n in ("README.md", "WORKLOAD_COVERAGE.md", "TEXT_GRID_COMPARISON.md", "pyproject.toml", "uv.lock")
    ]
    paths += sorted((bench / "sls_bench").glob("*.py"))
    # Two operational tests import private campaign scripts from data/jobs.
    # Neither those scripts nor their tests belong to the standalone runner.
    private_tests = {"test_image_extension_budget.py", "test_image_fleet_controller.py"}
    paths += [p for p in sorted((bench / "tests").glob("*.py")) if p.name not in private_tests]
    paths += [
        bench / "containers/diffusers" / n
        for n in ("Dockerfile", "README.md", "bootstrap.sh", "prepare_context.py", "verify_runtime.py")
    ]
    manifest = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Not a regular allowlisted file: {path}")
        relative = path.relative_to(source)
        target = destination / relative
        if target.is_symlink():
            raise ValueError(f"Refusing destination symlink: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[str(relative)] = sha(target.read_bytes())
    return manifest


def export_snapshot(snapshot, output):
    """Never publish raw JSON blobs, request bodies, artifacts, logs, or resources."""
    conn = sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        configs = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM configs")}
        runs = []
        run_keys = {}
        workload_rows = {}
        for r in conn.execute("SELECT * FROM runs ORDER BY id"):
            spec = json.loads(r["spec"])
            key = sha(r["id"].encode())[:20]
            run_keys[r["id"]] = key
            workloads = []
            for w in spec.get("workloads", []):
                safe = pick(w, WORKLOAD_FIELDS)
                wkey = sha(canonical(safe).encode())
                workload_rows[wkey] = safe
                workloads.append({"workload_key": wkey})
                # Keep the frozen revision, not a newer default configuration.
                if "config_id" in w:
                    run_keys[(r["id"], w["config_id"])] = wkey
            runs.append(
                {
                    "run_key": key,
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "finished_at": r["finished_at"],
                    "hourly_price": r["hourly_price"],
                    "estimated_cost": r["estimated_cost"],
                    "code_sha256": r["code_sha256"],
                    "target": pick(spec.get("target", {}), TARGET_FIELDS),
                    "replica_count": spec.get("replica_count"),
                    "gpus_per_replica": spec.get("gpus_per_replica"),
                    "client_location": spec.get("client_location"),
                    "worker_sha256": spec.get("worker_sha256"),
                    "workloads": workloads,
                }
            )
        metrics = {}
        for r in conn.execute("SELECT * FROM metrics ORDER BY trial_id,name"):
            if r["name"] in METRIC_FIELDS and math.isfinite(r["value"]):
                metrics.setdefault(r["trial_id"], {})[r["name"]] = r["value"]
        trials = []
        for t in conn.execute("SELECT * FROM trials ORDER BY run_id,ordinal"):
            result = json.loads(t["result"])
            config = configs[t["workload_config_id"]]
            wkey = run_keys.get((t["run_id"], t["workload_config_id"]))
            if wkey is None:
                safe = pick(json.loads(config["spec"]), WORKLOAD_FIELDS)
                wkey = sha(canonical(safe).encode())
                workload_rows[wkey] = safe
            trials.append(
                {
                    "run_key": run_keys[t["run_id"]],
                    "ordinal": t["ordinal"],
                    "workload_key": wkey,
                    "concurrency": t["concurrency"],
                    "repeat": t["repeat"],
                    "status": t["status"],
                    "recorded": pick(result, RESULT_FIELDS),
                    "engine_config": pick(result.get("engine_config", {}), ENGINE_FIELDS),
                    "metrics": metrics.get(t["id"], {}),
                }
            )
        # Hardware is narrowly selected: raw nvidia-smi/topology may expose UUIDs.
        hardware = []
        for h in conn.execute("SELECT * FROM hardware ORDER BY run_id"):
            details = json.loads(h["details"])
            hardware.append(
                {
                    "run_key": run_keys[h["run_id"]],
                    **pick(details, {"cuda", "torch", "python"}),
                    "torch_devices": [
                        pick(d, {"name", "compute_capability", "memory_bytes"})
                        for d in details.get("torch_devices", [])
                        if isinstance(d, dict)
                    ],
                }
            )
        output.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("runs", runs),
            ("trials", trials),
            ("workloads", workload_rows),
            ("hardware", hardware),
        ):
            json_file(output / f"{name}.json", data)
        fields = ["run_key", "ordinal", "workload_key", "concurrency", "repeat", "status"] + sorted(
            METRIC_FIELDS
        )
        with (output / "trials.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for t in trials:
                writer.writerow({**{k: t[k] for k in fields[:6]}, **t["metrics"]})
        return {
            "runs": len(runs),
            "trials": len(trials),
            "trial_statuses": dict(sorted(Counter(t["status"] for t in trials).items())),
        }
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--database", type=Path, help="Defaults to source/benchmarks/data/benchmarks.sqlite3")
    args = parser.parse_args()
    source = args.source.expanduser().resolve(strict=True)
    database = (
        (args.database or source / "benchmarks/data/benchmarks.sqlite3").expanduser().resolve(strict=True)
    )
    if source == ROOT or ROOT.is_relative_to(source) or source.is_relative_to(ROOT):
        parser.error("Source and publication must be separate, non-nested directories")
    # Backup uses SQLite directly: do NOT instantiate Database, which runs migrations.
    # All raw state lives outside the publication repo and is deleted on exit.
    with tempfile.TemporaryDirectory(prefix="benchmark-publication-") as tmp:
        snapshot = Path(tmp) / "snapshot.sqlite3"
        src = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=30)
        dst = sqlite3.connect(snapshot)
        try:
            src.backup(dst, pages=256, sleep=0.05)
        finally:
            dst.close()
            src.close()
        files = copy_source(source, ROOT)
        counts = export_snapshot(snapshot, ROOT / "results")
    json_file(
        ROOT / "results/manifest.json",
        {
            "export_schema": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_files_sha256": files,
            "counts": counts,
            "exporter_sha256": sha(Path(__file__).read_bytes()),
            "result_files_sha256": {
                p.name: sha(p.read_bytes())
                for p in sorted((ROOT / "results").glob("*"))
                if p.suffix in {".json", ".csv"} and p.name != "manifest.json"
            },
            "publication_status": "sanitized historical export; see docs/PUBLICATION_REVIEW.md",
            "raw_database_published": False,
        },
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
