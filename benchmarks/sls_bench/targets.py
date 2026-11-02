"""GPU selection comes from CLI arguments/live catalog, never a baked-in SKU list."""

import json
import re

from .config import Target
from .db import digest, encode


def materialize_target(db, gpu_type, catalog, **options):
    if gpu_type not in catalog:
        raise ValueError(
            f"Exact GPU SKU not in live Runpod catalog: {gpu_type}; use `bench gpus --match ...`"
        )
    identity = f"{gpu_type}|{options.get('cloud', 'SECURE')}|{options.get('gpu_count', 1)}"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", gpu_type).strip("-").lower()[:60]
    name = f"{slug}-{digest(identity.encode())[:8]}"
    config = Target(name=name, gpu_type=gpu_type, **options).model_dump(mode="json")
    memory = catalog[gpu_type].get("memoryInGb")
    config["expected_memory_gb"] = memory if memory and memory > 0 else None
    config["require_mig"] = "MIG" in gpu_type.upper()
    return db.save_config(config)


def hydrate_target(row, catalog):
    value = Target.model_validate(json.loads(row["spec"])).model_dump(mode="json")
    gpu = catalog.get(value["gpu_type"])
    if not gpu:
        raise ValueError(f"Exact GPU SKU not in live Runpod catalog: {value['gpu_type']}")
    if value["expected_memory_gb"] is None:
        memory = gpu.get("memoryInGb")
        if not memory or memory <= 0:
            raise ValueError("GPU catalog has no valid VRAM metadata; cannot verify this device safely")
        value["expected_memory_gb"] = memory
    if value["require_mig"] is None:
        value["require_mig"] = "MIG" in value["gpu_type"].upper()
    return {**row, "spec": encode(value)}
