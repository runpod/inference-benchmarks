"""Benchmark categories are independent of models, workload profiles, and GPU SKUs."""

from dataclasses import dataclass

from .config import Image, ImageProcessing, Text
from .db import encode


@dataclass(frozen=True)
class BenchmarkType:
    key: str
    label: str
    config_model: type
    adapter: str | None
    default_suite: str | None


TYPES = {
    t.key: t
    for t in [
        BenchmarkType("text_gen", "Text generation", Text, "vllm", "text"),
        BenchmarkType("image_gen", "Image generation", Image, "comfyui+diffusers", "image"),
        BenchmarkType("image_processing", "Image processing", ImageProcessing, None, None),
    ]
}


def type_for(config):
    key = config.get("benchmark_type") or {"text": "text_gen", "image": "image_gen"}.get(config["kind"])
    if key not in TYPES:
        raise ValueError(f"Unknown benchmark type: {key}")
    return TYPES[key]


def sync_types(db):
    with db.conn:
        for value in TYPES.values():
            db.conn.execute(
                "INSERT INTO benchmark_types(type_id,label,adapter,default_suite,schema_json) VALUES(?,?,?,?,?) "
                "ON CONFLICT(type_id) DO UPDATE SET label=excluded.label,adapter=excluded.adapter,"
                "default_suite=excluded.default_suite,schema_json=excluded.schema_json",
                (
                    value.key,
                    value.label,
                    value.adapter,
                    value.default_suite,
                    encode(value.config_model.model_json_schema()),
                ),
            )
