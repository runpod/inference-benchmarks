"""One SQLite matrix per benchmark type; expansion is deterministic and never rents Pods."""

import json

from .config import validate
from .db import digest, encode
from .registry import type_for


def select(db, selector, kind=None):
    name, separator, revision = selector.partition("@")
    return db.config(name, kind, int(revision) if separator else None)


def expand(db, selector):
    row = select(db, selector, "matrix")
    matrix = validate(json.loads(row["spec"]))
    workloads = []
    seen = set()
    for model in matrix["rows"]:
        for profile in model["profiles"]:
            source = select(db, profile)
            base = validate(json.loads(source["spec"]))
            if type_for(base).key != matrix["benchmark_type"]:
                raise ValueError(f"Matrix {selector}: {profile} has the wrong benchmark type")
            identity = (model["name"], source["id"])
            if identity in seen:
                raise ValueError(f"Duplicate matrix profile: {model['name']} / {profile}")
            seen.add(identity)
            value = validate({**base, **matrix["parameters"], **model["parameters"]})
            suffix = digest(
                encode(
                    {"matrix_id": row["id"], "profile_id": source["id"], "row": model["name"], "value": value}
                ).encode()
            )[:8]
            value["name"] = f"{matrix['name'][:24]}-{model['name'][:24]}-{source['name'][:18]}-{suffix}"
            workloads.append(
                {
                    "row": model["name"],
                    "profile": profile,
                    "profile_id": source["id"],
                    "value": value,
                    "placement": {
                        k: model[k]
                        for k in (
                            "gpu_count",
                            "min_device_vram_gb",
                            "allowed_gpu_generations",
                            "require_full_gpu",
                        )
                    },
                }
            )
    return row, workloads


def materialize(db, selector):
    row, workloads = expand(db, selector)
    return row, [{**db.save_config(w["value"]), "matrix_placement": w["placement"]} for w in workloads]
