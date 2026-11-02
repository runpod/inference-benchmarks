import json
import re
from pathlib import Path

import httpx

from .config import validate
from .db import digest


def resolve_image(image, client):
    if "@sha256:" in image:
        return image
    # Initial implementation supports public Docker Hub images; no registry secrets on Pods.
    if "/" not in image or image.count("/") > 1:
        raise ValueError("Use a Docker Hub namespace/image:tag or an image pinned by digest")
    repository, tag = image.rsplit(":", 1) if ":" in image else (image, "latest")
    token = client.get(
        "https://auth.docker.io/token",
        params={"service": "registry.docker.io", "scope": f"repository:{repository}:pull"},
    )
    token.raise_for_status()
    result = client.get(
        f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
        headers={
            "Authorization": "Bearer " + token.json()["token"],
            "Accept": ", ".join(
                [
                    "application/vnd.oci.image.index.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.docker.distribution.manifest.v2+json",
                ]
            ),
        },
    )
    result.raise_for_status()
    return f"{repository}@{result.headers['docker-content-digest']}"


def source_snapshot():
    root = Path(__file__).parent
    return {p.name: p.read_text() for p in sorted(root.glob("*.py"))}


def source_hash(sources=None):
    sources = sources if sources is not None else source_snapshot()
    return digest(b"".join(name.encode() + sources[name].encode() for name in sorted(sources)))


def resolve(target, workloads, client=None):
    target = json.loads(target["spec"])
    resolved = []
    with httpx.Client(timeout=60, follow_redirects=True) if client is None else client as http:
        target["image"] = resolve_image(target["image"], http)
        revisions = {}
        dataset_revisions = {}
        comfy_revisions = {}
        for row in workloads:
            value = validate(json.loads(row["spec"]))
            key = (value["model"], value["model_revision"])
            if key not in revisions:
                result = http.get(f"https://huggingface.co/api/models/{key[0]}/revision/{key[1]}")
                result.raise_for_status()
                revisions[key] = result.json()["sha"]
            value["model_revision"] = revisions[key]
            if value.get("dataset_repo"):
                dataset_key = (value["dataset_repo"], value["dataset_revision"])
                if dataset_key not in dataset_revisions:
                    result = http.get(
                        f"https://huggingface.co/api/datasets/{dataset_key[0]}/revision/{dataset_key[1]}"
                    )
                    result.raise_for_status()
                    dataset_revisions[dataset_key] = result.json()["sha"]
                value["dataset_revision"] = dataset_revisions[dataset_key]
            if (
                value["kind"] == "image"
                and value.get("backend", "comfyui") == "comfyui"
                and not re.fullmatch(r"[0-9a-f]{40}", value["comfy_revision"])
            ):
                ref = value["comfy_revision"]
                if ref not in comfy_revisions:
                    result = http.get(f"https://api.github.com/repos/comfyanonymous/ComfyUI/commits/{ref}")
                    result.raise_for_status()
                    comfy_revisions[ref] = result.json()["sha"]
                value["comfy_revision"] = comfy_revisions[ref]
            resolved.append({"config_id": row["id"], **value})
    from .worker import text_engine_plan

    sources = source_snapshot()
    lock = Path(__file__).parent.parent / "uv.lock"
    return {
        "schema_version": 1,
        "code_sha256": source_hash(sources),
        "worker_sha256": digest(sources["worker.py"].encode()),
        "sources": sources,
        "controller_lock": lock.read_text() if lock.exists() else None,
        "replica_count": 1,
        "gpus_per_replica": target["gpu_count"],
        "load_mode": "closed_loop_concurrency",
        "client_location": "pod_localhost",
        "target": target,
        "workloads": resolved,
        "text_engine_plan": text_engine_plan(resolved),
    }
