import hashlib
import importlib.metadata
import json
from pathlib import Path

import pytest

from sls_bench import worker


@pytest.mark.parametrize("drift", [None, "hash", "actual", "ledger", "missing", "malformed"])
def test_preinstalled_runtime_requires_hash_and_live_versions(tmp_path, monkeypatch, drift):
    requirements = ["torch==2.9.1", "diffusers==0.42.1"]
    versions = {"torch": "2.9.1+cu128", "diffusers": "0.42.1"}
    manifest = {
        "requirements_sha256": hashlib.sha256(("\n".join(requirements) + "\n").encode()).hexdigest(),
        "versions": dict(versions),
    }
    if drift == "hash":
        manifest["requirements_sha256"] = "wrong"
    if drift == "actual":
        versions["diffusers"] = "0.42.2"
    if drift == "ledger":
        manifest["versions"]["torch"] = "2.9.1+cu130"
    if drift == "malformed":
        manifest = []
    path = tmp_path / "runtime.json"
    if drift != "missing":
        path.write_text(json.dumps(manifest))
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda name: versions[name])
    assert worker.preinstalled_diffusers(requirements, path) is (drift is None)


def test_preinstalled_runtime_missing_distribution_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "runtime.json"
    requirements = ["not-installed==1"]
    path.write_text(json.dumps({
        "requirements_sha256": hashlib.sha256(b"not-installed==1\n").hexdigest(),
        "versions": {"not-installed": "1"},
    }))

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(worker.importlib.metadata, "version", missing)
    assert not worker.preinstalled_diffusers(requirements, path)


def test_docker_context_is_allowlisted_and_contains_no_data(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    target = tmp_path / "context"
    subprocess.run(
        [sys.executable, str(root / "containers/diffusers/prepare_context.py"), str(target)],
        check=True, cwd=root,
    )
    assert {p.name for p in target.iterdir()} == {
        "Dockerfile", "bootstrap.sh", "verify_runtime.py", "requirements.txt", ".dockerignore",
    }
    from sls_bench.image_diffusers import REQUIREMENTS

    assert (target / "requirements.txt").read_text() == "\n".join(REQUIREMENTS) + "\n"
    assert "HF_TOKEN" not in (target / "Dockerfile").read_text()
