"""Build-time CPU verification; never downloads weights or uses model credentials."""

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

root = Path("/opt/sls-bench")
requirements = (root / "requirements.txt").read_bytes()
versions = {}
for requirement in requirements.decode().splitlines():
    name, expected = requirement.split("==")
    actual = importlib.metadata.version(name)
    if actual.split("+")[0] != expected:
        raise RuntimeError(f"{name}: expected {expected}, found {actual}")
    versions[name] = actual

# Check imports without CUDA hardware, including the explicit FP8 dependency.
for module in ("diffusers", "torchao", "transformers"):
    importlib.import_module(module)
torch = importlib.import_module("torch")
pandas = importlib.import_module("pandas")

assert torch.version.cuda == "12.8"
assert pandas.__version__ == "2.2.3"
(root / "diffusers-runtime.json").write_text(json.dumps({
    "requirements_sha256": hashlib.sha256(requirements).hexdigest(),
    "versions": versions,
    "cuda": torch.version.cuda,
    "python_runtime": "3.11",
}, indent=2))
