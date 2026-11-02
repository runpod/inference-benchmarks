import io
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sls_bench import worker
from sls_bench.config import Image, sdxl_workflow


@pytest.fixture
def image_setup(tmp_path, monkeypatch):
    (tmp_path / "ComfyUI" / "models" / "checkpoints").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"model")
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=Mock(return_value=str(checkpoint)))
    )
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "REPORT", {"status": "running", "trials": [], "phases": []})
    monkeypatch.setattr(worker, "command", Mock())
    monkeypatch.setattr(worker, "capture", Mock(return_value="packages==pinned"))
    server = Mock()
    monkeypatch.setattr(worker, "start", Mock(return_value=server))
    monkeypatch.setattr(worker, "ready", Mock())
    stopper = Mock()
    monkeypatch.setattr(worker, "stop", stopper)
    monkeypatch.setattr(worker.urllib.request, "urlopen", lambda _: io.BytesIO(b"PNG"))
    config = Image(
        name="image",
        comfy_revision="commit",
        workflow=sdxl_workflow(),
        requests=2,
        repeats=2,
        warmup_requests=1,
    ).model_dump()
    config["config_id"] = 123
    yield config, stopper, tmp_path


def test_image_adapter_queues_unique_seeds_persists_samples_and_stops(image_setup, monkeypatch):
    config, stopper, path = image_setup
    graphs = []

    def api(endpoint, body=None, **kwargs):
        if endpoint == "/prompt":
            graphs.append(body["prompt"])
            return {"prompt_id": str(len(graphs))}
        prompt_id = endpoint.split("/")[-1]
        return {
            prompt_id: {
                "status": {"status_str": "success", "messages": []},
                "outputs": {"9": {"images": [{"filename": "image.png", "subfolder": "", "type": "output"}]}},
            }
        }

    monkeypatch.setattr(worker, "api", api)
    worker.image_workload(config)
    assert len(graphs) == 5
    assert len({g["3"]["inputs"]["seed"] for g in graphs}) == 5
    assert config["workflow"]["3"]["inputs"]["seed"] == 42
    assert len(worker.REPORT["trials"]) == 2
    assert all(t["status"] == "completed" for t in worker.REPORT["trials"])
    assert all(t["metrics"]["images"] == 2 for t in worker.REPORT["trials"])
    assert len(list(path.glob("image-*.png"))) == 2
    assert (path / "report.json").exists()
    stopper.assert_called_once()


def test_image_api_failure_still_stops_owned_engine(image_setup, monkeypatch):
    config, stopper, _ = image_setup

    def api(endpoint, body=None, **kwargs):
        if endpoint == "/prompt":
            return {"prompt_id": "failed"}
        return {"failed": {"status": {"status_str": "error", "messages": [["execution_error", {}]]}}}

    monkeypatch.setattr(worker, "api", api)
    with pytest.raises(RuntimeError, match="ComfyUI execution failed"):
        worker.image_workload(config)
    stopper.assert_called_once()
    assert not worker.REPORT["trials"]
