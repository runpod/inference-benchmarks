import json
import sqlite3
import subprocess
import sys
import zlib
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from sls_bench.cli import app
from sls_bench.config import Image, Text, defaults, sdxl_workflow, validate
from sls_bench.controller import cleanup
from sls_bench.db import Database
from sls_bench.resolve import resolve_image
from sls_bench.runpod import Runpod, safe_metadata
from sls_bench.worker import bench_command, image_graph, normalize_text, verify_hardware

from .helpers import sample_targets, seed_test_targets


@pytest.fixture
def db(tmp_path):
    value = Database(tmp_path / "bench.sqlite3")
    value.seed()
    seed_test_targets(value)
    yield value
    value.conn.close()


def make_run(db, **kwargs):
    return db.new_run(db.config("4090"), [db.config("qwen8b-chat")], {"code_sha256": "test"}, 60, **kwargs)


def test_defaults_and_context_lengths():
    for config in defaults():
        assert validate(config) == config
    context = next(c for c in defaults() if c["name"] == "qwen8b-context32k")
    assert context["input_tokens"] + context["output_tokens"] == 32768
    assert context["input_tokens"] > 30000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_tokens": 4096},
        {"concurrency": [0]},
        {"concurrency": [33]},
        {"concurrency": [1, 1]},
        {"surprise": 1},
        {"gpu_memory_utilization": 1},
    ],
)
def test_invalid_configs(kwargs):
    with pytest.raises(ValidationError):
        Text(name="invalid", **kwargs)


def test_revisions_and_seed_dont_overwrite(db):
    original = db.config("qwen8b-chat")
    config = json.loads(original["spec"])
    config["input_tokens"] = 1024
    changed = db.save_config(config)
    assert changed["revision"] == 2
    assert changed["id"] != original["id"]
    assert db.config("qwen8b-chat", revision=1)["spec"] == original["spec"]
    assert db.save_config(config)["id"] == changed["id"]
    db.seed()
    assert db.config("qwen8b-chat")["id"] == changed["id"]
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("UPDATE configs SET name='oops' WHERE id=?", (original["id"],))
    db.conn.rollback()


def test_ingestion_idempotent_and_artifact_latest(db):
    run_id = make_run(db)
    report = {
        "hardware": {"gpu": "4090"},
        "trials": [
            {
                "config_id": db.config("qwen8b-chat")["id"],
                "concurrency": 1,
                "repeat": 0,
                "status": "completed",
                "metrics": {"output_throughput": 100},
                "requests": [{"ttft_seconds": 0.1}],
            }
        ],
    }
    db.ingest(run_id, report)
    db.ingest(run_id, report)
    assert db.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1
    db.artifact(run_id, "report.json", b"first")
    db.artifact(run_id, "report.json", b"latest")
    data = db.conn.execute("SELECT content FROM artifacts").fetchone()[0]
    assert zlib.decompress(data) == b"latest"


def test_ingestion_rolls_back(db):
    run_id = make_run(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.ingest(
            run_id,
            {
                "hardware": {"gpu": "4090"},
                "trials": [{"config_id": 99999, "concurrency": 1, "repeat": 0, "status": "completed"}],
            },
        )
    assert db.conn.execute("SELECT COUNT(*) FROM hardware").fetchone()[0] == 0


def test_cleanup_owned_only_and_idempotent(db):
    run_id = make_run(db)
    db.update("resources", run_id, pod_id="owned")
    attached = make_run(db, owned=False, pod_id="attached")
    api = Mock()
    cleanup(db, api, attached)
    api.delete.assert_not_called()
    cleanup(db, api, run_id, expired_only=True)
    api.delete.assert_not_called()
    cleanup(db, api, run_id)
    cleanup(db, api, run_id)
    api.delete.assert_called_once_with("owned")
    assert db.run(run_id)["resource_state"] == "deleted"


def test_existing_pod_can_be_reused_after_detaching_but_not_concurrently(db):
    first = make_run(db, owned=False, pod_id="existing")
    with pytest.raises(sqlite3.IntegrityError):
        make_run(db, owned=False, pod_id="existing")
    db.update("resources", first, state="detached")
    second = make_run(db, owned=False, pod_id="existing")
    assert db.run(first)["pod_id"] == db.run(second)["pod_id"] == "existing"
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2


def test_cleanup_ambiguous_create(db):
    run_id = make_run(db)
    name = db.run(run_id)["pod_name"]
    api = Mock()
    api.pods.return_value = [{"id": "lost-response", "name": name}, {"id": "other", "name": "unrelated"}]
    cleanup(db, api, run_id)
    api.delete.assert_called_once_with("lost-response")


def test_cleanup_failure_retained(db):
    run_id = make_run(db)
    db.update("resources", run_id, pod_id="owned")
    api = Mock()
    api.delete.side_effect = RuntimeError("network")
    with pytest.raises(RuntimeError):
        cleanup(db, api, run_id)
    assert db.run(run_id)["resource_state"] == "cleanup_failed"


def test_create_payload_has_no_org_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_ORGS_API_KEY", "controller-secret")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"data": {"podFindAndDeployOnDemand": {"id": "pod"}}})

    api = Runpod(httpx.Client(transport=httpx.MockTransport(respond)))
    target = sample_targets()[1]
    api.create("test", target, "ssh-ed25519 public", 60)
    body = json.loads(requests[0].content)["variables"]["input"]
    assert body["gpuTypeId"] == target["gpu_type"]
    assert body["gpuCount"] == 1
    assert "controller-secret" not in requests[0].content.decode()
    assert {e["key"]: e["value"] for e in body["env"]} == {
        "PUBLIC_KEY": "ssh-ed25519 public",
        "BENCH_TTL_SECONDS": "60",
    }
    assert "terminateAfter" in body
    assert json.loads(body["dockerArgs"])["entrypoint"] == ["/bin/bash"]
    assert body["volumeInGb"] == 0
    assert "containerRegistryAuthId" not in body
    private_target = {**target, "container_registry_auth_id": "saved-registry-id"}
    api.create("private-test", private_target, "ssh-ed25519 public", 60)
    private_body = json.loads(requests[-1].content)["variables"]["input"]
    assert private_body["containerRegistryAuthId"] == "saved-registry-id"
    assert "password" not in private_body

    # Exercise the actual bootstrap fragment with a Conda-like Python location.
    bootstrap = json.loads(body["dockerArgs"])["cmd"][1]
    fragment = bootstrap.split('python_bin="$(command -v python3)"', 1)[1].split(
        "mkdir -p /root/.ssh", 1
    )[0]
    conda = tmp_path / "conda" / "bin"
    conda.mkdir(parents=True)
    (conda / "python3").symlink_to(sys.executable)
    ssh_bin = tmp_path / "ssh-bin"
    ssh_bin.mkdir()
    link = ssh_bin / "python3"
    script = ('python_bin="$(command -v python3)"' + fragment).replace(
        "/usr/local/bin/python3", str(link)
    )
    for _ in range(2):  # Must also be safe when restarted with the link already present.
        subprocess.run(
            ["/bin/bash", "-ec", script], check=True,
            env={"PATH": f"{conda}:/usr/bin:/bin"},
        )
        assert link.readlink() == conda / "python3"
        result = subprocess.run(
            ["/bin/bash", "-ec", "command -v python3; python3 --version"],
            env={"PATH": f"{ssh_bin}:/usr/bin:/bin"}, capture_output=True, text=True, check=True,
        )
        assert result.stdout.splitlines()[0] == str(link)


def test_metadata_does_not_leak_env():
    assert "env" not in safe_metadata({"id": "p", "env": {"SECRET": "secret"}})


def test_normalization():
    metrics, requests = normalize_text(
        {
            "completed": 1,
            "output_throughput": 10,
            "input_lens": [512, 512],
            "output_lens": [256, 0],
            "ttfts": [0.1, 0],
            "itls": [[0.2, 0.3], []],
            "errors": ["", "OOM"],
        },
        2,
    )
    assert metrics["failed_requests"] == 1
    assert requests[0]["stream_token_span_seconds"] == pytest.approx(0.6)
    assert requests[1]["error"] == "OOM"


def test_bench_command_fixed_lengths_and_localhost():
    config = Text(name="test").model_dump()
    args = bench_command(config, "/model", 4, "trial.json", 32, 42)
    assert args[args.index("--base-url") + 1] == "http://127.0.0.1:8000"
    assert "--ignore-eos" in args
    assert args[args.index("--random-range-ratio") + 1] == "0"
    assert args[args.index("--max-concurrency") + 1] == "4"


def test_mig_verification_rejects_parent_and_wrong_size():
    target = sample_targets()[1]
    with pytest.raises(RuntimeError, match="not a MIG"):
        verify_hardware(target, {"torch_devices": [{"name": "RTX PRO 6000", "memory_bytes": 96 * 1024**3}]})
    verify_hardware(
        target, {"torch_devices": [{"name": "RTX PRO 6000 MIG 1g.24gb", "memory_bytes": 23 * 1024**3}]}
    )
    with pytest.raises(RuntimeError, match="VRAM"):
        verify_hardware(target, {"torch_devices": [{"name": "MIG 2g.48gb", "memory_bytes": 48 * 1024**3}]})


def test_image_seed_changes_graph_without_mutating_config():
    config = Image(name="image", comfy_revision="commit", workflow=sdxl_workflow()).model_dump()
    a = image_graph(config, 42, "teapot")
    b = image_graph(config, 43, "teapot")
    assert a != b
    assert config["workflow"]["3"]["inputs"]["seed"] == 42
    assert b["6"]["inputs"]["text"] == "teapot"


def test_pinned_image_resolution_skips_network():
    client = Mock()
    pinned = "vllm/vllm-openai@sha256:abc"
    assert resolve_image(pinned, client) == pinned
    client.get.assert_not_called()


def test_backup_includes_wal(db, tmp_path):
    run_id = make_run(db)
    path = tmp_path / "backup.sqlite3"
    db.backup(path)
    copy = Database(path)
    assert copy.run(run_id)["status"] == "planned"
    copy.conn.close()


def test_cli_initialization_plan_and_approval(tmp_path):
    runner = CliRunner()
    args = ["--db", str(tmp_path / "test.sqlite3")]
    assert runner.invoke(app, [*args, "init"]).exit_code == 0
    db = Database(tmp_path / "test.sqlite3")
    seed_test_targets(db)
    db.conn.close()
    result = runner.invoke(app, [*args, "plan", "-t", "4090", "-w", "qwen8b-context32k"])
    assert result.exit_code == 0, result.output
    assert "31744 input" in result.output
    result = runner.invoke(app, [*args, "sweep", "-t", "4090", "-w", "qwen8b-chat"])
    assert result.exit_code != 0
