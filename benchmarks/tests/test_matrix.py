import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from sls_bench.cli import app
from sls_bench.db import Database
from sls_bench.matrix import expand, materialize
from sls_bench.sweeps import assess, workloads_for


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "matrix.sqlite3")
    db.seed()
    yield db
    db.conn.close()


def test_text_matrix_is_authoritative_and_covers_revised_models_and_load_modes(db):
    row, workloads = expand(db, "text_gen")
    assert row["kind"] == "matrix"
    assert len(workloads) == 15
    assert {w["value"]["model"] for w in workloads} == {
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-32B-FP8",
        "Qwen/Qwen3-30B-A3B",
    }
    for w in workloads:
        v = w["value"]
        assert v["requests"] == 1024 and v["repeats"] == 2
        assert v["concurrency"] == ([1] if v["load_mode"] == "open_loop_request_rate" else [1, 8, 48, 128])
        assert v["requests_by_concurrency"] == (
            {} if v["load_mode"] == "open_loop_request_rate" else {"1": 512}
        )
        assert v["prompt_mode"] == "chat_template" and v["enable_thinking"] is False
        assert v["prefix_caching"] is False
        assert v["saturation_max_concurrency"] is None
        placement = w["placement"]
        gpu = {"id": "example", "manufacturer": "Nvidia", "memoryInGb": 24, "gpuGeneration": "HOPPER"}
        work = {"spec": json.dumps(v), "matrix_placement": placement}
        status, reason = assess(gpu, work, {"gpu_count": placement["gpu_count"]})
        assert status == ("eligible_unvalidated" if v["model"] == "Qwen/Qwen3-8B" else "blocked")
        gpu["memoryInGb"] = 80
        assert assess(gpu, work, {"gpu_count": placement["gpu_count"]})[0] == "eligible_unvalidated"
        if v["model"] == "Qwen/Qwen3-32B":
            assert placement["gpu_count"] == 2
        if v["quantization"] == "fp8":
            assert v["model"].endswith("-FP8") and v["min_compute_capability"] == 90
            gpu["gpuGeneration"] = "AMPERE"
            assert assess(gpu, work, {"gpu_count": 1})[0] == "blocked"
        if w["profile"] == "qwen8b-decode4k":
            assert (v["input_tokens"], v["output_tokens"], v["max_model_len"]) == (512, 4096, 4608)


def test_matrix_materialization_and_revision_are_immutable(db):
    row, first = materialize(db, "text_gen")
    assert [r["id"] for r in materialize(db, "text_gen")[1]] == [r["id"] for r in first]
    value = json.loads(row["spec"])
    value["parameters"]["requests"] = 2048
    revision = db.save_config(value)
    assert revision["revision"] == 2
    _, updated = materialize(db, "text_gen")
    assert all(json.loads(r["spec"])["requests"] == 2048 for r in updated)
    assert all(json.loads(r["spec"])["requests"] == 1024 for r in materialize(db, "text_gen@1")[1])
    assert not set(r["id"] for r in first) & set(r["id"] for r in updated)


def test_expansion_rejects_wrong_type_and_conflicting_overrides(db):
    value = json.loads(db.config("text_gen", "matrix")["spec"])
    value["rows"][0]["profiles"] = ["sdxl-1024"]
    db.save_config(value)
    with pytest.raises(ValueError, match="wrong benchmark type"):
        expand(db, "text_gen")
    with pytest.raises(ValueError, match="authoritative"):
        workloads_for(db, matrix="image_gen", parameters={"requests": 2})
    assert len(expand(db, "image_gen")[1]) == 2


def test_matrix_cli_is_readable_and_never_allocates(db, monkeypatch):
    api = Mock()
    monkeypatch.setattr("sls_bench.commands.Runpod", api)
    args = ["--db", str(db.path), "matrix", "text_gen"]
    result = CliRunner().invoke(app, args + ["--json"])
    assert result.exit_code == 0, (result.output, result.exception)
    value = json.loads(result.output)
    assert len(value["workloads"]) == 15
    assert value["catalog_source"] == "cached"
    api.assert_not_called()
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Matrix text_gen" in result.output
