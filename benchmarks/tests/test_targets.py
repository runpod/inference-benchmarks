import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from sls_bench.cli import app
from sls_bench.config import defaults
from sls_bench.db import Database
from sls_bench.targets import hydrate_target, materialize_target

SKU = "Future Accelerator 144GB"
CATALOG = {SKU: {"id": SKU, "memoryInGb": 144, "securePrice": 2, "communityPrice": None}}


def test_no_gpu_types_seeded_into_production_defaults(tmp_path):
    assert all(c["kind"] != "target" for c in defaults())
    db = Database(tmp_path / "test.sqlite3")
    db.seed()
    assert db.conn.execute("SELECT COUNT(*) FROM configs WHERE kind='target'").fetchone()[0] == 0
    db.conn.close()


def test_any_catalog_gpu_can_be_used_without_code_changes(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    row = materialize_target(db, SKU, CATALOG, gpu_count=2, cloud="SECURE")
    value = json.loads(row["spec"])
    assert value["gpu_type"] == SKU
    assert value["expected_memory_gb"] == 144
    assert value["require_mig"] is False
    assert value["gpu_count"] == 2
    assert materialize_target(db, SKU, CATALOG, gpu_count=2, cloud="SECURE")["id"] == row["id"]
    db.conn.close()


def test_saved_target_hydrates_memory_from_catalog(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    row = db.save_config({"kind": "target", "name": "custom", "gpu_type": SKU})
    assert json.loads(row["spec"])["expected_memory_gb"] is None
    assert json.loads(hydrate_target(row, CATALOG)["spec"])["expected_memory_gb"] == 144
    db.conn.close()


def test_gpu_arg_queries_catalog_persists_config_and_never_allocates_in_plan(tmp_path, monkeypatch):
    api = Mock()
    api.gpus.return_value = list(CATALOG.values())
    monkeypatch.setattr("sls_bench.cli.Runpod", Mock(return_value=api))
    runner = CliRunner()
    args = ["--db", str(tmp_path / "test.sqlite3")]
    assert runner.invoke(app, [*args, "init"]).exit_code == 0
    result = runner.invoke(app, [*args, "plan", "--gpu", SKU, "--gpu-count", "2", "-w", "qwen8b-chat"])
    assert result.exit_code == 0, result.output
    assert SKU in result.output
    api.create.assert_not_called()
    db = Database(tmp_path / "test.sqlite3")
    row = db.conn.execute("SELECT spec FROM configs WHERE kind='target'").fetchone()
    assert json.loads(row[0])["gpu_count"] == 2
    db.conn.close()
    result = runner.invoke(app, [*args, "plan", "--gpu", "not-a-catalog-sku", "-w", "qwen8b-chat"])
    assert result.exit_code != 0
    assert "not in live Runpod catalog" in result.output
    api.create.assert_not_called()


def test_unknown_gpu_never_materialized(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    with pytest.raises(ValueError, match="not in live"):
        materialize_target(db, "unknown", CATALOG)
    assert db.conn.execute("SELECT COUNT(*) FROM configs").fetchone()[0] == 0
    db.conn.close()
