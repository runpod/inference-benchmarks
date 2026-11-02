import json
import sqlite3

import pytest
from typer.testing import CliRunner

from sls_bench.cli import app, selection
from sls_bench.config import Text
from sls_bench.db import Database
from sls_bench.worker import bench_command

from .helpers import seed_test_targets


def test_single_replica_is_distinct_from_request_concurrency(tmp_path):
    db = Database(tmp_path / "bench.sqlite3")
    db.seed()
    seed_test_targets(db)
    target, workload = db.config("mig24"), db.config("qwen8b-chat")
    spec = {"code_sha256": "test", "replica_count": 1, "gpus_per_replica": 1}
    run_id = db.new_run(target, [workload], spec, 60)
    db.update("runs", run_id, hourly_price=0.69)
    db.ingest(
        run_id,
        {
            "trials": [
                {
                    "config_id": workload["id"],
                    "concurrency": 16,
                    "repeat": 0,
                    "status": "completed",
                    "metrics": {
                        "output_throughput": 100,
                        "request_throughput": 2,
                        "p95_ttft_ms": 100,
                        "p95_e2el_ms": 500,
                        "completed": 32,
                        "failed_requests": 0,
                    },
                }
            ]
        },
    )
    row = dict(db.conn.execute("SELECT * FROM replica_trials").fetchone())
    assert row["replica_count"] == 1
    assert row["gpus_per_replica"] == 1
    assert row["concurrent_requests"] == 16
    assert (
        row["output_tokens_per_second"] == 100
    )  # Not divided by concurrency or multiplied by replica count.
    assert row["ttft_p95_ms"] == 100
    assert row["e2e_p95_ms"] == 500
    cost = db.conn.execute(
        "SELECT value FROM metrics WHERE name='estimated_dollars_per_million_output_tokens'"
    ).fetchone()[0]
    assert cost == pytest.approx(0.69 / 3600 / 100 * 1_000_000)
    result = CliRunner().invoke(app, ["--db", str(db.path), "results", run_id, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["replica_count"] == 1
    assert data[0]["concurrency"] == 16
    db.conn.close()


def test_historical_config_selection(tmp_path):
    db = Database(tmp_path / "bench.sqlite3")
    db.seed()
    seed_test_targets(db)
    old = db.config("qwen8b-chat")
    value = json.loads(old["spec"])
    value["input_tokens"] = 1024
    db.save_config(value)
    _, workloads = selection(db, ["mig24"], ["qwen8b-chat@1"])
    assert workloads[0]["id"] == old["id"]
    db.conn.close()


def test_v1_database_migrates_view_without_losing_configs(tmp_path):
    path = tmp_path / "bench.sqlite3"
    db = Database(path)
    db.seed()
    seed_test_targets(db)
    original = db.config("mig24")
    db.conn.execute("DROP VIEW replica_trials")
    db.conn.execute("CREATE VIEW replica_trials AS SELECT 1 AS old_column")
    db.conn.execute("PRAGMA user_version=1")
    db.conn.commit()
    db.conn.close()
    migrated = Database(path)
    assert migrated.config("mig24")["id"] == original["id"]
    columns = {r[1] for r in migrated.conn.execute("PRAGMA table_info(replica_trials)")}
    assert "replica_count" in columns and "engine_image" in columns
    assert migrated.conn.execute("PRAGMA user_version").fetchone()[0] == 4
    with pytest.raises(sqlite3.IntegrityError):
        migrated.conn.execute("DELETE FROM configs WHERE id=?", (original["id"],))
    migrated.conn.rollback()
    migrated.conn.close()


def test_optional_slo_flags():
    config = Text(name="slo", ttft_slo_ms=500, tpot_slo_ms=40).model_dump()
    args = bench_command(config, "/model", 1, "trial.json", 32, 42)
    offset = args.index("--goodput")
    assert args[offset + 1 :] == ["ttft:500.0", "tpot:40.0"]


def test_run_id_cli_is_positional(tmp_path):
    path = tmp_path / "bench.sqlite3"
    db = Database(path)
    db.seed()
    seed_test_targets(db)
    run_id = db.new_run(db.config("mig24"), [db.config("qwen8b-chat")], {"code_sha256": "test"}, 60)
    db.conn.close()
    result = CliRunner().invoke(app, ["--db", str(path), "status", run_id])
    assert result.exit_code == 0, result.output
    assert run_id in result.output
