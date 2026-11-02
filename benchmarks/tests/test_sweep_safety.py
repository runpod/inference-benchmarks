import json
from unittest.mock import Mock

import pytest

from sls_bench import sweeps
from sls_bench.config import validate
from sls_bench.db import Database


def test_catalog_placeholders_are_recorded_blocked_not_dropped(tmp_path, monkeypatch):
    db = Database(tmp_path / "bench.sqlite3")
    api = Mock()
    api.gpus.return_value = [{"id": "unknown", "manufacturer": None, "memoryInGb": 0, "securePrice": None}]

    def resolved(target, workloads):
        return {
            "code_sha256": "code",
            "target": json.loads(target["spec"]),
            "workloads": [{**validate(json.loads(w["spec"])), "config_id": w["id"]} for w in workloads],
        }

    monkeypatch.setattr(sweeps, "resolve", resolved)
    jid = sweeps.prepare(db, api, all_gpus=True, suite="image")
    row = db.conn.execute("SELECT status,assessments FROM sweep_items WHERE sweep_id=?", (jid,)).fetchone()
    assert row["status"] == "blocked"
    assert all("VRAM metadata" in a["reason"] for a in json.loads(row["assessments"]))
    api.create.assert_not_called()
    db.conn.close()


def test_cleanup_failure_stops_entire_queue(tmp_path, monkeypatch):
    db = Database(tmp_path / "bench.sqlite3")
    api = Mock()
    api.gpus.return_value = [
        {"id": name, "manufacturer": "Nvidia", "memoryInGb": 24, "securePrice": 1}
        for name in ("card-one", "card-two")
    ]
    monkeypatch.setattr(sweeps, "Runpod", Mock(return_value=api))

    def resolved(target, workloads):
        return {
            "code_sha256": "code",
            "target": json.loads(target["spec"]),
            "workloads": [{**validate(json.loads(w["spec"])), "config_id": w["id"]} for w in workloads],
        }

    monkeypatch.setattr(sweeps, "resolve", resolved)
    jid = sweeps.prepare(db, api, all_gpus=True, suite="image-smoke")
    with db.conn:
        db.conn.execute("UPDATE sweeps SET approved_budget=10 WHERE id=?", (jid,))

    def failed(db, api, rid, key, progress):
        db.update("runs", rid, status="failed")
        db.update("resources", rid, state="cleanup_failed")
        raise RuntimeError("Delete failed")

    execute = Mock(side_effect=failed)
    monkeypatch.setattr(sweeps, "execute", execute)
    with pytest.raises(RuntimeError, match="Cleanup unresolved"):
        sweeps.run(db.path, jid)
    assert execute.call_count == 1
    assert sweeps.job(db, jid)["status"] == "interrupted"
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    db.conn.close()
