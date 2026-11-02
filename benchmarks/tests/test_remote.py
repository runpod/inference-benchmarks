import io
import json
import sqlite3
import tarfile
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from sls_bench import remote, sweeps
from sls_bench.cli import app
from sls_bench.db import SCHEMA, Database, encode
from tests.test_sweeps import CATALOG
from tests.test_sweeps import setup as sweep_fixture

setup = sweep_fixture


def prepared(setup):
    db, api, key, _ = setup
    identity = sweeps.prepare(db, api, gpu_types=[CATALOG[0]["id"]], suite="image", ssh_key=key)
    return db, api, key, identity


def snapshot(db, identity, tmp_path):
    data = remote.bundle(db, identity, 100, 1)
    folder = tmp_path / "staged"
    folder.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(folder, filter="data")
    return Database(folder / "data/benchmarks.sqlite3")


def test_bundle_is_scoped_and_rewrites_only_host_options(setup, tmp_path):
    db, _, key, identity = prepared(setup)
    staged = snapshot(db, identity, tmp_path)
    job = sweeps.job(staged, identity)
    assert json.loads(job["options"])["ssh_key"] == f"{remote.ROOT}/data/controller_key"
    assert key.read_text() == "private"
    assert staged.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert job["approved_budget"] == 100
    staged.conn.close()


def test_import_remaps_trials_and_is_repeatable(setup, tmp_path):
    db, _, _, identity = prepared(setup)
    staged = snapshot(db, identity, tmp_path)
    item = dict(staged.conn.execute("SELECT * FROM sweep_items WHERE sweep_id=?", (identity,)).fetchone())
    spec = json.loads(item["spec"])
    target = dict(
        staged.conn.execute("SELECT * FROM configs WHERE id=?", (item["target_config_id"],)).fetchone()
    )
    workloads = [
        dict(staged.conn.execute("SELECT * FROM configs WHERE id=?", (w["config_id"],)).fetchone())
        for w in spec["workloads"]
    ]
    spec.update(sweep_id=identity, sweep_item_id=item["id"])
    # An unrelated local trial consumes the same integer ID as the remote trial.
    local_run = db.new_run(target, workloads, spec, 60)
    trial = {
        "config_id": workloads[0]["id"],
        "concurrency": 1,
        "repeat": 0,
        "status": "completed",
        "metrics": {"images_per_second": 2},
        "requests": [{"ok": True}],
    }
    db.ingest(local_run, {"trials": [trial]})
    run_id = staged.new_run(target, workloads, spec, 60)
    staged.ingest(run_id, {"trials": [trial], "hardware": {"gpu": "test"}})
    staged.artifact(run_id, "worker.log", b"result")
    staged.event(run_id, "complete", {"ok": True})
    staged.update("runs", run_id, status="completed")
    staged.update("resources", run_id, state="deleted")
    with staged.conn:
        staged.conn.execute(
            "UPDATE sweep_items SET run_id=?,status='completed' WHERE id=?", (run_id, item["id"])
        )
        staged.conn.execute("UPDATE sweeps SET status='completed' WHERE id=?", (identity,))
    path = tmp_path / "export.sqlite3"
    staged.backup(path)
    for _ in range(2):
        assert remote.merge(db, path, identity) == "completed"
    assert db.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 2
    trial_id = db.conn.execute("SELECT id FROM trials WHERE run_id=?", (run_id,)).fetchone()[0]
    assert trial_id != 1
    assert db.conn.execute("SELECT value FROM metrics WHERE trial_id=?", (trial_id,)).fetchone()[0] == 2
    assert db.conn.execute("SELECT COUNT(*) FROM requests WHERE trial_id=?", (trial_id,)).fetchone()[0] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (run_id,)).fetchone()[0] == 1
    assert db.run(local_run)["status"] == "planned"
    assert db.run(run_id)["status"] == "completed"
    staged.conn.close()


def test_remote_ownership_blocks_local_start(setup):
    db, _, key, identity = prepared(setup)
    remote.setup(db)
    with db.conn:
        db.conn.execute(
            "INSERT INTO remote_controllers VALUES(?,?,?,?,?,?,?,?)",
            (identity, "cpu", "pod", str(key), "running", 1, None, None),
        )
    with pytest.raises(ValueError, match="remote controller"):
        sweeps.start(db, identity, 100)


def test_fleet_diagnostics_share_ownership_but_cannot_release_or_resume_cpu(setup):
    db, api, key, owner = prepared(setup)
    diagnostic = sweeps.prepare(db, api, gpu_types=[CATALOG[0]["id"]], suite="image", ssh_key=key)
    options = json.loads(sweeps.job(db, diagnostic)["options"])
    options["fleet_owner"] = owner
    remote.setup(db)
    with db.conn:
        db.conn.execute("UPDATE sweeps SET options=? WHERE id=?", (encode(options), diagnostic))
        db.conn.execute("INSERT INTO remote_controllers VALUES(?,?,?,?,?,?,?,?)",
                        (owner, "cpu", "pod", str(key), "running", 1, None, None))
    assert remote.record(db, diagnostic)["sweep_id"] == owner
    with pytest.raises(ValueError, match="remote controller"):
        sweeps.start(db, diagnostic, 100)
    with pytest.raises(ValueError, match="fleet owner"):
        remote.release(db, api, diagnostic)
    with pytest.raises(ValueError, match="fleet owner"):
        remote.resume(db, api, diagnostic)
    api.delete.assert_not_called()


def test_ambiguous_create_is_recorded_and_never_retried(setup, monkeypatch):
    db, api, key, identity = prepared(setup)
    monkeypatch.setenv("RUNPOD_ORGS_API_KEY", "secret")
    api.rest.side_effect = TimeoutError("ambiguous")
    with pytest.raises(TimeoutError):
        remote.launch(db, api, identity, 100, 1, key)
    assert remote.record(db, identity)["state"] == "creating"
    with pytest.raises(ValueError, match="already recorded"):
        remote.launch(db, api, identity, 100, 1, key)
    api.rest.assert_called_once()
    payload = api.rest.call_args.kwargs["json"]
    assert payload["computeType"] == "CPU"
    assert "secret" not in encode(payload)
    assert "private" not in encode(payload)


def test_release_refuses_live_sweep(setup, monkeypatch):
    db, api, _, identity = prepared(setup)
    monkeypatch.setattr(remote, "pull", Mock(return_value="running"))
    with pytest.raises(ValueError, match="retained"):
        remote.release(db, api, identity)
    api.delete.assert_not_called()


def test_launch_invokes_existing_dispatcher_without_uploading_local_private_key(setup, monkeypatch):
    db, api, key, identity = prepared(setup)
    monkeypatch.setenv("RUNPOD_ORGS_API_KEY", "api-secret")
    api.rest.return_value = {"id": "cpu"}
    api.pod.return_value = {"id": "cpu", "costPerHr": 0.05}
    ssh = Mock()
    monkeypatch.setattr(remote, "connection", Mock(return_value=ssh))
    remote.launch(db, api, identity, 100, 0.10, key)
    assert remote.record(db, identity)["state"] == "running"
    calls = ssh.call.call_args_list
    assert any("-m sls_bench.remote" in call.args[0] for call in calls)
    assert any("uv sync --frozen --no-dev" in call.args[0] for call in calls)
    assert any("ssh-keygen" in call.args[0] for call in calls)
    for call in calls:
        assert "api-secret" not in call.args[0]
        assert len(call.args) < 2 or call.args[1] != b"private"


def test_stage_only_never_starts_a_gpu_dispatcher(setup, monkeypatch):
    db, api, key, identity = prepared(setup)
    monkeypatch.setenv("RUNPOD_ORGS_API_KEY", "secret")
    api.rest.return_value = {"id": "cpu"}
    api.pod.return_value = {"id": "cpu", "costPerHr": 0.05}
    ssh = Mock()
    monkeypatch.setattr(remote, "connection", Mock(return_value=ssh))
    from sls_bench.resolve import source_hash

    spec = json.loads(db.conn.execute("SELECT spec FROM sweep_items WHERE sweep_id=?", (identity,)).fetchone()[0])
    spec["code_sha256"] = source_hash(spec["sources"])
    with db.conn:
        db.conn.execute("UPDATE sweep_items SET spec=? WHERE sweep_id=?", (encode(spec), identity))
    remote.launch(db, api, identity, 100, 0.10, key, stage_only=True, frozen_sources=True)
    assert remote.record(db, identity)["state"] == "staged"
    assert not any("nohup" in call.args[0] for call in ssh.call.call_args_list)


def test_frozen_bundle_uses_captured_sources_not_mutable_checkout(setup):
    db, _, _, identity = prepared(setup)
    spec = json.loads(db.conn.execute("SELECT spec FROM sweep_items WHERE sweep_id=?", (identity,)).fetchone()[0])
    from sls_bench.resolve import source_hash

    spec["code_sha256"] = source_hash(spec["sources"])
    with db.conn:
        db.conn.execute("UPDATE sweep_items SET spec=? WHERE sweep_id=?", (encode(spec), identity))
    payload = remote.bundle(db, identity, 100, 1, frozen_sources=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for name, content in spec["sources"].items():
            assert archive.extractfile("sls_bench/" + name).read() == content.encode()
    spec["code_sha256"] = "wrong"
    with db.conn:
        db.conn.execute("UPDATE sweep_items SET spec=? WHERE sweep_id=?", (encode(spec), identity))
    with pytest.raises(ValueError, match="source hash mismatch"):
        remote.bundle(db, identity, 100, 1, frozen_sources=True)


def test_cpu_over_price_is_deleted_before_staging(setup, monkeypatch):
    db, api, key, identity = prepared(setup)
    monkeypatch.setenv("RUNPOD_ORGS_API_KEY", "secret")
    api.rest.return_value = {"id": "cpu"}
    api.pod.return_value = {"id": "cpu", "costPerHr": 2}
    with pytest.raises(ValueError, match="exceeds approval"):
        remote.launch(db, api, identity, 100, 0.10, key)
    api.delete.assert_called_once_with("cpu")
    assert remote.record(db, identity)["state"] == "deleted"


def test_wrong_lineage_import_does_not_change_local_sweep(setup, tmp_path):
    db, _, _, identity = prepared(setup)
    staged = snapshot(db, identity, tmp_path)
    with staged.conn:
        staged.conn.execute("UPDATE sweep_items SET spec=? WHERE sweep_id=?", ("{}", identity))
        staged.conn.execute("UPDATE sweeps SET status='completed' WHERE id=?", (identity,))
    path = tmp_path / "bad.sqlite3"
    staged.backup(path)
    with pytest.raises(ValueError, match="identity mismatch"):
        remote.merge(db, path, identity)
    assert sweeps.job(db, identity)["status"] == "prepared"
    staged.conn.close()


def test_repeated_pull_uses_new_remote_backup_paths(setup, tmp_path, monkeypatch):
    db, api, _, identity = prepared(setup)
    remote.setup(db)
    staged = snapshot(db, identity, tmp_path)
    export = tmp_path / "export.sqlite3"
    staged.backup(export)
    ssh = Mock()

    def call(command, *args, **kwargs):
        if command.startswith("cat "):
            return export.read_bytes()
        if command.startswith("tail "):
            return b"controller log"
        return b""

    ssh.call.side_effect = call
    monkeypatch.setattr(remote, "connection", Mock(return_value=ssh))
    assert remote.pull(db, api, identity) == "queued"
    assert remote.pull(db, api, identity) == "queued"
    backups = [c.args[0] for c in ssh.call.call_args_list if " backup " in c.args[0]]
    assert len(backups) == 2 and backups[0] != backups[1]
    staged.conn.close()


def test_nullable_metadata_schema_is_migrated_before_dispatch(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.replace("metadata IS NULL OR ", ""))
    conn.close()
    db = Database(path)
    sql = db.conn.execute("SELECT sql FROM sqlite_master WHERE name='resources'").fetchone()[0]
    assert "metadata IS NULL OR json_valid(metadata)" in sql
    assert db.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    db.conn.close()


def test_explicit_cpu_capacity_rejection_allows_retry(setup, monkeypatch):
    db, api, key, identity = prepared(setup)
    monkeypatch.setenv("RUNPOD_ORGS_API_KEY", "secret")
    api.rest.side_effect = RuntimeError("There are no longer any instances available")
    api.pods.return_value = []
    with pytest.raises(RuntimeError):
        remote.launch(db, api, identity, 100, 1, key)
    assert remote.record(db, identity)["state"] == "rejected"
    assert sweeps.job(db, identity)["status"] == "prepared"
    with pytest.raises(RuntimeError):
        remote.launch(db, api, identity, 100, 1, key, cpu_flavor="cpu3c,cpu5c")
    assert api.rest.call_args.kwargs["json"]["cpuFlavorIds"] == ["cpu3c", "cpu5c"]


def test_cli_pull_accepts_positional_sweep_id(tmp_path, monkeypatch):
    from sls_bench import remote_commands

    monkeypatch.setattr(remote_commands, "Runpod", Mock())
    puller = Mock(return_value="running")
    monkeypatch.setattr(remote, "pull", puller)
    result = CliRunner().invoke(app, ["--db", str(tmp_path / "db"), "remote", "pull", "sweep-id"])
    assert result.exit_code == 0
    assert puller.call_args.args[2] == "sweep-id"


def test_cli_requires_explicit_paid_approval(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "--db",
            str(tmp_path / "db"),
            "remote",
            "launch",
            "id",
            "--budget",
            "100",
            "--cpu-hourly-limit",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "--yes" in result.output
