import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "export", Path(__file__).resolve().parents[1] / "tools/export_publication.py"
)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class ExportTests(unittest.TestCase):
    def test_allowlist_and_snapshot_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live.sqlite3"
            c = sqlite3.connect(live)
            c.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE configs(id INTEGER, spec TEXT);
                CREATE TABLE runs(id TEXT, spec TEXT, status TEXT, created_at REAL, finished_at REAL,
                    hourly_price REAL, estimated_cost REAL, code_sha256 TEXT);
                CREATE TABLE trials(id INTEGER, run_id TEXT, workload_config_id INTEGER, ordinal INTEGER,
                    concurrency INTEGER, repeat INTEGER, status TEXT, result TEXT);
                CREATE TABLE metrics(trial_id INTEGER, name TEXT, value REAL);
                CREATE TABLE hardware(run_id TEXT, details TEXT);
            """)
            workload = {"config_id": 1, "name": "test", "model_revision": "frozen", "prompts": ["SECRET"]}
            c.execute("INSERT INTO configs VALUES(1,?)", (json.dumps({"model_revision": "newer"}),))
            c.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)",
                (
                    "private-run",
                    json.dumps(
                        {
                            "workloads": [workload],
                            "target": {"gpu_type": "test", "container_registry_auth_id": "SECRET"},
                            "sources": {"secret": "SECRET"},
                        }
                    ),
                    "completed",
                    0,
                    1,
                    0.5,
                    1,
                    "code-hash",
                ),
            )
            c.execute(
                "INSERT INTO trials VALUES(1,'private-run',1,0,1,0,'completed',?)",
                (
                    json.dumps(
                        {
                            "requests": [{"prompt": "SECRET"}],
                            "error": "SECRET",
                            "sample_class": "smoke",
                            "planned_requests": 100,
                            "requested_requests": 2,
                        }
                    ),
                ),
            )
            c.execute("INSERT INTO trials VALUES(2,'private-run',1,1,1,0,'incomplete_time_budget','{}')")
            c.execute("INSERT INTO metrics VALUES(1,'output_throughput',3)")
            c.execute("INSERT INTO metrics VALUES(1,'unknown_secret_metric',9)")
            c.execute(
                "INSERT INTO hardware VALUES('private-run',?)",
                (json.dumps({"nvidia_smi": "SECRET", "torch_devices": [{"name": "GPU", "uuid": "SECRET"}]}),),
            )
            c.commit()
            before = c.total_changes
            # Real online backup while the source connection remains open in WAL mode.
            src = sqlite3.connect(live.as_uri() + "?mode=ro", uri=True)
            snapshot = root / "snapshot.sqlite3"
            dst = sqlite3.connect(snapshot)
            src.backup(dst)
            dst.close()
            src.close()
            counts = export.export_snapshot(snapshot, root / "out")
            self.assertEqual(counts["trials"], 2)
            self.assertEqual(c.total_changes, before)
            self.assertEqual(c.execute("SELECT count(*) FROM trials").fetchone()[0], 2)
            c.close()
            text = "".join(p.read_text() for p in (root / "out").iterdir())
            self.assertNotIn("SECRET", text)
            self.assertNotIn("private-run", text)
            self.assertNotIn("unknown_secret_metric", text)
            self.assertIn("frozen", text)
            self.assertNotIn("newer", text)
            trials = json.loads((root / "out/trials.json").read_text())
            self.assertEqual(trials[0]["recorded"]["sample_class"], "smoke")
            self.assertEqual(trials[1]["status"], "incomplete_time_budget")
            self.assertEqual(trials[1]["metrics"], {})


if __name__ == "__main__":
    unittest.main()
