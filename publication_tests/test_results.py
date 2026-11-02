import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ResultIntegrityTests(unittest.TestCase):
    def test_export_hashes_and_joins(self):
        manifest = json.loads((ROOT / "results/manifest.json").read_text())
        for name, expected in manifest["source_files_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)
        for name, expected in manifest["result_files_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest(), expected, name
            )
        self.assertEqual(
            hashlib.sha256((ROOT / "tools/export_publication.py").read_bytes()).hexdigest(),
            manifest["exporter_sha256"],
        )
        runs = json.loads((ROOT / "results/runs.json").read_text())
        trials = json.loads((ROOT / "results/trials.json").read_text())
        workloads = json.loads((ROOT / "results/workloads.json").read_text())
        keys = {r["run_key"] for r in runs}
        self.assertEqual(len(keys), len(runs))
        self.assertEqual(len(trials), manifest["counts"]["trials"])
        identities = set()
        for t in trials:
            self.assertIn(t["run_key"], keys)
            self.assertIn(t["workload_key"], workloads)
            identity = t["run_key"], t["ordinal"]
            self.assertNotIn(identity, identities)
            identities.add(identity)


if __name__ == "__main__":
    unittest.main()
