import hashlib
import json
import sqlite3
import time
import uuid
import zlib
from pathlib import Path

from .config import defaults, validate

SCHEMA = """
CREATE TABLE IF NOT EXISTS configs (
 id INTEGER PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
 revision INTEGER NOT NULL, spec TEXT NOT NULL CHECK(json_valid(spec)),
 sha256 TEXT NOT NULL, created_at REAL NOT NULL, UNIQUE(kind,name,revision)
);
CREATE TRIGGER IF NOT EXISTS configs_no_update BEFORE UPDATE ON configs
 BEGIN SELECT RAISE(ABORT,'config revisions are immutable; insert a new revision'); END;
CREATE TRIGGER IF NOT EXISTS configs_no_delete BEFORE DELETE ON configs
 BEGIN SELECT RAISE(ABORT,'config revisions are referenced by historical runs'); END;
CREATE TABLE IF NOT EXISTS runs (
 id TEXT PRIMARY KEY, created_at REAL NOT NULL, finished_at REAL,
 status TEXT NOT NULL, target_config_id INTEGER NOT NULL REFERENCES configs(id),
 spec TEXT NOT NULL CHECK(json_valid(spec)), error TEXT,
 lease_expires_at REAL NOT NULL, hourly_price REAL, estimated_cost REAL,
 code_sha256 TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS runs_spec_immutable BEFORE UPDATE OF spec,target_config_id,code_sha256 ON runs
 BEGIN SELECT RAISE(ABORT,'run configuration snapshots are immutable'); END;
CREATE TABLE IF NOT EXISTS run_configs (
 run_id TEXT NOT NULL REFERENCES runs(id), config_id INTEGER NOT NULL REFERENCES configs(id),
 PRIMARY KEY(run_id, config_id)
);
CREATE TABLE IF NOT EXISTS resources (
 run_id TEXT PRIMARY KEY REFERENCES runs(id), pod_id TEXT, pod_name TEXT NOT NULL UNIQUE,
 owned INTEGER NOT NULL, state TEXT NOT NULL, metadata TEXT CHECK(metadata IS NULL OR json_valid(metadata)),
 cleanup_error TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS resources_active_pod ON resources(pod_id)
 WHERE state NOT IN ('deleted','detached');
CREATE TABLE IF NOT EXISTS hardware (
 run_id TEXT PRIMARY KEY REFERENCES runs(id), details TEXT NOT NULL CHECK(json_valid(details))
);
CREATE TABLE IF NOT EXISTS trials (
 id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
 workload_config_id INTEGER NOT NULL REFERENCES configs(id), ordinal INTEGER NOT NULL,
 concurrency INTEGER NOT NULL, repeat INTEGER NOT NULL, status TEXT NOT NULL,
 result TEXT NOT NULL CHECK(json_valid(result)), UNIQUE(run_id, ordinal)
);
CREATE TABLE IF NOT EXISTS metrics (
 trial_id INTEGER NOT NULL REFERENCES trials(id), name TEXT NOT NULL, value REAL NOT NULL,
 PRIMARY KEY(trial_id,name)
);
CREATE TABLE IF NOT EXISTS requests (
 trial_id INTEGER NOT NULL REFERENCES trials(id), ordinal INTEGER NOT NULL,
 details TEXT NOT NULL CHECK(json_valid(details)), PRIMARY KEY(trial_id,ordinal)
);
CREATE TABLE IF NOT EXISTS artifacts (
 run_id TEXT NOT NULL REFERENCES runs(id), path TEXT NOT NULL,
 sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, codec TEXT NOT NULL DEFAULT 'zlib',
 content BLOB NOT NULL, PRIMARY KEY(run_id,path)
);
CREATE TABLE IF NOT EXISTS events (
 id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), at REAL NOT NULL,
 kind TEXT NOT NULL, details TEXT NOT NULL CHECK(json_valid(details))
);
CREATE TABLE IF NOT EXISTS benchmark_types (
 type_id TEXT PRIMARY KEY,label TEXT NOT NULL,adapter TEXT,default_suite TEXT,
 schema_json TEXT NOT NULL CHECK(json_valid(schema_json))
);
CREATE TABLE IF NOT EXISTS gpu_catalog (
 gpu_type TEXT PRIMARY KEY,discovered_at REAL NOT NULL,metadata TEXT NOT NULL CHECK(json_valid(metadata))
);
CREATE TABLE IF NOT EXISTS sweeps (
 id TEXT PRIMARY KEY,created_at REAL NOT NULL,status TEXT NOT NULL,suite_config_id INTEGER REFERENCES configs(id),
 options TEXT NOT NULL CHECK(json_valid(options)),estimated_ceiling REAL NOT NULL,
 approved_budget REAL,pid INTEGER,heartbeat_at REAL,finished_at REAL,error TEXT
);
CREATE TABLE IF NOT EXISTS sweep_items (
 id INTEGER PRIMARY KEY,sweep_id TEXT NOT NULL REFERENCES sweeps(id),ordinal INTEGER NOT NULL,
 target_config_id INTEGER NOT NULL REFERENCES configs(id),status TEXT NOT NULL,
 reason TEXT,assessments TEXT NOT NULL CHECK(json_valid(assessments)),
 spec TEXT NOT NULL CHECK(json_valid(spec)),run_id TEXT REFERENCES runs(id),attempts INTEGER NOT NULL DEFAULT 0,
 UNIQUE(sweep_id,ordinal)
);
CREATE INDEX IF NOT EXISTS sweep_item_state ON sweep_items(sweep_id,status,ordinal);
CREATE INDEX IF NOT EXISTS trials_workload ON trials(workload_config_id,run_id);
CREATE VIEW IF NOT EXISTS replica_trials AS
 SELECT t.id trial_id,t.run_id,tc.name target,c.name workload,c.revision workload_revision,
   json_extract(tc.spec,'$.gpu_type') gpu_type,
   COALESCE(json_extract(r.spec,'$.replica_count'),1) replica_count,
   COALESCE(json_extract(r.spec,'$.gpus_per_replica'),json_extract(tc.spec,'$.gpu_count')) gpus_per_replica,
   json_extract(c.spec,'$.model') model,json_extract(c.spec,'$.dtype') dtype,
   r.code_sha256,json_extract(r.spec,'$.target.image') engine_image,
   json_extract(r.spec,'$.client_location') client_location,
   COALESCE(json_extract(t.result,'$.load_mode'),json_extract(c.spec,'$.load_mode'),json_extract(r.spec,'$.load_mode')) load_mode,
   json_extract(c.spec,'$.input_tokens') input_tokens,json_extract(c.spec,'$.output_tokens') output_tokens,
   t.concurrency concurrent_requests,t.repeat,t.status,
   json_extract(t.result,'$.metrics.output_throughput') output_tokens_per_second,
   json_extract(t.result,'$.metrics.request_throughput') requests_per_second,
   json_extract(t.result,'$.metrics.request_goodput') requests_per_second_meeting_slo,
   json_extract(t.result,'$.metrics.p50_ttft_ms') ttft_p50_ms,
   json_extract(t.result,'$.metrics.p95_ttft_ms') ttft_p95_ms,
   json_extract(t.result,'$.metrics.p99_ttft_ms') ttft_p99_ms,
   json_extract(t.result,'$.metrics.p95_tpot_ms') tpot_p95_ms,
   json_extract(t.result,'$.metrics.p95_itl_ms') itl_p95_ms,
   json_extract(t.result,'$.metrics.p50_e2el_ms') e2e_p50_ms,
   json_extract(t.result,'$.metrics.p95_e2el_ms') e2e_p95_ms,
   json_extract(t.result,'$.metrics.p99_e2el_ms') e2e_p99_ms,
   json_extract(t.result,'$.metrics.completed') completed_requests,
   json_extract(t.result,'$.metrics.failed_requests') failed_requests,
   json_extract(t.result,'$.request_rate') request_rate
 FROM trials t JOIN runs r ON r.id=t.run_id JOIN configs c ON c.id=t.workload_config_id
 JOIN configs tc ON tc.id=r.target_config_id;
PRAGMA user_version=4;
"""


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data: bytes):
    return hashlib.sha256(data).hexdigest()


class Database:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2, 3, 4):
            raise ValueError(f"Unsupported database version {version}")
        if version in (1, 2):
            # Reusing an attached Pod must not conflict with its historical, detached runs.
            # Keep uniqueness only for active executions, not for all time.
            self.conn.executescript("""
                BEGIN IMMEDIATE;
                ALTER TABLE resources RENAME TO resources_old;
                CREATE TABLE resources (
                  run_id TEXT PRIMARY KEY REFERENCES runs(id), pod_id TEXT,
                  pod_name TEXT NOT NULL UNIQUE, owned INTEGER NOT NULL, state TEXT NOT NULL,
                  metadata TEXT CHECK(json_valid(metadata)), cleanup_error TEXT);
                INSERT INTO resources SELECT * FROM resources_old;
                DROP TABLE resources_old;
                DROP VIEW IF EXISTS replica_trials;
                PRAGMA user_version=3;
                COMMIT;
            """)
        # SQLite versions differ on json_valid(NULL); optional metadata must be
        # explicitly nullable, including databases staged onto another host.
        resource_schema = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='resources'"
        ).fetchone()
        if resource_schema and "CHECK(json_valid(metadata))" in resource_schema[0]:
            body = SCHEMA.split("CREATE TABLE IF NOT EXISTS resources (", 1)[1].split(";", 1)[0]
            self.conn.executescript(
                "BEGIN IMMEDIATE; CREATE TABLE resources_nullable (" + body + ";"
                "INSERT INTO resources_nullable SELECT * FROM resources;"
                "DROP TABLE resources; ALTER TABLE resources_nullable RENAME TO resources; COMMIT;"
            )
        self.conn.executescript(SCHEMA)
        # Compatible v4 view refresh: old collectors keep using the same columns.
        # New mixed runs must not inherit a replica-wide closed-loop label.
        view = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name='replica_trials'"
        ).fetchone()[0]
        if "$.request_rate" not in view:
            body = SCHEMA.split("CREATE VIEW IF NOT EXISTS replica_trials AS", 1)[1].split(
                ";\nPRAGMA user_version", 1
            )[0]
            self.conn.executescript(
                "BEGIN IMMEDIATE; DROP VIEW replica_trials; CREATE VIEW replica_trials AS"
                + body
                + "; COMMIT;"
            )

    def seed(self):
        for value in defaults():
            if not self.conn.execute(
                "SELECT 1 FROM configs WHERE kind=? AND name=?", (value["kind"], value["name"])
            ).fetchone():
                self.save_config(value)

    def save_config(self, value):
        value = validate(value)
        spec = encode(value)
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            row = self.conn.execute(
                "SELECT * FROM configs WHERE kind=? AND name=? ORDER BY revision DESC LIMIT 1",
                (value["kind"], value["name"]),
            ).fetchone()
            if row and row["spec"] == spec:
                return dict(row)
            revision = row["revision"] + 1 if row else 1
            cur = self.conn.execute(
                "INSERT INTO configs(kind,name,revision,spec,sha256,created_at) VALUES(?,?,?,?,?,?)",
                (value["kind"], value["name"], revision, spec, digest(spec.encode()), time.time()),
            )
        return dict(self.conn.execute("SELECT * FROM configs WHERE id=?", (cur.lastrowid,)).fetchone())

    def config(self, name, kind=None, revision=None):
        clauses, args = ["name=?"], [name]
        if kind:
            clauses.append("kind=?")
            args.append(kind)
        if revision is not None:
            clauses.append("revision=?")
            args.append(revision)
        rows = self.conn.execute(
            "SELECT * FROM configs WHERE " + " AND ".join(clauses) + " ORDER BY revision DESC", args
        ).fetchall()
        if not rows:
            raise ValueError(f"Unknown configuration: {name}")
        if not kind and len({r["kind"] for r in rows}) > 1:
            raise ValueError("Ambiguous configuration; specify kind")
        return dict(rows[0])

    def event(self, run_id, kind, details):
        with self.conn:
            self.conn.execute(
                "INSERT INTO events(run_id,at,kind,details) VALUES(?,?,?,?)",
                (run_id, time.time(), kind, encode(details)),
            )

    def new_run(self, target, workloads, spec, ttl_seconds, owned=True, pod_id=None):
        run_id = uuid.uuid4().hex
        now = time.time()
        spec = {**spec, "lease_expires_at": now + ttl_seconds}
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs(id,created_at,status,target_config_id,spec,lease_expires_at,code_sha256) "
                "VALUES(?,?,?,?,?,?,?)",
                (run_id, now, "planned", target["id"], encode(spec), now + ttl_seconds, spec["code_sha256"]),
            )
            self.conn.executemany(
                "INSERT INTO run_configs VALUES(?,?)", [(run_id, r["id"]) for r in [target, *workloads]]
            )
            self.conn.execute(
                "INSERT INTO resources(run_id,pod_id,pod_name,owned,state) VALUES(?,?,?,?,?)",
                (run_id, pod_id, f"sls-bench-{run_id}", int(owned), "pending"),
            )
        return run_id

    def run(self, run_id):
        row = self.conn.execute(
            "SELECT r.*,p.pod_id,p.pod_name,p.owned,p.state AS resource_state,p.cleanup_error "
            "FROM runs r JOIN resources p ON r.id=p.run_id WHERE r.id=?",
            (run_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Unknown run: {run_id}")
        return dict(row)

    def update(self, table, run_id, **values):
        allowed = {
            "runs": {"status", "finished_at", "error", "hourly_price", "estimated_cost"},
            "resources": {"pod_id", "state", "metadata", "cleanup_error"},
        }
        if not values or not set(values) <= allowed[table]:
            raise ValueError("Invalid update")
        key = "id" if table == "runs" else "run_id"
        with self.conn:
            self.conn.execute(
                f"UPDATE {table} SET " + ",".join(f"{k}=?" for k in values) + f" WHERE {key}=?",
                [*values.values(), run_id],
            )

    def artifact(self, run_id, path, data):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?)",
                (run_id, path, digest(data), len(data), "zlib", zlib.compress(data)),
            )

    def ingest(self, run_id, report):
        """One transaction; safe to collect the same partial/full report repeatedly."""
        with self.conn:
            if report.get("hardware"):
                self.conn.execute(
                    "INSERT OR REPLACE INTO hardware VALUES(?,?)", (run_id, encode(report["hardware"]))
                )
            for ordinal, trial in enumerate(report.get("trials", [])):
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO trials(run_id,workload_config_id,ordinal,concurrency,repeat,status,result) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (
                        run_id,
                        trial["config_id"],
                        ordinal,
                        trial["concurrency"],
                        trial["repeat"],
                        trial["status"],
                        encode({k: v for k, v in trial.items() if k != "requests"}),
                    ),
                )
                if not cur.rowcount:
                    continue
                trial_id = cur.lastrowid
                metrics = dict(trial.get("metrics", {}))
                price = self.conn.execute("SELECT hourly_price FROM runs WHERE id=?", (run_id,)).fetchone()[0]
                if price is not None and metrics.get("output_throughput", 0) > 0:
                    metrics["estimated_dollars_per_million_output_tokens"] = (
                        price / 3600 / metrics["output_throughput"] * 1_000_000
                    )
                if price is not None and metrics.get("images_per_second", 0) > 0:
                    metrics["estimated_dollars_per_thousand_images"] = (
                        price / 3600 / metrics["images_per_second"] * 1000
                    )
                self.conn.executemany(
                    "INSERT INTO metrics VALUES(?,?,?)",
                    [
                        (trial_id, k, v)
                        for k, v in metrics.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    ],
                )
                self.conn.executemany(
                    "INSERT INTO requests VALUES(?,?,?)",
                    [(trial_id, i, encode(r)) for i, r in enumerate(trial.get("requests", []))],
                )

    def backup(self, destination):
        destination = Path(destination).expanduser().resolve()
        if destination == self.path:
            raise ValueError("Backup destination must differ from source")
        target = sqlite3.connect(destination)
        try:
            self.conn.backup(target)
        finally:
            target.close()
