"""Detached lease reaper. Recovery is also available via `bench cleanup --expired`."""

import sys
import time

from .controller import cleanup
from .db import Database
from .runpod import Runpod


def main(path, run_id):
    db = Database(path)
    api = Runpod()
    while True:
        row = db.run(run_id)
        if row["resource_state"] == "deleted" or not row["owned"]:
            return
        if row["resource_state"] == "creation_ambiguous":
            try:
                cleanup(db, api, run_id)
                return
            except Exception:
                pass  # No matching Pod yet; retain the ambiguity and try again.
        remaining = row["lease_expires_at"] - time.time()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 15))
    # Retain the resource record on failure and keep retrying instead of losing the orphan.
    for attempt in range(20):
        try:
            cleanup(db, api, run_id, expired_only=True)
            if db.run(run_id)["status"] in ("planned", "provisioning", "running"):
                db.update("runs", run_id, status="timed_out", finished_at=time.time(), error="Lease expired")
            return
        except Exception as error:
            db.event(run_id, "watchdog_cleanup_retry", {"attempt": attempt, "error": str(error)})
            time.sleep(30)
    raise RuntimeError(f"Unable to delete expired Pod for {run_id}; run bench cleanup")


if __name__ == "__main__":
    main(*sys.argv[1:])
