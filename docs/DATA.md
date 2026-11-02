# Published data

- `manifest.json`: exporter/schema identity, export time, source/result file SHA256s,
  and recorded outcome counts. Hashes identify content; they are not signatures.
- `runs.json`: pseudonymous run key, status/timestamps, original code/worker hashes,
  allowlisted target, frozen workload references, replica/device counts and costs.
- `workloads.json`: content-addressed, allowlisted **frozen** workload settings.
  Where a run lacks a frozen workload entry, the immutable referenced config is
  used as a fallback. Keys hash the sanitized view, not the original config body;
  omitted prompts can make distinct original workloads share a published key.
- `trials.json`: one row per recorded workload/load/repeat outcome, including
  failures. Join `run_key` and `workload_key`; `ordinal` is unique within a run.
  `recorded` preserves sampling and failure-category metadata when available;
  `engine_config` preserves the actual engine controls when available.
  `metrics` includes only known, finite numeric scalar metrics.
- `trials.csv`: flat trial identifiers and numeric metrics; blanks mean absent,
  never zero. Use JSON for sampling labels and frozen configuration joins.
- `hardware.json`: recorded Python/Torch/CUDA versions and allowlisted device
  names, compute capabilities, and memory bytes. Missing fields remain missing.
- `SUMMARY.md`: reproducible outcome/sample-label inventory, not performance ranking.

Run keys are truncated SHA256s of internal run IDs; pseudonymization is not a
promise of anonymity. Public GPU SKU names, model repositories/commits, engine
images, timestamps, pricing, and code hashes are intentionally retained as
benchmark provenance. These are historical observations, not current price quotes.

## Excluded

No raw SQLite files, resource/Pod IDs, SSH keys or host mappings, controller
ownership state, registry-auth IDs, environment files, model tokens, request or
partial-request bodies, prompt lists, workflow JSON, raw errors, logs, artifacts,
model weights, reference images, internal plans, or original Git history.
Unknown result/config/metric keys are omitted unless explicitly allowlisted.
Raw `nvidia-smi`, package listings, CPU/topology output, and device UUIDs are omitted.

This is a compact disclosure-controlled export, not the complete audit trail.
In particular it does not contain raw evidence sufficient to re-audit image
correctness, exact prompt equivalence, provider billing, or pooled request tails.
No new numerical performance observations have been fabricated or extrapolated
by the exporter. Historical derived cost metrics remain estimates.
