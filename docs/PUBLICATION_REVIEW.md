# Publication review

## Scope

Reviewed the allowlisted standalone source and sanitized historical result export.
The source project was not reorganized, migrated, restarted, or pushed. Its Git
history is not part of this repository. No paid infrastructure was provisioned.

## Checks performed

- SQLite online backup via a read-only source connection, with no runner Database
  initialization or migrations; temporary raw snapshot deleted after export.
- Source file allowlist excludes local data, credentials, operational job scripts,
  dashboard state, private plans, and original Git metadata.
- Result field allowlists exclude raw prompts/outputs, request arrays, workflow
  payloads, artifacts/logs, free-text errors, resource/Pod identifiers, registry
  authentication references, and raw host/device telemetry.
- **285 standalone runner tests passed**, using mocked cloud/SSH interfaces.
- **2 publication tests passed**: online WAL backup/export, frozen revision
  precedence, smoke labels, incomplete outcomes, secret-field omission, manifest
  hashes, and result referential integrity.
- Standalone CLI help and source lint passed in a fresh dependency environment.
- Gitleaks **8.30.1** scanned the exact staged tree. Initial matches were exclusively
  generated `run_key`/`workload_key` hashes. `.gitleaks.toml` narrowly allows their
  exact hex lengths and JSON field shapes in three result files, only for the
  generic-api-key rule. No files are wholesale excluded. Reviewed scan: **no leaks**.
- Additional staged-content scan: no matches against **11 available local
  credential/config values**, no private-key blocks, provider token patterns,
  credential-bearing URLs, user-home paths, or forbidden state-file paths.
  Credential values and local scan reports are not published.
- Retained container repositories are public upstream PyTorch/vLLM images. Retained
  model names are upstream model repositories, not internal deployment endpoints.

Two tests intentionally omitted from the original suite:
`test_image_extension_budget.py` and `test_image_fleet_controller.py`. These import
private campaign scripts from `data/jobs/`; copying those scripts would cross the
publication boundary. The standalone runner's tests are otherwise copied unchanged.

## Limits and release conditions

A clean secrets scan is not proof of absence of every sensitive value. These checks
are a disclosure review, not a penetration test or dependency/CVE audit. Historical
prices, timestamps, GPU types, model identities and runtime hashes are intentionally
public benchmark provenance. Results retain sampling/coverage caveats and are not
presented as a matched or independently certified hardware ranking.

No license has been invented on the owner's behalf. Public visibility is separate
from granting reuse rights; select code/data licenses in a follow-up owner decision.
Upstream model/data access and license terms still apply. No weights, source
datasets, prompt bodies, or generated image samples are redistributed.

## Repeat before updating

1. Run the one-way exporter and regenerate the summary.
2. Review source/result diffs, copied-file manifest, metadata, and new string values.
3. Run runner tests and publication tests.
4. Stage only intended publication files; run `python3 tools/check_publication.py`
   (optionally pass `--source /path/to/original` for private known-value comparison).
5. Materialize the Git index into a temporary directory using
   `git archive "$(git write-tree)"`; run `gitleaks dir` on that directory with this
   repository's `.gitleaks.toml`. Scan the Git history too before pushing.
6. Do not upload audit reports or raw snapshots. New allowlist fields need a new
   disclosure review. Secret scanning and review must be repeated on every refresh.
