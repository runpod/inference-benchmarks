# Runpod inference benchmarks

Benchmark methodology, a standalone Python runner, and sanitized historical GPU
inference results. This is an experimental measurement toolkit, **not a certified
GPU leaderboard**. It contains text generation and image generation campaigns with
different software revisions, sampling policies, and coverage.

## Start with the results

- [Results inventory](results/SUMMARY.md): outcomes, including incomplete/failed trials.
- [Methodology and comparison rules](docs/METHODOLOGY.md).
- [Data dictionary](docs/DATA.md): JSON/CSV fields and intentional omissions.
- [Publication review](docs/PUBLICATION_REVIEW.md): checks and remaining limitations.

`results/trials.json` contains trial metadata and metrics; `results/trials.csv` is
the flat metric export. Join `run_key` and `workload_key` to `runs.json`,
`workloads.json`, and `hardware.json` for comparison controls. No raw operational
database, requests, prompts, model outputs, logs, or credentials are included.

Regenerate the inventory without cloud access or dependencies:

```sh
python3 tools/summarize_results.py
python3 -m unittest discover -s publication_tests -v
```

## Standalone runner

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and OpenSSH for paid runs.
The runner has no dependency on the original dashboard or working directory.

```sh
cd benchmarks
uv sync --frozen --group dev
uv run pytest -q
uv run bench --help
uv run bench init                 # initializes ONLY your local database
uv run bench config list
```

[Runner documentation](benchmarks/README.md) covers workload configuration, safety
limits, immutable revisions, and explicit paid execution. Its dashboard-specific
sections describe the originating system; that dashboard is not included here.
Do not run `sweep`, `start`, or `remote launch` unless you intend to rent resources.
Budgets are estimates, not provider-enforced spending caps. Never put API keys in
configs or commit local state. Model access and upstream license requirements apply.

The copied runner represents the export-time source. Historical results retain
their original code/model/image identities and **must not be relabeled as runs of
this exact checkout**. This repo supports inspection and methodology reproduction,
not bit-for-bit replay of every historical campaign. Prompt bodies and some raw
runtime/correctness evidence are intentionally withheld.

## Updating the publication

This is a one-way export, not a second controller. From this repository:

```sh
python3 tools/export_publication.py --source /path/to/original/project
python3 tools/summarize_results.py
python3 -m unittest discover -s publication_tests -v
(cd benchmarks && uv run pytest -q)
```

The exporter opens the source SQLite database in read-only mode and uses its online
backup API. The temporary snapshot is outside this repo and is deleted on exit;
no migrations or benchmark code run against the original database. Only explicit
source-file and result-field allowlists cross the publication boundary. Review the
diff and repeat the disclosure/secret checks before each push. Export updates copied
files; it does not delete unrelated files or automatically publish anything.

## Licensing

No license is granted yet for this repository's original code or result dataset;
reuse terms require a separate owner decision. Dependencies, models, and referenced
datasets retain their respective upstream licenses. No weights or source datasets
are redistributed here. Public availability alone is not an open-source license.
