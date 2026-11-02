# Methodology

## Scope and provenance

These are recorded, mixed historical campaigns, not a controlled fleet-wide ranking.
Text runs use vLLM. Image runs include ComfyUI and Diffusers execution paths; never
pool these backends. The current runner and its tests are copied without modifying
the operating benchmark system. `results/manifest.json` records export time, file
hashes, and counts. Each run retains its historical code and worker hashes, engine
image, frozen workload settings, target SKU/count, and recorded prices. Historical
source bundles themselves are not redistributed; hashes establish identity, not a
claim that the current checkout reproduces every historical implementation.

## Workloads and execution

Text campaigns include synthetic fixed-length chat, prefill/decode-heavy and long
context profiles, plus separately configured mixed-arrival traffic. Inspect each
frozen workload: defaults have changed over time. Client concurrency is not engine
batch size or scheduling capacity; queued requests can exceed engine slots. Compare
engine capacities, context limits, caching, precision, model commits, token lengths,
arrival modes, seed schedules, request counts, and repeats explicitly.

Image campaigns vary model/backend, resolution, batch, guidance, steps, execution
mode, and precision. Keep those controls fixed when attributing effects to hardware.
Delivered image throughput and latency differ from denoiser-only timing. Sample
outputs are not a quality evaluation, and a completed trial alone is not correctness
certification. Prompts, reference-image payloads, and image artifacts are omitted
from this publication, so exact prompt-controlled replay is not available here.

## Sampling and incomplete coverage

Historical text policies include fixed counts, shrink-to-budget, full-or-smoke,
coverage-first, and cliff-aware search. Policies are not interchangeable:

- Full-or-smoke fallback can retain only two requests at low concurrency or one
  occupancy wave at higher concurrency. These support coarse capability/timing
  observations, not stable tail-latency claims.
- `reduced_sample`, planned/selected request counts, and `sample_class` are preserved
  when recorded. Missing historical labels/counts mean **unknown**, not full samples.
- Pilots and extrapolations are not completed measurements. Failed, unavailable,
  skipped, truncated, and time-budget-limited cells are not measured zeros.
- Recorded repeats need not represent independent physical-machine samples.
  Do not infer fleet-wide uncertainty or confidence intervals from them alone.

The [runner reference](../benchmarks/README.md) explains pilot admission, time
budgets, hard timeouts, and historical sampling-policy changes in detail. The
[four-point grid analysis](../benchmarks/TEXT_GRID_COMPARISON.md) includes explicitly
labeled interpolations and planning estimates; those are not new measurements.

## Metrics and comparisons

Metric names preserve source units: `_ms` milliseconds, `_seconds` seconds,
`_bytes` bytes; token throughput is tokens/second and image throughput images/second.
`duration` is benchmark serving duration, not GPU kernel-active time or total Pod
wall time. `hourly_price` and estimated per-output costs are recorded estimates,
not audited invoices. Startup/download/idle time can make full-job economics differ
substantially from steady-state throughput-based cost.

Do not average trial percentiles and call the result a pooled request percentile.
This publication omits request arrays, so pooled percentile/goodput recomputation
is not possible. Retained historical p95/p99 values can be statistically weak for
small samples. Do not turn an aggregate latency threshold into request-level goodput.

Filter to compatible, completed measurements with adequate sample sizes before
comparing GPUs. Match exact GPU/slice count and engine parallelism: a MIG slice is
not the parent card. No automatic ranking is provided because pooling these
campaigns would hide meaningful controls and gaps.

## Implemented versus proposed coverage

See [workload coverage](../benchmarks/WORKLOAD_COVERAGE.md) for the broader taxonomy.
That document includes proposed/unimplemented families. Their presence is not a
claim that this repository has measurements for them. The historical document may
lag the current runner's additional adapters; frozen result metadata is authoritative.
