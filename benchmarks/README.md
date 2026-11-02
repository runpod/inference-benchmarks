# Runpod inference benchmarks

A Python controller on your machine rents **Pods**, starts a worker inside each Pod,
collects results into **SQLite**, then deletes its Pods. No Serverless endpoints.
The Next.js dashboard does not manage executions. Its `/benchmarks` page reads recorded
text trials server-side, directly from SQLite in read-only mode (including committed WAL data).
From the repository root, it defaults to `benchmarks/data/benchmarks.sqlite3`; set
`BENCHMARK_DB_PATH` to override this path. The dashboard needs Node.js 22.13+ with
`node:sqlite` support. Use Refresh on the page to load new results; it never starts paid runs.

## First-class, fire-and-forget sweeps

From `benchmarks/`, one command selects a GPU and runs the complete stored suite:

```sh
uv run bench sweep --gpu "$GPU" --type image_gen --detach --yes --budget 3
# Text suite: --type text_gen
uv run bench jobs
uv run bench job SWEEP_ID
uv run bench results RUN_ID --json
```

`--gpu` accepts any live catalog SKU. `--type` is a benchmark category, **not a model**:
`text_gen` dispatches to vLLM, `image_gen` to ComfyUI. Models and parameters live in
workload definitions. `image_processing` has a registered schema but no implemented
adapter yet; it is explicitly blocked, never routed through image generation.

```sh
uv run bench types
uv run bench sweep --gpu "$GPU" --type image_gen --suite image-smoke --detach --yes --budget 1
# Model/parameter overrides become validated, immutable SQLite workload revisions:
uv run bench sweep --gpu "$GPU" --type image_gen --model ORG/MODEL \
  --params '{"checkpoint":"weights.safetensors","steps":30}' --detach --yes --budget 3
```

Default SQLite suites: `text` (four Qwen context/concurrency profiles), `image`
(SD1.5 512×512 plus SDXL 1024×1024), and `image-smoke` (one warmup + one SD1.5 image).
Suites themselves are `kind=suite` configs, editable through `config show`/`config put`.
A suite references workload names/revisions; **preparation freezes the selected config
IDs, resolved model commits, engine digest, source code, and catalog metadata**.

Prepare an entire catalog without spending anything:

```sh
uv run bench prepare --all-gpus --type image_gen --suite image --max-minutes 30
uv run bench job SWEEP_ID --json
# Only after reviewing coverage/cost and explicitly approving an adequate budget:
uv run bench start SWEEP_ID --budget DOLLARS --yes
```

`prepare` never allocates a Pod. Every catalog entry gets a row, including entries
blocked by missing pricing/VRAM metadata or incompatible runtime. The current image
runtime is CUDA: AMD requires a separately validated ROCm runtime and remains blocked.
Per-profile VRAM floors are **estimates**, not proof a workload fits. Small cards can
run the light profile while SDXL is blocked; those gaps remain visible in assessments.
Catalog preflight candidates are labeled `eligible_unvalidated`, not GPU-certified.

`start` is detached by default. The persistent queue defaults to one active Pod;
`bench start ID --parallel 4 --budget DOLLARS --yes` enables up to four concurrent
Pods (supported range: 1–8). Parallelism is persisted for resume. Each slot has its
own SQLite connection and API client. The queue retries only capacity failures with
bounded backoff, checkpoints into SQLite, and stops new admissions on unresolved
cleanup. Cancellation signals all active slots to collect and clean up their Pods. Logs are next to the DB under `jobs/ID.log`.
A file lock prevents duplicate dispatchers. `start ID --yes --budget ...` recovers an
interrupted dispatcher without rerunning completed items or allocating a replacement
for an existing run; it attempts to recover that run's results and cleanup first.
`start ID --retry-unavailable --yes --budget ...` explicitly retries no-capacity items.
`cancel ID` signals the verified local dispatcher, whose controller cleans up its Pod.

**Detached is not laptop-independent:** it survives closing the terminal, but the
controller host must remain awake/online. Use an always-on VM for multi-hour fleet
sweeps. A sleeping/offline controller cannot collect results; provider TTLs still
bound Pod lifetime. Recovery never extends an existing lease. Budget checks are
conservative estimates, not provider-enforced dollar caps.

## Quick start

From this directory:

```sh
uv sync --group dev
# If necessary: source ~/.zshrc
# RUNPOD_ORGS_API_KEY or runpod_orgs_api_key must be exported; never put it in configs.
uv run bench init
uv run bench config list
uv run bench gpus --match MIG
# Pass exact SKU(s) from `bench gpus`; there is no hardcoded GPU list.
GPU='paste an exact Runpod GPU SKU here'
OTHER_GPU='paste another exact Runpod GPU SKU here'
uv run bench plan --gpu "$GPU" --gpu "$OTHER_GPU" -w qwen8b-chat -w qwen8b-context16k -w qwen8b-context32k

# Paid, sequential, one Pod per target, all workloads on that Pod.
uv run bench sweep --gpu "$GPU" --gpu "$OTHER_GPU" \
  -w qwen8b-chat -w qwen8b-prefill8k -w qwen8b-context16k -w qwen8b-context32k \
  --max-minutes 45 --budget 3 --yes

uv run bench status
uv run bench results
uv run bench status RUN_ID
uv run bench logs RUN_ID
uv run bench cleanup --expired
uv run bench backup backup.sqlite3
```

Default database: `data/benchmarks.sqlite3`, relative to the working directory.

### Time-bounded text measurements

`measurement_budget_seconds` (optional, default disabled) bounds pilot plus sampling
for one fixed-concurrency engine capacity, across its configured repeats. It does
not include server startup/model download; the Pod's independent total lease also
bounds those phases. The pilot is capped at **40% of the measurement budget**
(120 seconds for a 300-second budget), not an unrelated fixed 60-second cutoff.
For fixed-length tests it uses the **same concurrency and token lengths**, with
one occupancy wave: `min(requests, max(warmup_requests, concurrency))`. There is no
second wave just for admission. Testing C512 still needs 512 requests to occupy
512 slots; those are not silently replaced by a lower-concurrency probe.

The vLLM result's actual benchmark `duration` is scaled to the full requested count
and repeats with a 15% safety margin. Non-serving wall time (client startup, test
prompt, tokenization, etc.) is charged **once per measured repeat**, not multiplied
by the sample-count ratio. Raw duration, wall time, overhead, estimator version,
margin and decision are retained in pilot evidence. Missing/invalid durations do
not produce a guessed estimate. If projected measurements cannot fit the remaining
budget, sampling is skipped. This is an operational admission rule, not a guaranteed
runtime prediction; actual measurements retain the absolute deadline.

The original two-wave/wall-time estimator is superseded. Its saved skips remain
historical diagnostic evidence and are not reclassified as completed measurements.

Both pilot and measurement commands have hard timeouts. A skip/timeout is stored as
`skipped_time_budget` with counts, budget and pilot evidence, **without throughput
metrics**. Incomplete samples are not promoted to measurements; configured sample
counts/repeats are never reduced. The adaptive search may try higher concurrency
after a skip, resets plateau evidence across skips, and reports time-budget-limited
coverage rather than a measured saturation frontier. Previously completed repeats
remain valid if a later repeat times out.

Mixed-arrival profiles use a shared budget across their rates/repeats, rejecting
an offered schedule whose arrival duration alone cannot fit and bounding actual
execution; they do not extrapolate serial pilot throughput to an arrival-rate test.
A one-hour total GPU lease can therefore leave later profiles unmeasured. Compare
only completed, compatible measurements; time-budget skips are censored coverage,
not evidence that hardware is slow or incapable.

### Coverage-first text sampling

For broad coverage, explicitly set `measurement_sample_policy: "shrink_to_budget"`,
`coverage_first: true`, and `measurement_budget_seconds: 900`. The historical
`fixed` policy above is still the default for existing configurations.

The pilot estimates how many requests fit **all configured repeats** within the
remaining point budget. Counts may shrink, but model, precision, input/output lengths,
target concurrency, request rate, seed schedule and number of repeats do not change.
Closed-loop counts never fall below the target concurrency (one occupancy wave);
open-loop points retain their offered rate and use at least one request. Each repeat
can shrink further based on the speed of the preceding real measurement. Counts
never grow above their planned values. A pilot timeout falls back to attempting a
minimum-sized real measurement instead of vetoing the point.

Every result records planned and selected requests, completed requests, policy,
`reduced_sample`, and an explicit sample-size warning. Small samples support coverage
and coarse throughput/latency observations, **not precise tail-latency claims**.
The dashboard's recorded-trial table exposes completed/planned counts and reduction
labels. Pilot evidence remains separate from completed measurements.

Coverage-first execution visits the complete configured concurrency/rate grid:
plateau and low-throughput early stops are disabled. Points are scheduled round-robin
across workload profiles, so C1 of later profiles is attempted before one profile
uses hours exploring higher concurrency. `report.json` retains an explicit coverage
plan with pending/running/completed/incomplete point status. Mixed-arrival rates get
independent point budgets, not one shared budget for all rates.

The current fleet policy uses **one GPU Pod per exact GPU type, no capacity retries,
and an eight-hour total lease** including provisioning/setup. This bounds cumulative
execution for each type in this sweep; restarting its existing dispatcher recovers
owned runs rather than granting a fresh eight-hour attempt. Do not prepare a second
sweep to bypass that cap. A type that is unavailable is recorded as such.

This is coverage-first, not a guarantee that every cell can complete: unsupported
hardware, OOM/setup errors, a minimum sample that itself exceeds 15 minutes, or the
eight-hour type cap produce explicit incomplete coverage. A measurement timeout is
`incomplete_time_budget`, not `skipped_time_budget` and never a fabricated result.
No new allocation or lease extension is used to conceal those gaps.

### Cliff-aware text search

Opt in with `cliff_aware: true`, `coverage_first: true` and adaptive budgeted
sampling for closed-loop profiles. Keep the full concurrency ceiling: cards discover
an observed useful-performance boundary independently. Mixed arrival-rate tests
remain separate and must use `cliff_aware: false`.

A validated occupancy-wave pilot detects an absolute output tok/s floor, a drop below
`saturation_collapse_ratio` of the best earlier measured throughput, or
`saturation_plateau_steps` consecutive improvements below `saturation_min_gain`.
A candidate boundary forces two requests per repeat (at least one per concurrent
slot), preserving token lengths, precision and repeats. Completed measurement,
not the pilot alone, must confirm the throughput stopping reason. If it contradicts
the pilot, continue the search. An unexpected measured cliff can also stop the search.

Persist actual counts, pilot candidates, measured history and `stopping_reason`;
mark later coverage-plan entries `truncated`, not completed or measured zeros.
Only that profile stops. A deadline or point failure also truncates its remaining
levels with an explicit incomplete/failure reason; retain any completed repeats.
Smoke samples do not establish reliable tail latency, and this boundary is not
proof of the GPU's absolute hardware capacity. Historical policies are unchanged.

### Full-or-smoke fallback (saved policy; no automatic restart)

Use `measurement_sample_policy: "full_or_smoke"` with `coverage_first: true` and
`measurement_budget_seconds: 900` when the goal is to get a small observation at
expensive points instead of filling the whole budget with a reduced sample.

- If the full planned count/repeats fit the pilot prediction, retain them.
- Otherwise use **two requests per repeat** at low concurrency. For closed-loop
  concurrency above two, use **one request per concurrent slot** instead; a C128
  measurement cannot honestly be performed with two requests. Never exceed the
  planned count, even if it was already less than two.
- The same rule applies to mixed-rate arrival-duration limits and repeat refitting.
  A pilot timeout falls back to this tiny count, not to a time-based skip.
- Keep the configured repeats, token lengths, precision and arrival rate unchanged.
  Fifteen minutes remains a hard point ceiling, **not a duration target**. There is
  no middle-sized "largest fitting count" in this policy. `shrink_to_budget` remains
  available explicitly for reproducing the older behavior.
- Mark fallback trials `sample_class: "smoke"`, retain planned/selected/completed
  counts, and label them **smoke sample** in the recorded-trial UI. These are
  capability/coarse timing observations, not reliable tail-latency estimates.

This does not remove startup, pilot or per-request generation time. Two long
completions can still take minutes; if even the minimum cannot complete before the
hard deadline, record `incomplete_time_budget`, never invented throughput. The
saved GPU-type lease remains eight hours, with no automatic capacity retries.
Previous runs/results remain immutable; changing the saved policy does not restart
a stopped controller or rewrite old reduced-sample measurements.

### Remote CPU controller (laptop may disconnect)

The remote wrapper stages a **fresh prepared sweep** on a Runpod CPU Pod and invokes
exactly the existing sweep dispatcher there. GPU workers, budgets, leases, watchdogs,
and collection are unchanged. Only the wrapper provisions the CPU and synchronizes
its database; the local dashboard continues to read your local SQLite database.

```bash
# Run from benchmarks/; prepare and inspect your sweep as usual first.
uv run bench remote launch SWEEP_ID --budget 20 --cpu-hourly-limit 0.10 --yes
# Once launch returns successfully, your laptop can disconnect.
uv run bench remote status                     # local ownership records, no network
uv run bench remote pull                       # reconnect/pull ALL retained CPU controllers
uv run bench remote pull SWEEP_ID              # or one; safe during execution and repeatable
uv run bench remote resume SWEEP_ID --yes       # restart an interrupted existing dispatcher
uv run bench remote release SWEEP_ID --yes      # final pull, verify cleanup, delete CPU
```

`--budget` approves the existing estimated GPU ceiling; CPU charges are **additional**.
`--cpu-hourly-limit` checks the allocated CPU rate before dispatch, not a total-dollar
cap. The CPU remains billable after the sweep finishes until explicitly released.
The default is 2 vCPUs, `cpu3g`, and a 30 GB Pod volume; override with `--vcpus`,
`--cpu-flavor` (comma-separated fallback flavors supported), and `--volume-gb`. Account for result artifacts when sizing storage.
The CPU volume survives Pod restarts, **not Pod deletion**. There is deliberately no
CPU auto-delete/TTL that could discard results before your laptop reconnects.

Sync is explicit (`remote pull`), not a hidden network operation on every local CLI
command. Pull takes an online SQLite backup remotely, validates original config/item
identity, and atomically merges only that sweep's runs, trials, requests, metrics,
artifacts and events. Trial integer IDs are remapped; repeat pulls do not duplicate
results. Other local runs and newer local configs are left alone. Snapshots and bounded
controller logs are retained under `data/remote/SWEEP_ID/`. Keep using the **originating
local database**; importing into an unrelated database is intentionally rejected.

A dedicated GPU-worker SSH key is generated on the CPU; your local private SSH key
is never uploaded. The API credential (and optional `data/private/hf_token`) is staged
over SSH into mode-0600 files, never provider env or CLI arguments. This CPU is a
trusted controller host with account access: use appropriate scoped credentials.
Code plus `uv.lock` are staged and installed with `uv sync --frozen --no-dev`.

A recorded remote sweep cannot be started or cleaned up locally while its CPU owns
it. Failed/ambiguous CPU creation is recorded **before** the API call and never blindly
retried: inspect `remote status` and the provider's exact recorded Pod name. An explicit
provider capacity rejection with no matching Pod is marked `rejected`, allowing an
explicit launch retry with a different CPU flavor. A staging
failure retains the CPU for inspection rather than risking deletion of results;
manual recovery may be needed. `release` refuses an active/interrupted sweep or
unresolved GPU cleanup. Cross-host `after_sweep` dependencies and migration of an
already-started local sweep are not supported. This path has unit/mocked coverage;
run a small paid smoke test before a long unattended fleet run.

CPU provisioning API reference: https://docs.runpod.io/api-reference/pods/POST/pods
Use the global `--db PATH` **before** the command to choose a stable absolute location.
From the repo root, for example:

```sh
uv run --project benchmarks bench --db benchmarks/data/benchmarks.sqlite3 status
```

The default SSH key is `~/.ssh/id_ed25519`; override with `--ssh-key PATH`.
Both the private key and matching `.pub` must exist. Only the public key is injected
into Pods. OpenSSH must be installed locally. New Pod host keys use TOFU (`accept-new`)
and a dedicated `known_hosts` next to the database; changed keys are rejected.

## Dashboard measurement model

See [GPU workload coverage](WORKLOAD_COVERAGE.md) for additional use cases
(embeddings/reranking, speech, multimodal inference, training, scientific computing,
and rendering), their useful metrics and comparison controls, plus text-serving
patterns such as RAG, agents, batch processing, and multi-LoRA serving. The reference
distinguishes existing adapters from unimplemented coverage.

The `/benchmarks` page separates **Text gen** from **Image / media gen**. The media
selector distinguishes diffusion, image processing, video generation, and video
processing. Only text and experimental diffusion adapters currently exist; unsupported
families show instrumentation requirements, never borrowed or synthetic results.
`lib/benchmark-catalog.ts` defines this measurement taxonomy and coverage, not execution
configuration. SQLite remains the source of truth for workload revisions and results.

The image tab includes delivered throughput/latency and images-per-dollar trade-offs,
GPU-timed denoiser seconds/step versus batch, fixed-batch log-log resolution scaling,
per-trial stage breakdowns, and a batch/resolution memory-feasibility grid. Filters
preserve model commit, backend, precision, guidance, prompts, scheduler, code and engine
identity; curves additionally retain run/device count and execution track. Resolution
and batch vary only in their respective sweeps. Points are individual trials, not
pooled request distributions or a statistically established hardware frontier.

Diffusers batch size comes from each recorded point, not the profile's initial batch.
Delivered images/s uses total images divided by total retained-call E2E time; core
images/s remains separate. Latency is time to deliver the batch, not latency divided
by batch size. p99 is read when recorded, otherwise linearly interpolated from retained
call timings, with sample counts exposed. Tiny diagnostic samples do not establish
stable tail latency. Missing retained timings mean missing delivered throughput, not
a substitution of core throughput. Legacy ComfyUI throughput is polling-inclusive.
Cost efficiency uses recorded whole-replica hourly price and excludes startup/idle.
MIG normalization counts slices as allocated devices, not parent GPUs.

Memory cells report worker CUDA-device peak allocated GiB (reserved on hover), not
inferred total multi-GPU VRAM. Only structured CUDA OOMs are labeled OOM. Stage bars
contain medians on mixed GPU/CPU clocks; their sum is not median E2E. Pixel-steps/s
uses configured steps and core throughput; it is not FLOPs, quality, or a universally
comparable cross-architecture score. Environment and machine differences can still
confound observed comparisons even when model/code/container identities match.

Serving-load/SLA-goodput, paired optimization speedups, and quality/speed panels
explicitly report missing instrumentation rather than fabricate plots. They require
new adapters or paired/evaluated datasets; static sweeps cannot provide production
serving capacity, and PNG sanity samples are not quality scores. Video stays a
separate unimplemented family, requiring frame/duration-normalized metrics.

Text's primary chart is the observed throughput/latency trade-off: output tokens/sec
per allocated device versus p95 TPOT, one point per concurrency and execution version.
Points preserve GPU count, engine image, resolved model commit, and harness code hash.
Repeated trials use median throughput and median trial p95 TPOT; these are **not pooled
request percentiles**. Outlined nondominated observations are not a statistically
established hardware frontier. A MIG slice counts as one allocated device, not its parent.
The chart does not infer per-user speed by dividing aggregate throughput by concurrency.

Optional p95 TTFT / TPOT limits filter individual completed trials **before** aggregating
cost. Missing required latency never qualifies. This is a trial-summary gate, not
request-level goodput, a realistic traffic mix, or a production SLO guarantee. A cost
plot versus concurrency uses the same target-qualified trials as the table. Cost
retains its recorded unit: dollars per million output tokens for text and dollars per
thousand images for diffusion. Startup/idle cost is outside these steady-state estimates.

The text diagnostics section adds p95 TTFT versus input length, and selectable p95
TPOT/ITL versus concurrency and input context length. These sweeps span the selected
model/controller version across workload profiles, independently of the workload
selector. Output length is fixed by a separate selector; input-length plots also fix
concurrency. Engine/cache settings (including maximum model length), precision, model
commit, device count and load mode remain separate series. Dots do not interpolate
unmeasured conditions. Concurrency is not measured engine batch size, and initial
input length is not instantaneous decode context occupancy.

For the selected workload, TTFT/E2E p50/p95/p99 plots show medians of trial percentiles,
with trial/request sample counts on hover. Optional recorded goodput is plotted only
when collection-time TTFT/TPOT SLOs are present; distinct targets remain separate.
Changing cost-target inputs does not retrospectively recompute request goodput.
Current runs are synthetic closed-loop load, not realistic arrival/mixed-length traffic.
The KV-cache panel explicitly reports missing instrumentation: it does not substitute
total process memory or context × concurrency estimates for measured KV bytes or a
VRAM exhaustion boundary. No additional paid benchmarks are launched by these views.

Each family also has a measurement map distinguishing collected scalar metrics,
missing visualization/sweep coverage, and instrumentation still needed. In particular,
KV bytes, scoped peak VRAM, per-stage GPU spans, MFU, quality scores, LoRA swaps,
multi-GPU scaling, and NVENC/NVDEC utilization must not be inferred from unrelated
throughput or process-memory measurements. Future adapters need pinned task-specific
inputs and explicit units (for example, output-video-seconds rather than token prices).
The dashboard never provisions resources or mutates benchmark configurations.

## Central benchmark matrices

`kind=matrix` configs are the central, versioned definitions for a benchmark type.
Inspect or edit them in one place—no second YAML/JSON manifest to drift from SQLite:

```sh
uv run bench matrix text_gen
uv run bench matrix text_gen --json       # expanded profiles + cached per-SKU exclusions
uv run bench matrix text_gen --live       # inspect against the live catalog; never rents
uv run bench config show text_gen         # authoritative editable definition
# Pipe edited JSON into: uv run bench config put
uv run bench matrix image_gen
```

A matrix has shared `parameters` plus named `rows`, each with profile references,
explicit parameter overrides and placement requirements (GPU count/TP, per-device
VRAM, full GPU versus MIG, and supported generations). Expansion merges profile → shared parameters → row
parameters, validates every resulting workload, and saves immutable workload revisions
when prepared. Matrix IDs, expanded config IDs, model commits, runtime digest and code
are frozen in the prepared sweep. Profile selectors can use `name@revision` to pin
historical templates. `init` only inserts missing matrices, never overwrites edits.

The default `text_gen` matrix uses **four concurrency points: 1, 8, 48, 128**,
with 512 requests at C1, 1,024 at the other points, and two repeats.
Mixed-arrival trials retain their separate offered-rate grid and 1,024 requests:

| Row | GPU/TP | Profiles | Eligibility floor |
|---|---|---|---|
| Qwen3-8B BF16 | 1 / TP1 | chat, prefill8k, decode4k, context16k, context32k | 18GB |
| Qwen3-32B BF16 | 2 / TP2 | same five lengths | two full devices ≥48GB each |
| Qwen3-32B-FP8 | 1 / TP1 | chat, context32k | ≥48GB, Hopper/Blackwell, compute capability ≥90 |
| Qwen3-30B-A3B BF16 MoE | 1 / TP1 | chat, decode4k | ≥72GB |
| Qwen3-8B BF16 mixed | 1 / TP1 | filtered ShareGPT, 0.5–32 offered req/s | 18GB |

Every fixed-length profile visits the explicit **1, 8, 48, 128** grid (no adaptive
doubling), uses chat templates with thinking explicitly disabled, and disables prefix caching.
This removes 60% of planned fixed-length trials versus the ten-level C1–512 grid;
actual runtime savings depend on samples, failures and previously truncated searches.
FP8 is a separate checkpoint, never an automatic fallback. KV cache dtype remains
`auto` and is recorded separately from weight precision. GPU topology is captured;
TP2 eligibility does not guarantee fast interconnect or available two-GPU capacity.
On 24–32GB cards only the six 8B profiles are eligible. Exclusions remain inspectable.
The legacy `--type text_gen` / `text` suite is separate; select the central matrix
explicitly with `--matrix text_gen`. Existing SQLite matrices need an explicit new
revision; updating source defaults alone does not replace a saved matrix.

```sh
uv run bench prepare --matrix text_gen --gpu "$GPU" --disk-gb 200 --max-minutes 2880
uv run bench start SWEEP_ID --parallel 8 --budget DOLLARS --yes
# Queue a follow-up behind an existing sweep, without increasing concurrent Pods:
uv run bench start FOLLOWUP_ID --after EXISTING_ID --parallel 8 --budget DOLLARS --yes
```

An `--after` dispatcher stays in `waiting`, heartbeat-only, and allocates nothing
until the predecessor finishes (`completed` or `partial`) and its owned resources
are cleaned up. An interrupted predecessor must be resumed before the follow-up can
proceed. Dependencies persist across dispatcher restarts; `cancel` also handles
waiting dispatchers. The predecessor and follow-up have separate spending estimates.

### Adaptive text saturation

`saturation_max_concurrency` enables doubling from the single configured starting
concurrency up to that cap (at most 512, no larger than the default request count).
`requests_by_concurrency` supplies explicit per-level counts for either an adaptive
search or an explicit closed-loop grid (default C1: 512); unlisted levels use
`requests` (1,024). Explicit-grid keys must belong to `concurrency`; adaptive keys
must be power-of-two levels within the cap. Each scheduled count must be at least its
concurrency and no larger than the default. Actual counts are recorded in trial
metadata, metrics and search history. Prepared/running specs are immutable: changing
this policy requires a revised sweep, never relabeling unfinished old trials.
New text configs default to `engine_capacity_policy: "fixed"`: sweep **client
concurrency**, without changing server settings. Requests above `max_num_seqs` can
queue; this is production-serving load, not a claim that every client request runs
on the GPU simultaneously. Frozen specs without this field and explicit
`"match_concurrency"` configs retain the historical per-level engine-capacity search
and restarts. Prepared/running sweeps are never changed in place.

Before allocation, resolve a separate `text_engine_plan`. Compatible profiles of
one model/revision/precision share the **largest profile context limit** and largest
explicitly configured `max_num_seqs` in that group. Keep the batched-token budget
fixed (`max_num_batched_tokens`, default 8192) and chunked prefill explicit (default
true). Different memory utilization, prefix-caching, seed, KV dtype, quantization,
token budget or chunked-prefill settings are separate engine groups. One worker
owns at most one engine; finish each compatible group before loading another model.
Profiles within a group retain breadth-first concurrency ordering and cliff rules.

Paged KV allocation does not reserve `max_model_len` tokens for every sequence.
Nevertheless the largest context must fit startup checks and graph/activation/KV
headroom. **Capacity is an explicit configuration choice, not an automatic
KV-derived estimate:** choose a per-card cap before the sweep, with headroom; do not
use the 32K simultaneous-sequence estimate as a universal cap on short chat traffic.
The planner does not silently tune settings or inflate capacity to C512.

Start/warm the engine once and reuse its exact launch command. Keep a short pilot
at each workload level. A point may reuse the engine only after all its benchmark
requests completed, so the previous queue is drained. Failed clients, deadlines,
cancellation, dead engines and incompatible settings close the engine; successful
subsequent work can start a fresh one with the same fixed configuration. If prefix
caching is enabled, reset it before the next point's warmup to avoid cross-point
cache inheritance; the current cache-disabled methodology remains unchanged.

Record actual `engine_config`, `engine_instance_id`, reuse/start phases and client
concurrency separately; preserve per-point pilot/log/result files. The dashboard
compares actual shared context/capacity/token-budget settings rather than treating
profile context limits as executed engine settings. Smoke samples remain coarse
observations, not steady-state production or reliable tail-latency measurements.

This follows the server/load separation in the official
[vLLM sweep CLI](https://docs.vllm.ai/en/latest/cli/bench/sweep/serve/) and
[AIPerf concurrency sweep example](https://github.com/ai-dynamo/aiperf/blob/main/docs/comprehensive-llm-benchmarking.md).
Its [control-hook guide](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/benchmark-control-hooks.md)
separates per-cell cache reset, warmup and measured profiling without restarting
an unchanged inference server.

The default stopping rule is two consecutive doublings with <5% gain over the prior
best median output throughput. Failure/OOM/timeout stops the profile and remains a
failure, not a successful saturation finding. Reaching the concurrency cap is labeled
`concurrency_cap`, not saturation. Decisions/history are checkpointed in each run's
`report.json` artifact under `saturation`. This is an empirical throughput heuristic,
not a statistically established maximum, latency-constrained goodput, or an SLO.
The matrix also has a **25 aggregate output tok/s floor** and **50% of prior-best
throughput collapse cutoff**, applied to every GPU, including MIG. A warmup probe
can skip the full 1,024-request batch. During warmup and fixed-length trials, the
[vLLM generated-token counter](https://docs.vllm.ai/en/v0.11.0/usage/metrics.html)
is checked every ~10 seconds: after a three-minute grace, a two-minute rolling
window below either applicable threshold aborts the client and stops that profile.
Repeated missing monitoring metrics fail closed rather than running an unguarded
multi-day trial. Decisions use `low_throughput_cutoff`; skipped/aborted probes are
not completed trials or proof of saturation. Full raw results are retained as
`.json.gz` artifacts; compact reports retain request lengths/timings but not duplicate
all token-level ITL arrays. This keeps 1,024×4,096-token reports collectable.

Mixed runs use pinned ShareGPT first-turn prompts/reference output lengths, filtered
to input ≤1,024 and total ≤2,048 tokens, without oversampling. Gamma arrivals have
burstiness 0.5 and **no client concurrency semaphore** (engine capacity 64). Dataset
commit and source/corpus hashes are recorded. Mixed measurements have separate
request-rate tables, never fixed-length concurrency curves. The live aggregate
throughput floor is not applied to intentionally low offered rates; the unthrottled
warmup still checks service capability. This is not a production replay or quality test.

At high concurrency, 1,024 requests may still provide few waves of sustained load.
The full Pod lease remains a hard lifetime bound and can expire before all profiles
finish; partial results are retained.

## Configurations live in SQLite

There are **no YAML workload files or authoritative manifest files**.
`config put` accepts JSON on stdin, validates it, and inserts a new immutable database
revision. Running `init` again only inserts missing defaults; it never resets edits.

```sh
# Increase the 32K test's request count. Both old and new revisions remain queryable.
uv run bench config show qwen8b-context32k | \
  python3 -c 'import json,sys; c=json.load(sys.stdin); c["requests"]=64; print(json.dumps(c))' | \
  uv run bench config put

uv run bench config show qwen8b-context32k --revision 1
# Select historical config revisions when rerunning:
uv run bench plan --gpu "$GPU" -w qwen8b-context32k@1
```

**GPU types are arguments, not source-code defaults.** `--gpu` accepts any exact SKU
returned by the live Runpod catalog, including new cards without a code change. VRAM
and MIG expectations are derived from that catalog, not from a hardcoded card table.
Each invocation saves/idempotently reuses the resulting target config in SQLite for
reproducibility. Repeat `--gpu` to select multiple types. Set `--gpu-count`, `--cloud`,
`--min-vcpu`, `--min-ram-gb`, and `--image` as needed.

Optional `--target NAME@REVISION` reuses an existing SQLite target profile; it is not
a GPU allowlist. `init` seeds workloads only, never GPU types. Old target revisions
remain in the database so historical runs keep their foreign keys.

To add a model, copy an existing workload through the same pipe and change its
`name` and model settings. Configs are strict:
unknown fields, invalid lengths and duplicate concurrency entries are rejected.
Use a distinct configuration name for a quantized checkpoint or a different dtype.
Never silently change precision or offload to CPU to make a configuration fit.

Runs select current database revisions, resolve model branches to Hugging Face commit
SHAs and Docker tags to image digests **before** renting anything, and save the resolved
settings in the `runs.spec` JSON column with foreign keys to the configuration revisions.
The run also captures the worker/controller Python sources and controller dependency lock;
worker uploads use this captured source, not whatever files happen to be on disk later.
The Pod receives a temporary execution spec and emits temporary checkpoints; neither
is an external source of truth. Reports, raw results, logs, package lists, and sample
images are imported into SQLite. The database survives Pod deletion.

### Built-in profiles

| Configuration | Input | Output | Maximum total context |
|---|---:|---:|---:|
| `qwen8b-chat` | 512 | 256 | 4,096 capacity; requests use 768 |
| `qwen8b-prefill8k` | 8,192 | 256 | 8,448 |
| `qwen8b-decode4k` | 512 | 4,096 | 4,608 |
| `qwen8b-context16k` | 15,360 | 1,024 | 16,384 |
| `qwen8b-context32k` | 31,744 | 1,024 | 32,768 |

These are deliberately different tests: 8K is a prefill profile, **not** our longest
context test. No YaRN/context extension is enabled. Longer-than-native tests must be
added as separate model/engine configurations with explicit scaling support first.

The legacy `--type text_gen` suite uses concurrency 1/4/16, 32 requests per trial, three repeats,
and two warmup requests. This is a starting sweep, **not enough samples to establish
stable p99s**. Increase requests and sample independent machines before publishing
fleet-wide claims. One Pod per target is one machine sample, not a GPU fleet average.

For larger samples, prepare a new immutable workload variant rather than changing a
running spec. For example, `prepare --type text_gen --gpu "$GPU" --params
'{"requests":1024,"repeats":2,"trial_timeout_seconds":86400}' --max-minutes 2880`
sets 1,024 requests per trial, two repeats, a 24-hour trial timeout and a 48-hour
maximum Pod lease. These are upper bounds, not completion guarantees or predicted
runtime. `prepare` supports finite leases up to 20,160 minutes (14 days); text trial
timeouts support up to 259,200 seconds (72 hours). The revised matrix uses that
72-hour trial ceiling; long decode C1 trials can take hours to days even above the
throughput floor. Review the substantially larger cost estimate before `start`.
Old small-sample trials remain separate workload revisions in the dashboard.

`qwen32b-chat` is provided for larger cards, but **unquantized 32B will not fit a
single 24GB device**. Qwen3-8B BF16 also needs weights + KV cache + workspace; the 32K
profile may fail on 24GB. Such failures are preserved. No automatic shortening.

### Single-replica throughput and latency

Every run has **one replica**: one engine process across `gpus_per_replica` devices
(one device for the initial targets). A MIG slice is one allocated device, not a full
parent GPU. Concurrency is **requests sharing that replica**, not additional replicas.
The database persists these dimensions independently.

The `replica_trials` SQL view exposes output tokens/sec, requests/sec, TTFT/ITL/TPOT,
E2E latency, request counts and failures for each target/workload/concurrency/repeat:

```sql
SELECT target, model, dtype, input_tokens, output_tokens,
       replica_count, gpus_per_replica, concurrent_requests,
       output_tokens_per_second, requests_per_second, ttft_p95_ms, e2e_p95_ms
FROM replica_trials
WHERE status = 'completed'
ORDER BY target, workload, concurrent_requests, repeat;
```

This gives a per-card **throughput versus latency curve**, not just an unqualified
maximum-throughput number. Request latency starts when the benchmark client sends
that request; time waiting for its client-side concurrency semaphore is not included.
With `request-rate=inf`, this is a closed-loop saturation/concurrency test, not an
open-loop arrival-rate trace. Multi-GPU throughput (including the TP2 matrix row) is per
replica across those devices, not a per-GPU claim.

Optional `ttft_slo_ms` / `tpot_slo_ms` config fields enable vLLM's request goodput
(requests/sec satisfying the specified SLOs). Steady-state estimated dollars per
million output tokens / thousand images are stored in `metrics`; full-run cost,
including startup time, is separately recorded in `runs`.

### Text methodology

- Pinned `vllm/vllm-openai:v0.11.0` resolved to a digest, same baseline across targets.
- `vllm bench serve` with a custom, seeded synthetic word corpus, pinned model/tokenizer,
  verified exact input lengths, `ignore_eos`, zero temperature, no beam search.
- Corpus JSONL files are temporary Pod inputs and imported as SQLite BLOBs with SHA256.
  They are not configuration files. We avoid vLLM 0.11's random-token generator because
  decode/re-encode can silently produce shorter prompts even with range ratio zero.
- The Pod bootstrap installs pinned `pandas==2.2.3` for vLLM's custom JSONL reader;
  the final package environment is captured alongside the pinned base image digest.
- Uses `/v1/completions`; matrix prompts are explicitly chat-template-rendered with
  `enable_thinking=False` before submission. Exact synthetic input lengths include
  template overhead. Legacy raw-completion profiles remain separately identified.
  Forced token lengths are not reasoning quality or meaningful chain-of-thought.
- Client and server on localhost inside the Pod; no public network latency in the metric.
- Prefix caching disabled by default. Startup, model download/hashing, server compilation
  and warmup are outside steady-state trials and recorded separately.
- Engine restarted for each workload so its context/KV capacity is explicit.
- Reports include throughput, TTFT, TPOT, ITL, E2E percentiles and per-request data.
  Exact aggregate E2E comes from vLLM. Its 0.11 detailed output omits individual final
  completion latencies, so per-request `stream_token_span_seconds` is explicitly the
  first-token delay plus inter-token intervals, not mislabeled as exact final E2E.
- Check actual token counts; mismatches or incomplete requests fail a trial.
- Raw nvidia-smi process-memory telemetry is stored as an artifact. On MIG, telemetry
  may be unavailable or parent-scoped; do not treat it as measured slice peak VRAM.

Both client and server share CPU allocation. Always inspect actual vCPU/RAM and
machine metadata before attributing an observed difference solely to the GPU.

### Image methodology / current limits

`sdxl-1024` stores its entire built-in-node ComfyUI API graph **in SQLite**, including
model, seed schedule, resolution, sampler, steps and prompts. The adapter installs
ComfyUI at a resolved Git commit, downloads a pinned SDXL checkpoint, uses FP16,
disables result caching, changes the seed for every request, and saves sample PNGs.
Only sequential requests/batch-size sweeps are supported initially, not a concurrent
ComfyUI load test. E2E timings include HTTP polling overhead; execution event timestamps
are retained for analysis. Images are sanity samples, not a quality benchmark.

The ComfyUI adapter has passed a GPU smoke test on a 24GB Blackwell MIG slice,
including an inspected SDXL output image. It explicitly uses PyTorch SDPA, disables
xformers/custom nodes/async offload, keeps models GPU-resident, and uses FP32 VAE
alongside FP16 model execution. This avoids per-card automatic attention/precision
choices; the initial xformers-selected Hopper kernel crashed on the Blackwell slice
and that failed attempt is preserved in SQLite. Its runtime dependency installation
records `pip freeze`, but is not a fully locked environment. For a published
image comparison, use a dedicated prebuilt image pinned by digest with ComfyUI and all
requirements locked. Do not mix image and text results from a Pod after dependency
installation changes its environment. The first live validation targets text only.

### Direct Diffusers runtime and gated device checks

Direct image profiles set `backend=diffusers`, separate from the ComfyUI track.
They use true single-call batching, native SDPA, explicit component dtypes and no
implicit offload. Core throughput excludes postprocessing/transfer/PNG; delivered
E2E includes them. The worker enforces a whole-row alarm and reserves subprocess
termination grace inside that bound; absolute expiry SIGKILLs without blocking
reaping or final copies. Already synchronized points survive; the active point may
be incomplete.

A batch sweep measures B1 once, then B2/B4/etc. Each point supplies latency,
throughput, stage spans and memory telemetry together. New profiles default to
`execution_modes=["batch"]`; old configs requesting both `latency` and `batch`
are deduplicated by the worker. An explicit latency-only profile still measures
B1. Curve profiles require exactly 40 measured calls per successful point, plus
three warmups. Denoising steps and image settings are unchanged.

`image_matrix.tail_profile(...)` creates a **separate** `tail` track at B1 with
400 measured calls and three warmups. Its minimum count is also 400: insufficient
time is a failed/incomplete tail study, never a smaller cohort mislabeled as 400.
Tail jobs are staged after curves and need explicit time/spend approval; long tail
budgets are not allowed for ordinary curves. At 13 seconds per image, 400 calls
alone take about 87 minutes. Tail p99 remains an empirical estimate based on few
extreme observations, not a serving SLO or evidence of a stable population tail.

Warmup/compilation time remains charged to the hard point and row deadlines.
Affordability and the per-call deadline guard use the slower of the final two
warmups, rather than an earlier one-time compilation cost. Retained trial metadata
records all warmup durations and the chosen pilot. This estimate does not certify
thermal stability; a late recompilation still makes it conservative.

`prepare(..., validate_image_diagnostics=True)` opts into fail-closed diagnostic
validation **before the dispatcher admits a replacement GPU**. Current diagnostic
specs are BATCH B1/B2, three warmups and eight measured calls: retained artifacts
must prove steps/forwards, CFG batch dimensions, counts, CUDA residency/dtypes,
plausible timing separation, PNG CRC/dimensions/nonblank contents, and Pod deletion.
A clean B2 CUDA OOM is an explicit capacity outcome, not a successful B2 point.
Missing B1 proof or a correctness/runtime failure interrupts the queue. This option
is not enabled for text or ordinary measurement sweeps.

`python -m sls_bench.image_gate --parallel 4` can gate a separately prepared stock followup on
those device checks and prior first-GPU family proofs. It checks immutable model,
image and adapter identities, full family/device coverage, and aggregate budgets.
Unavailable diagnostic cards are blocked from the followup, never assumed valid.
All GPU selection remains in the prepared SQLite specs; no hardcoded fleet.

Reusable image build files and setup details: [`containers/diffusers/`](containers/diffusers/README.md).
The context is allowlisted and excludes credentials, measurements and model weights.
The worker skips installation only when a preinstalled manifest's requirements hash
**and live installed versions** agree. Private images use a saved registry-auth ID,
never registry passwords in target specs. A saved bootstrap-only Pod template is
not a prebuilt dependency image; build/deployment status is recorded separately.

## Storage schema

SQLite uses foreign keys, WAL, a 30s busy timeout, and `PRAGMA user_version` for schema
compatibility. No ORM or database server is needed for this initial schema.

- `configs`: validated target/text/image definitions, content hash and immutable revisions.
- `runs`: resolved settings, lifecycle, code hash, lease, prices and estimated run cost.
- `run_configs`: links to every source configuration revision.
- `resources`: ownership, Pod identity, sanitized allocation metadata and cleanup state.
- `hardware`: CUDA-visible device(s), driver/torch/CUDA, CPU/RAM, package versions.
- `trials`: each workload/concurrency/repeat, including setup failures; compact summaries
  avoid duplicating request arrays in dashboard queries.
- `metrics`: indexed scalar numeric measurements per trial.
- `requests`: per-request timings, lengths, errors, image execution events.
- `artifacts`: compressed BLOBs with SHA256, including raw logs/results/sample images.
- `events`: controller lifecycle and cleanup history.
- `benchmark_types`: category/adapter metadata and per-type configuration schemas.
- `gpu_catalog`: discovered live SKUs, vendor/generation/VRAM, prices, discovery time.
- `sweeps`: durable preparation, budget approval, process/heartbeat and queue state.
- `sweep_items`: every GPU's frozen execution spec, per-profile compatibility decisions,
  attempts, result status, and linked run. Each retry also retains its own `runs` row.

Collection is transactional and idempotent. Completed trials are checkpointed into
SQLite while the Pod is still running. A failed final transfer does not erase trials
already collected. Export a BLOB with:

```sh
uv run bench artifact RUN_ID worker.log --output worker.log
```

Use `bench backup` rather than copying just the `.sqlite3` file while WAL writes are
active. Store the database on a local persistent filesystem, **not an NFS mount or
inside a disposable Pod**. Backups can contain prompts and model outputs; protect them
accordingly. API credentials and existing-Pod environment variables are not stored.

## Spending and cleanup

- `sweep` requires `--yes`, checks catalog price estimates against `--budget`, checks
  actual allocated hourly price against an approved ceiling, and defaults to sequential
  execution. `start --parallel N` changes concurrency, not the approved workload budget.
- Budget is an **estimate**, not a provider-enforced dollar cap. Includes 20% price
  headroom plus a disk allowance. Wall-clock leases include provisioning/startup.
- Creation uses GraphQL because REST v1's enum rejects valid new MIG catalog SKUs.
  No automatic retry of a create mutation: an ambiguous timeout can create duplicates.
- Unique resource names are persisted **before** creation; cleanup reconciles lost
  create responses using exact names. An ambiguous create remains tracked even when
  the first lookup returns no Pod; the watchdog checks for late allocations, and no
  further target is allocated until the ambiguity is resolved. Unrelated Pods are
  never selected for deletion.
- Normal exit, exceptions and Ctrl-C delete owned Pods. A separate local watchdog
  also deletes expired Pods; an in-container timer stops its container as a fallback.
  A server-side `terminateAfter` is submitted as another safeguard.
- None of these replace checking the provider after a laptop/network outage. A local
  watchdog cannot reach Runpod while your computer sleeps or is offline. Run
  `cleanup --expired` after reconnecting and verify the console.
- Failed deletion retains `cleanup_failed` and stops further allocation in the sweep.
- `--pod-id POD` attaches to an existing SSH-enabled compatible Pod and **never
  terminates that Pod**. Cancellation signals only the run-specific worker PID after
  checking its command line; the worker stops its own engine process groups. The worker
  also has a lease alarm. Cancellation failures are recorded for manual recovery.
- A failed/unavailable target is recorded and the next target is tried; there is no
  substitution of a different SKU, cloud, precision or memory size.

## Development

```sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests use temporary SQLite databases and mocked HTTP/SSH, never paid Pods. Live
hardware smoke tests are separate, explicit invocations of `sweep`.

Primary references:
- https://docs.runpod.io/sdks/graphql/manage-pods.md
- https://docs.runpod.io/pods/configuration/use-ssh.md
- https://docs.runpod.io/api-reference/pods/DELETE/pods/podId
- https://raw.githubusercontent.com/vllm-project/vllm/v0.11.0/vllm/benchmarks/serve.py
- https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/script_examples/websockets_api_example.py
