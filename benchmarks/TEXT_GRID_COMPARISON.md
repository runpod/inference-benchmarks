# Four-point text concurrency grid

## Saved configuration

`text_gen_four_point_v7@1` (SQLite config 483) is based on the latest
`text_gen_engine_reuse_smoke_v6@1`, not a replacement of historical measurements.
Select it with `--matrix text_gen_four_point_v7` when preparing a future sweep.
No sweep was prepared or launched for this change.

- Preserve all five fixed-length profiles: chat, prefill8k, decode4k, context16k,
  context32k; preserve the separate seven-rate mixed ShareGPT profile.
- Fixed-length client concurrency: **1, 8, 48, 128**, instead of
  1, 2, 4, 8, 16, 32, 64, 128, 256, 512.
- Preserve two repeats, token lengths, precision, fixed engine capacity (64
  scheduling slots), 32K engine context, point budgets and other v6 settings.
  C128 is client concurrency; requests can queue above the engine's 64 slots.
- Preserve smoke counts: two requests at C1, one per concurrent slot otherwise.
  Mixed counts/rates are unchanged. Disable cliff/plateau stopping so the explicit
  four-point grid is attempted; failures and hard deadlines can still prevent completion.
- Source defaults for the broader `text_gen` matrix also use the four-point grid,
  retaining its existing models, larger request counts and two repeats. Existing
  saved matrices and frozen sweeps are not automatically overwritten.

## Trial reduction

A trial here is one workload × load point × repeat, not one request or pilot.

| Per GPU, six-profile v6/v7 setup | Previous full grid | New grid | Removed |
|---|---:|---:|---:|
| Five fixed profiles × two repeats | 100 | 40 | 60 (60%) |
| Mixed: seven rates × two repeats | 14 | 14 | 0 |
| Total | 114 | 54 | 60 (52.6%) |

Across the two GPUs in the latest timing smoke, that is **228 → 108 planned
trials**, removing **120**. These are full-plan counts, not claims that the old
partial runs completed all 228 trials.

## Inference-time estimate from recorded measurements

Source: `benchmarks/data/benchmarks.sqlite3`, completed trials' `result.metrics.duration`.
Durations are summed benchmark serving time, not GPU kernel-active time or Pod wall
clock. Exclude startup, pilots, client overhead and failed/incomplete commands.

The latest fixed-engine smoke (`f3ebce4d37d04514adfb111e59af95f0`) did **not**
measure C48 or C128. Its measured fixed-length time was:

| GPU | Completed trials | Measured inference time | Time at levels removed by new grid |
|---|---:|---:|---:|
| RTX 4090 | 44 | 39.1 min | 26.3 min (28 trials) |
| RTX PRO 4500 Blackwell | 37 | 26.2 min | 11.8 min (21 trials) |

The 38.0 minutes in the final column are **gross removed work**, not a net saving:
new C48/C128 measurements must be added. Several long-context points had already
failed/timed out at C4/C8. Forcing new high-concurrency points can therefore cost
more than the old cliff-truncated execution for those profiles.

The earlier cliff smoke (`eb87599c42c64e3aaf9ad7ca7567e6c9`) has enough
**short-chat** measurements to make a rough comparison: retain measured C1, C8
and C128, and estimate C48 as the midpoint of measured C32/C64 durations, per repeat.
Compare against all completed short-chat trials on each GPU:

| GPU | Old recorded short-chat time | Estimated four-point time | Reduction |
|---|---:|---:|---:|
| L4 | 21.1 min | 6.0 min | 72% |
| RTX 4090 | 22.1 min | 13.3 min | 40% |
| RTX PRO 4500 Blackwell | 29.9 min | 13.2 min | 56% |
| Sum | 73.1 min | 32.4 min | 56% |

This is an interpolation, **not a new benchmark**. The old L4/4090 chat runs had
19/18 completed trials versus 20 on Blackwell, and the old engines matched capacity
to concurrency, unlike v6/v7. C48 can have nonlinear cache/queueing effects.

**Planning takeaway:** 60% fewer fixed-length trials is exact for the full grid;
roughly half the inference time is plausible for short chat (observed estimate
40–72%, 56% pooled). There is no defensible single net-runtime reduction for all
five lengths from the partial previous benchmark. In particular, do not promise
60% less whole-run time or extrapolate short-chat timings to long-context C128.
