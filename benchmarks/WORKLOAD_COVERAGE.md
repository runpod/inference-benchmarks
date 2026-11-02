# GPU workload coverage

Reference for the GPU use cases this benchmark dashboard covers and those it does
not. This is a measurement taxonomy, not a list of implemented adapters or measured
hardware rankings. Adding an entry here does not enable execution or launch a paid run.

## Current coverage

- **Text generation:** vLLM single-replica adapter, synthetic fixed-length prompts,
  closed-loop concurrency sweeps, recorded throughput and latency percentiles.
  Dashboard views include prefill/decode comparisons and trial-summary
  latency-qualified cost. Goodput requires collection-time SLO measurements.
- **Image generation:** experimental ComfyUI / SDXL diffusion adapter with image
  throughput, polling-inclusive E2E latency, and sample outputs. Samples are not
  a quality score.
- **Image processing, video generation, video processing:** measurement maps exist
  in the dashboard, but adapters are not implemented. They are not equivalent to
  diffusion, and diffusion results must not stand in for them.
- **KV-cache capacity and realistic traffic:** not currently measured. Total
  process memory is not KV occupancy; fixed-length closed-loop load is not a
  realistic arrival trace. Pilot request counts do not establish stable p99s.

See [README.md](README.md#dashboard-measurement-model) for the dashboard's units,
aggregation rules, and current limitations.

## Additional GPU use cases

The priority column reflects relevance to a serverless inference offering, not a
claim about market size or GPU performance. None of these additional families has
an implemented adapter in this repository.

| Use case | Representative workloads | Useful measurements | Comparison controls | Priority |
| --- | --- | --- | --- | --- |
| Embeddings | Search/RAG document indexing; online query embeddings | Documents/s and input tokens/s; p50/p95 latency; cost/million input tokens; retrieval quality | Model revision, input-length distribution, batch size, vector dimension, precision, pooling, normalization, truncation | High |
| Reranking | Cross-encoder scoring of retrieved candidates | Query-document pairs/s; end-to-end query latency vs candidate count; cost/query; ranking quality | Model, query/document lengths, candidates/query, batching, precision, fixed relevance judgments | High |
| Speech-to-text | Offline transcription; streaming transcription | Audio hours/GPU-hour; real-time factor (processing time/audio duration; lower is better); streaming partial/final latency; concurrent streams; word error rate | Model, language/accent/noise mix, sample rate, audio duration, chunk size, decoding settings, diarization, timestamps | High |
| Text-to-speech | Interactive voice; offline narration | Time to first audio; generated audio seconds/wall-second; concurrent real-time streams; cost/audio minute; quality and intelligibility | Model/voice, language, output duration, sample rate, streaming chunk size, vocoder, evaluation corpus | High |
| Vision-language inference | Image understanding; document/OCR extraction; multi-image questions | TTFT vs image count/resolution; pages/s; output tokens/s; p95 request latency; extraction/task accuracy; cost/page | Model, image preprocessing, visual-token count, page/image count, resolution, prompt/output length, precision | High |
| Training & fine-tuning | LoRA/QLoRA adaptation; full fine-tuning; pretraining | Training tokens/s; peak device VRAM; time/cost to target validation quality; checkpoint time; multi-GPU scaling | Model, dataset/order, optimizer, precision/quantization, sequence length, effective batch, gradient accumulation/checkpointing, adapter rank, topology | Medium; separate training suite |
| Scientific computing & simulation | Molecular dynamics; numerical solvers; scientific ML | Workload-specific steps/s or time-to-solution; numerical accuracy/convergence; memory limits; multi-GPU efficiency | Fixed problem size, precision, solver/tolerance, software version, CPU allocation, communication/topology | Specialized; task-specific suite |
| 3D rendering | Offline path tracing; animation frames; interactive rendering | Seconds/frame; samples/s at matched quality; peak VRAM/scene capacity; scene-load time | Renderer/version, scene/assets, resolution, sample count, denoising, ray-tracing backend, quality threshold | Specialized; rendering suite |

Important distinctions:

- Embedding input tokens, generation output tokens, audio seconds, image pages,
  training steps, and rendered frames are different billing/performance units.
  Do not collapse them into a single GPU score or token-cost chart.
- Reranking throughput per pair is not query throughput: each query can require
  many candidate scores.
- Training throughput without comparable validation quality can reward a faster
  but worse training recipe. LoRA, QLoRA, and full fine-tuning belong in separate
  comparisons; checkpoint loading and saving also matter for short jobs.
- Quality controls are required for meaningful speech, multimodal, rendering,
  and scientific comparisons. A successful request alone does not establish
  equivalent output quality or numerical correctness.

## Text-generation serving patterns

These are different ways of using text models, not additional model modalities.
The current fixed-length benchmark does not fully represent them.

| Pattern | What it exposes | Measurements and controls needed |
| --- | --- | --- |
| Offline batch processing | Sustained utilization and cost when interactive latency is less important | Completed jobs/hour; output tok/s; total completion time; cost/million output tokens; queue drain time; representative prompt/output distributions and batch deadlines |
| Long-context / RAG | Prefill cost, long-context memory pressure, prefix reuse, and retrieval overhead | TTFT vs input length; measured KV bytes vs active context/concurrency; cache-hit rate; preemption/OOM events; cold vs warm prefix cache; retrieval time separated from model time; answer quality |
| Tool-using agents | Repeated short calls, sequential dependencies, tool waits, and bursty follow-up requests | End-to-end task latency/cost/success; per-model-call TTFT/TPOT; model vs tool/queue time; requests/task; representative tool traces with fixed tool behavior |
| Many-model / multi-LoRA serving | Weight/adaptor residency, switching costs, cache eviction, and scheduling interference | Cold/warm model load and adapter-swap latency; hit/miss rate; per-model latency and goodput under a fixed popularity mix; resident VRAM; eviction frequency; base-model and adapter revisions |
| Interactive mixed traffic | Tail latency and SLO-compliant capacity under arrivals that are not synchronized to completions | Open-loop arrival-rate sweeps or replayed traces; mixed input/output lengths; p50/p95/p99 TTFT and E2E; request-level goodput; offered/completed/failed request rates; explicit queue-time inclusion and SLOs |

The existing long-context profiles provide some input-length observations, but they
are not an end-to-end RAG test. Likewise, measured concurrency is not instantaneous
engine batch size, and median trial p95 latency is not a pooled request p95.

## Cross-cutting measurements

For every applicable family, distinguish:

- **Cold start vs warm execution:** provisioning, download, model load, compilation,
  warmup, and steady-state service time. Publish full-job economics separately
  from steady-state cost estimates.
- **Latency vs throughput vs goodput:** total work completed, per-request latency,
  and requests/s meeting explicit targets answer different questions. Compute
  goodput from actual request observations or a documented benchmark metric,
  not from whether an aggregate p95 passed a threshold.
- **Device vs replica:** record allocated GPU/slice count and model parallelism.
  A MIG slice is not its parent card. Measure multi-GPU scaling against a matched
  baseline and retain interconnect/topology metadata.
- **Memory scope:** distinguish weights, KV cache, activations/workspaces, reserved
  memory, and used memory. Parent-scoped MIG telemetry cannot establish slice usage.
- **Bottlenecks outside the GPU:** include CPU preprocessing, host/device transfers,
  storage/network input, decode/encode engines, and downstream services where relevant.
- **Reproducibility and uncertainty:** pin software/model/data revisions, retain
  failures and raw timings, report sample counts, and repeat across independent
  machine samples before making fleet-wide claims.
