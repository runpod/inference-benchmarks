# Preinstalled Diffusers Pod runtime

Build context is an explicit allowlist: Dockerfile, runtime verification, SSH bootstrap,
and requirements generated from `sls_bench.image_diffusers.REQUIREMENTS`. It contains
**no benchmark data, model weights, HF tokens, SSH private keys, registry secrets,
or organization credentials**. Do not build using the repository root as context.

```sh
# From benchmarks/
uv run python containers/diffusers/prepare_context.py data/diffusers-image-context
docker buildx build --builder default --platform linux/amd64 \
  --load -t REGISTRY/NAMESPACE/sls-bench-diffusers:VERSION data/diffusers-image-context
docker push REGISTRY/NAMESPACE/sls-bench-diffusers:VERSION
```

Use an authenticated private registry and pin the resulting **remote manifest digest**
in all new immutable target specs. Do not retag or mutate an in-flight sweep.
The base remains the CUDA 12.8 / PyTorch 2.9.1 image used in CUDA diagnostics.
The build installs pinned Diffusers dependencies, pandas 2.2.3 and SSH once, verifies
imports/versions without GPU/model downloads, and writes a runtime manifest.

A private Runpod Pod template should use this digest, TCP 22, no persistent volume,
and the image entrypoint; provide only PUBLIC_KEY and a finite BENCH_TTL_SECONDS
(default 3300). Standalone template startup stops PID 1 after its lease; **it does
not replace controller/provider deletion**. Normal benchmark dispatch retains the
controller's absolute lease, provider terminateAfter, independent watchdog and
finally-cleanup. Its bootstrap detects already installed SSH/pandas.

The controller uses the pinned image directly (plus a saved registry-auth ID for
private pulls), rather than inheriting mutable template settings. Model credentials
are still delivered separately over SSH, outside collected artifacts.

The worker skips pip only when the image manifest hash matches the exact requested
requirements AND installed versions verify; mismatches fall back to the normal
bounded installation path. A prebuilt marker alone is not proof of CUDA correctness.
Actual first-GPU diagnostics remain required after changing the image.

Official references:
- https://docs.runpod.io/pods/templates/overview.md
- https://docs.runpod.io/api-reference/templates/POST/templates.md
- https://docs.runpod.io/api-reference/container-registry-auths/POST/containerregistryauth.md
