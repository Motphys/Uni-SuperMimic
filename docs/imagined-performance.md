# Imagined frozen-WM performance benchmark

This document and `tools/bench_imagined.py` provide a reproducible way to measure
the **learner update** in `ImaginedSuperMimicLearner`. The benchmark isolates the
now pure MLP-policy/frozen-MLP-world-model path from environment collection and
world-model optimization.

## Quick start

The default synthetic task requires no robot, motion library, or checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run tools/bench_imagined.py \
  --task fake --batch 4096 --horizon 32 \
  --policy-hidden 2048,1024,512,256 --wm-hidden 2048,2048,2048 \
  --num-joints 29 --context-dim 33 \
  --warmup 10 --iterations 50 \
  --amp bf16 --compile --tf32 auto --fused-adam auto \
  --label imagined-bf16 \
  --output benchmarks/imagined-bf16
```

Use `--device cpu` (or omit it on a host without CUDA) for a CPU-only run. Every
run writes `benchmark.json`; stdout reports iterations/s and imagined steps/s.
The synthetic task does not substitute for accuracy evaluation. It exists to
make kernel and scheduling changes comparable before loading production assets.

A production checkpoint task can be measured when all assets exist:

```bash
CUDA_VISIBLE_DEVICES=0 uv run tools/bench_imagined.py \
  --task real \
  --checkpoint runs/world_model/last.pt \
  --motion-file data/library/bones_seed_compact.pt \
  --batch 4096 --horizon 32 \
  --warmup 10 --iterations 50 --amp bf16 --compile
```

`--mjcf-path` overrides the bundled MJCF. The frozen checkpoint must contain the
`wm` state dict expected by `build_g1_imagined_task`.

## Measured quantities and artifacts

- Wall-clock seconds per learner update, including update/reset bookkeeping.
- CUDA event seconds per learner update, followed by one final synchronization;
  CPU reports this field as `null`.
- Iterations/s and imagined steps/s, where imagined steps are
  `iterations × batch × horizon`.
- CUDA allocator peak/reserved memory growth after warmup (unavailable on CPU).
- Sampled CPU and GPU utilization. `nvidia-smi` is tried first, then `rocm-smi`
  and `amd-smi`; if none is available, the report records the degradation rather
  than inventing GPU values.
- Optional PyTorch profiler table and Chrome trace through `--profile torch`.
  Override output names with `--trace-path` and `--profile-table-path`.
- Complete `BenchmarkSettings`, PyTorch/build versions, device name, TF32 flags,
  AMP/compile/optimizer flags, and sampled utilization in `benchmark.json`.

Keep benchmark process priority, power/clock policy, input dimensions, warmup,
and iteration count fixed when comparing reports. Put traces and `benchmark.json`
in the same labeled directory. More iterations reduce timing noise but can change
thermal behavior; keep both fixed for A/B tests.

## Acceleration matrix

| Knob | CUDA | ROCm/HIP | Notes and recommendation |
|---|---|---|---|
| Eager baseline | available | available | Always collect an eager baseline. It is the fallback for unsupported compilation or graph capture. |
| `torch.compile` | available | available in recent ROCm PyTorch builds; subject to Triton/kernel coverage | Compare `--compile` against eager. Keep a non-compiled CI path. On ROCm, test the exact model shape and PyTorch version; compilation failures should be treated as configuration errors, not silently ignored. |
| BF16 AMP | widely available on data-center/recent consumer GPUs | commonly available on MI GPUs; depends on supported native kernels | This learner currently supports `none` and `bf16` (`--amp bf16`). Compare numerically on real tasks before using it. CPU execution intentionally remains fp32. |
| TF32 | NVIDIA Ampere and newer | not applicable | `--tf32 auto` enables TF32 for fp32 matmuls only when AMP is `none`. Set `--tf32 off` when bit-compatibility matters. |
| Fused Adam | available | support is backend/version dependent | `--fused-adam auto` enables it on CUDA and disables it elsewhere. Check `benchmark.json`'s effective flag rather than relying on the request. |
| CUDA graphs | available through PyTorch | not portable; ROCm graph support depends on PyTorch/ROCm/driver | `--cuda-graph update` attempts full-update capture. The current update contains reset counts and other host synchronization/control flow, so full capture may fail until that path is graph-safe. A failure is reported rather than replaced with an eager result. Do not compare a failed graph run to a successful run. |

## Current benchmark status on this branch

Verified on 2026-11-04 with the repository's CUDA lockfile:

- CPU synthetic-task smoke benchmark with profiler trace/table export.
- CUDA synthetic-task smoke benchmarks with profiler trace/table export.
- AMP `none`/TF32/fused Adam on CUDA.
- AMP `bf16` plus `torch.compile` on CUDA.
- Full CUDA graph capture attempt on CUDA. It currently fails during capture
  (`cudaErrorStreamCaptureInvalidated` after a backward-pass error), consistent
  with host synchronization/control flow in the present update/reset path. This
  is reported as a failure rather than silently falling back to eager timing.
- Separate eager-vs-compile A/B with all other factors fixed was **not run**;
  the verified BF16 run combined the two knobs and is therefore only a smoke
  measurement, not an optimization recommendation.

ROCm was **not available on this machine** and is therefore **not verified here**.
The matrix above documents expected APIs and required validation, not measured
performance. On an AMD host, run at least eager and compiled CPU/GPU synthetic
benchmarks plus one real-checkpoint benchmark before promoting any configuration.
Record `torch.version.hip`, the ROCm runtime version, driver version, and kernel
messages if compilation fails.

## Measurement limits

- `nvidia-smi` utilization is sampled, not exact per-kernel utilization; use the
  Chrome trace for kernel-level attribution.
- With `CUDA_VISIBLE_DEVICES`, the NVIDIA text query cannot identify the visible
  ordinal. If several host GPUs are present, the sample is explicitly labeled
  `nvidia-smi[first-visible-host]`.
- CUDA graph capture changes allocator and synchronization behavior. Compare it
  only after the eager/compiled baseline has been warmed up in a separate run.
- A CPU benchmark validates CLI/report behavior but is not a useful predictor of
  GPU throughput.
- The sampled CUDA host memory/utilization shown by `nvidia-smi` includes other
  processes on the same GPU. Use profiler CUDA time and allocator memory for the
  benchmark attribution itself.
