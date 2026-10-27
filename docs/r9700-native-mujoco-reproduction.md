# Radeon AI PRO R9700: native MuJoCo reproduction

This is an **independent local run**, not a same-backend GPU benchmark against the
README's `mjwarp` result. The public-inference policy uses the original 40
G1-retargeted LAFAN1 clips; evaluation starts each clip at frame zero, turns
noise off, and runs until the original hard termination or the resampled clip
end. These are training motions, not a held-out generalization set.

## Environment and run

- Training worktree based on revision
  `faf7818e8abac3ec6abb8f00e6e904d3577f800b`, with the numerical and
  native-sensor patches described here. See the post-run correction below:
  the submitted PR is **not byte-for-byte identical** to the training code.
- One Radeon AI PRO R9700 (`gfx1201`, 32 GB); ROCm 7.2, PyTorch
  `2.14.1+rocm7.2`, UniLab 1.5.1, native MuJoCo host bridge (not `mjwarp`).
- One fresh seed (`algo.seed=0`), 4096 training environments, 12,000 updates,
  bf16 AMP, `algo.compile=false`, 500 world-model-only warmup updates. A
  checkpoint was saved every 500 updates. No previous policy was loaded.
- The 40 CSVs contain 264,705 source frames at 30 Hz. MotionLib resamples to
  50 Hz. The evaluator's equal-clip `norm_return`, not frame-weighted return,
  is reported below.

On a **fresh checkout**, select the existing ROCm dependency profile and prepare
assets as in the main README. On a multi-GPU host, first verify that your HIP
mask exposes the **R9700**, not the integrated GPU; ordinal mappings can differ
between `rocm-smi` and PyTorch. On the measured host, `HIP_VISIBLE_DEVICES=0`
selected the R9700:

```bash
make sync-rocm
HIP_VISIBLE_DEVICES=0 .venv/bin/python -c \
  'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)'
.venv/bin/python tools/prepare_supermimic_assets.py --validate-only

HIP_VISIBLE_DEVICES=0 .venv/bin/python -m unisupermimic.scripts.train_supermimic \
  task=g1_super_mimic/mujoco training.sim_backend=mujoco \
  training.log_dir=runs/r9700_mujoco4096_12000_fresh_seed0 \
  algo.num_envs=4096 algo.seed=0 algo.iterations=12000 \
  algo.wm_only_iters=500 algo.ckpt_every=500 algo.log_every=25 \
  algo.compile=false algo.resume=false
```

From a new process, evaluate each saved checkpoint with the **same** backend
and saved run configuration (the evaluator writes `ckpt_<iteration>_eval.json`
next to the checkpoint). For example:

```bash
HIP_VISIBLE_DEVICES=0 .venv/bin/python -m unisupermimic.scripts.train_supermimic \
  task=g1_super_mimic/mujoco training.play_only=true \
  training.sim_backend=mujoco \
  algo.load_run=runs/r9700_mujoco4096_12000_fresh_seed0 \
  algo.checkpoint=11500 algo.seed=0 algo.num_envs=4096 \
  algo.iterations=12000 algo.wm_only_iters=500 \
  algo.ckpt_every=500 algo.log_every=25 algo.compile=false
```

The recorded experiment evaluated **all 24** saved checkpoints with that
original full-clip protocol; `r9700-native-mujoco-checkpoints.csv` contains
iteration, failed-clip count, and unrounded macro return. The best checkpoint
was independently re-evaluated in a fresh process with identical per-clip
JSON, and each of its 40 nonfailed trajectories reached the CSV-derived full
resampled length (`r9700-native-mujoco-best-per-clip.csv`).

| Saved checkpoint | Training wall time | Full-clip failures | Equal-clip normalized return |
|---|---:|---:|---:|
| 11,000 | ~6 h 04 min | 2 / 40 | 0.8111440539360046 |
| **11,500 (best)** | **6 h 21 min 26 s** | **0 / 40** | **0.8350659608840942** |
| 12,000 | 6 h 38 min 30 s in trainer (6 h 38 min 50 s for whole process) | 0 / 40 | 0.829927921295166 |

**Post-run correction / provenance:** Independent review found that the
original training code's identity-heading fallback had a discontinuity when
the root forward axis was almost vertical. This PR replaces that hard switch
with a continuous bounded projection; it also checks **every** world-model
update for nonfinite metrics, whereas the training worktree checked only the
last update of each iteration. The 12,000-update training above predates these
two corrections and has **not** been rerun on this exact PR commit. The saved
11,500 policy was loaded through this PR's corrected code and re-evaluated on
all 40 original full clips in a fresh process: **0/40 failures,
0.8347989320755005** macro return, and 40/40 full resampled lengths
(`r9700-native-mujoco-pr-recheck.csv`). Its failure flags and step counts
match the archived evaluation; the return differs. Do not attribute the
archived training curve or its exact `0.8350659608840942` return to a fresh
training run with this post-run heading correction.

The first **saved and evaluated** checkpoint above the local joint gate
(0/40 failures and unrounded return **strictly greater than 0.82**) was 11,500;
no claim is made about individual unsaved updates between 11,000 and 11,500.
Evaluation and rendering time are excluded from training wall time. No MI300X
same-code, same-backend comparison is established by this run. The upstream
README's `mjwarp` result uses a different backend and dependency version, so
the two scores do not isolate a GPU or simulator effect.

## Why the code changes are needed

- The full motion library and the 10-frame goal batch can expose ROCm
  small-matrix batch-size numerical faults missed by a clip-start smoke test.
  Componentwise rotation contractions preserve the intended math, while
  tests check selected full-library values against independent CPU float64
  references and compare identical goal rows at large and small batch sizes.
- Imagined reward and critic-feature batches at 32×4096 (and 33×4096) are
  processed in order in chunks of at most 4096 rows. Autograd still flows
  through concatenated chunks; tests cover order and derivatives.
- The native MuJoCo tracked-body packet is refreshed before the first
  post-physics Manager read; the slow integration test checks packet root
  position, orientation, and velocity against native physics state.
- Heading becomes undefined near vertical forward direction; contract the
  horizontal heading block continuously toward zero there rather than
  switching to a potentially opposite identity heading. Duplicate same-tick
  return-bin writes are mean-aggregated before the single EMA update.
  Nonfinite metrics on **each** world-model or policy update abort before
  producing another checkpoint.

These are tested fixes, not a claim that any one fix alone caused the final
return. The optional asset- and GPU-dependent numerical/integration tests
require `pytest -m slow`; normal CI can run the unit suite without downloaded
motion assets. Checkpoints, raw training logs, and rendered video are not
committed to the repository.
