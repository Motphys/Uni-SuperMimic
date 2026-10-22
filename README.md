<h1 align="center">UniSuperMimic</h1>

<h3 align="center">SuperMimic reproduction on UniLab</h3>

<p align="center">Languages: English | <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.28480"><img src="https://img.shields.io/badge/paper-arXiv--2610.28480-red" alt="Paper"></a>
  <a href="https://pypi.org/project/unilab/"><img src="https://img.shields.io/pypi/v/unilab" alt="UniLab PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

UniSuperMimic is an independent downstream reproduction of [SuperMimic](https://supermimic.github.io) on **UniLab** for Unitree G1 whole-body tracking of LAFAN1 motions, with **mjwarp, MuJoCo, and Motrix** backends. It includes a differentiable world model, a goal-conditioned policy, a Manager-Based task, tools for 40 motion assets, and entry points for training and deterministic full-library evaluation.

<p align="center">
  <a href="docs/assets/supermimic_comparison.mp4"><img src="docs/assets/supermimic_comparison.gif" alt="SuperMimic policy and reference motion comparison" width="95%"></a>
</p>

<p align="center"><em>Solid robot: SuperMimic policy. Blue ghost: reference motion. Click the image to view the MP4.</em></p>

Paper: [*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*](https://arxiv.org/abs/2610.28480), Qiayuan Liao et al. The implementation references `supermimic_vibed` and depends on PyPI `unilab`.

## Repository Structure

```text
unisupermimic/
├── src/unisupermimic/
│   ├── rl/
│   │   ├── algos/supermimic/       # World model, policy, critic, replay, and learner
│   │   └── contracts/supermimic.py # Environment/algorithm tensor contract
│   ├── tasks/motion_tracking/g1/supermimic/
│   │   ├── env.py                  # Manager-Based environment and backend factory
│   │   ├── manager_terms.py        # Command/action/observation/event/reward terms
│   │   ├── motion.py, mdp.py       # Motion library and differentiable MDP
│   │   ├── evaluate.py             # Deterministic full-clip evaluation from frame zero
│   │   ├── render.py               # Policy rollout and reference-motion comparison video
│   │   └── registry.py             # G1SuperMimic registration
│   ├── scripts/train_supermimic.py # Training, resume, and evaluation entry point
│   ├── conf/supermimic/            # Hydra configuration
│   ├── assets/robots/g1/           # G1 MJCF; meshes/URDF are downloaded and untracked
│   ├── assets/motions/             # LAFAN1 CSV files, downloaded and untracked
│   └── sim2sim.py                  # Policy I/O, network shape, and asset identity checks
├── docs/assets/                    # Result curves, MP4, and GIF
├── tools/prepare_supermimic_assets.py
├── tests/
├── pyproject.toml, uv.lock
├── LICENSE, LICENSES/, NOTICE
└── Makefile
```

## Installation

Requirements: Linux, Python 3.10–3.13, and [uv](https://docs.astral.sh/uv/). A CUDA GPU is required for mjwarp training. Run all commands from the repository root. `uv run` uses `uv.lock` to synchronize the project, PyPI `unilab`, all three simulation backends, and development dependencies.

```bash
uv run unisupermimic-train --help
```

Run `uv lock` after changing dependencies.

On AMD ROCm hosts, install from the pre-resolved ROCm lockfile instead (a ROCm
PyTorch build with the same pinned dependency set):

```bash
make sync-rocm
```

CUDA-only backends (mjwarp) stay unsupported there; train with the MuJoCo
host-bridge backend (`task=g1_super_mimic/mujoco`). `make sync-rocm` swaps
`pyproject.toml`/`uv.lock` for the ROCm profile; run `make sync-default` to
switch back. Maintainers regenerate the ROCm lockfile after dependency changes
with `make lock-rocm`.

### Prepare Assets

```bash
# Download LAFAN1/G1 data from Hugging Face, then generate and validate MJCF.
uv run tools/prepare_supermimic_assets.py

# Validate existing assets offline.
uv run tools/prepare_supermimic_assets.py --validate-only
```

Assets come from `lvhaidong/LAFAN1_Retargeting_Dataset`. A complete asset set contains 40 CSV files with 264,705 source frames and 36 columns per frame. Meshes, URDF, CSV files, logs, and checkpoints are excluded by `.gitignore`.

## Training

Run the following commands from the repository root and set `CUDA_VISIBLE_DEVICES` to an available GPU when using CUDA.

```bash
# Default: mjwarp, 4096 parallel environments, 12000 iterations.
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  training.log_dir=runs/supermimic_mjwarp_001

# Resume from last.pt in the same run and train to 16000 iterations.
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  algo.resume=true \
  algo.load_run=runs/supermimic_mjwarp_001 \
  algo.checkpoint=last.pt \
  algo.iterations=16000

# Select MuJoCo through the task owner.
uv run unisupermimic-train \
  task=g1_super_mimic/mujoco \
  training.log_dir=runs/supermimic_mujoco_001

# Select Motrix through the task owner.
uv run unisupermimic-train \
  task=g1_super_mimic/motrix \
  training.log_dir=runs/supermimic_motrix_001
```

Use the same `task=g1_super_mimic/<backend>` override when resuming or evaluating a MuJoCo or Motrix run. Motrix uses fixed friction; center-of-mass and joint-offset domain randomization remain enabled. Use Hydra syntax for other configuration overrides. For a quick smoke run, override options such as `algo.num_envs`, `algo.iterations`, `algo.wm_hidden`, `algo.policy_hidden`, and `algo.compile=false`. See `src/unisupermimic/conf/supermimic/config.yaml` for the complete default configuration.

## Evaluation

Evaluation creates one environment per motion, starts from its first frame, disables observation and action noise, and runs the complete clip. Strict sim2sim validation checks policy I/O, network shape, motion library, and MJCF identity before environment construction.

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  training.play_only=true \
  algo.load_run=runs/supermimic_mjwarp_001 \
  algo.checkpoint=12000
```

Results are written to `ckpt_012000_eval.json` in the same directory. You can also evaluate a checkpoint directly. The standalone evaluator and video exporter accept `--backend mjwarp`, `--backend mujoco`, or `--backend motrix` and default to mjwarp:

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-eval \
  --backend mjwarp \
  --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt
```

## Video Export

The following command rolls out `jumps1_subject1` continuously from a checkpoint, trims the result to the jumping segment from 6–14 seconds, and writes a 25 fps comparison MP4:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run unisupermimic-video \
  --backend mjwarp \
  --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt \
  --clip jumps1_subject1 \
  --start 6 \
  --seconds 8 \
  --output docs/assets/supermimic_comparison.mp4
```

By default, the reference root XY is aligned with the policy to highlight whole-body tracking. Pass `--global-ghost` to preserve the reference motion's global displacement.

## Results

Configuration: PyPI `unilab 1.5.0`, mjwarp, and 4096 training environments. Evaluation covers 40 LAFAN1/G1 motions. `norm_return` is the macro average across clips; `frame-weighted` is weighted by clip length.

| checkpoint | norm_return | frame-weighted | failed clips |
|---|---:|---:|---:|
| ckpt_012000.pt | 0.827227 | 0.829952 | 0 / 40 |

![Training and evaluation curves through iteration 12000](docs/assets/success_curve.png)

## Tests

```bash
make check       # ruff, mypy, pyright
make test        # Non-slow tests
make test-all    # Include production backend slow tests; mjwarp requires CUDA
uv run tools/prepare_supermimic_assets.py --validate-only
```

## License

The repository code is released under [Apache-2.0](LICENSE). The reference implementation's MIT license and original copyright notice are preserved in [LICENSES/MIT-supermimic.txt](LICENSES/MIT-supermimic.txt); see also [NOTICE](NOTICE). UniLab, UniSim, MuJoCo/mjwarp, MotrixSim, and the datasets retain their respective upstream licenses. This repository does not relicense those external dependencies or downloaded assets.
