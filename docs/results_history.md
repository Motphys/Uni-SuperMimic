# Result history

Historical BONES-SEED / LAFAN1 result tables archived from the README. The
README now presents only the current default recipe with reset-segment CUDA
graph capture ([`perf/unilab-115-masked-autoreset-compile`]). Everything below
was produced with the configurations described at the time and is kept for
provenance.

## LAFAN1 library (40 clips)

Configuration: PyPI `unilab 1.5.0`, mjwarp, and 4096 training environments. Evaluation covers 40 LAFAN1/G1 motions. `norm_return` is the macro average across clips; `frame-weighted` is weighted by clip length.

| checkpoint | norm_return | frame-weighted | failed clips |
|---|---:|---:|---:|
| [ckpt_012000.pt](https://github.com/Motphys/Uni-SuperMimic/releases/download/checkpoints-v1/lafan1_ckpt_012000.pt) | 0.827227 | 0.829952 | 0 / 40 |

![Training and evaluation curves through iteration 12000](../assets/success_curve.png)

## BONES-SEED library (129,487 clips)

Same configuration on the BONES-SEED G1 compact library (129,487 kept clips, 47.3M frames, 6.72 GB fp16): one RTX 4090 (48 GB), 12000 iterations, 1.69 h learner wall time. Evaluation covers the complete kept library with deterministic full-clip rollouts. The recorded result was produced by the original one-environment-per-clip evaluator; current evaluation recycles finished environments instead. The LAFAN1 row is a baseline run with identical settings.

| policy | eval set | norm_return | frame-weighted | failed clips |
|---|---|---:|---:|---:|
| BONES-SEED [ckpt_012000.pt](https://github.com/Motphys/Uni-SuperMimic/releases/download/checkpoints-v1/boneseed_ckpt_012000.pt) | all 129,487 kept clips | 0.9243 | 0.9267 | 350 / 129,487 (0.27%) |
| LAFAN1 [ckpt_012000.pt](https://github.com/Motphys/Uni-SuperMimic/releases/download/checkpoints-v1/lafan1_ckpt_012000.pt) (baseline) | all 40 clips | 0.8336 | 0.8366 | 0 / 40 |

![BONES-SEED vs LAFAN1: full-library evaluation and training curves through iteration 12000](../assets/boneseed_curve.png)

Per-clip distribution on BONES-SEED: median 0.9528, P10 0.8337, P90 0.9841. BONES-SEED converges much faster (early-termination 0.83% vs 29.4% at iteration 2000) and finishes higher on both training reward and evaluation return. An independent RTX 5090 run reproduces these numbers (0.9262 on a random 10,000-clip subset, within 0.002 of the full-library value).

## BONES-SEED with EMA and no-sync training (8000 iterations)

Building on the fused 3×3 rotations and fused Adam already in this repository, an EMA policy (`pi_ema: 0.995`), no host-device syncs (`no_sync: true`), and `lr: 4e-4` reach a higher return in two-thirds of the iterations on the same one RTX 4090 (48 GB). Every checkpoint is evaluated on the complete 129,487-clip library.

| checkpoint | iterations | norm_return | frame-weighted | failed clips |
|---|---:|---:|---:|---:|
| ckpt_002000.pt | 2000 | 0.8092 | 0.8187 | 1901 / 129,487 (1.47%) |
| ckpt_004000.pt | 4000 | 0.9125 | 0.9165 | 660 / 129,487 (0.51%) |
| ckpt_006000.pt | 6000 | 0.9306 | 0.9342 | 206 / 129,487 (0.16%) |
| [ckpt_008000.pt](https://github.com/Motphys/Uni-SuperMimic/releases/download/checkpoints-v1/boneseed_ema_nosync_ckpt_008000.pt) | 8000 | 0.9395 | 0.9433 | 193 / 129,487 (0.15%) |

![EMA + no-sync + lr: full-library evaluation and training reward through iteration 8000](../assets/boneseed_optimized_curve.png)

At 8000 iterations this configuration already exceeds the 12000-iteration baseline on normalized return (0.9395 vs 0.9243) and halves the number of failed clips (193 vs 350).

## BONES-SEED with Muon hidden-layer optimization (8000 iterations)

The Muon optimizers update only the hidden `Linear.weight` matrices of the policy encoder/decoder and world-model MLP; input/output layers, vectors, normalizers, and the critic use Adam. Muon uses `match_rms_adamw` scaling, so the existing learning rate and cosine schedule are reused. Muon is now the default for both networks (`pi_opt: muon`, `wm_opt: muon`); Adam remains explicitly selectable.

The structured world model (`algo.wm_structured`) predicts velocity increments plus residual position corrections on top of a trapezoidal velocity drift, following InstantTrack's semi-implicit formulation. It does not change the policy input/output contract.

The three seed-0 runs use mjwarp, 4096 environments, the complete 129,487-clip BONES-SEED library, and the same schedule.

Muon reaches the Adam 8000-iteration macro return in 4000 iterations (0.939500 versus 0.939490), although its 423 failed clips are not a strict success-rate match for Adam's 193 at 8000. At 8000 iterations, Muon improves macro return by 0.0108 and reduces failed clips from 193 to 166.

Adding the structured world model improves macro and frame-weighted return at every checkpoint and greatly improves early success: failed clips drop from 1126 to 722 at 2000 iterations and from 423 to 242 at 4000. Its additional 3000-iteration checkpoint reaches 0.929833 macro return, 0.932974 frame-weighted return, and 297 failed clips. At 6000/8000 iterations it trades slightly more failures (195 versus 186, then 176 versus 166) for the higher mean return. Its 8000-iteration world-model 8-step error is 0.05635 versus 0.06252 for Muon (-9.9%), with only 0.54% lower measured throughput. The default recipe below adopts `wm_structured=true` together with the retuned 8k budget.

![Adam, Muon, and Muon + structured WM: full-library evaluation and learner wall time at 2000, 3000, 4000, 6000, and 8000 iterations](../assets/muon_vs_adam_evaluation.png)

## BONES-SEED default recipe, pre-CUDA-graph (8000 iterations)

The default training configuration moved to an 8000-iteration BONES-SEED budget. Relative to the previous defaults it changed `iterations` 12000 -> 8000, `lr_decay_iters` 250000 -> 12000, `wm_structured` false -> true, `wm_hidden` [2048, 2048, 2048] -> [1024, 1024, 1024], `pi_batch` 4096 -> 2048 with `wm_batches` 10 -> 5, and the base angular-velocity observation-noise half-width 0.05 -> 0.2 (gravity 0.05, joint_pos 0.01, joint_vel 0.5, and zero action noise unchanged).

A seed-0 comparison on one RTX 6000D (mjwarp, 4096 environments, the complete 129,487-clip library, deterministic full-library evaluation with 16384 environments):

| policy (8000 iterations) | norm return | frame-weighted | failed clips | learner wall |
|---|---:|---:|---:|---:|
| previous defaults with Adam override (`pi_opt=adam wm_opt=adam`) | 0.9073 | 0.9088 | 340 | 0.80 h |
| new defaults [ckpt_008000.pt](https://github.com/Motphys/Uni-SuperMimic/releases/download/checkpoints-v1/boneseed_default8k_ckpt_008000.pt) | 0.9320 | 0.9279 | 202 | 0.54 h |

The new defaults already pass the previous 8000-iteration return at iteration 4000 (0.9122, 391 failed clips). Attribution from existing runs and logs, without additional training: Muon plus the structured world model reproduce +0.044 training reward over Adam plus the unstructured model at 8000 iterations; the 1024-wide structured world model matches the 2048-wide one at that horizon (training reward 0.891 vs 0.890) and, together with the halved update batches, raises throughput from 2.85 to 4.09 it/s; the budget-aligned schedule anneals the policy by iteration 8000 (lr 1.09e-4 versus 3.99e-4 unannealed), and two 150k runs kept on the old 250k decay peaked near 0.91 around iteration 60000 before degrading to 0.25-0.64, so the shorter decay also removes a long-run instability. The recovered failures spread across motion families (jump 24, crawl 22, sit 15, dance 9, stand 9, push 8, kick 8); flips and rolls remain the hardest residual failures.

## Graph-captured default recipe on unilab 1.6.0 (8000 iterations)

The first results with `env.autoreset_cuda_graph` enabled, measured on unilab 1.6.0 before the learner launch-overhead reductions (#68–#70). Three seeds (mjwarp, 4096 environments, one RTX 4090 per seed, the complete 129,487-clip library); deterministic full-library evaluation with 16,384 environments from frame zero without noise. Training time is the learner wall time with the one-off compilation still included.

| policy (8000 iterations) | norm return | frame-weighted | failed clips | success rate | learner wall |
|---|---:|---:|---:|---:|---:|
| default recipe with `env.autoreset_cuda_graph` | 0.9322 | 0.9281 | 228 | 99.82% | 0.372 h (22.3 min) |

Steady-state learner throughput was ~6.2 it/s.

## Default recipe before the optimizer and collector CUDA graphs (8000 iterations)

Measured on main at `5cbbf13` (unilab 1.6.4, after #68–#72 and before the optimizer-step and collect-step CUDA graphs and the asynchronous iteration reads). Three seeds (mjwarp, 4096 environments, one RTX 4090 per seed, the complete 129,487-clip library); deterministic full-library evaluation with 16,384 environments from frame zero without noise. Training time is the learner wall time with the one-off compilation removed, as in the README.

| policy (8000 iterations) | norm return | frame-weighted | failed clips | success rate | training time (compile excluded) |
|---|---:|---:|---:|---:|---:|
| default recipe with `env.autoreset_cuda_graph` | 0.9314 | 0.9271 | 226 | 99.83% | 15.3 min |

Steady-state learner throughput was ~8.7 it/s.

[`perf/unilab-115-masked-autoreset-compile`]: https://github.com/Motphys/Uni-SuperMimic/pull/61
