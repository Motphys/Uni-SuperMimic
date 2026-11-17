# Uni-SuperMimic G1 Sim2Real Deployment (29-DoF, PC2, C++)

[中文](README_zh.md) | English

Reference sim2real deployment of a Uni-SuperMimic BONES-SEED policy on the
Unitree G1 (29-DoF). Inference chain: IMU/motor state → 99-D observation ×
10-frame history (frame-major, 990) + clip goal table (730) → ONNX policy
(ORT C++) → 29 joint position targets → low-level PD (500 Hz lowcmd).
Term-for-term identical to the training contract `unisupermimic_v1`.

**Damping is always available: `L2+B` enters DAMP (kp=0, kd=3.0) from any
state; `R1+A` resets to the initial PASSIVE state without restarting.**

## Supported assets

- **Weight**: `assets/policy.onnx` — Uni-SuperMimic trained on **BONES-SEED**
  (~129k retargeted G1 clips, 12,000 iterations), inputs obs[1,990] +
  goal[1,730], output action[1,29], IR9/opset18 (compatible with the PC2's
  onnxruntime 1.16).
- **Demo dataset** (`assets/clips/`, 12 BONES-SEED clips, goal tables
  bit-identical to the training `MotionLib.goal`):

  | # | Clip | Content |
  |---|---|---|
  | 1 | `cellphone_record_stand_idle_R_001__A423` | stand, holding a phone pose (10.2 s) |
  | 2 | `drinking_standing_mug_R_001__A282` | stand, drinking from a mug (11.0 s) |
  | 3 | `bow_R_001__A428` | bow in place (3.6 s) |
  | 4 | `clap_enthusiastic_001__A029` | enthusiastic clapping (5.0 s) |
  | 5 | `pels_air_punch_001__A492` | air punches (3.4 s) |
  | 6 | `fancy_spin_R_001__A428` | spin in place (4.1 s) |
  | 7 | `dance_basic_chaines_180_R_001__A306` | basic chaînés dance (4.1 s) |
  | 8 | `dance_basic_chaines_180_R_001__A307` | longer chaînés dance (10.4 s) |
  | 9 | `Loop_Forward_Walk_001__A019` | forward walk (4.3 s, 2.1 m) |
  | 10 | `Relaxed_walk_forward_001__A057` | relaxed long walk (14.8 s, 8.1 m) |
  | 11 | `Neutral_kick_trash_001__A057` | kicks (9.2 s, highly dynamic) |
  | 12 | `skipping_R_001__A359` | skipping forward (8.1 s, most dynamic) |

  All 12 pass the deployment-equivalent sim2sim (full operator flow, zero
  falls/aborts). The last two are the most dynamic — try them last.

## From clone to robot (PC2)

```bash
git clone https://github.com/Motphys/Uni-SuperMimic.git
cd Uni-SuperMimic/deploy/uni_g1

python3 tools/npz_to_bin.py            # build clips_bin from the demo npz (numpy only)
cd cpp
bash scripts/setup_ort.sh              # download ONNX Runtime C++ 1.16.3 (aarch64, online once)
bash scripts/build.sh                  # build check_numeric and g1_uni_ctrl (does NOT start anything)
bash scripts/run_numeric_check.sh      # MUST print NUMERIC PASS before any real-robot run
```

The numeric gate checks: asset/contract consistency; 97 golden vectors vs the
training MDP (obs ≤1e-4, goal ≤1e-4, action ≤1e-3, target formula ≤1e-5,
target end-to-end ≤1e-3); FSM walk-through (PASSIVE → MOVE_TO_START →
STANDBY → POLICY → DAMP → R1+A reset, gains verified).

### Run (operator starts manually)

Gantry/fall protection on, robot in development mode (L2+R2), stock
controller exited, `mode_machine=5` (checked at startup). Replace `eth0`
with the robot-facing NIC (not the Wi-Fi one).

```bash
cd Uni-SuperMimic/deploy/uni_g1/cpp
./build/g1_uni_ctrl eth0 ../assets                 # default clip: cellphone stand
./build/g1_uni_ctrl eth0 ../assets bow_R_001__A428 # or pick a clip by name
```

## Buttons and expected behavior (debounced, hold ~0.1 s)

| Button | Transition | Expected behavior | On trouble |
|---|---|---|---|
| `L2+B` | any → DAMP | whole body goes limp (kp=0, kd=3), latched | **first thing to verify after power-up** |
| `R1+A` | any (incl. DAMP) → reset | back to the initial PASSIVE state, no restart needed (re-hang first if fallen) | — |
| `L2+↑` | PASSIVE/STANDBY → MOVE_TO_START | robot straightens up smoothly on the harness (~4 s), facing its CURRENT heading, then auto STANDBY | unstable → L2+B |
| auto | → STANDBY | self-balancing stand, harness released, recovers from pushes | cannot stand → L2+B |
| `L2+←/→` | in STANDBY | terminal prints `clip -> <name>`, posture unchanged | — |
| `R1+X` | STANDBY → POLICY | plays the armed clip; walking starts from stand, arm motions catch up within ~2 s | bad start → L2+B |
| auto | POLICY → STANDBY | auto-returns to a self-balancing stand (facing the current heading) | — |
| `L2+↓` | STANDBY → PASSIVE | motors relax | — |
| `Ctrl+C` | any | DAMP for 1 s, then the process exits | — |

Automatic DAMP triggers: tilt deviation vs reference > 0.8, |dq| > 0.8 ×
velocity limit, |a| > 50, non-finite values.

## Layout

```
assets/    policy.onnx (weight), clips/*.npz (demo goal tables), configs, golden/ (acceptance vectors)
cpp/       C++ controller + offline numeric gate (build on PC2)
uni_deploy/  Python reference runtime (same semantics, locked by the golden vectors)
scripts/   run_sim.py (deployment-equivalent sim2sim), validate_bundle.py
tools/     pack_assets.py (regenerate golden on the training machine), npz_to_bin.py, sim_controller.py
```

**Adding a clip**: drop its goal-table npz into `assets/clips/`, run
`python3 tools/npz_to_bin.py`, and it is armed automatically (the controller
scans `clips_bin/` at startup — no code change). Regenerate golden with
`uv run python deploy/uni_g1/tools/pack_assets.py --checkpoint <ckpt.pt> --motion <compact.pt>`.

Derived artifacts are NOT in the repo (`.gitignore`): `assets/clips_bin/`
(`tools/npz_to_bin.py`), `assets/models/` (`python -m unisupermimic.deploy.export --cmd model`),
`reports/`, `cpp/build/`, `cpp/.deps/`, `manifest.json`.
