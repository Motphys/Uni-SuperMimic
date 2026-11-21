# Same-repository deployment export and sim2sim viewer

Languages: English | [简体中文](SIM2SIM_zh.md)

This module produces a contract-explicit deployment bundle for a Uni-SuperMimic
checkpoint and can replay that bundle through an interactive, same-repository
MuJoCo deployment viewer. It does not change training, evaluation, or the policy
contract. The generated artifacts can also be consumed by a deployment runtime
that recognizes `contract=unisupermimic_v1`.

## Contract

| item | Uni artifact |
|---|---:|
| proprioceptive row | 99 |
| flattened observation | `10 x 99 = 990`, frame-major |
| goal | `10 x 73 = 730` |
| action | 29 joint actions |
| action target | `q0 + joint_offset + action_scale * action`; deployment uses `joint_offset=0` |

The observation row is:

```text
[q-q0-joint_offset, dq, body gyro, projected gravity, previous action, relative rotation 6D]
```

Every clip row stores the exact result of `MotionLib.goal(frame, 10)`, using future
frames `frame+1 ... frame+10` clamped at the clip end. The exporter deliberately
does not convert these rows to Repro's native 68-column goal format.

## Export

Run the exporter in a training environment. `onnx` and `onnxruntime` are optional
at runtime, so use `uv run --with onnxruntime` or provide an environment that
already contains them:

```bash
uv run --with onnxruntime python -m unisupermimic.deploy.sim2sim --cmd all \
  --checkpoint runs/supermimic_001/ckpt_012000.pt \
  --motion src/unisupermimic/assets/motions/g1/lafan1_supermimic/walk1_subject1.csv \
  --clip-name walk1_subject1 \
  --out deploy_artifacts_uni
```

A compact BONES-SEED library is memory-mapped and contains many clips. Name the
clips to export explicitly:

```bash
uv run --with onnxruntime python -m unisupermimic.deploy.sim2sim --cmd all \
  --checkpoint runs/supermimic_boneseed_001/ckpt_012000.pt \
  --motion data/library/bones_seed_compact.pt \
  --clip-name CLIP_NAME \
  --out deploy_artifacts_uni
```

The output contains:

```text
<out>/<checkpoint>/policy.onnx        # obs [1,990], goal [1,730] -> action [1,29]
<out>/<checkpoint>/policy_ir9.onnx    # same graph at ONNX IR 9
<out>/<checkpoint>/deploy_config.json # contract, joint order, PD, and limits
<out>/clips/<clip>.npz                # exact 730-column goals and reference state
<out>/models/g1_train_scene.xml       # same-model MJCF and assets
```

Use `--cmd config`, `--cmd clips`, `--cmd onnx`, or `--cmd model` to regenerate
part of an existing bundle. `--cmd check --check-clip PATH` validates the stored
contract after export.

## Interactive sim2sim viewer

After training, open the native MuJoCo viewer directly from a checkpoint and one
motion source. The command creates a temporary deployment bundle, runs ONNX at
50 Hz, and advances the repository's G1 training MJCF at 1 kHz:

```bash
uv run --with onnxruntime python -m unisupermimic.deploy.sim2sim --cmd view \
  --checkpoint runs/supermimic_001/ckpt_012000.pt \
  --motion src/unisupermimic/assets/motions/g1/lafan1_supermimic/walk1_subject1.csv \
  --clip-name walk1_subject1 \
  --loop
```

For a compact BONES-SEED library, specify the clip explicitly:

```bash
uv run --with onnxruntime python -m unisupermimic.deploy.sim2sim --cmd view \
  --checkpoint runs/supermimic_boneseed_001/ckpt_012000.pt \
  --motion data/library/bones_seed_compact.pt \
  --clip-name CLIP_NAME \
  --loop
```

The solid robot is commanded by the exported ONNX policy. The translucent ghost
is a contact-free reference body. Useful options are:

```text
--start-frame N   start at a reference frame
--loop            restart after the final frame
--no-ghost        show only the policy robot
--headless        run without a window (also pass --max-steps N)
```

An already exported bundle can be viewed without recreating it:

```bash
uv run python -m unisupermimic.deploy.sim2sim --cmd view \
  --config deploy_artifacts_uni/ckpt_012000/deploy_config.json \
  --onnx deploy_artifacts_uni/ckpt_012000/policy.onnx \
  --clip deploy_artifacts_uni/clips/walk1_subject1.npz \
  --loop
```

This viewer uses the repository's own G1 MJCF and training position actuators.
It is a deployment-contract replay, not a substitute for staged real-robot safety
testing on an external gantry.

## Artifact safety

The viewer and exporter deliberately reject a policy, configuration, or clip whose
contract does not match `unisupermimic_v1`. Do not manually mix artifacts from a
different policy family: observation width, history layout, joint order, action
target formula, and goal layout must all match.
