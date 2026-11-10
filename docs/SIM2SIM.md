# Deployment export for Repro sim2sim

This exporter produces a contract-explicit deployment bundle for a Uni-SuperMimic
checkpoint. It does not change training, evaluation, or the policy contract. The
generated artifacts can be consumed by a SuperMimic-Repro-Sim2Real deployment
runtime that recognizes `contract=unisupermimic_v1`.

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
uv run --with onnxruntime python -m unisupermimic.deploy.export --cmd all \
  --checkpoint runs/supermimic_001/ckpt_012000.pt \
  --motion src/unisupermimic/assets/motions/g1/lafan1_supermimic/walk1_subject1.csv \
  --clip-name walk1_subject1 \
  --out deploy_artifacts_uni
```

A compact BONES-SEED library is memory-mapped and contains many clips. Name the
clips to export explicitly:

```bash
uv run --with onnxruntime python -m unisupermimic.deploy.export --cmd all \
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

## Repro sim2sim

From a SuperMimic-Repro-Sim2Real checkout with `unisupermimic_v1` runtime support,
pass the exported artifact root and clip directory to its existing deployment
gates:

```bash
python scripts/deploy/sim2sim.py --cmd g2,g3 \
  --ckpt ckpt_012000 \
  --artifacts /path/to/deploy_artifacts_uni \
  --clips walk1_subject1 \
  --tag uni_walk1 \
  --workers 8
```

Do not copy a Uni policy into a native Repro artifact directory without its
`deploy_config.json` and exported Uni clip. Repro native and Uni use different
observation widths, history layouts, and goal tables.
