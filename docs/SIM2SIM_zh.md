# 面向 Repro sim2sim 的部署导出

Languages: [English](SIM2SIM.md) | 简体中文

这个导出器为 Uni-SuperMimic checkpoint 生成显式契约的部署 bundle。它不改变训练、
评估或策略契约。生成的 artifacts 可由识别 `contract=unisupermimic_v1` 的
SuperMimic-Repro-Sim2Real 部署 runtime 消费。

## 契约

| 项 | Uni artifact |
|---|---:|
| 本体感知行 | 99 |
| 展平 observation | `10 x 99 = 990`，frame-major |
| goal | `10 x 73 = 730` |
| action | 29 个关节动作 |
| action target | `q0 + joint_offset + action_scale * action`；部署使用 `joint_offset=0` |

observation 行为：

```text
[q-q0-joint_offset, dq, body gyro, projected gravity, previous action, relative rotation 6D]
```

每个 clip 行保存 `MotionLib.goal(frame, 10)` 的精确结果，使用在 clip 末尾截断的
未来帧 `frame+1 ... frame+10`。导出器刻意不把这些行转换成 Repro native 的
68 列 goal 格式。

## 导出

在训练环境中运行导出器。`onnx` 与 `onnxruntime` 不是运行时强依赖，因此可使用
`uv run --with onnxruntime`，或提供已经安装它们的环境：

```bash
uv run --with onnxruntime python -m unisupermimic.deploy.sim2sim --cmd all \
  --checkpoint runs/supermimic_001/ckpt_012000.pt \
  --motion src/unisupermimic/assets/motions/g1/lafan1_supermimic/walk1_subject1.csv \
  --clip-name walk1_subject1 \
  --out deploy_artifacts_uni
```

紧凑 BONES-SEED 库采用 memory-map，且包含大量 clip。请显式指定要导出的 clip：

```bash
uv run --with onnxruntime python -m unisupermimic.deploy.sim2sim --cmd all \
  --checkpoint runs/supermimic_boneseed_001/ckpt_012000.pt \
  --motion data/library/bones_seed_compact.pt \
  --clip-name CLIP_NAME \
  --out deploy_artifacts_uni
```

输出包含：

```text
<out>/<checkpoint>/policy.onnx        # obs [1,990], goal [1,730] -> action [1,29]
<out>/<checkpoint>/policy_ir9.onnx    # 同一图的 ONNX IR 9 版本
<out>/<checkpoint>/deploy_config.json # 契约、关节顺序、PD 与限位
<out>/clips/<clip>.npz                # 精确 730 列 goal 与参考状态
<out>/models/g1_train_scene.xml       # same-model MJCF 与 assets
```

使用 `--cmd config`、`--cmd clips`、`--cmd onnx` 或 `--cmd model` 可重建已有
bundle 的部分内容。`--cmd check --check-clip PATH` 在导出后校验存储契约。

## Repro sim2sim

在支持 `unisupermimic_v1` runtime 的 SuperMimic-Repro-Sim2Real checkout 中，将
导出的 artifact 根目录和 clip 目录传给现有部署 gates：

```bash
python scripts/deploy/sim2sim.py --cmd g2,g3 \
  --ckpt ckpt_012000 \
  --artifacts /path/to/deploy_artifacts_uni \
  --clips walk1_subject1 \
  --tag uni_walk1 \
  --workers 8
```

不要在没有对应 `deploy_config.json` 和 Uni 导出 clip 的情况下，把 Uni policy 复制到
Repro native artifact 目录。Repro native 与 Uni 使用不同的 observation 宽度、
history layout 和 goal 表。
