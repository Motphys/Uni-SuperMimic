# 本仓库内部署导出与 sim2sim 查看器

Languages: [English](SIM2SIM.md) | 简体中文

这个模块为 Uni-SuperMimic checkpoint 生成显式契约的部署 bundle，并能在本仓库内用
MuJoCo 交互查看器回放该 bundle。它不改变训练、评估或策略契约。生成的 artifacts
也可以交给识别 `contract=unisupermimic_v1` 的部署 runtime 消费。

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

## 交互式 sim2sim 查看器

训练完成后，可以直接从一个 checkpoint 和一个动作源打开原生 MuJoCo viewer。该命令
会创建临时部署 bundle，以 50 Hz 运行 ONNX，并以 1 kHz 推进本仓库的 G1 训练 MJCF：

```bash
uv run --with onnxruntime python -m unisupermimic.deploy.sim2sim --cmd view \
  --checkpoint runs/supermimic_001/ckpt_012000.pt \
  --motion src/unisupermimic/assets/motions/g1/lafan1_supermimic/walk1_subject1.csv \
  --clip-name walk1_subject1 \
  --loop
```

紧凑 BONES-SEED 库需要显式指定 clip：

```bash
uv run --with onnxruntime python -m unisupermimic.deploy.sim2sim --cmd view \
  --checkpoint runs/supermimic_boneseed_001/ckpt_012000.pt \
  --motion data/library/bones_seed_compact.pt \
  --clip-name CLIP_NAME \
  --loop
```

实心机器人由导出的 ONNX policy 控制；半透明 ghost 是无接触的参考机体。常用选项：

```text
--start-frame N   从指定参考帧开始
--loop            播完后重播
--no-ghost        只显示 policy 机器人
--headless        无窗口运行（还需 --max-steps N）
```

已有 bundle 可直接查看，无需重新导出：

```bash
uv run python -m unisupermimic.deploy.sim2sim --cmd view \
  --config deploy_artifacts_uni/ckpt_012000/deploy_config.json \
  --onnx deploy_artifacts_uni/ckpt_012000/policy.onnx \
  --clip deploy_artifacts_uni/clips/walk1_subject1.npz \
  --loop
```

该查看器使用本仓库自己的 G1 MJCF 和训练 position actuator。它是部署契约回放，
不能替代外部吊架下的分阶段实机安全测试。

## Artifact 安全

viewer 和导出器会显式拒绝与 `unisupermimic_v1` 不匹配的 policy、配置或 clip。
不要手工混用来自不同 policy family 的 artifacts：observation 宽度、history
layout、关节顺序、action target 公式和 goal layout 都必须一致。
