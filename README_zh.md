<h1 align="center">UniSuperMimic</h1>

<h3 align="center">基于 UniLab 的 SuperMimic 复现</h3>

<p align="center">语言：简体中文 | <a href="README.md">English</a></p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.28480"><img src="https://img.shields.io/badge/paper-arXiv--2610.28480-red" alt="论文"></a>
  <a href="https://pypi.org/project/unilab/"><img src="https://img.shields.io/pypi/v/unilab" alt="UniLab PyPI"></a>
  <a href="#bones-seed-动作库129487-段"><img src="https://img.shields.io/badge/BONES--SEED-129%2C487%20clips-blue" alt="BONES-SEED：129,487 段动作"></a>
  <a href="#结果"><img src="https://img.shields.io/badge/%E5%8D%95%E5%8D%A1-%E7%BA%A640%20%E5%88%86%E9%92%9F-green" alt="单卡：约 40 分钟"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 许可"></a>
</p>

UniSuperMimic 是 [SuperMimic](https://supermimic.github.io) 在 **UniLab** 上的独立下游复现，支持 **mjwarp、MuJoCo 和 Motrix** 后端，面向 Unitree G1 全身运动跟踪。仓库包含可微世界模型、目标条件策略、Manager-Based 任务、覆盖 40 段 LAFAN1 与 129,487 段 BONES-SEED 紧凑动作库的数据管线，以及训练和确定性全库评估入口。

### 特性亮点

- **大规模 BONES-SEED**：在同一个仓库内完成 129,487 段、4730 万帧动作库的下载、转换、过滤、打包、训练和确定性评估。
- **单卡高效**：mjwarp、MuJoCo 和 Motrix 三个后端都支持高效单卡工作流，约 40 分钟即可得到表现良好的 BONES-SEED checkpoint；下方同时给出 12000 轮完整验证结果。
- **三种仿真后端**：策略和数据集配置不变，可在 mjwarp、MuJoCo、Motrix 之间切换。

<p align="center">
  <a href="docs/assets/supermimic_comparison.mp4"><img src="docs/assets/supermimic_comparison.gif" alt="SuperMimic 与参考动作对比" width="95%"></a>
</p>

<p align="center"><em>实体机器人为 SuperMimic 策略，蓝色半透明机器人为参考动作。点击图片查看 MP4。</em></p>

论文：[*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*](https://arxiv.org/abs/2610.28480)，Qiayuan Liao 等。实现参考 `supermimic_vibed`，上游依赖为 PyPI `unilab`。

## 仓库结构

```text
unisupermimic/
├── src/unisupermimic/
│   ├── rl/
│   │   ├── algos/supermimic/       # 世界模型、策略、critic、replay 和 learner
│   │   └── contracts/supermimic.py # 环境/算法张量协议
│   ├── tasks/motion_tracking/g1/supermimic/
│   │   ├── env.py                  # Manager-Based 环境与后端工厂
│   │   ├── manager_terms.py        # command/action/observation/event/reward terms
│   │   ├── motion.py, mdp.py       # 运动库与可微 MDP
│   │   ├── evaluate.py             # 首帧确定性全 clip 评估
│   │   ├── render.py               # 策略采样与参考动作对比视频
│   │   └── registry.py             # G1SuperMimic 注册
│   ├── scripts/train_supermimic.py # 训练、续训和 eval 入口
│   ├── conf/supermimic/            # Hydra 配置
│   ├── data/boneseed.py            # BONES-SEED 下载/转换/过滤/打包命令行入口
│   ├── assets/robots/g1/           # G1 MJCF；mesh/URDF 下载后不纳入 git
│   ├── assets/motions/             # LAFAN1 CSV，下载后不纳入 git
│   └── sim2sim.py                  # 策略 I/O、网络形状和资产身份校验
├── docs/assets/                    # 结果曲线、MP4 和 GIF
├── tools/prepare_supermimic_assets.py
├── tests/
├── pyproject.toml, uv.lock
├── LICENSE, LICENSES/, NOTICE
└── Makefile
```

## 安装

环境：Linux、Python 3.10–3.13、[uv](https://docs.astral.sh/uv/)；mjwarp 训练需要 CUDA GPU。以下命令均在仓库根目录执行，`uv run` 按 `uv.lock` 同步项目、PyPI `unilab`、三种仿真后端和开发依赖。

```bash
uv run unisupermimic-train --help
```

依赖变更后运行 `uv lock`。

AMD ROCm 主机改用预生成的 ROCm lockfile 安装（同样的依赖钉版，torch 为
ROCm 构建）：

```bash
make sync-rocm
```

CUDA-only 后端（mjwarp）在 ROCm 上不可用，请使用 MuJoCo host-bridge 后端
训练（`task=g1_super_mimic/mujoco`）。`make sync-rocm` 会把
`pyproject.toml`/`uv.lock` 换成 ROCm 配置档，切回默认配置档用
`make sync-default`。维护者在依赖变更后用 `make lock-rocm` 重新生成
ROCm lockfile。

### 准备资产并选择数据集

两种训练方式使用同一个 G1 机器人模型，但**只需下载你准备用于训练的数据集**。LAFAN1 和 BONES-SEED 是两个独立选项，选择其中一个并不需要下载另一个。

### 共享 G1 机器人资产

如果只使用 BONES-SEED，可以只准备机器人，不下载任何 LAFAN1 动作：

```bash
uv run tools/prepare_supermimic_assets.py --robot-only

# 可选：离线校验已有的机器人资产
uv run tools/prepare_supermimic_assets.py --robot-only --validate-only
```

### 选项 A：LAFAN1

适合默认的 40 段动作训练。以下命令会同时准备共享机器人和 LAFAN1/G1 动作，无需再执行上面的 `--robot-only` 命令：

```bash
uv run tools/prepare_supermimic_assets.py

# 离线校验机器人和全部 LAFAN1 动作
uv run tools/prepare_supermimic_assets.py --validate-only
```

数据源为 `lvhaidong/LAFAN1_Retargeting_Dataset`。完整动作集包含 40 段 CSV、264705 个源帧，每帧 36 列。

### 选项 B：BONES-SEED

适合大规模动作训练。如果共享机器人尚未准备好，先执行上面的 `--robot-only` 命令，然后运行：

```bash
# 完整 BONES-SEED 流程：下载 -> 转换 -> 过滤 -> 打包
uv run unisupermimic-boneseed all --workers 16

# 已有原始 CSV 时，可断点续跑或单独执行各阶段
uv run unisupermimic-boneseed convert --workers 16
uv run unisupermimic-boneseed filter --workers 16
uv run unisupermimic-boneseed pack
```

该流程与 `SuperMimic-Reproduced` 一致：把 120 fps CSV 重采样为带 MuJoCo 正运动学结果的 50 Hz NPZ，执行名称与可行性过滤，再将保留的 clip 打包为 71 维 fp16 运动库。数据源是 Hugging Face `bones-studio/seed` 的 `g1.tar.gz`。

默认原始 CSV 目录为 `data/raw/bones_seed`，NPZ 与 manifest 目录为 `data/npz/bones_seed`，训练库输出为 `data/library/bones_seed_compact.pt`。转换会跳过已有 NPZ，`--overwrite` 可强制重建，`--limit` 可先做小规模试跑。参考全量结果约有 12.95 万段运动，fp16 库约 6.7 GB 并常驻训练设备，原始 CSV 和中间 NPZ 还需要额外磁盘空间。

mesh、URDF、动作文件、日志、checkpoint 和生成的数据集目录均被 `.gitignore` 排除；数据集许可证仍归各自所有者。

## 训练

以下命令从仓库根目录运行；使用 CUDA 时，把 `CUDA_VISIBLE_DEVICES` 改为实际空闲 GPU。两个数据集都使用 `unisupermimic-train`，通过运动文件配置选择对应的训练入口。

### 使用 LAFAN1 训练

LAFAN1 是默认数据集，无需覆盖运动文件：

```bash
# mjwarp、4096 个并行环境、12000 iterations
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  training.log_dir=runs/supermimic_mjwarp_001

# 从该 LAFAN1 run 的 last.pt 续训到 16000 iterations
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  algo.resume=true \
  algo.load_run=runs/supermimic_mjwarp_001 \
  algo.checkpoint=last.pt \
  algo.iterations=16000
```

### 使用 BONES-SEED 训练

生成紧凑运动库后，通过 `env.commands.motion.motion_files` 显式选择它。该参数会替换默认的 LAFAN1 列表，而不是把两个数据集合并：

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  'env.commands.motion.motion_files=[data/library/bones_seed_compact.pt]' \
  training.log_dir=runs/supermimic_boneseed_001

# 使用相同的数据集配置续训 BONES-SEED run
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  'env.commands.motion.motion_files=[data/library/bones_seed_compact.pt]' \
  algo.resume=true \
  algo.load_run=runs/supermimic_boneseed_001 \
  algo.checkpoint=last.pt \
  algo.iterations=16000
```

带引号的 Hydra 列表里是一个运动库文件，不代表只有一段动作；实际 clip 信息从库中读取。通过 `unisupermimic-train` 续训或评估 BONES-SEED run 时，也要保留这个数据集覆盖参数。

### 后端与其他选项

后端选择与数据集选择相互独立。以下示例使用默认 LAFAN1；如需 BONES-SEED，在命令中加上面的运动文件覆盖参数即可：

```bash
# 通过 task owner 选择 MuJoCo
uv run unisupermimic-train \
  task=g1_super_mimic/mujoco \
  training.log_dir=runs/supermimic_mujoco_001

# 通过 task owner 选择 Motrix
uv run unisupermimic-train \
  task=g1_super_mimic/motrix \
  training.log_dir=runs/supermimic_motrix_001
```

续训或评估 MuJoCo、Motrix run 时，使用与训练相同的 `task=g1_super_mimic/<backend>` 覆盖。Motrix 使用固定摩擦，保留质心和关节偏置域随机化。其他配置覆盖使用 Hydra 语法，例如快速 smoke 可覆盖 `algo.num_envs`、`algo.iterations`、`algo.wm_hidden`、`algo.policy_hidden` 和 `algo.compile=false`。默认完整配置见 `src/unisupermimic/conf/supermimic/config.yaml`。

## 评估

评估从每段动作的首帧开始，关闭观测/动作噪声，并跑完整个 clip。已结束的环境会被回收给待评估动作；调度器优先启动较长 clip，避免固定 batch 出现长空闲尾巴。独立评估器默认最多使用 16,384 个并行环境，可用 `unisupermimic-eval --num_envs N` 适配较小 GPU。严格 sim2sim 校验会在环境构造前检查策略 I/O、网络形状、运动库和 MJCF 身份。

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  training.play_only=true \
  algo.load_run=runs/supermimic_mjwarp_001 \
  algo.checkpoint=12000
```

结果写入同目录的 `ckpt_012000_eval.json`。也可直接评估指定 checkpoint。独立评估和视频导出入口支持 `--backend mjwarp`、`--backend mujoco`、`--backend motrix`，默认为 mjwarp：

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-eval \
  --backend mjwarp \
  --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt
```

### Viser 批量渲染

直接评估 checkpoint 时添加 `--viser`，即可在浏览器中同时查看多个确定性回放。该功能复用 UniLab 的 Viser 渲染器，无需本地显示器或 `MUJOCO_GL`：

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-eval \
  --backend mjwarp \
  --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt \
  --viser \
  --viser_max_envs 16 \
  --viser_port 8080
```

评估器会读取 checkpoint 所在目录的 `run_config.json`，复用其中记录的动作库，无需额外传入动作文件覆盖。上面的示例使用该 run 的 LAFAN1 动作库。若要渲染 BONES-SEED，改为选择 BONES-SEED checkpoint 即可；其 run 配置会自动选择 `data/library/bones_seed_compact.pt`：

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-eval \
  --backend mjwarp \
  --ckpt runs/supermimic_boneseed_001/ckpt_012000.pt \
  --viser \
  --viser_max_envs 16
```

如果 `run_config.json` 不存在，评估器回退到默认 LAFAN1 动作库。

打开 <http://localhost:8080>。**Controls** 面板可切换批量网格和单环境视图，并提供暂停、单步及变速控制。Viser 模式默认最多创建 16 个仿真环境；使用 `--num_envs N` 调整实际并行 batch 大小，使用 `--viser_max_envs N` 限制网格中显示的环境数。已结束的环境会装载待播放 clip，动作库全部播放完后循环回放各自分配的 clip。其他选项包括 `--render_spacing`、`--viser_display single`、`--viser_env_idx`、`--start_paused` 和 `--speed`。在终端按 Ctrl+C 停止服务。

对于远程 GPU 服务器，先执行 `ssh -L 8080:localhost:8080 <server>` 转发端口，再打开浏览器。Viser 用于交互式查看，不会写出 MP4；如需视频文件，请使用下面的视频导出入口。

## 视频导出

以下命令从 checkpoint 连续采样 `jumps1_subject1`，截取第 6–14 秒的跳跃片段，生成 25 fps 的参考动作对比 MP4：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run unisupermimic-video \
  --backend mjwarp \
  --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt \
  --clip jumps1_subject1 \
  --start 6 \
  --seconds 8 \
  --output docs/assets/supermimic_comparison.mp4
```

默认将参考动作的 root XY 对齐策略，便于观察全身跟踪；传入 `--global-ghost` 可保留参考动作的全局位移。

## 结果

### LAFAN1 动作库（40 段）

配置：PyPI `unilab 1.5.0`、mjwarp、4096 个训练环境。评估覆盖 40 段 LAFAN1/G1 运动；`norm_return` 是 clip 宏平均，`frame-weighted` 按帧数加权。

| checkpoint | norm_return | frame-weighted | 失败段数 |
|---|---:|---:|---:|
| ckpt_012000.pt | 0.827227 | 0.829952 | 0 / 40 |

![截至 12000 轮的训练与评估曲线](docs/assets/success_curve.png)

### BONES-SEED 动作库（129,487 段）

相同配置在 BONES-SEED G1 紧凑动作库（过滤后 129,487 段、4730 万帧、6.72 GB fp16）上验证：单张 RTX 4090（48 GB）、12000 轮、learner 墙钟 1.69 小时。评估覆盖过滤后全库并使用确定性整段回放。下表结果由早期的“每段一个并行环境”评估器产生；当前评估器会回收已结束环境。LAFAN1 行为相同配置的基线 run。

| 策略 | 评估集 | norm_return | frame-weighted | 失败段数 |
|---|---|---:|---:|---:|
| BONES-SEED ckpt_012000.pt | 全部 129,487 段 | 0.9243 | 0.9267 | 350 / 129,487（0.27%） |
| LAFAN1 ckpt_012000.pt（基线） | 全部 40 段 | 0.8336 | 0.8366 | 0 / 40 |

![截至 12000 轮的 BONES-SEED 与 LAFAN1 全库评估和训练曲线](docs/assets/boneseed_curve.png)

BONES-SEED 逐段分布：中位数 0.9528，P10 0.8337，P90 0.9841。BONES-SEED 收敛显著更快（2000 轮时早停率 0.83%，LAFAN1 同期 29.4%），训练奖励与评估回报均更高。另一次独立 RTX 5090 run 复现了这些数字（随机 1 万段子集 0.9262，与全库结果相差小于 0.002）。

### 加入 EMA 与 no-sync 训练的 BONES-SEED（8000 轮）

在仓库已有的融合 3×3 旋转和融合 Adam 基础上，EMA 策略（`pi_ema: 0.995`）、去除 host-device 同步（`no_sync: true`）、`lr: 4e-4`——用三分之二的迭代轮数，在同样单张 RTX 4090（48 GB）上达到了更高的回报。每个 checkpoint 都在完整的 129,487 段库上评估。

| checkpoint | 迭代轮数 | norm_return | 帧加权 | 失败段 |
|---|---|---:|---:|---:|
| ckpt_002000.pt | 2000 | 0.8092 | 0.8187 | 1901 / 129,487（1.47%） |
| ckpt_004000.pt | 4000 | 0.9125 | 0.9165 | 660 / 129,487（0.51%） |
| ckpt_006000.pt | 6000 | 0.9306 | 0.9342 | 206 / 129,487（0.16%） |
| ckpt_008000.pt | 8000 | 0.9395 | 0.9433 | 193 / 129,487（0.15%） |

![EMA + no-sync + lr：截至 8000 轮的全库评估与训练奖励](docs/assets/boneseed_optimized_curve.png)

在 8000 轮时，该配置已超过 12000 轮基线的归一化回报（0.9395 vs 0.9243），并将失败段数减半（193 vs 350）。

## 测试

```bash
make check       # ruff、mypy、pyright
make test        # 非 slow 测试
make test-all    # 包含真实后端 slow 测试；mjwarp 需要 CUDA
uv run tools/prepare_supermimic_assets.py --validate-only
```

## License

本仓库代码采用 [Apache-2.0](LICENSE)。参考实现的 MIT 许可及原始版权声明完整保存在 [LICENSES/MIT-supermimic.txt](LICENSES/MIT-supermimic.txt) 和 [LICENSES/MIT-supermimic-reproduced.txt](LICENSES/MIT-supermimic-reproduced.txt)，另见 [NOTICE](NOTICE)。UniLab、UniSim、MuJoCo/mjwarp、MotrixSim 和数据集分别遵循其上游许可；本仓库不重新许可这些外部依赖或下载资产。
