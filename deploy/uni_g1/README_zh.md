# Uni-SuperMimic G1 Sim2Real 部署（29-DoF，PC2，C++）

[English](README.md) | 中文

Uni-SuperMimic BONES-SEED 策略在 Unitree G1（29-DoF）上的 sim2real 参考部署。推理链路：IMU/电机状态 → 99 维观测 ×10 帧历史（frame-major, 990）+ clip goal 表（730）→ ONNX 策略（ORT C++）→ 29 维关节位置目标 → 低层 PD（500 Hz lowcmd）。与训练端契约 `unisupermimic_v1` 逐项一致。

**任何时候都可以下阻尼：任意状态下按 `L2+B` 立即进入 DAMP（kp=0, kd=3.0）；`R1+A` 复位回初始 PASSIVE，无需重启进程。**

## 支持的资产

- **权重**：`assets/policy.onnx` — Uni-SuperMimic 在 **BONES-SEED**（约 12.9 万条 G1 重定向 clip，12,000 次迭代）上训练；输入 obs[1,990] + goal[1,730]，输出 action[1,29]，IR9/opset18（兼容 PC2 的 onnxruntime 1.16）。
- **演示数据集**（`assets/clips/`，12 条 BONES-SEED clip，goal 表与训练 `MotionLib.goal` 逐位一致）：

  | # | clip | 内容 |
  |---|---|---|
  | 1 | `cellphone_record_stand_idle_R_001__A423` | 原地站立，持机姿势（10.2s） |
  | 2 | `drinking_standing_mug_R_001__A282` | 原地站立喝水（11.0s） |
  | 3 | `bow_R_001__A428` | 原地鞠躬（3.6s） |
  | 4 | `clap_enthusiastic_001__A029` | 原地热情鼓掌（5.0s） |
  | 5 | `pels_air_punch_001__A492` | 原地冲拳（3.4s） |
  | 6 | `fancy_spin_R_001__A428` | 原地转体（4.1s） |
  | 7 | `dance_basic_chaines_180_R_001__A306` | 基础链步舞蹈（4.1s） |
  | 8 | `dance_basic_chaines_180_R_001__A307` | 同系长版舞蹈（10.4s） |
  | 9 | `Loop_Forward_Walk_001__A019` | 前向步行（4.3s，2.1m） |
  | 10 | `Relaxed_walk_forward_001__A057` | 放松长步行（14.8s，8.1m） |
  | 11 | `Neutral_kick_trash_001__A057` | 站立踢（9.2s，动态大） |
  | 12 | `skipping_R_001__A359` | 蹦跳步前进（8.1s，动态最大） |

  全部 12 条均通过部署等价 sim2sim（完整操作流程，0 摔倒 0 abort）。后两条动态最大，建议最后试。

## 从克隆到上机（PC2）

```bash
git clone https://github.com/Motphys/Uni-SuperMimic.git
cd Uni-SuperMimic/deploy/uni_g1

python3 tools/npz_to_bin.py            # 由演示 npz 生成 clips_bin（仅需 numpy）
cd cpp
bash scripts/setup_ort.sh              # 下载 ONNX Runtime C++ 1.16.3（aarch64，仅首次需联网）
bash scripts/build.sh                  # 编译 check_numeric 和 g1_uni_ctrl（不启动任何东西）
bash scripts/run_numeric_check.sh      # 必须看到 NUMERIC PASS 才能上真机
```

数值门检查：资产/契约一致性；97 组 golden 向量逐项比对（obs ≤1e-4、goal ≤1e-4、action ≤1e-3、target 公式 ≤1e-5、target 端到端 ≤1e-3）；FSM 走查（PASSIVE → MOVE_TO_START → STANDBY → POLICY → DAMP → R1+A 复位，增益正确）。

### 实机运行（操作员手动启动）

做好吊挂/防坠保护，机器人进入开发模式（L2+R2），退出原厂运控，确认 `mode_machine=5`（启动时强制检查）。`eth0` 换成机器人通信网卡（不要用 Wi-Fi 网卡）。

```bash
cd Uni-SuperMimic/deploy/uni_g1/cpp
./build/g1_uni_ctrl eth0 ../assets                 # 默认动作：cellphone 站立
./build/g1_uni_ctrl eth0 ../assets bow_R_001__A428 # 或按名指定动作
```

## 手柄按键与对应现象（去抖，按住约 0.1s）

| 按键 | 状态变化 | 应当看到的现象 | 异常处理 |
|---|---|---|---|
| `L2+B` | 任意 → DAMP | 全身立即软下（kp=0, kd=3），锁定 | **上电先按这个验证阻尼** |
| `R1+A` | 任意（含 DAMP）→ 复位 | 回到初始 PASSIVE，无需重启（已摔倒请先人工扶起） | — |
| `L2+↑` | PASSIVE/STANDBY → MOVE_TO_START | 带绳平稳站直（~4s），**朝向=按键瞬间实际朝向**，自动进 STANDBY | 站不稳 → L2+B |
| 自动 | → STANDBY | 放开吊绳，无支撑自平衡站立，手推有恢复力 | 无法站稳 → L2+B |
| `L2+←/→` | STANDBY 中 | 终端打印 `clip -> <动作名>`，姿态不变 | — |
| `R1+X` | STANDBY → POLICY | 播放选中动作；步行从站姿起步，手臂动作 ~2s 内跟上 | 起步异常 → L2+B |
| 自动 | POLICY → STANDBY | 播完自动回自平衡站姿（朝向=当时朝向） | — |
| `L2+↓` | STANDBY → PASSIVE | 电机卸力 | — |
| `Ctrl+C` | 任意 | 先 DAMP 1s 再退进程 | — |

自动下阻尼的保护（任一触发即 DAMP）：相对参考倾角偏差 >0.8、|dq| > 0.8×速度限位、|a| > 50、出现非有限值。

## 目录结构

```
assets/    policy.onnx（权重）、clips/*.npz（演示 goal 表）、配置、golden/（验收向量）
cpp/       C++ 控制器 + 离线数值门（PC2 上构建）
uni_deploy/  Python 参考实现（与 C++ 语义由 golden 向量锁定一致）
scripts/   run_sim.py（部署等价 sim2sim）、validate_bundle.py
tools/     pack_assets.py（训练机上再生成 golden）、npz_to_bin.py、sim_controller.py
```

**新增动作**：把 goal 表 npz 放入 `assets/clips/`，执行 `python3 tools/npz_to_bin.py` 即可（控制器启动时自动扫描 `clips_bin/`，无需改代码）。golden 再生成：`uv run python deploy/uni_g1/tools/pack_assets.py --checkpoint <ckpt.pt> --motion <compact.pt>`。

派生/构建产物不入库（`.gitignore`）：`assets/clips_bin/`（`tools/npz_to_bin.py` 生成）、`assets/models/`（`python -m unisupermimic.deploy.export --cmd model` 生成）、`reports/`、`cpp/build/`、`cpp/.deps/`、`manifest.json`。
