"""Export Uni-SuperMimic checkpoints and replay them in same-repository MuJoCo sim2sim.

The output bundle follows this deployment layout:

```
<out>/<ckpt_tag>/policy.onnx           # obs [1,990], goal [1,730] -> action [1,29]
<out>/<ckpt_tag>/deploy_config.json    # Uni joint constants + contract=unisupermimic_v1
<out>/models/g1_train_scene.xml        # same-model MJCF and meshes
<out>/clips/<clip>.npz                 # (T,730) exact MotionLib goal and reference state
```
"""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import mujoco
import mujoco.viewer
import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
    G1_JOINT_NAMES,
    _kd,
    _kp,
    action_scale,
    default_joint_pos_list,
    motor_of,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MJCF = REPO / "src/unisupermimic/assets/robots/g1/supermimic.xml"
DEFAULT_SCENE = REPO / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"
DEFAULT_ASSETS = DEFAULT_MJCF.parent / "assets"
OBS_DIM = 99
GOAL_FRAME_DIM = 73
GOAL_DIM = 730
NUM_JOINTS = 29


def checkpoint_tag(path: str | Path) -> str:
    return Path(path).stem


def _velocity_limit(joint_name: str) -> float:
    """Uni motor model's actuator velocity limit."""
    if "hip_pitch" in joint_name or "hip_yaw" in joint_name or "waist_yaw" in joint_name:
        return 32.0
    if "hip_roll" in joint_name or "knee" in joint_name:
        return 20.0
    if "wrist_pitch" in joint_name or "wrist_yaw" in joint_name:
        return 22.0
    return 37.0


def write_config(checkpoint: str | Path, out: Path, mjcf: str | Path = DEFAULT_MJCF) -> Path:
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    name_to_joint = {model.joint(i).name: i for i in range(model.njnt)}
    name_to_actuator = {model.actuator(i).name: i for i in range(model.nu)}
    lower, upper, armature = [], [], []
    for name in G1_JOINT_NAMES:
        if name not in name_to_joint:
            raise ValueError(f"{mjcf}: missing training joint {name}")
        jid = name_to_joint[name]
        lower.append(float(model.jnt_range[jid, 0]))
        upper.append(float(model.jnt_range[jid, 1]))
        armature.append(float(model.dof_armature[model.jnt_dofadr[jid]]))

    motors = [motor_of(name) for name in G1_JOINT_NAMES]
    for name, motor in zip(G1_JOINT_NAMES, motors):
        if name not in name_to_actuator:
            raise ValueError(f"{mjcf}: missing training position actuator {name}")
        aid = name_to_actuator[name]
        expected = (float(_kp(motor[2])), float(_kd(motor[2])), float(motor[0]))
        actual = (
            float(model.actuator_gainprm[aid, 0]),
            float(-model.actuator_biasprm[aid, 2]),
            float(model.actuator_forcerange[aid, 1]),
        )
        if not np.allclose(actual, expected, rtol=0, atol=1e-9):
            raise ValueError(f"{mjcf}: actuator {name} {actual} != robot constants {expected}")
    cfg = {
        "contract": "unisupermimic_v1",
        "source_checkpoint": str(checkpoint),
        "joint_names": G1_JOINT_NAMES,
        "default_joint_pos": default_joint_pos_list(),
        "action_scale": [float(action_scale(name)) for name in G1_JOINT_NAMES],
        "kp": [float(_kp(motor[2])) for motor in motors],
        "kd": [float(_kd(motor[2])) for motor in motors],
        "effort_limit": [float(motor[0]) for motor in motors],
        "armature": armature,
        "velocity_limit": [_velocity_limit(name) for name in G1_JOINT_NAMES],
        "joint_lower": lower,
        "joint_upper": upper,
        "soft_limit_factor": 0.9,
        "control_dt": 0.02,
        "obs_dim": OBS_DIM,
        "history": 10,
        "goal_frame_dim": GOAL_FRAME_DIM,
        "future": 10,
        "joint_offset": [0.0] * NUM_JOINTS,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "deploy_config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    return out / "deploy_config.json"


def load_policy(checkpoint: str | Path, device: str = "cpu") -> tuple[Policy, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    args = ckpt.get("args", {})
    required = {
        "goal_frames": 10,
        "hist": 10,
        "z_dim": 32,
    }
    for key, expected in required.items():
        value = args.get(key)
        if key not in args:
            raise ValueError(f"{checkpoint}: checkpoint args is missing algo.{key}")
        if value != expected:
            raise ValueError(f"{checkpoint}: expected algo.{key}={expected}, got {value!r}")
    if "pi" not in ckpt:
        raise ValueError(f"{checkpoint}: no 'pi' policy state dictionary")
    policy = Policy(
        obs_dim=OBS_DIM,
        goal_dim=GOAL_FRAME_DIM * int(args["goal_frames"]),
        hist=int(args["hist"]),
        z_dim=int(args["z_dim"]),
        num_joints=NUM_JOINTS,
        hidden=tuple(args.get("policy_hidden", (2048, 1024, 512, 256))),
    )
    policy.load_state_dict(ckpt["pi"], strict=True)
    return policy.eval(), dict(args)


class _OnnxPolicy(torch.nn.Module):
    def __init__(self, policy: Policy, history: int = 10):
        super().__init__()
        self.policy = policy
        self.history = history

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        obs_hist = obs.reshape(-1, self.history, OBS_DIM)
        return self.policy(obs_hist, goal)


def export_onnx(checkpoint: str | Path, out: Path, opset: int = 18) -> dict:
    try:
        import onnx as onnx_lib
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on export environment
        raise RuntimeError("ONNX export needs onnx and onnxruntime") from exc

    policy, _ = load_policy(checkpoint)
    destination = out / checkpoint_tag(checkpoint) / "policy.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    example = (torch.zeros(1, 990), torch.zeros(1, 730))
    with torch.no_grad():
        try:
            program = torch.onnx.export(
                _OnnxPolicy(policy).eval(),
                example,
                input_names=["obs", "goal"],
                output_names=["action"],
                opset_version=opset,
                dynamo=True,
                external_data=False,
            )
            assert program is not None
            program.save(str(destination), external_data=False)
        except TypeError:
            torch.onnx.export(
                _OnnxPolicy(policy).eval(),
                example,
                str(destination),
                input_names=["obs", "goal"],
                output_names=["action"],
                opset_version=opset,
            )

    model = onnx_lib.load(str(destination))
    onnx_lib.checker.check_model(model)
    ir9_destination = destination.with_name("policy_ir9.onnx")
    model.ir_version = 9
    onnx_lib.checker.check_model(model)
    onnx_lib.save(model, str(ir9_destination))
    shapes = []
    for value in (*model.graph.input, *model.graph.output):
        shapes.append((value.name, [d.dim_value for d in value.type.tensor_type.shape.dim]))
    if shapes != [("obs", [1, 990]), ("goal", [1, 730]), ("action", [1, 29])]:
        raise ValueError(f"bad ONNX I/O contract: {shapes}")

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(32, 990)).astype(np.float32)
    goal = rng.normal(size=(32, 730)).astype(np.float32)
    with torch.no_grad():
        torch_obs = torch.as_tensor(obs).reshape(32, 10, 99)
        ref = policy(torch_obs, torch.as_tensor(goal)).numpy()
    got = np.concatenate(
        [session.run(["action"], {"obs": o[None], "goal": g[None]})[0] for o, g in zip(obs, goal)]
    )
    max_diff = float(np.abs(ref - got).max())
    if max_diff > 2e-4:
        raise ValueError(f"ONNX/Torch max action diff {max_diff:.3e} exceeds 2e-4")
    return {
        "path": str(destination),
        "ir9_path": str(ir9_destination),
        "bytes": destination.stat().st_size,
        "opset": opset,
        "io": shapes,
        "n_random": 32,
        "max_abs_action_diff": max_diff,
    }


def export_clips(
    motion_files: Sequence[str | Path],
    out: Path,
    *,
    mjcf: str | Path = DEFAULT_MJCF,
    src_fps: float = 30.0,
    names: Sequence[str] | None = None,
    goal_frames: int = 10,
    chunk: int = 65536,
) -> list[Path]:
    kin = G1Kinematics(str(mjcf), "cpu")
    motion = MotionLib([str(path) for path in motion_files], kin, "cpu", src_fps=src_fps)
    selected = set(names) if names is not None else set(motion.names)
    unknown = sorted(selected - set(motion.names))
    if unknown:
        raise ValueError(f"requested clips not in source motion library: {unknown}")
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for clip_name in motion.names:
        if clip_name not in selected:
            continue
        ci = motion.names.index(clip_name)
        start, end = int(motion.clip_start[ci]), int(motion.clip_end[ci])
        goal_parts, ref_parts = [], []
        for lo in range(start, end + 1, chunk):
            frames = torch.arange(lo, min(lo + chunk, end + 1), dtype=torch.int64)
            goal_parts.append(motion.goal(frames, goal_frames).numpy().astype(np.float32))
            p, R, v, w, q, qd = motion.ref_state(frames)
            ref_parts.append(
                (
                    p.numpy(),
                    R.numpy(),
                    mat_to_quat(R).numpy(),
                    v.numpy(),
                    w.numpy(),
                    q.numpy(),
                    qd.numpy(),
                )
            )
        goal = np.concatenate(goal_parts)
        p, R, quat, v, w, q, qd = (
            np.concatenate([part[i] for part in ref_parts]) for i in range(7)
        )
        destination = clips_dir / f"{clip_name}.npz"
        np.savez(
            destination,
            name=np.array(clip_name),
            contract=np.array("unisupermimic_v1"),
            fps=np.float32(50.0),
            src_fps=np.float32(src_fps),
            joint_names=np.array(G1_JOINT_NAMES),
            default_joint_pos=np.asarray(default_joint_pos_list(), dtype=np.float32),
            goal=goal,
            joint_pos=q.astype(np.float32),
            proj_grav=(-R[:, 2, :]).astype(np.float32),
            root_pos=p.astype(np.float32),
            root_quat_wxyz=quat.astype(np.float32),
            root_lin_vel_w=v.astype(np.float32),
            root_ang_vel_w=w.astype(np.float32),
            joint_vel=qd.astype(np.float32),
        )
        written.append(destination)
        print(f"  {clip_name}: T={len(goal)} -> {destination}")
    return written


def export_model(out: Path, scene: Path = DEFAULT_SCENE, assets: Path = DEFAULT_ASSETS) -> Path:
    destination = out / "models"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene, destination / "g1_train_scene.xml")
    shutil.copy2(scene.parent / "supermimic.xml", destination / "supermimic.xml")
    if assets.is_dir():
        shutil.copytree(assets, destination / "assets", dirs_exist_ok=True)
    import mujoco

    mujoco.MjModel.from_xml_path(  # pyright: ignore[reportAttributeAccessIssue]
        str(destination / "g1_train_scene.xml")
    )
    return destination / "g1_train_scene.xml"


def check(out: Path, checkpoint: str | Path, clip: str | Path) -> dict:
    config = json.loads((out / checkpoint_tag(checkpoint) / "deploy_config.json").read_text())
    if config.get("contract") != "unisupermimic_v1":
        raise ValueError("deploy_config contract is not unisupermimic_v1")
    with np.load(clip) as archive:
        if str(archive["contract"]) != "unisupermimic_v1" or archive["goal"].shape[-1] != GOAL_DIM:
            raise ValueError(f"{clip}: invalid Uni clip contract")
        frames = int(archive["goal"].shape[0])
    policy, args = load_policy(checkpoint)
    return {
        "checkpoint": str(checkpoint),
        "goal_frames": args["goal_frames"],
        "history": args["hist"],
        "obs_flattened": OBS_DIM * args["hist"],
        "goal_flattened": GOAL_FRAME_DIM * args["goal_frames"],
        "clip": str(clip),
        "clip_frames": frames,
    }


# ============================================================================
# Deployment runtime
# ============================================================================

HISTORY = 10


def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a rotation matrix."""
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 0.5:
        raise ValueError(f"invalid quaternion: {quat}")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Return the world gravity direction in the body frame."""
    return quat_to_mat(quat_wxyz).T @ np.array([0.0, 0.0, -1.0])


def mat_to_6d(rotation: np.ndarray) -> np.ndarray:
    """Use the first two rotation columns as a continuous 6D orientation."""
    return np.asarray(rotation, dtype=np.float64)[:, :2].reshape(-1)


@dataclass
class SafetyConfig:
    clamp_soft_limits: bool = False
    clamp_torque: bool = True
    torque_clamp_factor: float = 1.5
    max_tilt_err: float = 0.35
    max_vel_frac: float = 0.8
    max_abs_action: float = 50.0


@dataclass
class DeployConfig:
    contract: str
    joint_names: list[str]
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    velocity_limit: np.ndarray
    soft_limit_factor: float = 0.9
    control_dt: float = 0.02
    joint_offset: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))

    @classmethod
    def load(cls, path: str | Path) -> "DeployConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("contract") != "unisupermimic_v1":
            raise ValueError(f"{path}: expected contract unisupermimic_v1")
        arrays = {
            name: np.asarray(data[name], dtype=np.float64)
            for name in (
                "default_joint_pos",
                "action_scale",
                "kp",
                "kd",
                "effort_limit",
                "joint_lower",
                "joint_upper",
                "velocity_limit",
                "joint_offset",
            )
        }
        config = cls(
            contract=str(data["contract"]),
            joint_names=list(data["joint_names"]),
            soft_limit_factor=float(data.get("soft_limit_factor", 0.9)),
            control_dt=float(data.get("control_dt", 0.02)),
            **arrays,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.joint_names) != NUM_JOINTS:
            raise ValueError(f"expected {NUM_JOINTS} joint names, got {len(self.joint_names)}")
        for name in (
            "default_joint_pos",
            "action_scale",
            "kp",
            "kd",
            "effort_limit",
            "joint_lower",
            "joint_upper",
            "velocity_limit",
            "joint_offset",
        ):
            value = getattr(self, name)
            if value.shape != (NUM_JOINTS,):
                raise ValueError(f"{name} must have shape ({NUM_JOINTS},), got {value.shape}")
        if not 0.0 < self.control_dt < 0.1:
            raise ValueError(f"invalid control_dt {self.control_dt}")

    @property
    def soft_lower(self) -> np.ndarray:
        middle = 0.5 * (self.joint_lower + self.joint_upper)
        half = 0.5 * (self.joint_upper - self.joint_lower)
        return middle - self.soft_limit_factor * half

    @property
    def soft_upper(self) -> np.ndarray:
        middle = 0.5 * (self.joint_lower + self.joint_upper)
        half = 0.5 * (self.joint_upper - self.joint_lower)
        return middle + self.soft_limit_factor * half


@dataclass
class UniClip:
    name: str
    goal: np.ndarray
    joint_pos: np.ndarray
    proj_grav: np.ndarray
    joint_vel: np.ndarray
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "UniClip":
        with np.load(path) as archive:
            files = set(archive.files)
            required = {
                "name",
                "contract",
                "goal",
                "joint_pos",
                "joint_vel",
                "proj_grav",
                "root_pos",
                "root_quat_wxyz",
                "root_lin_vel_w",
                "root_ang_vel_w",
                "joint_names",
            }
            missing = required - files
            if missing:
                raise ValueError(f"{path}: clip is missing {sorted(missing)}")
            if str(archive["contract"]) != "unisupermimic_v1":
                raise ValueError(f"{path}: expected contract unisupermimic_v1")
            clip = cls(
                name=str(archive["name"]),
                goal=archive["goal"].astype(np.float32),
                joint_pos=archive["joint_pos"].astype(np.float64),
                joint_vel=archive["joint_vel"].astype(np.float64),
                proj_grav=archive["proj_grav"].astype(np.float64),
                root_pos=archive["root_pos"].astype(np.float64),
                root_quat_wxyz=archive["root_quat_wxyz"].astype(np.float64),
                root_lin_vel_w=archive["root_lin_vel_w"].astype(np.float64),
                root_ang_vel_w=archive["root_ang_vel_w"].astype(np.float64),
            )
        if clip.goal.ndim != 2 or clip.goal.shape[1] != GOAL_DIM:
            raise ValueError(f"{path}: goal must have shape (T, {GOAL_DIM}), got {clip.goal.shape}")
        expected = (clip.goal.shape[0], NUM_JOINTS)
        if clip.joint_pos.shape != expected or clip.joint_vel.shape != expected:
            raise ValueError(f"{path}: joint reference shapes do not match goal table")
        reference_shapes = {
            "root_pos": 3,
            "root_quat_wxyz": 4,
            "root_lin_vel_w": 3,
            "root_ang_vel_w": 3,
        }
        for name, width in reference_shapes.items():
            if getattr(clip, name).shape != (clip.goal.shape[0], width):
                raise ValueError(f"{path}: invalid {name} shape {getattr(clip, name).shape}")
        return clip

    @property
    def num_frames(self) -> int:
        return int(self.goal.shape[0])


class PolicyProtocol(Protocol):
    def __call__(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray: ...


@dataclass
class StepResult:
    q_target: np.ndarray
    action: np.ndarray
    frame: int
    done: bool
    abort: str | None
    n_clamped: int


@dataclass
class UniDeployCore:
    config: DeployConfig
    policy: PolicyProtocol
    clip: UniClip
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def __post_init__(self) -> None:
        if self.config.contract != "unisupermimic_v1":
            raise ValueError("UniDeployCore requires contract unisupermimic_v1")
        if self.clip.goal.shape[-1] != GOAL_DIM:
            raise ValueError(f"Uni goal must have {GOAL_DIM} features")
        self.reset()

    def reset(self, start_frame: int = 0) -> None:
        self.frame = int(np.clip(start_frame, 0, self.clip.num_frames - 1))
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.history: np.ndarray | None = None

    def obs_frame(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        gyro: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        frame = min(self.frame, self.clip.num_frames - 1)
        relative_rotation = mat_to_6d(
            quat_to_mat(quat_wxyz).T @ quat_to_mat(self.clip.root_quat_wxyz[frame])
        )
        return np.concatenate(
            (
                q - self.config.default_joint_pos - self.config.joint_offset,
                dq,
                gyro,
                projected_gravity(quat_wxyz),
                self.last_action,
                relative_rotation,
            )
        )

    def observation(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        gyro: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        frame = self.obs_frame(q, dq, gyro, quat_wxyz)
        if self.history is None:
            self.history = np.repeat(frame[None], HISTORY, axis=0)
        else:
            self.history = np.concatenate((self.history[1:], frame[None]))
        return self.history.reshape(-1)

    def goal(self) -> np.ndarray:
        return self.clip.goal[min(self.frame, self.clip.num_frames - 1)]

    def targets(self, action: np.ndarray, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, int]:
        config = self.config
        raw = action * config.action_scale + config.default_joint_pos + config.joint_offset
        target = raw.copy()
        if self.safety.clamp_torque:
            torque = self.safety.torque_clamp_factor * config.effort_limit
            lower = q + (-torque + config.kd * dq) / config.kp
            upper = q + (torque + config.kd * dq) / config.kp
            target = np.clip(target, lower, upper)
        if self.safety.clamp_soft_limits:
            target = np.clip(target, config.soft_lower, config.soft_upper)
        return target, int(np.count_nonzero(np.abs(target - raw) > 1e-9))

    def _abort_reason(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        gyro: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> str | None:
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
            return "non-finite joint state"
        if not (np.all(np.isfinite(gyro)) and np.all(np.isfinite(quat_wxyz))):
            return "non-finite IMU"
        norm = float(np.linalg.norm(quat_wxyz))
        if abs(norm - 1.0) > 0.05:
            return f"bad quaternion norm {norm:.3f}"
        gravity = projected_gravity(quat_wxyz / norm)
        reference = self.clip.proj_grav[min(self.frame, self.clip.num_frames - 1)]
        denominator = np.linalg.norm(gravity) * np.linalg.norm(reference) + 1e-9
        cosine = float(np.clip(np.dot(gravity, reference) / denominator, -1.0, 1.0))
        tilt = float(np.arccos(cosine))
        if tilt > self.safety.max_tilt_err:
            return f"tilt error {np.degrees(tilt):.0f} deg"
        over = np.abs(dq) > self.safety.max_vel_frac * self.config.velocity_limit
        if np.any(over):
            joint = int(np.argmax(np.abs(dq) / self.config.velocity_limit))
            return f"joint velocity {self.config.joint_names[joint]} {dq[joint]:.1f} rad/s"
        return None

    def step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        gyro: np.ndarray,
        quat_wxyz: np.ndarray,
        *,
        advance: bool = True,
    ) -> StepResult:
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
        abort = self._abort_reason(q, dq, gyro, quat_wxyz)
        norm = float(np.linalg.norm(quat_wxyz))
        if np.isfinite(norm) and norm > 0.5:
            quat_wxyz = quat_wxyz / norm
        else:
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        obs = self.observation(q, dq, gyro, quat_wxyz)
        action = np.asarray(self.policy(obs, self.goal()), dtype=np.float64)
        if action.shape != (NUM_JOINTS,):
            raise ValueError(f"policy action must have shape ({NUM_JOINTS},), got {action.shape}")
        if not np.all(np.isfinite(action)):
            abort = abort or "non-finite action"
            action = np.zeros(NUM_JOINTS)
        elif np.max(np.abs(action)) > self.safety.max_abs_action:
            abort = abort or f"action magnitude {np.max(np.abs(action)):.1f}"
        target, n_clamped = self.targets(action, q, dq)
        self.last_action = action.copy()
        if advance:
            self.frame += 1
        done = advance and self.frame >= self.clip.num_frames - 1
        return StepResult(target, action, self.frame, done, abort, n_clamped)


class OnnxPolicy:
    def __init__(self, path: str | Path, threads: int = 1):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def __call__(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        outputs: Sequence[Any] = self.session.run(
            ["action"],
            {"obs": obs[None].astype(np.float32), "goal": goal[None].astype(np.float32)},
        )
        return outputs[0][0].astype(np.float64)


# ============================================================================
# Interactive MuJoCo viewer
# ============================================================================


def _short(name: str) -> str:
    return name.split("/")[-1]


def build_ghost_model(scene: Path, robot: Path) -> mujoco.MjModel:
    """Attach a contact-free visual reference robot to the deployment scene."""
    parent = mujoco.MjSpec.from_file(str(scene))
    child = mujoco.MjSpec.from_file(str(robot))
    frame = parent.worldbody.add_frame(name="reference_ghost")
    parent.attach(child, prefix="ref_", frame=frame)

    for key in list(parent.keys):
        parent.delete(key)
    for actuator in list(parent.actuators):
        if actuator.name.startswith("ref_"):
            parent.delete(actuator)
    for sensor in list(parent.sensors):
        if sensor.name.startswith("ref_"):
            parent.delete(sensor)
    for body in parent.bodies:
        if not body.name.startswith("ref_"):
            continue
        body.gravcomp = 1.0
        for joint in body.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                joint.damping = np.array([1000.0, 0.0, 0.0])
                joint.frictionloss = 1000.0
                joint.armature = 1000.0
        for geom in body.geoms:
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 1
            geom.rgba = [0.35, 0.65, 1.0, 0.25]
    return parent.compile()


class ViewerRobot:
    """Extract the state and command the actuators in the training joint order."""

    def __init__(self, model: mujoco.MjModel, config: DeployConfig, *, ghost: bool):
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config
        self.ghost = ghost
        names = [_short(model.joint(i).name) for i in range(model.njnt)]
        if len(config.joint_names) != len(set(config.joint_names)):
            raise ValueError("deploy_config joint_names contains duplicates")
        unknown = sorted(set(config.joint_names) - set(names))
        if unknown:
            raise ValueError(f"model is missing deployment joints: {unknown}")
        self.jid = np.array([names.index(name) for name in config.joint_names], dtype=int)
        self.qadr = model.jnt_qposadr[self.jid]
        self.dadr = model.jnt_dofadr[self.jid]

        free_joints = [
            i
            for i in range(model.njnt)
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and not names[i].startswith("ref_")
        ]
        if len(free_joints) != 1:
            raise RuntimeError("expected exactly one robot free joint")
        self.free = free_joints[0]
        self.fq = model.jnt_qposadr[self.free]
        self.fv = model.jnt_dofadr[self.free]

        transmission_joints = [int(model.actuator_trnid[i, 0]) for i in range(model.nu)]
        self.act = np.array([transmission_joints.index(j) for j in self.jid], dtype=int)
        if model.nu != len(config.joint_names):
            raise ValueError(f"expected {len(config.joint_names)} actuators, got {model.nu}")

        if ghost:
            ghost_names = [f"ref_{name}" for name in config.joint_names]
            missing = sorted(set(ghost_names) - set(names))
            if missing:
                raise RuntimeError(f"ghost model is missing joints: {missing}")
            self.ghost_jid = np.array([names.index(name) for name in ghost_names], dtype=int)
            self.ghost_qadr = model.jnt_qposadr[self.ghost_jid]
            ghost_free = [
                i
                for i in range(model.njnt)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and names[i].startswith("ref_")
            ]
            if len(ghost_free) != 1:
                raise RuntimeError("expected exactly one ghost free joint")
            self.ghost_fq = model.jnt_qposadr[ghost_free[0]]

    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self.data.qpos[self.qadr].copy()
        dq = self.data.qvel[self.dadr].copy()
        quat = self.data.qpos[self.fq + 3 : self.fq + 7].copy()
        gyro = self.data.qvel[self.fv + 3 : self.fv + 6].copy()
        return q, dq, gyro, quat

    def set_state(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.fq : self.fq + 3] = root_pos
        self.data.qpos[self.fq + 3 : self.fq + 7] = root_quat
        self.data.qpos[self.qadr] = joint_pos
        mujoco.mj_forward(self.model, self.data)

    def set_ghost(self, clip: UniClip, frame: int) -> None:
        if not self.ghost:
            return
        index = int(np.clip(frame, 0, clip.num_frames - 1))
        self.data.qpos[self.ghost_fq : self.ghost_fq + 3] = clip.root_pos[index]
        self.data.qpos[self.ghost_fq + 3 : self.ghost_fq + 7] = clip.root_quat_wxyz[index]
        self.data.qpos[self.ghost_qadr] = clip.joint_pos[index]
        self.data.qvel[self.ghost_fq : self.ghost_fq + 6] = 0.0
        self.data.qvel[self.model.jnt_dofadr[self.ghost_jid]] = 0.0

    def foot_clearance(self) -> float:
        lowest = np.inf
        for geom in range(self.model.ngeom):
            if not (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]):
                continue
            body_name = _short(self.model.body(self.model.geom_bodyid[geom]).name)
            if "ankle_roll" not in body_name:
                continue
            radius = float(self.model.geom_size[geom, 0])
            lowest = min(lowest, float(self.data.geom_xpos[geom, 2]) - radius)
        if not np.isfinite(lowest):
            raise RuntimeError("model has no ankle-roll collision geoms")
        return float(lowest)

    def place_on_ground(self) -> float:
        delta = -self.foot_clearance()
        self.data.qpos[self.fq + 2] += delta
        mujoco.mj_forward(self.model, self.data)
        return delta

    def command(self, target: np.ndarray) -> None:
        self.data.ctrl[self.act] = target


def launch(
    *,
    config_path: Path,
    policy_path: Path,
    clip_path: Path,
    scene: Path,
    robot: Path,
    start_frame: int = 0,
    loop: bool = False,
    ghost: bool = True,
    headless: bool = False,
    max_steps: int | None = None,
) -> None:
    """Launch a real-time ONNX deployment rollout in MuJoCo's native viewer."""
    if headless and max_steps is None:
        raise ValueError("--headless requires --max-steps")
    step_limit = max_steps if max_steps is not None else 0
    config = DeployConfig.load(config_path)
    policy = OnnxPolicy(policy_path, threads=1)
    clip = UniClip.load(clip_path)
    core = UniDeployCore(config, policy, clip)
    core.reset(start_frame)

    if ghost:
        model = build_ghost_model(scene, robot)
    else:
        model = mujoco.MjModel.from_xml_path(str(scene))  # pyright: ignore[reportAttributeAccessIssue]
    model.opt.timestep = 0.001
    robot_view = ViewerRobot(model, config, ghost=ghost)
    robot_view.set_state(
        clip.root_pos[start_frame],
        clip.root_quat_wxyz[start_frame],
        clip.joint_pos[start_frame],
    )
    robot_view.place_on_ground()
    robot_view.set_ghost(clip, start_frame)

    steps_per_policy_step = int(round(config.control_dt / model.opt.timestep))
    if steps_per_policy_step != 20:
        raise ValueError(f"expected 20 physics steps per policy step, got {steps_per_policy_step}")

    print(
        f"viewing {clip.name}: frame={start_frame}/{clip.num_frames - 1}, "
        f"duration={clip.num_frames * config.control_dt:.2f}s, loop={loop}, ghost={ghost}",
        file=sys.stderr,
    )

    def reset() -> None:
        core.reset(start_frame)
        robot_view.set_state(
            clip.root_pos[start_frame],
            clip.root_quat_wxyz[start_frame],
            clip.joint_pos[start_frame],
        )
        robot_view.place_on_ground()
        robot_view.set_ghost(clip, start_frame)

    def run_one() -> StepResult:
        q, dq, gyro, quat = robot_view.state()
        result = core.step(q, dq, gyro, quat)
        robot_view.command(result.q_target)
        for _ in range(steps_per_policy_step):
            mujoco.mj_step(model, robot_view.data)
        robot_view.set_ghost(clip, core.frame)
        mujoco.mj_forward(model, robot_view.data)
        return result

    if headless:
        for _ in range(step_limit):
            result = run_one()
            if result.abort:
                print(f"abort={result.abort} frame={core.frame}", file=sys.stderr)
                raise RuntimeError(result.abort)
            if result.done and not loop:
                break
        return

    with mujoco.viewer.launch_passive(model, robot_view.data) as view:
        policy_step = 0
        while view.is_running():
            result = run_one()
            view.sync()

            if result.abort:
                print(f"abort={result.abort} frame={core.frame}", file=sys.stderr)
                break
            if result.done:
                if not loop:
                    break
                reset()
            policy_step += 1
            if max_steps is not None and policy_step >= max_steps:
                break
            time.sleep(config.control_dt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmd",
        default="all",
        help="config | clips | onnx | model | all | check | view",
    )
    parser.add_argument("--checkpoint", type=Path, help="Uni ckpt_XXXXXX.pt / last.pt")
    parser.add_argument("--out", type=Path, default=REPO / "deploy_artifacts_uni")
    parser.add_argument("--motion", nargs="+", type=Path, help="CSV motion(s) or one compact .pt")
    parser.add_argument(
        "--clip-name",
        action="append",
        help="clip to export; repeatable; default all CSV clips / all is too large for compact PT",
    )
    parser.add_argument("--src-fps", type=float, default=30.0)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check-clip", type=Path)
    parser.add_argument("--config", type=Path, help="existing deploy_config.json for --cmd view")
    parser.add_argument("--onnx", type=Path, help="existing policy.onnx for --cmd view")
    parser.add_argument("--clip", type=Path, help="existing exported clip .npz for --cmd view")
    parser.add_argument("--robot", type=Path, help="robot MJCF used to build the visual ghost")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--loop", action="store_true", help="restart the clip in --cmd view")
    parser.add_argument(
        "--no-ghost",
        action="store_true",
        help="do not attach the visual reference robot in --cmd view",
    )
    parser.add_argument("--headless", action="store_true", help="run --cmd view without a window")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="stop --cmd view after this many policy steps; required by --headless",
    )
    args = parser.parse_args()
    cmds = set(args.cmd.split(","))
    if cmds == {"all"}:
        cmds = {"config", "clips", "onnx", "model"}
    if not cmds <= {"config", "clips", "onnx", "model", "all", "check", "view"}:
        parser.error(f"unknown command in {args.cmd}")
    if "view" in cmds and len(cmds) != 1:
        parser.error("--cmd view cannot be combined with other commands")
    if "view" in cmds and args.headless and args.max_steps is None:
        parser.error("--headless requires --max-steps")
    if cmds & {"config", "onnx", "check"} and args.checkpoint is None:
        parser.error("--checkpoint is required for config/onnx/check")
    if "clips" in cmds and not args.motion:
        parser.error("--motion is required for clips")
    needs_compact_selection = (
        cmds != {"check"} and "clips" in cmds and args.motion[0].suffix == ".pt"
    )
    if needs_compact_selection and not args.clip_name:
        parser.error("compact .pt export requires one or more --clip-name values")

    if "view" in cmds:
        existing_artifacts = (
            args.config is not None or args.onnx is not None or args.clip is not None
        )
        if existing_artifacts:
            if not (args.config is not None and args.onnx is not None and args.clip is not None):
                parser.error("--cmd view needs --config, --onnx, and --clip together")
            launch(
                config_path=args.config,
                policy_path=args.onnx,
                clip_path=args.clip,
                scene=args.scene,
                robot=args.robot or args.mjcf,
                start_frame=args.start_frame,
                loop=args.loop,
                ghost=not args.no_ghost,
                headless=args.headless,
                max_steps=args.max_steps,
            )
        else:
            if args.checkpoint is None or not args.motion or len(args.clip_name or []) != 1:
                parser.error(
                    "--cmd view needs either existing artifacts or "
                    "--checkpoint, --motion, and exactly one --clip-name"
                )
            with tempfile.TemporaryDirectory(prefix="unisupermimic-sim2sim-") as temporary:
                temporary_out = Path(temporary)
                config_out = temporary_out / checkpoint_tag(args.checkpoint)
                write_config(args.checkpoint, config_out, args.mjcf)
                export_onnx(args.checkpoint, temporary_out, args.opset)
                exported = export_clips(
                    args.motion,
                    temporary_out,
                    mjcf=args.mjcf,
                    src_fps=args.src_fps,
                    names=args.clip_name,
                )
                launch(
                    config_path=config_out / "deploy_config.json",
                    policy_path=temporary_out / checkpoint_tag(args.checkpoint) / "policy.onnx",
                    clip_path=exported[0],
                    scene=args.scene,
                    robot=args.robot or args.mjcf,
                    start_frame=args.start_frame,
                    loop=args.loop,
                    ghost=not args.no_ghost,
                    headless=args.headless,
                    max_steps=args.max_steps,
                )
        return

    report: dict[str, object] = {}
    if "model" in cmds:
        report["model"] = str(export_model(args.out, args.scene, args.assets))
    if "config" in cmds:
        config_out = args.out / checkpoint_tag(args.checkpoint)
        report["config"] = str(write_config(args.checkpoint, config_out, args.mjcf))
    if "clips" in cmds:
        report["clips"] = [
            str(path)
            for path in export_clips(
                args.motion,
                args.out,
                mjcf=args.mjcf,
                src_fps=args.src_fps,
                names=args.clip_name,
            )
        ]
    if "onnx" in cmds:
        report["onnx"] = export_onnx(args.checkpoint, args.out, args.opset)
    if "check" in cmds:
        if args.check_clip is None:
            parser.error("--check-clip is required for check")
        report["check"] = check(args.out, args.checkpoint, args.check_clip)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
