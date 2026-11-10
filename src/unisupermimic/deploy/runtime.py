"""NumPy deployment runtime for the ``unisupermimic_v1`` artifact contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

HISTORY = 10
NUM_JOINTS = 29
OBS_DIM = 99
GOAL_DIM = 730


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
