"""Loading and validation of Uni-SuperMimic deployment assets.

Assets:
  assets/policy.onnx        obs [1,990], goal [1,730] -> action [1,29]
  assets/deploy_config.json joint constants, contract=unisupermimic_v1
  assets/clips/<name>.npz   goal table (T,730) + reference state (T,...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTRACT = "unisupermimic_v1"
NUM_JOINTS = 29
OBS_DIM = 99
HISTORY = 10
GOAL_FRAME_DIM = 73
FUTURE = 10


@dataclass
class DeployAssets:
    contract: str
    joint_names: list
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    velocity_limit: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    soft_limit_factor: float
    control_dt: float
    joint_offset: np.ndarray

    @property
    def soft_lower(self) -> np.ndarray:
        c = 0.5 * (self.joint_lower + self.joint_upper)
        h = 0.5 * (self.joint_upper - self.joint_lower) * self.soft_limit_factor
        return c - h

    @property
    def soft_upper(self) -> np.ndarray:
        c = 0.5 * (self.joint_lower + self.joint_upper)
        h = 0.5 * (self.joint_upper - self.joint_lower) * self.soft_limit_factor
        return c + h


def _vec(d: dict, key: str) -> np.ndarray:
    v = np.asarray(d[key], dtype=np.float64)
    if v.shape != (NUM_JOINTS,):
        raise ValueError(f"deploy_config {key}: expected ({NUM_JOINTS},), got {v.shape}")
    return v


def load_assets(config_path) -> DeployAssets:
    d = json.loads(Path(config_path).read_text())
    if d.get("contract") != CONTRACT:
        raise ValueError(f"deploy_config contract must be {CONTRACT}, got {d.get('contract')!r}")
    if int(d.get("obs_dim", OBS_DIM)) != OBS_DIM or int(d.get("history", HISTORY)) != HISTORY:
        raise ValueError("deploy_config obs_dim/history mismatch")
    if int(d.get("goal_frame_dim", GOAL_FRAME_DIM)) != GOAL_FRAME_DIM:
        raise ValueError("deploy_config goal_frame_dim mismatch")
    if int(d.get("future", FUTURE)) != FUTURE:
        raise ValueError("deploy_config future mismatch")
    joint_names = [str(x) for x in d["joint_names"]]
    if len(joint_names) != NUM_JOINTS:
        raise ValueError(f"expected {NUM_JOINTS} joints, got {len(joint_names)}")
    control_dt = float(d.get("control_dt", 0.02))
    if abs(control_dt - 0.02) > 1e-9:
        raise ValueError(f"control_dt must be 0.02, got {control_dt}")
    return DeployAssets(
        contract=CONTRACT,
        joint_names=joint_names,
        default_joint_pos=_vec(d, "default_joint_pos"),
        action_scale=_vec(d, "action_scale"),
        kp=_vec(d, "kp"),
        kd=_vec(d, "kd"),
        effort_limit=_vec(d, "effort_limit"),
        velocity_limit=_vec(d, "velocity_limit"),
        joint_lower=_vec(d, "joint_lower"),
        joint_upper=_vec(d, "joint_upper"),
        soft_limit_factor=float(d.get("soft_limit_factor", 0.9)),
        control_dt=control_dt,
        joint_offset=np.asarray(d.get("joint_offset", [0.0] * NUM_JOINTS), dtype=np.float64),
    )


@dataclass
class Clip:
    name: str
    goal: np.ndarray          # (T, FUTURE*GOAL_FRAME_DIM) float32, exact training window rows
    joint_pos: np.ndarray     # (T, 29)
    proj_grav: np.ndarray     # (T, 3)
    root_pos: np.ndarray      # (T, 3)
    root_quat_wxyz: np.ndarray  # (T, 4)
    root_lin_vel_w: np.ndarray  # (T, 3)
    root_ang_vel_w: np.ndarray  # (T, 3)
    joint_vel: np.ndarray     # (T, 29)
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.goal.shape[0])

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps


def load_clip(path) -> Clip:
    path = Path(path)
    with np.load(path) as z:
        files = set(z.files)
        required = {"name", "contract", "goal", "joint_pos", "proj_grav", "root_quat_wxyz"}
        missing = required - files
        if missing:
            raise ValueError(f"{path}: clip missing keys {sorted(missing)}")
        if str(z["contract"]) != CONTRACT:
            raise ValueError(f"{path}: clip contract must be {CONTRACT}")
        goal = np.asarray(z["goal"], dtype=np.float32)
        expected = FUTURE * GOAL_FRAME_DIM
        if goal.ndim != 2 or goal.shape[1] != expected:
            raise ValueError(f"{path}: goal must be (T, {expected}), got {goal.shape}")
        t = goal.shape[0]
        joint_pos = np.asarray(z["joint_pos"], dtype=np.float64)
        if joint_pos.shape != (t, NUM_JOINTS):
            raise ValueError(f"{path}: joint_pos must be (T, {NUM_JOINTS}), got {joint_pos.shape}")
        for key in ("goal", "joint_pos", "proj_grav", "root_quat_wxyz"):
            if not np.all(np.isfinite(z[key])):
                raise ValueError(f"{path}: {key} contains non-finite values")
        def _opt(key, cols, fill=0.0):
            if key in files:
                arr = np.asarray(z[key], dtype=np.float64)
                if arr.shape != (t, cols):
                    raise ValueError(f"{path}: {key} must be (T, {cols}), got {arr.shape}")
                return arr
            return np.full((t, cols), fill, dtype=np.float64)
        fps = float(np.asarray(z["fps"]).reshape(-1)[0]) if "fps" in files else 50.0
        return Clip(
            name=str(z["name"]),
            goal=goal,
            joint_pos=joint_pos,
            proj_grav=_opt("proj_grav", 3),
            root_pos=_opt("root_pos", 3),
            root_quat_wxyz=_opt("root_quat_wxyz", 4, fill=1.0),
            root_lin_vel_w=_opt("root_lin_vel_w", 3),
            root_ang_vel_w=_opt("root_ang_vel_w", 3),
            joint_vel=_opt("joint_vel", NUM_JOINTS),
            fps=fps,
        )


def _yaw_mat(rotation: np.ndarray) -> np.ndarray:
    """NumPy port of the training geometry.yaw_mat (clamped planar heading)."""
    fx, fy = rotation[0, 0], rotation[1, 0]
    norm = max(float(np.hypot(fx, fy)), 0.02)
    c, s = fx / norm, fy / norm
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _quat_to_mat(q_wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64) / np.linalg.norm(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_static_clip(assets: DeployAssets, root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                     root_pos=(0.0, 0.0, 0.76), name: str = "static_stand") -> Clip:
    """Frozen stand reference for the self-balancing STANDBY state.

    Goal rows are the training-consistent "future = same pose" window:
    position delta 0, heading-relative rotation 6D, zero velocities,
    default stand joint positions, zero joint rates.
    """
    R = _quat_to_mat(root_quat_wxyz)
    rot6 = (_yaw_mat(R).T @ R)[:, :2].reshape(-1)
    row = np.zeros(GOAL_FRAME_DIM, dtype=np.float64)
    row[3:9] = rot6
    row[15:44] = assets.default_joint_pos
    window = np.repeat(row[None], FUTURE, axis=0).reshape(-1).astype(np.float32)
    t = 2
    return Clip(
        name=name,
        goal=np.repeat(window[None], t, axis=0).astype(np.float32),
        joint_pos=np.repeat(assets.default_joint_pos[None], t, axis=0),
        proj_grav=np.tile((-_quat_to_mat(root_quat_wxyz))[:, 2], (t, 1)),
        root_pos=np.tile(np.asarray(root_pos, dtype=np.float64), (t, 1)),
        root_quat_wxyz=np.tile(np.asarray(root_quat_wxyz, dtype=np.float64), (t, 1)),
        root_lin_vel_w=np.zeros((t, 3)),
        root_ang_vel_w=np.zeros((t, 3)),
        joint_vel=np.zeros((t, NUM_JOINTS)),
        fps=50.0,
    )


def make_frozen_clip(assets: DeployAssets, clip: Clip, frame: int = 0,
                     name: str | None = None) -> Clip:
    """Frozen reference holding `clip` at `frame` (pre-play STANDBY).

    Same window semantics as make_static_clip but anchored to the clip's
    own frame pose/orientation, so switching into POLICY at that frame is
    continuous for the policy.
    """
    frame = int(np.clip(frame, 0, clip.num_frames - 1))
    R = _quat_to_mat(clip.root_quat_wxyz[frame])
    rot6 = (_yaw_mat(R).T @ R)[:, :2].reshape(-1)
    row = np.zeros(GOAL_FRAME_DIM, dtype=np.float64)
    row[3:9] = rot6
    row[15:44] = clip.joint_pos[frame]
    window = np.repeat(row[None], FUTURE, axis=0).reshape(-1).astype(np.float32)
    t = 2
    return Clip(
        name=name or f"{clip.name}@{frame}_frozen",
        goal=np.repeat(window[None], t, axis=0).astype(np.float32),
        joint_pos=np.repeat(clip.joint_pos[frame][None], t, axis=0),
        proj_grav=np.repeat(clip.proj_grav[frame][None], t, axis=0),
        root_pos=np.repeat(clip.root_pos[frame][None], t, axis=0),
        root_quat_wxyz=np.repeat(clip.root_quat_wxyz[frame][None], t, axis=0),
        root_lin_vel_w=np.zeros((t, 3)),
        root_ang_vel_w=np.zeros((t, 3)),
        joint_vel=np.zeros((t, NUM_JOINTS)),
        fps=clip.fps,
    )


def yaw_rebase_clip(clip: Clip, delta_yaw: float, name: str | None = None) -> Clip:
    """Rotate the clip's root trajectory by delta_yaw about z at its frame-0 position.

    The goal table is invariant under a global yaw rotation (all goal terms
    are heading-frame relative), so it is carried over unchanged; only the
    root state used by obs/ghost/termination is rebased. Frame 0 keeps its
    position; the trajectory shape is preserved.
    """
    c, s = np.cos(delta_yaw), np.sin(delta_yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p0 = clip.root_pos[0].copy()
    root_pos = p0 + (Rz @ (clip.root_pos - p0).T).T
    root_quat = np.stack([_mat_to_quat(Rz @ _quat_to_mat(q)) for q in clip.root_quat_wxyz])
    root_lin = (Rz @ clip.root_lin_vel_w.T).T
    root_ang = (Rz @ clip.root_ang_vel_w.T).T
    proj_grav = np.stack([-_quat_to_mat(q)[:, 2] for q in root_quat])
    return Clip(
        name=name or f"{clip.name}@rebased",
        goal=clip.goal.copy(),
        joint_pos=clip.joint_pos.copy(),
        proj_grav=proj_grav,
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        root_lin_vel_w=root_lin,
        root_ang_vel_w=root_ang,
        joint_vel=clip.joint_vel.copy(),
        fps=clip.fps,
    )


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2.0
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def yaw_of_quat(q_wxyz) -> float:
    R = _quat_to_mat(q_wxyz)
    return float(np.arctan2(R[1, 0], R[0, 0]))
