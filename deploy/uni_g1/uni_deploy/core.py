"""Uni-SuperMimic deployment core: obs/goal construction, ONNX policy, safety.

Every formula mirrors the Uni training MDP term for term:
  mdp.obs  = [q - q0 - offset, dq, R^T @ w_world, -R[:, 2, :], a_prev,
              mat_to_6d(R^T @ R_ref_current)]
  mdp.goal = clip.goal[f]  (precomputed 10x73 training window row)
  target   = q0 + action_scale * a, clamped so |PD torque| <= torque_clamp x effort
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .contract import FUTURE, HISTORY, NUM_JOINTS, OBS_DIM, Clip, DeployAssets


def quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 0.5:
        raise ValueError("invalid quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_6d(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation, dtype=np.float64)[:, :2].reshape(-1)


def yaw_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """Upright quaternion carrying only the yaw of q_wxyz."""
    R = quat_to_mat(q_wxyz)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


@dataclass
class SafetyCfg:
    tilt_abort: float = 0.8          # training term_tilt: |R_ref[2,2] - R[2,2]|
    vel_abort_factor: float = 0.8    # |dq| > factor x velocity_limit
    action_abort: float = 50.0       # |a| max
    clamp_torque: bool = True        # clamp q_target so |PD torque| <= factor x effort
    torque_clamp_factor: float = 1.0
    clamp_soft_limits: bool = False  # DeployCore default off, like training


class OnnxPolicy:
    def __init__(self, path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        inputs = {i.name: tuple(i.shape) for i in self.session.get_inputs()}
        if inputs.get("obs") != (1, HISTORY * OBS_DIM) or inputs.get("goal") != (1, FUTURE * 73):
            raise ValueError(f"ONNX input contract mismatch: {inputs}")
        outputs = [tuple(o.shape) for o in self.session.get_outputs()]
        if outputs != [(1, NUM_JOINTS)]:
            raise ValueError(f"ONNX output contract mismatch: {outputs}")

    def __call__(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        out = self.session.run(
            None,
            {
                "obs": obs.reshape(1, -1).astype(np.float32),
                "goal": goal.reshape(1, -1).astype(np.float32),
            },
        )[0]
        return np.asarray(out[0], dtype=np.float64)


@dataclass
class DeployCore:
    assets: DeployAssets
    clip: Clip
    policy: OnnxPolicy
    safety: SafetyCfg = field(default_factory=SafetyCfg)

    def __post_init__(self):
        self.frame = 0
        self.hist = np.zeros((HISTORY, OBS_DIM), dtype=np.float64)
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self._warmed = False

    def reset(self, frame: int = 0):
        self.frame = int(np.clip(frame, 0, self.clip.num_frames - 1))
        self.hist = np.zeros((HISTORY, OBS_DIM), dtype=np.float64)
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self._warmed = False

    def switch_clip(self, clip, frame: int = 0, keep_history: bool = True):
        """Switch the reference clip mid-stream (STANDBY <-> POLICY).

        The observation history and last action are robot-state quantities and
        are kept continuous across the switch by default.
        """
        self.clip = clip
        self.frame = int(np.clip(frame, 0, clip.num_frames - 1))
        if not keep_history:
            self.hist = np.zeros((HISTORY, OBS_DIM), dtype=np.float64)
            self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
            self._warmed = False

    @property
    def clip_done(self) -> bool:
        return self.frame >= self.clip.num_frames - 1

    def obs_frame(self, q, dq, gyro_body, quat_wxyz) -> np.ndarray:
        R = quat_to_mat(quat_wxyz)
        f = min(self.frame, self.clip.num_frames - 1)
        R_ref = quat_to_mat(self.clip.root_quat_wxyz[f])
        rel6 = mat_to_6d(R.T @ R_ref)
        return np.concatenate(
            [
                q - self.assets.default_joint_pos - self.assets.joint_offset,
                dq,
                np.asarray(gyro_body, dtype=np.float64),
                -R[2, :],
                self.last_action,
                rel6,
            ]
        )

    def observation(self, q, dq, gyro_body, quat_wxyz) -> np.ndarray:
        row = self.obs_frame(q, dq, gyro_body, quat_wxyz)
        if not self._warmed:
            self.hist = np.repeat(row[None], HISTORY, axis=0)
            self._warmed = True
        else:
            self.hist = np.concatenate([self.hist[1:], row[None]], axis=0)
        return self.hist.reshape(-1)

    def goal(self) -> np.ndarray:
        return self.clip.goal[min(self.frame, self.clip.num_frames - 1)]

    def _check(self, q, dq, quat_wxyz, action):
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq)) and np.all(np.isfinite(action))):
            return "non_finite"
        f = min(self.frame, self.clip.num_frames - 1)
        R = quat_to_mat(quat_wxyz)
        R_ref = quat_to_mat(self.clip.root_quat_wxyz[f])
        if abs(R_ref[2, 2] - R[2, 2]) > self.safety.tilt_abort:
            return "tilt"
        if np.any(np.abs(dq) > self.safety.vel_abort_factor * self.assets.velocity_limit):
            return "joint_velocity"
        if np.max(np.abs(action)) > self.safety.action_abort:
            return "action_magnitude"
        return None

    def targets(self, action, q, dq) -> tuple:
        a = self.assets
        raw = a.default_joint_pos + a.action_scale * action
        q_t = raw.copy()
        if self.safety.clamp_torque:
            tau = self.safety.torque_clamp_factor * a.effort_limit
            lo = q + (-tau + a.kd * dq) / a.kp
            hi = q + (tau + a.kd * dq) / a.kp
            q_t = np.clip(q_t, lo, hi)
        if self.safety.clamp_soft_limits:
            q_t = np.clip(q_t, a.soft_lower, a.soft_upper)
        return q_t, int(np.count_nonzero(np.abs(q_t - raw) > 1e-9))

    def step(self, q, dq, gyro_body, quat_wxyz):
        """One 50 Hz control tick. Returns (q_target, abort_reason, action)."""
        obs = self.observation(q, dq, gyro_body, quat_wxyz)
        goal = self.goal()
        action = self.policy(obs, goal)
        abort = self._check(q, dq, quat_wxyz, action)
        q_target, _ = self.targets(action, q, dq)
        self.last_action = action.copy()
        self.frame += 1
        return q_target, abort, action

    def warmup(self, iterations: int = 3):
        z_obs = np.zeros(HISTORY * OBS_DIM, dtype=np.float32)
        z_goal = np.zeros(FUTURE * 73, dtype=np.float32)
        for _ in range(iterations):
            self.policy(z_obs, z_goal)


def time_policy(core: DeployCore, n: int = 200) -> dict:
    """Single-thread timing of one DeployCore step on random valid input."""
    core.reset()
    rng = np.random.default_rng(0)
    q = core.assets.default_joint_pos.copy()
    dq = np.zeros(NUM_JOINTS)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        core.step(q, dq, 0.01 * rng.standard_normal(3), quat)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {"p50_ms": samples[n // 2], "p99_ms": samples[int(n * 0.99)], "max_ms": samples[-1]}
