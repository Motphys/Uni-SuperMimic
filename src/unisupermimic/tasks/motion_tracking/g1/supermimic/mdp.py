"""Differentiable SuperMimic observations, goals, rewards, and terminations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, overload

import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    invariant_features,
    mat_to_6d,
    matmul,
    matvec,
    rot_angle_sq,
    yaw_mat,
)
from unisupermimic.rl.contracts.supermimic import State

from .kinematics import G1Kinematics
from .motion import CONTROL_DT, MotionLib

STATE_DIM = 71
NUM_J = 29
# Upstream-aligned sensor proprio: [gravity_b(3), base_ang_vel(3),
# joint_pos_rel(N), joint_vel(N), last_action(N)].
OBS_DIM = 3 + 3 + NUM_J + NUM_J + NUM_J
# Actor goal per frame: [joint_pos(N), joint_vel(N)] content plus the anchor's
# relative rot6d (6). The critic's wm goal additionally carries rel_pos(3).
GOAL_FRAME_DIM = 2 * NUM_J + 6
WM_GOAL_FRAME_DIM = 2 * NUM_J + 3 + 6
XI_DIM = 33
INV_DIM = 1 + 6 + 3 + 3 + 29 + 29

# Noise half-widths match upstream's proprio spec: gravity 0.05, ang_vel 0.2,
# joint_pos 0.01, joint_vel 0.5, actions 0.
OBS_NOISE = torch.cat(
    (
        torch.full((3,), 0.05),
        torch.full((3,), 0.05),
        torch.full((29,), 0.01),
        torch.full((29,), 0.5),
        torch.zeros(29),
    )
)
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_amp: float = 0.0
    amp_tol: float = 0.02
    amp_scale: float = 0.5
    amp_min_height: float = 0.8
    amp_ankles: tuple[int, int] = (3, 6)
    w_speed: float = 0.0
    speed_min: float = 2.0
    speed_tol: float = 0.3
    speed_scale: float = 1.5
    speed_bodies: tuple[int, int, int, int] = (3, 6, 9, 11)
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


class MDP:
    num_joints = NUM_J
    context_dim = XI_DIM
    obs_dim = OBS_DIM
    goal_frame_dim = GOAL_FRAME_DIM
    max_step_reward = MAX_STEP_REWARD

    def __init__(
        self,
        kin: G1Kinematics,
        motion: MotionLib,
        default_q: torch.Tensor,
        cfg: RewardCfg | None = None,
        goal_frames: int = 10,
    ):
        if default_q.shape != (NUM_J,):
            raise ValueError(
                f"SuperMimic default_q must have shape {(NUM_J,)}, got {default_q.shape}"
            )
        self.kin = kin
        self.motion = motion
        self.cfg = RewardCfg() if cfg is None else cfg
        if isinstance(goal_frames, bool) or goal_frames <= 0:
            raise ValueError(f"goal_frames must be positive, got {goal_frames}")
        self.goal_frames = goal_frames
        self.default_q = default_q
        middle = (kin.q_lo + kin.q_hi) / 2
        half_range = (kin.q_hi - kin.q_lo) / 2
        self.soft_lo = middle - half_range * self.cfg.soft_limit_factor
        self.soft_hi = middle + half_range * self.cfg.soft_limit_factor
        device = default_q.device
        self.obs_noise = OBS_NOISE.to(device)

    def to(self, device: str | torch.device) -> MDP:
        """Return a copy on ``device``; the original instance is left untouched."""
        device = torch.device(device)
        if device == self.default_q.device:
            return self
        new = copy.copy(self)
        new.kin = self.kin.to(device)
        new.motion = self.motion.to(device)
        for name in ("default_q", "soft_lo", "soft_hi", "obs_noise"):
            setattr(new, name, getattr(self, name).to(device))
        return new

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Return the upstream-aligned 93D noise-free sensor observation.

        Layout: ``[projected_gravity(3), base_ang_vel(3), joint_pos_rel(N),
        joint_vel(N), last_action(N)]`` — the same channel order and widths as
        the original implementation's proprio group.
        """
        _, rotation, _, angular_velocity, joint_pos, joint_vel = state
        rotation_t = rotation.transpose(-1, -2)
        gravity = -rotation[..., 2, :]
        return torch.cat(
            (
                gravity,
                matvec(rotation_t, angular_velocity),
                joint_pos - self.default_q,
                joint_vel,
                previous_action,
            ),
            -1,
        )

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2 * torch.rand_like(obs) - 1) * self.obs_noise

    def goal(self, state: State, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Actor goal: joint-target window plus anchor orientations relative to the root."""
        content, _, rel_ori = self._goal_parts(state, frames, num_frames)
        return torch.cat((content, rel_ori), -1)

    def _goal_parts(
        self, state: State, frames: torch.Tensor, num_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(content, rel_pos_b, rel_ori)`` of the goal window vs. the current root."""
        position, rotation = state[0], state[1]
        target_pos, target_rot = self.motion.goal_anchor(frames, num_frames)
        rotation_t = rotation.unsqueeze(-3)
        rel_pos = matvec(rotation_t, target_pos - position.unsqueeze(-2)).flatten(-2)
        rel_ori = mat_to_6d(matmul(rotation_t, target_rot)).flatten(-2)
        content = self.motion.goal_content(frames, num_frames)
        return content, rel_pos, rel_ori

    @overload
    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        terms: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        terms: Literal[True],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        terms: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compare a state with its reference and return the dt-weighted reward."""
        cfg, motion = self.cfg, self.motion
        position, rotation, linear_velocity, angular_velocity, joint_pos, joint_vel = state
        body_p, body_R, body_v, body_w = self.kin.forward(
            position,
            rotation,
            linear_velocity,
            angular_velocity,
            joint_pos,
            joint_vel,
        )
        ref_p, ref_R = motion.ref_state(frames)[:2]
        reference_body_p, reference_body_R, reference_body_v, reference_body_w = motion.body_state(
            frames
        )
        delta_rotation = yaw_mat(matmul(rotation, ref_R.transpose(-1, -2)))
        anchor = torch.cat((position[..., :2], ref_p[..., 2:3]), -1)
        ref_body_p = anchor.unsqueeze(-2) + matvec(
            delta_rotation.unsqueeze(-3), reference_body_p - ref_p.unsqueeze(-2)
        )
        ref_body_R = matmul(delta_rotation.unsqueeze(-3), reference_body_R)
        error_pos = ((ref_body_p - body_p) ** 2).sum(-1).mean(-1)
        error_ori = rot_angle_sq(ref_body_R, body_R).mean(-1)
        error_linvel = ((reference_body_v - body_v) ** 2).sum(-1).mean(-1)
        error_angvel = ((reference_body_w - body_w) ** 2).sum(-1).mean(-1)
        error_root_pos = ((ref_p - position) ** 2).sum(-1)
        error_root_ori = rot_angle_sq(ref_R, rotation)
        tracking = (
            torch.exp(-error_pos / cfg.body_pos_std**2)
            + torch.exp(-error_ori / cfg.body_ori_std**2)
            + torch.exp(-error_linvel / cfg.body_linvel_std**2)
            + torch.exp(-error_angvel / cfg.body_angvel_std**2)
            + 0.5 * torch.exp(-error_root_pos / cfg.root_pos_std**2)
            + 0.5 * torch.exp(-error_root_ori / cfg.root_ori_std**2)
        )
        if cfg.w_amp > 0.0:
            ankle_ids = list(cfg.amp_ankles)
            z_pol = body_p[..., ankle_ids, 2].max(dim=-1).values
            z_ref = reference_body_p[..., ankle_ids, 2].max(dim=-1).values
            amp_error = (z_pol - z_ref).abs().sub(cfg.amp_tol).clamp(min=0.0)
            amp = (1.0 - amp_error / cfg.amp_scale).clamp(0.0, 1.0)
            active = (z_ref > cfg.amp_min_height).float()
            tracking = tracking + cfg.w_amp * amp * active
        if cfg.w_speed > 0.0:
            speed_ids = list(cfg.speed_bodies)
            v_pol = body_v[..., speed_ids, :].norm(dim=-1)
            v_ref = reference_body_v[..., speed_ids, :].norm(dim=-1)
            speed_error = (v_pol - v_ref).abs().sub(cfg.speed_tol).clamp(min=0.0)
            speed = (1.0 - speed_error / cfg.speed_scale).clamp(0.0, 1.0)
            active = (v_ref > cfg.speed_min).float()
            tracking = tracking + cfg.w_speed * (speed * active).sum(-1) / active.sum(-1).clamp(
                min=1.0
            )
        action_rate = ((action - previous_action) ** 2).sum(-1)
        joint_limit = (
            torch.relu(joint_pos - self.soft_hi) + torch.relu(self.soft_lo - joint_pos)
        ).sum(-1)
        reward = (
            tracking + cfg.w_action_rate * action_rate + cfg.w_joint_limit * joint_limit
        ) * CONTROL_DT
        if terms:
            return reward, {
                "track": tracking * CONTROL_DT,
                "e_body_pos": error_pos,
                "e_body_ori": error_ori,
                "e_root_pos": error_root_pos,
                "action_rate": action_rate,
                "joint_limit": joint_limit,
            }
        return reward

    def termination(self, state: State, frames: torch.Tensor) -> torch.Tensor:
        """Return hard termination from root height and tilt deviations."""
        position, rotation = state[0], state[1]
        ref_p, ref_R = self.motion.ref_state(frames)[:2]
        error_height = (ref_p[..., 2] - position[..., 2]).abs()
        error_tilt = (ref_R[..., 2, 2] - rotation[..., 2, 2]).abs()
        return (error_height > self.cfg.term_height) | (error_tilt > self.cfg.term_tilt)

    def soft_termination(
        self,
        state: State,
        frames: torch.Tensor,
        height_temperature: float = 0.02,
        tilt_temperature: float = 0.04,
    ) -> torch.Tensor:
        """Return the differentiable imagined-rollout termination probability."""
        position, rotation = state[0], state[1]
        ref_p, ref_R = self.motion.ref_state(frames)[:2]
        error_height = (ref_p[..., 2] - position[..., 2]).abs()
        error_tilt = (ref_R[..., 2, 2] - rotation[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((error_height - self.cfg.term_height) / height_temperature)) * (
            1 - torch.sigmoid((error_tilt - self.cfg.term_tilt) / tilt_temperature)
        )
        return 1 - alive

    def critic_features(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return the upstream-aligned privileged critic input.

        Layout ``rep_state || wm_goal || sim_ctx``: the invariant state
        features, the full goal window (joint-target content plus the anchor's
        relative position and orientation), and the normalized sim context —
        mirroring the original critic observation group.
        """
        content, rel_pos, rel_ori = self._goal_parts(state, frames, self.goal_frames)
        return torch.cat(
            (
                invariant_features(state),
                content,
                rel_pos,
                rel_ori,
                normalized_context,
            ),
            -1,
        )

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                (context[..., :1] - 0.75) / 0.45,
                context[..., 1:30] / 0.01,
                context[..., 30:33] / 0.05,
            ),
            -1,
        )

    def valid_imagination_start(self, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        clip_end = self.motion.clip_end[self.motion.clip_id[frames]]
        return frames + horizon <= clip_end

    def ref_state(self, frames: torch.Tensor) -> State:
        return self.motion.ref_state(frames)

    def clip_info(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return owning clip, its inclusive bounds, and the frame's one-second bin."""
        clips = self.motion.clip_id[frames]
        return {
            "clip_id": clips,
            "clip_start": self.motion.clip_start[clips],
            "clip_end": self.motion.clip_end[clips],
            "frame_bin": self.motion.frame_bin[frames],
        }
