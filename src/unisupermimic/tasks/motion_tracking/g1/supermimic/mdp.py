"""Differentiable SuperMimic observations, goals, rewards, and terminations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, overload

import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    invariant_features,
    mat_to_6d,
    rot_angle_sq,
    yaw_mat,
)
from unisupermimic.rl.contracts.supermimic import State

from .kinematics import G1Kinematics
from .motion import CONTROL_DT, MotionLib

STATE_DIM = 71
NUM_J = 29
OBS_DIM = 29 + 29 + 3 + 3 + 29 + 6
GOAL_FRAME_DIM = 73
XI_DIM = 33
INV_DIM = 1 + 6 + 3 + 3 + 29 + 29
CRITIC_EXTRA = 29 + 14 * 3 + 3 + 6 + XI_DIM

OBS_NOISE = torch.cat(
    (
        torch.full((29,), 0.01),
        torch.full((29,), 0.5),
        torch.full((3,), 0.2),
        torch.full((3,), 0.05),
        torch.zeros(29),
        torch.full((6,), 0.05),
    )
)
OBS_SCALE = torch.cat(
    (
        torch.ones(29),
        torch.full((29,), 0.1),
        torch.full((3,), 0.25),
        torch.ones(3),
        torch.ones(29),
        torch.ones(6),
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
    ):
        if default_q.shape != (NUM_J,):
            raise ValueError(
                f"SuperMimic default_q must have shape {(NUM_J,)}, got {default_q.shape}"
            )
        self.kin = kin
        self.motion = motion
        self.cfg = RewardCfg() if cfg is None else cfg
        self.default_q = default_q
        middle = (kin.q_lo + kin.q_hi) / 2
        half_range = (kin.q_hi - kin.q_lo) / 2
        self.soft_lo = middle - half_range * self.cfg.soft_limit_factor
        self.soft_hi = middle + half_range * self.cfg.soft_limit_factor
        device = default_q.device
        self.obs_noise = OBS_NOISE.to(device)
        self.obs_scale = OBS_SCALE.to(device)

    def to(self, device: str | torch.device) -> MDP:
        """Return a copy on ``device``; the original instance is left untouched."""
        device = torch.device(device)
        if device == self.default_q.device:
            return self
        new = copy.copy(self)
        new.kin = self.kin.to(device)
        new.motion = self.motion.to(device)
        for name in ("default_q", "soft_lo", "soft_hi", "obs_noise", "obs_scale"):
            setattr(new, name, getattr(self, name).to(device))
        return new

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 99D noise-free proprioceptive observation."""
        _, rotation, _, angular_velocity, joint_pos, joint_vel = state
        rotation_t = rotation.transpose(-1, -2)
        default_offset = context[..., 1:30]
        gravity = -rotation[..., 2, :]
        relative_rotation = mat_to_6d(rotation_t @ self.motion.root_R[frames])
        return torch.cat(
            (
                joint_pos - self.default_q - default_offset,
                joint_vel,
                (rotation_t @ angular_velocity.unsqueeze(-1)).squeeze(-1),
                gravity,
                previous_action,
                relative_rotation,
            ),
            -1,
        )

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2 * torch.rand_like(obs) - 1) * self.obs_noise

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        return self.motion.goal(frames, num_frames)

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
        ref_p, ref_R = motion.root_p[frames], motion.root_R[frames]
        delta_rotation = yaw_mat(rotation @ ref_R.transpose(-1, -2))
        anchor = torch.cat((position[..., :2], ref_p[..., 2:3]), -1)
        ref_body_p = anchor.unsqueeze(-2) + (
            delta_rotation.unsqueeze(-3)
            @ (motion.body_p[frames] - ref_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        ref_body_R = delta_rotation.unsqueeze(-3) @ motion.body_R[frames]
        error_pos = ((ref_body_p - body_p) ** 2).sum(-1).mean(-1)
        error_ori = rot_angle_sq(ref_body_R, body_R).mean(-1)
        error_linvel = ((motion.body_v[frames] - body_v) ** 2).sum(-1).mean(-1)
        error_angvel = ((motion.body_w[frames] - body_w) ** 2).sum(-1).mean(-1)
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
        ref_p, ref_R = self.motion.root_p[frames], self.motion.root_R[frames]
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
        ref_p, ref_R = self.motion.root_p[frames], self.motion.root_R[frames]
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
        """Return invariant state, goal errors, and normalized simulation context."""
        motion = self.motion
        position, rotation, linear_velocity, angular_velocity, joint_pos, joint_vel = state
        heading_t = yaw_mat(rotation).transpose(-1, -2)
        ref_p, ref_R = motion.root_p[frames], motion.root_R[frames]
        body_p = self.kin.forward(
            position,
            rotation,
            linear_velocity,
            angular_velocity,
            joint_pos,
            joint_vel,
        )[0]
        delta_rotation = yaw_mat(rotation @ ref_R.transpose(-1, -2))
        anchor = torch.cat((position[..., :2], ref_p[..., 2:3]), -1)
        ref_body_p = anchor.unsqueeze(-2) + (
            delta_rotation.unsqueeze(-3)
            @ (motion.body_p[frames] - ref_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        body_error = (
            (heading_t.unsqueeze(-3) @ (ref_body_p - body_p).unsqueeze(-1)).squeeze(-1).flatten(-2)
        )
        root_error = (heading_t @ (ref_p - position).unsqueeze(-1)).squeeze(-1)
        relative_rotation = mat_to_6d(rotation.transpose(-1, -2) @ ref_R)
        return torch.cat(
            (
                invariant_features(state),
                motion.q[frames] - joint_pos,
                body_error,
                root_error,
                relative_rotation,
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
