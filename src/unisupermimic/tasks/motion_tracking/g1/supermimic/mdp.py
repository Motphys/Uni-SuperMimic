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
from .motion import (
    CONTROL_DT,
    GOAL_FRAME_DIM,
    MotionLib,
    ReferenceStatePack,
    ReferenceTimeline,
)

STATE_DIM = 71
NUM_J = 29
OBS_DIM = 29 + 29 + 3 + 3 + 29 + 6
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
        reference_rotation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the 99D noise-free proprioceptive observation.

        ``reference_rotation`` is an optional prefetched ``root_R`` batch. It
        keeps rollout observation construction from revisiting the motion cache.
        """
        _, rotation, _, angular_velocity, joint_pos, joint_vel = state
        rotation_t = rotation.transpose(-1, -2)
        default_offset = context[..., 1:30]
        gravity = -rotation[..., 2, :]
        if reference_rotation is None:
            reference_rotation = self.motion.ref_state_pack(frames).root_R
        relative_rotation = mat_to_6d(matmul(rotation_t, reference_rotation))
        return torch.cat(
            (
                joint_pos - self.default_q - default_offset,
                joint_vel,
                matvec(rotation_t, angular_velocity),
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
        reference: ReferenceStatePack | None = None,
        reference_body_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
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
        reference = reference if reference is not None else motion.ref_state_pack(frames)
        ref_p, ref_R = reference.root_p, reference.root_R
        if reference_body_state is None:
            reference_body_state = motion.body_state(frames)
        reference_body_p, reference_body_R, reference_body_v, reference_body_w = (
            reference_body_state
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

    def _termination_errors(
        self,
        state: State,
        reference: ReferenceStatePack | None = None,
        frames: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position, rotation = state[0], state[1]
        if reference is None:
            if frames is None:
                raise ValueError("termination queries need either reference or frames")
            reference = self.motion.ref_state_pack(frames)
        error_height = (reference.root_p[..., 2] - position[..., 2]).abs()
        error_tilt = (reference.root_R[..., 2, 2] - rotation[..., 2, 2]).abs()
        return error_height, error_tilt

    def termination(
        self, state: State, frames: torch.Tensor, reference: ReferenceStatePack | None = None
    ) -> torch.Tensor:
        """Return hard termination from root height and tilt deviations."""
        error_height, error_tilt = self._termination_errors(state, reference, frames)
        return (error_height > self.cfg.term_height) | (error_tilt > self.cfg.term_tilt)

    def soft_termination(
        self,
        state: State,
        frames: torch.Tensor,
        height_temperature: float = 0.02,
        tilt_temperature: float = 0.04,
        reference: ReferenceStatePack | None = None,
    ) -> torch.Tensor:
        """Return the differentiable soft termination probability."""
        error_height, error_tilt = self._termination_errors(state, reference, frames)
        alive = (1 - torch.sigmoid((error_height - self.cfg.term_height) / height_temperature)) * (
            1 - torch.sigmoid((error_tilt - self.cfg.term_tilt) / tilt_temperature)
        )
        return 1 - alive

    def critic_features(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
        reference: ReferenceStatePack | None = None,
        reference_body_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return invariant state, goal errors, and normalized simulation context."""
        motion = self.motion
        position, rotation, linear_velocity, angular_velocity, joint_pos, joint_vel = state
        heading_t = yaw_mat(rotation).transpose(-1, -2)
        reference = reference if reference is not None else motion.ref_state_pack(frames)
        ref_p, ref_R, ref_q = reference.root_p, reference.root_R, reference.q
        if reference_body_p is None:
            reference_body_p = motion.body_state(frames)[0]
        body_p = self.kin.forward(
            position,
            rotation,
            linear_velocity,
            angular_velocity,
            joint_pos,
            joint_vel,
        )[0]
        delta_rotation = yaw_mat(matmul(rotation, ref_R.transpose(-1, -2)))
        anchor = torch.cat((position[..., :2], ref_p[..., 2:3]), -1)
        ref_body_p = anchor.unsqueeze(-2) + matvec(
            delta_rotation.unsqueeze(-3), reference_body_p - ref_p.unsqueeze(-2)
        )
        body_error = matvec(heading_t.unsqueeze(-3), ref_body_p - body_p).flatten(-2)
        root_error = matvec(heading_t, ref_p - position)
        relative_rotation = mat_to_6d(matmul(rotation.transpose(-1, -2), ref_R))
        return torch.cat(
            (
                invariant_features(state),
                ref_q - joint_pos,
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

    def ref_state_pack(self, frames: torch.Tensor) -> ReferenceStatePack:
        """Return all reference state fields from one contiguous-cache gather."""
        return self.motion.ref_state_pack(frames)

    def timeline_frames(self, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        """Return a contiguous unified rollout frame axis owned by the task."""
        return self.motion.timeline_frames(frames, horizon)

    def reference_timeline(
        self, frames: torch.Tensor, horizon: int, goal_frames: int
    ) -> ReferenceTimeline:
        """Prefetch reference states, body states, and future goals once."""
        if horizon < 0:
            raise ValueError("horizon cannot be negative")
        if goal_frames <= 0:
            raise ValueError("goal_frames must be positive")
        return self.motion.reference_timeline(frames, horizon, goal_frames)

    @staticmethod
    def timeline_reference(
        reference: ReferenceStatePack, time_index: int, batch_size: int
    ) -> ReferenceStatePack:
        """Slice a prefetched flat reference pack to one rollout time step."""
        start = time_index * batch_size
        stop = start + batch_size
        return ReferenceStatePack(
            reference.root_p[start:stop],
            reference.root_R[start:stop],
            reference.root_v[start:stop],
            reference.root_w[start:stop],
            reference.q[start:stop],
            reference.qd[start:stop],
        )

    @staticmethod
    def timeline_body_state(
        body_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        time_index: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice prefetched body states to one rollout time step."""
        start = time_index * batch_size
        stop = start + batch_size
        return tuple(value[start:stop] for value in body_state)  # type: ignore[return-value]

    def clip_info(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return owning clip, its inclusive bounds, and the frame's one-second bin."""
        clips = self.motion.clip_id[frames]
        return {
            "clip_id": clips,
            "clip_start": self.motion.clip_start[clips],
            "clip_end": self.motion.clip_end[clips],
            "frame_bin": self.motion.frame_bin[frames],
        }
