"""Small deterministic fake task for imagined-learner benchmarks.

The real ``G1ImaginedTask`` needs a robot model, motion library, and frozen
checkpoint.  These stand-ins preserve the relevant state, observation, goal, and
MLP dimensions while keeping smoke tests and CI benchmarks asset-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.rl.contracts.supermimic import State


@dataclass
class _FakeMotion:
    num_frames: int
    num_joints: int
    clip_start: tuple[int, ...] = (0,)
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self.clip_end = torch.tensor([self.num_frames - 1], device=self.device)
        self.clip_id = torch.zeros(self.num_frames, dtype=torch.long, device=self.device)

    def sample_start(self, horizon: int) -> torch.Tensor:
        # The fake task has one clip. Keep sampled frames away from its end so
        # reset construction has valid reference states without host syncs.
        maximum = max(self.num_frames - horizon - 2, 0)
        return torch.randint(0, maximum + 1, (1,), device=self.device)[0].expand(1)

    def ref_state(self, frames: torch.Tensor) -> State:
        device = frames.device
        batch = frames.shape[0]
        phase = frames.float()[:, None] / max(self.num_frames - 1, 1)
        position = torch.cat(
            (
                phase,
                0.5 * phase,
                torch.full((batch, 1), 0.8, device=device, dtype=phase.dtype),
            ),
            -1,
        )
        rotation = torch.eye(3, device=device).expand(batch, 3, 3).clone()
        linear = torch.zeros(batch, 3, device=device, dtype=phase.dtype)
        linear[:, 0] = 1.0
        angular = torch.zeros(batch, 3, device=device, dtype=phase.dtype)
        joints = 0.1 * torch.sin(
            phase + torch.linspace(0.0, 0.5, self.num_joints, device=phase.device)
        )
        return position, rotation, linear, angular, joints, torch.zeros_like(joints)

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        return _FakeMdp._goal_batch(frames, num_frames)


class _FakeMdp:
    def __init__(self, batch: int, num_joints: int, context_dim: int, device: torch.device):
        self.num_joints = num_joints
        self.context_dim = context_dim
        self.obs_dim = 3 + 3 + 3 * num_joints + context_dim
        self.goal_frame_dim = 3
        self.max_step_reward = 1.0
        self.default_q = torch.zeros((), device=device)
        self._frames = torch.arange(batch, device=device).remainder(16)

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        del frames
        return torch.cat(
            (
                state[0],
                state[1][:, :, 0],
                state[4],
                state[5],
                previous_action,
                context,
            ),
            -1,
        )

    def add_noise(self, observation: torch.Tensor) -> torch.Tensor:
        return observation

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        return self._goal_batch(frames, num_frames)

    @staticmethod
    def _goal_batch(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        phase = (frames.float()[:, None] + torch.arange(num_frames, device=frames.device)) / 64.0
        return phase.repeat_interleave(3, dim=1).contiguous()

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        tracking = -state[0].square().mean(-1)
        smoothness = -(action - previous_action).square().mean(-1)
        return tracking + 0.01 * smoothness

    def soft_termination(self, state: State, frames: torch.Tensor) -> torch.Tensor:
        # Keep all trajectories alive for deterministic benchmark coverage. The
        # learner's termination/reset control flow still runs on every update.
        return torch.zeros_like(state[0][:, 2], dtype=torch.float32)

    def critic_features(
        self, state: State, frames: torch.Tensor, normalized_context: torch.Tensor
    ) -> torch.Tensor:
        del frames
        return torch.cat((state[0], state[1][:, :, 0], normalized_context), -1)

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(context * 0.5)

    def valid_imagination_start(self, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        clip_end = self.motion.clip_end[self.motion.clip_id[frames]]
        return frames + horizon <= clip_end


@dataclass
class _FakeTask:
    mdp: _FakeMdp
    motion: _FakeMotion
    world_model: WorldModel
    wm_hidden: tuple[int, ...]
    wm_structured: bool = False


def make_fake_imagined_task(
    batch: int = 8,
    num_joints: int = 4,
    context_dim: int = 33,
    wm_hidden: Sequence[int] = (16,),
    num_frames: int = 128,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> _FakeTask:
    """Construct an asset-free task with dimensions chosen by the benchmark."""
    if batch <= 0:
        raise ValueError("batch must be positive")
    if num_joints <= 0 or context_dim < 33 or num_frames < 2:
        raise ValueError("context_dim must be >= 33, joint dimension positive, num_frames >= 2")
    resolved = torch.device(device)
    generator = torch.Generator().manual_seed(seed)

    motion = _FakeMotion(num_frames=num_frames, num_joints=num_joints, device=resolved)
    mdp = _FakeMdp(batch, num_joints, context_dim, resolved)
    mdp.motion = motion
    world_model = WorldModel(tuple(wm_hidden), num_joints, context_dim).to(resolved)
    # Dense deterministic weights avoid the zero-initialized output making every
    # benchmark kernel numerically trivial; the model remains frozen by learner.
    with torch.no_grad():
        for parameter in world_model.parameters():
            noise = torch.rand(parameter.shape, generator=generator)
            parameter.mul_(0.02).add_(noise.to(device=parameter.device) * 0.01)
        for name, value in world_model.state_dict().items():
            if name.endswith(".var"):
                value.fill_(1.0)
    world_model.requires_grad_(False)
    world_model.eval()
    return _FakeTask(mdp, motion, world_model, tuple(wm_hidden))


__all__ = ["make_fake_imagined_task"]
