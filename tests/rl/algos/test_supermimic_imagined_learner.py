"""Tests for the pure frozen-world-model imagination learner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.rl.algos.supermimic_imagined.learner import (
    ImaginedSuperMimicConfig,
    ImaginedSuperMimicLearner,
)


class FakeMotion:
    def __init__(self) -> None:
        self.num_frames = 101
        self.clip_start = torch.tensor([0])
        self.clip_end = torch.tensor([100])
        self.clip_id = torch.zeros(101, dtype=torch.long)

    def ref_state(self, frames: torch.Tensor):
        batch = frames.shape[0]
        return (
            torch.zeros(batch, 3),
            torch.eye(3, device=frames.device).expand(batch, 3, 3),
            torch.zeros(batch, 3),
            torch.zeros(batch, 3),
            torch.zeros(batch, 29),
            torch.zeros(batch, 29),
        )

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        return frames.float()[:, None].expand(-1, num_frames * 3)

    def body_state(self, frames: torch.Tensor):
        batch = frames.shape[0]
        zeros = torch.zeros(batch, 2, 3)
        return zeros, torch.eye(3).expand(batch, 2, 3, 3), zeros, zeros


class FakeMdp:
    num_joints = 29
    context_dim = 33
    obs_dim = 39
    goal_frame_dim = 3
    max_step_reward = 0.1

    def __init__(self) -> None:
        self.default_q = torch.zeros(29)

    def obs(self, state, previous_action, context, frames):
        return torch.cat(
            (state[0], state[4][..., :1], previous_action[..., :2], torch.zeros_like(context)), -1
        )

    def add_noise(self, obs):
        return obs

    def goal(self, frames, num_frames):
        return frames.float()[:, None].expand(-1, num_frames * self.goal_frame_dim)

    def reward(self, state, frames, action, previous_action):
        del frames
        tracking = self.max_step_reward * torch.exp(-(state[0] - 0.1).square().mean(-1))
        return tracking - 0.001 * (action - previous_action).square().mean(-1)

    def soft_termination(self, state, frames):
        del state
        return torch.zeros_like(frames, dtype=torch.float32)

    def critic_features(self, state, frames, normalized_context):
        return torch.cat((state[0], state[4][..., :1], normalized_context), -1)

    def obs_with_reference(self, state, previous_action, context, frames, reference_rotation):
        del frames
        return torch.cat(
            (
                state[0],
                state[4][..., :1],
                previous_action[..., :2],
                torch.zeros_like(context),
            ),
            -1,
        )

    def reward_term_critic_features_with_reference(
        self,
        state,
        frames,
        normalized_context,
        reference_state,
        reference_body_state,
        action,
        previous_action,
    ):
        scored_state = tuple(item[1:].reshape(-1, *item.shape[2:]) for item in state)
        reward = self.reward(scored_state, frames, action, previous_action)
        dones = self.soft_termination(scored_state, frames)
        features = self.critic_features(
            state, torch.zeros(frames.shape[0] + 1, dtype=torch.long), normalized_context
        )
        return reward, dones, features

    def normalize_context(self, context):
        return torch.tanh(context)

    def valid_imagination_start(self, frames, horizon):
        return torch.ones_like(frames, dtype=torch.bool)

    def to(self, device):
        return self


@dataclass
class FakeTask:
    mdp: FakeMdp
    motion: FakeMotion
    world_model: WorldModel
    wm_hidden: tuple[int, ...]
    wm_structured: bool


def _task() -> FakeTask:
    return FakeTask(FakeMdp(), FakeMotion(), WorldModel((8,), 29, 33), (8,), False)


def _config(**overrides) -> ImaginedSuperMimicConfig:
    values = {
        "batch": 3,
        "horizon": 2,
        "goal_frames": 2,
        "hist": 2,
        "policy_hidden": (16, 16),
        "critic_hidden": (8,),
        "pi_ema": 0.0,
        "max_episode_steps": 100,
        "critic_epochs": 1,
        "critic_minibatches": 2,
        "compile": False,
        "test_hidden_dims": (8,),
        "log_backend": "none",
    }
    values.update(overrides)
    return ImaginedSuperMimicConfig(**values)


def test_update_backprops_through_frozen_wm_without_wm_gradients(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(_task(), _config(), tmp_path)
    before = {
        name: value.clone()
        for name, value in learner.wm.state_dict().items()
        if value.dtype.is_floating_point
    }
    policy_before = {
        name: value.clone()
        for name, value in learner.pi.state_dict().items()
        if value.dtype.is_floating_point
    }
    result = learner.update()

    assert all(torch.isfinite(item) for item in result[:5])
    assert all(not parameter.requires_grad for parameter in learner.wm.parameters())
    assert all(parameter.grad is None for parameter in learner.wm.parameters())
    for name, old in before.items():
        torch.testing.assert_close(learner.wm.state_dict()[name], old)
    assert any(
        not torch.equal(value, policy_before[name])
        for name, old_value in policy_before.items()
        for value in [learner.pi.state_dict()[name]]
    )


def test_update_reuses_persistent_imagined_state(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(_task(), _config(), tmp_path)
    initial_frames = learner._frames.clone()
    learner.update()

    assert learner._frames is not None and torch.equal(learner._frames, initial_frames + 2)
    assert learner._ep_step is not None and torch.equal(
        learner._ep_step, torch.full_like(initial_frames, 2)
    )


def test_reset_samples_motion_starts_and_fills_history(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(_task(), _config(), tmp_path)
    learner.reset_all()

    assert learner._state is not None
    assert learner._context is not None
    assert learner._context.shape == (3, 33)
    assert learner._obs_hist is not None and learner._obs_hist.shape == (3, 2, 39)
    assert (
        learner._previous_action is not None and torch.count_nonzero(learner._previous_action) == 0
    )


def test_checkpoint_round_trip(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(_task(), _config(), tmp_path)
    learner.update()
    path = learner.save(tmp_path / "ckpt.pt")

    restored = ImaginedSuperMimicLearner(_task(), _config(), tmp_path / "restored")
    restored.load(path)

    assert restored.iteration == learner.iteration
    assert restored.imag_steps == learner.imag_steps
    for name, value in learner.pi.state_dict().items():
        torch.testing.assert_close(restored.pi.state_dict()[name], value)


def test_learner_has_no_real_environment_or_replay() -> None:
    learner = ImaginedSuperMimicLearner(_task(), _config(), Path(tempfile_mkdtemp()))
    assert not hasattr(learner, "env")
    assert not hasattr(learner, "buffer")
    assert not hasattr(learner, "collect")
    assert not hasattr(learner, "wm_update")
    assert not hasattr(learner, "opt_wm")


def tempfile_mkdtemp() -> str:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        return directory
