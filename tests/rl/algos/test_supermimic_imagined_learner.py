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
        self.clip_start = torch.tensor([0, 80])
        self.clip_end = torch.tensor([79, 100])
        self.clip_id = torch.cat(
            (torch.zeros(80, dtype=torch.long), torch.ones(21, dtype=torch.long))
        )

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
        return frames + horizon <= self._clip_end[self._clip_id[frames]]

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
    motion = FakeMotion()
    mdp = FakeMdp()
    mdp._clip_end = motion.clip_end
    mdp._clip_id = motion.clip_id
    return FakeTask(mdp, motion, WorldModel((8,), 29, 33), (8,), False)


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
    clip_end = learner.motion.clip_end[learner.motion.clip_id[initial_frames]]
    surviving = (initial_frames + learner.cfg.horizon < clip_end).nonzero().squeeze(-1)
    learner.update()

    assert learner._frames is not None
    assert torch.equal(
        learner._frames[: surviving.numel()], initial_frames[surviving] + learner.cfg.horizon
    )
    assert learner._ep_step is not None
    assert torch.equal(learner._ep_step[: surviving.numel()], torch.full_like(surviving, 2))
    assert learner._frames.numel() == learner.cfg.batch


def test_reset_samples_valid_clip_starts_and_preserves_unselected_rows(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(_task(), _config(), tmp_path)
    learner._reset_batch(None)
    original_frames = learner._frames.clone()
    original_history = learner._obs_hist.clone()
    selected = torch.tensor([1], dtype=torch.long)
    learner._reset_batch(selected)

    assert learner._frames is not None
    assert learner.mdp.valid_imagination_start(learner._frames[selected], learner.cfg.horizon).all()
    assert torch.equal(learner._frames[0], original_frames[0])
    assert torch.equal(learner._frames[2], original_frames[2])
    assert learner._obs_hist is not None
    assert torch.equal(learner._obs_hist[0], original_history[0])
    assert torch.equal(learner._obs_hist[2], original_history[2])


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


def _clone_state(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if value.dtype.is_floating_point
    }


def _assert_close_state(actual, expected, *, message):
    assert actual.keys() == expected.keys()
    for name, old in expected.items():
        torch.testing.assert_close(
            actual[name], old, msg=lambda error: f"{message} {name}: {error}"
        )


def test_critic_forward_folds_running_norm_equivalently(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(_task(), _config(), tmp_path)
    critic_input = torch.randn(11, 3, learner.critic.norm.mean.numel(), device=learner.device)
    expected = learner.critic(critic_input)
    actual = learner._critic_forward(critic_input)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    folded_input = critic_input.clone().requires_grad_(True)
    reference_input = critic_input.clone().requires_grad_(True)
    learner._critic_forward(folded_input).sum().backward()
    learner.critic(reference_input).sum().backward()
    torch.testing.assert_close(folded_input.grad, reference_input.grad, rtol=2e-5, atol=2e-5)


def test_critic_norm_subsample_larger_than_data_is_exact(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(
        _task(), _config(critic_norm_update="subsample", critic_norm_subsample=1000), tmp_path
    )
    reference = ImaginedSuperMimicLearner(_task(), _config(), tmp_path / "reference")
    reference.critic.load_state_dict(learner.critic.state_dict())
    critic_input = torch.randn(3, 3, learner.critic.norm.mean.numel(), device=learner.device)

    reference.critic.norm.update(critic_input)
    learner._update_critic_norm(critic_input)

    torch.testing.assert_close(learner.critic.norm.mean, reference.critic.norm.mean)
    torch.testing.assert_close(learner.critic.norm.var, reference.critic.norm.var)
    torch.testing.assert_close(learner.critic.norm.count, reference.critic.norm.count)


def test_critic_norm_subsample_is_on_device_budget(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(
        _task(), _config(critic_norm_update="subsample", critic_norm_subsample=5), tmp_path
    )
    learner.iteration = 1
    critic_input = torch.randn(3, 3, learner.critic.norm.mean.numel(), device=learner.device)

    update_input = []
    original_update = learner.critic.norm.update
    learner.critic.norm.update = lambda x: update_input.append(x)
    learner._update_critic_norm(critic_input)
    learner.critic.norm.update = original_update

    assert len(update_input) == 1
    assert update_input[0].shape == (5, learner.critic.norm.mean.numel())
    assert update_input[0].device == learner.device


def test_critic_norm_interval_skips_updates_but_does_not_freeze_early(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(
        _task(), _config(critic_norm_update="interval", critic_norm_interval=3), tmp_path
    )
    critic_input = torch.randn(3, 3, learner.critic.norm.mean.numel(), device=learner.device)

    learner.iteration = 0
    learner._update_critic_norm(critic_input)
    updated = learner.critic.norm.count.clone()
    learner.iteration = 1
    learner._update_critic_norm(critic_input)
    learner.iteration = 2
    learner._update_critic_norm(critic_input)
    learner.iteration = 3
    learner._update_critic_norm(critic_input)

    assert updated.item() == critic_input.numel() // critic_input.shape[-1]
    assert learner.critic.norm.count.item() == 2 * updated.item()
    assert learner.critic.norm.max_count == 2e8


def test_critic_norm_off_and_max_count_are_configurable(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(
        _task(),
        _config(critic_norm_update="off", critic_norm_max_count=123.0),
        tmp_path,
    )
    critic_input = torch.randn(3, 3, learner.critic.norm.mean.numel(), device=learner.device)

    learner._update_critic_norm(critic_input)

    assert learner.critic.norm.count.item() == 0.0
    assert learner.critic.norm.max_count == 123.0


def test_optimizer_modes_auto_backoff_and_cuda_guard(tmp_path) -> None:
    Learner = ImaginedSuperMimicLearner

    assert Learner._resolve_optimizer_mode("auto", torch.device("cpu")) == "single"
    assert Learner._resolve_optimizer_mode("foreach", torch.device("cpu")) == "foreach"
    with pytest.raises(ValueError, match="fused optimizers require CUDA"):
        Learner._resolve_optimizer_mode("fused", torch.device("cpu"))

    learner = ImaginedSuperMimicLearner(
        _task(),
        _config(pi_opt="adam", pi_amsgrad=True, critic_amsgrad=True, optimizer_mode="foreach"),
        tmp_path,
    )
    assert learner.opt_pi.defaults["amsgrad"] is True
    assert learner.opt_v.defaults["amsgrad"] is True
    assert learner.opt_pi.defaults["fused"] is False
    assert learner.opt_v.defaults["fused"] is False
    assert learner.opt_pi.defaults["foreach"] is True
    assert learner.opt_v.defaults["foreach"] is True

    default = ImaginedSuperMimicLearner(_task(), _config(pi_opt="adam"), tmp_path)
    assert default.opt_pi.defaults["fused"] is False
    assert default.opt_v.defaults["fused"] is False
    # Auto CPU fallback leaves foreach unspecified, exactly as the legacy constructor did.
    assert default.opt_pi.defaults["foreach"] is None
    assert default.opt_v.defaults["foreach"] is None


def test_critic_minibatch_update_is_equivalent_to_legacy_path(tmp_path) -> None:
    fast = ImaginedSuperMimicLearner(_task(), _config(), tmp_path / "fast")
    legacy = ImaginedSuperMimicLearner(_task(), _config(), tmp_path / "legacy")
    legacy.critic.load_state_dict(fast.critic.state_dict())
    fast.critic.norm.update(torch.randn(20, fast.critic.norm.mean.numel()))
    legacy.critic.norm.load_state_dict(fast.critic.norm.state_dict())
    flat_input = torch.randn(12, fast.critic.norm.mean.numel(), device=fast.device)
    flat_targets = torch.randn(12, device=fast.device)
    indices = torch.tensor([11, 0, 7, 4, 2, 9], device=fast.device)
    fast_input = flat_input.index_select(0, indices)
    fast_targets = flat_targets.index_select(0, indices)
    legacy_input = flat_input[indices]
    legacy_targets = flat_targets[indices]

    fast_loss = fast._critic_loss_fn(fast_input, fast_targets)
    legacy_loss = ((legacy.critic(legacy_input) - legacy_targets) ** 2).mean()
    torch.testing.assert_close(fast_loss, legacy_loss)

    fast.opt_v.zero_grad(set_to_none=True)
    fast_loss.backward()
    fast.opt_v.step()
    legacy.opt_v.zero_grad(set_to_none=True)
    legacy_loss.backward()
    legacy.opt_v.step()
    _assert_close_state(
        fast.critic.state_dict(), legacy.critic.state_dict(), message="minibatch optimizer"
    )


def test_invalid_acceleration_configuration_rejected() -> None:
    with pytest.raises(ValueError, match="critic_norm_update"):
        _config(critic_norm_update="sometimes")
    with pytest.raises(ValueError, match="critic_norm_subsample"):
        _config(critic_norm_subsample=0)
    with pytest.raises(ValueError, match="critic_norm_interval"):
        _config(critic_norm_interval=0)
    with pytest.raises(ValueError, match="critic_norm_max_count"):
        _config(critic_norm_max_count=-1.0)
    with pytest.raises(ValueError, match="optimizer_mode"):
        _config(optimizer_mode="apex")
    with pytest.raises(ValueError, match="pi_amsgrad"):
        _config(pi_amsgrad=True)


def test_critic_input_layout_matches_legacy_flatten_and_targets_match_gae(tmp_path) -> None:
    learner = ImaginedSuperMimicLearner(_task(), _config(), tmp_path)
    horizon, batch = learner.cfg.horizon, learner.cfg.batch
    state_shapes = ((3,), (3, 3), (3,), (3,), (29,), (29,))
    initial_state = tuple(torch.randn(batch, *shape) for shape in state_shapes)
    states = [tuple(torch.randn(batch, *shape) for shape in state_shapes) for _ in range(horizon)]
    frames0 = torch.arange(batch)
    all_frames = [torch.arange(batch) + index + 1 for index in range(horizon)]
    initial_goal = torch.randn(batch, learner.mdp.goal_frame_dim * learner.cfg.goal_frames)
    goals = [torch.randn_like(initial_goal) for _ in range(horizon)]
    normalized_context = torch.randn(batch, learner.mdp.context_dim)

    optimized_states = tuple(
        torch.stack((initial_state[index], *(item[index] for item in states))) for index in range(6)
    )
    optimized_input = learner._batched_critic_input(
        optimized_states,
        torch.stack((frames0, *all_frames)),
        normalized_context.unsqueeze(0).expand(horizon + 1, -1, -1),
        torch.stack((initial_goal, *goals)),
    )
    legacy_states = tuple(
        torch.cat(
            (
                initial_state[index].unsqueeze(0),
                torch.stack([item[index] for item in states]),
            ),
            0,
        ).flatten(0, 1)
        for index in range(6)
    )
    legacy_input = learner._batched_critic_input(
        legacy_states,
        torch.cat((frames0.unsqueeze(0), torch.stack(all_frames)), 0).flatten(),
        normalized_context.unsqueeze(0).expand(horizon + 1, -1, -1).flatten(0, 1),
        torch.cat((initial_goal.unsqueeze(0), torch.stack(goals)), 0).flatten(0, 1),
    )
    torch.testing.assert_close(optimized_input.flatten(0, 1), legacy_input, rtol=0, atol=0)

    values = torch.randn(horizon + 1, batch)
    rewards = torch.randn(horizon, batch)
    dones = torch.rand(horizon, batch)
    target = values[-1]
    expected = []
    for index in range(horizon - 1, -1, -1):
        target = rewards[index] + learner.cfg.gamma * (1 - dones[index]) * (
            (1 - learner.cfg.lam) * values[index + 1] + learner.cfg.lam * target
        )
        expected.append(target)
    expected = torch.stack(expected[::-1])
    torch.testing.assert_close(
        learner._critic_targets_fn(values, rewards, dones), expected, rtol=0, atol=0
    )
