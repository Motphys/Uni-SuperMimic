"""CPU training-loop tests for the protocol-driven SuperMimic learner."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    State,
)


class FakeMdp:
    num_joints = 2
    context_dim = 2
    obs_dim = 6
    goal_frame_dim = 3
    max_step_reward = 0.1

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (state[0][..., :1], previous_action, context, frames.float().unsqueeze(-1)), -1
        )

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        return frames.float().unsqueeze(-1).expand(-1, num_frames * self.goal_frame_dim)

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        del frames
        tracking = self.max_step_reward * torch.exp(-(state[0] - 0.1).square().mean(-1))
        return tracking - 0.001 * (action - previous_action).square().mean(-1)

    def soft_termination(self, state: State, frames: torch.Tensor) -> torch.Tensor:
        del state
        return torch.zeros_like(frames, dtype=torch.float32)

    def critic_features(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((state[0], state[4], normalized_context, frames.float().unsqueeze(-1)), -1)

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(context)

    def valid_imagination_start(self, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        del horizon
        return torch.ones_like(frames, dtype=torch.bool)


@dataclass
class FakeState:
    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict
    final_observation: None = None


class FakeEnv:
    def __init__(self, num_envs: int = 2):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.mdp = FakeMdp()
        self.state: FakeState | None = None
        self.initial_state = torch.zeros(num_envs, 13 + 2 * self.mdp.num_joints)
        self.initial_state[:, 3] = 1.0
        self.initial_frames = torch.arange(num_envs, dtype=torch.long)
        self.context = torch.full((num_envs, self.mdp.context_dim), 0.2)

    @property
    def supermimic_mdp(self) -> FakeMdp:
        return self.mdp

    def init_state(self) -> FakeState:
        info = {
            "s": self.initial_state.clone(),
            "f": self.initial_frames.clone(),
            "xi": self.context.clone(),
            "a_prev": torch.zeros(self.num_envs, self.mdp.num_joints),
            "ep_step": torch.zeros(self.num_envs, dtype=torch.int32),
        }
        self.state = FakeState(
            obs={},
            reward=torch.zeros(self.num_envs),
            terminated=torch.zeros(self.num_envs, dtype=torch.bool),
            truncated=torch.zeros(self.num_envs, dtype=torch.bool),
            info={SUPERMIMIC_STATE_KEY: info},
        )
        return self.state

    def step(self, actions: torch.Tensor) -> FakeState:
        assert self.state is not None
        current = self.state.info[SUPERMIMIC_STATE_KEY]
        next_state = current["s"].clone()
        next_state[:, :3] += 0.001
        next_state[:, 13 : 13 + self.mdp.num_joints] += 0.01 * actions
        next_state[:, 13 + self.mdp.num_joints :] += 0.001 * actions
        next_frames = current["f"] + 1
        episode_steps = current["ep_step"] + 1
        done = episode_steps >= 3
        reward = self.mdp.reward(
            (
                next_state[:, 0:3],
                torch.eye(3).expand(self.num_envs, 3, 3),
                next_state[:, 7:10],
                next_state[:, 10:13],
                next_state[:, 13:15],
                next_state[:, 15:17],
            ),
            next_frames,
            actions,
            current["a_prev"],
        )
        transition = {
            "s": current["s"].clone(),
            "s1": next_state.clone(),
            "a": actions.clone(),
            "a_prev": current["a_prev"].clone(),
            "xi": current["xi"].clone(),
            "r": reward,
            "f": current["f"].clone(),
            "ep_step": current["ep_step"].clone(),
            "done": done,
            "term": torch.zeros_like(done),
        }
        if bool(done.any()):
            transition["ep_len"] = episode_steps[done].float()
            transition["ep_ret_norm"] = reward[done] / (
                episode_steps[done].float() * self.mdp.max_step_reward
            )
        post_reset_state = next_state.clone()
        post_reset_state[done] = self.initial_state[done]
        post_reset_action = torch.where(done.unsqueeze(-1), torch.zeros_like(actions), actions)
        post_reset_steps = torch.where(done, torch.zeros_like(episode_steps), episode_steps)
        info = {
            "s": post_reset_state,
            "f": next_frames,
            "xi": current["xi"].clone(),
            "a_prev": post_reset_action,
            "ep_step": post_reset_steps,
        }
        self.state = FakeState(
            obs={},
            reward=reward,
            terminated=done,
            truncated=torch.zeros_like(done),
            info={
                SUPERMIMIC_STATE_KEY: info,
                SUPERMIMIC_TRANSITION_KEY: transition,
            },
        )
        return self.state


def _config() -> SuperMimicConfig:
    return SuperMimicConfig(
        iterations=1,
        buffer_steps=16,
        utd=1.0,
        horizon=2,
        goal_frames=2,
        hist=2,
        wm_window=2,
        wm_batches=1,
        wm_batch=4,
        pi_batch=4,
        lr=1e-3,
        critic_epochs=1,
        critic_minibatches=2,
        warmup_steps=3,
        wm_only_iters=0,
        ckpt_every=1,
        log_every=1,
        log_backend="no_print",
        amp="none",
        compile=False,
        test_hidden_dims=(8,),
    )


def test_tf32_only_enabled_for_fp32_training(tmp_path) -> None:
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        SuperMimicLearner(FakeEnv(), _config(), tmp_path / "fp32")
        assert torch.backends.cuda.matmul.allow_tf32 is True
        SuperMimicLearner(FakeEnv(), replace(_config(), amp="bf16"), tmp_path / "bf16")
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_collect_and_updates_run_on_cpu(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)

    learner.collect(3)
    wm_loss, wm_error = learner.wm_update()
    policy_loss, value_loss, imagined_reward, grad_norm, survival = learner.policy_update()

    assert learner.buffer.size == 3
    assert learner.sim_steps == 6
    assert learner.obs_hist.shape == (2, 2, 6)
    for metric in (
        wm_loss,
        wm_error,
        policy_loss,
        value_loss,
        imagined_reward,
        grad_norm,
        survival,
    ):
        assert metric.ndim == 0
        assert torch.isfinite(metric)


def test_checkpoint_roundtrip_and_resume(tmp_path) -> None:
    cfg = _config()
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    final = learner.train()

    assert final["completed"] is True
    assert final["iteration"] == 1
    checkpoint_path = tmp_path / "last.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert set(checkpoint) == {
        "wm",
        "pi",
        "critic",
        "opt_wm",
        "opt_pi",
        "opt_v",
        "iteration",
        "wall_time",
        "sim_steps",
        "args",
        "saved_at",
    }
    small = torch.load(tmp_path / "ckpt_000001.pt", weights_only=True)
    assert set(small) == {"wm", "pi", "iteration", "wall_time", "sim_steps", "args", "saved_at"}
    with (tmp_path / "train_log.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    assert rows[0][:3] == ["iteration", "wall_h", "sim_steps"]
    assert rows[1][0] == "1"

    resumed = SuperMimicLearner(FakeEnv(), replace(cfg, iterations=2), tmp_path / "resumed")
    with pytest.raises(ValueError, match="requires a full last.pt"):
        resumed.load(tmp_path / "ckpt_000001.pt")
    resumed.load(checkpoint_path)
    assert resumed.iteration == learner.iteration
    assert resumed.sim_steps == learner.sim_steps
    assert resumed.wall_time == checkpoint["wall_time"]
    for name in ("wm", "pi", "critic"):
        original_state = getattr(learner, name).state_dict()
        restored_state = getattr(resumed, name).state_dict()
        assert original_state.keys() == restored_state.keys()
        for key in original_state:
            torch.testing.assert_close(restored_state[key], original_state[key])

    resumed_final = resumed.train()
    assert resumed_final["completed"] is True
    assert resumed_final["iteration"] == 2
    assert resumed_final["sim_steps"] > final["sim_steps"]


def test_resume_rebuilds_warmup_history(tmp_path) -> None:
    cfg = _config()
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    learner.train()
    resumed = SuperMimicLearner(
        FakeEnv(),
        replace(cfg, iterations=1, warmup_steps=6),
        tmp_path / "resumed",
    )

    resumed.train(resume=tmp_path / "last.pt")

    assert resumed.buffer.size == 6


def test_tensorboard_backend_logs_real_metric_names(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), replace(_config(), log_backend="tensorboard"), tmp_path)
    learner.train()

    event_files = list(tmp_path.glob("events.out.tfevents.*"))
    assert len(event_files) == 1
    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    expected = {
        "Loss/world_model",
        "Loss/world_model_err8",
        "Loss/policy",
        "Loss/value",
        "Loss/imagined_reward",
        "Loss/imagined_survival",
        "Train/grad_norm",
        "Train/learning_rate",
        "Reward/sim_step_norm",
        "Episode/length",
        "Episode/return_norm",
        "Episode/terminated_frac",
        "Perf/iteration_time",
        "Perf/collect_time",
        "Perf/learning_time",
        "Perf/it_per_s",
        "Perf/sim_steps_per_s",
    }
    assert expected <= tags


def test_no_print_backend_writes_no_events(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.train()

    assert not list(tmp_path.glob("events.out.tfevents.*"))
