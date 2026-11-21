"""Async collection (overlap of env stepping with WM updates) tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_supermimic_learner import FakeEnv, FakeState, _config

from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer


def _transition(num_envs: int = 2) -> dict[str, torch.Tensor]:
    return {
        "s": torch.randn(num_envs, 71),
        "s1": torch.randn(num_envs, 71),
        "a": torch.randn(num_envs, 29),
        "a_prev": torch.randn(num_envs, 29),
        "xi": torch.randn(num_envs, 33),
        "r": torch.randn(num_envs),
        "f": torch.randint(0, 100, (num_envs,)),
        "ep_step": torch.randint(0, 50, (num_envs,), dtype=torch.int32),
        "done": torch.zeros(num_envs, dtype=torch.bool),
    }


def test_staged_rows_invisible_until_commit() -> None:
    buffer = ReplayBuffer(T=8, N=2, device="cpu")
    buffer.add_staged(_transition())
    assert buffer.size == 0
    with pytest.raises(ValueError, match="empty replay buffer"):
        buffer.sample_starts(1, 1)
    buffer.commit(1)
    assert buffer.size == 1
    start, _ = buffer.sample_starts(4, 1)
    assert start["s"].shape == (4, 71)


def test_async_collect_completes_full_training_loop(tmp_path) -> None:
    cfg = replace(_config(), iterations=5, utd=0.5, async_collect=True)
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)

    result = learner.train()

    assert result["completed"] is True
    assert result["iteration"] == 5
    # utd=0.5 -> exactly 2 env steps per iteration on top of the 2-step warmup.
    assert learner.sim_steps == (2 + 2 * 5) * learner.env.num_envs
    assert torch.isfinite(torch.tensor(result["wm_loss"]))
    assert torch.isfinite(torch.tensor(result["pi_loss"]))


def test_async_collect_matches_sync_sim_step_schedule(tmp_path) -> None:
    sync_learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path / "sync")
    async_cfg = replace(_config(), async_collect=True)
    async_learner = SuperMimicLearner(FakeEnv(), async_cfg, tmp_path / "async")

    sync_learner.train()
    async_learner.train()

    assert async_learner.sim_steps == sync_learner.sim_steps
    assert async_learner.buffer.size == sync_learner.buffer.size
    assert async_learner.buffer.ptr == sync_learner.buffer.ptr


class _FailingEnv(FakeEnv):
    def __init__(self, fail_after: int = 4):
        super().__init__()
        self._steps = 0
        self._fail_after = fail_after

    def step(self, actions: torch.Tensor) -> FakeState:
        self._steps += 1
        if self._steps > self._fail_after:
            raise RuntimeError("boom")
        return super().step(actions)


def test_async_collect_worker_error_propagates(tmp_path) -> None:
    cfg = replace(_config(), iterations=5, async_collect=True)
    learner = SuperMimicLearner(_FailingEnv(), cfg, tmp_path)

    with pytest.raises(RuntimeError, match="collect worker failed") as excinfo:
        learner.train()

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "boom" in str(excinfo.value.__cause__)
