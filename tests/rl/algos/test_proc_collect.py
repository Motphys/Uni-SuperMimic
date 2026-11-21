"""Process-isolated collector tests (unilab IPC ring + SharedWeightSync)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_supermimic_learner import FakeEnv, _config

from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.rl.algos.supermimic.proc_collector import (
    TRANSITION_FIELDS,
    ProcCollectDriver,
    TransitionRing,
    weight_state_subset,
)


def fake_env_factory() -> FakeEnv:
    return FakeEnv()


def _fake_transition(num_envs: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "s": torch.randn(num_envs, 71, generator=g),
        "s1": torch.randn(num_envs, 71, generator=g),
        "a": torch.randn(num_envs, 29, generator=g),
        "a_prev": torch.randn(num_envs, 29, generator=g),
        "xi": torch.randn(num_envs, 33, generator=g),
        "r": torch.randn(num_envs, generator=g),
        "f": torch.randint(2**24, 47_300_000, (num_envs,), generator=g),
        "ep_step": torch.randint(0, 50, (num_envs,), dtype=torch.int32, generator=g),
        "done": torch.rand(num_envs, generator=g) > 0.5,
        "term": torch.zeros(num_envs, dtype=torch.bool),
        "ep_len": torch.rand(num_envs, generator=g),
        "ep_ret_norm": torch.rand(num_envs, generator=g),
    }


def test_transition_ring_roundtrip() -> None:
    ring = TransitionRing(num_envs=4, max_steps=3, obs_dim=6, goal_dim=6, create=True)
    try:
        views = ring.write_buffer_views()
        for step in range(2):
            tr = _fake_transition(4, step)
            for name in TRANSITION_FIELDS:
                views[f"tr:{name}"][step].copy_(tr[name])
            views["meta:has_ep"][step] = 1
        views["meta:n_steps"].fill_(2)
        ring.signal_write_done()

        assert ring.available() == 1
        read = ring.read_tensor_views()
        for step in range(2):
            tr = _fake_transition(4, step)
            for name in TRANSITION_FIELDS:
                got = read[f"tr:{name}"][step]
                assert got.dtype == tr[name].dtype, name
                assert torch.equal(got, tr[name]), name
        # int64 frame ids above the fp32-exact range survive intact
        assert int(read["tr:f"].max()) >= 2**24
        assert int(read["meta:n_steps"]) == 2
        ring.advance_read()
        assert ring.available() == 0
    finally:
        ring.cleanup()


def test_weight_subset_excludes_norm_buffers(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    subset = weight_state_subset(learner.pi)
    assert subset
    assert all(name.startswith(("enc.", "dec.")) for name in subset)
    assert not any("norm" in name for name in subset)


def test_threaded_async_requires_host_bridge(tmp_path) -> None:
    cfg = replace(_config(), async_collect=True)
    with pytest.raises(ValueError, match="HOST_BRIDGE"):
        SuperMimicLearner(FakeEnv(), cfg, tmp_path, backend="mjwarp")


def test_proc_collect_completes_training_loop(tmp_path) -> None:
    cfg = replace(_config(), iterations=5, utd=0.5, proc_collect=True)
    spec = {
        "env_factory": "test_proc_collect.fake_env_factory",
        "num_envs": 2,
        "sim_backend": "mujoco",
        "device": "cpu",
        "seed": 0,
    }
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path, collector_env_spec=spec)

    result = learner.train()

    assert result["completed"] is True
    assert result["iteration"] == 5
    assert learner.sim_steps == (2 + 2 * 5) * 2
    assert learner.buffer.size == 2 + 2 * 5
    assert torch.isfinite(torch.tensor(result["wm_loss"]))
    assert torch.isfinite(torch.tensor(result["pi_loss"]))
    # collector-side norm snapshots landed in the learner's policy
    assert learner.pi.obs_norm.count > 0
    assert learner.pi.goal_norm.count > 0


def test_three_way_requires_proc_collect(tmp_path) -> None:
    cfg = replace(_config(), three_way=True)
    with pytest.raises(ValueError, match="requires proc_collect"):
        SuperMimicLearner(FakeEnv(), cfg, tmp_path)


def test_three_way_completes_training_loop(tmp_path) -> None:
    cfg = replace(_config(), iterations=5, utd=0.5, proc_collect=True, three_way=True)
    spec = {
        "env_factory": "test_proc_collect.fake_env_factory",
        "num_envs": 2,
        "sim_backend": "mujoco",
        "device": "cpu",
        "seed": 0,
    }
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path, collector_env_spec=spec)

    result = learner.train()

    assert result["completed"] is True
    assert result["iteration"] == 5
    assert learner.sim_steps == (2 + 2 * 5) * 2
    assert learner.wm_shadow is not None
    assert torch.isfinite(torch.tensor(result["wm_loss"]))
    assert torch.isfinite(torch.tensor(result["pi_loss"]))
