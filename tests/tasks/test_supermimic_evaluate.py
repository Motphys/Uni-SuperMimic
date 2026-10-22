"""Fake-environment tests for deterministic SuperMimic evaluation and watching."""

from __future__ import annotations

import csv
import json
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate as supermimic_evaluate
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import EVAL_CSV_FIELDS


class FakeMotion:
    def __init__(self) -> None:
        self.names = ["hard", "truncated", "complete"]
        self.clip_start = torch.tensor([10, 20, 30])
        self.clip_len = torch.tensor([3, 4, 5])


class FakeMdp:
    num_joints = 2
    obs_dim = 3
    goal_frame_dim = 2
    max_step_reward = 0.1

    def __init__(self) -> None:
        self.motion = FakeMotion()
        self.goal_calls = []

    def obs(self, state, previous_action, context, frames):
        del state
        return torch.stack((frames.float(), previous_action[:, 0], context[:, 0]), dim=-1)

    def add_noise(self, obs):
        raise AssertionError("evaluation must not add observation noise")

    def goal(self, frames, num_frames):
        self.goal_calls.append(num_frames)
        return frames.float().unsqueeze(-1).expand(-1, self.goal_frame_dim * num_frames)


@dataclass
class FakeState:
    info: dict


class FakeEnv:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.mdp = FakeMdp()
        self.num_envs = len(self.mdp.motion.names)
        self.reset_frames = None
        self.actions = []
        self.state = None
        self.transitions = (
            {
                "r": torch.tensor([0.1, 0.1, 0.1]),
                "term": torch.tensor([False, False, False]),
                "done": torch.tensor([False, False, False]),
            },
            {
                "r": torch.tensor([0.05, 0.1, 0.1]),
                "term": torch.tensor([True, False, False]),
                "done": torch.tensor([True, False, False]),
            },
            {
                "r": torch.tensor([50.0, 0.05, 0.1]),
                "term": torch.tensor([True, False, False]),
                "done": torch.tensor([True, True, False]),
            },
            {
                "r": torch.tensor([50.0, 50.0, 0.05]),
                "term": torch.tensor([True, True, False]),
                "done": torch.tensor([True, True, True]),
            },
        )

    @property
    def supermimic_mdp(self):
        return self.mdp

    def _current_info(self, frames, previous_action=None):
        state = torch.zeros(self.num_envs, 13 + 2 * self.mdp.num_joints)
        state[:, 3] = 1.0
        if previous_action is None:
            previous_action = torch.zeros(self.num_envs, self.mdp.num_joints)
        return {
            "s": state,
            "f": frames.clone(),
            "xi": torch.full((self.num_envs, 1), 0.25),
            "a_prev": previous_action,
            "ep_step": torch.zeros(self.num_envs, dtype=torch.int64),
        }

    def reset_to_frames(self, frames):
        self.reset_frames = frames
        current = self._current_info(frames)
        self.state = FakeState({SUPERMIMIC_STATE_KEY: current})
        return {}, self.state.info

    def step(self, action):
        assert self.state is not None
        step_index = len(self.actions)
        self.actions.append(action.clone())
        current = self.state.info[SUPERMIMIC_STATE_KEY]
        transition = self.transitions[step_index]
        done = transition["done"]
        next_frames = torch.where(done, self.reset_frames, current["f"] + 1)
        next_action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
        next_current = self._current_info(next_frames, next_action)
        self.state = FakeState(
            {
                SUPERMIMIC_STATE_KEY: next_current,
                SUPERMIMIC_TRANSITION_KEY: transition,
            }
        )
        return self.state


class FakePolicy:
    instances = []

    def __init__(self, obs_dim, goal_dim, hist, z_dim, num_joints, hidden):
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.hist = hist
        self.z_dim = z_dim
        self.num_joints = num_joints
        self.hidden = hidden
        self.histories = []
        self.goals = []
        self.loaded = None
        self.eval_mode = False
        self.instances.append(self)

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state_dict):
        self.loaded = state_dict

    def eval(self):
        self.eval_mode = True

    def __call__(self, history, goal):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        self.histories.append(history.clone())
        self.goals.append(goal.clone())
        return torch.zeros(history.shape[0], self.num_joints)


def test_evaluate_first_done_history_and_return_aggregation(monkeypatch) -> None:
    FakePolicy.instances = []
    monkeypatch.setattr(networks, "Policy", FakePolicy)
    monkeypatch.setattr(networks, "AMP_DTYPE", None)
    env = FakeEnv()
    checkpoint = {
        "pi": {"marker": torch.tensor(7)},
        "args": {
            "goal_frames": 2,
            "hist": 3,
            "amp": "bf16",
            "z_dim": 16,
            "policy_hidden": [64, 32],
            "test_hidden_dims": [8, 4],
        },
    }

    result = supermimic_evaluate.evaluate(env, checkpoint, device="cpu")

    assert result["n_failed"] == 1
    assert result["failed"] == ["hard"]
    assert result["norm_return"] == pytest.approx((0.75 + 0.25 / 0.3 + 0.875) / 3)
    assert result["norm_return_framew"] == pytest.approx(0.75 / 0.9)
    assert result["per_clip"] == {
        "hard": {"norm_return": pytest.approx(0.75), "steps": 2, "failed": True},
        "truncated": {
            "norm_return": pytest.approx(0.25 / 0.3),
            "steps": 3,
            "failed": False,
        },
        "complete": {"norm_return": pytest.approx(0.875), "steps": 4, "failed": False},
    }
    assert len(env.actions) == 4
    assert all(torch.count_nonzero(action) == 0 for action in env.actions)
    torch.testing.assert_close(env.reset_frames, env.mdp.motion.clip_start)
    assert env.reset_frames.data_ptr() != env.mdp.motion.clip_start.data_ptr()

    policy = FakePolicy.instances[0]
    assert (policy.obs_dim, policy.goal_dim, policy.hist, policy.num_joints) == (3, 4, 3, 2)
    assert policy.z_dim == 16
    assert policy.hidden == (8, 4)
    assert policy.eval_mode
    assert policy.loaded is checkpoint["pi"]
    assert env.mdp.goal_calls == [2, 2, 2, 2]
    torch.testing.assert_close(
        policy.histories[0][:, :, 0],
        env.mdp.motion.clip_start.float().unsqueeze(-1).repeat(1, 3),
    )
    torch.testing.assert_close(policy.histories[1][2, :, 0], torch.tensor([30.0, 30.0, 31.0]))
    assert networks.AMP_DTYPE is torch.bfloat16


def _write_checkpoint(path, iteration: int) -> None:
    torch.save(
        {
            "pi": {},
            "args": {"goal_frames": 1, "hist": 1, "amp": "none"},
            "iteration": iteration,
            "wall_time": 7200.0,
            "saved_at": "2026-01-02 03:04:05",
        },
        path,
    )


def _watch_result(iteration: int) -> dict:
    del iteration
    return {
        "norm_return": 0.9,
        "norm_return_framew": 0.8,
        "n_failed": 0,
        "failed": [],
        "per_clip": {"clip": {"norm_return": 0.9, "steps": 10, "failed": False}},
    }


def test_watch_once_recovers_csv_and_skips_non_2000(monkeypatch, tmp_path) -> None:
    for iteration in (1000, 2000, 4000):
        _write_checkpoint(tmp_path / f"ckpt_{iteration:06d}.pt", iteration)
    with (tmp_path / "eval.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(EVAL_CSV_FIELDS)
        writer.writerow((2000, 1.0, "saved", "evaluated", 0.7, 0.7, 1, "old"))
    calls = []

    def fake_evaluate(env, checkpoint, device=None):
        del env, device
        calls.append(checkpoint["iteration"])
        return _watch_result(checkpoint["iteration"])

    monkeypatch.setattr(supermimic_evaluate, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        supermimic_evaluate.time, "sleep", lambda _: pytest.fail("once must not sleep")
    )
    env = SimpleNamespace(device=torch.device("cpu"))

    result = supermimic_evaluate.watch(env, tmp_path, once=True)

    assert result == _watch_result(4000)
    assert calls == [4000]
    assert not (tmp_path / "ckpt_001000_eval.json").exists()
    assert not (tmp_path / "ckpt_002000_eval.json").exists()
    with (tmp_path / "ckpt_004000_eval.json").open(encoding="utf-8") as stream:
        assert json.load(stream) == _watch_result(4000)
    with (tmp_path / "eval.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == list(EVAL_CSV_FIELDS)
    assert rows[-1][:4] == ["4000", "2.0000", "2026-01-02 03:04:05", rows[-1][3]]
    assert rows[-1][4:] == ["0.9000", "0.8000", "0", ""]
    assert not list(tmp_path.glob(".*.tmp"))


def test_watch_every_counts_2000_iteration_checkpoints(monkeypatch, tmp_path) -> None:
    for iteration in (2000, 4000, 6000):
        _write_checkpoint(tmp_path / f"ckpt_{iteration:06d}.pt", iteration)
    calls = []

    def fake_evaluate(env, checkpoint, device=None):
        del env, device
        calls.append(checkpoint["iteration"])
        result = _watch_result(checkpoint["iteration"])
        result["norm_return"] = 0.5
        return result

    monkeypatch.setattr(supermimic_evaluate, "evaluate", fake_evaluate)
    env = SimpleNamespace(device=torch.device("cpu"))

    supermimic_evaluate.watch(env, tmp_path, every=2, once=True)

    assert calls == [4000]
    assert (tmp_path / "ckpt_004000_eval.json").is_file()
    assert not (tmp_path / "ckpt_002000_eval.json").exists()
    assert not (tmp_path / "ckpt_006000_eval.json").exists()


def test_watch_returns_on_success_without_waiting(monkeypatch, tmp_path) -> None:
    for iteration in (2000, 4000):
        _write_checkpoint(tmp_path / f"ckpt_{iteration:06d}.pt", iteration)
    calls = []

    def fake_evaluate(env, checkpoint, device=None):
        del env, device
        calls.append(checkpoint["iteration"])
        result = _watch_result(checkpoint["iteration"])
        result["norm_return"] = 0.82
        return result

    monkeypatch.setattr(supermimic_evaluate, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        supermimic_evaluate.time, "sleep", lambda _: pytest.fail("success must stop watching")
    )
    env = SimpleNamespace(device=torch.device("cpu"))

    result = supermimic_evaluate.watch(env, tmp_path, success_threshold=0.82)

    assert result["norm_return"] == 0.82
    assert calls == [2000]
    assert not (tmp_path / "ckpt_004000_eval.json").exists()


@pytest.mark.parametrize("backend", supermimic_evaluate.SUPPORTED_BACKENDS)
def test_make_eval_env_composes_backend_owner(monkeypatch, tmp_path, backend: str) -> None:
    import hydra
    import unilab.base.config_adapter as config_adapter
    import unilab.base.process_device as process_device
    import unilab.training as training
    import unilab.utils.device as device

    import unisupermimic.scripts.train_supermimic as train
    import unisupermimic.sim2sim as sim2sim

    cfg = OmegaConf.create(
        {
            "training": {"sim_backend": backend, "sim2sim_strict": True},
            "env": {"commands": {"motion": {"motion_files": ["first.csv", "second.csv"]}}},
        }
    )
    sentinel = object()
    calls = {}

    def fake_compose(config_name, overrides):
        assert config_name == "config"
        calls["overrides"] = overrides
        return cfg

    def fake_create_env(received_cfg, num_envs, env_cfg_override):
        assert received_cfg is cfg
        assert num_envs == 2
        assert env_cfg_override == {"eval_mode": True}
        return sentinel

    monkeypatch.setattr(hydra, "compose", fake_compose)
    monkeypatch.setattr(hydra, "initialize_config_dir", lambda **kwargs: nullcontext())
    monkeypatch.setattr(config_adapter, "create_env", fake_create_env)
    monkeypatch.setattr(process_device, "configure_backend_process_device", lambda *args: None)
    monkeypatch.setattr(training, "ensure_registries", lambda: None)
    monkeypatch.setattr(device, "get_default_device", lambda: "cpu")
    monkeypatch.setattr(
        train,
        "build_env_override",
        lambda cfg, eval_mode, learner_device=None: {"eval_mode": eval_mode},
    )
    monkeypatch.setattr(sim2sim, "resolve_sim2sim_config", lambda *args, **kwargs: cfg)

    assert supermimic_evaluate.make_eval_env(tmp_path, backend=backend) is sentinel
    assert calls["overrides"] == [
        "training.play_only=true",
        f"task=g1_super_mimic/{backend}",
    ]


@pytest.mark.parametrize("backend", supermimic_evaluate.SUPPORTED_BACKENDS)
def test_eval_parser_accepts_backend(backend: str) -> None:
    args = supermimic_evaluate._argument_parser().parse_args(
        ["--ckpt", "run/ckpt.pt", "--backend", backend]
    )

    assert args.backend == backend
