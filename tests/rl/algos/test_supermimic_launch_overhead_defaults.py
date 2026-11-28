"""The launch-overhead switches default to ``auto``: on for NVIDIA CUDA, original path elsewhere."""

from __future__ import annotations

import csv
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
import yaml
from tests.rl.algos.test_supermimic_learner import FakeEnv, _config

import unisupermimic
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
# ``None`` / ``"auto"`` follow the device; the C++ inductor wrapper stays opt-in.
AUTO = {
    "critic_update": "auto",
    "wm_graph": None,
    "policy_graph": None,
    "optimizer_graph": None,
    "stats_on_device": None,
    "inductor_cpp_wrapper": False,
}
ORIGINAL = {
    "critic_update": "eager",
    "wm_graph": False,
    "policy_graph": False,
    "optimizer_graph": False,
    "stats_on_device": False,
    "inductor_cpp_wrapper": False,
}


def _original(**overrides) -> SuperMimicConfig:
    return replace(_config(), **ORIGINAL, **overrides)


def _assert_original_paths(learner: SuperMimicLearner) -> None:
    assert learner._critic_graph_enabled is False
    assert learner._wm_graph is None and learner._pi_graph is None
    assert learner._wm_step_graph is None and learner._pi_step_graph is None
    assert learner._stats_on_device is False
    assert learner._dev_stats is None


def test_switch_defaults_are_auto_in_the_dataclass_and_the_task_config() -> None:
    config = SuperMimicConfig()
    assert {name: getattr(config, name) for name in AUTO} == AUTO
    assert set(AUTO) <= {field.name for field in fields(SuperMimicConfig)}
    task_config = Path(unisupermimic.__file__).parent / "conf" / "supermimic" / "config.yaml"
    algo = yaml.safe_load(task_config.read_text(encoding="utf-8"))["algo"]
    assert {name: algo[name] for name in AUTO} == AUTO


def test_auto_keeps_every_original_path_off_cuda(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    _assert_original_paths(learner)
    learner.collect(12)
    for _ in range(5):  # past the capture call of every graph
        learner.wm_update()
    for _ in range(4):
        learner.policy_update()
    _assert_original_paths(learner)
    assert learner._critic_graph_state is None
    # The synced statistics still fill the lists.
    assert len(learner.stats["r"]) == 12


def test_explicit_switches_need_nvidia_cuda(tmp_path) -> None:
    for name, value in (
        ("wm_graph", True),
        ("policy_graph", True),
        ("optimizer_graph", True),
        ("critic_update", "graph"),
    ):
        with pytest.raises(ValueError, match=f"{name}=?.?'?{value}'? requires"):
            SuperMimicLearner(FakeEnv(), replace(_config(), **{name: value}), tmp_path / name)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
def test_explicit_off_keeps_the_original_paths_on_any_device(tmp_path, device) -> None:
    learner = SuperMimicLearner(FakeEnv(), _original(), tmp_path / device, device=device)
    _assert_original_paths(learner)


@needs_cuda
def test_auto_enables_every_switch_on_nvidia_cuda(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path, device="cuda")
    assert learner._cuda_graphs
    assert learner._critic_graph_enabled
    assert learner._wm_graph is not None and learner._pi_graph is not None
    assert learner._wm_step_graph is not None and learner._pi_step_graph is not None
    assert learner._stats_on_device and learner._dev_stats is not None

    learner.collect(12)
    assert learner._dev_stats[0].item() == 12  # the statistics never went through the host
    assert learner.stats["r"] == []
    for _ in range(5):
        learner.wm_update()
    for _ in range(4):
        learner.policy_update()
    # Past their warm-up calls, every update replays from its CUDA graph.
    assert learner._wm_graph.replaying and learner._wm_step_graph.replaying
    assert learner._pi_graph.replaying and learner._pi_step_graph.replaying
    assert learner._critic_graph_state is not None


@needs_cuda
def test_optimizer_step_graph_stays_eager_without_what_it_needs(tmp_path) -> None:
    # ``auto`` applies the step graph where its prerequisites hold and leaves the other
    # step eager; an explicit request is a configuration error instead.
    config = replace(_config(), no_sync=False)
    learner = SuperMimicLearner(FakeEnv(), config, tmp_path / "synced", device="cuda")
    assert learner._wm_step_graph is not None
    assert learner._pi_step_graph is None  # a step skipped on the host is a host branch
    learner = SuperMimicLearner(
        FakeEnv(), replace(_config(), pi_opt="adam"), tmp_path / "adam", device="cuda"
    )
    assert learner._wm_step_graph is not None
    assert learner._pi_step_graph is None  # the stock Adam keeps its float learning rate
    assert learner.opt_pi.defaults["capturable"] is False

    for overrides in ({"no_sync": False}, {"wm_graph": False}, {"pi_opt": "adam"}):
        with pytest.raises(ValueError, match="optimizer_graph=True requires"):
            SuperMimicLearner(
                FakeEnv(),
                replace(_config(), optimizer_graph=True, **overrides),
                tmp_path / "-".join(overrides),
                device="cuda",
            )


@needs_cuda
def test_default_training_loop_runs_on_cuda_graphs(tmp_path) -> None:
    config = replace(_config(), iterations=3, log_every=1, ckpt_every=1000)
    summary = SuperMimicLearner(FakeEnv(), config, tmp_path, device="cuda").train()
    assert summary["completed"]
    with (tmp_path / "train_log.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["iteration"]) for row in rows] == [1, 2, 3]
