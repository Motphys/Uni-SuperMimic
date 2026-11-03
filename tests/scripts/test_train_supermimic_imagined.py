"""Configuration tests for the standalone imagined-training entrypoint."""

from __future__ import annotations

from dataclasses import fields

from hydra import compose, initialize_config_module

from unisupermimic.rl.algos.supermimic_imagined.learner import (
    ImaginedSuperMimicConfig,
)
from unisupermimic.scripts.train_supermimic_imagined import (
    build_learner_config,
    build_task_config,
)


def _compose():
    with initialize_config_module(config_module="unisupermimic.conf.imagined", version_base="1.3"):
        return compose(config_name="config")


def test_imagined_config_composes_without_env() -> None:
    cfg = _compose()

    assert cfg.task_name == "G1SuperMimicImagined"
    assert "env" not in cfg
    assert "backend" not in cfg.training
    assert cfg.task.motion.motion_files
    assert cfg.algo.wm_checkpoint is None


def test_task_config_builds_from_hydra() -> None:
    task = build_task_config(_compose())

    assert task.motion_files
    assert task.mjcf_path.endswith("supermimic.xml")
    assert task.gamma == 0.99


def test_learner_config_builds_from_hydra() -> None:
    learner = build_learner_config(_compose())

    assert isinstance(learner, ImaginedSuperMimicConfig)
    assert learner.batch == 4096
    assert learner.horizon == 32
    assert learner.policy_hidden == (2048, 1024, 512, 256)
    assert learner.test_hidden_dims is None
    assert all(field.name != "wm_checkpoint" for field in fields(ImaginedSuperMimicConfig))
