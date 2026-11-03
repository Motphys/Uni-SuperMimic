"""Tests for the standalone frozen-world-model imagination task."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from tests.tasks.test_supermimic_mdp import _write_clip, _write_model

from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import default_joint_pos_list
from unisupermimic.tasks.motion_tracking.g1.supermimic_imagined.task import (
    G1ImaginedConfig,
    build_g1_imagined_task,
)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("imagined_task")
    model = root / "g1.xml"
    clip = root / "clip.csv"
    _write_model(model)
    _write_clip(clip, 8)
    return model, clip


def _checkpoint(device: str, structured: bool = False) -> dict[str, object]:
    wm = WorldModel((8, 8), num_joints=29, context_dim=33, structured=structured).to(device)
    with torch.no_grad():
        for parameter in wm.parameters():
            parameter.add_(0.125)
    if structured:
        with torch.no_grad():
            wm.inc_norm.var.fill_(4.0)
    return {"wm": wm.state_dict()}


def test_builds_offline_task_and_infers_checkpoint_layout(inputs) -> None:
    model, clip = inputs
    task = build_g1_imagined_task(
        G1ImaginedConfig(motion_files=(str(clip),), mjcf_path=str(model)), _checkpoint("cpu"), "cpu"
    )

    assert task.wm_hidden == (8, 8)
    assert task.wm_structured is False
    assert task.mdp.obs_dim == 99
    assert task.mdp.num_joints == 29
    assert all(not parameter.requires_grad for parameter in task.world_model.parameters())
    assert not task.world_model.training


def test_rejects_checkpoint_without_world_model(inputs) -> None:
    model, clip = inputs
    with pytest.raises(TypeError, match="wm"):
        build_g1_imagined_task(
            G1ImaginedConfig(motion_files=(str(clip),), mjcf_path=str(model)), {}, "cpu"
        )


def test_structured_checkpoint_is_detected(inputs) -> None:
    model, clip = inputs
    task = build_g1_imagined_task(
        G1ImaginedConfig(motion_files=(str(clip),), mjcf_path=str(model)),
        _checkpoint("cpu", structured=True),
        "cpu",
    )
    assert task.wm_structured is True
