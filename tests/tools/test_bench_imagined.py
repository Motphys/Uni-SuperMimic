"""Asset-free smoke and configuration tests for the imagined benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from unisupermimic.rl.algos.supermimic_imagined.benchmark import (
    BenchmarkError,
    BenchmarkSettings,
    build_learner_settings,
    configure_optimizer,
    main,
    run_benchmark,
)
from unisupermimic.rl.algos.supermimic_imagined.fake import make_fake_imagined_task
from unisupermimic.rl.algos.supermimic_imagined.learner import ImaginedSuperMimicLearner


def _settings(**overrides):
    values = {
        "iterations": 2,
        "warmup": 1,
        "utilization_interval": 0.01,
        "label": "smoke",
        "batch": 4,
        "horizon": 2,
        "goal_frames": 2,
        "hist": 2,
        "num_frames": 32,
        "pi_ema": 0.0,
        "context_dim": 33,
    }
    values.update(overrides)
    return BenchmarkSettings(**values)


def test_fake_task_shapes_match_dimensions() -> None:
    task = make_fake_imagined_task(batch=5, num_joints=4, context_dim=33, wm_hidden=(12, 8))

    assert task.mdp.obs_dim == 3 + 3 + 3 * 4 + 33
    assert task.world_model.in_dim == 13 + 12 + 33
    assert task.world_model.out_dim == 12 + 8
    assert task.world_model.net[0].in_features == task.world_model.in_dim
    assert all(not parameter.requires_grad for parameter in task.world_model.parameters())


def test_benchmark_cpu_smoke_with_profiler_artifacts(tmp_path: Path) -> None:
    report = run_benchmark(_settings(profile="torch"), tmp_path)

    assert report["timing"]["iterations"] == 2
    assert report["timing"]["imagined_steps_total"] == 4 * 2 * 2
    assert report["timing"]["iterations_per_second"] > 0
    assert report["timing"]["imagined_steps_per_second"] > 0
    assert report["timing"]["seconds_per_iteration"]["median"] > 0
    assert Path(report["profile"]["chrome_trace"]).is_file()
    assert Path(report["profile"]["profiler_table"]).is_file()
    parsed = json.loads((tmp_path / "benchmark.json").read_text())
    assert parsed["schema_version"] == 1
    assert parsed["settings"]["task"] == "fake"


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--device",
            "cpu",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--batch",
            "3",
            "--horizon",
            "1",
            "--critic-minibatches",
            "1",
            "--profile",
            "torch",
            "--utilization-interval",
            "0.01",
            "--label",
            "cli",
            "--output",
            str(tmp_path / "cli"),
        ]
    )

    assert exit_code == 0
    assert "imagined steps/s" in capsys.readouterr().out
    assert (tmp_path / "cli" / "benchmark.json").is_file()


def test_tf32_modes_are_reported_without_mutating_unsupported_device(tmp_path: Path) -> None:
    report = run_benchmark(_settings(tf32="on", iterations=1, warmup=0), tmp_path)

    if torch.cuda.is_available():
        assert report["precision"]["tf32_matmul"] is True
    else:
        assert report["precision"]["tf32_matmul"] is False


def test_fused_adam_is_disabled_on_cpu(tmp_path: Path) -> None:
    device = torch.device("cpu")
    task = make_fake_imagined_task(device=device)
    learner = ImaginedSuperMimicLearner(
        task,
        build_learner_settings(BenchmarkSettings(iterations=1)),
        tmp_path / "learner",
        device,
    )
    with pytest.raises(BenchmarkError, match="fused Adam is only available on CUDA"):
        configure_optimizer(learner, BenchmarkSettings(fused_adam="on", device="cpu"), device)
    result = configure_optimizer(learner, BenchmarkSettings(fused_adam="off"), device)

    assert result["fused_adam_effective"] is False
    assert learner.opt_v.defaults.get("fused", False) is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"amp": "fp16"}, "AMP mode"),
        ({"pi_opt": "muon", "policy_hidden": (16,)}, "MuonAdam"),
        ({"fused_adam": "on", "device": "cpu"}, "fused Adam"),
        ({"task": "real"}, "checkpoint"),
    ],
)
def test_invalid_settings_are_rejected(overrides, match) -> None:
    with pytest.raises((BenchmarkError, ValueError), match=match):
        run_benchmark(_settings(**overrides), "/tmp/unisupermimic-invalid-bench")
