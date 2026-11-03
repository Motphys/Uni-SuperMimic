"""Train G1 SuperMimic entirely inside a frozen differentiable world model."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from unisupermimic.rl.algos.supermimic_imagined.learner import (
    ImaginedSuperMimicConfig,
    ImaginedSuperMimicLearner,
)
from unisupermimic.sim2sim import extract_contract_snapshot
from unisupermimic.tasks.motion_tracking.g1.supermimic_imagined.task import (
    G1ImaginedConfig,
    build_g1_imagined_task,
)


def _checkpoint_path(cfg: DictConfig) -> Path:
    value = OmegaConf.select(cfg, "algo.wm_checkpoint", default=None)
    if value is None or str(value) == "":
        raise ValueError("G1SuperMimicImagined requires algo.wm_checkpoint")
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"frozen world-model checkpoint does not exist: {path}")
    return path


def build_task_config(cfg: DictConfig) -> G1ImaginedConfig:
    raw = OmegaConf.to_container(cfg.task.motion, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("cfg.task.motion must resolve to a mapping")
    raw.pop("_target_", None)
    raw["motion_files"] = tuple(raw.get("motion_files", ()))
    raw["amp_ankles"] = tuple(raw.get("amp_ankles", (3, 6)))
    raw["speed_bodies"] = tuple(raw.get("speed_bodies", (3, 6, 9, 11)))
    return G1ImaginedConfig(**raw)


def build_learner_config(cfg: DictConfig) -> ImaginedSuperMimicConfig:
    raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("cfg.algo must resolve to a mapping")
    values = {
        field.name: raw[field.name]
        for field in fields(ImaginedSuperMimicConfig)
        if field.name in raw and field.name != "wm_checkpoint"
    }
    for name in ("policy_hidden", "critic_hidden", "test_hidden_dims"):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    return ImaginedSuperMimicConfig(**values)


def _log_dir(cfg: DictConfig) -> Path:
    configured = OmegaConf.select(cfg, "training.log_dir", default=None)
    if configured:
        return Path(str(configured)).expanduser()
    root = OmegaConf.select(cfg, "training.log_root", default=None)
    base = Path(str(root).expanduser()) if root else Path.cwd() / "runs"
    return base / cfg.training.task_name / "imagined"


def main(cfg: DictConfig) -> None:
    checkpoint_file = _checkpoint_path(cfg)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    task_cfg = build_task_config(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(cfg.algo.seed))
    task = build_g1_imagined_task(task_cfg, checkpoint, device)
    learner_cfg = build_learner_config(cfg)
    if float(learner_cfg.gamma) != float(task_cfg.gamma):
        raise ValueError("G1SuperMimicImagined requires algo.gamma == task.motion.gamma")

    log_dir = _log_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = log_dir / "config.yaml"
    OmegaConf.save(config=cfg, f=str(snapshot_path))
    snapshot = extract_contract_snapshot(cfg)
    (log_dir / "contract_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )

    learner = ImaginedSuperMimicLearner(task, learner_cfg, log_dir)
    summary = learner.train()
    print(json.dumps(summary, indent=2))


@hydra.main(
    version_base="1.3",
    config_path="../conf/imagined",
    config_name="config",
)
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()
