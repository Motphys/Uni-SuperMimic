"""Hydra and checkpoint routing tests for the SuperMimic entrypoint."""

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

from unisupermimic.scripts.train_supermimic import (
    build_env_override,
    build_learner_config,
    main,
    resolve_supermimic_checkpoint,
)
from unisupermimic.sim2sim import extract_contract_snapshot
from unisupermimic.tasks.motion_tracking.g1.supermimic.env import SUPPORTED_BACKENDS


def _compose(backend: str = "mjwarp") -> Any:
    root = Path(__file__).resolve().parents[2]
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(root / "src" / "unisupermimic" / "conf" / "supermimic"),
        version_base="1.3",
    ):
        return compose("config", overrides=[f"task=g1_super_mimic/{backend}"])


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_owner_materializes_complete_task(backend: str) -> None:
    cfg = _compose(backend)
    override = EnvConfigAdapter(
        cfg, root_dir=Path.cwd(), algo_name="supermimic"
    ).build_task_env_cfg_override()
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config("G1SuperMimic")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()

    assert cfg.training.sim_backend == backend
    assert len(env_cfg.commands["motion"].motion_files) == 40
    assert env_cfg.sim_dt == pytest.approx(0.005)
    assert env_cfg.ctrl_dt == pytest.approx(0.02)
    assert env_cfg.scale_rewards_by_dt is False
    assert env_cfg.auto_reset is True
    assert env_cfg.mjwarp_nconmax == (128 if backend == "mjwarp" else None)
    assert env_cfg.mjwarp_njmax == (500 if backend == "mjwarp" else None)
    assert env_cfg.commands["motion"].goal_frames == 10
    assert env_cfg.rewards["tracking"].weight == pytest.approx(1.0)
    for event_name in ("startup_context", "reset_context"):
        randomize_friction = env_cfg.events[event_name].params.get("randomize_friction", True)
        assert randomize_friction is (backend != "motrix")


def test_backend_owners_have_identical_sim2sim_contracts() -> None:
    snapshots = [extract_contract_snapshot(_compose(backend)) for backend in SUPPORTED_BACKENDS]

    assert snapshots == [snapshots[0]] * len(SUPPORTED_BACKENDS)


def test_learner_config_uses_algorithm_dataclass_fields() -> None:
    cfg = _compose()
    OmegaConf.update(cfg, "algo.test_hidden_dims", [32, 16])

    learner_cfg = build_learner_config(cfg)

    assert learner_cfg.iterations == 12000
    assert learner_cfg.goal_frames == 10
    assert learner_cfg.hist == 10
    assert learner_cfg.wm_window == 8
    assert learner_cfg.horizon == 32
    assert learner_cfg.test_hidden_dims == (32, 16)
    assert not hasattr(learner_cfg, "num_envs")


def test_eval_override_only_changes_evaluation_lifecycle() -> None:
    cfg = _compose()
    override = build_env_override(cfg, eval_mode=True)

    assert override["commands"]["motion"]["eval_mode"] is True
    assert override["commands"]["motion"]["reset_noise"] is False
    assert override["auto_reset"] is True
    assert override["observations"]["policy"]["enable_corruption"] is False
    assert cfg.env.commands.motion.eval_mode is False
    assert cfg.env.commands.motion.reset_noise is True


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("mujoco", "cuda:3"), ("motrix", "cuda:3"), ("mjwarp", None)],
)
def test_collector_tensor_device_auto_follows_learner_device(
    backend: str, expected: str | None
) -> None:
    override = build_env_override(_compose(backend), learner_device="cuda:3")

    assert override.get("collector_tensor_device") == expected


def test_collector_tensor_device_auto_routes_cpu_learner_to_cpu() -> None:
    override = build_env_override(_compose("mujoco"), learner_device="cpu")

    assert override["collector_tensor_device"] == "cpu"


def test_collector_tensor_device_auto_is_noop_without_learner_device() -> None:
    assert "collector_tensor_device" not in build_env_override(_compose("mujoco"))


def test_collector_tensor_device_explicit_training_value_wins() -> None:
    cfg = _compose("mujoco")
    OmegaConf.update(cfg, "training.collector_tensor_device", "cuda")

    override = build_env_override(cfg, learner_device="cpu")

    assert override["collector_tensor_device"] == "cuda"


def test_training_rejects_mismatched_mdp_discount() -> None:
    cfg = _compose()
    OmegaConf.update(cfg, "algo.gamma", 0.98)
    with pytest.raises(ValueError, match="algo.gamma == env.commands.motion.gamma"):
        main(cfg)


def test_checkpoint_resolver_supports_resume_and_numbered_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _compose()
    run_dir = tmp_path / "logs" / "supermimic" / "G1SuperMimic" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "last.pt").touch()
    (run_dir / "ckpt_002000.pt").touch()
    monkeypatch.chdir(tmp_path)
    cfg.training.log_root = str(tmp_path / "logs" / "supermimic")
    cfg.algo.load_run = "run"

    cfg.algo.checkpoint = -1
    assert resolve_supermimic_checkpoint(cfg) == run_dir / "last.pt"
    cfg.algo.checkpoint = 2000
    assert resolve_supermimic_checkpoint(cfg) == run_dir / "ckpt_002000.pt"
    cfg.algo.checkpoint = "missing.pt"
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        resolve_supermimic_checkpoint(cfg)
