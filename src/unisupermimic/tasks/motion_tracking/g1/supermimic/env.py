"""SuperMimic Manager-Based environment and backend factory."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.cpu_runtime import apply_cpu_confinement
from unilab.base.entity import EntityCfg
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.base.torch_env import TorchEnvState
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from unisim.backend.base import SimBackend

from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    SuperMimicMdpProtocol,
)

from .manager_terms import (
    SuperMimicCommand,
    make_current_info,
    make_transition_info,
    merge_current_info,
)
from .motion import CONTROL_DT
from .robot import COLLISION_GEOM_NAMES


class SuperMimicManagerBasedRlEnv(ManagerBasedRlEnv):
    """Publish detached SuperMimic transitions around the Manager reset boundary."""

    def __init__(self, cfg: ManagerBasedRlEnvCfg, backend: SimBackend, num_envs: int):
        if cfg.scale_rewards_by_dt:
            raise ValueError(
                "SuperMimic requires scale_rewards_by_dt=False; rewards already include dt"
            )
        if not math.isclose(cfg.ctrl_dt, CONTROL_DT, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"SuperMimic requires ctrl_dt={CONTROL_DT}")
        super().__init__(cfg, backend, num_envs)
        commands: list[SuperMimicCommand] = []
        for name in self.command_manager.active_terms:
            command = self.command_manager.get_term(name)
            if isinstance(command, SuperMimicCommand):
                commands.append(command)
        if len(commands) != 1:
            raise ValueError("SuperMimic requires exactly one SuperMimicCommand term")
        self._supermimic_command = commands[0]
        self._pre_action: dict[str, Any] | None = None

    @property
    def supermimic_mdp(self) -> SuperMimicMdpProtocol:
        return self._supermimic_command.mdp

    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
        command = self._supermimic_command
        self._pre_action = {
            "state": tuple(value.detach().clone() for value in command.state()),
            "frames": command.f.detach().clone(),
            "context": command.xi.detach().clone(),
            "previous_action": command.a_prev.detach().clone(),
            "ep_step": command.ep_step.detach().clone(),
            "action": actions.detach().clone(),
        }
        return super().apply_action(actions, state)

    def _update_state_in_read_phase(self, state: TorchEnvState) -> TorchEnvState:
        updated = super()._update_state_in_read_phase(state)
        if self._pre_action is None:
            raise RuntimeError("SuperMimic update_state requires a preceding apply_action snapshot")
        command = self._supermimic_command
        transition = make_transition_info(
            self._pre_action["state"],
            command.state(),
            self._pre_action["action"],
            self._pre_action["previous_action"],
            self._pre_action["context"],
            updated.reward,
            self._pre_action["frames"],
            self._pre_action["ep_step"],
            updated.terminated,
            updated.truncated,
        )
        transition.update(command.last_episode_stats)
        updated.info[SUPERMIMIC_TRANSITION_KEY] = transition
        updated.info[SUPERMIMIC_STATE_KEY] = self._current_info()
        self._pre_action = None
        return updated

    def _current_info(self) -> dict[str, torch.Tensor]:
        command = self._supermimic_command
        return make_current_info(
            command.state(),
            command.f,
            command.xi,
            command.a_prev,
            command.ep_step,
        )

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if env_ids is None:
            reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            ids = resolve_reset_env_ids(
                env_ids,
                num_envs=self.num_envs,
                device=self.device,
                owner="SuperMimicManagerBasedRlEnv.reset",
            )
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[ids] = True
        obs, info = super().reset(env_ids, seed=seed, options=options)
        assert self._state is not None
        current = self._current_info()
        previous = self._state.info.get(SUPERMIMIC_STATE_KEY)
        if previous is not None:
            current = merge_current_info(previous, current, reset_mask)
        self._state.info[SUPERMIMIC_STATE_KEY] = current
        info[SUPERMIMIC_STATE_KEY] = current
        return obs, info

    def reset_to_frames(
        self,
        frames: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset every or selected environment to requested motion-library frames."""
        self._supermimic_command.request_frames(frames, env_ids=env_ids)
        return self.reset(env_ids)


SUPPORTED_BACKENDS = ("mjwarp", "mujoco", "motrix")


def make_supermimic_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mjwarp",
) -> SuperMimicManagerBasedRlEnv:
    """Construct SuperMimic through the common public backend factory path."""
    if backend_type not in SUPPORTED_BACKENDS:
        supported = ", ".join(repr(backend) for backend in SUPPORTED_BACKENDS)
        raise ValueError(f"make_supermimic_env supports backends {supported}; got {backend_type!r}")
    if not isinstance(cfg, ManagerBasedRlEnvCfg):
        raise TypeError("make_supermimic_env expected ManagerBasedRlEnvCfg")
    assert cfg.scene is not None
    robot_cfg = cfg.scene.entities["robot"]
    if not isinstance(robot_cfg, EntityCfg):
        raise TypeError("SuperMimic robot scene entity must be an EntityCfg")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
    if robot_cfg.root_body_name is None:
        raise ValueError("SuperMimic requires robot.root_body_name")
    if robot_cfg.geom_names is None:
        cfg.scene.entities["robot"] = replace(robot_cfg, geom_names=COLLISION_GEOM_NAMES)
    cfg.validate()
    if cfg.fixed_model_variants is not None:
        raise ValueError("SuperMimic does not support fixed model variants")
    apply_cpu_confinement(cfg.cpu_ids)
    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=True,
        base_name=robot_cfg.root_body_name,
        **env_backend_kwargs(cfg),
    )
    try:
        return SuperMimicManagerBasedRlEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise


__all__ = ["SuperMimicManagerBasedRlEnv", "make_supermimic_env"]
