"""SuperMimic Manager-Based environment and backend factory."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.cpu_runtime import apply_cpu_confinement
from unilab.base.entity import EntityCfg
from unilab.base.reset_selection import ResetSelection
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.base.torch_env import TorchEnvState
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from unisim.backend.base import SimBackend
from unisim.backend.mujoco.backend import MuJoCoBackend

from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    SuperMimicMdpProtocol,
)

from .manager_terms import (
    SuperMimicCommand,
    make_current_info,
    make_transition_info,
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
        # Fixed-address carriers for SUPERMIMIC_STATE_KEY: the reset-segment
        # CUDA graph replays recorded device work only, so host-side dict
        # re-assignment inside the captured body (the eager merge) would run
        # once at capture and never again. Publishing through persistent
        # tensors that refresh in place keeps every replay semantically equal
        # to eager execution.
        self._sm_state: dict[str, torch.Tensor] | None = None
        self._sm_reset_scratch: dict[str, torch.Tensor] | None = None
        # The pre-reset transition materialization is step-hot and pure-tensor;
        # compile it with the command's MDP when the task opts in. The reset
        # path (current-info snapshots) stays eager: it runs inside the
        # reset-segment CUDA graph body.
        self._make_transition_info = (
            torch.compile(make_transition_info) if commands[0].cfg.compile else make_transition_info
        )

    @property
    def supermimic_mdp(self) -> SuperMimicMdpProtocol:
        return self._supermimic_command.mdp

    @property
    def supermimic_exam_mask(self) -> torch.Tensor:
        """Full-width bool mask of the exam worlds (all false until activated)."""
        return self._supermimic_command.exam_mask

    def enable_supermimic_exams(self, num_rows: int) -> None:
        """Activate BlinkMimic2-style exam worlds on the last ``num_rows`` rows.

        The assignment happens once (no later reshuffling): those rows run exam
        episodes from here on — clip-first-frame starts, no early termination,
        EMA-policy actions from the learner — and their resets continue through
        the ordinary (captured) masked autoreset, which the command's exam
        override routes to the next clip's first frame. Runs eagerly at an
        iteration boundary, outside any captured segment.
        """
        if isinstance(num_rows, bool) or not isinstance(num_rows, int):
            raise TypeError(f"SuperMimic exam row count must be an int, got {num_rows!r}")
        if not 0 < num_rows < self.num_envs:
            raise ValueError(
                f"SuperMimic exam row count must be in (0, {self.num_envs}), got {num_rows}"
            )
        frames = self._supermimic_command.activate_exam_worlds(num_rows)
        env_ids = torch.arange(
            self.num_envs - num_rows, self.num_envs, dtype=torch.int64, device=self.device
        )
        self.reset_to_frames(frames[env_ids], env_ids=env_ids)

    def configure_supermimic_exam_margin(self, *, frontier_frac: float, fail_ema: float) -> None:
        """Turn on exams-v2 margin mode on the command (eager, pre-capture).

        Per-clip failure statistics start accumulating from the next step and
        practice resets may draw frontier starts; exam clip selection switches
        from the round-robin cursor to the margin-priority table. Runs once at
        learner construction, before any reset-segment graph capture, so the
        captured segment bakes the same Python-level branches.
        """
        self._supermimic_command.configure_exam_margin(
            frontier_frac=frontier_frac, fail_ema=fail_ema
        )

    def refresh_supermimic_exam_clips(self, num_rows: int) -> torch.Tensor:
        """Re-select the exam clips by margin priority (iteration boundary).

        Runs eagerly outside any captured segment: the fixed-address exam clip
        table takes the top-K margin-priority clips (one exam row per clip)
        and every later captured-reset replay gathers their first frames.
        Returns the selected clips' mean failure EMA for logging.
        """
        return self._supermimic_command.refresh_exam_clips(num_rows)

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

    def update_state(self, state: TorchEnvState) -> TorchEnvState:
        # Native mj_step may leave tracked body sensors one substep behind qpos/qvel.
        # Refresh before the Manager packs the first post-physics read packet.
        if isinstance(self.backend, MuJoCoBackend):
            self.backend._sync_tracked_body_state()
        return super().update_state(state)

    def _update_state_in_read_phase(self, state: TorchEnvState) -> TorchEnvState:
        updated = super()._update_state_in_read_phase(state)
        if self._pre_action is None:
            raise RuntimeError("SuperMimic update_state requires a preceding apply_action snapshot")
        command = self._supermimic_command
        # Refresh the fixed-address live-state mirror while the read-plan packet
        # is fresh: the captured reset segment publishes through it and must
        # never read the packet tensors, which the read plan reallocates on
        # every refresh.
        command.refresh_live_state()
        transition = self._make_transition_info(
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
        self._sm_state = self._current_info_into(self._sm_state)
        updated.info[SUPERMIMIC_STATE_KEY] = self._sm_state
        self._pre_action = None
        return updated

    def _current_info_into(
        self,
        target: dict[str, torch.Tensor] | None,
        reset_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        command = self._supermimic_command
        if reset_mask is None:
            state: tuple[torch.Tensor, ...] = command.state()
        else:
            # Post-reset publication inside the captured segment: done rows take
            # the state the reset wrote and every other row takes the fixed-
            # address live mirror refreshed by update_state this step. Neither
            # may read the read-plan packet: its derived body tensors are
            # reallocated on every refresh, so a captured read would fetch
            # whichever allocation now owns the capture-time address.
            written_state = command.reset_state_carrier
            live_state = command._live_state_carrier
            state = tuple(
                torch.where(
                    reset_mask.reshape((reset_mask.shape[0],) + (1,) * (written.ndim - 1)),
                    written,
                    read,
                )
                for written, read in zip(written_state, live_state, strict=True)
            )
        fresh = make_current_info(
            state,
            command.f,
            command.xi,
            command.a_prev,
            command.ep_step,
        )
        if target is None:
            return fresh
        for name, value in fresh.items():
            target[name].copy_(value)
        return target

    def reset(
        self,
        env_ids: torch.Tensor | ResetSelection | None = None,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if isinstance(env_ids, ResetSelection):
            # Fixed-shape autoreset: the env-owned selection mask aliases the
            # persistent buffer, so no allocation and no host read happen here.
            reset_mask = env_ids.mask
        elif env_ids is None:
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
        obs, info = super().reset(env_ids, seed=seed)
        assert self._state is not None
        # CUDA-graph contract: the autoreset body may execute inside the
        # reset-segment graph, where recorded device work replays but host
        # dict re-assignment does not. Publish the reset through fixed-address
        # carriers instead: the scratch materializes the just-reset state, the
        # persistent buffer still holds the pre-reset snapshot written by
        # update_state this step, and the masked copy merges them — exactly the
        # eager-path merge_current_info semantics, expressed as pure device
        # work on stable tensors.
        current = self._current_info_into(self._sm_reset_scratch, reset_mask=reset_mask)
        if self._sm_state is None:
            self._sm_state = {name: value.clone() for name, value in current.items()}
        else:
            for name, value in self._sm_state.items():
                mask = reset_mask.reshape(
                    (reset_mask.shape[0],) + (1,) * (value.ndim - 1)
                )
                value.copy_(torch.where(mask, current[name], value))
        self._state.info[SUPERMIMIC_STATE_KEY] = self._sm_state
        info[SUPERMIMIC_STATE_KEY] = self._sm_state
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
