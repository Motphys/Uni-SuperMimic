"""Manager-Based terms for the G1 SuperMimic mjwarp task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers import (
    ActionTerm,
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    EventTermCfg,
    ManagerRng,
    ManagerTermBase,
    SceneEntityCfg,
)

from unisupermimic.rl.algos.supermimic.geometry import (
    flatten_state,
    mat_to_quat,
    quat_to_mat,
    so3_exp,
)
from unisupermimic.rl.contracts.supermimic import State

from .kinematics import G1Kinematics
from .mdp import MAX_STEP_REWARD, MDP, NUM_J, XI_DIM
from .motion import BIN_FRAMES, MotionLib
from .robot import G1_JOINT_NAMES, action_scale

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv as RuntimeEnv
    from unilab.managers.types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def start_bin_probabilities(
    bin_G: torch.Tensor,
    bin_w_avg: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Update the past-performance EMA and return the exact bin mixture."""
    soft = torch.softmax(-25.0 * bin_G / (MAX_STEP_REWARD / (1.0 - gamma)), dim=0)
    bin_w_avg.mul_(0.999).add_(0.001 * soft)
    return 0.5 / bin_G.numel() + 0.4 * soft + 0.1 * bin_w_avg / bin_w_avg.sum()


def sample_start_frames(
    motion: MotionLib,
    bin_G: torch.Tensor,
    bin_w_avg: torch.Tensor,
    gamma: float,
    num_samples: int,
    rng: ManagerRng,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one-second bins and uniformly sample a frame inside each bin."""
    probabilities = start_bin_probabilities(bin_G, bin_w_avg, gamma)
    bins = rng.choice(motion.num_bins, (num_samples,), p=probabilities)
    start = motion.bin_start[bins]
    end = torch.minimum(start + BIN_FRAMES, motion.clip_end[motion.bin_clip[bins]])
    frames = start + (rng.random((num_samples,)) * (end - start).float()).long()
    return frames, bins


def update_start_bin_returns(
    bin_G: torch.Tensor,
    start_bin: torch.Tensor,
    ep_step: torch.Tensor,
    ep_ret: torch.Tensor,
    ep_disc_ret: torch.Tensor,
    hard: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Update completed-episode returns and return done-row logging statistics."""
    if not bool(done.any()):
        return {}
    lengths = ep_step[done].float()
    bootstrap = torch.where(
        hard[done],
        0.0,
        gamma**lengths * ep_ret[done] / lengths / (1.0 - gamma),
    )
    returns = ep_disc_ret[done] + bootstrap
    bins = start_bin[done]
    bin_G.index_put_((bins,), 0.8 * bin_G[bins] + 0.2 * returns)
    return {
        "ep_len": lengths,
        "ep_ret_norm": ep_ret[done] / (lengths * MAX_STEP_REWARD),
    }


def make_current_info(
    state: State,
    frames: torch.Tensor,
    context: torch.Tensor,
    previous_action: torch.Tensor,
    ep_step: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Materialize the learner's current-state input without backend views."""
    return {
        "s": flatten_state(state).detach().clone(),
        "f": frames.detach().clone(),
        "xi": context.detach().clone(),
        "a_prev": previous_action.detach().clone(),
        "ep_step": ep_step.detach().clone(),
    }


def make_transition_info(
    state: State,
    next_state: State,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    context: torch.Tensor,
    reward: torch.Tensor,
    frames: torch.Tensor,
    ep_step: torch.Tensor,
    hard: torch.Tensor,
    timeout: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Materialize one pre-reset transition, including both done causes."""
    return {
        "s": flatten_state(state).detach().clone(),
        "s1": flatten_state(next_state).detach().clone(),
        "a": action.detach().clone(),
        "a_prev": previous_action.detach().clone(),
        "xi": context.detach().clone(),
        "r": reward.detach().clone(),
        "f": frames.detach().clone(),
        "ep_step": ep_step.detach().clone(),
        "term": hard.detach().clone(),
        "done": (hard | timeout).detach().clone(),
    }


def merge_current_info(
    previous: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    reset_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Merge post-reset rows while preserving all non-reset current snapshots."""
    if set(previous) != set(current):
        raise ValueError("SuperMimic current-info keys changed across reset")
    merged: dict[str, torch.Tensor] = {}
    for name, values in current.items():
        mask = reset_mask.reshape((reset_mask.shape[0],) + (1,) * (values.ndim - 1))
        merged[name] = torch.where(mask, values, previous[name]).detach().clone()
    return merged


def joint_action_target(
    default_q: torch.Tensor,
    context: torch.Tensor,
    alpha: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """Build natural-joint-order position targets from the shared context."""
    return default_q + context[:, 1:30] + alpha * action


@dataclass(kw_only=True)
class SuperMimicCommandCfg(CommandTermCfg):
    """Hydra-owned SuperMimic motion, model, and episode parameters."""

    entity_name: str = "robot"
    motion_files: list[str]
    mjcf_path: str
    eval_mode: bool = False
    gamma: float = 0.99
    src_fps: float = 30.0
    reset_noise: bool = True
    max_episode_steps: int = 500
    goal_frames: int = 1
    context_dim: int = XI_DIM
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> SuperMimicCommand:
        return SuperMimicCommand(self, env)


class SuperMimicCommand(CommandTerm):
    """Reference motion, shared simulation context, and episode bookkeeping."""

    cfg: SuperMimicCommandCfg

    def __init__(self, cfg: SuperMimicCommandCfg, env: ManagerBasedRlEnv):
        self._validate_cfg(cfg)
        super().__init__(cfg, env)
        self.robot = cast("Entity", env.scene[cfg.entity_name])
        if tuple(self.robot.joint_names) != tuple(G1_JOINT_NAMES):
            raise ValueError("SuperMimic requires the G1 natural joint order")
        device = self._device
        self.kin = G1Kinematics(cfg.mjcf_path, device)
        self.motion = MotionLib(cfg.motion_files, self.kin, device, src_fps=cfg.src_fps)
        self.default_q = self.robot.data.default_joint_pos(device).clone()
        if self.default_q.shape != (self.num_envs, NUM_J):
            raise ValueError(
                f"SuperMimic default joint positions have shape {self.default_q.shape}; "
                f"expected {(self.num_envs, NUM_J)}"
            )
        self.mdp = MDP(self.kin, self.motion, self.default_q[0].clone())
        self.gamma = float(cfg.gamma)
        self.eval_mode = cfg.eval_mode
        self.reset_noise = cfg.reset_noise and not cfg.eval_mode
        self.max_episode_steps = 10**9 if cfg.eval_mode else cfg.max_episode_steps
        self._env_origins = env.scene.env_origins(device)
        self.f = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.xi = torch.zeros((self.num_envs, XI_DIM), dtype=torch.float32, device=device)
        self.a_prev = torch.zeros((self.num_envs, NUM_J), dtype=torch.float32, device=device)
        self.ep_step = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.start_bin = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.ep_disc_ret = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.ep_ret = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.bin_G = torch.zeros(self.motion.num_bins, dtype=torch.float32, device=device)
        self.bin_w_avg = torch.full(
            (self.motion.num_bins,), 1.0 / self.motion.num_bins, device=device
        )
        self.last_episode_stats: dict[str, torch.Tensor] = {}
        self._requested_frames: torch.Tensor | None = None
        self._read_phase_bound = False

    @staticmethod
    def _validate_cfg(cfg: SuperMimicCommandCfg) -> None:
        if not isinstance(cfg.entity_name, str) or not cfg.entity_name:
            raise ValueError("SuperMimicCommandCfg entity_name must be non-empty")
        if (
            not isinstance(cfg.motion_files, list)
            or not cfg.motion_files
            or any(not isinstance(path, str) or not path for path in cfg.motion_files)
        ):
            raise ValueError("SuperMimicCommandCfg motion_files must be a non-empty list of paths")
        if not isinstance(cfg.mjcf_path, str) or not cfg.mjcf_path:
            raise ValueError("SuperMimicCommandCfg mjcf_path must be non-empty")
        if not isinstance(cfg.eval_mode, bool) or not isinstance(cfg.reset_noise, bool):
            raise TypeError("SuperMimicCommandCfg eval_mode and reset_noise must be bool")
        if isinstance(cfg.gamma, bool) or not isinstance(cfg.gamma, (int, float)):
            raise TypeError("SuperMimicCommandCfg gamma must be a real number")
        if not math.isfinite(cfg.gamma) or not 0.0 < cfg.gamma < 1.0:
            raise ValueError("SuperMimicCommandCfg gamma must be in (0, 1)")
        if not math.isfinite(cfg.src_fps) or cfg.src_fps <= 0.0:
            raise ValueError("SuperMimicCommandCfg src_fps must be finite and positive")
        if isinstance(cfg.max_episode_steps, bool) or cfg.max_episode_steps <= 0:
            raise ValueError("SuperMimicCommandCfg max_episode_steps must be positive")
        if isinstance(cfg.goal_frames, bool) or cfg.goal_frames <= 0:
            raise ValueError("SuperMimicCommandCfg goal_frames must be positive")
        if cfg.context_dim != XI_DIM:
            raise ValueError(f"SuperMimic requires context_dim={XI_DIM}")

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("pelvis",)

    @property
    def command(self) -> torch.Tensor:
        return self.goal(self.cfg.goal_frames)

    def bind_read_phase(self) -> None:
        read_plan = self._env.scene.read_plan
        if read_plan is None:
            raise RuntimeError("SuperMimic requires the compiled scene tensor read plan")
        read_plan.refresh()
        self._read_phase_bound = True

    def state(self) -> State:
        """Read the current pelvis and joint state in the SuperMimic convention."""
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None or not read_plan.ready:
            if self._read_phase_bound:
                raise RuntimeError("SuperMimic requires a refreshed scene tensor read phase")
            return self.mdp.ref_state(self.f)
        body = read_plan.body_tensor_view(self.robot, self.tensor_body_names)
        joint = read_plan.joint_tensor_view(self.robot)
        return (
            body.pos_w[:, 0] - self._env_origins,
            quat_to_mat(body.quat_w[:, 0]),
            body.lin_vel_w[:, 0],
            body.ang_vel_w[:, 0],
            joint.joint_pos,
            joint.joint_vel,
        )

    def observation(self) -> torch.Tensor:
        return self.mdp.obs(self.state(), self.a_prev, self.xi, self.f)

    def goal(self, num_frames: int) -> torch.Tensor:
        return self.mdp.goal(self.f, num_frames)

    def request_frames(self, frames: torch.Tensor) -> None:
        """Set per-environment frames for the next full reset."""
        if not isinstance(frames, torch.Tensor):
            raise TypeError("SuperMimic reset frames must be a torch.Tensor")
        if frames.shape != (self.num_envs,):
            raise ValueError(f"SuperMimic reset frames must have shape {(self.num_envs,)}")
        if frames.dtype not in (torch.int32, torch.int64):
            raise TypeError("SuperMimic reset frames must use an integer dtype")
        frames = frames.to(device=self._device, dtype=torch.int64).clone()
        if bool((frames < 0).any()) or bool((frames >= self.motion.num_frames).any()):
            raise ValueError("SuperMimic reset frames are outside the motion library")
        self._requested_frames = frames

    def reset(
        self, env_ids: torch.Tensor | slice | None = None, *, publish_metrics: bool = True
    ) -> dict[str, float]:
        del publish_metrics
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self._device,
            owner="SuperMimicCommand.reset",
        )
        reset_mask = self._reset_mask_buffer.resolve(ids, owner="SuperMimicCommand.reset")
        if ids.numel() == 0:
            return {}
        if self._requested_frames is None:
            frames, bins = sample_start_frames(
                self.motion,
                self.bin_G,
                self.bin_w_avg,
                self.gamma,
                ids.numel(),
                self._rng,
            )
        else:
            if not bool(reset_mask.all()):
                raise ValueError("SuperMimic requested-frame reset must include every environment")
            frames = self._requested_frames
            self._requested_frames = None
            bins = self.motion.frame_bin[frames]

        full_frames = self.f.clone()
        full_frames[ids] = frames
        p, rotation, v, w, q, qd = (value.clone() for value in self.motion.ref_state(full_frames))
        if self.reset_noise:
            selected_p, selected_R, selected_v, selected_w, selected_q, _ = (
                value[ids].clone() for value in (p, rotation, v, w, q, qd)
            )
            count = ids.numel()
            selected_p += self._rng.uniform(
                torch.tensor([-0.05, -0.05, -0.01], device=self._device),
                torch.tensor([0.05, 0.05, 0.01], device=self._device),
                (count, 3),
            )
            rotation_noise = self._rng.uniform(
                torch.tensor([-0.1, -0.1, -0.2], device=self._device),
                torch.tensor([0.1, 0.1, 0.2], device=self._device),
                (count, 3),
            )
            selected_R = so3_exp(rotation_noise) @ selected_R
            selected_v += self._rng.uniform(
                torch.tensor([-0.5, -0.5, -0.2], device=self._device),
                torch.tensor([0.5, 0.5, 0.2], device=self._device),
                (count, 3),
            )
            selected_w += self._rng.uniform(
                torch.tensor([-0.52, -0.52, -0.78], device=self._device),
                torch.tensor([0.52, 0.52, 0.78], device=self._device),
                (count, 3),
            )
            selected_q += self._rng.uniform(-0.1, 0.1, (count, NUM_J))
            p[ids], rotation[ids], v[ids], w[ids], q[ids] = (
                selected_p,
                selected_R,
                selected_v,
                selected_w,
                selected_q,
            )
        q = torch.clamp(q, self.kin.q_lo, self.kin.q_hi)
        # MuJoCo free-root qvel stores angular velocity in the root/body frame,
        # while SuperMimic motion state and UniLab root-state consumers use world.
        w_local = (rotation.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
        root_state = torch.cat((p + self._env_origins, mat_to_quat(rotation), v, w_local), dim=-1)
        self.robot.write_motion_state_tensor_to_sim(
            root_state=root_state,
            position=q,
            velocity=qd,
            reset_mask=reset_mask,
        )
        self.f[ids] = frames
        self.ep_step[ids] = 0
        self.a_prev[ids] = 0.0
        self.start_bin[ids] = bins
        self.ep_disc_ret[ids] = 0.0
        self.ep_ret[ids] = 0.0
        self.time_left[ids] = self._resampling_time_range[0]
        self.command_counter[ids] += 1
        return {}

    def _update_metrics(self) -> None:
        return

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        if bool(reset_mask.any()):
            raise RuntimeError("SuperMimic commands resample only through the reset lifecycle")

    def _update_command(self) -> None:
        reward = cast("RuntimeEnv", self._env).reward_buf
        self.ep_disc_ret.add_(self.gamma ** self.ep_step.float() * reward)
        self.ep_ret.add_(reward)
        self.f.add_(1)
        self.ep_step.add_(1)
        self.a_prev.copy_(self._env.action_manager.action)
        if self.eval_mode:
            self.last_episode_stats = {}
            return
        self.last_episode_stats = update_start_bin_returns(
            self.bin_G,
            self.start_bin,
            self.ep_step,
            self.ep_ret,
            self.ep_disc_ret,
            self._env.termination_manager.terminated,
            self._env.reset_buf,
            self.gamma,
        )

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        del reset_mask

    def post_compute_reset(self) -> None:
        return


@dataclass(kw_only=True)
class SuperMimicActionCfg(ActionTermCfg):
    """Natural-joint-order SuperMimic position-action configuration."""

    entity_name: str = "robot"
    command_name: str = "motion"

    def build(self, env: ManagerBasedRlEnv) -> SuperMimicAction:
        return SuperMimicAction(self, env)


class SuperMimicAction(ActionTerm):
    """Apply default pose, episode joint offsets, and per-joint action scaling."""

    cfg: SuperMimicActionCfg

    def __init__(self, cfg: SuperMimicActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = _command(env, cfg.command_name)
        if self._command.robot is not self._entity:
            raise ValueError("SuperMimic action and command must control the same entity")
        self._raw_actions = torch.zeros(
            (self.num_envs, NUM_J), dtype=torch.float32, device=env.device
        )
        self._target = torch.zeros_like(self._raw_actions)
        self._alpha = torch.tensor(
            [action_scale(name) for name in G1_JOINT_NAMES],
            dtype=torch.float32,
            device=env.device,
        )

    @property
    def action_dim(self) -> int:
        return NUM_J

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def tensor_target(self) -> torch.Tensor:
        return self._target

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)

    def apply_actions(self) -> None:
        self._target.copy_(
            joint_action_target(
                self._command.default_q,
                self._command.xi,
                self._alpha,
                self._raw_actions,
            )
        )
        cast("Entity", self._entity).set_full_width_joint_position_target(self._target)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self._env.device,
            owner="SuperMimicAction.reset",
        )
        self._raw_actions[ids] = 0.0


def supermimic_obs(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    return _command(env, command_name).observation()


def supermimic_goal(
    env: ManagerBasedRlEnv, command_name: str = "motion", num_frames: int = 1
) -> torch.Tensor:
    return _command(env, command_name).goal(num_frames)


def supermimic_reward(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    return command.mdp.reward(
        command.state(),
        command.f + 1,
        env.action_manager.action,
        command.a_prev,
    )


def supermimic_termination(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    return command.mdp.termination(command.state(), command.f + 1)


def supermimic_timeout(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    return command.ep_step + 1 >= command.max_episode_steps


def supermimic_clip_end(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    clip_end = command.motion.clip_end[command.motion.clip_id[command.f]]
    return command.f + 1 >= clip_end


class SuperMimicContextEvent(ManagerTermBase):
    """Sample startup friction/CoM or replay those values on later resets."""

    uses_startup_rows = True

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        if cfg.mode not in ("startup", "reset"):
            raise ValueError("SuperMimicContextEvent requires startup or reset mode")
        self._startup = cfg.mode == "startup"
        self._command_name = _event_command_name(cfg)
        self._randomize_friction = cfg.params.get("randomize_friction", True)
        if not isinstance(self._randomize_friction, bool):
            raise TypeError("SuperMimicContextEvent randomize_friction must be bool")
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("SuperMimicContextEvent asset_cfg must be SceneEntityCfg")
        self._robot = cast("Entity", env.scene[asset_cfg.name])
        self._geom_ids = None
        self._friction_defaults = None
        if self._randomize_friction:
            self._geom_ids, self._friction_defaults = self._robot.bind_geom_friction_write(
                asset_cfg.geom_ids,
                term_name="supermimic_context",
            )
        torso_ids, _ = self._robot.find_bodies("torso_link", preserve_order=True)
        self._torso_ids, self._ipos_defaults = self._robot.bind_body_ipos_write(
            torso_ids,
            term_name="supermimic_context",
        )
        ground_cfg = cfg.params.get("ground_asset_cfg")
        if ground_cfg is not None and not isinstance(ground_cfg, SceneEntityCfg):
            raise TypeError(
                "SuperMimicContextEvent ground_asset_cfg must be SceneEntityCfg or None"
            )
        self._ground = (
            None
            if ground_cfg is None or not self._randomize_friction
            else cast("Entity", env.scene[ground_cfg.name])
        )
        self._ground_geom_ids = None
        self._ground_friction_defaults = None
        if self._ground is not None and ground_cfg is not None:
            self._ground_geom_ids, self._ground_friction_defaults = (
                self._ground.bind_geom_friction_write(
                    ground_cfg.geom_ids,
                    term_name="supermimic_ground_context",
                )
            )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        command_name: str = "motion",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        ground_asset_cfg: SceneEntityCfg | None = None,
        randomize_friction: bool = True,
    ) -> None:
        del command_name, asset_cfg, ground_asset_cfg, randomize_friction
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=env.num_envs,
            device=env.device,
            owner="SuperMimicContextEvent",
        )
        command = _command(env, self._command_name)
        if self._startup:
            expected = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
            if not torch.equal(ids, expected):
                raise ValueError("SuperMimic startup context requires every environment row")
            if command.eval_mode or not self._randomize_friction:
                command.xi[:, 0] = 1.0
            else:
                command.xi[:, 0] = env.rng.uniform(0.3, 1.2, (env.num_envs,))
            if command.eval_mode:
                command.xi[:, 30:33] = 0.0
            else:
                command.xi[:, 30:33] = env.rng.uniform(
                    torch.tensor([-0.025, -0.05, -0.05], device=env.device),
                    torch.tensor([0.025, 0.05, 0.05], device=env.device),
                    (env.num_envs, 3),
                )
        if self._randomize_friction:
            assert self._geom_ids is not None and self._friction_defaults is not None
            friction = _selected_defaults(self._friction_defaults, ids)
            friction[..., 0] = command.xi[ids, :1]
            self._robot.write_geom_friction_to_sim(
                friction,
                geom_ids=self._geom_ids,
                env_ids=ids,
                term_name="supermimic_context",
            )
        ipos = _selected_defaults(self._ipos_defaults, ids)
        ipos += command.xi[ids, 30:33].unsqueeze(1)
        if self._startup:
            self._robot.record_startup_body_ipos_to_sim(
                ipos,
                body_ids=self._torso_ids,
                env_ids=ids,
                term_name="supermimic_context",
            )
        else:
            self._robot.write_body_ipos_to_sim(
                ipos,
                body_ids=self._torso_ids,
                env_ids=ids,
                term_name="supermimic_context",
            )
        if self._ground is not None:
            assert self._ground_friction_defaults is not None
            ground_friction = _selected_defaults(self._ground_friction_defaults, ids)
            ground_friction[..., 0] = 1.0 if command.eval_mode else 0.3
            self._ground.write_geom_friction_to_sim(
                ground_friction,
                geom_ids=self._ground_geom_ids,
                env_ids=ids,
                term_name="supermimic_ground_context",
            )


class SuperMimicJointOffsetEvent(ManagerTermBase):
    """Sample reset joint offsets into the command's only simulation context."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        if cfg.mode != "reset":
            raise ValueError("SuperMimicJointOffsetEvent requires reset mode")
        self._command_name = _event_command_name(cfg)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        command_name: str = "motion",
    ) -> None:
        del command_name
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=env.num_envs,
            device=env.device,
            owner="SuperMimicJointOffsetEvent",
        )
        command = _command(env, self._command_name)
        if command.eval_mode:
            command.xi[ids, 1:30] = 0.0
        else:
            command.xi[ids, 1:30] = env.rng.uniform(-0.01, 0.01, (ids.numel(), NUM_J))


def _event_command_name(cfg: EventTermCfg) -> str:
    command_name = cfg.params.get("command_name", "motion")
    if not isinstance(command_name, str) or not command_name:
        raise ValueError("SuperMimic event command_name must be non-empty")
    return command_name


def _selected_defaults(defaults: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    values = defaults.to(device=env_ids.device, dtype=torch.float32)
    if values.ndim == 2:
        return values.expand((env_ids.numel(), *values.shape)).clone()
    if values.ndim == 3:
        return values.index_select(0, env_ids).clone()
    raise ValueError(f"SuperMimic event defaults have invalid shape {values.shape}")


def _command(env: ManagerBasedRlEnv, command_name: str) -> SuperMimicCommand:
    try:
        command = env.command_manager.get_term(command_name)
    except KeyError as exc:
        raise KeyError(f"SuperMimic command term '{command_name}' not found") from exc
    if not isinstance(command, SuperMimicCommand):
        raise TypeError(
            f"Command term '{command_name}' is {type(command).__name__}, expected SuperMimicCommand"
        )
    return command


__all__ = [
    "SuperMimicAction",
    "SuperMimicActionCfg",
    "SuperMimicCommand",
    "SuperMimicCommandCfg",
    "SuperMimicContextEvent",
    "SuperMimicJointOffsetEvent",
    "joint_action_target",
    "make_current_info",
    "make_transition_info",
    "merge_current_info",
    "sample_start_frames",
    "start_bin_probabilities",
    "supermimic_clip_end",
    "supermimic_goal",
    "supermimic_obs",
    "supermimic_reward",
    "supermimic_termination",
    "supermimic_timeout",
    "update_start_bin_returns",
]
