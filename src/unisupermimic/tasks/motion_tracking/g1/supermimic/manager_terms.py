"""Manager-Based terms for the G1 SuperMimic mjwarp task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import torch
from unilab.base.reset_selection import ResetSelection, fill_rows_, is_reset_selection
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
from unisim.dr.types import RESET_TERM_GEOM_FRICTION

from unisupermimic.rl.algos.supermimic.geometry import (
    flatten_state,
    mat_to_quat,
    quat_to_mat,
    so3_exp,
)
from unisupermimic.rl.contracts.supermimic import State

from .kinematics import G1Kinematics
from .mdp import MAX_STEP_REWARD, MDP, NUM_J, XI_DIM, RewardCfg, step_path_compile_backend
from .motion import BIN_FRAMES, MotionLib
from .robot import G1_JOINT_NAMES, action_scale

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv as RuntimeEnv
    from unilab.managers.types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# exams-v2 frontier practice: practice resets near a failed clip's failure
# frame start inside the window [failure frame - FRONTIER_FRAMES, clip end),
# matching BlinkMimic2's ``--front_win`` default.
FRONTIER_FRAMES = 50


def start_bin_probabilities(
    bin_G: torch.Tensor,
    bin_w_avg: torch.Tensor,
    gamma: float,
    update: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update the past-performance EMA and return the exact bin mixture.

    ``update`` is a device-side gate for the EMA step: a zero scalar skips the
    mixture update without any host read of the reset mask, so the fixed-shape
    reset (which samples every step) keeps per-reset EMA semantics.
    """
    soft = torch.softmax(-25.0 * bin_G / (MAX_STEP_REWARD / (1.0 - gamma)), dim=0)
    if update is None:
        delta = torch.full((), 0.001, device=bin_G.device)
    else:
        delta = 0.001 * update.reshape(())
    bin_w_avg.mul_(1.0 - delta).add_(delta * soft)
    return 0.5 / bin_G.numel() + 0.4 * soft + 0.1 * bin_w_avg / bin_w_avg.sum()


def sample_start_frames(
    motion: MotionLib,
    bin_G: torch.Tensor,
    bin_w_avg: torch.Tensor,
    gamma: float,
    num_samples: int,
    rng: ManagerRng,
    update: torch.Tensor | None = None,
    cdf_sampling: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one-second bins and uniformly sample a frame inside each bin.

    ``cdf_sampling`` draws the categorical bin choice through inverse-CDF
    ``searchsorted`` instead of ``rng.choice``: the choice wrapper validates
    its probability table with host reads, which the fixed-shape reset
    (sampling every step) must avoid.
    """
    probabilities = start_bin_probabilities(bin_G, bin_w_avg, gamma, update=update)
    if cdf_sampling:
        cdf = torch.cumsum(probabilities, dim=0)
        bins = torch.searchsorted(cdf, rng.random((num_samples,)) * cdf[-1])
        bins = bins.clamp(max=motion.num_bins - 1)
    else:
        bins = rng.choice(motion.num_bins, (num_samples,), p=probabilities)
    start = motion.bin_start[bins]
    end = torch.minimum(start + BIN_FRAMES, motion.clip_end[motion.bin_clip[bins]])
    frames = start + (rng.random((num_samples,)) * (end - start).float()).long()
    return frames, bins


def sample_frontier_frames(
    clip_fail_frame: torch.Tensor,
    clip_start: torch.Tensor,
    clip_end: torch.Tensor,
    window: int,
    u_clip: torch.Tensor,
    u_position: torch.Tensor,
) -> torch.Tensor:
    """Sample frontier practice starts: equal mass per failed clip.

    Each draw picks a failed clip uniformly (inverse-CDF ``searchsorted``
    over the failed-clip indicator — ``right=True`` so unfailed clips carry
    no mass), then a frame uniform in ``[failure frame - window, clip end)``
    clipped into the clip. With no failed clip anywhere the draws return
    clipped clip-0 frames; callers gate those rows off through the
    failed-clip availability flag so only the (fixed-count) RNG draws leak.
    Pure elementwise/searchsorted device work: safe inside the captured
    reset segment.
    """
    failed = (clip_fail_frame >= 0).to(torch.float32)
    cdf = torch.cumsum(failed, dim=0)
    clips = torch.searchsorted(cdf, u_clip * cdf[-1], right=True).clamp(max=failed.numel() - 1)
    fail = torch.minimum(torch.maximum(clip_fail_frame[clips], clip_start[clips]), clip_end[clips])
    low = torch.minimum(torch.maximum(fail - window, clip_start[clips]), clip_end[clips] - 1)
    span = (clip_end[clips] - low).float()
    return low + (u_position * span).long()


def update_clip_fail_stats(
    clip_fail_ema: torch.Tensor,
    clip_fail_frame: torch.Tensor,
    clip_id: torch.Tensor,
    frames: torch.Tensor,
    term: torch.Tensor,
    done: torch.Tensor,
    decay: float,
) -> None:
    """Update per-clip EMA failure rate and last failure frame in place.

    Sync-free like :func:`update_start_bin_returns`: episodes ending this
    step (practice and exam rows alike) scatter their hard-termination
    verdict into per-clip means for one EMA step (``decay`` keeps the old
    value), and failed endings scatter their frame into the last-failure-
    frame table (``amax`` reduction: deterministic under index clashes and
    never lowered by non-failing rows writing ``-1``).
    """
    ended = done.to(clip_fail_ema.dtype)
    failed = term & done
    sums = torch.zeros_like(clip_fail_ema).scatter_add_(0, clip_id, failed.to(ended.dtype))
    counts = torch.zeros_like(clip_fail_ema).scatter_add_(0, clip_id, ended)
    touched = counts > 0
    mean_fail = sums / counts.clamp(min=1.0)
    clip_fail_ema.copy_(
        torch.where(touched, decay * clip_fail_ema + (1.0 - decay) * mean_fail, clip_fail_ema)
    )
    clip_fail_frame.scatter_reduce_(
        0, clip_id, torch.where(failed, frames, -1), reduce="amax", include_self=True
    )


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
    """Update completed-episode returns; return full-width masked statistics.

    Sync-free: every reduction is masked and full-width, so the per-step
    command update never reads the device on the host. Non-done rows carry
    zero in the returned statistics; consumers aggregate with the done mask.
    """
    lengths = ep_step.float().clamp(min=1.0)
    bootstrap = torch.where(
        hard,
        0.0,
        gamma**lengths * ep_ret / lengths / (1.0 - gamma),
    )
    returns = (ep_disc_ret + bootstrap) * done
    # ``index_put_`` without accumulate is order-dependent when several episodes
    # end in the same bin (last write wins), so average each bin's returns first
    # and apply a single EMA update per bin.
    num_bins = bin_G.numel()
    summed_returns = torch.zeros(num_bins, dtype=returns.dtype, device=returns.device)
    summed_returns.scatter_add_(0, start_bin, returns)
    counts = torch.zeros(num_bins, dtype=returns.dtype, device=returns.device)
    counts.scatter_add_(0, start_bin, done.float())
    touched = counts > 0
    mean_returns = summed_returns / counts.clamp(min=1.0)
    bin_G.copy_(torch.where(touched, 0.8 * bin_G + 0.2 * mean_returns, bin_G))
    return {
        "ep_len": ep_step.float() * done,
        "ep_ret_norm": ep_ret / (lengths * MAX_STEP_REWARD) * done,
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
    # torch.compile the hot step-path MDP functions (obs/goal/reward/
    # termination/state/transition materialization). The collection step is
    # launch-bound, so fusing these elementwise chains removes many tiny
    # kernels per step.
    compile: bool = False
    gamma: float = 0.99
    src_fps: float = 30.0
    reset_noise: bool = True
    max_episode_steps: int = 500
    goal_frames: int = 1
    context_dim: int = XI_DIM
    w_amp: float = 0.0
    amp_tol: float = 0.02
    amp_scale: float = 0.5
    amp_min_height: float = 0.8
    amp_ankles: tuple[int, int] = (3, 6)
    w_speed: float = 0.0
    speed_min: float = 2.0
    speed_tol: float = 0.3
    speed_scale: float = 1.5
    speed_bodies: tuple[int, int, int, int] = (3, 6, 9, 11)
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
        self.mdp = MDP(
            self.kin,
            self.motion,
            self.default_q[0].clone(),
            RewardCfg(
                w_amp=cfg.w_amp,
                amp_tol=cfg.amp_tol,
                amp_scale=cfg.amp_scale,
                amp_min_height=cfg.amp_min_height,
                amp_ankles=cfg.amp_ankles,
                w_speed=cfg.w_speed,
                speed_min=cfg.speed_min,
                speed_tol=cfg.speed_tol,
                speed_scale=cfg.speed_scale,
                speed_bodies=cfg.speed_bodies,
            ),
            goal_frames=cfg.goal_frames,
        )
        self.gamma = float(cfg.gamma)
        self.eval_mode = cfg.eval_mode
        self.reset_noise = cfg.reset_noise and not cfg.eval_mode
        self.max_episode_steps = 10**9 if cfg.eval_mode else cfg.max_episode_steps
        self._env_origins = env.scene.env_origins(device)
        self.f = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.xi = torch.zeros((self.num_envs, XI_DIM), dtype=torch.float32, device=device)
        self.a_prev = torch.zeros((self.num_envs, NUM_J), dtype=torch.float32, device=device)
        # Fixed-address carrier holding the state each row was last reset to,
        # in the exact State layout command.state() returns. Under the
        # reset-segment CUDA graph the scene read plan does not refresh during
        # replay, so reading physics right after the masked write would return
        # the pre-reset state on done rows; the written reference is the
        # authoritative post-reset state and is staged here instead.
        self._reset_state_carrier = (
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device),
            torch.zeros((self.num_envs, 3, 3), dtype=torch.float32, device=device),
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device),
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device),
            torch.zeros((self.num_envs, NUM_J), dtype=torch.float32, device=device),
            torch.zeros((self.num_envs, NUM_J), dtype=torch.float32, device=device),
        )
        # Fixed-address carrier mirroring the live (pre-reset) state for the
        # captured reset segment. The read plan reallocates its derived body
        # packet tensors on every refresh, so a captured command.state() reads
        # whichever allocation now owns the capture-time address; the reset
        # publication must never read the packet, only these stable tensors.
        self._live_state_carrier = (
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device),
            torch.zeros((self.num_envs, 3, 3), dtype=torch.float32, device=device),
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device),
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device),
            torch.zeros((self.num_envs, NUM_J), dtype=torch.float32, device=device),
            torch.zeros((self.num_envs, NUM_J), dtype=torch.float32, device=device),
        )
        self.ep_step = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.start_bin = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.ep_disc_ret = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.ep_ret = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        # BlinkMimic2-style exam worlds (learner-activated, see
        # ``activate_exam_worlds``): rows running full-clip exam episodes. The
        # mask and the round-robin clip cursor live at fixed addresses from
        # construction and only their values change at activation, so the
        # captured reset segment keeps reading them after every replay. With
        # no exams active the mask is all false and every exam-dependent op is
        # a value no-op.
        self.exam_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._num_clips = int(self.motion.clip_start.numel())
        self._exam_clip_cursor = torch.arange(
            self.num_envs, dtype=torch.int64, device=device
        ) % max(self._num_clips, 1)
        # exams-v2 margin mode (learner-configured through
        # ``configure_exam_margin``; default off keeps the round-robin exam
        # behavior above bit-identical). The failure statistics and the exam
        # clip table live at fixed addresses from construction so a captured
        # reset segment keeps reading them after every replay; the switches
        # are Python-level and must be set once, eagerly, before any
        # reset-segment capture so the captured branch layout never changes.
        self._exam_margin = False
        self._frontier_frac = 0.0
        self._fail_ema = 0.7
        self._clip_fail_ema = torch.zeros(self._num_clips, dtype=torch.float32, device=device)
        self._clip_fail_frame = torch.full((self._num_clips,), -1, dtype=torch.int64, device=device)
        self._exam_clip_table = torch.arange(self.num_envs, dtype=torch.int64, device=device) % max(
            self._num_clips, 1
        )
        self.bin_G = torch.zeros(self.motion.num_bins, dtype=torch.float32, device=device)
        self.bin_w_avg = torch.full(
            (self.motion.num_bins,), 1.0 / self.motion.num_bins, device=device
        )
        self.last_episode_stats: dict[str, torch.Tensor] = {}
        self._requested_frames: torch.Tensor | None = None
        self._requested_env_ids: torch.Tensor | None = None
        self._read_phase_bound = False
        # Reset-noise bound tensors, built once: constructing them from Python
        # lists inside reset() would stage a host-to-device copy per call.
        self._noise_bounds = tuple(
            (
                torch.tensor(lower, dtype=torch.float32, device=device),
                torch.tensor(upper, dtype=torch.float32, device=device),
            )
            for lower, upper in (
                ((-0.05, -0.05, -0.01), (0.05, 0.05, 0.01)),
                ((-0.1, -0.1, -0.2), (0.1, 0.1, 0.2)),
                ((-0.5, -0.5, -0.2), (0.5, 0.5, 0.2)),
                ((-0.52, -0.52, -0.78), (0.52, 0.52, 0.78)),
            )
        )
        if cfg.compile:
            self._compile_step_path()

    def _compile_step_path(self) -> None:
        """Compile the per-step pure-tensor functions on this term and MDP.

        These run inside the Manager compute/read phases (outside the
        reset-segment CUDA graph body), where fusion removes the most tiny
        eager kernels per step. Only step-path methods are compiled: the
        reset path stays eager so the reset-segment capture never meets an
        inductor recompile.
        """
        self.mdp.compile_step_path()
        self.state = torch.compile(
            self.state, backend=step_path_compile_backend(torch.device(self._device))
        )

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
        if not isinstance(cfg.compile, bool):
            raise TypeError("SuperMimicCommandCfg compile must be bool")
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

    @property
    def reset_state_carrier(self) -> tuple[torch.Tensor, ...]:
        """Fixed-address per-row state each reset last wrote, in State layout."""
        return self._reset_state_carrier

    def refresh_live_state(self) -> None:
        """Mirror the live (pre-reset) state into the fixed-address carrier.

        Runs outside the captured reset segment every step, where the read
        plan packet is fresh; the captured segment then publishes through the
        carrier and never reads the (reallocated) packet.
        """
        for carrier, value in zip(self._live_state_carrier, self.state(), strict=True):
            carrier.copy_(value)

    def observation(self) -> torch.Tensor:
        return self.mdp.obs(self.state(), self.a_prev, self.xi, self.f)

    def goal(self, num_frames: int) -> torch.Tensor:
        return self.mdp.goal(self.state(), self.f, num_frames)

    def request_frames(self, frames: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Set frames for the next full or partial reset."""
        if not isinstance(frames, torch.Tensor):
            raise TypeError("SuperMimic reset frames must be a torch.Tensor")
        expected_frames = self.num_envs if env_ids is None else env_ids.numel()
        if frames.shape != (expected_frames,):
            raise ValueError(
                f"SuperMimic reset frames must have shape ({expected_frames},), "
                f"got {tuple(frames.shape)}"
            )
        if frames.dtype not in (torch.int32, torch.int64):
            raise TypeError("SuperMimic reset frames must use an integer dtype")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self._device)
        else:
            env_ids = resolve_reset_env_ids(
                env_ids,
                num_envs=self.num_envs,
                device=self._device,
                owner="SuperMimicCommand.request_frames",
            )
        requested = frames.to(device=self._device, dtype=torch.int64).clone()
        if bool((requested < 0).any()) or bool((requested >= self.motion.num_frames).any()):
            raise ValueError("SuperMimic reset frames are outside the motion library")
        self._requested_frames = requested
        self._requested_env_ids = env_ids

    supports_reset_selection: ClassVar[bool] = True

    def configure_exam_margin(self, *, frontier_frac: float, fail_ema: float) -> None:
        """Enable exams-v2 margin mode: failure statistics + frontier practice.

        Eager, one-time, and required to run before any reset-segment capture:
        the captured segment bakes these Python-level switches, and the
        per-clip statistics and exam clip table already live at fixed
        addresses from construction. ``frontier_frac`` 0 keeps the bin-EMA
        start sampling and only margin-selects the exam clips.
        """
        if isinstance(frontier_frac, bool) or not isinstance(frontier_frac, (int, float)):
            raise TypeError(
                f"SuperMimic frontier_frac must be a real number, got {frontier_frac!r}"
            )
        if not 0.0 <= float(frontier_frac) <= 1.0:
            raise ValueError(f"SuperMimic frontier_frac must be in [0, 1], got {frontier_frac}")
        if isinstance(fail_ema, bool) or not isinstance(fail_ema, (int, float)):
            raise TypeError(f"SuperMimic fail_ema must be a real number, got {fail_ema!r}")
        if not 0.0 <= float(fail_ema) < 1.0:
            raise ValueError(f"SuperMimic fail_ema must be in [0, 1), got {fail_ema}")
        self._frontier_frac = float(frontier_frac)
        self._fail_ema = float(fail_ema)
        self._exam_margin = True

    def _write_exam_clip_table(self, num_rows: int) -> torch.Tensor:
        """Select exam clips by margin priority into the fixed-address table.

        Simplified BlinkMimic2 margin: the per-clip EMA failure rate closest
        to 1 (failure danger boundary) is examined first. A deterministic
        device-side ``topk`` (ties broken by clip id) hands the ``num_rows``
        exam rows distinct clips — at most one exam instance per clip — and
        writes them into the last ``num_rows`` entries of the persistent
        table; the captured reset segment gathers clip first frames from it
        on every replay. Returns the selected clips' mean failure EMA (a
        device scalar for the caller to log).
        """
        if num_rows > self._num_clips:
            raise ValueError(
                f"SuperMimic exam row count {num_rows} exceeds the clip count {self._num_clips}"
            )
        margin = (self._clip_fail_ema - 1.0).abs()
        clips = torch.topk(margin, k=num_rows, largest=False, sorted=True).indices
        self._exam_clip_table[self.num_envs - num_rows :].copy_(clips)
        return self._clip_fail_ema[clips].mean()

    def refresh_exam_clips(self, num_rows: int) -> torch.Tensor:
        """Re-select the exam clips by margin priority (graph-out, eager).

        Runs at iteration boundaries outside any captured segment; only the
        values of the fixed-address table change, so later replays pick the
        new clips up without recapture.
        """
        if isinstance(num_rows, bool) or not isinstance(num_rows, int):
            raise TypeError(f"SuperMimic exam row count must be an int, got {num_rows!r}")
        if not 0 < num_rows < self.num_envs:
            raise ValueError(
                f"SuperMimic exam row count must be in (0, {self.num_envs}), got {num_rows}"
            )
        return self._write_exam_clip_table(num_rows)

    def _frontier_override(
        self, frames: torch.Tensor, bins: torch.Tensor, exam_selected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge BlinkMimic2 frontier starts into sampled practice starts.

        RNG layout stays deterministic: every call consumes exactly three
        further ``(n,)`` draws in a fixed order — the per-row Bernoulli mask
        (fixed ratio ``frontier_frac`` on every step, never a per-step mode
        switch), the failed-clip choice, and the window position — whether
        or not any row ends up taking a frontier start. Exam rows and the
        no-failed-clip case are gated off device-side (value no-ops), so the
        captured reset segment's RNG consumption never depends on data.
        """
        count = frames.shape[0]
        draw_mask = self._rng.random((count,))
        draw_clip = self._rng.random((count,))
        draw_position = self._rng.random((count,))
        frontier_frames = sample_frontier_frames(
            self._clip_fail_frame,
            self.motion.clip_start,
            self.motion.clip_end,
            FRONTIER_FRAMES,
            draw_clip,
            draw_position,
        )
        available = (self._clip_fail_frame >= 0).any()
        use_frontier = (draw_mask < self._frontier_frac) & ~exam_selected & available
        return (
            torch.where(use_frontier, frontier_frames, frames),
            torch.where(use_frontier, self.motion.frame_bin[frontier_frames], bins),
        )

    def activate_exam_worlds(self, num_rows: int) -> torch.Tensor:
        """Mark the last ``num_rows`` rows as exam worlds; return their frames.

        Exam worlds run BlinkMimic2-style exam episodes: every episode starts
        on a clip's first frame and runs to the clip end (hard termination and
        the episode cap do not apply), the learner collects them with the EMA
        policy, and their episodes stay outside the start-bin statistics. The
        returned frames are the first frames of the clips the newly activated
        rows run: margin mode selects them by failure-EMA priority (writing
        the fixed-address exam clip table), otherwise the round-robin cursor
        hands them out. The caller resets them explicitly. Only tensor values
        are written (the mask keeps its address), so a captured reset segment
        replays with the updated mask.
        """
        if isinstance(num_rows, bool) or not isinstance(num_rows, int):
            raise TypeError(f"SuperMimic exam row count must be an int, got {num_rows!r}")
        if not 0 < num_rows < self.num_envs:
            raise ValueError(
                f"SuperMimic exam row count must be in (0, {self.num_envs}), got {num_rows}"
            )
        rows = torch.arange(self.num_envs, device=self._device) >= self.num_envs - num_rows
        if self._exam_margin:
            self._write_exam_clip_table(num_rows)
            self.exam_mask.copy_(rows)
            return self.motion.clip_start[self._exam_clip_table].clone()
        self._exam_clip_cursor.copy_(
            torch.where(
                rows, (self._exam_clip_cursor + 1) % self._num_clips, self._exam_clip_cursor
            )
        )
        self.exam_mask.copy_(rows)
        return self.motion.clip_start[self._exam_clip_cursor].clone()

    def _exam_start_frames(self, reset_mask: torch.Tensor) -> torch.Tensor:
        """Per-row clip first frames for exam worlds (fixed-shape, graph-safe).

        Margin mode gathers the fixed-address exam clip table (refreshed
        eagerly at iteration boundaries): restarting exam rows relaunch the
        table's clip, so at most one exam instance per selected clip runs.
        The default round-robin path advances the clip cursor only on rows
        that are both resetting and exam, then gathers every row's cursor
        clip start frame; rows that are not resetting exam worlds produce
        values the callers discard. Both paths keep the shared training RNG
        stream untouched, so exam activation never perturbs the captured
        reset segment's RNG layout.
        """
        if self._exam_margin:
            return self.motion.clip_start[self._exam_clip_table]
        advance = reset_mask & self.exam_mask
        cursor = (self._exam_clip_cursor + 1) % self._num_clips
        self._exam_clip_cursor.copy_(torch.where(advance, cursor, self._exam_clip_cursor))
        return self.motion.clip_start[self._exam_clip_cursor]

    def reset(
        self,
        env_ids: torch.Tensor | slice | ResetSelection | None = None,
        *,
        publish_metrics: bool = True,
    ) -> dict[str, float]:
        del publish_metrics
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self._device,
            owner="SuperMimicCommand.reset",
        )
        fixed_shape = is_reset_selection(ids)
        reset_mask = self._reset_mask_buffer.resolve(ids, owner="SuperMimicCommand.reset")
        if isinstance(ids, torch.Tensor) and ids.numel() == 0:
            return {}
        if self._requested_frames is None:
            if fixed_shape:
                # Fixed-shape dispatch: sample full-width every step (an
                # all-false mask is a no-op through the masked writes) and gate
                # the past-performance EMA device-side, so the mixture keeps
                # per-reset semantics without any host read of the mask.
                frames, bins = sample_start_frames(
                    self.motion,
                    self.bin_G,
                    self.bin_w_avg,
                    self.gamma,
                    self.num_envs,
                    self._rng,
                    update=reset_mask.any(),
                    cdf_sampling=True,
                )
                # Frontier practice: a fixed-ratio Bernoulli share of the
                # practice rows restarts in a failed clip's frontier window
                # instead of its sampled bin (value-only merge; a no-op while
                # no clip has failed). Skipped entirely when margin mode is
                # off, keeping the default RNG stream bit-identical.
                if self._exam_margin and self._frontier_frac > 0.0:
                    frames, bins = self._frontier_override(frames, bins, self.exam_mask)
                # Exam worlds restart on a clip's first frame instead of the
                # sampled start bin. Value-only override on fixed-address
                # tensors: with no exam rows the mask is all false and the
                # sampled values pass through unchanged.
                exam_frames = self._exam_start_frames(reset_mask)
                frames = torch.where(self.exam_mask, exam_frames, frames)
                bins = torch.where(self.exam_mask, self.motion.frame_bin[exam_frames], bins)
                full_frames = frames
            else:
                frames, bins = sample_start_frames(
                    self.motion,
                    self.bin_G,
                    self.bin_w_avg,
                    self.gamma,
                    ids.numel(),
                    self._rng,
                )
                if self._exam_margin and self._frontier_frac > 0.0:
                    frames, bins = self._frontier_override(frames, bins, self.exam_mask[ids])
                exam_frames = self._exam_start_frames(reset_mask)
                selected = self.exam_mask[ids]
                frames = torch.where(selected, exam_frames[ids], frames)
                bins = torch.where(selected, self.motion.frame_bin[exam_frames[ids]], bins)
                full_frames = self.f.clone()
                full_frames[ids] = frames
        else:
            if fixed_shape:
                raise RuntimeError("SuperMimic requested-frame resets require explicit env_ids")
            assert self._requested_env_ids is not None
            if not torch.equal(ids, self._requested_env_ids):
                raise ValueError(
                    "SuperMimic requested-frame reset environment IDs do not match the request"
                )
            frames = self._requested_frames
            self._requested_frames = None
            self._requested_env_ids = None
            bins = self.motion.frame_bin[frames]
            full_frames = self.f.clone()
            full_frames[ids] = frames

        p, rotation, v, w, q, qd = (value.clone() for value in self.motion.ref_state(full_frames))
        if self.reset_noise:
            if fixed_shape:
                # Full-width noise: unselected rows are discarded by the
                # masked sim write and the masked memory updates below.
                bounds = self._noise_bounds
                p += self._rng.uniform(*bounds[0], (self.num_envs, 3))
                rotation = so3_exp(self._rng.uniform(*bounds[1], (self.num_envs, 3))) @ rotation
                v += self._rng.uniform(*bounds[2], (self.num_envs, 3))
                w += self._rng.uniform(*bounds[3], (self.num_envs, 3))
                q += self._rng.uniform(-0.1, 0.1, (self.num_envs, NUM_J))
            else:
                selected_p, selected_R, selected_v, selected_w, selected_q, _ = (
                    value[ids].clone() for value in (p, rotation, v, w, q, qd)
                )
                count = ids.numel()
                bounds = self._noise_bounds
                selected_p += self._rng.uniform(*bounds[0], (count, 3))
                rotation_noise = self._rng.uniform(*bounds[1], (count, 3))
                selected_R = so3_exp(rotation_noise) @ selected_R
                selected_v += self._rng.uniform(*bounds[2], (count, 3))
                selected_w += self._rng.uniform(*bounds[3], (count, 3))
                selected_q += self._rng.uniform(-0.1, 0.1, (count, NUM_J))
                p[ids], rotation[ids], v[ids], w[ids], q[ids] = (
                    selected_p,
                    selected_R,
                    selected_v,
                    selected_w,
                    selected_q,
                )
        q = torch.clamp(q, self.kin.q_lo, self.kin.q_hi)
        # Stage the exact written state into the fixed-address carrier (in the
        # command.state() layout: world position without env origins). The
        # reset-segment graph replays this alongside the sim write, so the
        # post-reset state publication never depends on a read-plan refresh.
        carrier = self._reset_state_carrier
        carrier[0].copy_(torch.where(reset_mask[:, None], p, carrier[0]))
        carrier[1].copy_(torch.where(reset_mask[:, None, None], rotation, carrier[1]))
        carrier[2].copy_(torch.where(reset_mask[:, None], v, carrier[2]))
        carrier[3].copy_(torch.where(reset_mask[:, None], w, carrier[3]))
        carrier[4].copy_(torch.where(reset_mask[:, None], q, carrier[4]))
        carrier[5].copy_(torch.where(reset_mask[:, None], qd, carrier[5]))
        root_state = torch.cat((p + self._env_origins, mat_to_quat(rotation), v, w), dim=-1)
        self.robot.write_motion_state_tensor_to_sim(
            root_state=root_state,
            position=q,
            velocity=qd,
            reset_mask=reset_mask,
        )
        if fixed_shape:
            self.f.copy_(torch.where(reset_mask, full_frames, self.f))
            self.start_bin.copy_(torch.where(reset_mask, bins, self.start_bin))
            self.command_counter += reset_mask.to(self.command_counter.dtype)
        else:
            self.f[ids] = frames
            self.start_bin[ids] = bins
            self.command_counter[ids] += 1
        fill_rows_(self.ep_step, ids, 0)
        fill_rows_(self.a_prev, ids, 0.0)
        fill_rows_(self.ep_disc_ret, ids, 0.0)
        fill_rows_(self.ep_ret, ids, 0.0)
        fill_rows_(self.time_left, ids, self._resampling_time_range[0])
        return {}

    def _update_metrics(self) -> None:
        return

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        # Fail closed without a host read: the verdict is latched on the device
        # and surfaces at the finite-validation poll boundary.
        self._env.require(
            ~reset_mask.any(),
            "supermimic.command_resample",
            "SuperMimic commands resample only through the reset lifecycle",
        )

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
        # Exam episodes stay outside the start-bin statistics (BlinkMimic2
        # keeps them out of ``bin_G``): their clip-end resets are masked out of
        # the done feed, and with no exam rows the mask term is a value no-op.
        done = self._env.reset_buf & ~self.exam_mask
        self.last_episode_stats = update_start_bin_returns(
            self.bin_G,
            self.start_bin,
            self.ep_step,
            self.ep_ret,
            self.ep_disc_ret,
            self._env.termination_manager.terminated,
            done,
            self.gamma,
        )
        # exams-v2 margin statistics: per-clip EMA failure rate and last
        # failure frame from every episode end — practice and exam rows alike
        # (an exam clip ends either failed at its hard termination or passed
        # at its clip end). Runs after ``f`` advanced to the ended episode's
        # frame (always within the clip, at its end at most), so the clip
        # lookup stays inside the library. Sync-free and skipped entirely
        # while margin mode is off.
        if self._exam_margin:
            update_clip_fail_stats(
                self._clip_fail_ema,
                self._clip_fail_frame,
                self.motion.clip_id[self.f],
                self.f,
                self._env.termination_manager.terminated,
                self._env.reset_buf,
                self._fail_ema,
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

    supports_reset_selection: ClassVar[bool] = True

    def reset(self, env_ids: torch.Tensor | ResetSelection | None = None) -> None:
        fill_rows_(self._raw_actions, env_ids, 0.0)


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
    # Exam worlds keep the hard termination (a fallen exam robot ends that
    # clip's rollout, matching BlinkMimic2); only the wall-clock timeout is
    # lifted for them so long clips run to their end.
    return command.mdp.termination(command.state(), command.f + 1)


def supermimic_timeout(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    # The episode cap does not apply to exam worlds (their length is the clip).
    return (command.ep_step + 1 >= command.max_episode_steps) & ~command.exam_mask


def supermimic_clip_end(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    clip_end = command.motion.clip_end[command.motion.clip_id[command.f]]
    return command.f + 1 >= clip_end


class SuperMimicContextEvent(ManagerTermBase):
    """Sample the startup friction/CoM; later resets replay only the friction.

    The torso CoM offset is a per-world constant. The startup term records it, which also
    promotes it into the reset baseline: a reset that does not stage ``body_ipos`` leaves it
    in place. So no reset writes it again. Writing it again would store the same values and
    only make the backend recompute the derived inertia constants with ``set_const`` for
    *every* world (one Cholesky solve per DOF and per body): on every step of a training
    run with masked autoreset, and on every evaluation step that recycles clips, which cost
    more than the physics step itself.
    """

    uses_startup_rows = True

    # On row-indexed resets the friction is written back explicitly. Under a
    # ResetSelection the row-index write facades cannot be used (they need host
    # row indices), so the fixed-shape path stages full-width final tables
    # through the packed transaction's public mask-based writer instead. Unilab
    # 1.5.11 exposes that writer only on the transaction object, so backend
    # column IDs are captured here once.
    supports_reset_selection: ClassVar[bool] = True

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
        self._ground_backend_geom_ids: np.ndarray | None = None
        # Backend column IDs for the fixed-shape (mask-based) staging path,
        # captured lazily: the public bind facades return only entity-local
        # ids (unilab 1.5.11), and the index path never needs them. The DR
        # default tables are moved to the env device here so the hot path
        # never stages a host-to-device copy.
        self._backend_geom_ids: np.ndarray | None = None
        self._backend_geom_ids_ready = False
        self._device_friction_defaults = (
            None
            if self._friction_defaults is None
            else self._friction_defaults.to(device=env.device, dtype=torch.float32)
        )
        self._device_ground_friction_defaults = (
            None
            if self._ground_friction_defaults is None
            else self._ground_friction_defaults.to(device=env.device, dtype=torch.float32)
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
        if is_reset_selection(env_ids):
            self._replay_fixed_shape(env, env_ids)
            return
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
        if self._startup:
            # The only write of the CoM offset (see the class docstring).
            ipos = _selected_defaults(self._ipos_defaults, ids)
            ipos += command.xi[ids, 30:33].unsqueeze(1)
            self._robot.record_startup_body_ipos_to_sim(
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

    def _replay_fixed_shape(self, env: ManagerBasedRlEnv, selection: ResetSelection) -> None:
        """Replay the startup friction through full-width masked staging."""
        reset_state = getattr(env, "_reset_state", None)
        packed = (
            None
            if reset_state is None
            else getattr(reset_state, "_require_packed_transaction", None)
        )
        if packed is None:
            raise RuntimeError(
                "SuperMimic fixed-shape reset requires the env-owned packed reset "
                "transaction (unilab >= 1.5.11); update unilab or disable "
                "env.autoreset_cuda_graph"
            )
        # Stage through the packed composer's cached-baseline mask writer.
        # (The transaction-level wrapper also caches its default table since
        # unilab 1.6.0rc1 #2623, but this full-width masked staging path is the
        # one verified under reset-segment graph capture.)
        packed = packed()
        if not self._backend_geom_ids_ready:
            self._backend_geom_ids = (
                None
                if self._geom_ids is None
                else np.asarray(self._robot._geom_ids)[np.asarray(self._geom_ids)]
            )
            if self._ground is not None and self._ground_geom_ids is not None:
                self._ground_backend_geom_ids = np.asarray(self._ground._geom_ids)[
                    np.asarray(self._ground_geom_ids)
                ]
            self._backend_geom_ids_ready = True
        command = _command(env, self._command_name)
        reset_mask = env.event_term_mask_buffer.scatter(selection)

        def full_width(defaults: torch.Tensor) -> torch.Tensor:
            if defaults.ndim == 2:
                return defaults.unsqueeze(0).expand(env.num_envs, *defaults.shape).clone()
            return defaults[: env.num_envs].clone()

        # Friction only: the torso CoM offset stays what the startup term recorded
        # (see the class docstring).
        if self._randomize_friction:
            assert self._backend_geom_ids is not None
            friction = full_width(self._device_friction_defaults)
            friction[..., 0] = command.xi[:, :1]
            packed.write_model_field_mask(
                RESET_TERM_GEOM_FRICTION, reset_mask, friction, columns=self._backend_geom_ids
            )
        if self._ground is not None:
            assert self._ground_backend_geom_ids is not None
            ground_friction = full_width(self._device_ground_friction_defaults)
            ground_friction[..., 0] = 0.3 if not command.eval_mode else 1.0
            packed.write_model_field_mask(
                RESET_TERM_GEOM_FRICTION,
                reset_mask,
                ground_friction,
                columns=self._ground_backend_geom_ids,
            )


class SuperMimicJointOffsetEvent(ManagerTermBase):
    """Sample reset joint offsets into the command's only simulation context."""

    supports_reset_selection: ClassVar[bool] = True

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        if cfg.mode != "reset":
            raise ValueError("SuperMimicJointOffsetEvent requires reset mode")
        self._command_name = _event_command_name(cfg)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | ResetSelection | None,
        command_name: str = "motion",
    ) -> None:
        del command_name
        command = _command(env, self._command_name)
        if is_reset_selection(env_ids):
            # Fixed-shape dispatch: sample full-width every step and merge
            # through the env-owned mask; unselected rows keep their offsets.
            reset_mask = env.event_term_mask_buffer.scatter(env_ids)
            if command.eval_mode:
                command.xi[:, 1:30] = torch.where(reset_mask[:, None], 0.0, command.xi[:, 1:30])
            else:
                sampled = env.rng.uniform(-0.01, 0.01, (env.num_envs, NUM_J))
                command.xi[:, 1:30] = torch.where(reset_mask[:, None], sampled, command.xi[:, 1:30])
            return
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=env.num_envs,
            device=env.device,
            owner="SuperMimicJointOffsetEvent",
        )
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
    "FRONTIER_FRAMES",
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
    "sample_frontier_frames",
    "sample_start_frames",
    "start_bin_probabilities",
    "supermimic_clip_end",
    "supermimic_goal",
    "supermimic_obs",
    "supermimic_reward",
    "supermimic_termination",
    "supermimic_timeout",
    "update_clip_fail_stats",
    "update_start_bin_returns",
]
