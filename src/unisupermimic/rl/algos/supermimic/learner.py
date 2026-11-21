"""SuperMimic training over the injected environment and MDP protocols."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .proc_collector import ProcCollectDriver

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from unilab.base.contracts.env import EnvProtocol

from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    State,
    get_supermimic_mdp,
)
from unisupermimic.rl.logging import SuperMimicLogger

from . import networks
from .geometry import unflatten_state
from .networks import Critic, Policy, WorldModel, mlp
from .optim import MuonAdam
from .replay import ReplayBuffer


@dataclass
class SuperMimicConfig:
    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    goal_frames: int = 10
    hist: int = 10
    z_dim: int = 32
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    wm_structured: bool = False
    policy_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 4e-4
    pi_opt: str = "muon"
    wm_opt: str = "muon"
    no_sync: bool = True
    pi_ema: float = 0.0
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    lr_decay_iters: int = 250_000
    fail_penalty: float = 0.0
    # BlinkMimic2-style exams, all off by default: ``exam_frac`` 0 disables the
    # whole mechanism and keeps the pre-exam behavior. When enabled, the last
    # ``exam_frac * num_envs`` worlds become exam worlds at iteration
    # ``exam_start_iter`` (one-time assignment, no reshuffling): their episodes
    # start on a clip's first frame, run to the clip end without early
    # termination, and are collected with the EMA policy. The share of policy-
    # update imagination starts drawn from those rows' transitions ramps
    # linearly from ``exam_frac`` to ``exam_share_ramp`` over
    # ``exam_ramp_iters`` iterations (0 disables the ramp).
    exam_start_iter: int = 0
    exam_frac: float = 0.0
    exam_share_ramp: float = 0.0
    exam_ramp_iters: int = 2500
    # exams-v2 margin mode, off by default (requires exam_frac > 0). When on,
    # the command keeps per-clip EMA failure statistics (``fail_ema`` decay,
    # BlinkMimic2 uses 0.7) and re-selects the exam clips every
    # ``exam_clip_window`` iterations by margin priority: the per-clip EMA
    # failure rate closest to 1 is examined first, one exam row per clip
    # (device-side topk, deterministic). ``frontier_frac`` of the practice
    # resets (fixed per-row Bernoulli ratio) start inside the frontier window
    # of a failed clip — [failure frame - 50, clip end) — instead of the
    # bin-EMA mixture; the rest keep the existing sampling.
    exam_margin: bool = False
    exam_clip_window: int = 250
    frontier_frac: float = 0.5
    fail_ema: float = 0.7
    ckpt_every: int = 2000
    log_every: int = 50
    log_backend: str = "tensorboard"
    seed: int = 0
    amp: str = "bf16"
    compile: bool = True
    compile_mode: str = "default"
    wm_graph: bool = False
    test_hidden_dims: tuple[int, ...] | None = None
    async_collect: bool = False
    proc_collect: bool = False
    three_way: bool = False


_LOG_FIELDS = (
    "iteration",
    "wall_h",
    "sim_steps",
    "wm_loss",
    "wm_err8",
    "pi_loss",
    "v_loss",
    "imag_rew",
    "pi_gradnorm",
    "imag_survival",
    "sim_rew",
    "ep_len",
    "ep_ret_norm",
    "term_frac",
    "it_per_s",
    "lr",
)

# ROCm's flattened small-matrix bmm path is numerically unreliable above a batch
# boundary (~32x4096 rollout); chunk the geometry-heavy reward/critic inputs
# there. Other backends keep a single fused call so their hot path is unchanged.
_SAFE_GEOMETRY_BATCH = 4096 if getattr(torch.version, "hip", None) is not None else 0


def _empty_stats() -> dict[str, list[torch.Tensor]]:
    return {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": [], "exam_surv": []}


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class _CollectWorker(threading.Thread):
    """Background thread running ``learner._collect_step`` during WM updates.

    The worker owns the environment between ``dispatch`` and ``join``; the main
    thread must not touch the env, the policy, or ``learner.obs_hist`` in that
    window. Staged replay rows stay invisible to samplers until the main thread
    calls ``ReplayBuffer.commit`` after ``join``.
    """

    def __init__(self, learner: SuperMimicLearner):
        super().__init__(daemon=True, name="supermimic-collect")
        self._learner = learner
        self.stream = torch.cuda.Stream(learner.device) if learner.device.type == "cuda" else None
        self.pending_steps = 0
        self.error: BaseException | None = None
        self.staged_stats: dict[str, list[torch.Tensor]] = _empty_stats()
        self.staged_transitions: list[dict[str, torch.Tensor]] = []
        self._request = threading.Event()
        self._completed = threading.Event()
        self._completed.set()
        self._shutdown = False

    def dispatch(self, n_steps: int) -> None:
        if not self._completed.is_set():
            raise RuntimeError("collect worker dispatched before previous join")
        self.pending_steps = n_steps
        self._completed.clear()
        self._request.set()

    def run(self) -> None:
        while True:
            self._request.wait()
            self._request.clear()
            if self._shutdown:
                self._completed.set()
                return
            learner = self._learner
            try:
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        self._collect_pending(learner)
                else:
                    self._collect_pending(learner)
            except BaseException as exc:  # surfaced to the main thread at join
                self.error = exc
            self._completed.set()

    def _collect_pending(self, learner: SuperMimicLearner) -> None:
        for _ in range(self.pending_steps):
            transition = learner._collect_step(self.staged_stats)
            self.staged_transitions.append(transition)

    def harvest(self) -> None:
        """Wait for in-flight steps, then publish staged rows, norms, and stats."""
        self._completed.wait()
        if self.error is not None:
            raise RuntimeError("collect worker failed") from self.error
        if self.stream is not None:
            torch.cuda.current_stream(self._learner.device).wait_stream(self.stream)
        learner = self._learner
        for transition in self.staged_transitions:
            learner._apply_wm_norms(transition)
        learner.buffer.commit(len(self.staged_transitions))
        learner.sim_steps += len(self.staged_transitions) * learner.env.num_envs
        for key, staged in self.staged_stats.items():
            learner.stats[key].extend(staged)
        self.staged_transitions = []
        self.staged_stats = _empty_stats()
        self.pending_steps = 0

    def shutdown(self) -> None:
        """Stop the worker loop; never raises (used from ``finally`` blocks)."""
        self._shutdown = True
        self._request.set()
        self._completed.wait()
        super().join(timeout=5.0)


def _maybe_start_profiler(tag: str):
    """Env-gated torch.profiler (kineto/CUPTI) wrapper for kernel-timeline
    analysis. Enabled when UNISUPERMIMIC_PROFILE_DIR is set; writes one
    chrome trace per process into that directory."""
    directory = os.environ.get("UNISUPERMIMIC_PROFILE_DIR")
    if not directory:
        return None
    from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

    os.makedirs(directory, exist_ok=True)
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(directory, worker_name=tag),
    ).__enter__()


class SuperMimicLearner:
    def __init__(
        self,
        env: EnvProtocol,
        cfg: SuperMimicConfig,
        log_dir: str | Path,
        device: str | torch.device | None = None,
        backend: str = "",
        collector_env_spec: dict[str, Any] | None = None,
    ):
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        if cfg.utd <= 0:
            raise ValueError("utd must be positive")
        if cfg.pi_opt not in ("adam", "muon"):
            raise ValueError(f"pi_opt must be 'adam' or 'muon', got {cfg.pi_opt!r}")
        if cfg.wm_opt not in ("adam", "muon"):
            raise ValueError(f"wm_opt must be 'adam' or 'muon', got {cfg.wm_opt!r}")
        if cfg.test_hidden_dims is not None and not cfg.test_hidden_dims:
            raise ValueError("test_hidden_dims must be nonempty when provided")
        exam_rows = 0
        if cfg.exam_frac > 0.0:
            if cfg.pi_ema <= 0.0:
                raise ValueError(
                    "SuperMimic exams require pi_ema > 0: exam worlds act with the EMA policy"
                )
            if not 0.0 < cfg.exam_frac < 1.0:
                raise ValueError(
                    f"exam_frac must be in (0, 1) when exams are enabled, got {cfg.exam_frac}"
                )
            if not 0.0 <= cfg.exam_share_ramp < 1.0:
                raise ValueError(f"exam_share_ramp must be in [0, 1), got {cfg.exam_share_ramp}")
            if cfg.exam_start_iter < 0:
                raise ValueError(f"exam_start_iter must be >= 0, got {cfg.exam_start_iter}")
            if cfg.exam_share_ramp > 0.0 and cfg.exam_ramp_iters <= 0:
                raise ValueError(
                    f"exam_ramp_iters must be positive when exam_share_ramp is set, "
                    f"got {cfg.exam_ramp_iters}"
                )
            if not callable(getattr(env, "enable_supermimic_exams", None)):
                raise TypeError(
                    f"{type(env).__name__} does not support SuperMimic exams "
                    "(missing enable_supermimic_exams)"
                )
            exam_rows = int(round(cfg.exam_frac * env.num_envs))
            if not 0 < exam_rows < env.num_envs:
                raise ValueError(
                    f"exam_frac {cfg.exam_frac} resolves to {exam_rows} exam rows of "
                    f"{env.num_envs} environments"
                )
        if cfg.exam_margin:
            if exam_rows == 0:
                raise ValueError(
                    "SuperMimic exam_margin requires exam_frac > 0: margin mode "
                    "selects the exam worlds' clips"
                )
            if not 0.0 <= cfg.frontier_frac <= 1.0:
                raise ValueError(f"frontier_frac must be in [0, 1], got {cfg.frontier_frac}")
            if not 0.0 <= cfg.fail_ema < 1.0:
                raise ValueError(f"fail_ema must be in [0, 1), got {cfg.fail_ema}")
            if isinstance(cfg.exam_clip_window, bool) or cfg.exam_clip_window <= 0:
                raise ValueError(f"exam_clip_window must be positive, got {cfg.exam_clip_window}")
            if not callable(getattr(env, "configure_supermimic_exam_margin", None)):
                raise TypeError(
                    f"{type(env).__name__} does not support SuperMimic margin exams "
                    "(missing configure_supermimic_exam_margin)"
                )
            # Configure before any stepping (and any reset-segment capture):
            # the command bakes these Python-level switches into the captured
            # segment's branch layout once, here.
            env.configure_supermimic_exam_margin(
                frontier_frac=cfg.frontier_frac, fail_ema=cfg.fail_ema
            )
        self._exam_rows_planned = exam_rows
        if cfg.compile_mode == "reduce-overhead":
            # policy_update stores compiled-fn outputs (states/rewards) across
            # loop generations; cudagraph_trees reuses those buffers unless
            # user-visible outputs are cloned.
            import torch._inductor.config as _inductor_config

            _inductor_config.triton.cudagraph_trees_generation_cloning = "user_visible"
            _inductor_config.triton.cudagraph_trees_history_recording = True
        if cfg.compile_mode not in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
            raise ValueError(
                f"compile_mode must be default/reduce-overhead/max-autotune/max-autotune-no-cudagraphs, got {cfg.compile_mode!r}"
            )
        if not isinstance(cfg.async_collect, bool):
            raise TypeError(
                f"async_collect must be a bool (use true/false, not on/off), got {cfg.async_collect!r}"
            )
        if not isinstance(cfg.proc_collect, bool):
            raise TypeError(
                f"proc_collect must be a bool (use true/false, not on/off), got {cfg.proc_collect!r}"
            )
        if cfg.async_collect and cfg.proc_collect:
            raise ValueError("async_collect and proc_collect are mutually exclusive")
        if cfg.async_collect and backend:
            from unilab.base.process_device import is_host_bridge_backend

            if not is_host_bridge_backend(backend):
                raise ValueError(
                    f"threaded async_collect is only safe for HOST_BRIDGE backends, got "
                    f"{backend!r}; use proc_collect (process-isolated collector) instead"
                )
        if cfg.proc_collect and collector_env_spec is None:
            raise ValueError("proc_collect requires collector_env_spec")
        if not isinstance(cfg.three_way, bool):
            raise TypeError(f"three_way must be a bool, got {cfg.three_way!r}")
        if cfg.three_way and not cfg.proc_collect:
            raise ValueError("three_way requires proc_collect")
        self._collector_env_spec = collector_env_spec

        self.env = env
        self.cfg = cfg
        self.backend = backend
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env_device = torch.device(env.device)
        self.device = self.env_device if device is None else torch.device(device)
        # HOST_BRIDGE backends (e.g. MuJoCo) keep env tensors on CPU while the
        # learner may run on an accelerator (MPS on macOS); when the two differ,
        # the learner works on its own MDP copy and bridges tensors at the env
        # boundary. On CUDA/ROCm the devices always match and nothing changes.
        self._bridged = self.device != self.env_device
        mdp = get_supermimic_mdp(env)
        self.mdp = mdp.to(self.device) if self._bridged else mdp

        # TF32 only pays off for fp32 (amp="none") training: under bf16 autocast
        # the heavy GEMMs already run in bf16, while on ROCm a TF32 bmm over the
        # misaligned strided views produced by unflatten_state triggers per-call
        # rocBLAS kernel selection costing ~100 ms per call.
        torch.backends.cuda.matmul.allow_tf32 = cfg.amp == "none"
        torch.backends.cudnn.allow_tf32 = True
        self._wm_graph_state: dict = {"warmups": 0, "graph": None}
        torch.manual_seed(cfg.seed)
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
        networks.RunningNorm.no_sync = cfg.no_sync

        self.wm = self._make_world_model().to(self.device)
        self.pi = self._make_policy().to(self.device)
        self.critic = Critic(self._critic_input_dim(), hidden=cfg.critic_hidden).to(self.device)
        if cfg.test_hidden_dims is not None:
            self.critic.net = mlp(self.critic.norm.mean.numel(), cfg.test_hidden_dims, 1).to(
                self.device
            )
        self.pi_avg = None
        if cfg.pi_ema > 0:
            # Polyak average of the policy weights; checkpoints and evaluation use it while the
            # raw ``self.pi`` keeps being trained. Buffers (running normalizers) are averaged too.
            self.pi_avg = AveragedModel(
                self.pi,
                multi_avg_fn=get_ema_multi_avg_fn(cfg.pi_ema),
                use_buffers=True,
            )
        fused_adam = self.device.type == "cuda"
        if cfg.wm_opt == "muon":
            self.opt_wm = MuonAdam(
                self.wm,
                [self.wm.net],
                lr=cfg.lr,
                betas=(0.9, 0.999),
                fused=fused_adam,
                capturable=cfg.wm_graph,
            )
        else:
            self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr, fused=fused_adam)
        if cfg.pi_opt == "muon":
            self.opt_pi = MuonAdam(
                self.pi,
                [self.pi.enc, self.pi.dec],
                lr=cfg.lr,
                betas=(0.9, 0.95),
                fused=fused_adam,
            )
        else:
            self.opt_pi = torch.optim.Adam(
                self.pi.parameters(), lr=cfg.lr, betas=(0.9, 0.95), fused=fused_adam
            )
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr, fused=fused_adam)
        self._v_mean = torch.zeros((), device=self.device)
        self._v_std = torch.ones((), device=self.device)
        self._v_ema = 0.01
        self._v_init = False

        self.buffer = ReplayBuffer(
            cfg.buffer_steps,
            env.num_envs,
            self.device,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
        )
        self.obs_hist = torch.zeros(env.num_envs, cfg.hist, self.mdp.obs_dim, device=self.device)
        self.err_floor = torch.cat(
            (
                torch.full((3,), 3e-3),
                torch.full((3,), 5e-3),
                torch.full((3,), 2e-2),
                torch.full((3,), 5e-2),
                torch.full((self.mdp.num_joints,), 3e-3),
                torch.full((self.mdp.num_joints,), 5e-2),
            )
        ).to(self.device)

        self.iteration = 0
        self.sim_steps = 0
        self.wall_time = 0.0
        # Exam schedule (all zeros while exams are off or not yet started):
        # ``_exam_rows`` counts the active exam worlds (the last rows), and
        # ``_exam_share`` is the probability each imagination start is drawn
        # from those rows' transitions. ``_exam_fail_ema`` carries the mean
        # failure EMA of the currently examined clips (margin mode only).
        self._exam_rows = 0
        self._exam_share = 0.0
        self._exam_fail_ema: float | None = None
        self.stats = _empty_stats()
        # three_way: frozen one-iteration-lagged WM so policy_update overlaps
        # WM training (collector ∥ WM ∥ policy). Params stay frozen forever.
        self.wm_shadow: WorldModel | None = None
        self._imag_step_shadow_fn = None
        self._pi_stream: torch.cuda.Stream | None = None
        if cfg.three_way:
            self.wm_shadow = self._make_world_model().to(self.device)
            _set_requires_grad(self.wm_shadow, False)
        self._pi_metrics: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._pi_error: BaseException | None = None
        self._wm_rollout_fn = self._compile(self._wm_rollout)
        self._wm_err_fn = self._compile(self._wm_err)
        self._imag_step_fn = self._compile(self._imag_step_main)
        self._reward_term_fn = self._compile(self._reward_term)
        self._critic_input_fn = self._compile(self._critic_input)
        if cfg.three_way:
            self._imag_step_shadow_fn = self._compile(self._imag_step_shadow)
            if self.device.type == "cuda":
                self._pi_stream = torch.cuda.Stream(self.device)

        with (self.log_dir / "args.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(cfg), stream, indent=2)

    def _compile(self, function):
        if not self.cfg.compile:
            return function
        if self.device.type == "mps":
            # Inductor Metal codegen is not usable on MPS; keep AOT graph
            # capture without kernel generation.
            return torch.compile(function, backend="aot_eager")
        if self.cfg.compile_mode != "default":
            return torch.compile(function, mode=self.cfg.compile_mode)
        return torch.compile(function)

    def _make_world_model(self) -> WorldModel:
        if self.cfg.test_hidden_dims is not None:
            return WorldModel(
                hidden=self.cfg.test_hidden_dims,
                num_joints=self.mdp.num_joints,
                context_dim=self.mdp.context_dim,
                structured=self.cfg.wm_structured,
            )
        return WorldModel(
            hidden=self.cfg.wm_hidden,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
            structured=self.cfg.wm_structured,
        )

    def _make_policy(self) -> Policy:
        goal_dim = self.mdp.goal_frame_dim * self.cfg.goal_frames
        policy = Policy(
            self.mdp.obs_dim,
            goal_dim,
            hist=self.cfg.hist,
            z_dim=self.cfg.z_dim,
            num_joints=self.mdp.num_joints,
            hidden=self.cfg.policy_hidden,
        )
        if self.cfg.test_hidden_dims is not None:
            enc_output = policy.enc[-1]
            assert isinstance(enc_output, nn.Linear)
            z_dim = enc_output.out_features
            policy.enc = mlp(goal_dim, self.cfg.test_hidden_dims, z_dim)
            policy.dec = mlp(
                z_dim + self.mdp.obs_dim * self.cfg.hist,
                self.cfg.test_hidden_dims,
                self.mdp.num_joints,
            )
            dec_output = policy.dec[-1]
            assert isinstance(dec_output, nn.Linear)
            with torch.no_grad():
                dec_output.weight.mul_(0.01)
                dec_output.bias.zero_()
        return policy

    def _state_info(self) -> dict[str, torch.Tensor]:
        state = self.env.state
        if state is None:
            state = self.env.init_state()
        info: dict[str, torch.Tensor] = state.info[SUPERMIMIC_STATE_KEY]
        if self._bridged:
            info = {key: value.to(self.device) for key, value in info.items()}
        return info

    def _critic_input_dim(self) -> int:
        info = self._state_info()
        with torch.no_grad():
            features = self.mdp.critic_features(
                unflatten_state(info["s"]),
                info["f"],
                self.mdp.normalize_context(info["xi"]),
            )
        return int(features.shape[-1])

    def _wm_rollout(
        self, states: torch.Tensor, actions: torch.Tensor, contexts: torch.Tensor
    ) -> State:
        state = unflatten_state(states[:, 0])
        predictions = []
        for index in range(self.cfg.wm_window):
            state = self.wm.step(
                state,
                actions[:, index],
                self.mdp.normalize_context(contexts[:, index]),
            )
            predictions.append(tuple(t.clone() for t in state))
        return (
            torch.stack([prediction[0] for prediction in predictions], 1),
            torch.stack([prediction[1] for prediction in predictions], 1),
            torch.stack([prediction[2] for prediction in predictions], 1),
            torch.stack([prediction[3] for prediction in predictions], 1),
            torch.stack([prediction[4] for prediction in predictions], 1),
            torch.stack([prediction[5] for prediction in predictions], 1),
        )

    def _wm_err(
        self, predicted: State, next_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.maximum(self.wm.inc_norm.std(), self.err_floor)
        error = (self.wm.state_error(predicted, unflatten_state(next_states)) / std) ** 2
        error = error.mean(-1)
        return (error * mask).sum() / next_states.shape[0], error[:, -1]

    def _imag_step(
        self,
        wm: WorldModel,
        state: State,
        obs_hist: torch.Tensor,
        previous_action: torch.Tensor,
        frames: torch.Tensor,
        context: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[State, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = self.pi(obs_hist, goal)
        next_state = wm.step(state, action, normalized_context)
        next_frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, context, next_frames))
        next_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        next_goal = self.mdp.goal(next_state, next_frames, self.cfg.goal_frames)
        return next_state, action, next_hist, next_frames, next_goal

    def _imag_step_main(
        self,
        state: State,
        obs_hist: torch.Tensor,
        previous_action: torch.Tensor,
        frames: torch.Tensor,
        context: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[State, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._imag_step(
            self.wm, state, obs_hist, previous_action, frames, context, normalized_context, goal
        )

    def _imag_step_shadow(
        self,
        state: State,
        obs_hist: torch.Tensor,
        previous_action: torch.Tensor,
        frames: torch.Tensor,
        context: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[State, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.wm_shadow is not None
        return self._imag_step(
            self.wm_shadow,
            state,
            obs_hist,
            previous_action,
            frames,
            context,
            normalized_context,
            goal,
        )

    def _reward_term(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mdp.reward(state, frames, action, previous_action), self.mdp.soft_termination(
            state, frames
        )

    def _critic_input(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        return self.mdp.critic_features(state, frames, normalized_context)

    def _batched_reward_term(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _SAFE_GEOMETRY_BATCH or len(frames) <= _SAFE_GEOMETRY_BATCH:
            reward, done = self._reward_term_fn(state, frames, action, previous_action)
            return reward.clone(), done.clone()
        reward_parts, done_parts = [], []
        for start in range(0, len(frames), _SAFE_GEOMETRY_BATCH):
            stop = start + _SAFE_GEOMETRY_BATCH
            reward, done = self._reward_term_fn(
                cast(State, tuple(item[start:stop] for item in state)),
                frames[start:stop],
                action[start:stop],
                previous_action[start:stop],
            )
            reward_parts.append(reward)
            done_parts.append(done)
        return torch.cat(reward_parts), torch.cat(done_parts)

    def _batched_critic_input(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        if not _SAFE_GEOMETRY_BATCH or len(frames) <= _SAFE_GEOMETRY_BATCH:
            return self._critic_input_fn(state, frames, normalized_context).clone()
        parts = []
        for start in range(0, len(frames), _SAFE_GEOMETRY_BATCH):
            stop = start + _SAFE_GEOMETRY_BATCH
            parts.append(
                self._critic_input_fn(
                    cast(State, tuple(item[start:stop] for item in state)),
                    frames[start:stop],
                    normalized_context[start:stop],
                )
            )
        return torch.cat(parts)

    @torch.no_grad()
    def collect(self, n_steps: int) -> None:
        for _ in range(n_steps):
            transition = self._collect_step(self.stats)
            self._apply_wm_norms(transition)
            self.buffer.commit(1)
            self.sim_steps += self.env.num_envs

    @torch.no_grad()
    def _collect_step(self, stats: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Run one environment step and stage the transition (not yet committed)."""
        info = self._state_info()
        state = unflatten_state(info["s"])
        obs = self.mdp.add_noise(self.mdp.obs(state, info["a_prev"], info["xi"], info["f"]))
        fresh = info["ep_step"] == 0
        shifted = torch.cat((self.obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        self.obs_hist = torch.where(
            fresh[:, None, None], obs.unsqueeze(1).expand_as(self.obs_hist), shifted
        )
        goal = self.mdp.goal(state, info["f"], self.cfg.goal_frames)
        self.pi.obs_norm.update(obs)
        self.pi.goal_norm.update(goal)
        action = self.pi(self.obs_hist, goal)
        action = torch.clamp(action, -20, 20)
        if self._exam_rows:
            # Exam worlds act with the deployed (EMA) policy, as the
            # evaluation would; their transitions still feed the replay
            # buffer and the world-model normalizers below.
            assert self.pi_avg is not None
            exam_action = self.pi_avg.module(
                self.obs_hist[-self._exam_rows :], goal[-self._exam_rows :]
            )
            action[-self._exam_rows :] = torch.clamp(exam_action, -20, 20).to(action.dtype)
        if self._bridged:
            step_state = self.env.step(action.to(self.env_device))
        else:
            step_state = self.env.step(action)
        transition = cast(dict[str, torch.Tensor], step_state.info[SUPERMIMIC_TRANSITION_KEY])
        if self._bridged:
            transition = {key: value.to(self.device) for key, value in transition.items()}
        self.buffer.add_staged(transition)
        stats["r"].append(transition["r"].mean() / self.mdp.max_step_reward)
        if self._exam_rows:
            # Per-step survival of the exam worlds: the share whose danger
            # probability stays under the termination band. These episodes
            # never hard-terminate, so this is the observable for whether
            # the exams are improving.
            rows = slice(self.env.num_envs - self._exam_rows, None)
            danger = self.mdp.soft_termination(
                unflatten_state(transition["s1"][rows]), transition["f"][rows] + 1
            )
            stats["exam_surv"].append((danger < 0.5).float().mean())
        if "ep_len" in transition:
            done = transition["done"]
            term = transition["term"]
            ep_len = transition["ep_len"]
            ep_ret = transition["ep_ret_norm"]
            if self._exam_rows:
                keep = self.env.num_envs - self._exam_rows
                done, term, ep_len, ep_ret = done[:keep], term[:keep], ep_len[:keep], ep_ret[:keep]
            done_count = float(done.sum())
            if done_count > 0.0:
                stats["ep_len"].append(float(ep_len.sum()) / done_count)
                stats["ep_ret"].append(float(ep_ret.sum()) / done_count)
            stats["term"].append(term.float().sum())
            stats["done"].append(done.float().sum())
        return transition

    def _apply_wm_norms(self, transition: dict[str, torch.Tensor]) -> None:
        initial = unflatten_state(transition["s"])
        next_state = unflatten_state(transition["s1"])
        self.wm.in_norm.update(
            self.wm.inputs(
                initial,
                transition["a"],
                self.mdp.normalize_context(transition["xi"]),
            )
        )
        target = self.wm.targets(initial, next_state)
        self.wm.out_norm.update(target)
        if self.cfg.wm_structured:
            self.wm.inc_norm.update(self.wm.increments(initial, next_state))
        else:
            self.wm.inc_norm.update(target)

    def wm_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window)
        if self.cfg.wm_graph:
            return self._wm_update_graphed(window)
        return self._wm_update_body(window)

    def _wm_update_body(self, window: dict) -> tuple[torch.Tensor, torch.Tensor]:
        predicted = self._wm_rollout_fn(window["s"], window["a"], window["xi"])
        loss, last_error = self._wm_err_fn(predicted, window["s1"], window["mask"])
        loss, last_error = loss.clone(), last_error.clone()
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        mask = window["mask"][:, -1]
        mean_last_error = (last_error.detach() * mask).sum() / mask.sum().clamp(min=1)
        return loss.detach() / self.cfg.wm_window, mean_last_error

    def _advance_exam_schedule(self, iteration: int) -> None:
        """Activate exams at ``exam_start_iter`` and ramp the imagination share.

        Activation is a one-time, eager, iteration-boundary event: the env
        flips its persistent exam mask in place and resets the newly assigned
        rows onto clip first frames (margin mode selects those clips by
        failure-EMA priority). Afterwards only the device-side sampling
        probability changes, linearly from ``exam_frac`` to
        ``exam_share_ramp``, and — in margin mode — the exam clip table is
        re-selected every ``exam_clip_window`` iterations (eagerly, outside
        every captured segment; the activation iteration already selected).
        """
        if self._exam_rows_planned == 0:
            return
        if self._exam_rows == 0:
            if iteration < self.cfg.exam_start_iter:
                return
            self.env.enable_supermimic_exams(self._exam_rows_planned)
            self._exam_rows = self._exam_rows_planned
            self._exam_share = self.cfg.exam_frac
        if self.cfg.exam_share_ramp > 0.0:
            progress = (iteration - self.cfg.exam_start_iter) / self.cfg.exam_ramp_iters
            progress = min(max(progress, 0.0), 1.0)
            self._exam_share = self.cfg.exam_frac + progress * (
                self.cfg.exam_share_ramp - self.cfg.exam_frac
            )
        if (
            self.cfg.exam_margin
            and iteration % self.cfg.exam_clip_window == 0
            and iteration != self.cfg.exam_start_iter
        ):
            mean_fail_ema = self.env.refresh_supermimic_exam_clips(self._exam_rows)
            # One host read per refresh window (never inside a captured
            # segment) so TensorBoard can watch the margin converge.
            self._exam_fail_ema = float(mean_fail_ema)
    def _wm_graph_body(self, window: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # zero_grad must be a captured kernel (set_to_none=False): a Python-level
        # ``grad = None`` assignment would not replay, leaving stale gradients.
        predicted = self._wm_rollout_fn(window["s"], window["a"], window["xi"])
        loss, last_error = self._wm_err_fn(predicted, window["s1"], window["mask"])
        self.opt_wm.zero_grad(set_to_none=False)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        mask = window["mask"][:, -1]
        mean_last_error = (last_error.detach() * mask).sum() / mask.sum().clamp(min=1)
        return loss.detach() / self.cfg.wm_window, mean_last_error

    def _wm_update_graphed(self, window: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._wm_graph_state
        if state["warmups"] < 3:
            state["warmups"] += 1
            return self._wm_update_body(window)
        if state["graph"] is None:
            statics = {key: value.clone() for key, value in window.items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._wm_graph_body(statics)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            # Free stale grads so capture-time backward allocates them INTO the
            # graph pool; otherwise replay keeps writing to freed general-heap
            # addresses that the allocator may hand to other tensors (opt state).
            self.opt_wm.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                outs = self._wm_graph_body(statics)
            state.update(graph=graph, statics=statics, outs=outs)
            return self._wm_update_body(window)
        for key, value in window.items():
            state["statics"][key].copy_(value, non_blocking=True)
        state["graph"].replay()
        out_loss, out_err = state["outs"]
        return out_loss.clone(), out_err.clone()

    def _sample_valid_starts(
        self, batch_size: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # While exams run, prefer exam-row transitions for imagination starts
        # (BlinkMimic2's practice/exam start split).
        exam_rows = self._exam_rows if self._exam_share > 0.0 else 0
        if self.cfg.no_sync:
            # Fixed-size selection without reading the valid count on the host: oversample by 2x,
            # stable-sort the valid starts first, and keep the leading ``batch_size``. Only starts
            # within ``horizon`` frames of a clip end are invalid (< 1 %), so 2x is enough.
            start, history = self.buffer.sample_starts(
                2 * batch_size, self.cfg.hist, exam_rows=exam_rows, exam_prob=self._exam_share
            )
            valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon)
            indices = torch.sort((~valid).to(torch.uint8), stable=True).indices[:batch_size]
            return (
                {key: value[indices] for key, value in start.items()},
                {key: value[indices] for key, value in history.items()},
            )

        starts = []
        histories = []
        count = 0
        while count < batch_size:
            start, history = self.buffer.sample_starts(
                2 * batch_size, self.cfg.hist, exam_rows=exam_rows, exam_prob=self._exam_share
            )
            valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon)
            indices = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - count]
            starts.append({key: value[indices] for key, value in start.items()})
            histories.append({key: value[indices] for key, value in history.items()})
            count += len(indices)

        def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        return concatenate(starts), concatenate(histories)


    def policy_update(
        self,
        *,
        wm: WorldModel | None = None,
        imag_step_fn=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wm = self.wm if wm is None else wm
        imag_step_fn = self._imag_step_fn if imag_step_fn is None else imag_step_fn
        start, history = self._sample_valid_starts(self.cfg.pi_batch)
        state = unflatten_state(start["s"])
        context, frames, previous_action = start["xi"], start["f"], start["a_prev"]
        normalized_context = self.mdp.normalize_context(context)
        with torch.no_grad():
            obs_hist = self.mdp.obs(
                unflatten_state(history["s"]),
                history["a_prev"],
                history["xi"],
                history["f"],
            )
        obs_hist = self.mdp.add_noise(obs_hist)
        _set_requires_grad(wm, False)
        _set_requires_grad(self.critic, False)

        batch_size = self.cfg.pi_batch
        goal = self.mdp.goal(state, frames, self.cfg.goal_frames)
        initial_state = state
        states, actions, previous_actions, all_frames = [], [], [], []
        try:
            for _ in range(self.cfg.horizon):
                if self.cfg.compile_mode == "reduce-overhead":
                    torch.compiler.cudagraph_mark_step_begin()
                previous_actions.append(previous_action.clone())
                next_out = imag_step_fn(
                    state,
                    obs_hist,
                    previous_action,
                    frames,
                    context,
                    normalized_context,
                    goal,
                )
                state = tuple(t.clone() for t in next_out[0])
                previous_action = next_out[1].clone()
                obs_hist = next_out[2].clone()
                frames = next_out[3].clone()
                goal = next_out[4].clone()
                states.append(tuple(t.clone() for t in state))
                actions.append(previous_action.clone())
                all_frames.append(frames)

            def flatten(parts: list[torch.Tensor]) -> torch.Tensor:
                return torch.stack(parts).flatten(0, 1)

            all_states = cast(
                State, tuple(flatten([item[index] for item in states]) for index in range(6))
            )
            flat_frames = flatten(all_frames)
            rewards, dones = self._batched_reward_term(
                all_states, flat_frames, flatten(actions), flatten(previous_actions)
            )
            rewards = rewards.view(self.cfg.horizon, batch_size)
            dones = dones.view(self.cfg.horizon, batch_size)
            alive = torch.cumprod(1 - dones, 0)
            continuation = torch.cat((torch.ones(1, batch_size, device=self.device), alive[:-1]), 0)
            discount = self.cfg.gamma ** torch.arange(
                self.cfg.horizon, device=self.device, dtype=torch.float32
            ).unsqueeze(-1)
            # BlinkMimic2-style actor-only failure penalty: the imagined reward
            # used by the policy objective is shaped by the soft-termination
            # probability, while the critic below keeps learning the plain
            # tracking reward (``detached_rewards``).
            if self.cfg.fail_penalty:
                actor_rewards = rewards - self.cfg.fail_penalty * dones
            else:
                actor_rewards = rewards
            objective = (discount * continuation * actor_rewards).sum(0)
            survival = alive[-1]
            reward_sum = (continuation * rewards).detach().sum(0)
            detached_rewards, detached_dones = rewards.detach(), dones.detach()
            if not self.cfg.no_critic:
                bootstrap = self.critic(self._critic_input_fn(state, frames, normalized_context))
                objective = objective + (self.cfg.gamma**self.cfg.horizon) * survival * (
                    bootstrap * self._v_std + self._v_mean
                )
            loss = -objective.mean()
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
            if self.cfg.no_sync:
                # Skip a non-finite update without reading the norm on the host: zero the
                # gradients on-device instead of branching on ``torch.isfinite``.
                finite = torch.isfinite(grad_norm)
                for parameter in self.pi.parameters():
                    if parameter.grad is not None:
                        parameter.grad.copy_(torch.where(finite, parameter.grad, 0.0))
                self.opt_pi.step()
            elif torch.isfinite(grad_norm):
                self.opt_pi.step()
            if self.pi_avg is not None:
                self.pi_avg.update_parameters(self.pi)
        finally:
            _set_requires_grad(wm, True)
            _set_requires_grad(self.critic, True)

        value_loss = torch.zeros((), device=self.device)
        if not self.cfg.no_critic:
            with torch.no_grad():
                critic_states = cast(
                    State,
                    tuple(
                        torch.cat(
                            (
                                initial_state[index].unsqueeze(0),
                                torch.stack([item[index].detach() for item in states]),
                            ),
                            0,
                        ).flatten(0, 1)
                        for index in range(6)
                    ),
                )
                critic_frames = torch.cat(
                    (start["f"].unsqueeze(0), torch.stack(all_frames)), 0
                ).flatten()
                critic_context = (
                    normalized_context.unsqueeze(0)
                    .expand(self.cfg.horizon + 1, -1, -1)
                    .flatten(0, 1)
                )
                critic_input = (
                    self._batched_critic_input(critic_states, critic_frames, critic_context)
                    .clone()
                    .view(self.cfg.horizon + 1, batch_size, -1)
                )
            self.critic.norm.update(critic_input)
            with torch.no_grad():
                values = self.critic(critic_input) * self._v_std + self._v_mean
                target = values[-1]
                targets = []
                for index in range(self.cfg.horizon - 1, -1, -1):
                    target = detached_rewards[index] + self.cfg.gamma * (
                        1 - detached_dones[index]
                    ) * ((1 - self.cfg.lam) * values[index + 1] + self.cfg.lam * target)
                    targets.append(target)
                targets = torch.stack(targets[::-1])
                target_mean = targets.mean()
                target_std = targets.std().clamp(min=1e-2)
                if not self._v_init:
                    self._v_mean = target_mean.detach().clone()
                    self._v_std = target_std.detach().clone()
                    self._v_init = True
                else:
                    self._v_mean = (1 - self._v_ema) * self._v_mean + self._v_ema * target_mean
                    self._v_std = (
                        ((1 - self._v_ema) * self._v_std**2 + self._v_ema * target_std**2)
                        .sqrt()
                        .clamp(min=1e-2)
                    )
                targets = ((targets - self._v_mean) / self._v_std).clamp(-20.0, 20.0)
            flat_input = critic_input[:-1].reshape(-1, critic_input.shape[-1])
            flat_targets = targets.reshape(-1)
            num_samples = flat_input.shape[0]
            minibatch_size = num_samples // self.cfg.critic_minibatches
            if minibatch_size == 0:
                raise ValueError("critic_minibatches exceeds the number of imagined transitions")
            for _ in range(self.cfg.critic_epochs):
                permutation = torch.randperm(num_samples, device=self.device)
                for index in range(self.cfg.critic_minibatches):
                    indices = permutation[index * minibatch_size : (index + 1) * minibatch_size]
                    value_loss = (
                        (self.critic(flat_input[indices]) - flat_targets[indices]) ** 2
                    ).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()

        imagined_reward = (reward_sum / self.cfg.horizon / self.mdp.max_step_reward).mean()
        return (
            loss.detach(),
            value_loss.detach(),
            imagined_reward,
            grad_norm.detach(),
            survival.mean().detach(),
        )

    def _run_policy_shadow(self) -> None:
        """Policy-update worker for three_way: uses the 1-iter-lagged shadow WM."""
        try:
            if self._pi_stream is not None:
                with torch.cuda.stream(self._pi_stream):
                    self._pi_metrics = self.policy_update(
                        wm=self.wm_shadow, imag_step_fn=self._imag_step_shadow_fn
                    )
            else:
                self._pi_metrics = self.policy_update(
                    wm=self.wm_shadow, imag_step_fn=self._imag_step_shadow_fn
                )
        except BaseException as exc:  # surfaced to the main thread at join
            self._pi_error = exc

    def lr_at(self, iteration: int) -> float:
        if iteration >= self.cfg.lr_decay_iters:
            return self.cfg.lr * 0.03
        return self.cfg.lr * (
            0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / self.cfg.lr_decay_iters))
        )

    def _checkpoint(self, include_optimizers: bool) -> dict[str, Any]:
        policy_state = (
            self.pi_avg.module.state_dict() if self.pi_avg is not None else self.pi.state_dict()
        )
        checkpoint = {
            "wm": self.wm.state_dict(),
            "pi": policy_state,
            "iteration": self.iteration,
            "wall_time": self.wall_time,
            "sim_steps": self.sim_steps,
            "args": asdict(self.cfg),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.pi_avg is not None:
            checkpoint["pi_live"] = self.pi.state_dict()
        if include_optimizers:
            checkpoint.update(
                {
                    "critic": self.critic.state_dict(),
                    "opt_wm": self.opt_wm.state_dict(),
                    "opt_pi": self.opt_pi.state_dict(),
                    "opt_v": self.opt_v.state_dict(),
                }
            )
        return checkpoint

    def save(self, path: str | Path | None = None, include_optimizers: bool = True) -> Path:
        destination = Path(path) if path is not None else self.log_dir / "last.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._checkpoint(include_optimizers), temporary)
        os.replace(temporary, destination)
        return destination

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        required = {"critic", "opt_wm", "opt_pi", "opt_v"}
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(
                f"SuperMimic resume requires a full last.pt checkpoint; missing {missing} in {path}"
            )
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_structured = bool(checkpoint_args.get("wm_structured", False))
        if checkpoint_structured != self.cfg.wm_structured:
            raise ValueError(
                "Cannot resume with a different algo.wm_structured setting "
                f"(checkpoint={checkpoint_structured}, requested={self.cfg.wm_structured})"
            )
        self.wm.load_state_dict(checkpoint["wm"])
        if "pi_live" in checkpoint:
            self.pi.load_state_dict(checkpoint["pi_live"])
        else:
            self.pi.load_state_dict(checkpoint["pi"])
        if self.pi_avg is not None:
            self.pi_avg.module.load_state_dict(checkpoint["pi"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.opt_wm.load_state_dict(checkpoint["opt_wm"])
        self.opt_pi.load_state_dict(checkpoint["opt_pi"])
        self.opt_v.load_state_dict(checkpoint["opt_v"])
        self.iteration = checkpoint["iteration"]
        self.wall_time = checkpoint["wall_time"]
        self.sim_steps = checkpoint["sim_steps"]

    @staticmethod
    def _mean(values: list[torch.Tensor]) -> float:
        return float(sum(float(value) for value in values) / max(len(values), 1))

    def train(self, resume: str | Path | None = None) -> dict[str, float | int | bool]:
        if resume is not None:
            self.load(resume)
        start_iteration = self.iteration
        if self.cfg.three_way and self.wm_shadow is not None and start_iteration > 0:
            # Resume: the shadow WM must mirror the restored live WM, not the
            # freshly initialized module.
            self.wm_shadow.load_state_dict(self.wm.state_dict())
        self.stats = _empty_stats()
        logger = SuperMimicLogger(
            env_name=type(self.env).__name__,
            backend=self.backend,
            max_iterations=self.cfg.iterations,
            start_iteration=start_iteration,
            num_envs=self.env.num_envs,
            log_dir=str(self.log_dir),
            log_backend=self.cfg.log_backend,
        )
        logger.start(status="Warming up...")
        worker: _CollectWorker | ProcCollectDriver | None = None
        profiler = None
        try:
            # Match upstream: start training as soon as the buffer holds one
            # full window; no extra collection warmup or WM-only phase.
            required_steps = max(self.cfg.hist, self.cfg.wm_window)
            if self.buffer.size < required_steps:
                self.collect(required_steps - self.buffer.size)
            logger.log_buffer_fill(self.buffer.size, self.cfg.buffer_steps)
            logger.start_training_timer()

            log_path = self.log_dir / "train_log.csv"
            with log_path.open("a", encoding="utf-8", newline="") as log_file:
                if start_iteration == 0:
                    log_file.write(",".join(_LOG_FIELDS) + "\n")
                train_start = time.time() - self.wall_time
                log_start = time.time()
                step_accumulator = 0.0
                if self.cfg.proc_collect:
                    from .proc_collector import ProcCollectDriver

                    env_spec = self._collector_env_spec
                    if env_spec is None:  # pragma: no cover - validated in __init__
                        raise ValueError("proc_collect requires collector_env_spec")
                    worker = ProcCollectDriver(self, env_spec, asdict(self.cfg))
                    worker.start()
                elif self.cfg.async_collect:
                    worker = _CollectWorker(self)
                    worker.start()
                if worker is not None:
                    step_accumulator += 1.0 / self.cfg.utd
                    dispatch_steps = int(step_accumulator)
                    step_accumulator -= dispatch_steps
                    worker.dispatch(dispatch_steps)
                if profiler is None:
                    profiler = _maybe_start_profiler("learner")
                aggregate: dict[str, list[torch.Tensor]] = {
                    key: [] for key in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")
                }
                last_metrics: dict[str, float | int] = {}
                for iteration in range(start_iteration, self.cfg.iterations):
                    if self.cfg.compile and iteration == start_iteration:
                        logger.log_status("Compiling world model (one-off, slow on first run)")
                    exam_rows_before = self._exam_rows
                    self._advance_exam_schedule(iteration)
                    if self._exam_rows != exam_rows_before:
                        logger.log_status(
                            f"Exams active: {self._exam_rows}/{self.env.num_envs} worlds, "
                            f"imagination exam share {self._exam_share:.3f}"
                        )
                    iteration_start = time.time()
                    learning_rate = self.lr_at(iteration)
                    for optimizer in (self.opt_wm, self.opt_pi, self.opt_v):
                        for group in optimizer.param_groups:
                            current_lr = group["lr"]
                            if isinstance(current_lr, torch.Tensor):
                                current_lr.fill_(learning_rate)
                            else:
                                group["lr"] = learning_rate
                    stats_mark = len(self.stats["r"])
                    join_wait = 0.0
                    if worker is None:
                        step_accumulator += 1.0 / self.cfg.utd
                        collect_steps = int(step_accumulator)
                        step_accumulator -= collect_steps
                        self.collect(collect_steps)
                    else:
                        # This iteration's steps were dispatched one iteration
                        # ahead and overlap the WM updates below.
                        collect_steps = worker.pending_steps
                    train_phase_start = time.time()
                    pi_thread: threading.Thread | None = None
                    if self.cfg.three_way and iteration > start_iteration:
                        self._pi_metrics = None
                        self._pi_error = None
                        pi_thread = threading.Thread(
                            target=self._run_policy_shadow,
                            daemon=True,
                            name="supermimic-policy",
                        )
                        pi_thread.start()
                    wm_loss = torch.zeros((), device=self.device)
                    wm_error = torch.zeros((), device=self.device)
                    for _ in range(self.cfg.wm_batches):
                        wm_loss, wm_error = self.wm_update()
                        for name, metric in (("wm_loss", wm_loss), ("wm_err8", wm_error)):
                            if not torch.isfinite(metric).all():
                                raise FloatingPointError(
                                    f"non-finite {name} at iteration {iteration + 1}"
                                )
                    aggregate["wm"].append(wm_loss)
                    aggregate["wm8"].append(wm_error)
                    pi_loss: float | None = None
                    v_loss: float | None = None
                    imag_rew: float | None = None
                    pi_gradnorm: float | None = None
                    imag_survival: float | None = None
                    if pi_thread is not None:
                        pi_thread.join()
                        if self._pi_error is not None:
                            raise RuntimeError("policy shadow worker failed") from self._pi_error
                        assert self._pi_metrics is not None
                        (
                            policy_loss,
                            value_loss,
                            imagined_reward,
                            grad_norm,
                            survival,
                        ) = self._pi_metrics
                    else:
                        policy_loss, value_loss, imagined_reward, grad_norm, survival = (
                            self.policy_update()
                        )
                    if worker is not None:
                        join_start = time.time()
                        worker.harvest()
                        join_wait = time.time() - join_start
                    if self.cfg.three_way and self.wm_shadow is not None:
                        self.wm_shadow.load_state_dict(self.wm.state_dict())
                    if worker is not None and iteration + 1 < self.cfg.iterations:
                        step_accumulator += 1.0 / self.cfg.utd
                        dispatch_steps = int(step_accumulator)
                        step_accumulator -= dispatch_steps
                        worker.dispatch(dispatch_steps)
                    # Fail fast on divergence instead of checkpointing a diverged model.
                    # ``grad_norm`` is deliberately excluded: ``no_sync`` training already
                    # handles a non-finite gradient by zeroing it and stepping, so a non-finite
                    # gradient is not a divergence signal on its own.
                    for name, metric in (
                        ("pi_loss", policy_loss),
                        ("v_loss", value_loss),
                        ("imag_rew", imagined_reward),
                    ):
                        if not torch.isfinite(metric).all():
                            raise FloatingPointError(
                                f"non-finite {name} at iteration {iteration + 1}"
                            )
                    aggregate["pi"].append(policy_loss)
                    aggregate["v"].append(value_loss)
                    aggregate["ir"].append(imagined_reward)
                    aggregate["gn"].append(grad_norm)
                    aggregate["surv"].append(survival)
                    pi_loss = float(policy_loss)
                    v_loss = float(value_loss)
                    imag_rew = float(imagined_reward)
                    pi_gradnorm = float(grad_norm)
                    imag_survival = float(survival)
                    train_time = time.time() - train_phase_start
                    self.iteration = iteration + 1
                    self.wall_time = time.time() - train_start
                    iteration_time = time.time() - iteration_start

                    iter_rewards = self.stats["r"][stats_mark:]
                    mean_reward = self._mean(iter_rewards) if iter_rewards else None
                    ep_len = self._mean(self.stats["ep_len"]) if self.stats["ep_len"] else None
                    ep_ret_norm = self._mean(self.stats["ep_ret"]) if self.stats["ep_ret"] else None
                    total_done = sum(float(value) for value in self.stats["done"])
                    term_frac = (
                        sum(float(value) for value in self.stats["term"]) / total_done
                        if total_done > 0
                        else None
                    )
                    exam_survival = (
                        self._mean(self.stats["exam_surv"]) if self.stats["exam_surv"] else None
                    )
                    logger.log_step(
                        self.iteration,
                        wm_loss=float(wm_loss),
                        wm_err8=float(wm_error),
                        pi_loss=pi_loss,
                        v_loss=v_loss,
                        imag_rew=imag_rew,
                        imag_survival=imag_survival,
                        pi_gradnorm=pi_gradnorm,
                        lr=learning_rate,
                        sim_rew=mean_reward,
                        ep_len=ep_len,
                        ep_ret_norm=ep_ret_norm,
                        term_frac=term_frac,
                        exam_survival=exam_survival,
                        exam_fail_ema=self._exam_fail_ema,
                        iter_sim_steps=collect_steps * self.env.num_envs,
                        collect_time=(
                            join_wait if worker is not None else train_phase_start - iteration_start
                        ),
                        learn_time=train_time,
                        iteration_time=iteration_time,
                    )

                    if self.iteration % self.cfg.log_every == 0:
                        iterations_per_second = self.cfg.log_every / (time.time() - log_start)
                        log_start = time.time()
                        term_frac = float(
                            sum(float(value) for value in self.stats["term"])
                            / max(sum(float(value) for value in self.stats["done"]), 1)
                        )
                        row = [
                            self.iteration,
                            self.wall_time / 3600,
                            self.sim_steps,
                            self._mean(aggregate["wm"]),
                            self._mean(aggregate["wm8"]),
                            self._mean(aggregate["pi"]),
                            self._mean(aggregate["v"]),
                            self._mean(aggregate["ir"]),
                            self._mean(aggregate["gn"]),
                            self._mean(aggregate["surv"]),
                            self._mean(self.stats["r"]),
                            self._mean(self.stats["ep_len"]),
                            self._mean(self.stats["ep_ret"]),
                            term_frac,
                            iterations_per_second,
                            learning_rate,
                        ]
                        log_file.write(
                            ",".join(
                                f"{value:.6g}" if isinstance(value, float) else str(value)
                                for value in row
                            )
                            + "\n"
                        )
                        log_file.flush()
                        last_metrics = dict(zip(_LOG_FIELDS, row, strict=True))
                        self.stats = _empty_stats()
                        aggregate = {key: [] for key in aggregate}

                    if (
                        self.iteration % self.cfg.ckpt_every == 0
                        or self.iteration == self.cfg.iterations
                    ):
                        self.wall_time = time.time() - train_start
                        self.save()
                        numbered = self.save(
                            self.log_dir / f"ckpt_{self.iteration:06d}.pt",
                            include_optimizers=False,
                        )
                        logger.log_save(str(numbered))

            self.wall_time = time.time() - train_start
            logger.finish()
            return {
                "completed": self.iteration >= self.cfg.iterations,
                "iteration": self.iteration,
                "sim_steps": self.sim_steps,
                "wall_time": self.wall_time,
                **last_metrics,
            }
        finally:
            if worker is not None:
                worker.shutdown()
            if profiler is not None:
                profiler.__exit__(None, None, None)
            logger.close()


__all__ = ["SuperMimicConfig", "SuperMimicLearner"]
