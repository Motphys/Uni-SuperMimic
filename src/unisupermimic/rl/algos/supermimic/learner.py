"""SuperMimic training over the injected environment and MDP protocols."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

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
from .cuda_graph import CapturedUpdate, cumprod_without_host_sync
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
    # How the ``critic_epochs x critic_minibatches`` loop of ``policy_update`` runs.
    # ``eager`` steps every minibatch from Python (the original path). ``graph`` captures
    # the whole loop (forward, backward, gradient clipping, Adam step) into one CUDA graph
    # on the first use and replays it once per iteration; the math is the same eager ops on
    # static buffers, with the permutations still drawn outside the graph. ``auto`` (the
    # default) picks ``graph`` on NVIDIA CUDA and ``eager`` anywhere else (CPU, MPS,
    # ROCm), where CUDA graphs are not available.
    critic_update: str = "auto"
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
    test_hidden_dims: tuple[int, ...] | None = None
    # Host launch-overhead reductions. None of them changes the update math. The
    # tri-state switches read ``None`` (the default) as "on for NVIDIA CUDA, off
    # anywhere else"; ``True`` requires NVIDIA CUDA and ``False`` forces the original
    # path. The training loop is launch-bound on fast GPUs, so by default the host
    # work is moved out of the way wherever the device supports it.
    # ``wm_graph`` / ``policy_graph`` replay the gradient step of ``wm_update`` / of
    # the policy in ``policy_update`` (imagination rollout + objective + backward) from
    # an explicit CUDA graph with persistent input buffers. Gradient clipping and the
    # optimizer step stay eager.
    wm_graph: bool | None = None
    policy_graph: bool | None = None
    # ``stats_on_device`` keeps the collect-side logging statistics (sim_rew, ep_len,
    # ep_ret_norm, term_frac, exam_survival) on the device: the synced path's one
    # ``float(done.sum())`` per collect step (plus two more when an episode ends)
    # stalls the host until the env pipeline has drained. On, the statistics accumulate
    # in float64 on the device and are read back once per ``log_every`` period;
    # ``train_log.csv`` keeps its definitions, while the matching TensorBoard scalars
    # get one point per log period instead of per iteration.
    stats_on_device: bool | None = None
    # ``inductor_cpp_wrapper`` makes torch.compile emit its wrapper code in C++ instead
    # of Python, so the Triton kernels of the compiled forward/backward are launched
    # without going through the Python launcher. Opt-in: it costs a slow C++ compile on
    # every cold cache, and with the graphs above the gain is small on a fast host.
    inductor_cpp_wrapper: bool = False


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


# Slots of the device-side statistics vector (``cfg.stats_on_device``): running
# float64 sums over the collect steps of the current log period.
_DEV_STAT_FIELDS = (
    "steps",  # collect steps
    "r_sum",  # sum of mean(r) / max_step_reward
    "done_sum",  # sum of sum(done)
    "term_sum",  # sum of sum(term)
    "ep_steps",  # steps with at least one episode end
    "ep_len_sum",  # over those steps: sum of sum(ep_len) / done_count
    "ep_ret_sum",  # over those steps: sum of sum(ep_ret_norm) / done_count
    "exam_steps",  # collect steps taken while exam worlds were active
    "exam_surv_sum",  # over those steps: sum of the exam survival share
)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@dataclass
class _CriticGraph:
    """One captured critic training loop and the static buffers it was captured on."""

    key: tuple[tuple[int, ...], ...]
    graph: "torch.cuda.CUDAGraph"
    input: torch.Tensor
    targets: torch.Tensor
    permutations: torch.Tensor
    loss: torch.Tensor
    # Storages the capture is bound to (critic parameters and buffers, Adam state).
    # A replay is only valid while every one of them is still the same tensor object.
    tensors: list[torch.Tensor]


class SuperMimicLearner:
    def __init__(
        self,
        env: EnvProtocol,
        cfg: SuperMimicConfig,
        log_dir: str | Path,
        device: str | torch.device | None = None,
        backend: str = "",
    ):
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        if cfg.utd <= 0:
            raise ValueError("utd must be positive")
        if cfg.pi_opt not in ("adam", "muon"):
            raise ValueError(f"pi_opt must be 'adam' or 'muon', got {cfg.pi_opt!r}")
        if cfg.wm_opt not in ("adam", "muon"):
            raise ValueError(f"wm_opt must be 'adam' or 'muon', got {cfg.wm_opt!r}")
        if cfg.critic_update not in ("auto", "eager", "graph"):
            raise ValueError(
                f"critic_update must be 'auto', 'eager' or 'graph', got {cfg.critic_update!r}"
            )
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
        # Explicit CUDA graphs only exist on NVIDIA CUDA (not on ROCm's HIP). The
        # launch-overhead switches default to ``None`` / ``"auto"`` and follow this;
        # an explicit request on any other device is a configuration error.
        self._cuda_graphs = (
            self.device.type == "cuda" and getattr(torch.version, "hip", None) is None
        )

        def follow_device(requested: bool | None, name: str) -> bool:
            if requested is None:
                return self._cuda_graphs
            if requested and not self._cuda_graphs:
                raise ValueError(
                    f"SuperMimicConfig.{name}=True requires a CUDA-graph capable device "
                    f"(NVIDIA CUDA), got {self.device}"
                )
            return requested

        if cfg.critic_update == "graph" and not self._cuda_graphs:
            raise ValueError(
                "SuperMimicConfig.critic_update='graph' requires an NVIDIA CUDA device, "
                f"got {self.device}"
            )
        self._wm_graph_enabled = follow_device(cfg.wm_graph, "wm_graph")
        self._pi_graph_enabled = follow_device(cfg.policy_graph, "policy_graph")
        # The device-side statistics run on any device (host bridges move the transition to
        # the learner device first); ``auto`` follows NVIDIA CUDA, where the per-step sync
        # they remove actually stalls the host.
        self._stats_on_device = (
            self._cuda_graphs if cfg.stats_on_device is None else cfg.stats_on_device
        )
        if cfg.inductor_cpp_wrapper and self.device.type != "cuda":
            raise ValueError(
                "SuperMimicConfig.inductor_cpp_wrapper=True requires a CUDA device, "
                f"got {self.device}"
            )
        if cfg.inductor_cpp_wrapper:
            # Process-wide inductor switch; it has to be set before the first compile.
            import torch._inductor.config as inductor_config

            inductor_config.cpp_wrapper = True
        mdp = get_supermimic_mdp(env)
        self.mdp = mdp.to(self.device) if self._bridged else mdp

        # TF32 only pays off for fp32 (amp="none") training: under bf16 autocast
        # the heavy GEMMs already run in bf16, while on ROCm a TF32 bmm over the
        # misaligned strided views produced by unflatten_state triggers per-call
        # rocBLAS kernel selection costing ~100 ms per call.
        torch.backends.cuda.matmul.allow_tf32 = cfg.amp == "none"
        torch.backends.cudnn.allow_tf32 = True
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
        self._critic_params = list(self.critic.parameters())
        self._critic_buffers = list(self.critic.buffers())
        # CUDA graphs only exist on NVIDIA CUDA; anywhere else ``auto`` / ``graph`` resolve to
        # the eager loop (``graph`` off CUDA is rejected above), so the training semantics are
        # the same on every device.
        self._critic_graph_enabled = (
            cfg.critic_update in ("auto", "graph") and not cfg.no_critic and self._cuda_graphs
        )
        self._critic_graph_state: _CriticGraph | None = None
        if self._critic_graph_enabled:
            # A captured graph freezes every Python float it reads, but ``train()`` re-binds
            # a float lr on the param group each iteration (cosine decay). The capturable
            # fused Adam takes the learning rate from a device tensor instead, which
            # ``_sync_critic_lr`` refreshes between replays. It is the same fused kernel, so
            # the updates stay bit-identical to the float-lr path.
            self._v_lr = torch.tensor(cfg.lr, dtype=torch.float32, device=self.device)
            self.opt_v = torch.optim.Adam(
                self._critic_params, lr=self._v_lr, fused=True, capturable=True
            )
        else:
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
        # Device-side collect statistics (``stats_on_device``), see ``_accumulate_stats``.
        self._dev_stats = None
        self._dev_unit = None
        if self._stats_on_device:
            self._dev_stats = torch.zeros(
                len(_DEV_STAT_FIELDS), dtype=torch.float64, device=self.device
            )
            # Scalar constants (one, zero) for the per-step row, without a host->device copy.
            self._dev_unit = torch.tensor([1.0, 0.0], dtype=torch.float64, device=self.device)
        self._wm_rollout_fn = self._compile(self._wm_rollout)
        self._wm_err_fn = self._compile(self._wm_err)
        self._imag_step_fn = self._compile(self._imag_step)
        self._reward_term_fn = self._compile(self._reward_term)
        self._critic_input_fn = self._compile(self._critic_input)
        # Explicit CUDA graphs over the gradient steps. The policy graph clones the outputs
        # the training loop keeps past the next replay.
        self._wm_graph = CapturedUpdate() if self._wm_graph_enabled else None
        self._pi_graph = CapturedUpdate(clone_outputs=(0, 4, 7)) if self._pi_graph_enabled else None

        with (self.log_dir / "args.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(cfg), stream, indent=2)

    def _compile(self, function):
        if not self.cfg.compile:
            return function
        if self.device.type == "mps":
            # Inductor Metal codegen is not usable on MPS; keep AOT graph
            # capture without kernel generation.
            return torch.compile(function, backend="aot_eager")
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
            predictions.append(state)
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
        state: State,
        obs_hist: torch.Tensor,
        previous_action: torch.Tensor,
        frames: torch.Tensor,
        context: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[State, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = self.pi(obs_hist, goal)
        next_state = self.wm.step(state, action, normalized_context)
        next_frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, context, next_frames))
        next_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        next_goal = self.mdp.goal(next_state, next_frames, self.cfg.goal_frames)
        return next_state, action, next_hist, next_frames, next_goal

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
            return reward, done
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
            return self._critic_input_fn(state, frames, normalized_context)
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
            transition = step_state.info[SUPERMIMIC_TRANSITION_KEY]
            if self._bridged:
                transition = {key: value.to(self.device) for key, value in transition.items()}
            self.buffer.add(transition)
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
            self.sim_steps += self.env.num_envs
            if self._dev_stats is not None:
                # The same statistics as the appends below, accumulated on the device:
                # no ``float()`` and no host branch, so the host runs on without
                # waiting for the env pipeline to drain.
                self._accumulate_stats(transition)
                continue
            self.stats["r"].append(transition["r"].mean() / self.mdp.max_step_reward)
            if self._exam_rows:
                # Per-step survival of the exam worlds: the share whose danger
                # probability stays under the termination band. These episodes
                # never hard-terminate, so this is the observable for whether
                # the exams are improving.
                rows = slice(self.env.num_envs - self._exam_rows, None)
                danger = self.mdp.soft_termination(
                    unflatten_state(transition["s1"][rows]), transition["f"][rows] + 1
                )
                self.stats["exam_surv"].append((danger < 0.5).float().mean())
            if "ep_len" in transition:
                # Episode statistics are full-width and masked (non-done rows
                # carry zero), so aggregate with the done count. Exam episodes
                # stay out of them: their only end is the clip end.
                done = transition["done"]
                term = transition["term"]
                if self._exam_rows:
                    done = done[: self.env.num_envs - self._exam_rows]
                    term = term[: self.env.num_envs - self._exam_rows]
                done_count = float(done.sum())
                if done_count > 0.0:
                    self.stats["ep_len"].append(float(transition["ep_len"].sum()) / done_count)
                    self.stats["ep_ret"].append(float(transition["ep_ret_norm"].sum()) / done_count)
                self.stats["term"].append(term.float().sum())
                self.stats["done"].append(done.float().sum())

    def _wm_loss_grads(self, window: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward, loss and backward of one window (eager and graph-replayed alike)."""
        predicted = self._wm_rollout_fn(window["s"], window["a"], window["xi"])
        loss, last_error = self._wm_err_fn(predicted, window["s1"], window["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        return loss, last_error

    def wm_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window)
        graph = self._wm_graph
        if graph is None:
            loss, last_error = self._wm_loss_grads(window)
        else:
            loss, last_error = graph.step(self._wm_loss_grads, window)
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

    def _pi_loss_grads(
        self,
        start: dict[str, torch.Tensor],
        history: dict[str, torch.Tensor],
        normalized_context: torch.Tensor,
        v_mean: torch.Tensor,
        v_std: torch.Tensor,
        set_to_none: bool,
        capturable: bool,
    ) -> tuple[Any, ...]:
        """Imagination rollout, policy objective and its backward.

        Shared by the eager path and the ``policy_graph`` replay, so the captured
        computation is the eager one by construction. ``v_mean`` / ``v_std`` are
        arguments because the graph has to read them from persistent buffers, and
        ``capturable`` swaps the survival cumprod for the variant whose backward can be
        captured (ATen's synchronizes with the host).
        """
        state = unflatten_state(start["s"])
        context, frames, previous_action = start["xi"], start["f"], start["a_prev"]
        with torch.no_grad():
            obs_hist = self.mdp.obs(
                unflatten_state(history["s"]),
                history["a_prev"],
                history["xi"],
                history["f"],
            )
        obs_hist = self.mdp.add_noise(obs_hist)

        batch_size = self.cfg.pi_batch
        goal = self.mdp.goal(state, frames, self.cfg.goal_frames)
        initial_state = state
        states, actions, previous_actions, all_frames = [], [], [], []
        for _ in range(self.cfg.horizon):
            previous_actions.append(previous_action)
            state, previous_action, obs_hist, frames, goal = self._imag_step_fn(
                state,
                obs_hist,
                previous_action,
                frames,
                context,
                normalized_context,
                goal,
            )
            states.append(state)
            actions.append(previous_action)
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
        alive = cumprod_without_host_sync(1 - dones) if capturable else torch.cumprod(1 - dones, 0)
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
                bootstrap * v_std + v_mean
            )
        loss = -objective.mean()
        self.opt_pi.zero_grad(set_to_none=set_to_none)
        loss.backward()
        return (
            loss,
            initial_state,
            states,
            all_frames,
            reward_sum,
            detached_rewards,
            detached_dones,
            survival,
        )

    def _pi_graph_body(self, inputs: dict[str, Any]) -> tuple[Any, ...]:
        # ``set_to_none=False`` keeps every gradient at the address the captured graph
        # writes to; the eager clip and optimizer step outside the graph read it.
        return self._pi_loss_grads(
            inputs["start"],
            inputs["history"],
            inputs["normalized_context"],
            inputs["v_mean"],
            inputs["v_std"],
            set_to_none=False,
            capturable=True,
        )

    def policy_update(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start, history = self._sample_valid_starts(self.cfg.pi_batch)
        normalized_context = self.mdp.normalize_context(start["xi"])
        batch_size = self.cfg.pi_batch
        _set_requires_grad(self.wm, False)
        _set_requires_grad(self.critic, False)
        graph = self._pi_graph
        try:
            if graph is None:
                (
                    loss,
                    initial_state,
                    states,
                    all_frames,
                    reward_sum,
                    detached_rewards,
                    detached_dones,
                    survival,
                ) = self._pi_loss_grads(
                    start,
                    history,
                    normalized_context,
                    self._v_mean,
                    self._v_std,
                    set_to_none=True,
                    capturable=False,
                )
            else:
                # ``_v_mean`` / ``_v_std`` are rebound every iteration by the critic
                # targets below; copying them into the graph's input buffers is what
                # keeps the replayed objective reading the current values.
                (
                    loss,
                    initial_state,
                    states,
                    all_frames,
                    reward_sum,
                    detached_rewards,
                    detached_dones,
                    survival,
                ) = graph.step(
                    self._pi_graph_body,
                    {
                        "start": start,
                        "history": history,
                        "normalized_context": normalized_context,
                        "v_mean": self._v_mean,
                        "v_std": self._v_std,
                    },
                )
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
            _set_requires_grad(self.wm, True)
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
                critic_input = self._batched_critic_input(
                    critic_states, critic_frames, critic_context
                ).view(self.cfg.horizon + 1, batch_size, -1)
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
                    self._v_mean = target_mean.detach()
                    self._v_std = target_std.detach()
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
            if self._critic_graph_enabled:
                value_loss = self._critic_train_graph(flat_input, flat_targets, num_samples)
            else:
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

    def _sync_critic_lr(self) -> None:
        """Push the param group's learning rate into the device tensor the graph reads.

        ``train()`` re-binds a Python float on the group every iteration (cosine decay) and a
        resume re-binds the checkpoint's value, so the tensor is refreshed from the group
        whenever the two went apart, and becomes the group's lr again.
        """
        group = self.opt_v.param_groups[0]
        if group["lr"] is not self._v_lr:
            if isinstance(group["lr"], torch.Tensor):
                self._v_lr.copy_(group["lr"])
            else:
                self._v_lr.fill_(group["lr"])
            group["lr"] = self._v_lr

    def _critic_graph_tensors(self) -> list[torch.Tensor]:
        """Return the storages a captured critic graph is bound to."""
        tensors = self._critic_params + self._critic_buffers
        for parameter in self._critic_params:
            state = self.opt_v.state.get(parameter)
            if state:
                tensors.extend(state.values())
        return tensors

    def _critic_train_captured(
        self, flat_input: torch.Tensor, flat_targets: torch.Tensor, permutations: torch.Tensor
    ) -> torch.Tensor:
        """Run the minibatch loop on static buffers; this is what the CUDA graph records.

        It mirrors the eager loop step by step, with two substitutions that leave the math
        bit-identical:

        * ``zero_grad(set_to_none=True)`` becomes an in-place zeroing: a captured graph
          cannot drop and re-allocate its gradients, so they are persistent buffers that
          backward accumulates into exactly as it would into fresh ones.
        * The input normalizer is frozen for the whole loop, and autocast casts the same
          minibatch to the same dtype in every epoch. The eager loop re-normalizes and
          re-casts each sample once per epoch; here every sample is normalized (and cast)
          once per iteration, right after the gather. Both are elementwise per sample, so
          every minibatch sees exactly the values ``self.critic`` would produce, with a
          fraction of the memory traffic (the loop is bandwidth-bound on its [16384, 834]
          input).
        """
        minibatch_size = flat_input.shape[0] // self.cfg.critic_minibatches
        normalized = self.critic.norm(flat_input)
        if networks.AMP_DTYPE is not None:
            normalized = normalized.to(networks.AMP_DTYPE)
        value_loss = torch.zeros((), device=self.device)
        for permutation in permutations:
            for index in range(self.cfg.critic_minibatches):
                indices = permutation[index * minibatch_size : (index + 1) * minibatch_size]
                prediction = networks.run_mlp(self.critic.net, normalized[indices]).squeeze(-1)
                value_loss = ((prediction - flat_targets[indices]) ** 2).mean()
                for parameter in self._critic_params:
                    parameter.grad.zero_()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self._critic_params, 1.0)
                self.opt_v.step()
        return value_loss.detach()

    def _capture_critic_graph(
        self, flat_input: torch.Tensor, flat_targets: torch.Tensor, permutations: torch.Tensor
    ) -> _CriticGraph:
        """Warm up on a side stream, then capture the whole loop into one CUDA graph.

        Warm-up steps execute for real (they create cuBLAS workspaces and the lazily
        allocated Adam state), so the critic parameters, its normalizer buffers, and the
        optimizer state are snapshotted first and restored in place afterwards; state the
        warm-up created is reset to the zeros a fresh Adam starts from. Nothing here touches
        the RNG.
        """
        params, buffers = self._critic_params, self._critic_buffers
        for parameter in params:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        live = params + buffers
        saved = [tensor.detach().clone() for tensor in live]
        saved_state = [
            {name: value.detach().clone() for name, value in self.opt_v.state.get(p, {}).items()}
            for p in params
        ]
        state = _CriticGraph(
            key=(
                tuple(flat_input.shape),
                tuple(flat_targets.shape),
                tuple(permutations.shape),
            ),
            graph=torch.cuda.CUDAGraph(),
            input=flat_input.clone(),
            targets=flat_targets.clone(),
            permutations=permutations.clone(),
            loss=torch.zeros((), device=self.device),
            tensors=[],
        )
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    state.loss = self._critic_train_captured(
                        state.input, state.targets, state.permutations
                    )
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(state.graph):
                state.loss = self._critic_train_captured(
                    state.input, state.targets, state.permutations
                )
        finally:
            with torch.no_grad():
                for tensor, snapshot in zip(live, saved, strict=True):
                    tensor.copy_(snapshot)
                for parameter, snapshot in zip(params, saved_state, strict=True):
                    for name, value in self.opt_v.state.get(parameter, {}).items():
                        if name in snapshot:
                            value.copy_(snapshot[name])
                        else:
                            value.zero_()
        state.tensors = self._critic_graph_tensors()
        return state

    def _critic_train_graph(
        self,
        flat_input: torch.Tensor,
        flat_targets: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Draw the permutations like the eager loop and replay the captured update."""
        # Same calls with the same shapes as the eager loop, so the global generator is
        # consumed identically and a seeded run sees the same minibatches. The captured
        # graph itself draws no random numbers.
        permutations = torch.stack(
            [torch.randperm(num_samples, device=self.device) for _ in range(self.cfg.critic_epochs)]
        )
        key = (
            tuple(flat_input.shape),
            tuple(flat_targets.shape),
            tuple(permutations.shape),
        )
        # Before any capture: the graph must record the fused Adam reading the lr tensor, so
        # the tensor has to be the group's lr by then (a float on the group would be baked in).
        self._sync_critic_lr()
        tensors = self._critic_graph_tensors()
        state = self._critic_graph_state
        if (
            state is None
            or state.key != key
            or len(tensors) != len(state.tensors)
            or any(a is not b for a, b in zip(tensors, state.tensors))
        ):
            state = self._critic_graph_state = self._capture_critic_graph(
                flat_input, flat_targets, permutations
            )
        state.input.copy_(flat_input)
        state.targets.copy_(flat_targets)
        state.permutations.copy_(permutations)
        state.graph.replay()
        # Every replay rewrites this buffer, while ``train()`` keeps the loss until the next
        # log point: hand out a private copy.
        return state.loss.clone()

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

    def _accumulate_stats(self, transition: dict[str, torch.Tensor]) -> None:
        """Add one collect step's logging statistics to the device-side sums.

        Device-side counterpart of the ``self.stats`` appends in ``collect``: every
        per-step value is still an fp32 reduction (as the synced path computes it)
        that only then turns into float64, and the accumulator is updated on the
        current stream right after the step, so stream order pairs each value with
        its step. A step without an episode end contributes zero to ``ep_len`` and
        ``ep_ret_norm`` through ``torch.where`` instead of a host branch.
        """
        if self._dev_stats is None or self._dev_unit is None:
            raise RuntimeError("device-side statistics require cfg.stats_on_device")
        float64 = torch.float64
        one, zero = self._dev_unit[0], self._dev_unit[1]
        row = [one, (transition["r"].mean() / self.mdp.max_step_reward).to(float64)]
        row += [zero] * 5
        if self._exam_rows:
            # Per-step survival of the exam worlds: the share whose danger
            # probability stays under the termination band. These episodes
            # never hard-terminate, so this is the observable for whether
            # the exams are improving.
            rows = slice(self.env.num_envs - self._exam_rows, None)
            danger = self.mdp.soft_termination(
                unflatten_state(transition["s1"][rows]), transition["f"][rows] + 1
            )
            row += [one, (danger < 0.5).float().mean().to(float64)]
        else:
            row += [zero, zero]
        if "ep_len" in transition:
            # Episode statistics are full-width and masked (non-done rows
            # carry zero), so aggregate with the done count. Exam episodes
            # stay out of them: their only end is the clip end.
            done = transition["done"]
            term = transition["term"]
            if self._exam_rows:
                done = done[: self.env.num_envs - self._exam_rows]
                term = term[: self.env.num_envs - self._exam_rows]
            done_count = done.float().sum()
            ended = done_count > 0.0
            divisor = torch.where(ended, done_count.to(float64), one)
            # ``ep_len`` / ``ep_ret_norm`` are full-width and summed uncut (as in the
            # synced path), so a step with no counted end must drop out entirely.
            row[2:7] = [
                done_count.to(float64),
                term.float().sum().to(float64),
                ended.to(float64),
                torch.where(ended, transition["ep_len"].sum().to(float64) / divisor, zero),
                torch.where(ended, transition["ep_ret_norm"].sum().to(float64) / divisor, zero),
            ]
        self._dev_stats.add_(torch.stack(row))

    def _read_dev_stats(self) -> dict[str, float]:
        """Read the period's collect statistics back once and reset the sums.

        The single host read of the device-side statistics (one small D2H copy per
        log period). The results follow the ``train_log.csv`` definitions: means
        over the steps of the period (unweighted for the per-step episode means;
        0.0 when no step qualified), and the terminated share of all episode ends.
        ``exam_survival`` is present only when exam worlds collected in the period.
        """
        if self._dev_stats is None:
            raise RuntimeError("device-side statistics require cfg.stats_on_device")
        sums = dict(zip(_DEV_STAT_FIELDS, self._dev_stats.tolist(), strict=True))
        self._dev_stats.zero_()
        stats = {
            "sim_rew": sums["r_sum"] / max(sums["steps"], 1),
            "ep_len": sums["ep_len_sum"] / max(sums["ep_steps"], 1),
            "ep_ret_norm": sums["ep_ret_sum"] / max(sums["ep_steps"], 1),
            "term_frac": sums["term_sum"] / max(sums["done_sum"], 1),
        }
        if sums["exam_steps"] > 0:
            stats["exam_survival"] = sums["exam_surv_sum"] / max(sums["exam_steps"], 1)
        return stats

    def train(self, resume: str | Path | None = None) -> dict[str, float | int | bool]:
        if resume is not None:
            self.load(resume)
        start_iteration = self.iteration
        self.stats = _empty_stats()
        if self._dev_stats is not None:
            # Device-side statistics start a fresh period as well (resume included).
            self._dev_stats.zero_()
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
                aggregate: dict[str, list[torch.Tensor]] = {
                    key: [] for key in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")
                }
                last_metrics: dict[str, float | int] = {}
                # Collect statistics handed to the logger; ``period`` is the closing
                # log period's row when this iteration writes one (device-side path).
                period: dict[str, float] | None
                mean_reward: float | None
                ep_len: float | None
                ep_ret_norm: float | None
                term_frac: float | None
                exam_survival: float | None
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
                            group["lr"] = learning_rate
                    step_accumulator += 1.0 / self.cfg.utd
                    collect_steps = int(step_accumulator)
                    step_accumulator -= collect_steps
                    stats_mark = len(self.stats["r"])
                    self.collect(collect_steps)
                    train_phase_start = time.time()
                    wm_loss = torch.zeros((), device=self.device)
                    wm_error = torch.zeros((), device=self.device)
                    # The finite verdicts stay on the device: reading each batch's metrics
                    # here would drain the GPU queue after every batch and leave the GPU
                    # waiting for the host. The first host read is the fail-fast check after
                    # the policy update, which syncs anyway. A non-finite batch cannot be
                    # masked by a later finite one, because the flags are only ever cleared.
                    wm_finite = torch.ones(2, dtype=torch.bool, device=self.device)
                    for _ in range(self.cfg.wm_batches):
                        wm_loss, wm_error = self.wm_update()
                        wm_finite &= torch.isfinite(torch.stack((wm_loss, wm_error)))
                    aggregate["wm"].append(wm_loss)
                    aggregate["wm8"].append(wm_error)
                    pi_loss: float | None = None
                    v_loss: float | None = None
                    imag_rew: float | None = None
                    pi_gradnorm: float | None = None
                    imag_survival: float | None = None
                    policy_loss, value_loss, imagined_reward, grad_norm, survival = (
                        self.policy_update()
                    )
                    # Fail fast on divergence instead of checkpointing a diverged model.
                    # ``grad_norm`` is deliberately excluded: ``no_sync`` training already
                    # handles a non-finite gradient by zeroing it and stepping, so a non-finite
                    # gradient is not a divergence signal on its own.
                    for name, finite in zip(
                        ("wm_loss", "wm_err8"), wm_finite.tolist(), strict=True
                    ):
                        if not finite:
                            raise FloatingPointError(
                                f"non-finite {name} at iteration {iteration + 1}"
                            )
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

                    period = None
                    if self._dev_stats is not None:
                        # Nothing is read per iteration: the collect statistics are
                        # read back once, here, when the log period closes, and the
                        # collect tags are logged at that iteration with the period's
                        # value (one point per log period instead of per iteration).
                        if self.iteration % self.cfg.log_every == 0:
                            period = self._read_dev_stats()
                        if period is None:
                            mean_reward = ep_len = ep_ret_norm = None
                            term_frac = exam_survival = None
                        else:
                            mean_reward = period["sim_rew"]
                            ep_len = period["ep_len"]
                            ep_ret_norm = period["ep_ret_norm"]
                            term_frac = period["term_frac"]
                            exam_survival = period.get("exam_survival")
                    else:
                        iter_rewards = self.stats["r"][stats_mark:]
                        mean_reward = self._mean(iter_rewards) if iter_rewards else None
                        ep_len = self._mean(self.stats["ep_len"]) if self.stats["ep_len"] else None
                        ep_ret_norm = (
                            self._mean(self.stats["ep_ret"]) if self.stats["ep_ret"] else None
                        )
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
                        collect_time=train_phase_start - iteration_start,
                        learn_time=train_time,
                        iteration_time=iteration_time,
                    )

                    if self.iteration % self.cfg.log_every == 0:
                        iterations_per_second = self.cfg.log_every / (time.time() - log_start)
                        log_start = time.time()
                        if period is not None:
                            # Device-side statistics: already read back once for this row.
                            collect_row = (
                                period["sim_rew"],
                                period["ep_len"],
                                period["ep_ret_norm"],
                                period["term_frac"],
                            )
                        else:
                            collect_row = (
                                self._mean(self.stats["r"]),
                                self._mean(self.stats["ep_len"]),
                                self._mean(self.stats["ep_ret"]),
                                float(
                                    sum(float(value) for value in self.stats["term"])
                                    / max(sum(float(value) for value in self.stats["done"]), 1)
                                ),
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
                            *collect_row,
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
            logger.close()


__all__ = ["SuperMimicConfig", "SuperMimicLearner"]
