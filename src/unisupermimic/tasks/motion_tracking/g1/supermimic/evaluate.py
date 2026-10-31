"""Deterministic full-clip evaluation and checkpoint watching for G1 SuperMimic."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    State,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.env import SUPPORTED_BACKENDS
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MAX_STEP_REWARD, MDP

if TYPE_CHECKING:
    from unisupermimic.tasks.motion_tracking.g1.supermimic.env import SuperMimicManagerBasedRlEnv

EVAL_CSV_FIELDS = (
    "iteration",
    "train_wall_h",
    "saved_at",
    "eval_at",
    "norm_return",
    "norm_return_framew",
    "n_failed",
    "failed",
)
DEFAULT_EVAL_NUM_ENVS = 16_384


def _resolve_policy_device(
    env_device: torch.device,
    device: str | torch.device | None,
    requested: str | torch.device,
) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if sys.platform == "darwin" and env_device.type == "cpu" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device(env_device if device is None else device)


def load_policy(
    env: SuperMimicManagerBasedRlEnv,
    checkpoint: dict,
    device: str | torch.device | None = None,
) -> networks.Policy:
    mdp = cast(MDP, env.supermimic_mdp)
    device = torch.device(env.device if device is None else device)
    args = checkpoint["args"]
    goal_frames = int(args["goal_frames"])
    hist = int(args["hist"])
    networks.AMP_DTYPE = torch.bfloat16 if args.get("amp", "none") == "bf16" else None
    hidden = args.get("test_hidden_dims") or args["policy_hidden"]
    policy: networks.Policy = networks.Policy(
        mdp.obs_dim,
        mdp.goal_frame_dim * goal_frames,
        hist=hist,
        z_dim=int(args["z_dim"]),
        num_joints=mdp.num_joints,
        hidden=tuple(hidden),
    ).to(device)
    policy.load_state_dict(checkpoint["pi"])
    policy.eval()
    return policy


@torch.inference_mode()
def evaluate(
    env: SuperMimicManagerBasedRlEnv,
    checkpoint: dict,
    device: str | torch.device | None = None,
    policy_device: str | torch.device = "auto",
) -> dict:
    """Evaluate every motion once from its first frame without observation/action noise."""
    mdp = cast(MDP, env.supermimic_mdp)
    motion = mdp.motion
    args = checkpoint["args"]
    goal_frames = int(args["goal_frames"])
    hist = int(args["hist"])
    inference_device = _resolve_policy_device(torch.device(env.device), device, policy_device)
    if inference_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS policy inference was requested but MPS is not available")
    policy = load_policy(env, checkpoint, inference_device)
    if inference_device.type == "mps":
        networks.AMP_DTYPE = None

    num_clips = len(motion.names)
    num_envs = int(env.num_envs)
    if num_envs > num_clips:
        raise ValueError(
            f"SuperMimic evaluation environment count must not exceed clips: "
            f"got {num_envs} environments for {num_clips} clips"
        )

    # Recycle finished environments. Starting the longest clips first makes the
    # final batch contain the shortest clips, so the fixed-size backend batch has
    # almost no idle tail after the last pending clip has been scheduled.
    clip_order = motion.clip_len.argsort(descending=True, stable=True)
    ordered_starts = motion.clip_start[clip_order].clone()
    _, reset_info = env.reset_to_frames(ordered_starts[:num_envs].clone())
    current = reset_info[SUPERMIMIC_STATE_KEY]
    returns = torch.zeros(num_clips, device=env.device)
    steps = torch.zeros(num_clips, dtype=torch.int64, device=env.device)
    failed = torch.zeros(num_clips, dtype=torch.bool, device=env.device)
    assigned = clip_order[:num_envs].to(env.device)
    next_clip = num_envs
    history = None

    def reset_history(info: dict[str, torch.Tensor], env_ids: torch.Tensor) -> None:
        assert history is not None
        state = cast(State, tuple(value[env_ids] for value in unflatten_state(info["s"])))
        obs = mdp.obs(
            state,
            info["a_prev"][env_ids],
            info["xi"][env_ids],
            info["f"][env_ids],
        )
        policy_env_ids = env_ids.to(inference_device)
        history[policy_env_ids] = obs.unsqueeze(1).expand(-1, hist, -1).to(inference_device)

    while bool((assigned >= 0).any()):
        obs = mdp.obs(
            unflatten_state(current["s"]),
            current["a_prev"],
            current["xi"],
            current["f"],
        ).to(inference_device)
        if history is None:
            history = obs.unsqueeze(1).repeat(1, hist, 1)
        else:
            history = torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1)
        goal = mdp.goal(current["f"], goal_frames).to(inference_device)
        action = policy(history, goal)
        state = env.step(action.to(env.device))
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        active = assigned >= 0
        active_ids = active.nonzero(as_tuple=False).flatten()
        active_clips = assigned[active_ids]
        returns[active_clips] += transition["r"][active_ids]
        steps[active_clips] += 1
        failed[active_clips] |= transition["term"][active_ids]
        current = state.info[SUPERMIMIC_STATE_KEY]

        completed = (transition["done"] & active).nonzero(as_tuple=False).flatten()
        reset_count = min(completed.numel(), num_clips - next_clip)
        if reset_count:
            reset_ids = completed[:reset_count]
            reset_clips = clip_order[next_clip : next_clip + reset_count].to(assigned.device)
            _, reset_info = env.reset_to_frames(
                ordered_starts[next_clip : next_clip + reset_count],
                env_ids=reset_ids,
            )
            current = reset_info[SUPERMIMIC_STATE_KEY]
            reset_history(current, reset_ids)
            assigned[reset_ids] = reset_clips
            next_clip += reset_count
        if reset_count < completed.numel():
            assigned[completed[reset_count:]] = -1

    max_returns = (motion.clip_len - 1).float() * MAX_STEP_REWARD
    normalized = returns / max_returns
    failed_indices = failed.nonzero(as_tuple=False).flatten().tolist()
    return {
        "norm_return": normalized.mean().item(),
        "norm_return_framew": (returns.sum() / max_returns.sum()).item(),
        "n_failed": len(failed_indices),
        "failed": [motion.names[index] for index in failed_indices],
        "per_clip": {
            name: {
                "norm_return": normalized[index].item(),
                "steps": steps[index].item(),
                "failed": failed[index].item(),
            }
            for index, name in enumerate(motion.names)
        },
    }


class SuperMimicPlaybackSession:
    """Deterministic batched rollouts behind UniLab's viewer session contract."""

    def __init__(
        self,
        env: SuperMimicManagerBasedRlEnv,
        checkpoint: dict,
        device: str | torch.device | None = None,
        policy_device: str | torch.device = "auto",
    ) -> None:
        self.env = env
        self.mdp = cast(MDP, env.supermimic_mdp)
        self.motion = self.mdp.motion
        self.inference_device = _resolve_policy_device(
            torch.device(env.device), device, policy_device
        )
        if self.inference_device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS policy inference was requested but MPS is not available")
        self.policy = load_policy(env, checkpoint, self.inference_device)
        if self.inference_device.type == "mps":
            networks.AMP_DTYPE = None
        args = checkpoint["args"]
        self.goal_frames = int(args["goal_frames"])
        self.hist = int(args["hist"])
        self.num_envs = int(env.num_envs)
        num_clips = len(self.motion.names)
        if self.num_envs > num_clips:
            raise ValueError(
                f"SuperMimic playback environment count must not exceed clips: "
                f"got {self.num_envs} environments for {num_clips} clips"
            )
        self.clip_order = self.motion.clip_len.argsort(descending=True, stable=True).to(env.device)
        self.clip_starts = self.motion.clip_start.to(env.device)
        self.assigned = self.clip_order[: self.num_envs].clone()
        self.next_clip = self.num_envs
        self.current: dict[str, torch.Tensor] | None = None
        self.history: torch.Tensor | None = None

    def reset(self) -> dict[str, torch.Tensor]:
        self.assigned = self.clip_order[: self.num_envs].clone()
        self.next_clip = self.num_envs
        _, info = self.env.reset_to_frames(self.clip_starts[self.assigned].clone())
        current = cast(dict[str, torch.Tensor], info[SUPERMIMIC_STATE_KEY])
        self.current = current
        self.history = None
        return current

    def _reset_history(self, env_ids: torch.Tensor) -> None:
        assert self.current is not None
        assert self.history is not None
        state = cast(State, tuple(value[env_ids] for value in unflatten_state(self.current["s"])))
        obs = self.mdp.obs(
            state,
            self.current["a_prev"][env_ids],
            self.current["xi"][env_ids],
            self.current["f"][env_ids],
        )
        policy_env_ids = env_ids.to(self.inference_device)
        self.history[policy_env_ids] = (
            obs.unsqueeze(1).expand(-1, self.hist, -1).to(self.inference_device)
        )

    @torch.inference_mode()
    def step_once(self) -> None:
        assert self.current is not None
        current = self.current
        obs = self.mdp.obs(
            unflatten_state(current["s"]),
            current["a_prev"],
            current["xi"],
            current["f"],
        ).to(self.inference_device)
        if self.history is None:
            self.history = obs.unsqueeze(1).repeat(1, self.hist, 1)
        else:
            self.history = torch.cat((self.history[:, 1:], obs.unsqueeze(1)), dim=1)
        goal = self.mdp.goal(current["f"], self.goal_frames).to(self.inference_device)
        action = self.policy(self.history, goal)
        state = self.env.step(action.to(self.env.device))
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        self.current = state.info[SUPERMIMIC_STATE_KEY]

        completed = transition["done"].nonzero(as_tuple=False).flatten()
        if completed.numel() == 0:
            return
        reset_clips = self.assigned[completed].clone()
        pending_count = min(completed.numel(), len(self.motion.names) - self.next_clip)
        if pending_count:
            reset_clips[:pending_count] = self.clip_order[
                self.next_clip : self.next_clip + pending_count
            ]
            self.next_clip += pending_count
        _, info = self.env.reset_to_frames(self.clip_starts[reset_clips].clone(), env_ids=completed)
        self.current = info[SUPERMIMIC_STATE_KEY]
        self.assigned[completed] = reset_clips
        self._reset_history(completed)

    def advance(self, controls: Any) -> bool:
        if not controls.consume_step_permission():
            return False
        self.step_once()
        return True

    def physics_state(self) -> Any:
        return self.env.get_physics_state_snapshot()

    @property
    def info(self) -> dict[str, Any]:
        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        return info if isinstance(info, dict) else {}


def run_viser_evaluation(
    env: SuperMimicManagerBasedRlEnv,
    checkpoint: dict,
    *,
    device: str | torch.device | None = None,
    policy_device: str | torch.device = "auto",
    port: int = 8080,
    max_visible_envs: int = 16,
    env_idx: int = 0,
    display_mode: str = "all",
    render_spacing: float = 1.0,
    start_paused: bool = False,
    speed: float = 1.0,
) -> None:
    from unilab.visualization.viser_playback import run_viser_playback

    session = SuperMimicPlaybackSession(env, checkpoint, device=device, policy_device=policy_device)
    run_viser_playback(
        session,
        port=port,
        max_visible_envs=max_visible_envs,
        env_idx=env_idx,
        display_mode=display_mode,
        render_spacing=render_spacing,
        start_paused=start_paused,
        speed=speed,
        entrypoint="unisupermimic-eval --viser",
    )


def _checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("ckpt_"))
    except ValueError as exc:
        raise ValueError(f"Invalid SuperMimic checkpoint name: {path.name}") from exc


def _evaluated_iterations(path: Path) -> set[int]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(encoding="utf-8", newline="") as stream:
        return {int(row["iteration"]) for row in csv.DictReader(stream) if row.get("iteration")}


def _atomic_write_json(path: Path, result: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=1)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _append_eval_row(
    path: Path,
    iteration: int,
    checkpoint: dict,
    evaluated_at: str,
    result: dict,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                iteration,
                f"{checkpoint['wall_time'] / 3600:.4f}",
                checkpoint["saved_at"],
                evaluated_at,
                f"{result['norm_return']:.4f}",
                f"{result['norm_return_framew']:.4f}",
                result["n_failed"],
                "|".join(result["failed"]),
            )
        )


def watch(
    env: SuperMimicManagerBasedRlEnv,
    run_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    every: int = 1,
    poll_seconds: float = 60.0,
    success_threshold: float | None = 0.82,
    once: bool = False,
) -> dict | None:
    """Evaluate new 2000-iteration checkpoints, optionally returning after one scan."""
    if every <= 0:
        raise ValueError("every must be positive")
    if poll_seconds < 0:
        raise ValueError("poll_seconds must be non-negative")
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"SuperMimic run directory does not exist: {run_dir}")
    device = torch.device(env.device if device is None else device)
    csv_path = run_dir / "eval.csv"
    evaluated = _evaluated_iterations(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerow(EVAL_CSV_FIELDS)

    latest: dict | None = None
    while True:
        checkpoints = sorted(run_dir.glob("ckpt_*.pt"), key=_checkpoint_iteration)
        for checkpoint_path in checkpoints:
            iteration = _checkpoint_iteration(checkpoint_path)
            if iteration in evaluated or iteration % 2000 != 0:
                continue
            if (iteration // 2000) % every != 0:
                continue
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            started = time.monotonic()
            result: dict = evaluate(env, checkpoint, device=device)
            evaluated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            _atomic_write_json(
                checkpoint_path.with_name(f"{checkpoint_path.stem}_eval.json"), result
            )
            _append_eval_row(csv_path, iteration, checkpoint, evaluated_at, result)
            evaluated.add(iteration)
            latest = result
            elapsed = time.monotonic() - started
            print(
                f"[eval] it {iteration} ({checkpoint['wall_time'] / 3600:.2f} h): "
                f"norm return {result['norm_return']:.4f}, "
                f"failed {result['n_failed']}/{len(result['per_clip'])} ({elapsed:.0f}s)",
                flush=True,
            )
            if (
                success_threshold is not None
                and result["norm_return"] >= success_threshold
                and result["n_failed"] == 0
            ):
                return result
        if once:
            return latest
        time.sleep(poll_seconds)


def make_eval_env(
    run_dir: Path,
    backend: str = "mjwarp",
    num_envs: int | None = None,
    *,
    default_num_envs: int = DEFAULT_EVAL_NUM_ENVS,
) -> SuperMimicManagerBasedRlEnv:
    from hydra import compose, initialize_config_dir
    from unilab.base.config_adapter import create_env
    from unilab.base.process_device import configure_backend_process_device
    from unilab.training import ensure_registries
    from unilab.utils.device import get_default_device

    import unisupermimic
    from unisupermimic.scripts.train_supermimic import _motion_count, build_env_override
    from unisupermimic.sim2sim import resolve_sim2sim_config

    config_dir = Path(unisupermimic.__file__).resolve().parent / "conf" / "supermimic"
    overrides = ["training.play_only=true", f"task=g1_super_mimic/{backend}"]
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    ensure_registries()
    run_config = run_dir / "run_config.json"
    if run_config.is_file():
        contract = json.loads(run_config.read_text(encoding="utf-8")).get("contract_snapshot", {})
        source_motion_files = contract.get("env.commands.motion.motion_files")
        if source_motion_files is not None:
            OmegaConf.update(cfg, "env.commands.motion.motion_files", source_motion_files)
    clip_count = _motion_count(cfg)
    if num_envs is None:
        num_envs = min(clip_count, default_num_envs)
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
        raise ValueError(f"evaluation num_envs must be a positive integer, got {num_envs!r}")
    if num_envs > clip_count:
        raise ValueError(f"evaluation num_envs must not exceed clip count {clip_count}: {num_envs}")
    resolve_sim2sim_config(
        run_dir,
        cfg,
        algo_name="supermimic",
        strict=bool(cfg.training.sim2sim_strict),
    )
    device = get_default_device()
    bound_device = configure_backend_process_device(str(cfg.training.sim_backend), device)
    if bound_device is not None:
        device = bound_device
    return cast(
        "SuperMimicManagerBasedRlEnv",
        create_env(
            cfg,
            num_envs=num_envs,
            env_cfg_override=build_env_override(cfg, eval_mode=True, learner_device=device),
        ),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="maximum parallel environments; defaults to min(clip count, 16384), or 16 with Viser",
    )
    parser.add_argument(
        "--backend", choices=SUPPORTED_BACKENDS, default="mjwarp", help="Simulation backend"
    )
    parser.add_argument(
        "--policy_device",
        default="auto",
        help="policy inference device; auto selects MPS for CPU environments on macOS",
    )
    parser.add_argument("--viser", action="store_true", help="serve batched playback in a browser")
    parser.add_argument("--viser_port", type=int, default=8080)
    parser.add_argument("--viser_max_envs", type=int, default=16, help="maximum visible envs")
    parser.add_argument("--viser_env_idx", type=int, default=0, help="initial single-view env slot")
    parser.add_argument("--viser_display", choices=("all", "single"), default="all")
    parser.add_argument("--render_spacing", type=float, default=1.0)
    parser.add_argument("--start_paused", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--success_threshold", type=float, default=0.82)
    return parser


def main(argv: list[str] | None = None) -> None:
    from unisupermimic.sim2sim import policy_load_dim_guard

    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.ckpt is None and args.run_dir is None:
        parser.error("one of --run_dir or --ckpt is required")
    if args.viser and args.ckpt is None:
        parser.error("--viser requires --ckpt")
    run_dir = args.ckpt.parent if args.ckpt is not None else args.run_dir
    env = make_eval_env(
        run_dir,
        backend=args.backend,
        num_envs=args.num_envs,
        default_num_envs=args.viser_max_envs if args.viser else DEFAULT_EVAL_NUM_ENVS,
    )
    try:
        if args.ckpt is not None:
            device = torch.device(env.device)
            checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
            with policy_load_dim_guard(
                env_obs_dim=env.supermimic_mdp.obs_dim,
                env_action_dim=env.action_manager.total_action_dim,
                algo_name="supermimic",
            ):
                if args.viser:
                    run_viser_evaluation(
                        env,
                        checkpoint,
                        device=device,
                        policy_device=args.policy_device,
                        port=args.viser_port,
                        max_visible_envs=args.viser_max_envs,
                        env_idx=args.viser_env_idx,
                        display_mode=args.viser_display,
                        render_spacing=args.render_spacing,
                        start_paused=args.start_paused,
                        speed=args.speed,
                    )
                    return
                result = evaluate(env, checkpoint, device=device, policy_device=args.policy_device)
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "per_clip"}, indent=1
                )
            )
            return
        watch(
            env,
            args.run_dir,
            every=args.every,
            poll_seconds=args.poll_seconds,
            success_threshold=args.success_threshold,
            once=not args.watch,
        )
    finally:
        env.close()
        cast(Any, env.backend).close()


__all__ = [
    "EVAL_CSV_FIELDS",
    "SuperMimicPlaybackSession",
    "evaluate",
    "load_policy",
    "main",
    "make_eval_env",
    "run_viser_evaluation",
    "watch",
]

if __name__ == "__main__":
    main()
