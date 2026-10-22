"""Deterministic full-clip evaluation and checkpoint watching for G1 SuperMimic."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
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
) -> dict:
    """Evaluate every motion once from its first frame without observation/action noise."""
    mdp = cast(MDP, env.supermimic_mdp)
    motion = mdp.motion
    args = checkpoint["args"]
    goal_frames = int(args["goal_frames"])
    hist = int(args["hist"])
    policy = load_policy(env, checkpoint, device)

    num_clips = len(motion.names)
    if env.num_envs != num_clips:
        raise ValueError(
            f"SuperMimic evaluation requires one environment per clip: "
            f"got {env.num_envs} environments for {num_clips} clips"
        )
    _, reset_info = env.reset_to_frames(motion.clip_start.clone())
    current = reset_info[SUPERMIMIC_STATE_KEY]
    returns = torch.zeros(num_clips, device=env.device)
    steps = torch.zeros(num_clips, dtype=torch.int64, device=env.device)
    finished = torch.zeros(num_clips, dtype=torch.bool, device=env.device)
    failed = torch.zeros_like(finished)
    history = None

    while not bool(finished.all()):
        obs = mdp.obs(
            unflatten_state(current["s"]),
            current["a_prev"],
            current["xi"],
            current["f"],
        )
        if history is None:
            history = obs.unsqueeze(1).repeat(1, hist, 1)
        else:
            history = torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1)
        goal = mdp.goal(current["f"], goal_frames)
        action = policy(history, goal)
        state = env.step(action)
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]

        alive = ~finished
        returns += torch.where(alive, transition["r"], torch.zeros_like(transition["r"]))
        steps += alive
        failed |= alive & transition["term"]
        finished |= alive & transition["done"]
        current = state.info[SUPERMIMIC_STATE_KEY]

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


def make_eval_env(run_dir: Path, backend: str = "mjwarp") -> SuperMimicManagerBasedRlEnv:
    from hydra import compose, initialize_config_dir
    from unilab.base.config_adapter import create_env
    from unilab.base.process_device import configure_backend_process_device
    from unilab.training import ensure_registries
    from unilab.utils.device import get_default_device

    import unisupermimic
    from unisupermimic.scripts.train_supermimic import build_env_override
    from unisupermimic.sim2sim import resolve_sim2sim_config

    config_dir = Path(unisupermimic.__file__).resolve().parent / "conf" / "supermimic"
    overrides = ["training.play_only=true", f"task=g1_super_mimic/{backend}"]
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    ensure_registries()
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
            num_envs=len(cfg.env.commands.motion.motion_files),
            env_cfg_override=build_env_override(cfg, eval_mode=True, learner_device=device),
        ),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument(
        "--backend", choices=SUPPORTED_BACKENDS, default="mjwarp", help="Simulation backend"
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--success_threshold", type=float, default=0.82)
    return parser


def main(argv: list[str] | None = None) -> None:
    from unisupermimic.sim2sim import policy_load_dim_guard

    args = _argument_parser().parse_args(argv)
    if args.ckpt is None and args.run_dir is None:
        _argument_parser().error("one of --run_dir or --ckpt is required")
    run_dir = args.ckpt.parent if args.ckpt is not None else args.run_dir
    env = make_eval_env(run_dir, backend=args.backend)
    try:
        if args.ckpt is not None:
            device = torch.device(env.device)
            checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
            with policy_load_dim_guard(
                env_obs_dim=env.supermimic_mdp.obs_dim,
                env_action_dim=env.action_manager.total_action_dim,
                algo_name="supermimic",
            ):
                result = evaluate(env, checkpoint, device=device)
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


__all__ = ["EVAL_CSV_FIELDS", "evaluate", "load_policy", "main", "make_eval_env", "watch"]

if __name__ == "__main__":
    main()
