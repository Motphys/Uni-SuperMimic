"""Compare SuperMimic world-model checkpoints.

The tool reports three complementary views:

* parameter distance: useful for checkpoints on one optimization trajectory;
* functional agreement on a deterministic common probe;
* linear CKA of hidden representations, which is more appropriate than raw
  parameter distance when models can have hidden-neuron permutations.

The synthetic probe makes pairwise comparisons reproducible without a saved
replay buffer. It does **not** replace a fixed held-out replay validation set.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from unisupermimic.rl.algos.supermimic.geometry import flatten_state, so3_exp, unflatten_state
from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.rl.contracts.supermimic import State

EPS = 1e-12
CONTEXT_DIM = 33
NUM_JOINTS = 29
STATE_BLOCKS = (
    ("position_local_m", 3),
    ("rotation_log_rad", 3),
    ("linear_velocity_m_s", 3),
    ("angular_velocity_rad_s", 3),
    ("joint_position_rad", NUM_JOINTS),
    ("joint_velocity_rad_s", NUM_JOINTS),
)


@dataclass
class LoadedModel:
    path: Path
    iteration: int
    model: WorldModel
    checkpoint_args: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path, help="ckpt_*.pt files")
    parser.add_argument("--probe-size", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=min(8, torch.get_num_threads()))
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="write JSON and PNG summaries")
    return parser.parse_args()


def load_world_model(path: Path, device: torch.device) -> LoadedModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    args = checkpoint["args"]
    hidden = args.get("test_hidden_dims") or args["wm_hidden"]
    model = WorldModel(
        hidden=tuple(hidden),
        num_joints=NUM_JOINTS,
        context_dim=CONTEXT_DIM,
        structured=bool(args.get("wm_structured", False)),
    ).to(device)
    model.load_state_dict(checkpoint["wm"])
    model.eval()
    return LoadedModel(path, int(checkpoint["iteration"]), model, args)


def _is_network_parameter(name: str) -> bool:
    return name.startswith("net.")


def _is_norm_statistic(name: str) -> bool:
    return name.startswith(("in_norm.", "out_norm.", "inc_norm.")) and not name.endswith(".count")


def _flatten_parameters(state: dict[str, torch.Tensor], predicate) -> torch.Tensor:
    return torch.cat(
        [value.detach().cpu().float().reshape(-1) for name, value in state.items() if predicate(name)]
    )


def parameter_metrics(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> dict[str, Any]:
    if first.keys() != second.keys():
        raise ValueError("checkpoint world-model keys differ")

    first_vector = _flatten_parameters(first, _is_network_parameter)
    second_vector = _flatten_parameters(second, _is_network_parameter)
    delta = first_vector - second_vector
    scale = torch.sqrt((first_vector.square().mean() + second_vector.square().mean()) / 2)
    cosine = torch.nn.functional.cosine_similarity(
        first_vector, second_vector, dim=0, eps=EPS
    ).item()

    tensor_metrics: dict[str, dict[str, float]] = {}
    for name in first:
        if not _is_network_parameter(name):
            continue
        a = first[name].detach().cpu().float()
        b = second[name].detach().cpu().float()
        d = a - b
        tensor_scale = torch.sqrt((a.square().mean() + b.square().mean()) / 2)
        tensor_metrics[name] = {
            "relative_l2": (d.norm() / (tensor_scale * torch.sqrt(torch.tensor(a.numel())))).item(),
            "rmse": d.square().mean().sqrt().item(),
            "cosine": torch.nn.functional.cosine_similarity(
                a.reshape(-1), b.reshape(-1), dim=0, eps=EPS
            ).item(),
        }

    changed = sorted(
        tensor_metrics,
        key=lambda name: tensor_metrics[name]["relative_l2"],
        reverse=True,
    )[:10]

    norm_stats = {}
    for prefix in ("in_norm", "out_norm", "inc_norm"):
        mean_keys = [f"{prefix}.mean", f"{prefix}.var"]
        if not all(key in first and key in second for key in mean_keys):
            continue
        mean_delta = first[f"{prefix}.mean"] - second[f"{prefix}.mean"]
        var_delta = first[f"{prefix}.var"] - second[f"{prefix}.var"]
        norm_stats[prefix] = {
            "mean_rmse": mean_delta.square().mean().sqrt().item(),
            "var_relative_rmse": (
                var_delta.square().mean()
                / ((first[f"{prefix}.var"].square().mean() + second[f"{prefix}.var"].square().mean()) / 2)
            )
            .sqrt()
            .item(),
        }

    return {
        "network_relative_l2": (delta.norm() / (scale * torch.sqrt(torch.tensor(first_vector.numel())))).item(),
        "network_rmse": delta.square().mean().sqrt().item(),
        "network_cosine": cosine,
        "normalizers": norm_stats,
        "largest_relative_tensor_changes": [
            {"name": name, **tensor_metrics[name]} for name in changed
        ],
    }


def make_probe(
    batch_size: int,
    horizon: int,
    device: torch.device,
    seed: int,
) -> tuple[State, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    position = torch.zeros(batch_size, 3, device=device)
    position[:, 2] = 0.75 + 0.12 * torch.randn(batch_size, generator=generator)
    rotation = so3_exp(0.35 * torch.randn(batch_size, 3, generator=generator)).to(device)
    lin_vel = (0.7 * torch.randn(batch_size, 3, generator=generator)).to(device)
    ang_vel = (0.7 * torch.randn(batch_size, 3, generator=generator)).to(device)
    joint_pos = (0.7 * torch.randn(batch_size, NUM_JOINTS, generator=generator)).to(device)
    joint_vel = (1.0 * torch.randn(batch_size, NUM_JOINTS, generator=generator)).to(device)
    state: State = (position, rotation, lin_vel, ang_vel, joint_pos, joint_vel)

    action = torch.tanh(
        torch.randn(horizon, batch_size, NUM_JOINTS, generator=generator)
    ).to(device)
    normalized_context = torch.randn(horizon, batch_size, CONTEXT_DIM, generator=generator).to(device)
    return state, action, normalized_context


@torch.inference_mode()
def forward_trace(
    model: WorldModel, state: State, action: torch.Tensor, normalized_context: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    normalized_input = model.in_norm(model.inputs(state, action, normalized_context))
    hidden: list[torch.Tensor] = []
    output = normalized_input
    for module in model.net:
        output = module(output)
        if isinstance(module, nn.SiLU):
            hidden.append(output)
    increment = output * model.out_norm.std() + model.out_norm.mean
    return increment, hidden


def linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    """Compute centered linear CKA; column permutations do not change it."""
    first = first.detach().float() - first.detach().float().mean(0, keepdim=True)
    second = second.detach().float() - second.detach().float().mean(0, keepdim=True)
    numerator = (first.T @ second).square().sum()
    first_denominator = (first.T @ first).square().sum().clamp_min(EPS).sqrt()
    second_denominator = (second.T @ second).square().sum().clamp_min(EPS).sqrt()
    return (numerator / (first_denominator * second_denominator)).item()


def agreement_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    first = first.detach().float()
    second = second.detach().float()
    flat_first = first.reshape(-1)
    flat_second = second.reshape(-1)
    delta = flat_first - flat_second
    rmse = delta.square().mean().sqrt()
    scale = torch.sqrt((flat_first.square().mean() + flat_second.square().mean()) / 2)
    centered_first = flat_first - flat_first.mean()
    centered_second = flat_second - flat_second.mean()
    covariance = (centered_first * centered_second).mean()
    global_pearson = covariance / (
        centered_first.square().mean().sqrt() * centered_second.square().mean().sqrt() + EPS
    )
    global_ccc = 2 * covariance / (
        centered_first.square().mean()
        + centered_second.square().mean()
        + (flat_first.mean() - flat_second.mean()).square()
        + EPS
    )

    dim_centered_first = first - first.mean(0, keepdim=True)
    dim_centered_second = second - second.mean(0, keepdim=True)
    dim_covariance = (dim_centered_first * dim_centered_second).mean(0)
    dim_first_std = dim_centered_first.square().mean(0).sqrt()
    dim_second_std = dim_centered_second.square().mean(0).sqrt()
    valid = (dim_first_std > 1e-8) & (dim_second_std > 1e-8)
    dim_pearson = (dim_covariance[valid] / (dim_first_std[valid] * dim_second_std[valid])).mean()

    dim_var_first = dim_centered_first.square().mean(0)
    dim_var_second = dim_centered_second.square().mean(0)
    dim_mean_delta = (first.mean(0) - second.mean(0)).square()
    dim_ccc = 2 * dim_covariance / (dim_var_first + dim_var_second + dim_mean_delta + EPS)

    return {
        "rmse": rmse.item(),
        "mae": delta.abs().mean().item(),
        "relative_rmse": (rmse / (scale + EPS)).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            flat_first, flat_second, dim=0, eps=EPS
        ).item(),
        "pearson_global": global_pearson.item(),
        "ccc_global": global_ccc.item(),
        "pearson_dimension_mean": dim_pearson.item(),
        "ccc_dimension_mean": dim_ccc.mean().item(),
    }


def _state_to_cpu(state: State) -> State:
    return tuple(value.cpu() for value in state)  # type: ignore[return-value]


def state_error_metrics(first: State, second: State) -> dict[str, dict[str, float]]:
    error = WorldModel.state_error(first, second)
    result: dict[str, dict[str, float]] = {}
    start = 0
    for name, width in STATE_BLOCKS:
        block = error[..., start : start + width]
        result[name] = {
            "rmse": block.square().mean().sqrt().item(),
            "mae": block.abs().mean().item(),
            "p95_abs": block.abs().reshape(-1).quantile(0.95).item(),
        }
        start += width
    return result


def ensemble_dispersion(samples: torch.Tensor) -> dict[str, float]:
    """Describe inter-model spread over a common probe."""
    values = samples.detach().float()
    mean = values.mean(0, keepdim=True)
    spread = (values - mean).square().mean().sqrt()
    signal = mean.square().mean().sqrt()
    centered = values - values.mean(0, keepdim=True)
    return {
        "inter_model_rmse": spread.item(),
        "signal_rms": signal.item(),
        "relative_dispersion": (spread / (signal + EPS)).item(),
        "dimension_mean_std": centered.square().mean(dim=(0, 1)).sqrt().mean().item(),
    }


@torch.inference_mode()
def functional_metrics(
    models: list[LoadedModel],
    initial_state: State,
    actions: torch.Tensor,
    contexts: torch.Tensor,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = initial_state[0].device
    increments: list[torch.Tensor] = []
    hidden_by_layer: list[list[torch.Tensor]] = []
    one_step_states: list[State] = []
    rollout_states = torch.empty(
        len(models), horizon, initial_state[0].shape[0], 13 + 2 * NUM_JOINTS,
        device=device,
    )

    for model_index, loaded in enumerate(models):
        model = loaded.model
        increment, hidden = forward_trace(
            model,
            initial_state,
            actions[0],
            contexts[0],
        )
        increments.append(increment.cpu())
        hidden_by_layer.append([activation.cpu() for activation in hidden])
        one_step_states.append(model.integrate(initial_state, increment))
        state = initial_state
        for step in range(horizon):
            state = model.step(state, actions[step], contexts[step])
            rollout_states[model_index, step] = flatten_state(state)

    pair_results: dict[str, Any] = {}
    increment_stack = torch.stack(increments)
    for first_index in range(len(models)):
        for second_index in range(first_index + 1, len(models)):
            first_label = _label(models[first_index])
            second_label = _label(models[second_index])
            hidden_cka = [
                linear_cka(hidden_by_layer[first_index][i], hidden_by_layer[second_index][i])
                for i in range(len(hidden_by_layer[first_index]))
            ]
            step_errors = []
            for step in range(horizon):
                first_state = unflatten_state(rollout_states[first_index, step].cpu())
                second_state = unflatten_state(rollout_states[second_index, step].cpu())
                block_metrics = state_error_metrics(first_state, second_state)
                step_errors.append(block_metrics)

            final_metrics = step_errors[-1]
            pair_results[f"{first_label}__{second_label}"] = {
                "one_step_increment": agreement_metrics(
                    increments[first_index], increments[second_index]
                ),
                "one_step_state": state_error_metrics(
                    _state_to_cpu(one_step_states[first_index]),
                    _state_to_cpu(one_step_states[second_index]),
                ),
                "hidden_layer_cka": hidden_cka,
                "hidden_layer_cka_mean": sum(hidden_cka) / len(hidden_cka),
                "rollout_mean_block_rmse": {
                    name: sum(step[name]["rmse"] for step in step_errors) / horizon
                    for name, _ in STATE_BLOCKS
                },
                "rollout_final_block": final_metrics,
            }

    ensemble = {
        "one_step_increment": ensemble_dispersion(increment_stack),
        "one_step_state": ensemble_dispersion(
            torch.stack([flatten_state(state).cpu() for state in one_step_states])
        ),
        "rollout_final_state": ensemble_dispersion(rollout_states[:, -1].cpu()),
    }
    return pair_results, ensemble


def _label(loaded: LoadedModel) -> str:
    return f"it{loaded.iteration}"


def parse_training_log(path: Path | None, iterations: list[int]) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {int(row["iteration"]): row for row in csv.DictReader(handle)}
    result = []
    for iteration in iterations:
        row = rows.get(iteration)
        if row is None:
            continue
        result.append(
            {
                "iteration": iteration,
                "wm_loss": float(row["wm_loss"]),
                "wm_err8": float(row["wm_err8"]),
                "note": "online replay-batch metric, not a fixed held-out validation set",
            }
        )
    return result


def write_plots(
    out_dir: Path,
    models: list[LoadedModel],
    parameter_pairs: dict[str, Any],
    functional_pairs: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [_label(model) for model in models]
    count = len(labels)

    def matrix(
        value_name: str, functional: bool, diagonal: float
    ) -> list[list[float]]:
        matrix_values = [[diagonal if i == j else 0.0 for j in range(count)] for i in range(count)]
        for first in range(count):
            for second in range(first + 1, count):
                key = f"{labels[first]}__{labels[second]}"
                source = functional_pairs[key] if functional else parameter_pairs[key]
                value = source
                for part in value_name.split("."):
                    value = value[part]
                matrix_values[first][second] = matrix_values[second][first] = float(value)
        return matrix_values

    panels = [
        ("Network relative L2", matrix("network_relative_l2", False, 0.0), "RMS-normalized distance"),
        ("One-step increment CCC", matrix("one_step_increment.ccc_global", True, 1.0), "CCC"),
        ("Hidden-representation CKA", matrix("hidden_layer_cka_mean", True, 1.0), "mean linear CKA"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for ax, (title, values, colorbar_label) in zip(axes, panels, strict=True):
        image = ax.imshow(values, cmap="viridis" if title != "Network relative L2" else "magma")
        ax.set_xticks(range(count), labels)
        ax.set_yticks(range(count), labels)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        for first in range(count):
            for second in range(count):
                ax.text(
                    second,
                    first,
                    f"{values[first][second]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
        fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(out_dir / "world_model_similarity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.probe_size < 2:
        raise ValueError("--probe-size must be at least 2")
    if args.horizon < 1:
        raise ValueError("--horizon must be at least 1")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    models = [load_world_model(path, device) for path in args.checkpoints]
    if len(models) < 2:
        raise ValueError("at least two checkpoints are required")

    parameter_pairs = {}
    for first in range(len(models)):
        for second in range(first + 1, len(models)):
            key = f"{_label(models[first])}__{_label(models[second])}"
            parameter_pairs[key] = parameter_metrics(
                models[first].model.state_dict(), models[second].model.state_dict()
            )

    initial_state, actions, contexts = make_probe(args.probe_size, args.horizon, device, args.seed)
    pairs, ensemble = functional_metrics(models, initial_state, actions, contexts, args.horizon)

    report = {
        "interpretation": {
            "parameter_distance": (
                "Valid for aligned checkpoints from one trajectory. Across independently "
                "trained models, permutation symmetry makes raw parameter distance difficult "
                "to interpret; use functional metrics or CKA instead."
            ),
            "functional_probe": (
                "All models receive the same deterministic synthetic state/action/context. "
                "This measures mutual agreement, not accuracy against ground truth."
            ),
            "ground_truth": (
                "For accuracy, rerun all checkpoints on the same fixed replay windows and "
                "compute paired errors relative to s1."
            ),
        },
        "checkpoints": [
            {
                "path": str(loaded.path.resolve()),
                "iteration": loaded.iteration,
                "structured": bool(loaded.checkpoint_args.get("wm_structured", False)),
                "wm_hidden": list(loaded.checkpoint_args["wm_hidden"]),
            }
            for loaded in models
        ],
        "probe": {
            "seed": args.seed,
            "batch_size": args.probe_size,
            "horizon": args.horizon,
            "device": str(device),
            "synthetic": True,
        },
        "parameter_pairs": parameter_pairs,
        "functional_pairs": pairs,
        "ensemble_dispersion": ensemble,
        "training_log": parse_training_log(
            args.train_log, [loaded.iteration for loaded in models]
        ),
    }

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "world_model_comparison.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        write_plots(args.out_dir, models, parameter_pairs, pairs)

    print(f"{'pair':<20} {'param_dL2':>10} {'incRMSE':>10} {'CCC':>8} {'CKA':>8}")
    for key, pair in pairs.items():
        parameter = parameter_pairs[key]
        increment = pair["one_step_increment"]
        print(
            f"{key:<20} "
            f"{parameter['network_relative_l2']:>10.4f} "
            f"{increment['rmse']:>10.5f} "
            f"{increment['ccc_global']:>8.4f} "
            f"{pair['hidden_layer_cka_mean']:>8.4f}"
        )
    print(
        "ensemble one-step increment relative dispersion: "
        f"{ensemble['one_step_increment']['relative_dispersion']:.4f}"
    )
    if args.out_dir is not None:
        print(f"wrote {args.out_dir / 'world_model_comparison.json'}")


if __name__ == "__main__":
    main()
