"""Render pairwise world-model comparison matrices as CSV grids and heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = {
    "param_relative_l2": {
        "title": "Parameter difference: relative L2",
        "short_title": "Parameter rel. L2",
        "path": None,
        "kind": "difference",
        "format": ".2f",
    },
    "increment_rmse": {
        "title": "One-step increment difference: RMSE",
        "short_title": "Increment RMSE",
        "path": ("one_step_increment", "rmse"),
        "kind": "difference",
        "format": ".2f",
    },
    "increment_ccc": {
        "title": "One-step increment agreement: CCC",
        "short_title": "Increment CCC",
        "path": ("one_step_increment", "ccc_global"),
        "kind": "agreement",
        "format": ".2f",
    },
    "hidden_cka": {
        "title": "Hidden representation agreement: linear CKA",
        "short_title": "Hidden CKA",
        "path": ("hidden_layer_cka_mean",),
        "kind": "agreement",
        "format": ".2f",
    },
    "rollout_final_rmse": {
        "title": "8-step final state difference: flattened RMSE",
        "short_title": "Rollout RMSE",
        "path": ("functional_pairs", "rollout_final_block"),
        "kind": "derived",
        "format": ".2f",
    },
}


def _labels(report: dict[str, Any]) -> list[str]:
    return [f"it{x['iteration']}" for x in report["checkpoints"]]


def _lookup(source: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        source = source[key]
    return source


def _pair_key(first: str, second: str) -> str:
    return f"{first}__{second}"


def build_matrix(report: dict[str, Any], metric: str) -> np.ndarray:
    labels = _labels(report)
    n = len(labels)
    matrix = np.full((n, n), np.nan, dtype=float)
    definition = METRICS[metric]

    diagonal = 1.0 if definition["kind"] == "agreement" else 0.0
    np.fill_diagonal(matrix, diagonal)
    for i, first in enumerate(labels):
        for j in range(i + 1, n):
            second = labels[j]
            key = _pair_key(first, second)
            pair = report["functional_pairs"][key]
            if metric == "param_relative_l2":
                value = float(report["parameter_pairs"][key]["network_relative_l2"])
            elif metric == "rollout_final_rmse":
                # The flattened state contains heterogeneous physical units, so
                # report the RMS of block RMSE values as a compact overall score.
                blocks = pair["rollout_final_block"]
                value = float(np.sqrt(np.mean([x["rmse"] ** 2 for x in blocks.values()])))
            else:
                value = float(_lookup(pair, definition["path"]))
            matrix[i, j] = matrix[j, i] = value
    return matrix


def write_grid_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow(
                [
                    label,
                    *(("" if np.isnan(value) else f"{value:.6g}") for value in row),
                ]
            )


def write_markdown(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    lines = ["| model | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for label, row in zip(labels, matrix, strict=True):
        cells = ["—" if np.isnan(value) else f"{value:.3f}" for value in row]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _value_format(decimal_digits: int) -> str:
    return f".{decimal_digits}f"


def _normalized_values(matrix: np.ndarray) -> tuple[float, float, np.ndarray]:
    finite = matrix[np.isfinite(matrix)]
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    normalized = (matrix - vmin) / (vmax - vmin + 1e-12)
    return vmin, vmax, normalized


def _text_color(normalized: float) -> str:
    """Choose readable text over the white middle of a blue-red colormap."""
    return "#1f1f1f" if 0.36 <= normalized <= 0.64 else "white"


def _cell_font_size(n: int, panel_count: int = 1, digits: int = 2) -> float:
    """Scale cell labels with matrix and dashboard size to prevent overlap."""
    size = 9.4 - 0.60 * max(0, n - 3) - 0.42 * max(0, panel_count - 3) - 0.35 * (digits - 2)
    return float(np.clip(size, 3.5, 9.0))


def _tick_font_size(n: int, panel_count: int = 1) -> float:
    return float(np.clip(9.0 - 0.20 * max(0, n - 3) - 0.25 * max(0, panel_count - 3), 5.5, 9.0))


def write_heatmap(
    path: Path,
    title: str,
    labels: list[str],
    matrix: np.ndarray,
    kind: str,
    value_format: str,
) -> None:
    del value_format  # retained in the public API for backward compatibility
    n = len(labels)
    digits = 2
    masked = np.ma.masked_invalid(matrix)
    cmap = "coolwarm"
    colorbar_label = "larger = more consistent" if kind == "agreement" else "larger = more different"

    fig, ax = plt.subplots(figsize=(7.0, 5.7))
    image = ax.imshow(masked, cmap=cmap)
    tick_size = _tick_font_size(n)
    ax.set_xticks(range(n), labels, rotation=45 if n > 7 else 0, ha="right" if n > 7 else "center")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("world model", fontsize=10)
    ax.set_ylabel("world model", fontsize=10)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(labelsize=tick_size)

    _, _, normalized = _normalized_values(matrix)
    cell_size = _cell_font_size(n, digits=digits)
    for i in range(n):
        for j in range(n):
            if np.isnan(matrix[i, j]):
                continue
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.{digits}f}",
                ha="center",
                va="center",
                fontsize=cell_size,
                color=_text_color(normalized[i, j]),
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_dashboard(path: Path, labels: list[str], matrices: dict[str, np.ndarray]) -> None:
    n = len(labels)
    panel_count = len(matrices)
    fig, axes = plt.subplots(1, panel_count, figsize=(3.9 * panel_count, 4.0))
    axes = np.atleast_1d(axes)
    tick_size = _tick_font_size(n, panel_count)
    cell_size = _cell_font_size(n, panel_count, digits=2)

    for ax, (metric, matrix) in zip(axes, matrices.items(), strict=True):
        definition = METRICS[metric]
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, cmap="coolwarm")
        ax.set_xticks(range(n), labels, rotation=45 if n > 5 else 0, ha="right" if n > 5 else "center")
        ax.set_yticks(range(n), labels)
        ax.set_title(definition["short_title"], loc="left", fontsize=10, fontweight="bold", pad=7)
        ax.set_xticks(np.arange(-0.5, n), minor=True)
        ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(labelsize=tick_size)

        _, _, normalized = _normalized_values(matrix)
        for i in range(n):
            for j in range(n):
                if np.isnan(matrix[i, j]):
                    continue
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=cell_size,
                    color=_text_color(normalized[i, j]),
                    fontweight="bold",
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=max(6, tick_size - 1))
        cbar.set_label(
            "consistency" if definition["kind"] == "agreement" else "difference",
            fontsize=max(6.5, tick_size - 0.5),
        )

    fig.suptitle(
        "Pairwise world-model differences (blue → red)",
        x=0.01,
        y=1.04,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report", type=Path, help="world_model_comparison.json produced by compare_world_models.py"
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--metric", choices=METRICS, default="increment_rmse")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    labels = _labels(report)
    out_dir = args.out_dir or args.report.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = build_matrix(report, args.metric)
    prefix = f"world_model_{args.metric}_grid"
    write_grid_csv(out_dir / f"{prefix}.csv", labels, selected)
    write_markdown(out_dir / f"{prefix}.md", labels, selected)
    write_heatmap(
        out_dir / f"{prefix}_heatmap.png",
        METRICS[args.metric]["title"],
        labels,
        selected,
        METRICS[args.metric]["kind"],
        METRICS[args.metric]["format"],
    )

    matrices = {name: build_matrix(report, name) for name in METRICS}
    write_dashboard(out_dir / "world_model_difference_grid_heatmaps.png", labels, matrices)
    for name, matrix in matrices.items():
        write_grid_csv(out_dir / f"world_model_{name}_grid.csv", labels, matrix)

    print(f"wrote grid and heatmap to {out_dir}")
    print(f"selected metric: {args.metric}")


if __name__ == "__main__":
    main()
