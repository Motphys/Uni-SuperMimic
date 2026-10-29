"""Plot full-library evaluation and training reward for the optimized run."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURF = "#fcfcfb"


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel("iteration", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_log", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/boneseed_optimized_curve.png",
    )
    args = parser.parse_args()

    with args.train_log.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_iterations = [int(row["iteration"]) for row in rows]
    train_rewards = [float(row["sim_rew"]) for row in rows]

    eval_iterations = [2000, 4000, 6000, 8000]
    eval_returns = [0.8092, 0.9125, 0.9306, 0.9395]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.94), facecolor=SURF)

    ax = axes[0]
    style(ax, "Full-library evaluation", "norm_return")
    ax.plot(
        eval_iterations,
        eval_returns,
        color="tab:blue",
        lw=2,
        marker="o",
        ms=4.5,
        mec=SURF,
        mew=1.2,
        zorder=3,
        label="EMA + no-sync + lr",
    )
    ax.axhline(
        0.9243,
        color="tab:orange",
        lw=1.5,
        ls="--",
        zorder=1,
        label="baseline 12k",
    )
    for it, ret, hours in (
        (6000, 0.9306, "0.60 h"),
        (8000, 0.9395, "0.80 h"),
    ):
        ax.annotate(
            hours,
            xy=(it, ret),
            xytext=(0, 18),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7),
        )
    ax.set_ylim(0.78, 1.00)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    style(ax, "Training reward", "reward")
    ax.plot(train_iterations, train_rewards, color="tab:blue", lw=2, zorder=3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURF)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
