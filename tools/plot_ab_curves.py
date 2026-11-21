"""Plot baseline vs async-collect training curves from two train_log.csv files."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "baseline (sync mjwarp)": "runs/boneseed_latest_20261117_161712",
    "async mujoco + async_collect": "runs/boneseed_async_mujoco_20261117_174721",
}

PANELS = [
    ("wm_loss", "WM loss", "log"),
    ("wm_err8", "WM error @8", "log"),
    ("sim_rew", "sim reward", None),
    ("ep_len", "episode length", None),
    ("imag_survival", "imagination survival", None),
    ("it_per_s", "throughput [it/s] (50-iter window)", None),
]


def load(run_dir: str) -> dict[str, list[float]]:
    rows = list(csv.DictReader(open(Path(run_dir) / "train_log.csv")))
    cols: dict[str, list[float]] = {}
    for key in rows[0]:
        try:
            cols[key] = [float(r[key]) for r in rows]
        except ValueError:
            pass
    return cols


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/ab_curves.png")
    data = {name: load(d) for name, d in RUNS.items()}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for ax, (key, title, yscale) in zip(axes.flat, PANELS):
        for name, cols in data.items():
            ax.plot(cols["iteration"], cols[key], label=name, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        if yscale:
            ax.set_yscale(yscale)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=9)
    # walltime panel overlay: sim_rew vs wall_h
    ax = axes.flat[-1]
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
