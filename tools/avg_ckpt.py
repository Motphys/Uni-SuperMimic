"""Average the deployed (EMA) policy weights of several checkpoints of one run.

Weights and biases are averaged with equal weights; the observation / goal
normalizers and other non-float buffers are taken from the last checkpoint.
The evaluator reads the output like any checkpoint (a run_config.json copy is
placed next to it).

    uv run python tools/avg_ckpt.py --run_dir logs/lra8k_seed0 \
        --ckpts 6000,6500,7000,7500,8000 --out_dir logs/lra8k_seed0/avg
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--ckpts", required=True, help="comma-separated iterations")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    its = [int(x) for x in args.ckpts.split(",")]
    run_dir = args.run_dir
    out_dir = args.out_dir
    if out_dir.exists():
        raise SystemExit(f"output directory already exists: {out_dir}")
    checkpoints = [
        torch.load(run_dir / f"ckpt_{it:06d}.pt", map_location="cpu", weights_only=True)
        for it in its
    ]
    last = checkpoints[-1]

    def keep(key: str) -> bool:
        return ("_norm." in key) or not torch.is_floating_point(last["pi"][key])

    averaged: dict[str, torch.Tensor] = {}
    for key, value in last["pi"].items():
        if keep(key):
            averaged[key] = value.clone()
        else:
            averaged[key] = (
                torch.stack([c["pi"][key].float() for c in checkpoints])
                .mean(0)
                .to(value.dtype)
            )
    out_dir.mkdir(parents=True)
    for snapshot_name in ("run_config.json", "args.json"):
        source = run_dir / snapshot_name
        if source.is_file():
            shutil.copy(source, out_dir / snapshot_name)
    out = dict(last, pi=averaged, averaged_from=its, averaged_run=str(run_dir))
    torch.save(out, out_dir / f"ckpt_{its[-1]:06d}.pt")
    json.dump(
        {
            "run_dir": str(run_dir),
            "ckpts": its,
            "kept_from_last": [k for k in averaged if keep(k)],
        },
        out_dir / "avg_info.json",
        indent=1,
    )
    print(f"averaged {len(its)} checkpoints -> {out_dir}/ckpt_{its[-1]:06d}.pt")


if __name__ == "__main__":
    main()
