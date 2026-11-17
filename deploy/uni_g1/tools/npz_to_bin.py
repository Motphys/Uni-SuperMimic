#!/usr/bin/env python3
"""Convert deployment clip npz files to the C++ binary format (numpy only).

Usage: python3 tools/npz_to_bin.py [assets_dir]
Reads assets/clips/*.npz, writes assets/clips_bin/<name>.bin:
  int32 T + goal (T*730 f32) + joint_pos (T*29 f32)
  + root_quat_wxyz (T*4 f32) + root_pos (T*3 f32), little-endian.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np


def convert(npz_path: Path, out_path: Path) -> int:
    with np.load(npz_path) as z:
        goal = np.asarray(z["goal"], dtype="<f4")
        joint_pos = np.asarray(z["joint_pos"], dtype="<f4")
        root_quat = np.asarray(z["root_quat_wxyz"], dtype="<f4")
        root_pos = np.asarray(z["root_pos"], dtype="<f4")
    t = goal.shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<i", t))
        f.write(goal.tobytes())
        f.write(joint_pos.tobytes())
        f.write(root_quat.tobytes())
        f.write(root_pos.tobytes())
    return t


def main() -> None:
    assets = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets"
    clips = sorted((assets / "clips").glob("*.npz"))
    if not clips:
        raise SystemExit(f"no clips under {assets / 'clips'}")
    for p in clips:
        t = convert(p, assets / "clips_bin" / f"{p.stem}.bin")
        print(f"{p.stem}: T={t}")


if __name__ == "__main__":
    main()
