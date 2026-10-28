"""Small end-to-end tests for the BONES-SEED preparation pipeline."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from unisupermimic.data.boneseed import (
    convert_directory,
    filter_directory,
    load_csv,
    longest_run,
    name_filter,
    pack_library,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
    G1_JOINT_NAMES,
    TRACKED_BODIES,
)


def _write_model(path: Path) -> None:
    inertial = '<inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>'
    lines = [
        "<mujoco>",
        '<compiler angle="radian"/>',
        "<worldbody>",
        '<body name="pelvis">',
        inertial,
        '<freejoint name="root"/>',
    ]
    for index, joint_name in enumerate(G1_JOINT_NAMES):
        body_name = (
            TRACKED_BODIES[index + 1] if index + 1 < len(TRACKED_BODIES) else f"link_{index}"
        )
        lines.append(f'<body name="{body_name}" pos="0 0 0">')
        lines.append(inertial)
        lines.append(f'<joint name="{joint_name}" type="hinge" axis="0 0 1" range="-2 2"/>')
        if body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            lines.append('<geom type="sphere" size="0.01" pos="0 0 -0.76"/>')
    lines.extend(["</body>"] * (len(G1_JOINT_NAMES) + 1))
    lines.extend(("</worldbody>", "</mujoco>"))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_bones_csv(path: Path) -> None:
    columns = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *(f"{name}_dof" for name in G1_JOINT_NAMES),
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for frame in range(13):
            writer.writerow((frame, frame, 0.0, 76.0, 0.0, 0.0, 90.0, *([1.0] * 29)))


def test_bones_csv_units_and_joint_order(tmp_path: Path) -> None:
    path = tmp_path / "motion.csv"
    _write_bones_csv(path)

    raw = load_csv(path)

    assert raw.shape == (13, 36)
    np.testing.assert_allclose(raw[1, :3], [0.01, 0.0, 0.76])
    np.testing.assert_allclose(raw[0, 3:7], [0.0, 0.0, 2**-0.5, 2**-0.5], atol=1e-7)
    np.testing.assert_allclose(raw[:, 7:], np.deg2rad(1.0))
    path.write_text(
        path.read_text(encoding="utf-8").replace("left_hip_pitch_joint_dof", "bad_dof"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="joint columns"):
        load_csv(path)


def test_convert_filter_and_pack(tmp_path: Path) -> None:
    model = tmp_path / "g1.xml"
    raw_dir = tmp_path / "raw"
    npz_dir = tmp_path / "npz"
    raw_dir.mkdir()
    _write_model(model)
    _write_bones_csv(raw_dir / "walk.csv")
    shutil.copy(raw_dir / "walk.csv", raw_dir / "bike_stunt.csv")

    convert_directory(raw_dir, npz_dir, workers=1, model_path=model)

    with np.load(npz_dir / "walk.npz") as motion:
        assert motion["joint_pos"].shape == (6, 29)
        assert motion["body_pos_w"].shape[0] == 6
        assert float(motion["fps"][0]) == 50.0
        assert float(motion["input_fps"][0]) == 120.0
        pelvis = list(motion["body_names"]).index("pelvis")
        np.testing.assert_allclose(motion["body_pos_w"][1, pelvis], [0.024, 0.0, 0.76], atol=1e-6)
    manifest = filter_directory(npz_dir, workers=1, model_path=model)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    assert {row["name"]: row["keep"] for row in rows} == {"bike_stunt": "0", "walk": "1"}

    output = tmp_path / "library.pt"
    pack_library(npz_dir, output, manifest=manifest)

    library = torch.load(output, map_location="cpu", weights_only=True)
    assert library["names"] == ["walk"]
    assert library["starts"] == [0]
    assert library["lengths"] == [6]
    assert library["ref"].shape == (6, 71)
    assert library["ref"].dtype == torch.float16


def test_name_filter_and_longest_run() -> None:
    assert name_filter("jump_box_high") == "name:_box_"
    assert name_filter("ordinary_walk") is None
    mask = np.array([False, True, True, True, False, True])
    assert longest_run(mask) == 3
    assert longest_run(np.zeros(4, dtype=bool)) == 0
