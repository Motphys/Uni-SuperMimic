import numpy as np
import pytest
import torch

from unisupermimic.deploy.export import DEFAULT_ASSETS, DEFAULT_MJCF, export_clips
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib


@pytest.mark.slow
def test_exported_goal_matches_motionlib(tmp_path):
    motion = DEFAULT_MJCF.parents[2] / "motions/g1/lafan1_supermimic/walk1_subject1.csv"
    if not DEFAULT_ASSETS.is_dir() or not motion.is_file():
        pytest.skip("full G1 assets and LAFAN fixture are not distributed")
    written = export_clips([motion], tmp_path, mjcf=DEFAULT_MJCF, src_fps=30.0)
    assert len(written) == 1
    with np.load(written[0]) as z:
        exported = z["goal"]
        assert exported.shape[1] == 730
    kin = G1Kinematics(str(DEFAULT_MJCF), "cpu")
    lib = MotionLib([str(motion)], kin, "cpu", src_fps=30.0)
    for frame in (0, 1, 77, exported.shape[0] // 2, exported.shape[0] - 1):
        expected = lib.goal(torch.tensor([frame]), 10).numpy()[0]
        np.testing.assert_allclose(exported[frame], expected, rtol=0, atol=1e-7)
