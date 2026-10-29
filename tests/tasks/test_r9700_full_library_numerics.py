"""Regressions for full-size R9700 reference features (not tiny smoke batches)."""

from pathlib import Path

import pytest
import torch

from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib

ROOT = Path(__file__).resolve().parents[2]
MOTIONS = ROOT / "src/unisupermimic/assets/motions/g1/lafan1_supermimic"
MJCF = ROOT / "src/unisupermimic/assets/robots/g1/supermimic.xml"
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def library() -> MotionLib:
    if not torch.cuda.is_available():
        pytest.skip("full-library numerical test requires a CUDA or ROCm GPU")
    paths = sorted(MOTIONS.glob("*.csv"))
    if not paths or not MJCF.exists():
        pytest.skip("retargeted motion and MJCF assets are not installed")
    assert len(paths) == 40, len(paths)
    kin = G1Kinematics(str(MJCF), "cuda")
    return MotionLib([str(p) for p in paths], kin, "cuda")


def test_entire_library_local_velocity_matches_independent_cpu64(library: MotionLib) -> None:
    assert library.num_frames > 65536
    indices = torch.tensor(
        [0, 100, 65535, 65536, 65537, 80000, library.num_frames - 2], device="cuda"
    )
    rot = library.root_R[indices].cpu().double()
    linear = library.root_v[indices].cpu().double()
    angular = library.root_w[indices].cpu().double()
    expected_v = (rot.transpose(-1, -2) * linear.unsqueeze(-2)).sum(-1).float()
    expected_w = (rot.transpose(-1, -2) * angular.unsqueeze(-2)).sum(-1).float()
    torch.testing.assert_close(library.root_v_local[indices].cpu(), expected_v, rtol=0, atol=2e-4)
    torch.testing.assert_close(library.root_w_local[indices].cpu(), expected_w, rtol=0, atol=2e-4)


def test_large_goal_batch_equals_same_rows_in_safe_chunks(library: MotionLib) -> None:
    frames = torch.arange(8192, device="cuda", dtype=torch.long)
    big = library.goal(frames, 10)
    small = torch.cat([library.goal(chunk, 10) for chunk in frames.split(512)])
    assert big.shape == small.shape
    torch.testing.assert_close(big, small, rtol=0, atol=2e-4)
