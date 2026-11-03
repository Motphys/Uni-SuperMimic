"""Regressions for full-size reference features (not tiny smoke batches)."""

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


def test_contiguous_reference_timeline_matches_per_step_queries(library: MotionLib) -> None:
    """The contiguous cache and unified timeline reproduce per-step access exactly."""
    generator = torch.Generator(device="cuda").manual_seed(1234)
    frames = torch.randint(0, library.num_frames, (257,), device="cuda", generator=generator)
    # Exercise every clip boundary and the library's global boundary.
    frames[:2] = torch.tensor([0, library.num_frames - 1], device="cuda")
    clip_ends = library.clip_end
    interior = (clip_ends[library.clip_id[frames]] - 1).clamp(min=0)
    frames[2:4] = interior[2:4]
    for actual, expected in zip(
        library.ref_state_pack(frames.reshape(1, -1)).state(),
        library.ref_state(frames.reshape(1, -1)),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    cache = library.reference_cache
    assert cache.shape == (library.num_frames, 76)
    assert cache.is_contiguous()
    for actual, expected in zip(
        library.ref_state_pack(frames).state(), library.ref_state(frames), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(
        library.body_state(frames), library.body_state(frames), strict=True
    ):
        assert actual.is_contiguous()
        assert actual.shape[:1] == expected.shape[:1]

    # Rollouts reset before crossing a clip/global end, so valid timeline starts
    # stay at least a horizon away from the final frame.
    timeline_frames = frames.clamp(max=library.num_frames - 9)
    timeline = library.reference_timeline(timeline_frames, 8, 10)
    assert timeline.frames.shape == (9, 257)
    assert timeline.reference.root_p.shape == (9 * 257, 3)
    assert timeline.goals.shape == (9, 257, 730)
    torch.testing.assert_close(
        timeline.frames,
        timeline_frames.unsqueeze(0) + torch.arange(9, device="cuda").unsqueeze(-1),
    )
    for index in range(9):
        expected = library.ref_state(timeline_frames + index)
        actual = timeline.step(index, len(timeline_frames)).state()
        for observed, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            timeline.goals[index], library.goal(timeline_frames + index, 10), rtol=0, atol=0
        )
