"""Tests for SuperMimic reference-ghost video rendering."""

from pathlib import Path

import numpy as np
import pytest
import torch

from unisupermimic.tasks.motion_tracking.g1.supermimic import render

SCENE = Path(__file__).resolve().parents[2] / render.DEFAULT_SCENE


def test_flattened_qpos_extracts_root_and_joint_positions() -> None:
    state = torch.arange(71, dtype=torch.float32).reshape(1, 71)

    qpos = render.flattened_qpos(state)

    assert qpos.shape == (1, 36)
    torch.testing.assert_close(qpos[0, :7], state[0, :7])
    torch.testing.assert_close(qpos[0, 7:], state[0, 13:42])


def test_comparison_xml_adds_noncolliding_reference_ghost() -> None:
    xml = render.comparison_xml(SCENE)

    assert '<body name="ref_pelvis">' in xml
    assert '<freejoint name="ref_floating_base_joint"' in xml
    assert 'rgba="0.35 0.65 1.0 0.25"' in xml
    assert '<geom contype="0" conaffinity="0" group="1"' in xml
    assert "<keyframe>" not in xml


def test_comparison_model_loads_and_removes_temporary_xml() -> None:
    temporary_files = set(SCENE.parent.glob(".render_*.xml"))

    model = render.load_comparison_model(SCENE)

    assert model.nq == 72
    assert set(SCENE.parent.glob(".render_*.xml")) == temporary_files


def test_comparison_frame_preserves_shape_and_adds_labels() -> None:
    image = np.full((540, 960, 3), 128, dtype=np.uint8)

    frame = render.make_comparison_frame(image, "dance1_subject1  t= 0.0s")

    assert frame.shape == image.shape
    assert frame.dtype == np.uint8
    assert not np.array_equal(frame, image)


def test_rollout_qpos_trims_continuous_rollout(monkeypatch) -> None:
    from types import SimpleNamespace

    num_frames = 200
    root_p = torch.zeros(num_frames, 3)
    root_p[:, 2] = torch.arange(num_frames)
    motion = SimpleNamespace(
        names=["jumps1_subject1"],
        clip_start=torch.tensor([0]),
        root_p=root_p,
        root_R=torch.eye(3).expand(num_frames, 3, 3),
        q=torch.zeros(num_frames, 29),
    )

    class FakeMdp:
        def __init__(self) -> None:
            self.motion = motion

        def obs(self, state, previous_action, context, frames):
            return torch.zeros(frames.shape[0], 1)

        def goal(self, state, frames, num_frames):
            del state
            return torch.zeros(frames.shape[0], num_frames)

    class FakeEnv:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.supermimic_mdp = FakeMdp()
            self.steps = 0

        def _current(self, frames):
            state = torch.zeros(1, 71)
            state[0, 2] = frames[0].float()
            state[0, 3] = 1.0
            return {
                "s": state,
                "f": frames.clone(),
                "a_prev": torch.zeros(1, 29),
                "xi": torch.zeros(1, 1),
            }

        def reset_to_frames(self, frames):
            self.current = self._current(frames)
            return {}, {render.SUPERMIMIC_STATE_KEY: self.current}

        def step(self, action):
            self.steps += 1
            self.current = self._current(self.current["f"] + 1)
            return SimpleNamespace(
                info={
                    render.SUPERMIMIC_STATE_KEY: self.current,
                    render.SUPERMIMIC_TRANSITION_KEY: {"done": torch.tensor([False])},
                }
            )

    def policy(history, goal):
        return torch.zeros(history.shape[0], 29)

    monkeypatch.setattr(render, "load_policy", lambda *_: policy)
    env = FakeEnv()

    reference, policy_qpos = render.rollout_qpos(
        env,
        {"args": {"hist": 1, "goal_frames": 1}},
        "jumps1_subject1",
        seconds=0.2,
        start=0.1,
    )

    assert env.steps == 14
    assert reference.shape == (10, 36)
    assert policy_qpos.shape == (10, 36)
    np.testing.assert_allclose(reference[:, 2], np.arange(5, 15))
    np.testing.assert_allclose(policy_qpos[:, 2], np.arange(5, 15))


@pytest.mark.parametrize("backend", render.SUPPORTED_BACKENDS)
def test_video_parser_accepts_backend(backend: str) -> None:
    args = render._argument_parser().parse_args(["--ckpt", "run/ckpt.pt", "--backend", backend])

    assert args.backend == backend
