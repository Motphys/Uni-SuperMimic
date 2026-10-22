"""Asset-free numerical tests for the G1 SuperMimic task modules."""

from __future__ import annotations

import csv

import pytest
import torch

from unisupermimic.rl.contracts.supermimic import SuperMimicMdpProtocol
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import (
    CRITIC_EXTRA,
    INV_DIM,
    MAX_STEP_REWARD,
    MDP,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
    G1_JOINT_NAMES,
    TRACKED_BODIES,
    action_scale,
    default_joint_pos_list,
    motor_of,
)


def _write_model(path) -> None:
    inertial = '<inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>'
    lines = [
        "<mujoco>",
        '<compiler angle="radian"/>',
        "<worldbody>",
        '<body name="pelvis">',
        inertial,
        '<freejoint name="root"/>',
    ]
    axes = ("1 0 0", "0 1 0", "0 0 1")
    for index, joint_name in enumerate(G1_JOINT_NAMES):
        body_name = (
            TRACKED_BODIES[index + 1] if index + 1 < len(TRACKED_BODIES) else f"link_{index}"
        )
        lines.append(f'<body name="{body_name}" pos="0.01 0.02 -0.03">')
        lines.append(inertial)
        lines.append(
            f'<joint name="{joint_name}" type="hinge" axis="{axes[index % 3]}" range="-2 2"/>'
        )
    lines.extend(["</body>"] * (len(G1_JOINT_NAMES) + 1))
    lines.extend(("</worldbody>", "</mujoco>"))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_clip(path, source_frames: int) -> None:
    default_q = default_joint_pos_list()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for index in range(source_frames):
            time = index / 30.0
            joints = [value + 0.001 * index for value in default_q]
            writer.writerow((time, 0.1 * time, 0.76, 0.0, 0.0, 0.0, 1.0, *joints))


@pytest.fixture(scope="module")
def library(tmp_path_factory) -> tuple[G1Kinematics, MotionLib, MDP]:
    root = tmp_path_factory.mktemp("supermimic_numeric")
    model_path = root / "g1.xml"
    first_clip = root / "first.csv"
    second_clip = root / "second.csv"
    _write_model(model_path)
    _write_clip(first_clip, 62)
    _write_clip(second_clip, 5)
    kin = G1Kinematics(str(model_path), "cpu")
    motion = MotionLib([str(first_clip), str(second_clip)], kin, "cpu")
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()))
    return kin, motion, mdp


def test_robot_constants_and_kinematics(library) -> None:
    kin, _, _ = library
    assert len(G1_JOINT_NAMES) == 29
    assert len(TRACKED_BODIES) == 14
    assert motor_of("left_hip_pitch_joint") == (88.0, 32.0, 0.010177520)
    assert action_scale("left_wrist_pitch_joint") > 0
    defaults = default_joint_pos_list()
    assert defaults[0] == -0.312
    assert defaults[3] == 0.669
    assert defaults[4] == -0.363

    position = torch.tensor([[1.0, 2.0, 3.0]])
    rotation = torch.eye(3).unsqueeze(0)
    linear_velocity = torch.tensor([[0.4, 0.5, 0.6]])
    angular_velocity = torch.tensor([[0.1, 0.2, 0.3]])
    zeros = torch.zeros(1, 29)
    body_p, body_R, body_v, body_w = kin.forward(
        position,
        rotation,
        linear_velocity,
        angular_velocity,
        zeros,
        zeros,
        all_bodies=True,
    )
    offset = torch.tensor([0.01, 0.02, -0.03])
    expected_p = position[0] + 29 * offset
    expected_v = linear_velocity[0] + torch.cross(angular_velocity[0], 29 * offset, dim=0)
    assert body_p.shape == (1, 30, 3)
    torch.testing.assert_close(body_p[0, -1], expected_p)
    torch.testing.assert_close(body_v[0, -1], expected_v)
    torch.testing.assert_close(body_R, torch.eye(3).expand(1, 30, 3, 3))
    torch.testing.assert_close(body_w, angular_velocity.expand(1, 30, 3))


def test_motion_resampling_fk_and_bins(library) -> None:
    _, motion, _ = library
    assert motion.num_frames == 109
    assert motion.clip_len.tolist() == [102, 7]
    assert motion.clip_start.tolist() == [0, 102]
    assert motion.clip_end.tolist() == [101, 108]
    assert motion.bin_start.tolist() == [0, 50, 100, 102]
    assert motion.bin_clip.tolist() == [0, 0, 0, 1]
    assert motion.frame_bin[torch.tensor([0, 49, 50, 99, 100, 101, 102])].tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
    ]
    assert motion.body_p.shape == (109, 14, 3)
    assert motion.body_R.shape == (109, 14, 3, 3)
    assert motion.body_v.shape == (109, 14, 3)
    assert motion.body_w.shape == (109, 14, 3)

    torch.testing.assert_close(motion.root_p[1], torch.tensor([0.02, 0.002, 0.76]))
    torch.testing.assert_close(motion.root_v[1], torch.tensor([1.0, 0.1, 0.0]))
    torch.testing.assert_close(motion.q[1, 0], torch.tensor(default_joint_pos_list()[0] + 0.0006))
    torch.testing.assert_close(motion.qd[1], torch.full((29,), 0.03))
    torch.testing.assert_close(motion.root_w, torch.zeros_like(motion.root_w))


def test_future_goal_clamps_at_clip_boundaries(library) -> None:
    _, motion, mdp = library
    frames = torch.tensor([100, 105])
    future = motion.future_frames(frames, 4)
    assert future.tolist() == [[101, 101, 101, 101], [106, 107, 108, 108]]
    goal = mdp.goal(frames, 4)
    assert goal.shape == (2, 4 * mdp.goal_frame_dim)
    torch.testing.assert_close(goal[0, :3], torch.tensor([0.02, 0.002, 0.0]))
    torch.testing.assert_close(goal[0, 3:6], goal[0, 76:79])

    first_goal = motion.goal(torch.tensor([0]), 1)
    assert first_goal.shape == (1, 73)
    torch.testing.assert_close(first_goal[0, :3], torch.tensor([0.02, 0.002, 0.0]))
    torch.testing.assert_close(first_goal[0, 9:12], torch.tensor([1.0, 0.1, 0.0]))


def test_observation_context_critic_and_clip_queries(library) -> None:
    _, motion, mdp = library
    assert isinstance(mdp, SuperMimicMdpProtocol)
    assert (mdp.num_joints, mdp.obs_dim, mdp.context_dim) == (29, 99, 33)
    frames = torch.tensor([0, 102])
    state = motion.ref_state(frames)
    previous_action = torch.zeros(2, 29)
    context = torch.zeros(2, 33)
    obs = mdp.obs(state, previous_action, context, frames)
    assert obs.shape == (2, 99)
    torch.testing.assert_close(obs[:, :29], torch.zeros(2, 29))
    torch.testing.assert_close(
        obs[:, -6:], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expand(2, 6)
    )
    noisy = mdp.add_noise(obs)
    assert noisy.shape == obs.shape
    torch.testing.assert_close(noisy[:, 64:93], previous_action)

    context[0, 0] = 0.75
    context[0, 1:30] = 0.01
    context[0, 30:33] = 0.05
    normalized = mdp.normalize_context(context)
    torch.testing.assert_close(normalized[0], torch.cat((torch.zeros(1), torch.ones(32))))
    critic = mdp.critic_features(state, frames, normalized)
    assert critic.shape == (2, INV_DIM + CRITIC_EXTRA)

    query_frames = torch.tensor([0, 50, 100, 102, 108])
    info = mdp.clip_info(query_frames)
    assert info["clip_id"].tolist() == [0, 0, 0, 1, 1]
    assert info["clip_start"].tolist() == [0, 0, 0, 102, 102]
    assert info["clip_end"].tolist() == [101, 101, 101, 108, 108]
    assert info["frame_bin"].tolist() == [0, 1, 2, 3, 3]
    for actual, expected in zip(mdp.ref_state(frames), state):
        torch.testing.assert_close(actual, expected)

    valid = mdp.valid_imagination_start(torch.tensor([98, 99, 105, 106]), 3)
    assert valid.tolist() == [True, False, True, False]


def test_observation_supports_bptt_history(library) -> None:
    _, motion, mdp = library
    frames = torch.tensor([[0, 1, 2], [102, 103, 104]])
    state = motion.ref_state(frames)
    previous_action = torch.zeros(2, 3, 29)
    context = torch.zeros(2, 3, 33)

    obs = mdp.obs(state, previous_action, context, frames)

    assert obs.shape == (2, 3, 99)
    torch.testing.assert_close(obs[..., :29], state[4] - mdp.default_q)


def test_mdp_rejects_batched_default_joint_positions(library) -> None:
    kin, motion, _ = library
    with pytest.raises(ValueError, match="default_q must have shape"):
        MDP(kin, motion, torch.zeros(2, 29))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mdp_to_copies_tensors_without_touching_original(library) -> None:
    kin, motion, mdp = library
    assert mdp.to("cpu") is mdp
    assert motion.to("cpu") is motion
    assert kin.to("cpu") is kin

    moved = mdp.to("mps")

    assert moved is not mdp
    assert moved.default_q.device.type == "mps"
    assert moved.motion.root_p.device.type == "mps"
    assert moved.motion.frame_bin.device.type == "mps"
    assert moved.kin.off_p.device.type == "mps"
    assert mdp.default_q.device.type == "cpu"
    assert mdp.motion.root_p.device.type == "cpu"

    frames = torch.tensor([0, 51, 102, 108], device="mps")
    goal = moved.goal(frames, 2)
    assert goal.device.type == "mps"
    torch.testing.assert_close(goal.cpu(), mdp.goal(frames.cpu(), 2))
    state = moved.motion.ref_state(frames)
    zeros = torch.zeros(4, 29, device="mps")
    reward = moved.reward(state, frames, zeros, zeros)
    torch.testing.assert_close(reward.cpu(), torch.full((4,), 0.1), rtol=0, atol=2e-6)


def test_perfect_tracking_reward_and_termination(library) -> None:
    _, motion, mdp = library
    frames = torch.tensor([0, 51, 102, 108])
    state = motion.ref_state(frames)
    action = torch.zeros(4, 29)
    reward, terms = mdp.reward(state, frames, action, action, terms=True)
    assert MAX_STEP_REWARD == pytest.approx(0.1)
    torch.testing.assert_close(reward, torch.full((4,), 0.1), rtol=0, atol=2e-6)
    torch.testing.assert_close(terms["track"], reward, rtol=0, atol=2e-6)
    torch.testing.assert_close(terms["e_root_pos"], torch.zeros(4))
    assert not mdp.termination(state, frames).any()
    assert (mdp.soft_termination(state, frames) < 1e-4).all()

    rate_reward = mdp.reward(state, frames, torch.ones_like(action), action)
    torch.testing.assert_close(rate_reward, reward - 0.1 * 29 * 0.02, rtol=0, atol=2e-6)

    height_state = tuple(value.clone() for value in state)
    height_state[0][0, 2] += 0.26
    terminated = mdp.termination(height_state, frames)
    assert terminated.tolist() == [True, False, False, False]
    assert mdp.soft_termination(height_state, frames)[0] > 0.5

    tilt_state = tuple(value.clone() for value in state)
    angle = 1.6
    tilt_state[1][1] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
            [0.0, torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))],
        ]
    )
    assert mdp.termination(tilt_state, frames).tolist() == [False, True, False, False]
