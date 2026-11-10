import numpy as np
import pytest
import torch

from unisupermimic.deploy.runtime import DeployConfig, UniClip, UniDeployCore
from unisupermimic.deploy.sim2sim import DEFAULT_ASSETS, DEFAULT_MJCF, export_clips
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


@pytest.mark.slow
def test_deploy_runtime_matches_uni_mdp_observation_and_goal(tmp_path):
    motion = DEFAULT_MJCF.parents[2] / "motions/g1/lafan1_supermimic/walk1_subject1.csv"
    if not DEFAULT_ASSETS.is_dir() or not motion.is_file():
        pytest.skip("full G1 assets and LAFAN fixture are not distributed")
    from unisupermimic.deploy.sim2sim import write_config
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import default_joint_pos_list

    checkpoint = tmp_path / "ckpt_000001.pt"
    checkpoint.touch()
    config_path = write_config(checkpoint, tmp_path, DEFAULT_MJCF)
    config = DeployConfig.load(config_path)
    exported = export_clips([motion], tmp_path, mjcf=DEFAULT_MJCF, src_fps=30.0)[0]
    clip = UniClip.load(exported)

    def policy(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.zeros(29)

    core = UniDeployCore(config, policy, clip)
    kin = G1Kinematics(str(DEFAULT_MJCF), "cpu")
    lib = MotionLib([str(motion)], kin, "cpu", src_fps=30.0)
    mdp = MDP(kin, lib, torch.tensor(default_joint_pos_list()))
    for frame in (0, 1, 77, clip.num_frames // 2, clip.num_frames - 1):
        core.reset(frame)
        root_pos, root_rot, _, root_w, q, qd = lib.ref_state(torch.tensor([frame]))
        previous_action = np.linspace(-0.2, 0.2, 29, dtype=np.float64)
        core.last_action = previous_action.copy()
        gyro = root_rot.numpy()[0].T @ root_w.numpy()[0]
        actual_obs = core.observation(
            q.numpy()[0],
            qd.numpy()[0],
            gyro,
            clip.root_quat_wxyz[frame],
        )
        state = (
            root_pos,
            root_rot,
            torch.zeros_like(root_pos),
            root_w,
            q,
            qd,
        )
        expected_obs = mdp.obs(
            state,
            torch.as_tensor(previous_action)[None],
            torch.zeros(1, 33),
            torch.tensor([frame]),
        ).numpy()[0]
        np.testing.assert_allclose(actual_obs.reshape(10, 99)[-1], expected_obs, rtol=0, atol=2e-7)
        np.testing.assert_allclose(
            core.goal(), mdp.goal(torch.tensor([frame]), 10).numpy()[0], atol=0
        )
