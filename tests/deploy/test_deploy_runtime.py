import numpy as np
import pytest

from unisupermimic.deploy.runtime import (
    NUM_JOINTS,
    DeployConfig,
    UniClip,
    UniDeployCore,
)


def _clip(frames: int = 4) -> UniClip:
    return UniClip(
        name="test",
        goal=np.zeros((frames, 730), dtype=np.float32),
        joint_pos=np.zeros((frames, NUM_JOINTS)),
        joint_vel=np.zeros((frames, NUM_JOINTS)),
        proj_grav=np.tile([[0.0, 0.0, -1.0]], (frames, 1)),
        root_pos=np.zeros((frames, 3)),
        root_quat_wxyz=np.tile([[1.0, 0.0, 0.0, 0.0]], (frames, 1)),
        root_lin_vel_w=np.zeros((frames, 3)),
        root_ang_vel_w=np.zeros((frames, 3)),
    )


def _config() -> DeployConfig:
    return DeployConfig(
        contract="unisupermimic_v1",
        joint_names=[f"joint_{i}" for i in range(NUM_JOINTS)],
        default_joint_pos=np.arange(NUM_JOINTS, dtype=float) * 0.01,
        action_scale=np.full(NUM_JOINTS, 0.25),
        kp=np.full(NUM_JOINTS, 40.0),
        kd=np.full(NUM_JOINTS, 2.0),
        effort_limit=np.full(NUM_JOINTS, 25.0),
        joint_lower=np.full(NUM_JOINTS, -1.0),
        joint_upper=np.full(NUM_JOINTS, 1.0),
        velocity_limit=np.full(NUM_JOINTS, 20.0),
    )


def test_uni_runtime_uses_exact_observation_goal_and_target_contract():
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def policy(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        calls.append((obs.copy(), goal.copy()))
        action = np.zeros(NUM_JOINTS)
        action[0] = len(calls)
        return action

    core = UniDeployCore(_config(), policy, _clip())
    q = np.full(NUM_JOINTS, 0.1)
    dq = np.full(NUM_JOINTS, 0.01)
    gyro = np.array([0.1, -0.2, 0.3])
    quat = np.array([1.0, 0.0, 0.0, 0.0])

    first = core.step(q, dq, gyro, quat)
    second = core.step(q, dq, gyro, quat)

    assert calls[0][0].shape == (990,)
    assert calls[0][1].shape == (730,)
    first_row = calls[0][0].reshape(10, 99)[-1]
    second_row = calls[1][0].reshape(10, 99)[-1]
    np.testing.assert_allclose(first_row[:29], q - core.config.default_joint_pos)
    np.testing.assert_allclose(second_row[29:58], dq)
    np.testing.assert_allclose(second_row[58:61], gyro)
    np.testing.assert_allclose(first_row[64:93], np.zeros(NUM_JOINTS))
    np.testing.assert_allclose(second_row[64:93], first.action)
    np.testing.assert_allclose(first_row[93:99], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(calls[0][1], core.clip.goal[0])
    np.testing.assert_allclose(calls[1][1], core.clip.goal[1])
    expected_target = core.config.default_joint_pos.copy()
    expected_target[0] += 0.25
    np.testing.assert_allclose(first.q_target, expected_target)
    assert second.frame == 2
    assert first.abort is None
    assert second.abort is None


def test_uni_runtime_rejects_changed_contract():
    def policy(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.zeros(NUM_JOINTS)

    config = _config()
    config.contract = "invalid"
    with pytest.raises(ValueError, match="unisupermimic_v1"):
        UniDeployCore(config, policy, _clip())
