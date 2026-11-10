from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from unisupermimic.deploy.sim2sim import DEFAULT_ASSETS, DEFAULT_MJCF, DEFAULT_SCENE
from unisupermimic.deploy.viewer import ViewerRobot, build_ghost_model


@pytest.mark.slow
def test_ghost_viewer_model_keeps_one_actuated_robot_and_contact_free_reference():
    if not DEFAULT_ASSETS.is_dir():
        pytest.skip("full G1 assets are not distributed")
    from unisupermimic.deploy.runtime import NUM_JOINTS
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES

    model = build_ghost_model(DEFAULT_SCENE, DEFAULT_MJCF)
    assert model.nu == NUM_JOINTS
    free_joints = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    ]
    assert len(free_joints) == 2
    ghost_contacts = sum(
        1
        for geom in range(model.ngeom)
        if model.body(model.geom_bodyid[geom]).name.startswith("ref_")
        and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    )
    assert ghost_contacts == 0

    config = SimpleNamespace(joint_names=G1_JOINT_NAMES)
    robot = ViewerRobot(model, config, ghost=True)  # type: ignore[arg-type]
    root_pos = np.array([0.0, 0.0, 0.8])
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.zeros(NUM_JOINTS)
    robot.set_state(root_pos, root_quat, q)
    assert robot.state()[0].shape == (NUM_JOINTS,)
