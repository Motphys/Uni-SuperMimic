"""Static contract tests for the generated G1 SuperMimic assets."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import PurePosixPath

import pytest
from tools import prepare_supermimic_assets as prepare

from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
    COLLISION_GEOM_NAMES,
    DAMPING_RATIO,
    G1_JOINT_NAMES,
    NATURAL_FREQ,
    TRACKED_BODIES,
    default_joint_pos_list,
    motor_of,
)

JOINT_RANGES = {
    "left_hip_pitch_joint": (-2.5307, 2.8798),
    "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798),
    "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576),
    "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618),
    "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.972222054, 1.972222054),
    "left_wrist_pitch_joint": (-1.614429558, 1.614429558),
    "left_wrist_yaw_joint": (-1.614429558, 1.614429558),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.972222054, 1.972222054),
    "right_wrist_pitch_joint": (-1.614429558, 1.614429558),
    "right_wrist_yaw_joint": (-1.614429558, 1.614429558),
}


def robot_root() -> ET.Element:
    return ET.parse(prepare.ROBOT_XML).getroot()


def test_robot_preserves_reference_kinematics_and_tracked_frames():
    root = robot_root()
    compiler = root.find("compiler")
    assert compiler is not None
    assert compiler.get("meshdir") == "assets"
    assert compiler.get("fusestatic") == "false"
    assert compiler.get("discardvisual") == "false"
    assert root.find("keyframe") is None

    pelvis = root.find("./worldbody/body[@name='pelvis']")
    assert pelvis is not None
    assert pelvis.find("./freejoint[@name='floating_base_joint']") is not None
    joints = {joint.get("name"): joint for joint in root.findall(".//joint")}
    assert list(joints) == G1_JOINT_NAMES
    for name, joint in joints.items():
        assert joint.get("type", "hinge") == "hinge"
        assert tuple(float(value) for value in joint.get("range", "").split()) == JOINT_RANGES[name]

    bodies = {body.get("name") for body in root.findall(".//body")}
    assert set(TRACKED_BODIES) <= bodies


def test_actuators_match_reference_implicit_pd_and_effort():
    root = robot_root()
    joints = {joint.get("name"): joint for joint in root.findall(".//joint")}
    actuators = root.findall("./actuator/position")
    assert [actuator.get("joint") for actuator in actuators] == G1_JOINT_NAMES
    for actuator in actuators:
        name = actuator.get("joint", "")
        effort, _, armature = motor_of(name)
        assert float(joints[name].get("armature", "nan")) == armature
        assert float(actuator.get("kp", "nan")) == pytest.approx(
            armature * NATURAL_FREQ**2, rel=1e-14
        )
        assert float(actuator.get("kv", "nan")) == pytest.approx(
            2.0 * DAMPING_RATIO * armature * NATURAL_FREQ, rel=1e-14
        )
        assert tuple(float(value) for value in actuator.get("forcerange", "").split()) == (
            -effort,
            effort,
        )


def test_mesh_references_are_portable_and_cover_reference_urdf():
    root = robot_root()
    meshes = root.findall("./asset/mesh")
    assert len(meshes) == 35
    for mesh in meshes:
        path = PurePosixPath(mesh.get("file", ""))
        assert not path.is_absolute()
        assert path.parts == (path.name,)
        assert path.suffix == ".STL"


def test_collision_geoms_disable_self_contact_and_keep_ground_contact():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(prepare.SCENE_XML))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] != 0
        and (model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0)
    ]
    assert floor_id >= 0
    assert tuple(model.geom(geom_id).name for geom_id in robot_ids) == COLLISION_GEOM_NAMES
    assert len(robot_ids) == 36
    for geom_id in robot_ids:
        assert model.geom_contype[geom_id] == 1
        assert model.geom_conaffinity[geom_id] == 0
        for other_id in robot_ids:
            assert not (
                (model.geom_contype[geom_id] & model.geom_conaffinity[other_id])
                or (model.geom_contype[other_id] & model.geom_conaffinity[geom_id])
            )
        assert (model.geom_contype[geom_id] & model.geom_conaffinity[floor_id]) or (
            model.geom_contype[floor_id] & model.geom_conaffinity[geom_id]
        )

    data = mujoco.MjData(model)
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    data.qpos[2] = 0.7
    mujoco.mj_forward(model, data)
    pairs = [(contact.geom1, contact.geom2) for contact in data.contact[: data.ncon]]
    assert any(floor_id in pair for pair in pairs)
    assert any(
        floor_id in pair and "ankle" in (model.geom(pair[0]).name + model.geom(pair[1]).name)
        for pair in pairs
    )
    assert all(floor_id in pair for pair in pairs)


def test_scene_has_training_physics_and_default_stand_controls():
    root = ET.parse(prepare.SCENE_XML).getroot()
    assert root.find("./include[@file='supermimic.xml']") is not None
    option = root.find("option")
    assert option is not None
    assert option.get("integrator") == "implicitfast"
    assert option.get("cone") == "pyramidal"
    assert float(option.get("timestep", "nan")) == prepare.TIMESTEP
    floor = root.find("./worldbody/geom[@name='floor']")
    assert floor is not None
    assert float(floor.get("friction", "").split()[0]) == prepare.GROUND_FRICTION

    stand = root.find("./keyframe/key[@name='stand']")
    assert stand is not None
    qpos = [float(value) for value in stand.get("qpos", "").split()]
    ctrl = [float(value) for value in stand.get("ctrl", "").split()]
    assert qpos == [0.0, 0.0, prepare.ROOT_HEIGHT, 1.0, 0.0, 0.0, 0.0, *default_joint_pos_list()]
    assert ctrl == default_joint_pos_list()


def test_preparation_script_uses_expected_remote_and_ignored_locations():
    assert prepare.REPO_ID == "lvhaidong/LAFAN1_Retargeting_Dataset"
    assert prepare.EXPECTED_CLIPS == 40
    assert prepare.EXPECTED_FRAMES == 264705
    assert prepare.CSV_COLUMNS == 36
    assert prepare.URDF_PATH.is_relative_to(prepare.MESH_DIR)
    assert prepare.ROBOT_XML.name == "supermimic.xml"
    assert prepare.SCENE_XML.name == "scene_supermimic.xml"
    assert prepare.MOTION_DIR.name == "lafan1_supermimic"


def test_robot_only_preparation_skips_motion_download_and_validation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        prepare,
        "download_assets",
        lambda *, include_motions: calls.append(("download", include_motions)),
    )
    monkeypatch.setattr(prepare, "generate_robot_xml", lambda: calls.append("generate"))
    monkeypatch.setattr(prepare, "validate_motions", lambda: calls.append("motions"))
    monkeypatch.setattr(prepare, "validate_robot", lambda: calls.append("robot"))
    monkeypatch.setattr(prepare, "validate_scene", lambda: calls.append("scene"))
    monkeypatch.setattr("sys.argv", ["prepare_supermimic_assets.py", "--robot-only"])

    prepare.main()

    assert calls == [("download", False), "generate", "robot", "scene"]
