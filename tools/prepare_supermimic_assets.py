#!/usr/bin/env python3
"""Download and validate the G1 SuperMimic motion and MJCF assets.

Run from any working directory with:
    uv run scripts/tools/prepare_supermimic_assets.py

Use --validate-only to check an existing local asset installation without network access.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
    COLLISION_GEOM_NAMES,
    DAMPING_RATIO,
    G1_JOINT_NAMES,
    NATURAL_FREQ,
    TRACKED_BODIES,
    default_joint_pos_list,
    motor_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
G1_DIR = REPO_ROOT / "src" / "unisupermimic" / "assets" / "robots" / "g1"
MESH_DIR = G1_DIR / "assets"
URDF_PATH = MESH_DIR / "supermimic" / "g1_29dof_rev_1_0.urdf"
ROBOT_XML = G1_DIR / "supermimic.xml"
SCENE_XML = G1_DIR / "scene_supermimic.xml"
MOTION_DIR = REPO_ROOT / "src" / "unisupermimic" / "assets" / "motions" / "g1" / "lafan1_supermimic"

REPO_ID = "lvhaidong/LAFAN1_Retargeting_Dataset"
URDF_REMOTE_PATH = "robot_description/g1/g1_29dof_rev_1_0.urdf"
EXPECTED_CLIPS = 40
EXPECTED_FRAMES = 264705
CSV_COLUMNS = 36
ROOT_HEIGHT = 0.76
GROUND_FRICTION = 0.3
TIMESTEP = 0.005


def download_assets(*, include_motions: bool = True) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    remote_files = set(HfApi().list_repo_files(REPO_ID, repo_type="dataset"))
    motion_paths = sorted(path for path in remote_files if re.fullmatch(r"g1/[^/]+\.csv", path))
    if include_motions and len(motion_paths) != EXPECTED_CLIPS:
        raise RuntimeError(
            f"Expected {EXPECTED_CLIPS} remote motion clips, found {len(motion_paths)}"
        )
    if URDF_REMOTE_PATH not in remote_files:
        raise FileNotFoundError(f"Missing remote URDF: {URDF_REMOTE_PATH}")

    work_dir = REPO_ROOT / ".tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    reused = 0
    with tempfile.TemporaryDirectory(prefix="supermimic-", dir=work_dir) as temp_dir:

        def fetch(remote_path: str, target: Path) -> None:
            nonlocal downloaded, reused
            if target.is_file() and target.stat().st_size > 0:
                reused += 1
                return
            source = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=remote_path,
                    repo_type="dataset",
                    local_dir=temp_dir,
                )
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            downloaded += 1

        fetch(URDF_REMOTE_PATH, URDF_PATH)
        if include_motions:
            for remote_path in motion_paths:
                fetch(remote_path, MOTION_DIR / PurePosixPath(remote_path).name)
        for mesh_name in required_mesh_names(URDF_PATH):
            remote_path = f"robot_description/g1/meshes/{mesh_name}"
            if remote_path not in remote_files:
                raise FileNotFoundError(f"Missing remote mesh: {remote_path}")
            fetch(remote_path, MESH_DIR / mesh_name)

    print(f"Asset download: {downloaded} downloaded, {reused} reused")


def required_mesh_names(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    names = sorted({PurePosixPath(mesh.get("filename", "")).name for mesh in root.iter("mesh")})
    if not names or any(not name for name in names):
        raise ValueError(f"No valid mesh references in {urdf_path}")
    return names


def generate_robot_xml() -> None:
    import mujoco

    urdf_root = ET.parse(URDF_PATH).getroot()
    mujoco_element = urdf_root.find("mujoco")
    if mujoco_element is None:
        mujoco_element = ET.SubElement(urdf_root, "mujoco")
    compiler = mujoco_element.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_element, "compiler")
    compiler.attrib.clear()
    compiler.attrib.update(
        {
            "meshdir": "../assets",
            "balanceinertia": "true",
            "discardvisual": "false",
            "fusestatic": "false",
            "strippath": "false",
        }
    )
    for mesh in urdf_root.iter("mesh"):
        mesh.set("filename", PurePosixPath(mesh.get("filename", "")).name)

    with tempfile.TemporaryDirectory(prefix=".supermimic-build-", dir=G1_DIR) as temp_dir:
        temp_urdf = Path(temp_dir) / "g1_29dof_rev_1_0.urdf"
        temp_xml = Path(temp_dir) / "compiled.xml"
        ET.ElementTree(urdf_root).write(temp_urdf, encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(temp_urdf))
        mujoco.mj_saveLastXML(str(temp_xml), model)
        robot_root = ET.parse(temp_xml).getroot()

    robot_root.set("model", "g1_29dof_supermimic")
    compiler = robot_root.find("compiler")
    if compiler is None:
        raise ValueError("Compiled MJCF has no compiler element")
    compiler.set("meshdir", "assets")
    compiler.set("fusestatic", "false")
    compiler.set("discardvisual", "false")
    compiler.set("strippath", "false")

    collision_names = []
    for body in robot_root.iter("body"):
        body_name = body.get("name")
        if not body_name:
            raise ValueError("Compiled MJCF has an unnamed body")
        collision_geoms = [
            geom
            for geom in body.findall("geom")
            if geom.get("contype") != "0" or geom.get("conaffinity") != "0"
        ]
        for index, geom in enumerate(collision_geoms, start=1):
            suffix = "" if len(collision_geoms) == 1 else f"_{index}"
            name = f"{body_name}_collision{suffix}"
            geom.set("name", name)
            geom.set("contype", "1")
            geom.set("conaffinity", "0")
            collision_names.append(name)
    if tuple(collision_names) != COLLISION_GEOM_NAMES:
        raise ValueError("Compiled MJCF collision geometry differs from the expected G1 set")

    pelvis = robot_root.find("./worldbody/body[@name='pelvis']")
    if pelvis is None:
        raise ValueError("Compiled MJCF has no pelvis body")
    if pelvis.find("freejoint") is not None:
        raise ValueError("Compiled MJCF unexpectedly already has a pelvis freejoint")
    freejoint = ET.Element("freejoint", {"name": "floating_base_joint"})
    pelvis.insert(0, freejoint)

    joints = {joint.get("name"): joint for joint in robot_root.iter("joint")}
    if set(G1_JOINT_NAMES) != {name for name in joints if name in G1_JOINT_NAMES}:
        raise ValueError("Compiled MJCF does not contain the expected 29 hinges")
    for name in G1_JOINT_NAMES:
        joints[name].set("armature", format_float(motor_of(name)[2]))

    bodies = {body.get("name"): body for body in robot_root.iter("body")}
    for urdf_joint in urdf_root.findall("joint"):
        child = urdf_joint.find("child")
        origin = urdf_joint.find("origin")
        if child is not None and origin is not None:
            bodies[child.get("link")].set("pos", origin.get("xyz", "0 0 0"))
        if urdf_joint.get("type") == "revolute":
            limit = urdf_joint.find("limit")
            if limit is None:
                raise ValueError(f"Revolute URDF joint has no limit: {urdf_joint.get('name')}")
            joints[urdf_joint.get("name")].set(
                "range", f"{limit.get('lower')} {limit.get('upper')}"
            )

    old_actuators = robot_root.find("actuator")
    if old_actuators is not None:
        robot_root.remove(old_actuators)
    actuators = ET.SubElement(robot_root, "actuator")
    for name in G1_JOINT_NAMES:
        effort, _, armature = motor_of(name)
        kp = armature * NATURAL_FREQ**2
        kd = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
        ET.SubElement(
            actuators,
            "position",
            {
                "name": name,
                "joint": name,
                "kp": format_float(kp),
                "kv": format_float(kd),
                "forcerange": f"{format_float(-effort)} {format_float(effort)}",
            },
        )

    ET.indent(robot_root, space="  ")
    text = ET.tostring(robot_root, encoding="unicode", short_empty_elements=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=G1_DIR,
        prefix="tmp_supermimic_",
        suffix=".xml",
        delete=False,
    ) as output:
        output.write(text)
        output_path = Path(output.name)
    try:
        mujoco.MjModel.from_xml_path(str(output_path))
        os.replace(output_path, ROBOT_XML)
    finally:
        output_path.unlink(missing_ok=True)
    print(f"Generated {ROBOT_XML}")


def format_float(value: float) -> str:
    return format(value, ".17g")


def validate_motions() -> None:
    paths = sorted(MOTION_DIR.glob("*.csv"))
    if len(paths) != EXPECTED_CLIPS:
        raise ValueError(f"Expected {EXPECTED_CLIPS} CSV clips in {MOTION_DIR}, found {len(paths)}")
    total_frames = 0
    for path in paths:
        frames = 0
        with path.open(newline="", encoding="utf-8") as stream:
            for line_number, row in enumerate(csv.reader(stream), start=1):
                if not row:
                    continue
                if len(row) != CSV_COLUMNS:
                    raise ValueError(
                        f"{path.name}:{line_number}: expected {CSV_COLUMNS} columns, found {len(row)}"
                    )
                if not all(math.isfinite(float(value)) for value in row):
                    raise ValueError(f"{path.name}:{line_number}: non-finite motion value")
                frames += 1
        total_frames += frames
    if total_frames != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} frames, found {total_frames}")
    print(
        f"Motions: {len(paths)} clips, {total_frames} frames at source 30 fps, "
        f"{CSV_COLUMNS} columns per frame"
    )


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label}: expected {expected:.17g}, found {actual:.17g}")


def validate_robot() -> None:
    import mujoco

    robot_root = ET.parse(ROBOT_XML).getroot()
    compiler = robot_root.find("compiler")
    if compiler is None or compiler.get("fusestatic") != "false":
        raise ValueError("SuperMimic MJCF must set fusestatic=false")
    if compiler.get("meshdir") != "assets":
        raise ValueError("SuperMimic MJCF meshdir must be assets")
    if robot_root.find("keyframe") is not None:
        raise ValueError("Task keyframes are not allowed in the robot XML")

    xml_meshes = robot_root.findall("./asset/mesh")
    xml_mesh_names = {PurePosixPath(mesh.get("file", "")).name for mesh in xml_meshes}
    expected_mesh_names = set(required_mesh_names(URDF_PATH))
    if xml_mesh_names != expected_mesh_names:
        raise ValueError("MJCF mesh set differs from the reference URDF")
    for mesh in xml_meshes:
        relative = PurePosixPath(mesh.get("file", ""))
        if relative.is_absolute() or ".." in relative.parts or relative.name != relative.as_posix():
            raise ValueError(f"Non-portable mesh path: {relative}")
        mesh_path = MESH_DIR / relative.name
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Unresolved mesh: {mesh_path}")

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    hinge_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    hinge_names = [model.joint(joint_id).name for joint_id in hinge_ids]
    if hinge_names != G1_JOINT_NAMES:
        raise ValueError(f"Natural hinge order differs: {hinge_names}")
    if model.njnt != 30 or model.nq != 36 or model.nv != 35 or model.nu != 29:
        raise ValueError(
            f"Unexpected model dimensions: njnt={model.njnt} nq={model.nq} nv={model.nv} nu={model.nu}"
        )
    collision_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0
    ]
    collision_names = tuple(model.geom(geom_id).name for geom_id in collision_ids)
    if collision_names != COLLISION_GEOM_NAMES:
        raise ValueError(f"Collision geometry names differ: {collision_names}")
    if any(model.geom_contype[geom_id] != 1 for geom_id in collision_ids) or any(
        model.geom_conaffinity[geom_id] != 0 for geom_id in collision_ids
    ):
        raise ValueError("Robot collisions must use contype=1 and conaffinity=0")
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if free_id < 0 or model.jnt_type[free_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Missing pelvis floating_base_joint")
    if model.body(model.jnt_bodyid[free_id]).name != "pelvis":
        raise ValueError("floating_base_joint is not attached to pelvis")
    missing_bodies = [
        name
        for name in TRACKED_BODIES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0
    ]
    if missing_bodies:
        raise ValueError(f"Missing tracked bodies: {missing_bodies}")

    urdf_root = ET.parse(URDF_PATH).getroot()
    for urdf_joint in urdf_root.findall("joint"):
        name = urdf_joint.get("name", "")
        child = urdf_joint.find("child")
        origin = urdf_joint.find("origin")
        if child is None:
            continue
        body_name = child.get("link", "")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"URDF body was fused or omitted: {body_name}")
        expected_offset = (
            [float(value) for value in origin.get("xyz", "0 0 0").split()]
            if origin is not None
            else [0.0] * 3
        )
        for axis, expected in enumerate(expected_offset):
            assert_close(
                model.body_pos[body_id, axis], expected, f"{body_name} body offset[{axis}]"
            )
        if urdf_joint.get("type") != "revolute":
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        limit = urdf_joint.find("limit")
        if joint_id < 0 or limit is None:
            raise ValueError(f"Missing revolute joint data: {name}")
        for index, attribute in enumerate(("lower", "upper")):
            assert_close(
                model.jnt_range[joint_id, index],
                float(limit.get(attribute, "nan")),
                f"{name} {attribute}",
            )
        effort, _, armature = motor_of(name)
        assert_close(model.dof_armature[model.jnt_dofadr[joint_id]], armature, f"{name} armature")
        assert_close(float(limit.get("effort", "nan")), effort, f"{name} URDF effort")

    for actuator_id, name in enumerate(G1_JOINT_NAMES):
        actuator = model.actuator(actuator_id)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if (
            actuator.name != name
            or model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT
            or actuator.trnid[0] != joint_id
        ):
            raise ValueError(f"Actuator order or transmission differs at {name}")
        effort, _, armature = motor_of(name)
        kp = armature * NATURAL_FREQ**2
        kd = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
        assert_close(actuator.gainprm[0], kp, f"{name} Kp")
        assert_close(actuator.biasprm[1], -kp, f"{name} position bias")
        assert_close(actuator.biasprm[2], -kd, f"{name} Kd")
        assert_close(actuator.forcerange[0], -effort, f"{name} minimum effort")
        assert_close(actuator.forcerange[1], effort, f"{name} maximum effort")

    print(
        f"Robot: {len(hinge_ids)} natural-order hinges + pelvis freejoint, "
        f"{len(TRACKED_BODIES)} tracked frames, {len(xml_meshes)} meshes in {MESH_DIR}"
    )
    print("Robot numerics: URDF ranges/body offsets/efforts and reference armature/Kp/Kd match")


def validate_scene() -> None:
    import mujoco

    scene_root = ET.parse(SCENE_XML).getroot()
    include = scene_root.find("include")
    if include is None or include.get("file") != ROBOT_XML.name:
        raise ValueError("SuperMimic scene must include supermimic.xml")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    assert_close(model.opt.timestep, TIMESTEP, "scene timestep")
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise ValueError("SuperMimic scene must use implicitfast integration")
    if model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
        raise ValueError("SuperMimic scene must use pyramidal contacts")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("Missing ground plane")
    assert_close(model.geom_friction[floor_id, 0], GROUND_FRICTION, "ground friction")

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        raise ValueError("Missing scene stand keyframe")
    expected_qpos = [0.0, 0.0, ROOT_HEIGHT, 1.0, 0.0, 0.0, 0.0, *default_joint_pos_list()]
    expected_ctrl = default_joint_pos_list()
    for index, expected in enumerate(expected_qpos):
        assert_close(model.key_qpos[key_id, index], expected, f"stand qpos[{index}]")
    for index, expected in enumerate(expected_ctrl):
        assert_close(model.key_ctrl[key_id, index], expected, f"stand ctrl[{index}]")
    print(
        f"Scene: timestep={TIMESTEP}, implicit PD, ground friction={GROUND_FRICTION}, "
        f"stand root z={ROOT_HEIGHT}, default ctrl matches {len(expected_ctrl)} joints"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate local assets without downloading or regenerating",
    )
    parser.add_argument(
        "--robot-only",
        action="store_true",
        help="Prepare or validate only the shared G1 robot assets, without LAFAN1 motions",
    )
    args = parser.parse_args()
    if not args.validate_only:
        download_assets(include_motions=not args.robot_only)
        generate_robot_xml()
    if not args.robot_only:
        validate_motions()
    validate_robot()
    validate_scene()
    print("SuperMimic asset preparation and validation passed")


if __name__ == "__main__":
    main()
