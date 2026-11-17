#!/usr/bin/env python3
"""Kinematic G1 NPZ replay with MuJoCo foot-contact force visualization.

The NPZ stores only kinematics. This script teleports MuJoCo to each retargeted
frame, calls mj_forward(), reads contacts via mj_contactForce(), and draws
aggregate foot-force arrows. Press C and F in the viewer for MuJoCo's native
contact points and contact-force visualization.

Usage:
  uv run python deploy/uni_g1/scripts/replay_contact_forces.py \
    --npz deploy/uni_g1/assets/clips/Loop_Forward_Walk_001__A019.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    bundle = here.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--xml", type=Path,
                    default=bundle / "assets" / "models" / "g1_train_scene.xml")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--use-vel", action="store_true",
                    help="also set qvel from root/joint velocities (default: zero qvel)")
    ap.add_argument("--no-custom-force", action="store_true",
                    help="show only MuJoCo native C/F visuals, not aggregate foot arrows")
    ap.add_argument("--force-scale", type=float, default=0.0012,
                    help="arrow length in meters per newton (default: 0.0012)")
    return ap.parse_args()


def add_arrow(scn: mujoco.MjvScene, start: np.ndarray, force: np.ndarray,
              scale: float, rgba: np.ndarray) -> None:
    if scn.ngeom >= scn.maxgeom - 1:
        return
    force = np.asarray(force, dtype=np.float64)
    norm = float(np.linalg.norm(force))
    if norm < 1.0:
        return
    direction = force / norm
    end = np.asarray(start, dtype=np.float64) + direction * min(norm * scale, 0.65)
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    width = 0.004 + 0.005 * min(norm / 600.0, 1.5)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CYLINDER, width,
                         np.asarray(start, dtype=np.float64), end)
    scn.ngeom += 1


def foot_contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, dict]:
    sides = {
        "L": np.zeros(3),
        "R": np.zeros(3),
    }
    result = {
        "L": {"force": np.zeros(3), "position": None, "count": 0},
        "R": {"force": np.zeros(3), "position": None, "count": 0},
    }
    positions = {"L": [], "R": []}

    for i in range(data.ncon):
        con = data.contact[i]
        b1 = int(model.geom_bodyid[con.geom1])
        b2 = int(model.geom_bodyid[con.geom2])
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
        left = n1.startswith("left_ankle") or n2.startswith("left_ankle")
        right = n1.startswith("right_ankle") or n2.startswith("right_ankle")
        if not ((left or right) and (b1 == 0 or b2 == 0)):
            continue

        wrench_local = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench_local)
        frame = np.asarray(con.frame).reshape(3, 3)
        force_world_on_geom1 = frame @ wrench_local[:3]

        # mj_contactForce reports the force on geom1. Return the reaction on the
        # robot's foot, which is the useful quantity for gait/contact analysis.
        if b1 == 0:
            force_on_foot = -force_world_on_geom1
        else:
            force_on_foot = force_world_on_geom1

        side = "L" if left else "R"
        sides[side] += force_on_foot
        positions[side].append(np.asarray(con.pos, dtype=np.float64))

    for side in ("L", "R"):
        count = len(positions[side])
        result[side]["force"] = sides[side]
        result[side]["count"] = count
        result[side]["position"] = (
            np.mean(np.stack(positions[side]), axis=0) if count else None
        )
    return result


def set_frame(model: mujoco.MjModel, data: mujoco.MjData, clip: dict, frame: int,
              use_vel: bool) -> None:
    joint_names = clip["joint_names"]
    data.qpos[:3] = clip["root_pos"][frame]
    quat = clip["root_quat_wxyz"][frame] / max(
        np.linalg.norm(clip["root_quat_wxyz"][frame]), 1e-9
    )
    data.qpos[3:7] = quat
    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid < 0:
            raise ValueError(f"model has no joint {name!r}")
        adr = int(model.jnt_qposadr[jid])
        dof = int(model.jnt_dofadr[jid])
        data.qpos[adr] = clip["joint_pos"][frame, i]
        data.ctrl[model.actuator_trnid[:, 0] == jid] = clip["joint_pos"][frame, i]
        if use_vel:
            data.qvel[dof] = clip["joint_vel"][frame, i]

    if use_vel:
        data.qvel[:3] = clip["root_lin_vel_w"][frame]
        data.qvel[3:6] = clip["root_ang_vel_w"][frame]
    else:
        data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.time = frame / float(clip["fps"])
    mujoco.mj_forward(model, data)


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")

    with np.load(args.npz, allow_pickle=False) as archive:
        required = {
            "joint_names", "joint_pos", "joint_vel", "root_pos",
            "root_quat_wxyz", "root_lin_vel_w", "root_ang_vel_w", "fps",
        }
        missing = required.difference(archive.files)
        if missing:
            raise SystemExit(f"{args.npz}: missing {sorted(missing)}")
        clip = {key: np.asarray(archive[key]) for key in archive.files}

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    num_frames = int(clip["joint_pos"].shape[0])

    print(f"clip={Path(args.npz).name} frames={num_frames} fps={float(clip['fps']):.0f}")
    print("viewer: press C=contact points, F=contact forces; blue=left, orange=right")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.75]
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 105.0
        viewer.cam.elevation = -12.0

        next_frame_time = time.monotonic()
        last_print = -1.0
        frame = 0
        while viewer.is_running():
            now = time.monotonic()
            if now >= next_frame_time:
                set_frame(model, data, clip, frame, args.use_vel)
                forces = foot_contact_force(model, data)

                if not args.no_custom_force:
                    viewer.user_scn.ngeom = 0
                    add_arrow(
                        viewer.user_scn, forces["L"]["position"], forces["L"]["force"],
                        args.force_scale, [0.15, 0.45, 1.0, 0.95],
                    )
                    add_arrow(
                        viewer.user_scn, forces["R"]["position"], forces["R"]["force"],
                        args.force_scale, [1.0, 0.45, 0.05, 0.95],
                    )

                stamp = frame / float(clip["fps"])
                if stamp - last_print >= 0.20:
                    last_print = stamp
                    lf = forces["L"]["force"]
                    rf = forces["R"]["force"]
                    print(
                        f"frame={frame:03d}/{num_frames - 1:03d} t={stamp:5.2f}s  "
                        f"L n={forces['L']['count']} |F|={np.linalg.norm(lf):6.1f} Fz={lf[2]:6.1f}  "
                        f"R n={forces['R']['count']} |F|={np.linalg.norm(rf):6.1f} Fz={rf[2]:6.1f}",
                        flush=True,
                    )

                viewer.sync()
                frame = (frame + 1) % num_frames
                next_frame_time += 1.0 / (float(clip["fps"]) * args.speed)
                if next_frame_time < now - 0.5:
                    next_frame_time = now
            else:
                time.sleep(min(0.002, max(0.0, next_frame_time - now)))


if __name__ == "__main__":
    main()
