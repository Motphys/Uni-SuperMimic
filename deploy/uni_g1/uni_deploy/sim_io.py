"""MuJoCo-backed IO for the deployment-equivalent sim2sim.

Implements the same read/send interface as the real robot IO:
  * physics: training scene MJCF (position actuators), stepped at the XML
    timestep; per-joint kp/kd are reprogrammed every tick, exactly like
    writing kp/kd/q_target to the motor driver;
  * read(): joint q/dq, pelvis local-frame angular velocity, pelvis quat;
  * buttons come from a scripted timeline or the viewer keyboard.
"""

from __future__ import annotations

import numpy as np


class SimIO:
    def __init__(self, scene_xml: str, clip, joint_names, keymap=None,
                 start: str = "clip_frame0", gantry: bool = False, ghost: bool = False,
                 sim_dt: float | None = None):
        import mujoco

        self.mj = mujoco
        if ghost:
            scene_xml = self._make_ghost_scene(scene_xml)
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        if sim_dt is not None:
            self.model.opt.timestep = sim_dt
        self.ghost_joint_adr = None
        self.ghost_dof_adr = None
        self.ghost_root = None
        if ghost:
            self._setup_ghost(joint_names)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 1280)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 720)
        self.clip = clip
        self.sim_dt = self.model.opt.timestep
        self.control_dt = 0.02
        self.substeps = int(round(self.control_dt / self.sim_dt))
        self.keys = 0
        self._keymap = keymap or {}

        # actuator index per joint, resolved via actuator->joint transmission
        self.act_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"scene has no joint {name}")
            matches = [a for a in range(self.model.nu) if self.model.actuator_trnid[a, 0] == jid]
            if not matches:
                raise ValueError(f"scene has no actuator for joint {name}")
            self.act_ids.append(matches[0])
        self.act_ids = np.asarray(self.act_ids)
        self.joint_adr = np.asarray(
            [self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
        )
        self.dof_adr = np.asarray(
            [self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
        )
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_id < 0:
            raise ValueError("scene has no pelvis body")
        self.gantry = gantry
        self._gantry_released = not gantry
        self._gantry_active = False
        self._anchor_p = None
        self._anchor_R = None
        self.gantry_lin_k = 8000.0   # N/m, waist-harness spring
        self.gantry_lin_c = 400.0    # N s/m
        self.gantry_rot_k = 200.0    # N m/rad
        self.gantry_rot_c = 25.0     # N m s/rad
        self._push_force = None      # external disturbance, N (world frame, pelvis)
        self._gantry_force_norm = 0.0
        if start == "clip_frame0":
            self._teleport(clip, 0)
        elif start == "stand":
            kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            if kid >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_forward(self.model, self.data)
        if gantry:
            self._anchor_p = self.data.qpos[0:3].copy()
            self._anchor_R = self._quat_mat(self.data.qpos[3:7])

    def set_push(self, force_xyz):
        self._push_force = None if force_xyz is None else np.asarray(force_xyz, dtype=np.float64)

    @staticmethod
    def _quat_mat(q):
        w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def sync_gantry(self, state):
        """Waist-harness spring for the transitions only.

        Released (latched) once the robot reaches STANDBY, so the operator can
        verify gantry-free self-balancing before playing, matching the real
        harness flow. Stays released afterwards (POLICY and post-play).
        """
        if state in ("STANDBY", "POLICY"):
            self._gantry_released = True
        self._gantry_active = (
            self.gantry and not self._gantry_released and state in ("PASSIVE", "MOVE_TO_START")
        )

    def _apply_gantry(self):
        self.data.xfrc_applied[self.pelvis_id] = 0.0
        self._gantry_force_norm = 0.0
        if self._push_force is not None:
            self.data.xfrc_applied[self.pelvis_id, 0:3] += self._push_force
        if not self._gantry_active:
            return
        m, d = self.model, self.data
        p = d.qpos[0:3]
        v = d.qvel[0:3]
        R = self._quat_mat(d.qpos[3:7])
        vel = np.zeros(6)
        self.mj.mj_objectVelocity(m, d, self.mj.mjtObj.mjOBJ_BODY, self.pelvis_id, vel, 0)
        w_world = vel[0:3]
        force = self.gantry_lin_k * (self._anchor_p - p) - self.gantry_lin_c * v
        R_err = self._anchor_R.T @ R
        vee = np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
        sin = 0.5 * np.linalg.norm(vee)
        cos = 0.5 * (np.trace(R_err) - 1.0)
        angle = np.arctan2(sin, cos)
        if sin > 1e-6:
            axis = vee / (2.0 * sin)
        elif cos > 0.0:
            axis = np.zeros(3)
        else:  # ~180 deg: pick any unit axis orthogonal to the first error column
            axis = np.array([1.0, 0.0, 0.0]) - R_err[:, 0] * R_err[0, 0]
            n = np.linalg.norm(axis)
            axis = axis / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
        torque = -self.gantry_rot_k * angle * axis - self.gantry_rot_c * w_world
        self._gantry_force_norm = float(np.linalg.norm(force))
        self.data.xfrc_applied[self.pelvis_id, 0:3] += force
        self.data.xfrc_applied[self.pelvis_id, 3:6] += torque

    def draw_gantry(self, viewer):
        """Draw the waist-harness rope in the passive viewer (visual only)."""
        if self._anchor_p is None or viewer is None:
            return
        scn = viewer.user_scn
        if scn.ngeom >= scn.maxgeom - 1:
            return
        top = self._anchor_p + np.array([0.0, 0.0, 1.5])
        pelvis = self.data.qpos[0:3]
        if self._gantry_active:
            f = self._gantry_force_norm
            # green (slack) -> yellow -> red (high tension)
            heat = min(f / 600.0, 1.0)
            rgba = np.array([0.2 + 0.8 * heat, 0.8 - 0.6 * heat, 0.1, 0.9])
            width = 0.006 + 0.004 * heat
        else:
            rgba = np.array([0.5, 0.5, 0.5, 0.35])
            width = 0.003
        g = scn.geoms[scn.ngeom]
        self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                             np.eye(3).ravel(), rgba.astype(np.float32))
        self.mj.mjv_connector(g, self.mj.mjtGeom.mjGEOM_CAPSULE, width, top, pelvis)
        scn.ngeom += 1

    @staticmethod
    def _make_ghost_scene(scene_xml: str) -> str:
        """Inject a translucent ghost copy of the robot (reference playback).

        Manual merge (mujoco 3.11 has no <attach>): the robot file's worldbody
        content is copied with body/joint/geom/site names prefixed "ghost_";
        its actuators are dropped (the ghost is driven kinematically).
        """
        import re
        from pathlib import Path

        scene_path = Path(scene_xml)
        text = scene_path.read_text()
        m = re.search(r'<include file="([^"]+)"', text)
        if not m:
            raise ValueError(f"{scene_xml}: no <include> robot file found for ghost")
        robot_text = (scene_path.parent / m.group(1)).read_text()

        wm = re.search(r"<worldbody>(.*)</worldbody>", robot_text, flags=re.S)
        if not wm:
            raise ValueError("robot file has no worldbody")
        ghost_body = wm.group(1)

        def _prefix_tag(match):
            tag = match.group(0)
            if tag.startswith("<mesh") or tag.startswith("<material") or tag.startswith("<texture"):
                return tag
            return re.sub(r'name="([^"]+)"', lambda m: f'name="ghost_{m.group(1)}"', tag, count=1)

        ghost_body = re.sub(r'<(?:body|joint|freejoint|geom|site)\b[^>]*?name="[^"]+"', _prefix_tag, ghost_body)
        text = re.sub(r"<keyframe>.*?</keyframe>", "", text, flags=re.S)
        text = re.sub(r"</worldbody>", ghost_body + "\n  </worldbody>", text, count=1)
        tmp = scene_path.with_name(scene_path.stem + "_ghost.xml")
        tmp.write_text(text)
        return str(tmp)

    def _setup_ghost(self, joint_names):
        mj = self.mj
        adr, dof = [], []
        for name in joint_names:
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"ghost_{name}")
            if jid < 0:
                raise ValueError(f"ghost joint ghost_{name} missing")
            adr.append(self.model.jnt_qposadr[jid])
            dof.append(self.model.jnt_dofadr[jid])
        self.ghost_joint_adr = np.asarray(adr)
        self.ghost_dof_adr = np.asarray(dof)
        root_jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "ghost_floating_base_joint")
        if root_jid < 0:
            raise ValueError("ghost root joint missing")
        self.ghost_root = (self.model.jnt_qposadr[root_jid], self.model.jnt_dofadr[root_jid])
        ghost_body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "ghost_pelvis")
        for b in range(self.model.nbody):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, b) or ""
            if name.startswith("ghost_"):
                self.model.body_gravcomp[b] = 1.0
        for g in range(self.model.ngeom):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("ghost_") or self.model.geom_bodyid[g] == ghost_body or (
                mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])) or ""
            ).startswith("ghost_"):
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0
                self.model.geom_rgba[g] = np.array([0.3, 0.5, 1.0, 0.4])
        for a in range(self.model.nu):
            jname = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, int(self.model.actuator_trnid[a, 0])) or ""
            if jname.startswith("ghost_"):
                self.model.actuator_gainprm[a, 0] = 0.0
                self.model.actuator_biasprm[a, 1] = 0.0
                self.model.actuator_biasprm[a, 2] = 0.0

    def update_ghost(self, root_pos, root_quat_wxyz, joint_pos):
        """Pin the ghost to a reference pose (visual only)."""
        if self.ghost_joint_adr is None:
            return
        qadr, vadr = self.ghost_root
        self.data.qpos[qadr : qadr + 3] = root_pos
        self.data.qpos[qadr + 3 : qadr + 7] = root_quat_wxyz
        self.data.qpos[self.ghost_joint_adr] = joint_pos
        self.data.qvel[vadr : vadr + 6] = 0.0
        self.data.qvel[self.ghost_dof_adr] = 0.0
        self.mj.mj_forward(self.model, self.data)
    def _teleport(self, clip, frame: int, with_ref_velocity: bool = False):
        self.data.qpos[0:3] = clip.root_pos[frame]
        self.data.qpos[3:7] = clip.root_quat_wxyz[frame]
        self.data.qpos[self.joint_adr] = clip.joint_pos[frame]
        self.data.qvel[:] = 0.0
        if with_ref_velocity:
            R = self._quat_mat(clip.root_quat_wxyz[frame])
            self.data.qvel[0:3] = clip.root_lin_vel_w[frame]
            self.data.qvel[3:6] = R.T @ clip.root_ang_vel_w[frame]
            self.data.qvel[self.dof_adr] = clip.joint_vel[frame]

    def teleport_pose(self, root_pos, root_quat_wxyz, joint_pos):
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat_wxyz
        self.data.qpos[self.joint_adr] = joint_pos
        self.data.qvel[:] = 0.0
        self.mj.mj_forward(self.model, self.data)
        self._anchor_p = self.data.qpos[0:3].copy()
        self._anchor_R = self._quat_mat(self.data.qpos[3:7])

    def read(self):
        d, m = self.data, self.model
        q = d.qpos[self.joint_adr].astype(np.float64).copy()
        dq = d.qvel[self.dof_adr].astype(np.float64).copy()
        vel = np.zeros(6)
        self.mj.mj_objectVelocity(m, d, self.mj.mjtObj.mjOBJ_BODY, self.pelvis_id, vel, 1)
        gyro = vel[0:3].copy()
        quat = d.qpos[3:7].copy()
        return q, dq, gyro, quat, self.keys

    def send(self, q_target, kp, kd):
        m, d = self.model, self.data
        ids = self.act_ids
        m.actuator_gainprm[ids, 0] = kp
        m.actuator_biasprm[ids, 1] = -kp
        m.actuator_biasprm[ids, 2] = -kd
        d.ctrl[ids] = q_target

    def finish_tick(self):
        for _ in range(self.substeps):
            self._apply_gantry()
            self.mj.mj_step(self.model, self.data)
        self.data.xfrc_applied[self.pelvis_id] = 0.0

    def pelvis_z(self) -> float:
        return float(self.data.qpos[2])
