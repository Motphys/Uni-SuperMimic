"""Interactive same-repository MuJoCo viewer for exported deployment artifacts."""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from unisupermimic.deploy.runtime import (
    DeployConfig,
    OnnxPolicy,
    StepResult,
    UniClip,
    UniDeployCore,
)


def _short(name: str) -> str:
    return name.split("/")[-1]


def build_ghost_model(scene: Path, robot: Path) -> mujoco.MjModel:
    """Attach a contact-free visual reference robot to the deployment scene."""
    parent = mujoco.MjSpec.from_file(str(scene))
    child = mujoco.MjSpec.from_file(str(robot))
    frame = parent.worldbody.add_frame(name="reference_ghost")
    parent.attach(child, prefix="ref_", frame=frame)

    for key in list(parent.keys):
        parent.delete(key)
    for actuator in list(parent.actuators):
        if actuator.name.startswith("ref_"):
            parent.delete(actuator)
    for sensor in list(parent.sensors):
        if sensor.name.startswith("ref_"):
            parent.delete(sensor)
    for body in parent.bodies:
        if not body.name.startswith("ref_"):
            continue
        body.gravcomp = 1.0
        for joint in body.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                joint.damping = np.array([1000.0, 0.0, 0.0])
                joint.frictionloss = 1000.0
                joint.armature = 1000.0
        for geom in body.geoms:
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 1
            geom.rgba = [0.35, 0.65, 1.0, 0.25]
    return parent.compile()


class ViewerRobot:
    """Extract the state and command the actuators in the training joint order."""

    def __init__(self, model: mujoco.MjModel, config: DeployConfig, *, ghost: bool):
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config
        self.ghost = ghost
        names = [_short(model.joint(i).name) for i in range(model.njnt)]
        if len(config.joint_names) != len(set(config.joint_names)):
            raise ValueError("deploy_config joint_names contains duplicates")
        unknown = sorted(set(config.joint_names) - set(names))
        if unknown:
            raise ValueError(f"model is missing deployment joints: {unknown}")
        self.jid = np.array([names.index(name) for name in config.joint_names], dtype=int)
        self.qadr = model.jnt_qposadr[self.jid]
        self.dadr = model.jnt_dofadr[self.jid]

        free_joints = [
            i
            for i in range(model.njnt)
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and not names[i].startswith("ref_")
        ]
        if len(free_joints) != 1:
            raise RuntimeError("expected exactly one robot free joint")
        self.free = free_joints[0]
        self.fq = model.jnt_qposadr[self.free]
        self.fv = model.jnt_dofadr[self.free]

        transmission_joints = [int(model.actuator_trnid[i, 0]) for i in range(model.nu)]
        self.act = np.array([transmission_joints.index(j) for j in self.jid], dtype=int)
        if model.nu != len(config.joint_names):
            raise ValueError(f"expected {len(config.joint_names)} actuators, got {model.nu}")

        if ghost:
            ghost_names = [f"ref_{name}" for name in config.joint_names]
            missing = sorted(set(ghost_names) - set(names))
            if missing:
                raise RuntimeError(f"ghost model is missing joints: {missing}")
            self.ghost_jid = np.array([names.index(name) for name in ghost_names], dtype=int)
            self.ghost_qadr = model.jnt_qposadr[self.ghost_jid]
            ghost_free = [
                i
                for i in range(model.njnt)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and names[i].startswith("ref_")
            ]
            if len(ghost_free) != 1:
                raise RuntimeError("expected exactly one ghost free joint")
            self.ghost_fq = model.jnt_qposadr[ghost_free[0]]

    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self.data.qpos[self.qadr].copy()
        dq = self.data.qvel[self.dadr].copy()
        quat = self.data.qpos[self.fq + 3 : self.fq + 7].copy()
        gyro = self.data.qvel[self.fv + 3 : self.fv + 6].copy()
        return q, dq, gyro, quat

    def set_state(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.fq : self.fq + 3] = root_pos
        self.data.qpos[self.fq + 3 : self.fq + 7] = root_quat
        self.data.qpos[self.qadr] = joint_pos
        mujoco.mj_forward(self.model, self.data)

    def set_ghost(self, clip: UniClip, frame: int) -> None:
        if not self.ghost:
            return
        index = int(np.clip(frame, 0, clip.num_frames - 1))
        self.data.qpos[self.ghost_fq : self.ghost_fq + 3] = clip.root_pos[index]
        self.data.qpos[self.ghost_fq + 3 : self.ghost_fq + 7] = clip.root_quat_wxyz[index]
        self.data.qpos[self.ghost_qadr] = clip.joint_pos[index]
        self.data.qvel[self.ghost_fq : self.ghost_fq + 6] = 0.0
        self.data.qvel[self.model.jnt_dofadr[self.ghost_jid]] = 0.0

    def foot_clearance(self) -> float:
        lowest = np.inf
        for geom in range(self.model.ngeom):
            if not (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]):
                continue
            body_name = _short(self.model.body(self.model.geom_bodyid[geom]).name)
            if "ankle_roll" not in body_name:
                continue
            radius = float(self.model.geom_size[geom, 0])
            lowest = min(lowest, float(self.data.geom_xpos[geom, 2]) - radius)
        if not np.isfinite(lowest):
            raise RuntimeError("model has no ankle-roll collision geoms")
        return float(lowest)

    def place_on_ground(self) -> float:
        delta = -self.foot_clearance()
        self.data.qpos[self.fq + 2] += delta
        mujoco.mj_forward(self.model, self.data)
        return delta

    def command(self, target: np.ndarray) -> None:
        self.data.ctrl[self.act] = target


def launch(
    *,
    config_path: Path,
    policy_path: Path,
    clip_path: Path,
    scene: Path,
    robot: Path,
    start_frame: int = 0,
    loop: bool = False,
    ghost: bool = True,
    headless: bool = False,
    max_steps: int | None = None,
) -> None:
    """Launch a real-time ONNX deployment rollout in MuJoCo's native viewer."""
    if headless and max_steps is None:
        raise ValueError("--headless requires --max-steps")
    step_limit = max_steps if max_steps is not None else 0
    config = DeployConfig.load(config_path)
    policy = OnnxPolicy(policy_path, threads=1)
    clip = UniClip.load(clip_path)
    core = UniDeployCore(config, policy, clip)
    core.reset(start_frame)

    if ghost:
        model = build_ghost_model(scene, robot)
    else:
        model = mujoco.MjModel.from_xml_path(str(scene))  # pyright: ignore[reportAttributeAccessIssue]
    model.opt.timestep = 0.001
    robot_view = ViewerRobot(model, config, ghost=ghost)
    robot_view.set_state(
        clip.root_pos[start_frame],
        clip.root_quat_wxyz[start_frame],
        clip.joint_pos[start_frame],
    )
    robot_view.place_on_ground()
    robot_view.set_ghost(clip, start_frame)

    steps_per_policy_step = int(round(config.control_dt / model.opt.timestep))
    if steps_per_policy_step != 20:
        raise ValueError(f"expected 20 physics steps per policy step, got {steps_per_policy_step}")

    print(
        f"viewing {clip.name}: frame={start_frame}/{clip.num_frames - 1}, "
        f"duration={clip.num_frames * config.control_dt:.2f}s, loop={loop}, ghost={ghost}",
        file=sys.stderr,
    )

    def reset() -> None:
        core.reset(start_frame)
        robot_view.set_state(
            clip.root_pos[start_frame],
            clip.root_quat_wxyz[start_frame],
            clip.joint_pos[start_frame],
        )
        robot_view.place_on_ground()
        robot_view.set_ghost(clip, start_frame)

    def run_one() -> StepResult:
        q, dq, gyro, quat = robot_view.state()
        result = core.step(q, dq, gyro, quat)
        robot_view.command(result.q_target)
        for _ in range(steps_per_policy_step):
            mujoco.mj_step(model, robot_view.data)
        robot_view.set_ghost(clip, core.frame)
        mujoco.mj_forward(model, robot_view.data)
        return result

    if headless:
        for _ in range(step_limit):
            result = run_one()
            if result.abort:
                print(f"abort={result.abort} frame={core.frame}", file=sys.stderr)
                raise RuntimeError(result.abort)
            if result.done and not loop:
                break
        return

    with mujoco.viewer.launch_passive(model, robot_view.data) as view:
        policy_step = 0
        while view.is_running():
            result = run_one()
            view.sync()

            if result.abort:
                print(f"abort={result.abort} frame={core.frame}", file=sys.stderr)
                break
            if result.done:
                if not loop:
                    break
                reset()
            policy_step += 1
            if max_steps is not None and policy_step >= max_steps:
                break
            time.sleep(config.control_dt)
