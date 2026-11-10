"""Export Uni-SuperMimic policy checkpoints and references for Repro sim2sim.

The output bundle follows Repro's deployment layout:

```
<out>/<ckpt_tag>/policy.onnx           # obs [1,990], goal [1,730] -> action [1,29]
<out>/<ckpt_tag>/deploy_config.json    # Uni joint constants + contract=unisupermimic_v1
<out>/models/g1_train_scene.xml        # same-model MJCF and meshes
<out>/clips/<clip>.npz                 # (T,730) exact MotionLib goal and reference state
```
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
    G1_JOINT_NAMES,
    _kd,
    _kp,
    action_scale,
    default_joint_pos_list,
    motor_of,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MJCF = REPO / "src/unisupermimic/assets/robots/g1/supermimic.xml"
DEFAULT_SCENE = REPO / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"
DEFAULT_ASSETS = DEFAULT_MJCF.parent / "assets"
OBS_DIM = 99
GOAL_FRAME_DIM = 73
GOAL_DIM = 730
NUM_JOINTS = 29


def checkpoint_tag(path: str | Path) -> str:
    return Path(path).stem


def _velocity_limit(joint_name: str) -> float:
    """Uni motor model's actuator velocity limit."""
    if "hip_pitch" in joint_name or "hip_yaw" in joint_name or "waist_yaw" in joint_name:
        return 32.0
    if "hip_roll" in joint_name or "knee" in joint_name:
        return 20.0
    if "wrist_pitch" in joint_name or "wrist_yaw" in joint_name:
        return 22.0
    return 37.0


def write_config(checkpoint: str | Path, out: Path, mjcf: str | Path = DEFAULT_MJCF) -> Path:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))  # pyright: ignore[reportAttributeAccessIssue]
    name_to_joint = {model.joint(i).name: i for i in range(model.njnt)}
    name_to_actuator = {model.actuator(i).name: i for i in range(model.nu)}
    lower, upper, armature = [], [], []
    for name in G1_JOINT_NAMES:
        if name not in name_to_joint:
            raise ValueError(f"{mjcf}: missing training joint {name}")
        jid = name_to_joint[name]
        lower.append(float(model.jnt_range[jid, 0]))
        upper.append(float(model.jnt_range[jid, 1]))
        armature.append(float(model.dof_armature[model.jnt_dofadr[jid]]))

    motors = [motor_of(name) for name in G1_JOINT_NAMES]
    for name, motor in zip(G1_JOINT_NAMES, motors):
        if name not in name_to_actuator:
            raise ValueError(f"{mjcf}: missing training position actuator {name}")
        aid = name_to_actuator[name]
        expected = (float(_kp(motor[2])), float(_kd(motor[2])), float(motor[0]))
        actual = (
            float(model.actuator_gainprm[aid, 0]),
            float(-model.actuator_biasprm[aid, 2]),
            float(model.actuator_forcerange[aid, 1]),
        )
        if not np.allclose(actual, expected, rtol=0, atol=1e-9):
            raise ValueError(f"{mjcf}: actuator {name} {actual} != robot constants {expected}")
    cfg = {
        "contract": "unisupermimic_v1",
        "source_checkpoint": str(checkpoint),
        "joint_names": G1_JOINT_NAMES,
        "default_joint_pos": default_joint_pos_list(),
        "action_scale": [float(action_scale(name)) for name in G1_JOINT_NAMES],
        "kp": [float(_kp(motor[2])) for motor in motors],
        "kd": [float(_kd(motor[2])) for motor in motors],
        "effort_limit": [float(motor[0]) for motor in motors],
        "armature": armature,
        "velocity_limit": [_velocity_limit(name) for name in G1_JOINT_NAMES],
        "joint_lower": lower,
        "joint_upper": upper,
        "soft_limit_factor": 0.9,
        "control_dt": 0.02,
        "obs_dim": OBS_DIM,
        "history": 10,
        "goal_frame_dim": GOAL_FRAME_DIM,
        "future": 10,
        "joint_offset": [0.0] * NUM_JOINTS,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "deploy_config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    return out / "deploy_config.json"


def load_policy(checkpoint: str | Path, device: str = "cpu") -> tuple[Policy, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    args = ckpt.get("args", {})
    required = {
        "goal_frames": 10,
        "hist": 10,
        "z_dim": 32,
    }
    for key, expected in required.items():
        value = args.get(key)
        if key not in args:
            raise ValueError(f"{checkpoint}: checkpoint args is missing algo.{key}")
        if value != expected:
            raise ValueError(f"{checkpoint}: expected algo.{key}={expected}, got {value!r}")
    if "pi" not in ckpt:
        raise ValueError(f"{checkpoint}: no 'pi' policy state dictionary")
    policy = Policy(
        obs_dim=OBS_DIM,
        goal_dim=GOAL_FRAME_DIM * int(args["goal_frames"]),
        hist=int(args["hist"]),
        z_dim=int(args["z_dim"]),
        num_joints=NUM_JOINTS,
        hidden=tuple(args.get("policy_hidden", (2048, 1024, 512, 256))),
    )
    policy.load_state_dict(ckpt["pi"], strict=True)
    return policy.eval(), dict(args)


class _OnnxPolicy(torch.nn.Module):
    def __init__(self, policy: Policy, history: int = 10):
        super().__init__()
        self.policy = policy
        self.history = history

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        obs_hist = obs.reshape(-1, self.history, OBS_DIM)
        return self.policy(obs_hist, goal)


def export_onnx(checkpoint: str | Path, out: Path, opset: int = 18) -> dict:
    try:
        import onnx as onnx_lib
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on export environment
        raise RuntimeError("ONNX export needs onnx and onnxruntime") from exc

    policy, _ = load_policy(checkpoint)
    destination = out / checkpoint_tag(checkpoint) / "policy.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    example = (torch.zeros(1, 990), torch.zeros(1, 730))
    with torch.no_grad():
        try:
            program = torch.onnx.export(
                _OnnxPolicy(policy).eval(),
                example,
                input_names=["obs", "goal"],
                output_names=["action"],
                opset_version=opset,
                dynamo=True,
                external_data=False,
            )
            assert program is not None
            program.save(str(destination), external_data=False)
        except TypeError:
            torch.onnx.export(
                _OnnxPolicy(policy).eval(),
                example,
                str(destination),
                input_names=["obs", "goal"],
                output_names=["action"],
                opset_version=opset,
            )

    model = onnx_lib.load(str(destination))
    onnx_lib.checker.check_model(model)
    ir9_destination = destination.with_name("policy_ir9.onnx")
    model.ir_version = 9
    onnx_lib.checker.check_model(model)
    onnx_lib.save(model, str(ir9_destination))
    shapes = []
    for value in (*model.graph.input, *model.graph.output):
        shapes.append((value.name, [d.dim_value for d in value.type.tensor_type.shape.dim]))
    if shapes != [("obs", [1, 990]), ("goal", [1, 730]), ("action", [1, 29])]:
        raise ValueError(f"bad ONNX I/O contract: {shapes}")

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(32, 990)).astype(np.float32)
    goal = rng.normal(size=(32, 730)).astype(np.float32)
    with torch.no_grad():
        torch_obs = torch.as_tensor(obs).reshape(32, 10, 99)
        ref = policy(torch_obs, torch.as_tensor(goal)).numpy()
    got = np.concatenate(
        [session.run(["action"], {"obs": o[None], "goal": g[None]})[0] for o, g in zip(obs, goal)]
    )
    max_diff = float(np.abs(ref - got).max())
    if max_diff > 2e-4:
        raise ValueError(f"ONNX/Torch max action diff {max_diff:.3e} exceeds 2e-4")
    return {
        "path": str(destination),
        "ir9_path": str(ir9_destination),
        "bytes": destination.stat().st_size,
        "opset": opset,
        "io": shapes,
        "n_random": 32,
        "max_abs_action_diff": max_diff,
    }


def export_clips(
    motion_files: Sequence[str | Path],
    out: Path,
    *,
    mjcf: str | Path = DEFAULT_MJCF,
    src_fps: float = 30.0,
    names: Sequence[str] | None = None,
    goal_frames: int = 10,
    chunk: int = 65536,
) -> list[Path]:
    kin = G1Kinematics(str(mjcf), "cpu")
    motion = MotionLib([str(path) for path in motion_files], kin, "cpu", src_fps=src_fps)
    selected = set(names) if names is not None else set(motion.names)
    unknown = sorted(selected - set(motion.names))
    if unknown:
        raise ValueError(f"requested clips not in source motion library: {unknown}")
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for clip_name in motion.names:
        if clip_name not in selected:
            continue
        ci = motion.names.index(clip_name)
        start, end = int(motion.clip_start[ci]), int(motion.clip_end[ci])
        goal_parts, ref_parts = [], []
        for lo in range(start, end + 1, chunk):
            frames = torch.arange(lo, min(lo + chunk, end + 1), dtype=torch.int64)
            goal_parts.append(motion.goal(frames, goal_frames).numpy().astype(np.float32))
            p, R, v, w, q, qd = motion.ref_state(frames)
            ref_parts.append(
                (
                    p.numpy(),
                    R.numpy(),
                    mat_to_quat(R).numpy(),
                    v.numpy(),
                    w.numpy(),
                    q.numpy(),
                    qd.numpy(),
                )
            )
        goal = np.concatenate(goal_parts)
        p, R, quat, v, w, q, qd = (
            np.concatenate([part[i] for part in ref_parts]) for i in range(7)
        )
        destination = clips_dir / f"{clip_name}.npz"
        np.savez(
            destination,
            name=np.array(clip_name),
            contract=np.array("unisupermimic_v1"),
            fps=np.float32(50.0),
            src_fps=np.float32(src_fps),
            joint_names=np.array(G1_JOINT_NAMES),
            default_joint_pos=np.asarray(default_joint_pos_list(), dtype=np.float32),
            goal=goal,
            joint_pos=q.astype(np.float32),
            proj_grav=(-R[:, 2, :]).astype(np.float32),
            root_pos=p.astype(np.float32),
            root_quat_wxyz=quat.astype(np.float32),
            root_lin_vel_w=v.astype(np.float32),
            root_ang_vel_w=w.astype(np.float32),
            joint_vel=qd.astype(np.float32),
        )
        written.append(destination)
        print(f"  {clip_name}: T={len(goal)} -> {destination}")
    return written


def export_model(out: Path, scene: Path = DEFAULT_SCENE, assets: Path = DEFAULT_ASSETS) -> Path:
    destination = out / "models"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene, destination / "g1_train_scene.xml")
    shutil.copy2(scene.parent / "supermimic.xml", destination / "supermimic.xml")
    if assets.is_dir():
        shutil.copytree(assets, destination / "assets", dirs_exist_ok=True)
    import mujoco

    mujoco.MjModel.from_xml_path(  # pyright: ignore[reportAttributeAccessIssue]
        str(destination / "g1_train_scene.xml")
    )
    return destination / "g1_train_scene.xml"


def check(out: Path, checkpoint: str | Path, clip: str | Path) -> dict:
    config = json.loads((out / checkpoint_tag(checkpoint) / "deploy_config.json").read_text())
    if config.get("contract") != "unisupermimic_v1":
        raise ValueError("deploy_config contract is not unisupermimic_v1")
    with np.load(clip) as archive:
        if str(archive["contract"]) != "unisupermimic_v1" or archive["goal"].shape[-1] != GOAL_DIM:
            raise ValueError(f"{clip}: invalid Uni clip contract")
        frames = int(archive["goal"].shape[0])
    policy, args = load_policy(checkpoint)
    return {
        "checkpoint": str(checkpoint),
        "goal_frames": args["goal_frames"],
        "history": args["hist"],
        "obs_flattened": OBS_DIM * args["hist"],
        "goal_flattened": GOAL_FRAME_DIM * args["goal_frames"],
        "clip": str(clip),
        "clip_frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmd",
        default="all",
        help="config | clips | onnx | model | all | check | view",
    )
    parser.add_argument("--checkpoint", type=Path, help="Uni ckpt_XXXXXX.pt / last.pt")
    parser.add_argument("--out", type=Path, default=REPO / "deploy_artifacts_uni")
    parser.add_argument("--motion", nargs="+", type=Path, help="CSV motion(s) or one compact .pt")
    parser.add_argument(
        "--clip-name",
        action="append",
        help="clip to export; repeatable; default all CSV clips / all is too large for compact PT",
    )
    parser.add_argument("--src-fps", type=float, default=30.0)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check-clip", type=Path)
    parser.add_argument("--config", type=Path, help="existing deploy_config.json for --cmd view")
    parser.add_argument("--onnx", type=Path, help="existing policy.onnx for --cmd view")
    parser.add_argument("--clip", type=Path, help="existing exported clip .npz for --cmd view")
    parser.add_argument("--robot", type=Path, help="robot MJCF used to build the visual ghost")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--loop", action="store_true", help="restart the clip in --cmd view")
    parser.add_argument(
        "--no-ghost",
        action="store_true",
        help="do not attach the visual reference robot in --cmd view",
    )
    parser.add_argument("--headless", action="store_true", help="run --cmd view without a window")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="stop --cmd view after this many policy steps; required by --headless",
    )
    args = parser.parse_args()
    cmds = set(args.cmd.split(","))
    if cmds == {"all"}:
        cmds = {"config", "clips", "onnx", "model"}
    if not cmds <= {"config", "clips", "onnx", "model", "all", "check", "view"}:
        parser.error(f"unknown command in {args.cmd}")
    if "view" in cmds and len(cmds) != 1:
        parser.error("--cmd view cannot be combined with other commands")
    if "view" in cmds and args.headless and args.max_steps is None:
        parser.error("--headless requires --max-steps")
    if cmds & {"config", "onnx", "check"} and args.checkpoint is None:
        parser.error("--checkpoint is required for config/onnx/check")
    if "clips" in cmds and not args.motion:
        parser.error("--motion is required for clips")
    needs_compact_selection = (
        cmds != {"check"} and "clips" in cmds and args.motion[0].suffix == ".pt"
    )
    if needs_compact_selection and not args.clip_name:
        parser.error("compact .pt export requires one or more --clip-name values")

    if "view" in cmds:
        from unisupermimic.deploy.viewer import launch

        existing_artifacts = (
            args.config is not None or args.onnx is not None or args.clip is not None
        )
        if existing_artifacts:
            if not (args.config is not None and args.onnx is not None and args.clip is not None):
                parser.error("--cmd view needs --config, --onnx, and --clip together")
            launch(
                config_path=args.config,
                policy_path=args.onnx,
                clip_path=args.clip,
                scene=args.scene,
                robot=args.robot or args.mjcf,
                start_frame=args.start_frame,
                loop=args.loop,
                ghost=not args.no_ghost,
                headless=args.headless,
                max_steps=args.max_steps,
            )
        else:
            if args.checkpoint is None or not args.motion or len(args.clip_name or []) != 1:
                parser.error(
                    "--cmd view needs either existing artifacts or "
                    "--checkpoint, --motion, and exactly one --clip-name"
                )
            with tempfile.TemporaryDirectory(prefix="unisupermimic-sim2sim-") as temporary:
                temporary_out = Path(temporary)
                config_out = temporary_out / checkpoint_tag(args.checkpoint)
                write_config(args.checkpoint, config_out, args.mjcf)
                export_onnx(args.checkpoint, temporary_out, args.opset)
                exported = export_clips(
                    args.motion,
                    temporary_out,
                    mjcf=args.mjcf,
                    src_fps=args.src_fps,
                    names=args.clip_name,
                )
                launch(
                    config_path=config_out / "deploy_config.json",
                    policy_path=temporary_out / checkpoint_tag(args.checkpoint) / "policy.onnx",
                    clip_path=exported[0],
                    scene=args.scene,
                    robot=args.robot or args.mjcf,
                    start_frame=args.start_frame,
                    loop=args.loop,
                    ghost=not args.no_ghost,
                    headless=args.headless,
                    max_steps=args.max_steps,
                )
        return

    report: dict[str, object] = {}
    if "model" in cmds:
        report["model"] = str(export_model(args.out, args.scene, args.assets))
    if "config" in cmds:
        config_out = args.out / checkpoint_tag(args.checkpoint)
        report["config"] = str(write_config(args.checkpoint, config_out, args.mjcf))
    if "clips" in cmds:
        report["clips"] = [
            str(path)
            for path in export_clips(
                args.motion,
                args.out,
                mjcf=args.mjcf,
                src_fps=args.src_fps,
                names=args.clip_name,
            )
        ]
    if "onnx" in cmds:
        report["onnx"] = export_onnx(args.checkpoint, args.out, args.opset)
    if "check" in cmds:
        if args.check_clip is None:
            parser.error("--check-clip is required for check")
        report["check"] = check(args.out, args.checkpoint, args.check_clip)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
