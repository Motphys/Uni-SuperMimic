#!/usr/bin/env python3
"""Deployment-equivalent sim2sim for the Uni-SuperMimic G1 bundle.

Runs the exact runner/FSM/DeployCore code that runs on the robot, against
the training-scene MuJoCo model. Flow mirrors the real operation:

  PASSIVE --(L2+Up)--> MOVE_TO_START --> HOLD --(R1+X)--> POLICY
  POLICY --clip end--> END --> HOLD ;  any time --(L2+B)--> DAMP

With --auto the button presses are scripted; without it, use the viewer
keyboard: u=to_start  x=play  d=DAMP  p=PASSIVE.

Usage (Uni repo root):
  uv run --with onnxruntime --with imageio --with imageio-ffmpeg \
    python deploy/uni_g1/scripts/run_sim.py --clip cellphone_record_stand_idle_R_001__A423 \
    --auto --record deploy/uni_g1/reports/sim_<clip>.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE))

from uni_deploy.buttons import DebouncedButtons, KeyMap  # noqa: E402
from uni_deploy.contract import load_assets, load_clip  # noqa: E402
from uni_deploy.core import DeployCore, OnnxPolicy, SafetyCfg  # noqa: E402
from uni_deploy.fsm import FSM  # noqa: E402
from uni_deploy.runner import TickLog, run  # noqa: E402
from uni_deploy.sim_io import SimIO  # noqa: E402

ASSETS = BUNDLE / "assets"


def key_for(name: str) -> int:
    return {
        "u": (1 << KeyMap.L2) | (1 << KeyMap.UP),
        "x": (1 << KeyMap.R1) | (1 << KeyMap.X),
        "d": (1 << KeyMap.L2) | (1 << KeyMap.B),
        "p": (1 << KeyMap.L2) | (1 << KeyMap.DOWN),
        "n": (1 << KeyMap.L2) | (1 << KeyMap.RIGHT),
        "b": (1 << KeyMap.L2) | (1 << KeyMap.LEFT),
    }[name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--auto", action="store_true", help="script the button presses")
    ap.add_argument("--passive-s", type=float, default=0.5)
    ap.add_argument("--standby-s", type=float, default=2.0, help="STANDBY dwell before POLICY")
    ap.add_argument("--end-hold-s", type=float, default=3.0, help="STANDBY dwell after the clip")
    ap.add_argument("--no-gantry", action="store_true", help="disable the waist-harness spring")
    ap.add_argument("--start", choices=("stand", "clip_frame0"), default="stand")
    ap.add_argument("--direct-play", action="store_true",
                    help="eval-equivalent: spawn at clip frame 0 with reference velocity, POLICY from t=0")
    ap.add_argument("--backend", choices=("train", "unitree"), default="train",
                    help="train: training scene position actuators; unitree: official 29-DoF MJCF + 1 kHz torque PD")
    ap.add_argument("--unitree-scene", type=Path, default=None,
                    help="path to Unitree scene_29dof.xml (backend=unitree)")
    ap.add_argument("--sim-dt", type=float, default=None,
                    help="override the training-scene timestep (backend=train), e.g. 0.001")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--duration", type=float, default=None, help="override run duration [s]")
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    assets = load_assets(ASSETS / "deploy_config.json")
    clips = [load_clip(p) for p in sorted((ASSETS / "clips").glob("*.npz"))]
    if not clips:
        raise SystemExit(f"no clips under {ASSETS / 'clips'}")
    names = [c.name for c in clips]
    clip = clips[names.index(args.clip)] if args.clip in names else clips[0]
    policy = OnnxPolicy(ASSETS / "policy.onnx")
    core = DeployCore(assets, clip, policy, SafetyCfg())
    fsm = FSM(assets, core, clips=clips)
    fsm.motion_clip = clip
    fsm.clip_index = clips.index(clip)
    if args.backend == "unitree":
        from uni_deploy.unitree_io import UnitreeTorqueIO
        scene = args.unitree_scene
        if scene is None:
            env_scene = os.environ.get("G1_UNITREE_XML")
            if env_scene:
                scene = Path(env_scene)
            else:
                raise SystemExit(
                    "backend=unitree needs the Unitree G1 scene: pass --unitree-scene or set "
                    "G1_UNITREE_XML (e.g. <unitree_mujoco>/unitree_robots/g1/scene_29dof.xml)"
                )
        io = UnitreeTorqueIO(str(scene), clip, assets.joint_names, assets.effort_limit,
                             start=args.start, gantry=not args.no_gantry, ghost=args.viewer)
    else:
        io = SimIO(str(ASSETS / "models" / "g1_train_scene.xml"), clip, assets.joint_names,
                   start=args.start, gantry=not args.no_gantry, ghost=args.viewer,
                   sim_dt=args.sim_dt if args.sim_dt is not None else 0.001)
    if args.start == "stand":
        # synthesize the stand pose (ghost models carry no keyframe)
        io.teleport_pose((0.0, 0.0, 0.76), (1.0, 0.0, 0.0, 0.0), assets.default_joint_pos)
    if args.direct_play:
        io._teleport(clip, 0, with_ref_velocity=True)
        io.mj.mj_forward(io.model, io.data)
        fsm.request("to_start", io.read()[0])
        fsm._enter("STANDBY", io.read()[0])
        fsm.request("play", io.read()[0])
        io.sync_gantry(fsm.state)

    from uni_deploy.fsm import INTERP_S
    clip_s = clip.duration
    script = {}
    if args.direct_play:
        end_t = clip_s + 2.0
    elif args.auto:
        t = args.passive_s
        script[t] = "to_start"
        t += INTERP_S + args.standby_s
        script[t] = "play"
        end_t = t + clip_s + args.end_hold_s + 1.0
    else:
        end_t = args.passive_s + 8.0 + clip_s + 6.0
    if args.duration is not None:
        end_t = args.duration

    buttons = DebouncedButtons()
    log = TickLog()

    viewer = None
    renderer = None
    writer = None
    frames = []
    if args.record is not None:
        import imageio.v2 as imageio
        import mujoco
        renderer = mujoco.Renderer(io.model, height=540, width=960)
        writer = imageio.get_writer(str(args.record), fps=25)
        args.record.parent.mkdir(parents=True, exist_ok=True)

    _prev = {"state": None}
    _key = {"set_tick": -1}

    def on_key_wrap(ch, tick_ref):
        io.keys = key_for(ch)
        _key["set_tick"] = tick_ref[0]

    _tick_ref = [0]
    fifo_fd = None
    fifo_path = os.environ.get("UNI_SIM_FIFO", "/tmp/uni_sim_keys.fifo")
    try:
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        fifo_fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        print(f"[sim] controller fifo: {fifo_path} (run tools/sim_controller.py)", flush=True)
    except OSError as exc:
        print(f"[sim] controller fifo unavailable: {exc}", flush=True)

    def do_reset():
        """Reset to the just-started state: PASSIVE at stand, gantry re-armed."""
        motion, clips, idx = fsm.motion_clip, fsm.clips, fsm.clip_index
        fresh = FSM(assets, core)
        fsm.__dict__.update(fresh.__dict__)
        fsm.motion_clip, fsm.clips, fsm.clip_index = motion, clips, idx
        core.reset(0)
        io._gantry_released = False
        io._gantry_active = False
        io.teleport_pose((0.0, 0.0, 0.76), (1.0, 0.0, 0.0, 0.0), assets.default_joint_pos)
        _prev["state"] = None
        print("[sim] reset to initial state (PASSIVE, gantry armed)", flush=True)

    if args.viewer:
        import mujoco.viewer

        def on_key(keycode):
            try:
                ch = chr(keycode)
            except ValueError:
                return
            if ch in "uxdpnb":
                on_key_wrap(ch, _tick_ref)
            elif ch == "r":
                do_reset()

        viewer = mujoco.viewer.launch_passive(io.model, io.data, key_callback=on_key)

    def render_hook(tick, q, quat, state):
        _tick_ref[0] = tick
        if fifo_fd is not None:
            try:
                while True:
                    data = os.read(fifo_fd, 32)
                    if not data:
                        break
                    for byte in data:
                        ch = chr(byte)
                        if ch in "uxdpnb":
                            on_key_wrap(ch, _tick_ref)
                        elif ch == "r":
                            do_reset()
            except BlockingIOError:
                pass
        if state != _prev["state"]:
            print(f"[sim] t={tick * 0.02:6.2f}s  state -> {state}", flush=True)
            _prev["state"] = state
        if getattr(io, "ghost_joint_adr", None) is not None:
            cur = core.clip
            f = min(core.frame, cur.num_frames - 1)
            rp = list(cur.root_pos[f])
            if cur.num_frames <= 2:
                # static/frozen reference: anchor the ghost to the robot's current xy
                rp[0], rp[1] = io.data.qpos[0], io.data.qpos[1]
            io.update_ghost(rp, cur.root_quat_wxyz[f], cur.joint_pos[f])
        if viewer is not None and tick % 1 == 0:
            if hasattr(io, "draw_gantry"):
                io.draw_gantry(viewer)
            viewer.sync()
        if renderer is not None and tick % 2 == 0:
            renderer.update_scene(io.data)
            writer.append_data(renderer.render())
        if io.keys and tick - _key["set_tick"] > 8:
            io.keys = 0  # emulate key release after ~0.16 s held

    t_end = end_t
    run(fsm, core, io, t_end, buttons, log=log, scripted=script,
        realtime=args.realtime and not args.record, hooks=render_hook)

    if viewer is not None:
        viewer.close()
    if renderer is not None:
        renderer.close()
    if writer is not None:
        writer.close()

    summary = log.summary()
    summary["clip"] = args.clip
    summary["clip_frames"] = clip.num_frames
    summary["final_state"] = fsm.state
    summary["abort_reason"] = fsm.abort_reason
    print(json.dumps(summary, indent=1))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"summary": summary,
                                           "tilt_err": log.tilt_err,
                                           "q_rmse": log.q_rmse,
                                           "state": log.state}))
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
