#!/usr/bin/env python3
"""Pack C++-consumable assets and golden numeric vectors (runs in the Uni venv).

Outputs under deploy/uni_g1/assets/:
  uni_config.txt          flat key=value config (vectors comma-separated)
  clips_bin/<name>.bin    int32 T + goal (T*730 f32) + joint_pos (T*29 f32)
                          + root_quat_wxyz (T*4 f32) + root_pos (T*3 f32)
  golden/golden.bin       int32 n_cases; per case: clip_idx i32, frame i32,
                          q[29] dq[29] gyro[3] quat[4] prev_action[29]
                          obs[99] goal[730] action[29] target[29]  (all f32)
  golden/golden_manifest.json

Golden vectors are produced by the training-side MDP and the torch policy,
so the C++ controller can be checked term-for-term without torch.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
BUNDLE = REPO / "deploy" / "uni_g1"
ASSETS = BUNDLE / "assets"
sys.path.insert(0, str(BUNDLE))

from uni_deploy.contract import load_assets, load_clip, make_static_clip  # noqa: E402
from uni_deploy.core import DeployCore, SafetyCfg  # noqa: E402

from unisupermimic.deploy.export import load_policy  # noqa: E402
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics  # noqa: E402
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP, XI_DIM, RewardCfg  # noqa: E402
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib  # noqa: E402

CLIPS = [
    "cellphone_record_stand_idle_R_001__A423",
    "drinking_standing_mug_R_001__A282",
    "bow_R_001__A428",
    "clap_enthusiastic_001__A029",
    "pels_air_punch_001__A492",
    "fancy_spin_R_001__A428",
    "dance_basic_chaines_180_R_001__A306",
    "dance_basic_chaines_180_R_001__A307",
    "Loop_Forward_Walk_001__A019",
    "Relaxed_walk_forward_001__A057",
    "Neutral_kick_trash_001__A057",
    "skipping_R_001__A359",
]
FRAMES_PER_CLIP = 8
TOL = {"obs": 1e-4, "goal": 1e-4, "action": 1e-3, "target_formula": 1e-5, "target_e2e": 1e-3}


def write_config(assets) -> None:
    lines = [
        "contract unisupermimic_v1",
        "joint_names " + ",".join(assets.joint_names),
    ]
    for key in ("default_joint_pos", "action_scale", "kp", "kd", "effort_limit",
                "velocity_limit", "joint_lower", "joint_upper"):
        v = getattr(assets, key)
        lines.append(f"{key} " + ",".join(f"{x:.9g}" for x in v))
    lines.append(f"soft_limit_factor {assets.soft_limit_factor:.9g}")
    lines.append(f"control_dt {assets.control_dt:.9g}")
    (ASSETS / "uni_config.txt").write_text("\n".join(lines) + "\n")


def write_clip_bin(clip) -> None:
    out = ASSETS / "clips_bin" / f"{clip.name}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(struct.pack("<i", clip.num_frames))
        f.write(clip.goal.astype("<f4").tobytes())
        f.write(clip.joint_pos.astype("<f4").tobytes())
        f.write(clip.root_quat_wxyz.astype("<f4").tobytes())
        f.write(clip.root_pos.astype("<f4").tobytes())


def main() -> None:
    global ASSETS
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="training checkpoint (ckpt_XXXXXX.pt) used for the golden torch policy")
    ap.add_argument("--motion", type=Path, required=True,
                    help="motion source covering every clip in CLIPS (one compact .pt)")
    ap.add_argument("--mjcf", type=Path,
                    default=REPO / "src/unisupermimic/assets/robots/g1/supermimic.xml")
    ap.add_argument("--assets", type=Path, default=ASSETS)
    args = ap.parse_args()

    ASSETS = args.assets
    assets = load_assets(ASSETS / "deploy_config.json")
    write_config(assets)
    for name in CLIPS:
        write_clip_bin(load_clip(ASSETS / "clips" / f"{name}.npz"))

    kin = G1Kinematics(str(args.mjcf), "cpu")
    motion = MotionLib([str(args.motion)], kin, "cpu", src_fps=30.0)
    mdp = MDP(kin, motion, torch.tensor(assets.default_joint_pos, dtype=torch.float32), RewardCfg())
    pi, _ = load_policy(args.checkpoint)
    pi.eval()

    static_clip = make_static_clip(assets)
    _probe = load_clip(ASSETS / "clips" / f"{CLIPS[0]}.npz")
    target_core = DeployCore(assets, _probe, policy=None, safety=SafetyCfg())
    cases = []
    rng = np.random.default_rng(7)

    def emit(clip_idx: int, frame: int, q, dq, gyro, quat, obs99, goal730):
        obs_hist = torch.tensor(np.repeat(obs99[None], 10, axis=0)[None], dtype=torch.float32)
        with torch.no_grad():
            action = pi(obs_hist, torch.tensor(goal730[None], dtype=torch.float32))[0].numpy()
        target, _ = target_core.targets(action, np.asarray(q, dtype=np.float64),
                                        np.asarray(dq, dtype=np.float64))
        cases.append((clip_idx, frame, q, dq, gyro, quat, action, obs99, goal730, target))

    from uni_deploy.core import quat_to_mat
    for clip_idx, name in enumerate(CLIPS):
        ci = motion.names.index(name)
        s0 = int(motion.clip_start[ci])
        e0 = int(motion.clip_end[ci])
        clip = load_clip(ASSETS / "clips" / f"{name}.npz")
        for k in np.linspace(0, e0 - s0, FRAMES_PER_CLIP, dtype=int):
            f = torch.tensor([s0 + int(k)])
            kc = int(k)
            q = clip.joint_pos[kc] + rng.normal(0, 0.01, 29)
            dq = clip.joint_vel[kc] + rng.normal(0, 0.02, 29)
            quat = clip.root_quat_wxyz[kc]
            R = quat_to_mat(quat)
            gyro = R.T @ clip.root_ang_vel_w[kc]
            # MDP state with the SAME perturbed joint state (root from reference)
            p, Rm, v, w, _q, _qd = motion.ref_state(f)
            state = (p, Rm, v, w,
                     torch.tensor(q[None], dtype=torch.float32),
                     torch.tensor(dq[None], dtype=torch.float32))
            obs99 = mdp.obs(state, torch.zeros(1, 29), torch.zeros(1, XI_DIM), f)[0].numpy()
            goal730 = mdp.goal(f, 10)[0].numpy()
            emit(clip_idx, kc, q, dq, gyro, quat, obs99, goal730)

    # frozen stand (STANDBY) case; rel6 of identity rotation, row-major (R00,R01,R10,R11,R20,R21)
    obs_hist_row = np.concatenate([
        np.zeros(29), np.zeros(29), np.zeros(3), np.array([0.0, 0.0, -1.0]),
        np.zeros(29), np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    ])
    goal_row = static_clip.goal[0]
    emit(-1, 0, assets.default_joint_pos.copy(), np.zeros(29), np.zeros(3),
         np.array([1.0, 0.0, 0.0, 0.0]), obs_hist_row, goal_row)

    out = ASSETS / "golden" / "golden.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(struct.pack("<i", len(cases)))
        for clip_idx, frame, q, dq, gyro, quat, action, obs99, goal730, target in cases:
            f.write(struct.pack("<ii", clip_idx, frame))
            for arr in (q, dq, gyro, quat, np.zeros(29), obs99, goal730, action, target):
                f.write(np.asarray(arr, dtype="<f4").tobytes())
    manifest = {
        "layout": "int32 n_cases; per case: int32 clip_idx, int32 frame, "
                  "q[29] dq[29] gyro[3] quat_wxyz[4] prev_action[29] "
                  "obs[99] goal[730] action[29] target[29] (little-endian f32)",
        "clips": CLIPS,
        "clip_idx_-1": "generated static stand clip (STANDBY)",
        "history": "10 repeats of the same obs row (reset semantics)",
        "tolerances": TOL,
        "n_cases": len(cases),
    }
    (ASSETS / "golden" / "golden_manifest.json").write_text(json.dumps(manifest, indent=1))
    # ordered clip names consumed by the C++ numeric checker (no hardcoded list)
    (ASSETS / "golden" / "clips.txt").write_text("\n".join(CLIPS) + "\n")
    print(f"wrote {len(cases)} golden cases -> {out}")


if __name__ == "__main__":
    main()
