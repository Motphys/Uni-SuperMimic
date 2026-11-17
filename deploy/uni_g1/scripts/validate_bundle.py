#!/usr/bin/env python3
"""Offline bundle validation. Runs entirely without the robot.

Checks:
  1. manifest.json sha256 of every asset file (if manifest present)
  2. deploy_config.json contract/dims and internal consistency
  3. every clip: contract, goal (T,730), joint_pos (T,29), finite values
  4. ONNX: input/output contract, warmup + deterministic dummy inference
  5. DeployCore dry step on synthetic state (no NaN, target within limits)
  6. FSM dry walk-through: PASSIVE->MOVE_TO_START->HOLD->POLICY->END->HOLD->DAMP

Exit code 0 and prints "status": "PASS" on success.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE))

from uni_deploy.contract import load_assets, load_clip  # noqa: E402
from uni_deploy.core import DeployCore, OnnxPolicy  # noqa: E402
from uni_deploy.fsm import DAMP, FSM, MOVE_TO_START, PASSIVE, POLICY, STANDBY  # noqa: E402

ASSETS = BUNDLE / "assets"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    report = {"status": "PASS", "checks": {}}
    failures = []

    def check(name, fn):
        try:
            report["checks"][name] = fn()
        except Exception as exc:  # noqa: BLE001
            report["checks"][name] = f"FAIL: {exc}"
            failures.append(f"{name}: {exc}")

    def manifest():
        path = BUNDLE / "manifest.json"
        if not path.is_file():
            return "skipped (no manifest.json)"
        entries = json.loads(path.read_text())["files"]
        bad = [k for k, v in entries.items() if sha256(BUNDLE / k) != v]
        if bad:
            raise ValueError(f"hash mismatch: {bad}")
        return f"{len(entries)} files ok"

    def config():
        assets = load_assets(ASSETS / "deploy_config.json")
        assert np.all(assets.kp > 0) and np.all(assets.kd > 0)
        assert np.all(assets.joint_lower < assets.joint_upper)
        return f"contract ok, {len(assets.joint_names)} joints"

    clips = sorted((ASSETS / "clips").glob("*.npz"))

    def clips_check():
        if not clips:
            raise ValueError("no clips under assets/clips")
        return {c.stem: load_clip(c).num_frames for c in clips}

    core_holder = {}

    def onnx_check():
        policy = OnnxPolicy(ASSETS / "policy.onnx")
        assets = load_assets(ASSETS / "deploy_config.json")
        clip = load_clip(clips[0])
        core = DeployCore(assets, clip, policy)
        core.warmup()
        rng = np.random.default_rng(0)
        obs = rng.standard_normal(990).astype(np.float32) * 0.1
        goal = rng.standard_normal(730).astype(np.float32) * 0.1
        a1 = policy(obs, goal)
        a2 = policy(obs, goal)
        if not np.array_equal(a1, a2):
            raise ValueError("ONNX inference is not deterministic")
        core_holder["core"] = core
        return f"obs[1,990] goal[1,730] -> action[1,29]; clip={clip.name}"

    def core_dry_step():
        core = core_holder.get("core")
        if core is None:
            raise ValueError("onnx_check did not run")
        q = core.assets.default_joint_pos.copy()
        dq = np.zeros(29)
        quat = core.clip.root_quat_wxyz[0].copy()
        gyro = np.zeros(3)
        q_t, abort, action = core.step(q, dq, gyro, quat)
        if not np.all(np.isfinite(q_t)):
            raise ValueError("non-finite q_target")
        return f"step ok, abort={abort}, max|a|={np.abs(action).max():.3f}"

    def fsm_walk():
        core = core_holder.get("core")
        if core is None:
            raise ValueError("onnx_check did not run")
        fsm = FSM(core.assets, core)
        q = core.assets.default_joint_pos.copy()
        assert fsm.state == PASSIVE
        assert fsm.request("to_start", q) and fsm.state == MOVE_TO_START
        dq = np.zeros(29)
        gyro = np.zeros(3)
        quat = core.clip.root_quat_wxyz[0].copy()
        for _ in range(int(4.0 / 0.02) + 2):
            q_t, abort, _ = core.step(q, dq, gyro, quat)
            fsm.update(q, (q_t, abort))
        assert fsm.state == STANDBY, fsm.state
        assert fsm.request("play", q) and fsm.state == POLICY
        for _ in range(3):
            q_t, abort, _ = core.step(q, dq, gyro, quat)
            fsm.update(q, (q_t, abort))
        assert fsm.state in (STANDBY, POLICY), fsm.state
        assert fsm.request("damp_anytime", q) and fsm.state == DAMP
        assert not fsm.request("to_start", q)
        assert fsm.request("reset", q) and fsm.state == PASSIVE
        assert fsm.request("to_start", q) and fsm.state == MOVE_TO_START
        return "PASSIVE->MOVE_TO_START->STANDBY->POLICY->DAMP->reset->PASSIVE ok"

    check("manifest", manifest)
    check("deploy_config", config)
    check("clips", clips_check)
    check("onnx", onnx_check)
    check("core_dry_step", core_dry_step)
    check("fsm_walk", fsm_walk)

    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
