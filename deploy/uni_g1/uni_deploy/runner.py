"""Control loop shared by the MuJoCo sim2sim and the real-robot runner.

The IO object provides:
  read()  -> (q[29], dq[29], gyro_body[3], quat_wxyz[4], keys:int)
  send(q_target, kp, kd)
  finish_tick()  (sim advances physics; robot sleeps to the next 50 Hz tick)

Everything above the IO is exactly the code that runs on the robot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .buttons import EVENT_CYCLE, EVENT_TO_FSM, DebouncedButtons
from .core import DeployCore, quat_to_mat
from .fsm import DAMP, FSM, MOVE_TO_START, POLICY, STANDBY


@dataclass
class TickLog:
    t: list = field(default_factory=list)
    state: list = field(default_factory=list)
    frame: list = field(default_factory=list)
    q_rmse: list = field(default_factory=list)
    tilt_err: list = field(default_factory=list)
    max_dq_ratio: list = field(default_factory=list)
    clamps: list = field(default_factory=list)

    def summary(self) -> dict:
        out = {
            "ticks": len(self.t),
            "q_rmse_policy": None,
            "max_tilt_err": None,
            "max_dq_ratio": None,
        }
        rmse = [v for v, s in zip(self.q_rmse, self.state) if s == POLICY and np.isfinite(v)]
        if rmse:
            out["q_rmse_policy"] = float(np.sqrt(np.mean(np.square(rmse))))
            out["q_rmse_max"] = float(np.max(rmse))
        if self.tilt_err:
            out["max_tilt_err"] = float(np.max(self.tilt_err))
        if self.max_dq_ratio:
            out["max_dq_ratio"] = float(np.max(self.max_dq_ratio))
        return out


def run(fsm: FSM, core: DeployCore, io, duration_s: float, buttons: DebouncedButtons,
        log: TickLog | None = None, scripted: dict | None = None,
        realtime: bool = True, hooks=None) -> TickLog:
    core.warmup()
    log = log or TickLog()
    scripted = scripted or {}
    ticks = int(round(duration_s / fsm.dt))
    fired = set()
    for tick in range(ticks):
        t_now = tick * fsm.dt
        t0 = time.perf_counter()
        q, dq, gyro, quat, keys = io.read()
        state_before_events = fsm.state
        for ts in sorted(scripted):
            if ts not in fired and t_now >= ts:
                fired.add(ts)
                fsm.request(scripted[ts], q)
                if scripted[ts] == "to_start":
                    fsm.set_start_heading(quat)
        for event in buttons.poll(keys):
            if event in EVENT_CYCLE:
                name = fsm.cycle_clip(EVENT_CYCLE[event])
                print(f"[runner] clip -> {name}", flush=True)
            else:
                fsm.request(EVENT_TO_FSM[event], q)
                if event == "up":
                    fsm.set_start_heading(quat)
        if state_before_events == STANDBY and fsm.state == POLICY:
            fsm.reanchor_motion_clip(quat)
        policy_result = None
        rmse = float("nan")
        if fsm.state in (MOVE_TO_START, STANDBY, POLICY):
            core.switch_clip(fsm.policy_clip(), core.frame, keep_history=True)
            q_target, abort, _action = core.step(q, dq, gyro, quat)
            policy_result = (q_target, abort)
            f_now = min(core.frame, core.clip.num_frames - 1)
            ref = core.clip.joint_pos[f_now]
            rmse = float(np.sqrt(np.mean((q - ref) ** 2)))
        prev_state = fsm.state
        q_target, kp, kd = fsm.update(q, policy_result)
        if prev_state == POLICY and fsm.state == STANDBY:
            fsm.reanchor_post_standby(quat)
        io.send(q_target, kp, kd)
        if hasattr(io, "sync_gantry"):
            io.sync_gantry(fsm.state)
        f_now = min(core.frame, core.clip.num_frames - 1)
        R = quat_to_mat(quat)
        R_ref = quat_to_mat(core.clip.root_quat_wxyz[f_now])
        log.t.append(t_now)
        log.state.append(fsm.state)
        log.frame.append(f_now)
        log.q_rmse.append(rmse)
        log.tilt_err.append(float(abs(R_ref[2, 2] - R[2, 2])))
        log.max_dq_ratio.append(float(np.max(np.abs(dq) / core.assets.velocity_limit)))
        log.clamps.append(0)
        if hooks is not None:
            hooks(tick, q, quat, fsm.state)
        io.finish_tick()
        if realtime:
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, fsm.dt - elapsed))
    return log
