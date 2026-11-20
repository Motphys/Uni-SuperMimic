"""Torque-motor IO over Unitree's official 29-DoF G1 MJCF (sim backend B).

Models the real low-level chain: the 50 Hz runner updates (q_target, kp, kd);
a 1 kHz PD loop computes tau = kp*(q_t - q) - kd*dq from the latest target,
clipped to the per-joint effort limit, exactly like the motor driver.
"""

from __future__ import annotations

import numpy as np

from .sim_io import SimIO


class UnitreeTorqueIO(SimIO):
    def __init__(self, scene_xml: str, clip, joint_names, effort_limit,
                 keymap=None, start: str = "clip_frame0", gantry: bool = False,
                 pd_hz: float = 1000.0, ghost: bool = False):
        super().__init__(scene_xml, clip, joint_names, keymap=keymap, start=start, gantry=gantry,
                         ghost=ghost)
        self.model.opt.timestep = 1.0 / pd_hz
        self.sim_dt = self.model.opt.timestep
        self.substeps = int(round(self.control_dt / self.sim_dt))
        self.effort_limit = np.asarray(effort_limit, dtype=np.float64)
        self._qt = np.zeros(29)
        self._kp = np.zeros(29)
        self._kd = np.zeros(29)
        # motor actuators: gains live in ctrl directly, nothing to reprogram
        if not all(self.model.actuator_gaintype[a] == self.mj.mjtGain.mjGAIN_FIXED for a in self.act_ids):
            raise ValueError("expected fixed-gain motor actuators in the Unitree scene")

    def send(self, q_target, kp, kd):
        self._qt = np.asarray(q_target, dtype=np.float64).copy()
        self._kp = np.asarray(kp, dtype=np.float64).copy()
        self._kd = np.asarray(kd, dtype=np.float64).copy()

    def finish_tick(self):
        d = self.data
        ids = self.act_ids
        for _ in range(self.substeps):
            self._apply_gantry()
            q = d.qpos[self.joint_adr]
            dq = d.qvel[self.dof_adr]
            tau = self._kp * (self._qt - q) - self._kd * dq
            np.clip(tau, -self.effort_limit, self.effort_limit, out=tau)
            d.ctrl[ids] = tau
            self.mj.mj_step(self.model, d)
        d.xfrc_applied[self.pelvis_id] = 0.0
