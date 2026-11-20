"""Unitree G1 DDS IO (real robot).

Canonical low-level pattern for the 29-DoF G1:
  * lowcmd  (rt/lowcmd,  unitree_hg LowCmd_)  published every 2 ms;
    each motor cmd carries mode/kp/kd/q_target exactly like the sim IO;
  * lowstate (rt/lowstate, unitree_hg LowState_) provides q/dq, IMU
    quaternion + body gyro, mode_machine and the wireless controller.

G1 29-DoF motor order matches the Uni natural joint order 1:1
(left leg 0-5, right leg 6-11, waist 12-14, left arm 15-21, right arm 22-28).

This module is only exercised when the operator starts run_robot.py manually.
"""

from __future__ import annotations

import time

import numpy as np

NUM_JOINTS = 29
CMD_DT = 0.002  # 500 Hz lowcmd publish rate


class KeyMap:
    R1 = 0
    L1 = 1
    START = 2
    SELECT = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    UP = 12
    RIGHT = 13
    DOWN = 14
    LEFT = 15


class RobotIO:
    def __init__(self, network_interface: str):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, network_interface)
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_cmd.mode_pr = 0  # Mode.PR
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()
        self.low_state = None
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._on_state, 10)
        self.crc = CRC()
        self.keys = 0
        self._ticks = 0
        deadline = time.time() + 5.0
        while self.low_state is None:
            if time.time() > deadline:
                raise TimeoutError("no rt/lowstate within 5 s; check network and robot mode")
            time.sleep(0.01)

    def _on_state(self, msg):
        self.low_state = msg
        wc = getattr(msg, "wireless_controller", None)
        if wc:
            self.keys = int.from_bytes(bytes(wc[0:2]), "little")

    def read(self):
        s = self.low_state
        q = np.array([s.motor_state[i].q for i in range(NUM_JOINTS)], dtype=np.float64)
        dq = np.array([s.motor_state[i].dq for i in range(NUM_JOINTS)], dtype=np.float64)
        gyro = np.asarray(s.imu_state.gyroscope, dtype=np.float64)
        quat = np.asarray(s.imu_state.quaternion, dtype=np.float64)  # w, x, y, z
        return q, dq, gyro, quat, self.keys

    def mode_machine(self) -> int:
        return int(self.low_state.mode_machine)

    def send(self, q_target, kp, kd):
        self._last_target = np.asarray(q_target, dtype=np.float64).copy()
        self._last_kp = np.asarray(kp, dtype=np.float64).copy()
        self._last_kd = np.asarray(kd, dtype=np.float64).copy()
        self._publish()

    def _publish(self):
        cmd = self.low_cmd
        cmd.mode_machine = self.mode_machine()
        for i in range(NUM_JOINTS):
            mc = cmd.motor_cmd[i]
            mc.mode = 1
            mc.q = float(self._last_target[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self._last_kp[i])
            mc.kd = float(self._last_kd[i])
        cmd.crc = self.crc.Crc(cmd)
        self.cmd_pub.Write(cmd)

    def finish_tick(self):
        # the runner loop is 50 Hz; lowcmd is republished at 500 Hz between ticks
        for _ in range(int(0.02 / CMD_DT) - 1):
            time.sleep(CMD_DT)
            self._publish()
