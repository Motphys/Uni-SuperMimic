"""Deployment state machine with always-available damping.

States:
  PASSIVE       zero torque, startup state
  MOVE_TO_START cosine interpolation to the default stand pose (gains ramp in)
  STANDBY       self-balancing: policy tracks a frozen stand reference
  POLICY        self-balancing: policy tracks the selected motion clip
  DAMP          kp=0, kd=DAMP_KD everywhere; terminal until process restart

Transitions:
  PASSIVE/STANDBY --(L2+Up)--> MOVE_TO_START --auto--> STANDBY
  STANDBY --(R1+X)--> POLICY --clip end--auto--> STANDBY
  STANDBY --(L2+Down)--> PASSIVE
  any state --(L2+B)--> DAMP
"""

from __future__ import annotations

import numpy as np

from .contract import make_static_clip
from .core import yaw_quat

DAMP_KD = 3.0
INTERP_S = 4.0

PASSIVE = "PASSIVE"
MOVE_TO_START = "MOVE_TO_START"
STANDBY = "STANDBY"
POLICY = "POLICY"
DAMP = "DAMP"


def cosine_blend(q_from, q_to, t, duration):
    s = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / duration, 0.0, 1.0)))
    return (1.0 - s) * np.asarray(q_from) + s * np.asarray(q_to)


class FSM:
    def __init__(self, assets, core, control_dt=0.02, clips=None):
        self.assets = assets
        self.core = core
        self.clips = list(clips) if clips else [core.clip]
        self.clip_index = 0
        self.motion_clip = self.clips[0]
        # Pre-play reference is the frozen default stand, not the clip's
        # frame 0: frozen non-trivial poses are out-of-distribution for the
        # policy when reached from afar, while the stand is held robustly.
        self.standby_pre = make_static_clip(assets)
        self.standby_post = make_static_clip(assets)
        self.dt = control_dt
        self.state = PASSIVE
        self.t = 0.0
        self.state_t = 0.0
        self.q_from = assets.default_joint_pos.copy()
        self.abort_reason = None
        self.ever_played = False
        self.blend_start = None

    def cycle_clip(self, delta: int) -> str:
        """Switch the armed motion clip (allowed in PASSIVE/STANDBY)."""
        if self.state not in (PASSIVE, STANDBY):
            return self.motion_clip.name
        self.clip_index = (self.clip_index + delta) % len(self.clips)
        self.motion_clip = self.clips[self.clip_index]
        return self.motion_clip.name

    def policy_clip(self):
        """Clip the policy should track this tick."""
        return self.core.clip

    def _enter(self, state, q_now):
        self.state = state
        self.state_t = 0.0
        self.q_from = np.asarray(q_now, dtype=np.float64).copy()
        if state == MOVE_TO_START:
            # policy-driven settling on the stand reference (gravity compensated
            # by the policy, unlike an open-loop PD interpolation)
            self.core.switch_clip(self.standby_pre, 0, keep_history=False)
            self.blend_start = None
        elif state == STANDBY:
            clip = self.standby_post if self.ever_played else self.standby_pre
            self.core.switch_clip(clip, 0, keep_history=self.ever_played)
        elif state == POLICY:
            self.core.switch_clip(self.motion_clip, 0, keep_history=True)
            self.ever_played = True

    def request(self, event, q_now):
        """Button events: damp_anytime, to_start, play, to_passive, reset."""
        if event == "damp_anytime":
            self._enter(DAMP, q_now)
            return True
        if event == "reset":
            # recover from any state (incl. DAMP) to the just-started PASSIVE
            self.abort_reason = None
            self.ever_played = False
            self.core.reset(0)
            self._enter(PASSIVE, q_now)
            return True
        if self.state == DAMP:
            return False  # DAMP only leaves via reset
        if event == "to_start" and self.state in (PASSIVE, STANDBY):
            self._enter(MOVE_TO_START, q_now)
            return True
        if event == "play" and self.state == STANDBY:
            self._enter(POLICY, q_now)
            return True
        if event == "to_passive" and self.state == STANDBY:
            self._enter(PASSIVE, q_now)
            return True
        return False

    def set_start_heading(self, quat_wxyz):
        """Anchor the pre-play stand reference to the robot's current yaw.

        The frozen stand must face the way the robot actually faces at
        L2+Up time; an identity-yaw reference would make the policy rotate
        the whole body to world +x (observed on the real robot as a ~90 deg
        turn with the waist forcefully holding that heading).
        """
        self.standby_pre = make_static_clip(self.assets, root_quat_wxyz=yaw_quat(quat_wxyz))
        if self.state == MOVE_TO_START:
            self.core.switch_clip(self.standby_pre, 0, keep_history=True)

    def reanchor_motion_clip(self, quat_wxyz):
        """Yaw-rebase the motion clip to the robot's current heading.

        Called at STANDBY -> POLICY so the reference frame-0 heading matches
        the robot's actual heading (no instantaneous turn on play). The goal
        table is yaw-invariant and is kept unchanged. Always rebased from the
        ORIGINAL clip so names/frames never compound across plays.
        """
        from .contract import yaw_of_quat, yaw_rebase_clip
        original = self.clips[self.clip_index]
        delta = yaw_of_quat(quat_wxyz) - yaw_of_quat(original.root_quat_wxyz[0])
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        self.motion_clip = yaw_rebase_clip(original, delta)
        self.core.switch_clip(self.motion_clip, 0, keep_history=True)

    def reanchor_post_standby(self, quat_wxyz):
        """Anchor the post-play stand reference to the robot's current yaw.

        After a locomotion clip the robot stands several meters / some yaw away
        from the origin; the self-balancing stand reference must face the way
        the robot actually faces, otherwise the relative-rotation observation
        carries a large out-of-distribution yaw offset.
        """
        self.standby_post = make_static_clip(self.assets, root_quat_wxyz=yaw_quat(quat_wxyz))
        if self.state == STANDBY and self.ever_played:
            self.core.switch_clip(self.standby_post, 0, keep_history=True)

    def abort(self, reason, q_now):
        self.abort_reason = reason
        self._enter(DAMP, q_now)

    def update(self, q_now, policy_result=None):
        """Advance the FSM by one control tick; returns (q_target, kp, kd).

        In STANDBY and POLICY the caller must pass policy_result=(q_target,
        abort_reason) from DeployCore.step of this tick.
        """
        a = self.assets
        self.t += self.dt
        self.state_t += self.dt
        st = self.state
        if st == PASSIVE:
            return np.asarray(q_now, dtype=np.float64), np.zeros(29), np.zeros(29)
        if st == DAMP:
            return np.asarray(q_now, dtype=np.float64), np.zeros(29), np.full(29, DAMP_KD)
        if st == MOVE_TO_START:
            if policy_result is None:
                raise ValueError("MOVE_TO_START requires policy_result")
            q_t, abort = policy_result
            if abort is not None:
                self.abort(abort, q_now)
            elif self.state_t >= INTERP_S or self.core.clip_done:
                self._enter(STANDBY, q_now)
            return q_t, a.kp, a.kd
        if st in (STANDBY, POLICY):
            if policy_result is None:
                raise ValueError(f"{st} requires policy_result")
            q_t, abort = policy_result
            if abort is not None:
                self.abort(abort, q_now)
            elif st == POLICY and self.core.clip_done:
                self._enter(STANDBY, q_now)
            return q_t, a.kp, a.kd
        raise ValueError(f"unknown state {st!r}")
