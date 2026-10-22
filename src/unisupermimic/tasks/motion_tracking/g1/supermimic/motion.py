"""Reference G1 motions resampled to 50 Hz with tracked-body forward kinematics."""

from __future__ import annotations

import copy
import csv
import math
import os
from collections.abc import Sequence

import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    mat_to_6d,
    quat_slerp,
    quat_to_mat,
    so3_log,
    yaw_mat,
)
from unisupermimic.rl.contracts.supermimic import State

from .kinematics import G1Kinematics

CONTROL_DT = 0.02
BIN_FRAMES = 50


class MotionLib:
    def __init__(
        self,
        files: Sequence[str],
        kin: G1Kinematics,
        device: str | torch.device,
        src_fps: float = 30.0,
    ):
        if not files:
            raise ValueError("MotionLib requires at least one CSV file")
        self.device = torch.device(device)
        self.names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        positions, quaternions, joint_positions = [], [], []
        lengths = []
        for file in files:
            raw = _load_csv(file)
            if raw.shape[0] < 2 or raw.shape[1] != 36:
                raise ValueError(f"{file} must have at least two rows and 36 columns")
            position, quat_xyzw, joints = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
            quaternion = torch.cat((quat_xyzw[:, 3:4], quat_xyzw[:, :3]), -1)
            source_frames = raw.shape[0]
            duration = (source_frames - 1) / src_fps
            num_frames = math.floor(duration / CONTROL_DT) + 1
            sample = torch.arange(num_frames, dtype=torch.float32) * CONTROL_DT * src_fps
            lower = torch.clamp(sample.floor().long(), max=source_frames - 2)
            alpha = (sample - lower.float()).clamp(0, 1)
            position = position[lower] * (1 - alpha[:, None]) + position[lower + 1] * alpha[:, None]
            joints = joints[lower] * (1 - alpha[:, None]) + joints[lower + 1] * alpha[:, None]
            quaternion = quat_slerp(quaternion[lower], quaternion[lower + 1], alpha)
            positions.append(position)
            quaternions.append(quaternion)
            joint_positions.append(joints)
            lengths.append(num_frames)

        self.clip_len = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.clip_start = torch.cumsum(
            torch.tensor([0, *lengths[:-1]], dtype=torch.long, device=self.device), 0
        )
        self.clip_end = self.clip_start + self.clip_len - 1
        self.num_frames = sum(lengths)
        self.clip_id = torch.cat(
            [torch.full((length,), index, dtype=torch.long) for index, length in enumerate(lengths)]
        ).to(self.device)

        root_p, root_R, root_v, root_w, joint_q, joint_qd = [], [], [], [], [], []
        for position, quaternion, joints in zip(positions, quaternions, joint_positions):
            rotation = quat_to_mat(quaternion)
            linear_velocity = _gradient(position, CONTROL_DT)
            joint_velocity = _gradient(joints, CONTROL_DT)
            angular_velocity = torch.zeros_like(position)
            angular_velocity[1:-1] = so3_log(rotation[2:] @ rotation[:-2].transpose(-1, -2)) / (
                2 * CONTROL_DT
            )
            angular_velocity[0] = so3_log(rotation[1] @ rotation[0].transpose(-1, -2)) / CONTROL_DT
            angular_velocity[-1] = (
                so3_log(rotation[-1] @ rotation[-2].transpose(-1, -2)) / CONTROL_DT
            )
            root_p.append(position)
            root_R.append(rotation)
            root_v.append(linear_velocity)
            root_w.append(angular_velocity)
            joint_q.append(joints)
            joint_qd.append(joint_velocity)

        self.root_p, self.root_R, self.root_v, self.root_w = (
            torch.cat(values).to(self.device) for values in (root_p, root_R, root_v, root_w)
        )
        self.q, self.qd = (torch.cat(values).to(self.device) for values in (joint_q, joint_qd))
        with torch.no_grad():
            body_p, body_R, body_v, body_w = [], [], [], []
            for start in range(0, self.num_frames, 65536):
                end = min(start + 65536, self.num_frames)
                output = kin.forward(
                    self.root_p[start:end],
                    self.root_R[start:end],
                    self.root_v[start:end],
                    self.root_w[start:end],
                    self.q[start:end],
                    self.qd[start:end],
                )
                body_p.append(output[0])
                body_R.append(output[1])
                body_v.append(output[2])
                body_w.append(output[3])
            self.body_p, self.body_R, self.body_v, self.body_w = (
                torch.cat(values) for values in (body_p, body_R, body_v, body_w)
            )

        root_rotation_t = self.root_R.transpose(-1, -2)
        self.root_v_local = (root_rotation_t @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (root_rotation_t @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]

        bin_clip, bin_start = [], []
        for clip in range(len(lengths)):
            num_bins = math.ceil(lengths[clip] / BIN_FRAMES)
            for bin_index in range(num_bins):
                bin_clip.append(clip)
                bin_start.append(int(self.clip_start[clip]) + bin_index * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, dtype=torch.long, device=self.device)
        self.bin_start = torch.tensor(bin_start, dtype=torch.long, device=self.device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=self.device)
        for bin_index in range(self.num_bins):
            clip = bin_clip[bin_index]
            start = bin_start[bin_index]
            end = min(start + BIN_FRAMES, int(self.clip_end[clip]) + 1)
            self.frame_bin[start:end] = bin_index

    def to(self, device: str | torch.device) -> MotionLib:
        """Return a copy on ``device``; the original instance is left untouched."""
        device = torch.device(device)
        if device == self.device:
            return self
        new = copy.copy(self)
        new.device = device
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(new, name, value.to(device))
        return new

    def future_frames(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Return frames+1 through frames+K, clamped at each clip's last frame."""
        end = self.clip_end[self.clip_id[frames]]
        future = frames.unsqueeze(-1) + torch.arange(1, num_frames + 1, device=frames.device)
        return torch.minimum(future, end.unsqueeze(-1))

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Build future reference features in the current reference heading frame."""
        future = self.future_frames(frames, num_frames)
        heading = yaw_mat(self.root_R[frames])
        heading_t = heading.transpose(-1, -2).unsqueeze(1)
        position_delta = (
            heading_t @ (self.root_p[future] - self.root_p[frames].unsqueeze(1)).unsqueeze(-1)
        ).squeeze(-1)
        rotation = mat_to_6d(heading_t @ self.root_R[future])
        features = torch.cat(
            (
                position_delta,
                rotation,
                self.root_v_local[future],
                self.root_w_local[future],
                self.q[future],
                self.qd[future] * 0.1,
            ),
            -1,
        )
        return features.reshape(frames.shape[0], -1)

    def ref_state(self, frames: torch.Tensor) -> State:
        return (
            self.root_p[frames],
            self.root_R[frames],
            self.root_v[frames],
            self.root_w[frames],
            self.q[frames],
            self.qd[frames],
        )


def _load_csv(path: str) -> torch.Tensor:
    with open(path, encoding="utf-8", newline="") as stream:
        rows = [
            [float(value) for value in row]
            for row in csv.reader(stream)
            if row and not row[0].lstrip().startswith("#")
        ]
    if not rows:
        raise ValueError(f"{path} is empty")
    return torch.tensor(rows, dtype=torch.float32)


def _gradient(value: torch.Tensor, dt: float) -> torch.Tensor:
    gradient = torch.zeros_like(value)
    gradient[1:-1] = (value[2:] - value[:-2]) / (2 * dt)
    gradient[0] = (value[1] - value[0]) / dt
    gradient[-1] = (value[-1] - value[-2]) / dt
    return gradient
