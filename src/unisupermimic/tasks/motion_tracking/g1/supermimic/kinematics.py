"""Torch forward kinematics built from a G1 MJCF on the cold path."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch

from unisupermimic.rl.algos.supermimic.geometry import axis_angle_to_mat, quat_to_mat

from .robot import G1_JOINT_NAMES, TRACKED_BODIES


class G1Kinematics:
    def __init__(
        self,
        mjcf_path: str,
        device: str | torch.device,
        tracked: Sequence[str] = TRACKED_BODIES,
    ):
        import mujoco

        model = mujoco.MjModel.from_xml_path(mjcf_path)  # pyright: ignore[reportAttributeAccessIssue]
        self.device = torch.device(device)
        joint_name_to_idx = {name: index for index, name in enumerate(G1_JOINT_NAMES)}
        bodies: list[str] = []
        parent: list[int] = []
        joint_idx: list[int] = []
        axes, offsets_p, offsets_r = [], [], []
        mj_to_local = {}
        for body_id in range(1, model.nbody):
            name = model.body(body_id).name
            if model.body_jntnum[body_id] == 0:
                continue
            joint_id = model.body_jntadr[body_id]
            mj_to_local[body_id] = len(bodies)
            bodies.append(name)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:  # pyright: ignore[reportAttributeAccessIssue]
                parent.append(-1)
                joint_idx.append(-1)
            else:
                parent_id = model.body_parentid[body_id]
                while parent_id not in mj_to_local:
                    raise RuntimeError(f"fixed parent for {name}")
                parent.append(mj_to_local[parent_id])
                joint_idx.append(joint_name_to_idx[model.joint(joint_id).name])
            axes.append(model.jnt_axis[joint_id].tolist())
            offsets_p.append(model.body_pos[body_id].tolist())
            offsets_r.append(model.body_quat[body_id].tolist())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(axes, dtype=torch.float32, device=self.device)
        self.off_p = torch.tensor(offsets_p, dtype=torch.float32, device=self.device)
        self.off_R = quat_to_mat(torch.tensor(offsets_r, dtype=torch.float32, device=self.device))
        self.tracked_idx = [bodies.index(name) for name in tracked]
        self.num_bodies = len(bodies)

        lower = [0.0] * len(G1_JOINT_NAMES)
        upper = [0.0] * len(G1_JOINT_NAMES)
        for joint_id in range(model.njnt):
            name = model.joint(joint_id).name
            if name in joint_name_to_idx:
                index = joint_name_to_idx[name]
                lower[index] = float(model.jnt_range[joint_id][0])
                upper[index] = float(model.jnt_range[joint_id][1])
        self.q_lo = torch.tensor(lower, dtype=torch.float32, device=self.device)
        self.q_hi = torch.tensor(upper, dtype=torch.float32, device=self.device)

    def to(self, device: str | torch.device) -> G1Kinematics:
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

    def forward(
        self,
        p: torch.Tensor,
        R: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        all_bodies: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tracked-body world poses and link-origin linear/angular velocities."""
        positions = [p]
        rotations = [R]
        linear_velocities = [v]
        angular_velocities = [w]
        for index in range(1, self.num_bodies):
            parent_index = self.parent[index]
            joint_index = self.joint_idx[index]
            parent_rotation = rotations[parent_index]
            position = positions[parent_index] + (
                parent_rotation @ self.off_p[index].unsqueeze(-1)
            ).squeeze(-1)
            offset_rotation = parent_rotation @ self.off_R[index]
            rotation = offset_rotation @ axis_angle_to_mat(self.axes[index], q[..., joint_index])
            axis_world = (offset_rotation @ self.axes[index].unsqueeze(-1)).squeeze(-1)
            angular_velocity = (
                angular_velocities[parent_index]
                + axis_world * qd[..., joint_index : joint_index + 1]
            )
            linear_velocity = linear_velocities[parent_index] + torch.cross(
                angular_velocities[parent_index],
                position - positions[parent_index],
                dim=-1,
            )
            positions.append(position)
            rotations.append(rotation)
            linear_velocities.append(linear_velocity)
            angular_velocities.append(angular_velocity)
        indices = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([positions[index] for index in indices], -2),
            torch.stack([rotations[index] for index in indices], -3),
            torch.stack([linear_velocities[index] for index in indices], -2),
            torch.stack([angular_velocities[index] for index in indices], -2),
        )
