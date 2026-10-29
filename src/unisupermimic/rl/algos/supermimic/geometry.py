"""Batched tensor geometry for SuperMimic (quaternions use wxyz)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from unisupermimic.rl.contracts.supermimic import State


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    candidates = torch.stack(
        (
            torch.stack(
                (qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1
            ),
            torch.stack(
                ((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1
            ),
            torch.stack(
                ((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1
            ),
            torch.stack(
                ((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1
            ),
        ),
        dim=-2,
    )
    indices = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(
        candidates, -2, indices[..., None, None].expand(indices.shape + (1, 4))
    ).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    x, y, z = axis.unbind(-1)
    cos, sin = torch.cos(angle), torch.sin(angle)
    complement = 1 - cos
    return torch.stack(
        (
            cos + x * x * complement,
            x * y * complement - z * sin,
            x * z * complement + y * sin,
            y * x * complement + z * sin,
            cos + y * y * complement,
            y * z * complement - x * sin,
            z * x * complement - y * sin,
            z * y * complement + x * sin,
            cos + z * z * complement,
        ),
        dim=-1,
    ).reshape(angle.shape + (3, 3))


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def matmul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first.unsqueeze(-1) * second.unsqueeze(-3)).sum(-2)


def matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (matrix * vector.unsqueeze(-2)).sum(-1)


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    theta2 = (vector * vector).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    skew_matrix = skew(vector)
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype).expand(skew_matrix.shape)
    return eye + a[..., None] * skew_matrix + b[..., None] * matmul(skew_matrix, skew_matrix)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    vector = q[..., 1:]
    sin_half = torch.linalg.norm(vector, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., :1])
    scale = torch.where(
        sin_half < 1e-6,
        2.0 / torch.clamp(q[..., :1], min=1e-6),
        2 * half / (sin_half + 1e-12),
    )
    return vector * scale


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    return quat_to_rotvec(mat_to_quat(rotation))


def rot_angle_sq(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    trace = (first * second).sum((-1, -2))
    cosine = torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine) ** 2


def yaw_mat(rotation: torch.Tensor) -> torch.Tensor:
    forward_x, forward_y = rotation[..., 0, 0], rotation[..., 1, 0]
    norm = torch.sqrt(forward_x * forward_x + forward_y * forward_y + 1e-9)
    cos, sin = forward_x / norm, forward_y / norm
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    return torch.stack((cos, -sin, zero, sin, cos, zero, zero, zero, one), -1).reshape(
        rotation.shape[:-2] + (3, 3)
    )


def mat_to_6d(rotation: torch.Tensor) -> torch.Tensor:
    return rotation[..., :, :2].reshape(rotation.shape[:-2] + (6,))


def quat_slerp(first: torch.Tensor, second: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dot = (first * second).sum(-1, keepdim=True)
    second = torch.where(dot < 0, -second, second)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin = torch.sin(theta)
    t = t[..., None]
    first_weight = torch.where(sin < 1e-6, 1 - t, torch.sin((1 - t) * theta) / (sin + 1e-12))
    second_weight = torch.where(sin < 1e-6, t, torch.sin(t * theta) / (sin + 1e-12))
    return F.normalize(first_weight * first + second_weight * second, dim=-1)


def flatten_state(state: State) -> torch.Tensor:
    position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
    return torch.cat((position, mat_to_quat(rotation), lin_vel, ang_vel, joint_pos, joint_vel), -1)


def unflatten_state(value: torch.Tensor) -> State:
    num_joints = (value.shape[-1] - 13) // 2
    joint_end = 13 + num_joints
    return (
        value[..., 0:3],
        quat_to_mat(value[..., 3:7]),
        value[..., 7:10],
        value[..., 10:13],
        value[..., 13:joint_end],
        value[..., joint_end:],
    )


def invariant_features(state: State) -> torch.Tensor:
    position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
    heading = yaw_mat(rotation)
    rotation_t = rotation.transpose(-1, -2)
    return torch.cat(
        (
            position[..., 2:3],
            mat_to_6d(matmul(heading.transpose(-1, -2), rotation)),
            matvec(rotation_t, lin_vel),
            matvec(rotation_t, ang_vel),
            joint_pos,
            joint_vel,
        ),
        -1,
    )


__all__ = [
    "axis_angle_to_mat",
    "flatten_state",
    "invariant_features",
    "mat_to_6d",
    "mat_to_quat",
    "matmul",
    "matvec",
    "quat_slerp",
    "quat_to_mat",
    "quat_to_rotvec",
    "rot_angle_sq",
    "skew",
    "so3_exp",
    "so3_log",
    "unflatten_state",
    "yaw_mat",
]
