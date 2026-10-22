"""World model, goal-conditioned policy, and critic for SuperMimic."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from unisupermimic.rl.contracts.supermimic import State

from .geometry import invariant_features, so3_exp, so3_log, yaw_mat


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    act: type[nn.Module] = nn.ELU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for hidden_dim in hidden:
        layers.append(nn.Linear(dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean and variance, frozen once ``max_count`` is reached."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.mean: torch.Tensor
        self.var: torch.Tensor
        self.count: torch.Tensor
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        batch_count = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total
        ) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)


AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run an MLP under optional autocast, returning fp32 for state arithmetic."""
    if AMP_DTYPE is None:
        return net(x)
    device_type = x.device.type
    if device_type not in ("cuda", "mps"):
        return net(x)
    with torch.autocast(device_type, dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Predict frame-local state increments and integrate them in fp32."""

    def __init__(
        self,
        hidden: Sequence[int] = (2048, 2048, 2048),
        num_joints: int = 29,
        context_dim: int = 33,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.context_dim = context_dim
        self.in_dim = 13 + 3 * num_joints + context_dim
        self.out_dim = 12 + 2 * num_joints
        self.net = mlp(self.in_dim, hidden, self.out_dim, act=nn.SiLU, layer_norm=True)
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(
        self, state: State, action: torch.Tensor, normalized_context: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((invariant_features(state), action, normalized_context), -1)

    @staticmethod
    def targets(state: State, next_state: State) -> torch.Tensor:
        """Return frame-local increments from ``state`` to ``next_state``."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        next_position, next_rotation, next_lin_vel, next_ang_vel, next_joint_pos, next_joint_vel = (
            next_state
        )
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                (heading.transpose(-1, -2) @ (next_position - position).unsqueeze(-1)).squeeze(-1),
                so3_log(rotation_t @ next_rotation),
                (rotation_t @ (next_lin_vel - lin_vel).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (next_ang_vel - ang_vel).unsqueeze(-1)).squeeze(-1),
                next_joint_pos - joint_pos,
                next_joint_vel - joint_vel,
            ),
            -1,
        )

    def step(
        self,
        state: State,
        action: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> State:
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        prediction = run_mlp(
            self.net,
            self.in_norm(self.inputs(state, action, normalized_context)),
        )
        increment = prediction * self.out_norm.std() + self.out_norm.mean
        heading = yaw_mat(rotation)
        d_position, d_rotation, d_lin_vel, d_ang_vel, d_joint_pos, d_joint_vel = increment.split(
            (3, 3, 3, 3, self.num_joints, self.num_joints), -1
        )
        next_position = position + (heading @ d_position.unsqueeze(-1)).squeeze(-1)
        next_rotation = rotation @ so3_exp(d_rotation)
        next_lin_vel = lin_vel + (rotation @ d_lin_vel.unsqueeze(-1)).squeeze(-1)
        next_ang_vel = ang_vel + (rotation @ d_ang_vel.unsqueeze(-1)).squeeze(-1)
        return (
            next_position,
            next_rotation,
            next_lin_vel,
            next_ang_vel,
            joint_pos + d_joint_pos,
            joint_vel + d_joint_vel,
        )

    @staticmethod
    def state_error(predicted_state: State, state: State) -> torch.Tensor:
        """Return frame-local errors in the same layout as ``targets``."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        (
            predicted_position,
            predicted_rotation,
            predicted_lin_vel,
            predicted_ang_vel,
            predicted_joint_pos,
            predicted_joint_vel,
        ) = predicted_state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                (heading.transpose(-1, -2) @ (predicted_position - position).unsqueeze(-1)).squeeze(
                    -1
                ),
                so3_log(rotation_t @ predicted_rotation),
                (rotation_t @ (predicted_lin_vel - lin_vel).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (predicted_ang_vel - ang_vel).unsqueeze(-1)).squeeze(-1),
                predicted_joint_pos - joint_pos,
                predicted_joint_vel - joint_vel,
            ),
            -1,
        )


class Policy(nn.Module):
    """Encode a goal and decode it together with an observation history."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        num_joints: int = 29,
        hidden: Sequence[int] = (2048, 1024, 512, 256),
    ):
        super().__init__()
        self.hist = hist
        self.enc = mlp(goal_dim, hidden, z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, hidden, num_joints)
        output = self.dec[-1]
        assert isinstance(output, nn.Linear)
        with torch.no_grad():
            output.weight.mul_(0.01)
            output.bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Consume ``(batch, hist, obs_dim)`` observations, newest last."""
        latent = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        obs = self.obs_norm(obs_hist).flatten(-2)
        return run_mlp(self.dec, torch.cat((latent, obs), -1))


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 128)):
        super().__init__()
        self.net = mlp(in_dim, hidden, 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.norm(x)).squeeze(-1)


__all__ = ["AMP_DTYPE", "Critic", "Policy", "RunningNorm", "WorldModel", "run_mlp"]
