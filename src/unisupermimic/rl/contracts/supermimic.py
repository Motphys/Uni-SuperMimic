"""Injected task contract for SuperMimic world-model training."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

State = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

SUPERMIMIC_TRANSITION_KEY = "supermimic"
SUPERMIMIC_STATE_KEY = "supermimic_state"


@runtime_checkable
class SuperMimicMdpProtocol(Protocol):
    """Differentiable analytic heads and reference data owned by a task."""

    num_joints: int
    context_dim: int
    obs_dim: int
    goal_frame_dim: int
    max_step_reward: float

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor: ...

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor: ...

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor: ...

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor: ...

    def soft_termination(self, state: State, frames: torch.Tensor) -> torch.Tensor: ...

    def critic_features(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor: ...

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor: ...

    def valid_imagination_start(self, frames: torch.Tensor, horizon: int) -> torch.Tensor: ...

    def to(self, device: str | torch.device) -> SuperMimicMdpProtocol:
        """Return a copy on ``device`` for learner-side use off the env device."""
        ...


@runtime_checkable
class SupportsSuperMimicProtocol(Protocol):
    """Environment exposing its task-owned SuperMimic implementation."""

    @property
    def supermimic_mdp(self) -> SuperMimicMdpProtocol: ...


def get_supermimic_mdp(env: Any) -> SuperMimicMdpProtocol:
    """Return the injected task MDP, failing closed for unsupported envs."""
    if not isinstance(env, SupportsSuperMimicProtocol):
        raise TypeError(f"{type(env).__name__} does not provide the SuperMimic MDP contract")
    return env.supermimic_mdp


__all__ = [
    "SUPERMIMIC_STATE_KEY",
    "SUPERMIMIC_TRANSITION_KEY",
    "State",
    "SuperMimicMdpProtocol",
    "SupportsSuperMimicProtocol",
    "get_supermimic_mdp",
]
