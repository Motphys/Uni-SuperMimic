"""Numerical tests for SuperMimic networks."""

from __future__ import annotations

import warnings

import torch

from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.networks import (
    Critic,
    Policy,
    RunningNorm,
    WorldModel,
    mlp,
    run_mlp,
)
from unisupermimic.rl.contracts.supermimic import State


def _state(batch_size: int, num_joints: int) -> State:
    rotation = torch.eye(3).expand(batch_size, 3, 3).clone()
    return (
        torch.randn(batch_size, 3),
        rotation,
        torch.randn(batch_size, 3),
        torch.randn(batch_size, 3),
        torch.randn(batch_size, num_joints),
        torch.randn(batch_size, num_joints),
    )


def test_running_norm_tracks_feature_statistics() -> None:
    norm = RunningNorm(2)
    values = torch.tensor([[[1.0, 2.0], [3.0, 6.0]]])

    norm.update(values)
    normalized = norm(values)

    torch.testing.assert_close(norm.mean, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(norm.var, torch.tensor([1.0, 4.0]))
    assert norm.count.item() == 2
    assert normalized.shape == values.shape
    assert normalized.dtype == torch.float32


def test_world_model_zero_init_step_is_identity() -> None:
    num_joints = 3
    context_dim = 4
    model = WorldModel(hidden=(16, 16), num_joints=num_joints, context_dim=context_dim)
    state = _state(5, num_joints)
    action = torch.randn(5, num_joints)
    context = torch.randn(5, context_dim)

    inputs = model.inputs(state, action, context)
    targets = model.targets(state, state)
    next_state = model.step(state, action, context)
    error = model.state_error(next_state, state)

    assert inputs.shape == (5, 13 + 3 * num_joints + context_dim)
    assert targets.shape == (5, 12 + 2 * num_joints)
    assert error.shape == targets.shape
    assert inputs.dtype == targets.dtype == error.dtype == torch.float32
    for actual, expected in zip(next_state, state, strict=True):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(error, torch.zeros_like(error), rtol=0, atol=0)


def test_world_model_integrates_frame_local_increments() -> None:
    model = WorldModel(hidden=(4,), num_joints=2, context_dim=1)
    rotation = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    state = (
        torch.zeros(1, 3),
        rotation,
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
    )
    increment = torch.zeros(16)
    increment[0:3] = torch.tensor([1.0, 2.0, 3.0])
    increment[6:9] = torch.tensor([0.1, 0.2, 0.3])
    increment[9:12] = torch.tensor([0.4, 0.5, 0.6])
    increment[12:14] = torch.tensor([0.7, 0.8])
    increment[14:16] = torch.tensor([0.9, 1.0])
    with torch.no_grad():
        model.out_norm.mean.copy_(increment)

    next_state = model.step(state, torch.zeros(1, 2), torch.zeros(1, 1))

    torch.testing.assert_close(
        next_state[0],
        (rotation @ increment[0:3].unsqueeze(-1)).squeeze(-1),
    )
    torch.testing.assert_close(next_state[1], rotation)
    torch.testing.assert_close(
        next_state[2],
        (rotation @ increment[6:9].unsqueeze(-1)).squeeze(-1),
    )
    torch.testing.assert_close(
        next_state[3],
        (rotation @ increment[9:12].unsqueeze(-1)).squeeze(-1),
    )
    torch.testing.assert_close(next_state[4], increment[12:14].unsqueeze(0))
    torch.testing.assert_close(next_state[5], increment[14:16].unsqueeze(0))


def test_policy_and_critic_shapes_and_dtypes() -> None:
    batch_size = 4
    num_joints = 3
    policy = Policy(obs_dim=5, goal_dim=7, hist=4, z_dim=3, num_joints=num_joints)
    critic = Critic(in_dim=9)

    action = policy(torch.randn(batch_size, 4, 5), torch.randn(batch_size, 7))
    value = critic(torch.randn(batch_size, 9))

    assert action.shape == (batch_size, num_joints)
    assert value.shape == (batch_size,)
    assert action.dtype == value.dtype == torch.float32
    assert torch.count_nonzero(policy.dec[-1].bias) == 0


def test_run_mlp_autocast_follows_input_device() -> None:
    net = mlp(4, (8,), 1)
    x = torch.randn(3, 4)
    previous = networks.AMP_DTYPE
    try:
        networks.AMP_DTYPE = torch.bfloat16
        with warnings.catch_warnings():
            # CPU inputs must skip autocast silently (no "Disabling autocast").
            warnings.simplefilter("error")
            out = run_mlp(net, x)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, net(x))
    finally:
        networks.AMP_DTYPE = previous


def test_run_mlp_autocasts_on_mps() -> None:
    if not torch.backends.mps.is_available():
        return
    net = mlp(4, (8,), 1).to("mps")
    x = torch.randn(3, 4, device="mps")
    previous = networks.AMP_DTYPE
    try:
        networks.AMP_DTYPE = torch.bfloat16
        out = run_mlp(net, x)
        assert out.dtype == torch.float32
        assert out.device.type == "mps"
        torch.testing.assert_close(out.cpu(), net.cpu()(x.cpu()), rtol=2e-2, atol=2e-3)
    finally:
        networks.AMP_DTYPE = previous
