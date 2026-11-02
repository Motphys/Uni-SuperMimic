"""Numerical properties of the differentiable heading projection."""

import torch

from unisupermimic.rl.algos.supermimic.geometry import so3_exp, yaw_mat


def test_heading_gradient_remains_bounded_near_vertical() -> None:
    perturbation = torch.zeros(1, 3, requires_grad=True)
    vertical = so3_exp(torch.tensor([[0.0, torch.pi / 2, 0.0]]))
    heading = yaw_mat(so3_exp(perturbation) @ vertical)
    (gradient,) = torch.autograd.grad(heading[0, 1, 0], perturbation)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() < 100


def test_heading_is_exact_at_upright_and_bounded_at_vertical() -> None:
    upright = yaw_mat(torch.eye(3).unsqueeze(0))
    torch.testing.assert_close(upright, torch.eye(3).unsqueeze(0), atol=1e-6, rtol=0)
    vertical = yaw_mat(so3_exp(torch.tensor([[0.0, torch.pi / 2, 0.0]])))
    assert torch.isfinite(vertical).all()
    assert vertical[0, :2, :2].abs().max() < 1e-4
    torch.testing.assert_close(vertical[0, 2, 2], torch.tensor(1.0))


def test_near_vertical_headings_are_continuous_across_opposite_directions() -> None:
    pitch = torch.pi / 2
    two_sides = so3_exp(torch.tensor([[0.0, pitch - 1e-4, 0.0], [0.0, pitch + 1e-4, 0.0]]))
    assert (two_sides[0] - two_sides[1]).abs().max() < 3e-4
    headings = yaw_mat(two_sides)
    assert torch.isfinite(headings).all()
    assert (headings[0] - headings[1]).abs().max() < 0.02
