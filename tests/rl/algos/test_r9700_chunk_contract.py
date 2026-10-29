"""Ensure full imagination batches retain row order and differentiability."""

import torch

from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner


def test_reward_batches_chunk_at_4096_without_losing_gradients() -> None:
    learner = object.__new__(SuperMimicLearner)
    sizes = []

    def reward_fn(state, frames, actions, previous):
        sizes.append(len(frames))
        reward = state[0].square().sum(-1) + frames.float() + (actions - previous).sum(-1)
        return reward, torch.sigmoid(reward)

    learner._reward_term_fn = reward_fn
    count = 8197
    position = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / count
    state = (position.clone().requires_grad_(),) + tuple(position for _ in range(5))
    frames = torch.arange(count)
    action = position.clone()
    old = position * 0
    reward, done = learner._batched_reward_term(state, frames, action, old)
    expected_reward, expected_done = reward_fn(state, frames, action, old)
    assert sizes == [4096, 4096, 5, count]
    assert torch.equal(reward, expected_reward)
    assert torch.equal(done, expected_done)
    reward.sum().backward()
    assert torch.equal(state[0].grad, 2 * state[0].detach())


def test_critic_batches_chunk_at_4096_without_losing_gradients() -> None:
    learner = object.__new__(SuperMimicLearner)
    sizes = []

    def critic_fn(state, frames, context, goal):
        sizes.append(len(frames))
        return torch.cat((state[0].square(), frames[:, None].float(), context, goal), -1)

    learner._critic_input_fn = critic_fn
    count = 8201
    position = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / count
    state = (position.clone().requires_grad_(),) + tuple(position for _ in range(5))
    frames = torch.arange(count)
    context, goal = position[:, :1], position[:, 1:]
    output = learner._batched_critic_input(state, frames, context, goal)
    assert sizes == [4096, 4096, 9]
    assert torch.equal(output, critic_fn(state, frames, context, goal))
    output.sum().backward()
    assert torch.equal(state[0].grad, 2 * state[0].detach())
