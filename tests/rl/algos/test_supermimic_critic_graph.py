"""CUDA-graph critic training (``algo.critic_update=graph``) against the eager loop."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_supermimic_learner import FakeEnv, _config

from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
LEARNING_RATES = (1e-3, 1e-3, 2e-4, 5e-5)


def _config_for(**overrides):
    # Isolate the critic loop: the other launch-overhead switches stay on their original
    # (eager / synced) paths whichever device the test runs on.
    values = dict(
        amp="bf16",
        critic_epochs=3,
        critic_minibatches=2,
        critic_update="graph",
        wm_graph=False,
        policy_graph=False,
        stats_on_device=False,
    )
    values.update(overrides)
    return replace(_config(), **values)


def _set_lr(learner: SuperMimicLearner, learning_rate: float) -> None:
    """Re-bind a Python float on every param group, exactly like ``train()`` does."""
    for optimizer in (learner.opt_wm, learner.opt_pi, learner.opt_v):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


def _update(learner: SuperMimicLearner, seed: int):
    torch.manual_seed(seed)
    return learner.policy_update()


def _assert_same_critic(expected: SuperMimicLearner, actual: SuperMimicLearner) -> None:
    for (name, reference), (other, value) in zip(
        expected.critic.state_dict().items(), actual.critic.state_dict().items(), strict=True
    ):
        assert name == other
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    for reference, value in zip(expected._critic_params, actual._critic_params, strict=True):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                actual.opt_v.state[value][key], expected.opt_v.state[reference][key], rtol=0, atol=0
            )


def test_invalid_critic_update_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="critic_update must be 'auto', 'eager' or 'graph'"):
        SuperMimicLearner(FakeEnv(), replace(_config(), critic_update="compiled"), tmp_path)


def test_graph_critic_update_requires_nvidia_cuda(tmp_path) -> None:
    # An explicit ``graph`` is a hard request; only ``auto`` follows the device.
    with pytest.raises(ValueError, match="critic_update='graph' requires an NVIDIA CUDA"):
        SuperMimicLearner(FakeEnv(), _config_for(critic_update="graph"), tmp_path)


def test_auto_critic_update_keeps_the_eager_loop_off_cuda(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config_for(critic_update="auto"), tmp_path)

    assert learner._critic_graph_enabled is False
    assert learner.opt_v.defaults["capturable"] is False
    learner.collect(3)
    learner.wm_update()
    learner.policy_update()

    assert learner._critic_graph_state is None
    # The eager loop leaves gradients allocated (no capture turned them into buffers).
    assert all(parameter.grad is not None for parameter in learner._critic_params)


@needs_cuda
def test_graph_critic_update_matches_eager_with_a_changing_learning_rate(tmp_path) -> None:
    eager = SuperMimicLearner(
        FakeEnv(), _config_for(critic_update="eager"), tmp_path / "eager", device="cuda"
    )
    graph = SuperMimicLearner(FakeEnv(), _config_for(), tmp_path / "graph", device="cuda")
    assert eager._critic_graph_enabled is False
    assert graph._critic_graph_enabled is True
    assert graph.opt_v.defaults["capturable"] is True
    for learner in (eager, graph):
        learner.collect(3)

    for iteration, learning_rate in enumerate(LEARNING_RATES):
        losses, generators = [], []
        for learner in (eager, graph):
            _set_lr(learner, learning_rate)
            losses.append(_update(learner, 1234 + iteration))
            generators.append(torch.cuda.get_rng_state().clone())
        # The permutations are drawn exactly as in the eager loop: same stream consumed.
        assert torch.equal(generators[0], generators[1])
        # The reported loss is the last minibatch's, as in eager.
        torch.testing.assert_close(losses[1][1], losses[0][1], rtol=0, atol=0)
        _assert_same_critic(eager, graph)


@needs_cuda
def test_graph_critic_update_tracks_the_learning_rate_after_capture(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config_for(), tmp_path, device="cuda")
    learner.collect(3)
    _set_lr(learner, 1e-3)
    _update(learner, 7)  # captures the graph at lr 1e-3
    params = learner._critic_params
    saved = [parameter.detach().clone() for parameter in params]
    saved_state = [
        {key: value.clone() for key, value in learner.opt_v.state[parameter].items()}
        for parameter in params
    ]
    generator = torch.cuda.get_rng_state().clone()

    deltas = []
    for learning_rate in (1e-3, 1e-5):
        # Rewind in place, which also checks that replays read the live storages.
        with torch.no_grad():
            for parameter, snapshot in zip(params, saved, strict=True):
                parameter.copy_(snapshot)
            for parameter, snapshot in zip(params, saved_state, strict=True):
                for key, value in learner.opt_v.state[parameter].items():
                    value.copy_(snapshot[key])
        torch.cuda.set_rng_state(generator)
        _set_lr(learner, learning_rate)
        _update(learner, 99)
        deltas.append(
            max(
                (parameter.detach() - snapshot).abs().max().item()
                for parameter, snapshot in zip(params, saved, strict=True)
            )
        )
        assert learner.opt_v.param_groups[0]["lr"] is learner._v_lr
        assert learner._v_lr.item() == pytest.approx(learning_rate, rel=1e-6)

    # Adam's step scales with lr: a frozen (captured) lr would give two equal updates.
    assert 30.0 < deltas[0] / deltas[1] < 300.0


@needs_cuda
def test_graph_replay_and_permutations_leave_the_generator_as_eager(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config_for(), tmp_path, device="cuda")
    learner.collect(3)
    _set_lr(learner, 1e-3)
    _update(learner, 3)
    count = learner.cfg.horizon * learner.cfg.pi_batch
    inputs = torch.randn(count, learner.critic.norm.mean.numel(), device="cuda")
    targets = torch.randn(count, device="cuda")

    before = torch.cuda.get_rng_state().clone()
    learner._critic_graph_state.graph.replay()
    assert torch.equal(before, torch.cuda.get_rng_state())

    torch.manual_seed(11)
    for _ in range(learner.cfg.critic_epochs):
        torch.randperm(count, device="cuda")
    expected = torch.cuda.get_rng_state().clone()
    torch.manual_seed(11)
    learner._critic_train_graph(inputs, targets, count)
    assert torch.equal(expected, torch.cuda.get_rng_state())


@needs_cuda
def test_graph_critic_loss_outlives_later_replays(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config_for(), tmp_path, device="cuda")
    learner.collect(3)
    _set_lr(learner, 1e-3)
    first = _update(learner, 3)[1]
    kept = first.clone()
    second = _update(learner, 4)[1]

    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, kept)


@needs_cuda
def test_graph_critic_recaptures_after_resume(tmp_path) -> None:
    source = SuperMimicLearner(FakeEnv(), _config_for(), tmp_path / "source", device="cuda")
    source.collect(3)
    _set_lr(source, 1e-3)
    _update(source, 1)
    source.save()

    resumed = SuperMimicLearner(FakeEnv(), _config_for(), tmp_path / "resumed", device="cuda")
    resumed.collect(3)
    _set_lr(resumed, 1e-3)
    _update(resumed, 5)  # captured against the fresh optimizer state
    captured = resumed._critic_graph_state
    resumed.load(tmp_path / "source" / "last.pt")  # swaps the optimizer state tensors
    graph_out = _update(resumed, 42)

    reference = SuperMimicLearner(
        FakeEnv(), _config_for(critic_update="eager"), tmp_path / "reference", device="cuda"
    )
    reference.collect(3)
    # Same hidden value-normalizer state (``_v_mean`` / ``_v_std`` are not checkpointed).
    _set_lr(reference, 1e-3)
    _update(reference, 5)
    reference.load(tmp_path / "source" / "last.pt")
    eager_out = _update(reference, 42)

    assert resumed._critic_graph_state is not captured
    assert torch.equal(graph_out[1], eager_out[1])
    _assert_same_critic(reference, resumed)
