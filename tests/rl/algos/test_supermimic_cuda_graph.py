"""Explicit CUDA-graph replay of the SuperMimic world-model gradient step."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from tests.rl.algos.test_supermimic_learner import FakeEnv, FakeMdp, _config

from unisupermimic.rl.algos.supermimic.cuda_graph import CapturedUpdate, cumprod_without_host_sync
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner


def _eager(**overrides):
    """``_config()`` with every launch-overhead switch forced onto its original path.

    The switches default to ``auto`` (on for NVIDIA CUDA), so a test that isolates one of
    them pins the others to eager / synced whichever device it runs on.
    """
    values = dict(critic_update="eager", wm_graph=False, policy_graph=False, stats_on_device=False)
    values.update(overrides)
    return replace(_config(), **values)


def test_launch_overhead_switches_require_cuda_graphs(tmp_path) -> None:
    for switch in ("wm_graph", "policy_graph"):
        with pytest.raises(ValueError, match="requires a CUDA-graph capable device"):
            SuperMimicLearner(FakeEnv(), _eager(**{switch: True}), tmp_path / switch)
    with pytest.raises(ValueError, match="inductor_cpp_wrapper=True requires a CUDA device"):
        SuperMimicLearner(FakeEnv(), _eager(inductor_cpp_wrapper=True), tmp_path / "cpp")
    # ``auto`` (the default) keeps the eager path off NVIDIA CUDA: nothing is captured.
    learner = SuperMimicLearner(FakeEnv(), _eager(), tmp_path / "auto")
    assert learner._wm_graph is None
    assert learner._pi_graph is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_captured_update_draws_fresh_noise_and_keeps_the_eager_rng_sequence() -> None:
    def body(inputs: dict) -> torch.Tensor:
        return inputs["x"] + torch.rand_like(inputs["x"])

    captured = CapturedUpdate(warmup=1)
    inputs = {"x": torch.full((4096,), 2.0, device="cuda")}
    captured.step(body, inputs)  # eager warmup
    captured.step(body, inputs)  # capture, then the real (rewound) replay
    assert captured.replaying

    state = torch.cuda.get_rng_state()
    first = captured.step(body, inputs).clone()
    second = captured.step(body, inputs).clone()
    after_graph = torch.rand(4096, device="cuda")

    # Every replay draws new numbers ...
    assert not torch.equal(first, second)
    # ... and they are the numbers the eager sequence would have drawn, without
    # silently changing how much of the stream the graph consumes.
    torch.cuda.set_rng_state(state)
    eager_first = body(inputs)
    eager_second = body(inputs)
    after_eager = torch.rand(4096, device="cuda")
    torch.testing.assert_close(first, eager_first, rtol=0, atol=0)
    torch.testing.assert_close(second, eager_second, rtol=0, atol=0)
    torch.testing.assert_close(after_graph, after_eager, rtol=0, atol=0)

    with pytest.raises(ValueError, match="changed shape or dtype"):
        captured.step(body, {"x": torch.zeros(8, device="cuda")})
    with pytest.raises(ValueError, match="cannot switch bodies"):
        captured.step(lambda inputs: inputs["x"], inputs)
    with pytest.raises(ValueError, match="warmup must be positive"):
        CapturedUpdate(warmup=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("compile", "amp", "muon"),
    [(False, "none", False), (True, "bf16", True)],
)
def test_wm_update_graph_matches_eager_gradients_parameters_and_metrics(
    tmp_path, compile, amp, muon
) -> None:
    cfg = _eager(compile=compile, amp=amp, pi_opt="muon" if muon else "adam")
    eager = SuperMimicLearner(FakeEnv(), cfg, tmp_path / "eager", device="cuda")
    graph = SuperMimicLearner(
        FakeEnv(), replace(cfg, wm_graph=True), tmp_path / "graph", device="cuda"
    )
    for learner in (eager, graph):
        learner.collect(12)
    assert not graph._wm_graph.replaying

    worst = 0.0
    for step in range(7):  # 3 eager warmups, the capture call, then pure replays
        state = torch.cuda.get_rng_state()
        eager_metrics = eager.wm_update()
        torch.cuda.set_rng_state(state)
        graph_metrics = graph.wm_update()
        assert graph._wm_graph.replaying == (step >= 3)
        for expected, actual in zip(eager_metrics, graph_metrics, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        for parameter, reference in zip(graph.wm.parameters(), eager.wm.parameters(), strict=True):
            assert parameter.grad is not None and reference.grad is not None
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-3, atol=1e-5)
            torch.testing.assert_close(parameter, reference, rtol=1e-4, atol=1e-6)
            worst = max(
                worst,
                (parameter - reference).abs().max().item()
                / reference.abs().max().clamp(min=1e-12).item(),
            )
    assert worst < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_wm_update_graph_outputs_are_private_and_gradients_keep_their_address(
    tmp_path,
) -> None:
    learner = SuperMimicLearner(FakeEnv(), _eager(wm_graph=True), tmp_path / "graph", device="cuda")
    learner.collect(12)
    for _ in range(4):
        learner.wm_update()
    assert learner._wm_graph.replaying
    addresses = [parameter.grad.data_ptr() for parameter in learner.wm.parameters()]

    # Metrics that the training loop keeps across iterations must not alias the
    # graph's buffers, which the next replay overwrites.
    held = learner.wm_update()
    snapshot = [metric.clone() for metric in held]
    learner.wm_update()
    learner.wm_update()
    for metric, saved in zip(held, snapshot, strict=True):
        torch.testing.assert_close(metric, saved, rtol=0, atol=0)
    # The optimizer (outside the graph) keeps reading the same gradient storage.
    assert addresses == [parameter.grad.data_ptr() for parameter in learner.wm.parameters()]


class _GraphCheckMdp(FakeMdp):
    """FakeMdp as the real MDP behaves where the policy graph is sensitive.

    ``soft_termination`` depends on the state, so the survival ``cumprod`` is on the
    backward path (the plain ``FakeMdp`` returns constants, which would hide it), and
    ``add_noise`` optionally draws the real MDP's uniform observation noise, accumulated
    on-device so it also works inside a captured graph: ``first`` is overwritten in
    place by every update.
    """

    noise = None

    def soft_termination(self, state: State, frames: torch.Tensor) -> torch.Tensor:
        del frames
        return torch.sigmoid(2 * state[0][..., 0])

    def to(self, device: str | torch.device) -> _GraphCheckMdp:
        device = torch.device(device)
        self.sum = torch.zeros((), device=device)
        self.square_sum = torch.zeros((), device=device)
        self.count = torch.zeros((), device=device)
        self.first = torch.zeros((), device=device)
        self.moved_to = device
        return self

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        if self.noise is None:
            return obs
        unit = 2 * torch.rand_like(obs) - 1
        self.sum.add_(unit.sum())
        self.square_sum.add_(unit.square().sum())
        self.count.add_(unit.numel())
        self.first.copy_(unit.reshape(-1)[0])
        return obs + unit * self.noise


def _policy_pair(tmp_path, noise: bool, **overrides):
    """Eager and policy-graph learners with the same seed and MDP."""
    cfg = _eager(pi_batch=64, horizon=4, **overrides)
    learners = []
    for name, policy_graph in (("eager", False), ("graph", True)):
        env = FakeEnv()
        env.mdp = _GraphCheckMdp()
        env.mdp.noise = 0.05 if noise else None
        learner = SuperMimicLearner(
            env, replace(cfg, policy_graph=policy_graph), tmp_path / name, device="cuda"
        )
        learner.collect(12)
        learners.append(learner)
    return learners


def _noise_statistics(learner: SuperMimicLearner) -> tuple[float, float]:
    mdp = learner.mdp
    mean = (mdp.sum / mdp.count).item()
    return mean, (mdp.square_sum / mdp.count - mean**2).sqrt().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("zeros", ["none", "middle", "first", "last", "two", "all"])
def test_cumprod_without_host_sync_matches_autograd_and_captures(zeros) -> None:
    torch.manual_seed(0)
    steps, batch = 12, 4096
    values = torch.rand(steps, batch, device="cuda") * 1.5 + 0.25
    upstream = torch.randn(steps, batch, device="cuda") * (
        torch.rand(steps, batch, device="cuda") > 0.3
    )
    columns = torch.arange(batch, device="cuda")[::2]
    if zeros != "none":
        values[{"middle": 5, "first": 0, "last": steps - 1, "two": 3, "all": 0}[zeros], columns] = 0
        if zeros == "two":
            values[8, columns[::2]] = 0
        if zeros == "all":
            values[:, columns] = 0

    reference = values.clone().requires_grad_(True)
    (torch.cumprod(reference, 0) * upstream).sum().backward()
    actual = values.clone().requires_grad_(True)
    loss = (cumprod_without_host_sync(actual) * upstream).sum()
    loss.backward()
    tolerance = {"rtol": 0, "atol": 0} if zeros == "none" else {"rtol": 1e-5, "atol": 1e-6}
    torch.testing.assert_close(actual.grad, reference.grad, **tolerance)

    # The backward itself has to be replayable (ATen's cannot be captured). Capture on a
    # fresh autograd graph: capturing one that already ran on the default stream is
    # rejected, and the capture run itself writes the gradients once.
    del loss, actual
    replayed = values.clone().requires_grad_(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (cumprod_without_host_sync(replayed) * upstream).sum().backward()
    replayed.grad.zero_()
    graph.replay()
    torch.testing.assert_close(replayed.grad, reference.grad, **tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("compile", "amp", "muon"), [(False, "none", False), (True, "bf16", True)])
def test_policy_update_graph_matches_eager_without_noise(tmp_path, compile, amp, muon) -> None:
    optimizer = "muon" if muon else "adam"
    eager, graph = _policy_pair(
        tmp_path,
        noise=False,
        compile=compile,
        amp=amp,
        pi_opt=optimizer,
        wm_opt=optimizer,
        pi_ema=0.995,
    )
    assert not graph._pi_graph.replaying
    for step in range(7):  # 3 eager warmups, the capture call, then pure replays
        state = torch.cuda.get_rng_state()
        expected = eager.policy_update()
        torch.cuda.set_rng_state(state)
        actual = graph.policy_update()
        assert graph._pi_graph.replaying == (step >= 3)
        for reference, value in zip(expected, actual, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        for parameter, reference in zip(graph.pi.parameters(), eager.pi.parameters(), strict=True):
            assert parameter.grad is not None and reference.grad is not None
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
        for name in ("pi", "pi_avg", "critic", "wm"):
            for value, reference in zip(
                getattr(graph, name).state_dict().values(),
                getattr(eager, name).state_dict().values(),
                strict=True,
            ):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
        for name in ("_v_mean", "_v_std"):
            torch.testing.assert_close(getattr(graph, name), getattr(eager, name), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_policy_update_graph_draws_fresh_noise_with_eager_statistics(tmp_path) -> None:
    eager, graph = _policy_pair(tmp_path, noise=True)
    firsts = []
    for _ in range(8):
        for learner in (eager, graph):
            learner.policy_update()
        if graph._pi_graph.replaying:
            firsts.append(graph.mdp.first.clone())
    assert len(firsts) == 5
    # Never frozen ...
    assert any(not torch.equal(first, firsts[0]) for first in firsts[1:])
    # ... and the same distribution as eager: uniform on [-1, 1] has std 1 / sqrt(3).
    # Both learners run their own stream, so the two estimates are independent draws.
    eager_mean, eager_std = _noise_statistics(eager)
    graph_mean, graph_std = _noise_statistics(graph)
    assert abs(eager_mean) < 0.02 and abs(graph_mean) < 0.02
    assert abs(eager_std - 0.57735) < 0.01 and abs(graph_std - 0.57735) < 0.01
    assert abs(graph_mean - eager_mean) < 0.025
    assert abs(graph_std - eager_std) < 0.012


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_policy_update_graph_keeps_the_off_graph_rng_sequence(tmp_path) -> None:
    eager, graph = _policy_pair(tmp_path, noise=True)
    compared = 0
    for _ in range(8):
        # The same start: whatever the noisy update consumes inside the graph, the
        # draws after it (sampling, resets) must go on where eager would have.
        state = torch.cuda.get_rng_state()
        eager.policy_update()
        after_eager = torch.rand(16, device="cuda")
        torch.cuda.set_rng_state(state)
        graph.policy_update()
        after_graph = torch.rand(16, device="cuda")
        if graph._pi_graph.replaying:
            compared += 1
            # The first in-graph draw is the one eager makes from the same state, so the
            # replayed update is the eager update, noise included.
            torch.testing.assert_close(graph.mdp.first, eager.mdp.first, rtol=0, atol=0)
            torch.testing.assert_close(after_graph, after_eager, rtol=0, atol=0)
    assert compared == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_policy_update_graph_outputs_outlive_the_next_replay(tmp_path) -> None:
    _, graph = _policy_pair(tmp_path, noise=False)
    for _ in range(4):
        graph.policy_update()
    assert graph._pi_graph.replaying
    addresses = [parameter.grad.data_ptr() for parameter in graph.pi.parameters()]

    held = graph.policy_update()
    snapshot = [metric.clone() for metric in held]
    graph.policy_update()
    graph.policy_update()
    for metric, saved in zip(held, snapshot, strict=True):
        torch.testing.assert_close(metric, saved, rtol=0, atol=0)
    # The eager clip and optimizer keep reading the gradients the graph writes to.
    assert addresses == [parameter.grad.data_ptr() for parameter in graph.pi.parameters()]
