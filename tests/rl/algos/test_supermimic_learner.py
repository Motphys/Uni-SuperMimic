"""CPU training-loop tests for the protocol-driven SuperMimic learner."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.rl.algos.supermimic.optim import MuonAdam
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    State,
)


class FakeMdp:
    num_joints = 2
    context_dim = 2
    obs_dim = 6
    goal_frame_dim = 3
    max_step_reward = 0.1

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (state[0][..., :1], previous_action, context, frames.float().unsqueeze(-1)), -1
        )

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def goal(self, state: State, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        del state
        return frames.float().unsqueeze(-1).expand(-1, num_frames * self.goal_frame_dim)

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        del frames
        tracking = self.max_step_reward * torch.exp(-(state[0] - 0.1).square().mean(-1))
        return tracking - 0.001 * (action - previous_action).square().mean(-1)

    def soft_termination(self, state: State, frames: torch.Tensor) -> torch.Tensor:
        del state
        return torch.zeros_like(frames, dtype=torch.float32)

    def critic_features(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((state[0], state[4], normalized_context, frames.float().unsqueeze(-1)), -1)

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(context)

    def valid_imagination_start(self, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        del horizon
        return torch.ones_like(frames, dtype=torch.bool)

    def to(self, device: str | torch.device) -> FakeMdp:
        self.moved_to = torch.device(device)
        return self


@dataclass
class FakeState:
    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict
    final_observation: None = None


class FakeEnv:
    def __init__(self, num_envs: int = 2):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.mdp = FakeMdp()
        self.state: FakeState | None = None
        self.initial_state = torch.zeros(num_envs, 13 + 2 * self.mdp.num_joints)
        self.initial_state[:, 3] = 1.0
        self.initial_frames = torch.arange(num_envs, dtype=torch.long)
        self.context = torch.full((num_envs, self.mdp.context_dim), 0.2)

    @property
    def supermimic_mdp(self) -> FakeMdp:
        return self.mdp

    def init_state(self) -> FakeState:
        info = {
            "s": self.initial_state.clone(),
            "f": self.initial_frames.clone(),
            "xi": self.context.clone(),
            "a_prev": torch.zeros(self.num_envs, self.mdp.num_joints),
            "ep_step": torch.zeros(self.num_envs, dtype=torch.int32),
        }
        self.state = FakeState(
            obs={},
            reward=torch.zeros(self.num_envs),
            terminated=torch.zeros(self.num_envs, dtype=torch.bool),
            truncated=torch.zeros(self.num_envs, dtype=torch.bool),
            info={SUPERMIMIC_STATE_KEY: info},
        )
        return self.state

    def step(self, actions: torch.Tensor) -> FakeState:
        assert self.state is not None
        current = self.state.info[SUPERMIMIC_STATE_KEY]
        next_state = current["s"].clone()
        next_state[:, :3] += 0.001
        next_state[:, 13 : 13 + self.mdp.num_joints] += 0.01 * actions
        next_state[:, 13 + self.mdp.num_joints :] += 0.001 * actions
        next_frames = current["f"] + 1
        episode_steps = current["ep_step"] + 1
        done = episode_steps >= 3
        reward = self.mdp.reward(
            (
                next_state[:, 0:3],
                torch.eye(3).expand(self.num_envs, 3, 3),
                next_state[:, 7:10],
                next_state[:, 10:13],
                next_state[:, 13:15],
                next_state[:, 15:17],
            ),
            next_frames,
            actions,
            current["a_prev"],
        )
        transition = {
            "s": current["s"].clone(),
            "s1": next_state.clone(),
            "a": actions.clone(),
            "a_prev": current["a_prev"].clone(),
            "xi": current["xi"].clone(),
            "r": reward,
            "f": current["f"].clone(),
            "ep_step": current["ep_step"].clone(),
            "done": done,
            "term": torch.zeros_like(done),
        }
        if bool(done.any()):
            transition["ep_len"] = episode_steps[done].float()
            transition["ep_ret_norm"] = reward[done] / (
                episode_steps[done].float() * self.mdp.max_step_reward
            )
        post_reset_state = next_state.clone()
        post_reset_state[done] = self.initial_state[done]
        post_reset_action = torch.where(done.unsqueeze(-1), torch.zeros_like(actions), actions)
        post_reset_steps = torch.where(done, torch.zeros_like(episode_steps), episode_steps)
        info = {
            "s": post_reset_state,
            "f": next_frames,
            "xi": current["xi"].clone(),
            "a_prev": post_reset_action,
            "ep_step": post_reset_steps,
        }
        self.state = FakeState(
            obs={},
            reward=reward,
            terminated=done,
            truncated=torch.zeros_like(done),
            info={
                SUPERMIMIC_STATE_KEY: info,
                SUPERMIMIC_TRANSITION_KEY: transition,
            },
        )
        return self.state


def _config() -> SuperMimicConfig:
    return SuperMimicConfig(
        iterations=1,
        buffer_steps=16,
        utd=1.0,
        horizon=2,
        goal_frames=2,
        hist=2,
        wm_window=2,
        wm_batches=1,
        wm_batch=4,
        pi_batch=4,
        lr=1e-3,
        critic_epochs=1,
        critic_minibatches=2,
        ckpt_every=1,
        log_every=1,
        log_backend="no_print",
        amp="none",
        compile=False,
        test_hidden_dims=(8, 8),
    )


class ExamFakeEnv(FakeEnv):
    """FakeEnv with the exam activation hook the learner drives."""

    def __init__(self, num_envs: int = 8):
        super().__init__(num_envs)
        self.exam_activations: list[int] = []

    def enable_supermimic_exams(self, num_rows: int) -> None:
        assert self.state is not None
        self.exam_activations.append(num_rows)
        # Value-only activation: the last rows restart on a clip first frame
        # (frame 0 here) with a fresh episode.
        current = self.state.info[SUPERMIMIC_STATE_KEY]
        current["f"][self.num_envs - num_rows :] = 0
        current["ep_step"][self.num_envs - num_rows :] = 0


class MarginFakeEnv(ExamFakeEnv):
    """ExamFakeEnv with the exams-v2 margin hooks the learner drives."""

    def __init__(self, num_envs: int = 8):
        super().__init__(num_envs)
        self.margin_config: tuple[float, float] | None = None
        self.exam_refreshes: list[int] = []

    def configure_supermimic_exam_margin(self, *, frontier_frac: float, fail_ema: float) -> None:
        self.margin_config = (frontier_frac, fail_ema)

    def refresh_supermimic_exam_clips(self, num_rows: int) -> torch.Tensor:
        self.exam_refreshes.append(num_rows)
        return torch.tensor(0.42)


def _exam_config(**overrides) -> SuperMimicConfig:
    values = dict(
        iterations=3,
        pi_ema=0.995,
        exam_start_iter=1,
        exam_frac=0.5,
        exam_share_ramp=0.9,
        exam_ramp_iters=4,
    )
    values.update(overrides)
    return replace(_config(), **values)


def test_tf32_only_enabled_for_fp32_training(tmp_path) -> None:
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        SuperMimicLearner(FakeEnv(), _config(), tmp_path / "fp32")
        assert torch.backends.cuda.matmul.allow_tf32 is True
        SuperMimicLearner(FakeEnv(), replace(_config(), amp="bf16"), tmp_path / "bf16")
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_collect_and_updates_run_on_cpu(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)

    optimizers = (learner.opt_wm, learner.opt_pi, learner.opt_v)
    assert all(optimizer.defaults["fused"] is False for optimizer in optimizers)

    learner.collect(3)
    wm_loss, wm_error = learner.wm_update()
    policy_loss, value_loss, imagined_reward, grad_norm, survival = learner.policy_update()

    assert learner.buffer.size == 3
    assert learner.sim_steps == 6
    assert learner.obs_hist.shape == (2, 2, 6)
    for metric in (
        wm_loss,
        wm_error,
        policy_loss,
        value_loss,
        imagined_reward,
        grad_norm,
        survival,
    ):
        assert metric.ndim == 0
        assert torch.isfinite(metric)
    assert learner._v_init
    assert learner._v_std >= 0.01


def test_structured_world_model_mode_is_wired_into_updates(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), replace(_config(), wm_structured=True), tmp_path)

    assert learner.wm.structured is True
    assert learner.cfg.wm_structured is True

    learner.collect(3)
    _, _ = learner.wm_update()

    assert learner.wm.out_norm.count > 0
    assert learner.wm.inc_norm.count == learner.wm.out_norm.count


def test_invalid_optimizer_name_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="pi_opt must be 'adam' or 'muon'"):
        SuperMimicLearner(FakeEnv(), replace(_config(), pi_opt="sgd"), tmp_path)

    with pytest.raises(ValueError, match="wm_opt must be 'adam' or 'muon'"):
        SuperMimicLearner(FakeEnv(), replace(_config(), wm_opt="sgd"), tmp_path)


def test_muon_updates_hidden_matrices_and_runs_on_cpu(tmp_path) -> None:
    cfg = replace(
        _config(),
        pi_opt="muon",
        wm_opt="muon",
        test_hidden_dims=(8, 8),
    )
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)

    assert isinstance(learner.opt_wm, MuonAdam)
    assert isinstance(learner.opt_pi, MuonAdam)
    assert not isinstance(learner.opt_v, MuonAdam)

    world_model_muon = {
        id(parameter) for parameter in learner.opt_wm.muon.param_groups[0]["params"]
    }
    policy_muon = {
        id(parameter) for group in learner.opt_pi.muon.param_groups for parameter in group["params"]
    }
    world_model_hidden = learner.wm.net[3]
    policy_encoder_hidden = learner.pi.enc[2]
    policy_decoder_hidden = learner.pi.dec[2]
    assert isinstance(world_model_hidden, torch.nn.Linear)
    assert isinstance(policy_encoder_hidden, torch.nn.Linear)
    assert isinstance(policy_decoder_hidden, torch.nn.Linear)
    assert id(world_model_hidden.weight) in world_model_muon
    assert id(policy_encoder_hidden.weight) in policy_muon
    assert id(policy_decoder_hidden.weight) in policy_muon
    assert id(learner.wm.net[0].weight) not in world_model_muon
    assert id(learner.wm.net[-1].weight) not in world_model_muon
    assert id(learner.pi.enc[0].weight) not in policy_muon
    assert id(learner.pi.dec[-1].weight) not in policy_muon

    learner.collect(3)
    wm_loss, wm_error = learner.wm_update()
    policy_loss, value_loss, imagined_reward, grad_norm, survival = learner.policy_update()
    for metric in (
        wm_loss,
        wm_error,
        policy_loss,
        value_loss,
        imagined_reward,
        grad_norm,
        survival,
    ):
        assert metric.ndim == 0
        assert torch.isfinite(metric)


def test_muon_optimizer_checkpoint_roundtrip(tmp_path) -> None:
    cfg = replace(
        _config(),
        pi_opt="muon",
        wm_opt="muon",
        test_hidden_dims=(8, 8),
        iterations=1,
    )
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    learner.collect(3)
    learner.wm_update()
    learner.policy_update()
    learner.save()

    resumed = SuperMimicLearner(FakeEnv(), replace(cfg, iterations=2), tmp_path / "resumed")
    resumed.load(tmp_path / "last.pt")

    assert isinstance(resumed.opt_wm, MuonAdam)
    assert isinstance(resumed.opt_pi, MuonAdam)
    assert len(resumed.opt_wm.muon.state) == len(learner.opt_wm.muon.state)
    assert len(resumed.opt_pi.muon.state) == len(learner.opt_pi.muon.state)
    for name, value in learner.wm.named_parameters():
        torch.testing.assert_close(resumed.wm.state_dict()[name], value)
    for name, value in learner.pi.state_dict().items():
        if name in dict(learner.pi.named_parameters()):
            torch.testing.assert_close(resumed.pi.state_dict()[name], value)
    for original, restored in zip(learner.wm.parameters(), resumed.wm.parameters(), strict=True):
        if original in learner.opt_wm.muon.state:
            torch.testing.assert_close(
                resumed.opt_wm.muon.state[restored]["momentum_buffer"],
                learner.opt_wm.muon.state[original]["momentum_buffer"],
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_muon_updates_run_with_fused_adam_on_cuda(tmp_path) -> None:
    cfg = replace(
        _config(),
        pi_opt="muon",
        wm_opt="muon",
        test_hidden_dims=(8, 8),
    )
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path, device="cuda")

    assert learner.opt_wm.adam.defaults["fused"] is True
    assert learner.opt_pi.adam.defaults["fused"] is True
    learner.collect(3)
    wm_loss, _ = learner.wm_update()
    policy_loss, value_loss, _, _, _ = learner.policy_update()

    assert torch.isfinite(wm_loss)
    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)


def test_no_sync_and_host_sampling_return_full_batch(tmp_path) -> None:
    cfg = _config()
    for no_sync in (False, True):
        learner = SuperMimicLearner(
            FakeEnv(), replace(cfg, no_sync=no_sync), tmp_path / f"no_sync_{no_sync}"
        )
        learner.collect(4)
        starts, histories = learner._sample_valid_starts(cfg.pi_batch)
        for value in list(starts.values()) + list(histories.values()):
            assert value.shape[0] == cfg.pi_batch


def test_pi_ema_disabled_by_default(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    assert learner.pi_avg is None


def test_ema_policy_averages_weights(tmp_path) -> None:
    decay = 0.995
    learner = SuperMimicLearner(FakeEnv(), replace(_config(), pi_ema=decay), tmp_path)
    learner.collect(3)
    learner.policy_update()  # first call initializes the average to the live weights
    previous = {
        name: parameter.detach().clone() for name, parameter in learner.pi.named_parameters()
    }

    learner.policy_update()

    for name, parameter in learner.pi.named_parameters():
        expected = decay * previous[name] + (1 - decay) * parameter.detach()
        torch.testing.assert_close(learner.pi_avg.module.state_dict()[name], expected)


def test_ema_policy_checkpoint_and_resume(tmp_path) -> None:
    cfg = replace(_config(), pi_ema=0.995)
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    learner.collect(3)
    learner.policy_update()

    learner.save()
    checkpoint = torch.load(tmp_path / "last.pt", weights_only=True)
    assert set(checkpoint) == {
        "wm",
        "pi",
        "pi_live",
        "critic",
        "opt_wm",
        "opt_pi",
        "opt_v",
        "iteration",
        "wall_time",
        "sim_steps",
        "args",
        "saved_at",
    }
    live = learner.pi.state_dict()
    averaged = learner.pi_avg.module.state_dict()
    for name, value in live.items():
        torch.testing.assert_close(checkpoint["pi_live"][name], value)
    for name, value in averaged.items():
        torch.testing.assert_close(checkpoint["pi"][name], value)

    resumed = SuperMimicLearner(FakeEnv(), replace(cfg, iterations=1), tmp_path / "resumed")
    resumed.load(tmp_path / "last.pt")
    for name, value in live.items():
        torch.testing.assert_close(resumed.pi.state_dict()[name], value)
    for name, value in averaged.items():
        torch.testing.assert_close(resumed.pi_avg.module.state_dict()[name], value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_collect_and_updates_run_with_fused_adam_on_cuda(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path, device="cuda")

    assert learner.device.type == "cuda"
    optimizers = (learner.opt_wm, learner.opt_pi, learner.opt_v)
    assert all(optimizer.defaults["fused"] is True for optimizer in optimizers)

    learner.collect(3)
    wm_loss, _ = learner.wm_update()
    policy_loss, value_loss, _, _, _ = learner.policy_update()

    assert torch.isfinite(wm_loss)
    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)


def test_matching_explicit_device_disables_bridging(tmp_path) -> None:
    env = FakeEnv()
    learner = SuperMimicLearner(env, _config(), tmp_path, device="cpu")

    assert learner._bridged is False
    assert learner.mdp is env.mdp


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_bridged_learner_moves_training_to_mps(tmp_path) -> None:
    env = FakeEnv()
    learner = SuperMimicLearner(env, _config(), tmp_path, device="mps")

    assert learner._bridged is True
    assert learner.device == torch.device("mps")
    assert learner.mdp.moved_to == torch.device("mps")

    learner.collect(3)
    wm_loss, _ = learner.wm_update()
    policy_loss, value_loss, imagined_reward, grad_norm, survival = learner.policy_update()

    assert learner.buffer.data["s"].device.type == "mps"
    assert learner.obs_hist.device.type == "mps"
    for metric in (wm_loss, policy_loss, value_loss, imagined_reward, grad_norm, survival):
        assert metric.ndim == 0
        assert torch.isfinite(metric)
    # The env stays on its own device across the bridge.
    assert env.state is not None
    current = env.state.info[SUPERMIMIC_STATE_KEY]
    assert current["s"].device.type == "cpu"


def test_checkpoint_roundtrip_and_resume(tmp_path) -> None:
    cfg = _config()
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    final = learner.train()

    assert final["completed"] is True
    assert final["iteration"] == 1
    checkpoint_path = tmp_path / "last.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert set(checkpoint) == {
        "wm",
        "pi",
        "critic",
        "opt_wm",
        "opt_pi",
        "opt_v",
        "iteration",
        "wall_time",
        "sim_steps",
        "args",
        "saved_at",
    }
    small = torch.load(tmp_path / "ckpt_000001.pt", weights_only=True)
    assert set(small) == {"wm", "pi", "iteration", "wall_time", "sim_steps", "args", "saved_at"}
    with (tmp_path / "train_log.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    assert rows[0][:3] == ["iteration", "wall_h", "sim_steps"]
    assert rows[1][0] == "1"

    resumed = SuperMimicLearner(FakeEnv(), replace(cfg, iterations=2), tmp_path / "resumed")
    with pytest.raises(ValueError, match="requires a full last.pt"):
        resumed.load(tmp_path / "ckpt_000001.pt")
    resumed.load(checkpoint_path)
    assert resumed.iteration == learner.iteration
    assert resumed.sim_steps == learner.sim_steps
    assert resumed.wall_time == checkpoint["wall_time"]
    for name in ("wm", "pi", "critic"):
        original_state = getattr(learner, name).state_dict()
        restored_state = getattr(resumed, name).state_dict()
        assert original_state.keys() == restored_state.keys()
        for key in original_state:
            torch.testing.assert_close(restored_state[key], original_state[key])

    resumed_final = resumed.train()
    assert resumed_final["completed"] is True
    assert resumed_final["iteration"] == 2
    assert resumed_final["sim_steps"] > final["sim_steps"]


def test_resume_rebuilds_required_buffer_history(tmp_path) -> None:
    cfg = _config()
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    learner.train()
    resumed = SuperMimicLearner(
        FakeEnv(),
        replace(cfg, iterations=1, wm_window=6),
        tmp_path / "resumed",
    )

    resumed.train(resume=tmp_path / "last.pt")

    assert resumed.buffer.size >= 6


def test_tensorboard_backend_logs_real_metric_names(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), replace(_config(), log_backend="tensorboard"), tmp_path)
    learner.train()

    event_files = list(tmp_path.glob("events.out.tfevents.*"))
    assert len(event_files) == 1
    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    expected = {
        "Loss/world_model",
        "Loss/world_model_err8",
        "Loss/policy",
        "Loss/value",
        "Loss/imagined_reward",
        "Loss/imagined_survival",
        "Train/grad_norm",
        "Train/learning_rate",
        "Reward/sim_step_norm",
        "Episode/length",
        "Episode/return_norm",
        "Episode/terminated_frac",
        "Perf/iteration_time",
        "Perf/collect_time",
        "Perf/learning_time",
        "Perf/it_per_s",
        "Perf/sim_steps_per_s",
    }
    assert expected <= tags


def test_no_print_backend_writes_no_events(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.train()

    assert not list(tmp_path.glob("events.out.tfevents.*"))


def test_nonfinite_actor_metric_aborts_before_checkpoint(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.policy_update = lambda: (
        torch.tensor(float("nan")),
        torch.zeros(()),
        torch.zeros(()),
        torch.zeros(()),
        torch.ones(()),
    )
    with pytest.raises(FloatingPointError, match="non-finite pi_loss"):
        learner.train()
    assert not (tmp_path / "last.pt").exists()


def test_nonfinite_world_model_metric_aborts_before_checkpoint(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.wm_update = lambda: (torch.tensor(float("nan")), torch.zeros(()))
    with pytest.raises(FloatingPointError, match="non-finite wm_loss"):
        learner.train()
    assert not (tmp_path / "last.pt").exists()


def test_early_nonfinite_world_model_batch_cannot_be_masked_by_later_batches(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), replace(_config(), wm_batches=2), tmp_path)
    calls = 0

    def wm_update() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return torch.tensor(float("nan") if calls == 1 else 0.0), torch.zeros(())

    learner.wm_update = wm_update
    # The verdicts stay on the device until the fail-fast check after the policy update,
    # so the remaining batches of the iteration do run; the AND keeps the first one's.
    with pytest.raises(FloatingPointError, match="non-finite wm_loss"):
        learner.train()
    assert calls == 2
    assert not (tmp_path / "ckpt_000001.pt").exists()
    assert not (tmp_path / "last.pt").exists()


def test_nonfinite_world_model_error_is_reported_after_the_policy_update(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.wm_update = lambda: (torch.zeros(()), torch.tensor(float("inf")))
    updates = 0
    policy_update = learner.policy_update

    def counting_update():
        nonlocal updates
        updates += 1
        return policy_update()

    learner.policy_update = counting_update
    with pytest.raises(FloatingPointError, match="non-finite wm_err8"):
        learner.train()
    assert updates == 1
    assert not (tmp_path / "last.pt").exists()


def test_exam_default_config_keeps_exams_fully_off(tmp_path) -> None:
    env = FakeEnv()
    learner = SuperMimicLearner(env, _config(), tmp_path)

    assert learner._exam_rows_planned == 0
    assert learner._exam_rows == 0
    assert learner._exam_share == 0.0
    learner._advance_exam_schedule(10**6)
    assert learner._exam_rows == 0  # a plain env without the hook is never touched

    learner.collect(3)
    learner.wm_update()
    learner.policy_update()
    starts, histories = learner._sample_valid_starts(4)
    for value in list(starts.values()) + list(histories.values()):
        assert value.shape[0] == 4
    assert learner.stats["exam_surv"] == []


def test_exam_config_requires_ema_policy_and_exam_capable_env(tmp_path) -> None:
    with pytest.raises(ValueError, match="pi_ema > 0"):
        SuperMimicLearner(ExamFakeEnv(), _exam_config(pi_ema=0.0), tmp_path)
    with pytest.raises(TypeError, match="enable_supermimic_exams"):
        SuperMimicLearner(FakeEnv(), _exam_config(), tmp_path)
    with pytest.raises(ValueError, match="exam_frac"):
        SuperMimicLearner(ExamFakeEnv(), _exam_config(exam_frac=2.0), tmp_path)


def test_exam_schedule_activates_once_and_ramps_imagination_share(tmp_path) -> None:
    env = ExamFakeEnv()
    learner = SuperMimicLearner(env, _exam_config(), tmp_path)

    learner._advance_exam_schedule(0)
    assert env.exam_activations == []
    learner._advance_exam_schedule(1)
    assert env.exam_activations == [4]  # exam_frac 0.5 of 8 envs, assigned once
    assert learner._exam_rows == 4
    assert learner._exam_share == pytest.approx(0.5)
    learner._advance_exam_schedule(3)
    assert env.exam_activations == [4]
    assert learner._exam_share == pytest.approx(0.5 + 0.5 * (0.9 - 0.5))
    learner._advance_exam_schedule(100)
    assert learner._exam_share == pytest.approx(0.9)


def test_exam_rows_act_with_the_ema_policy_and_record_survival(tmp_path) -> None:
    learner = SuperMimicLearner(ExamFakeEnv(), _exam_config(), tmp_path)
    learner._advance_exam_schedule(1)

    def constant_policy(obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del goal
        return torch.full((obs_hist.shape[0], learner.mdp.num_joints), 5.0)

    learner.pi_avg.module.forward = constant_policy
    learner.collect(1)

    row = (learner.buffer.ptr - 1) % learner.buffer.T
    actions = learner.buffer.data["a"][row]
    torch.testing.assert_close(actions[-4:], torch.full((4, learner.mdp.num_joints), 5.0))
    # Practice rows keep the live policy (not the constant EMA output).
    assert not torch.allclose(actions[:-4], torch.full_like(actions[:-4], 5.0))
    # FakeMdp reports zero danger, so every exam row survives.
    assert len(learner.stats["exam_surv"]) == 1
    assert learner.stats["exam_surv"][0] == pytest.approx(1.0)


def test_exam_done_statistics_count_practice_rows_only(tmp_path) -> None:
    learner = SuperMimicLearner(ExamFakeEnv(), _exam_config(), tmp_path)
    learner._advance_exam_schedule(1)

    learner.collect(4)  # FakeEnv episodes end at episode step 3

    # All 8 rows of the fake env finish together on the third step; only the
    # 4 practice rows enter the done statistic.
    assert len(learner.stats["done"]) == 1
    assert float(learner.stats["done"][0]) == pytest.approx(4.0)


def test_exam_starts_flow_into_valid_start_sampling(tmp_path) -> None:
    learner = SuperMimicLearner(ExamFakeEnv(), _exam_config(), tmp_path)
    learner.collect(3)
    learner._advance_exam_schedule(1)
    calls: list[tuple[int, float]] = []
    original = learner.buffer.sample_starts

    def spy(batch: int, hist: int, exam_rows: int = 0, exam_prob: float = 0.0):
        calls.append((exam_rows, exam_prob))
        return original(batch, hist, exam_rows=exam_rows, exam_prob=exam_prob)

    learner.buffer.sample_starts = spy
    starts, histories = learner._sample_valid_starts(4)

    assert starts["s"].shape[0] == 4
    assert histories["s"].shape[0] == 4
    assert calls == [(4, pytest.approx(0.5))]


def test_exam_survival_is_logged_to_tensorboard(tmp_path) -> None:
    learner = SuperMimicLearner(ExamFakeEnv(), _exam_config(log_backend="tensorboard"), tmp_path)
    learner.train()

    event_files = list(tmp_path.glob("events.out.tfevents.*"))
    assert len(event_files) == 1
    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert "Exam/survival" in tags
    events = accumulator.Scalars("Exam/survival")
    # Iteration 0 runs without exams; iterations 1 and 2 log survival.
    assert [event.step for event in events] == [2, 3]


def test_margin_mode_requires_exams_and_a_margin_capable_env(tmp_path) -> None:
    with pytest.raises(ValueError, match="exam_margin requires exam_frac"):
        SuperMimicLearner(MarginFakeEnv(), _exam_config(exam_frac=0.0, exam_margin=True), tmp_path)
    with pytest.raises(TypeError, match="configure_supermimic_exam_margin"):
        SuperMimicLearner(ExamFakeEnv(), _exam_config(exam_margin=True), tmp_path)
    with pytest.raises(ValueError, match="frontier_frac"):
        SuperMimicLearner(
            MarginFakeEnv(), _exam_config(exam_margin=True, frontier_frac=1.5), tmp_path
        )
    with pytest.raises(ValueError, match="fail_ema"):
        SuperMimicLearner(MarginFakeEnv(), _exam_config(exam_margin=True, fail_ema=1.0), tmp_path)
    with pytest.raises(ValueError, match="exam_clip_window"):
        SuperMimicLearner(
            MarginFakeEnv(), _exam_config(exam_margin=True, exam_clip_window=0), tmp_path
        )


def test_margin_mode_configures_the_env_and_refreshes_exam_clips_per_window(tmp_path) -> None:
    env = MarginFakeEnv()
    learner = SuperMimicLearner(env, _exam_config(exam_margin=True), tmp_path)

    # Configured once at construction (before any capture): the command bakes
    # the frontier ratio and failure-EMA decay into its reset layout.
    assert env.margin_config == (0.5, 0.7)
    learner._advance_exam_schedule(0)  # before exam_start_iter (1): nothing
    assert env.exam_activations == []
    assert env.exam_refreshes == []
    learner._advance_exam_schedule(1)  # activation already selects the clips
    assert env.exam_activations == [4]
    assert env.exam_refreshes == []
    learner._advance_exam_schedule(249)
    assert env.exam_refreshes == []
    learner._advance_exam_schedule(250)
    assert env.exam_refreshes == [4]
    assert learner._exam_fail_ema == pytest.approx(0.42)
    learner._advance_exam_schedule(499)
    assert env.exam_refreshes == [4]
    learner._advance_exam_schedule(500)
    assert env.exam_refreshes == [4, 4]


def test_margin_fail_ema_is_logged_to_tensorboard(tmp_path) -> None:
    learner = SuperMimicLearner(
        MarginFakeEnv(),
        _exam_config(
            log_backend="tensorboard", exam_start_iter=0, exam_clip_window=1, exam_margin=True
        ),
        tmp_path,
    )
    learner.train()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    assert "Exam/margin_fail_ema" in set(accumulator.Tags()["scalars"])
    events = accumulator.Scalars("Exam/margin_fail_ema")
    # Exams activate at iteration 0 (that activation iteration logs no
    # refresh yet); every later window-1 iteration carries the refreshed mean.
    assert [event.step for event in events] == [2, 3]
    assert all(event.value == pytest.approx(0.42) for event in events)
