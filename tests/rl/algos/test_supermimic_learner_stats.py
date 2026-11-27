"""Device-side collect statistics (``cfg.stats_on_device``) match the synced path.

The device-side path must reproduce the ``train_log.csv`` definitions exactly (they
only differ in where the sums are taken), without any host read inside ``collect``.
Both paths are driven through the real ``collect`` / ``train`` code by a scripted
fake env whose episode ends are random, mostly absent, and absent for whole periods.
"""

from __future__ import annotations

import csv
from dataclasses import fields, replace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from test_supermimic_learner import ExamFakeEnv, FakeEnv, FakeMdp, _config, _exam_config

from unisupermimic.rl.algos.supermimic import learner as learner_module
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY

_CSV_STATS = ("sim_rew", "ep_len", "ep_ret_norm", "term_frac")
_STAT_KEYS = _CSV_STATS + ("exam_survival",)
_WARMUP_STEPS = 2  # max(hist, wm_window) of ``_config()``


class PatternMdp(FakeMdp):
    """FakeMdp whose danger depends on the row and the frame: survival is non-trivial."""

    def soft_termination(self, state, frames: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(frames.numel(), device=frames.device)
        return ((rows * 7 + frames) % 5 < 3).float()


class ScriptedEnv(FakeEnv):
    """FakeEnv whose episode statistics are the next scripted step.

    ``step`` keeps the env's own dynamics and resets, but replaces the reward,
    ``done`` / ``term`` and the masked episode statistics of the transition with the
    scripted tensors (full-width, zero on non-done rows, like the real env). A
    ``None`` step yields a transition without episode statistics (the env's eval mode).
    """

    def __init__(self, steps: list[dict[str, torch.Tensor] | None], **kwargs) -> None:
        super().__init__(**kwargs)
        self.steps = steps
        self.cursor = 0

    def step(self, actions: torch.Tensor):
        state = super().step(actions)
        step = self.steps[self.cursor]
        self.cursor += 1
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        if step is None:
            transition.pop("ep_len", None)
            transition.pop("ep_ret_norm", None)
            return state
        for key in ("r", "done", "term", "ep_len", "ep_ret_norm"):
            transition[key] = step[key].clone()
        return state


class ScriptedExamEnv(ScriptedEnv, ExamFakeEnv):
    """ExamFakeEnv with scripted statistics (the last rows are the exam worlds)."""


def _script(num_envs: int, plan: list[str], seed: int = 0) -> list[dict[str, torch.Tensor] | None]:
    """One scripted step per plan entry.

    ``none`` never ends an episode, ``rare`` / ``sparse`` end a few (many steps end
    none), ``single`` ends exactly one, ``exam`` ends only the last row (an exam
    world once exams are on: counted in the full-width statistics, not in ``done``),
    ``dense`` ends many, ``stale`` ends none but carries non-zero episode statistics
    on some rows (they must be ignored), and ``missing`` carries no episode
    statistics at all.
    """
    generator = torch.Generator().manual_seed(seed)
    probability = {"none": 0.0, "rare": 0.01, "sparse": 0.04, "single": 0.0, "dense": 0.4}
    probability.update({"exam": 0.0, "stale": 0.0})
    steps: list[dict[str, torch.Tensor] | None] = []
    for kind in plan:
        if kind == "missing":
            steps.append(None)
            continue
        done = torch.rand(num_envs, generator=generator) < probability[kind]
        if kind == "single":
            done = torch.zeros(num_envs, dtype=torch.bool)
            done[int(torch.randint(num_envs, (1,), generator=generator))] = True
        if kind == "exam":
            done = torch.zeros(num_envs, dtype=torch.bool)
            done[-1] = True
        term = done & (torch.rand(num_envs, generator=generator) < 0.8)
        counted = done
        if kind == "stale":
            counted = torch.rand(num_envs, generator=generator) < 0.2
        steps.append(
            {
                "r": torch.rand(num_envs, generator=generator) * 0.1,
                "done": done,
                "term": term,
                "ep_len": (1.0 + 150.0 * torch.rand(num_envs, generator=generator)) * counted,
                "ep_ret_norm": (1.4 * torch.rand(num_envs, generator=generator) - 0.5) * counted,
            }
        )
    return steps


def _synced_row(learner: SuperMimicLearner) -> dict[str, float | None]:
    """The CSV definitions evaluated on the synced path's lists since its last reset."""
    stats = learner.stats
    total_done = sum(float(value) for value in stats["done"])
    return {
        "sim_rew": SuperMimicLearner._mean(stats["r"]),
        "ep_len": SuperMimicLearner._mean(stats["ep_len"]),
        "ep_ret_norm": SuperMimicLearner._mean(stats["ep_ret"]),
        "term_frac": float(sum(float(value) for value in stats["term"]) / max(total_done, 1)),
        "exam_survival": SuperMimicLearner._mean(stats["exam_surv"])
        if stats["exam_surv"]
        else None,
    }


def _relative_error(reference: float | None, value: float | None) -> float:
    assert (reference is None) == (value is None), (reference, value)
    if reference is None or value is None:
        return 0.0
    return abs(value - reference) / max(abs(reference), abs(value), 1e-300)


def _assert_rows_match(reference: dict[str, float | None], value: dict[str, float]) -> float:
    """Both rows carry the same keys; return the largest relative deviation."""
    expected = _STAT_KEYS if reference["exam_survival"] is not None else _CSV_STATS
    assert set(value) == set(expected)
    return max(_relative_error(reference[key], value[key]) for key in value)


def _learner_pair(env_factory, config: SuperMimicConfig, log_dir, device=None):
    """Two learners on identical scripted steps: synced path and device-side path."""
    pair = []
    for on_device in (False, True):
        learner = SuperMimicLearner(
            env_factory(),
            replace(config, stats_on_device=on_device),
            log_dir / f"on_device_{on_device}",
            device=device,
        )
        pair.append(learner)
    return pair


# Three steps per period. The second period counts no episode end (yet carries
# stale statistics on two steps), the fourth carries no episode statistics at all.
_PERIODS = (
    ("sparse", "none", "rare"),
    ("none", "stale", "stale"),
    ("single", "dense", "sparse"),
    ("missing", "missing", "missing"),
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_statistics_match_the_synced_path_every_period(tmp_path, device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    synced, on_device = _learner_pair(
        lambda: ScriptedEnv(_script(32, sum(_PERIODS, ()), seed=3), num_envs=32),
        _config(),
        tmp_path,
        device=device,
    )
    assert on_device._dev_stats is not None and synced._dev_stats is None

    worst = 0.0
    for index, kinds in enumerate(_PERIODS):
        for _ in kinds:
            synced.collect(1)
            on_device.collect(1)
        # The device path kept nothing on the host: its lists stay empty.
        assert all(on_device.stats[key] == [] for key in on_device.stats)
        reference, value = _synced_row(synced), on_device._read_dev_stats()
        assert on_device._dev_stats.tolist() == [0.0] * len(learner_module._DEV_STAT_FIELDS)
        synced.stats = learner_module._empty_stats()
        if index == 1:
            assert reference["ep_len"] == 0.0 and reference["ep_ret_norm"] == 0.0
            assert reference["term_frac"] == 0.0 and reference["sim_rew"] > 0.0
        if index == 3:
            assert reference["ep_len"] == 0.0 and reference["term_frac"] == 0.0
        worst = max(worst, _assert_rows_match(reference, value))
        print(f"period {index} max relative deviation so far: {worst:.3e}")
    # Only the float64 summation order can differ; the observed deviation is 0.
    assert worst <= 1e-12


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_statistics_match_the_synced_path_in_exam_mode(tmp_path, device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # Before activation one step; then periods of 2, 3 and 1 steps. ``exam`` steps
    # end an exam world only: no counted end, but a non-zero full-width sum.
    steps = _script(8, ("sparse", "exam", "single", "none", "exam", "rare", "exam"), seed=5)

    def make_env() -> ScriptedExamEnv:
        env = ScriptedExamEnv(steps, num_envs=8)
        env.mdp = PatternMdp()
        return env

    # Exams start at iteration 1: the warm-up steps run without exam worlds.
    config = replace(_exam_config(iterations=2), exam_margin=False)
    synced, on_device = _learner_pair(make_env, config, tmp_path, device=device)
    for learner in (synced, on_device):
        learner.collect(1)  # before activation: no exam survival is collected
        learner._advance_exam_schedule(1)
        assert learner._exam_rows == 4

    worst = 0.0
    for index, length in enumerate((2, 3, 1)):
        for _ in range(length):
            synced.collect(1)
            on_device.collect(1)
        reference, value = _synced_row(synced), on_device._read_dev_stats()
        synced.stats = learner_module._empty_stats()
        assert reference["exam_survival"] is not None
        if index == 2:
            # An exam-only end: no counted episode, so the episode statistics stay 0.
            assert reference["ep_len"] == 0.0 and reference["term_frac"] == 0.0
        worst = max(worst, _assert_rows_match(reference, value))
        print(f"exam period {index} max relative deviation so far: {worst:.3e}")
    assert worst <= 1e-12


# Two warm-up steps, then one collect step per iteration; log_every = 2. The second
# period (iterations 2-3) counts no episode end (stale statistics only).
_TRAIN_PLAN = ("sparse", "none", "rare", "dense", "stale", "stale", "single", "sparse")
# Same, with episode ends on an exam world only (exams start at iteration 1).
_EXAM_TRAIN_PLAN = ("sparse", "none", "exam", "dense", "stale", "exam", "single", "sparse")


def _train_rows(tmp_path, config: SuperMimicConfig, env_factory) -> tuple[list, dict]:
    rows = []
    summaries = {}
    for on_device in (False, True):
        learner = SuperMimicLearner(
            env_factory(), replace(config, stats_on_device=on_device), tmp_path / f"{on_device}"
        )
        summaries[on_device] = learner.train()
        with (learner.log_dir / "train_log.csv").open(encoding="utf-8", newline="") as stream:
            rows.append(list(csv.DictReader(stream)))
    return rows, summaries


def test_train_log_csv_rows_match_the_synced_path(tmp_path) -> None:
    config = replace(_config(), iterations=6, log_every=2, ckpt_every=1000)
    rows, summaries = _train_rows(
        tmp_path,
        config,
        lambda: ScriptedEnv(_script(32, _TRAIN_PLAN, seed=7), num_envs=32),
    )
    (synced, on_device) = rows
    assert [row["iteration"] for row in synced] == ["2", "4", "6"]
    assert [row["iteration"] for row in on_device] == ["2", "4", "6"]
    # Rows are written with six significant digits: identical strings mean the two
    # paths agreed to better than the CSV resolution on every period.
    for reference, value in zip(synced, on_device):
        for key in ("sim_steps",) + _CSV_STATS:
            assert value[key] == reference[key], (key, reference[key], value[key])
    assert synced[1]["ep_len"] == "0" and synced[1]["term_frac"] == "0"
    assert float(synced[0]["ep_len"]) > 0 and float(synced[2]["ep_len"]) > 0
    # The returned summary is the last CSV row on both paths.
    for key in _CSV_STATS:
        assert summaries[True][key] == pytest.approx(summaries[False][key], rel=1e-12)
    assert summaries[True]["completed"] is True and summaries[True]["iteration"] == 6


def test_train_log_csv_rows_match_the_synced_path_in_exam_mode(tmp_path) -> None:
    config = replace(_exam_config(iterations=6, log_every=2, ckpt_every=1000), exam_margin=False)

    def make_env() -> ScriptedExamEnv:
        env = ScriptedExamEnv(_script(8, _EXAM_TRAIN_PLAN, seed=11), num_envs=8)
        env.mdp = PatternMdp()
        return env

    (synced, on_device) = _train_rows(tmp_path, config, make_env)[0]
    assert float(synced[1]["ep_len"]) == 0.0
    for reference, value in zip(synced, on_device):
        for key in _CSV_STATS:
            assert value[key] == reference[key], (key, reference[key], value[key])


def _scalars(log_dir, tag: str) -> list[tuple[int, float]]:
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        return []
    return [(event.step, event.value) for event in accumulator.Scalars(tag)]


def test_device_statistics_log_one_point_per_period(tmp_path) -> None:
    config = replace(
        _config(),
        iterations=4,
        log_every=2,
        ckpt_every=1000,
        log_backend="tensorboard",
    )
    # Every iteration ends episodes, so the synced path logs its collect tags
    # on all four iterations.
    plan = ("dense", "none", "single", "dense", "single", "dense")
    (synced, on_device) = _train_rows(
        tmp_path,
        config,
        lambda: ScriptedEnv(_script(32, plan, seed=13), num_envs=32),
    )[0]
    collect_tags = (
        "Reward/sim_step_norm",
        "Episode/length",
        "Episode/return_norm",
        "Episode/terminated_frac",
    )
    for tag in collect_tags:
        # Synced: one point per iteration. Device-side: one point per log period.
        assert [step for step, _ in _scalars(tmp_path / "False", tag)] == [1, 2, 3, 4]
        assert [step for step, _ in _scalars(tmp_path / "True", tag)] == [2, 4]
    for tag in ("Loss/world_model", "Train/grad_norm", "Perf/iteration_time"):
        assert [step for step, _ in _scalars(tmp_path / "False", tag)] == [1, 2, 3, 4]
        assert [step for step, _ in _scalars(tmp_path / "True", tag)] == [1, 2, 3, 4]
    # Exams are off: the exam tag is not written on either path.
    assert _scalars(tmp_path / "True", "Exam/survival") == []
    # A period's point carries that period's CSV value (six significant digits).
    for tag, column in zip(collect_tags, _CSV_STATS):
        points = dict(_scalars(tmp_path / "True", tag))
        for row in on_device:
            assert points[int(row["iteration"])] == pytest.approx(
                float(row[column]), rel=5e-6, abs=1e-9
            )


def test_exam_survival_is_logged_once_per_period_on_device(tmp_path) -> None:
    config = replace(
        _exam_config(iterations=4, log_every=2, ckpt_every=1000, log_backend="tensorboard"),
        exam_margin=False,
    )

    def make_env() -> ScriptedExamEnv:
        env = ScriptedExamEnv(_script(8, _TRAIN_PLAN[:6], seed=17), num_envs=8)
        env.mdp = PatternMdp()
        return env

    (synced, on_device) = _train_rows(tmp_path, config, make_env)[0]
    # Exams start at iteration 1 (iteration 0 logs none); log points land on 2 and 4.
    assert [step for step, _ in _scalars(tmp_path / "False", "Exam/survival")] == [2, 3, 4]
    assert [step for step, _ in _scalars(tmp_path / "True", "Exam/survival")] == [2, 4]


def test_collect_never_reads_the_host_when_statistics_stay_on_device(tmp_path, monkeypatch) -> None:
    steps = _script(8, ("single", "dense", "none", "sparse", "rare"), seed=19)

    def no_host_read(name):
        def read(self, *args, **kwargs):
            raise AssertionError(f"host read ({name}) inside collect")

        return read

    for name in ("item", "tolist", "cpu", "__float__", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, no_host_read(name))
    synced = SuperMimicLearner(ScriptedEnv(steps, num_envs=8), _config(), tmp_path / "synced")
    with pytest.raises(AssertionError, match="host read"):
        synced.collect(2)

    on_device = SuperMimicLearner(
        ScriptedEnv(steps, num_envs=8),
        replace(_config(), stats_on_device=True),
        tmp_path / "on_device",
    )
    on_device.collect(len(steps))
    assert on_device.sim_steps == 8 * len(steps)
    assert all(on_device.stats[key] == [] for key in on_device.stats)
    monkeypatch.undo()
    # Outside collect, the period's row is read back once and the sums restart.
    sums = on_device._dev_stats.tolist()
    assert sums[learner_module._DEV_STAT_FIELDS.index("steps")] == len(steps)
    row = on_device._read_dev_stats()
    assert sums[0] == 5 and row["ep_len"] > 0.0 and 0.0 < row["term_frac"] <= 1.0
    assert on_device._dev_stats.tolist() == [0.0] * len(sums)


def test_stats_on_device_is_auto_by_default(tmp_path) -> None:
    assert SuperMimicConfig().stats_on_device is None
    assert "stats_on_device" in {field.name for field in fields(SuperMimicConfig)}
    learner = SuperMimicLearner(FakeEnv(num_envs=8), _config(), tmp_path)
    # ``auto`` keeps the synced path anywhere but NVIDIA CUDA (this learner is on CPU).
    assert learner.cfg.stats_on_device is None
    assert learner._stats_on_device is False
    assert learner._dev_stats is None and learner._dev_unit is None
    with pytest.raises(RuntimeError, match="cfg.stats_on_device"):
        learner._accumulate_stats({})
    with pytest.raises(RuntimeError, match="cfg.stats_on_device"):
        learner._read_dev_stats()
    learner.collect(3)  # the synced path still fills the lists
    assert len(learner.stats["r"]) == 3 and len(learner.stats["done"]) == 1


def test_device_statistics_restart_each_training_run(tmp_path) -> None:
    # 2 warm-up + iteration 0, then 2 steps outside train(), then iteration 1.
    steps = _script(8, ("dense", "dense", "dense", "single", "dense", "sparse"), seed=23)
    learner = SuperMimicLearner(
        ScriptedEnv(steps, num_envs=8),
        replace(_config(), iterations=1, stats_on_device=True),
        tmp_path,
    )
    learner.train()
    assert learner._dev_stats.tolist() == [0.0] * len(learner_module._DEV_STAT_FIELDS)
    learner.collect(2)  # steps collected after a stopped run must not leak into the next
    assert learner._dev_stats.tolist()[0] == 2.0
    learner.cfg = replace(learner.cfg, iterations=2)
    learner.train()
    assert learner._dev_stats.tolist() == [0.0] * len(learner_module._DEV_STAT_FIELDS)
    with (tmp_path / "train_log.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["iteration"] for row in rows] == ["1", "2"]
    # The second row covers exactly iteration 1's single collect step.
    expected = float(steps[5]["r"].mean() / learner.mdp.max_step_reward)
    assert float(rows[1]["sim_rew"]) == pytest.approx(expected, rel=5e-6)
