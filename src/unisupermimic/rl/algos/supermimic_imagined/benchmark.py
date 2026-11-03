"""Reproducible performance benchmark for frozen-world-model imagined training.

The benchmark can use either an asset-free synthetic task or a real frozen G1
task/checkpoint.  It reports learner-update wall/CUDA time, imagined throughput,
device memory, sampled hardware utilization, and optional profiler artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from unisupermimic.rl.algos.supermimic import networks as networks_module
from unisupermimic.rl.algos.supermimic.optim import MuonAdam
from unisupermimic.rl.algos.supermimic_imagined.fake import make_fake_imagined_task
from unisupermimic.rl.algos.supermimic_imagined.learner import (
    ImaginedSuperMimicConfig,
    ImaginedSuperMimicLearner,
)


class BenchmarkError(RuntimeError):
    """Raised for an unsupported or failed benchmark configuration."""


@dataclass(frozen=True)
class BenchmarkSettings:
    """All knobs that can change measured performance."""

    task: str = "fake"
    device: str = "auto"
    batch: int = 8
    horizon: int = 2
    goal_frames: int = 2
    hist: int = 2
    z_dim: int = 8
    policy_hidden: tuple[int, ...] = (32, 32, 32)
    critic_hidden: tuple[int, ...] = (16,)
    wm_hidden: tuple[int, ...] = (16,)
    num_joints: int = 4
    context_dim: int = 33
    num_frames: int = 128
    iterations: int = 3
    warmup: int = 1
    seed: int = 0
    amp: str = "none"
    compile: bool = False
    tf32: str = "auto"
    pi_opt: str = "muon"
    fused_adam: str = "auto"
    cuda_graph: str = "off"
    profile: str = "off"
    trace_path: str | None = None
    profile_table_path: str | None = None
    profile_row_limit: int = 40
    deterministic: bool = True
    critic_epochs: int = 1
    critic_minibatches: int = 2
    no_critic: bool = False
    no_sync: bool = True
    pi_ema: float = 0.0
    utilization: bool = True
    utilization_interval: float = 0.2
    label: str = "imagined"
    checkpoint: str | None = None
    motion_files: tuple[str, ...] = ()
    mjcf_path: str | None = None


@dataclass
class HardwareSample:
    cpu_percent: float | None = None
    gpu_util_percent: float | None = None
    gpu_memory_mib: float | None = None
    source: str = "unavailable"


class HardwareMonitor:
    """Best-effort CPU/GPU sampling without making external packages mandatory."""

    def __init__(self, interval: float, enabled: bool = True):
        if interval <= 0:
            raise ValueError("utilization interval must be positive")
        self.interval = interval
        self.enabled = enabled
        self.samples: list[HardwareSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="imagined-bench-monitor", daemon=True
        )
        self._cpu_ticks: tuple[int, int] | None = None
        self._query = self._select_query()

    def start(self) -> None:
        if self.enabled:
            self._thread.start()

    def stop(self) -> list[HardwareSample]:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(self.interval * 4.0, 1.0))
        return self.samples

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.samples.append(self.sample())

    def _select_query(self) -> Callable[[], HardwareSample] | None:
        if not self.enabled:
            return None
        if _executable("nvidia-smi"):
            return lambda: self._sample_nvidia()
        for command, sampler in (
            ("rocm-smi", self._sample_rocm_smi),
            ("amd-smi", self._sample_amd_smi),
        ):
            if _executable(command):
                return sampler
        return lambda: HardwareSample(cpu_percent=_cpu_percent(self), source="cpu-only")

    def sample(self) -> HardwareSample:
        cpu = _cpu_percent(self)
        if self._query is None:
            return HardwareSample(cpu_percent=cpu, source="cpu-only")
        sample = self._query()
        return HardwareSample(
            cpu_percent=cpu,
            gpu_util_percent=sample.gpu_util_percent,
            gpu_memory_mib=sample.gpu_memory_mib,
            source=sample.source or "unavailable",
        )

    @staticmethod
    def _sample_nvidia() -> HardwareSample:
        output = _run_query(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        rows = [row for row in output.splitlines() if row.strip()]
        if not rows:
            return HardwareSample(source="nvidia-smi(unavailable)")
        # With CUDA_VISIBLE_DEVICES the visible ordinal is not exposed by this
        # query. A single row is unambiguous; otherwise report the first GPU and
        # label the sample as aggregated rather than pretending exact selection.
        try:
            first = rows[0].split(",")
            gpu_util = float(first[0].strip().rstrip("%"))
            memory = float(first[1].strip().rstrip("MiB"))
        except (IndexError, ValueError):
            return HardwareSample(source="nvidia-smi(parse-error)")
        source = "nvidia-smi" if len(rows) == 1 else "nvidia-smi[first-visible-host]"
        return HardwareSample(gpu_util_percent=gpu_util, gpu_memory_mib=memory, source=source)

    @staticmethod
    def _sample_rocm_smi() -> HardwareSample:
        output = _run_query(["rocm-smi", "--csv", "--showuse", "--showmemuse", "vram"])
        gpu_util = _parse_metric(output, ("gpu_use", "gpu utilization"))
        memory = _parse_metric(output, ("vram_used", "memory used"))
        return HardwareSample(
            gpu_util_percent=gpu_util,
            gpu_memory_mib=memory,
            source="rocm-smi"
            if gpu_util is not None or memory is not None
            else "rocm-smi(unparsed)",
        )

    @staticmethod
    def _sample_amd_smi() -> HardwareSample:
        output = _run_query(["amd-smi", "monitor", "--csv"])
        gpu_util = _parse_metric(output, ("gpu_activity", "gpu_util", "gpu utilization"))
        memory = _parse_metric(output, ("vram_used", "memory_used", "memory used"))
        return HardwareSample(
            gpu_util_percent=gpu_util,
            gpu_memory_mib=memory,
            source="amd-smi" if gpu_util is not None or memory is not None else "amd-smi(unparsed)",
        )


def _executable(name: str) -> bool:
    for directory in os.get_exec_path():
        if (Path(directory) / name).is_file():
            return True
    return False


def _run_query(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def _parse_metric(output: str, names: Sequence[str]) -> float | None:
    lines = output.splitlines()
    if len(lines) < 2:
        return None
    header = [cell.strip().strip('"').lower() for cell in lines[0].split(",")]
    column = next(
        (index for index, cell in enumerate(header) if any(n in cell for n in names)), None
    )
    if column is None:
        return None
    values: list[float] = []
    for line in lines[1:]:
        cells = line.split(",")
        if len(cells) <= column:
            continue
        text = cells[column].strip().strip('%"MiB ')
        try:
            values.append(float(text))
        except ValueError:
            continue
    return statistics.median(values) if values else None


def _cpu_percent(monitor: HardwareMonitor) -> float | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        ticks = tuple(int(value) for value in fields[:4])
    except (OSError, ValueError, IndexError):
        return None
    total = sum(ticks)
    idle = ticks[3]
    previous = monitor._cpu_ticks
    monitor._cpu_ticks = ticks
    if previous is None:
        return None
    total_delta = total - sum(previous)
    idle_delta = idle - previous[3]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta))


def parse_hidden(value: str) -> tuple[int, ...]:
    try:
        hidden = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "hidden dimensions must be comma-separated integers"
        ) from error
    if not hidden or any(size <= 0 for size in hidden):
        raise argparse.ArgumentTypeError("hidden dimensions must be positive")
    return hidden


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


def build_fake_task(settings: BenchmarkSettings, device: torch.device):
    return make_fake_imagined_task(
        batch=settings.batch,
        num_joints=settings.num_joints,
        context_dim=settings.context_dim,
        wm_hidden=settings.wm_hidden,
        num_frames=settings.num_frames,
        device=device,
        seed=settings.seed,
    )


def build_real_task(settings: BenchmarkSettings, device: torch.device):
    if not settings.checkpoint:
        raise BenchmarkError("--checkpoint is required with --task real")
    if not settings.motion_files:
        raise BenchmarkError("at least one --motion-file is required with --task real")
    from unisupermimic.tasks.motion_tracking.g1.supermimic_imagined.task import (
        G1ImaginedConfig,
        build_g1_imagined_task,
    )

    checkpoint_path = Path(settings.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise BenchmarkError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    task = G1ImaginedConfig(
        motion_files=tuple(str(path) for path in settings.motion_files),
        mjcf_path=settings.mjcf_path or "src/unisupermimic/assets/robots/g1/supermimic.xml",
        gamma=0.99,
    )
    return build_g1_imagined_task(task, checkpoint, device)


def build_learner_settings(settings: BenchmarkSettings) -> ImaginedSuperMimicConfig:
    if settings.amp not in ("none", "bf16"):
        raise BenchmarkError(
            f"AMP mode {settings.amp!r} is not implemented by ImaginedSuperMimicLearner; "
            "use none or bf16"
        )
    if settings.pi_opt == "muon" and len(settings.policy_hidden) < 3:
        raise BenchmarkError(
            "MuonAdam needs a hidden policy Linear; use at least three policy hidden dimensions"
        )
    return ImaginedSuperMimicConfig(
        iterations=settings.iterations,
        batch=settings.batch,
        horizon=settings.horizon,
        goal_frames=settings.goal_frames,
        hist=settings.hist,
        z_dim=settings.z_dim,
        policy_hidden=settings.policy_hidden,
        critic_hidden=settings.critic_hidden,
        pi_opt=settings.pi_opt,
        no_sync=settings.no_sync,
        pi_ema=settings.pi_ema,
        critic_epochs=settings.critic_epochs,
        critic_minibatches=settings.critic_minibatches,
        no_critic=settings.no_critic,
        reset_noise=True,
        lr_decay_iters=max(settings.iterations, 1),
        ckpt_every=max(settings.iterations + 1, 1),
        log_every=max(settings.iterations + 1, 1),
        log_backend="none",
        seed=settings.seed,
        amp=settings.amp,
        compile=settings.compile,
        test_hidden_dims=settings.critic_hidden,
    )


def configure_precision(settings: BenchmarkSettings, device: torch.device) -> dict[str, Any]:
    if settings.tf32 not in ("auto", "on", "off"):
        raise BenchmarkError(f"unknown TF32 mode {settings.tf32!r}")
    matmul_tf32 = (settings.tf32 == "on") or (settings.tf32 == "auto" and settings.amp == "none")
    cudnn_tf32 = settings.tf32 != "off"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
    else:
        matmul_tf32 = False
        cudnn_tf32 = False
    return {
        "tf32_matmul": bool(matmul_tf32),
        "tf32_cudnn": bool(cudnn_tf32),
    }


def configure_optimizer(
    learner: ImaginedSuperMimicLearner,
    settings: BenchmarkSettings,
    device: torch.device,
) -> dict[str, Any]:
    if settings.fused_adam not in ("auto", "on", "off"):
        raise BenchmarkError(f"unknown fused Adam mode {settings.fused_adam!r}")
    enabled = (
        device.type == "cuda" if settings.fused_adam == "auto" else settings.fused_adam == "on"
    )
    if device.type != "cuda" and enabled:
        raise BenchmarkError("fused Adam is only available on CUDA")
    if settings.pi_opt == "muon":
        learner.opt_pi = MuonAdam(
            learner.pi,
            [learner.pi.enc, learner.pi.dec],
            lr=learner.cfg.lr,
            betas=(0.9, 0.95),
            fused=enabled,
        )
    else:
        learner.opt_pi = torch.optim.Adam(
            learner.pi.parameters(),
            lr=learner.cfg.lr,
            betas=(0.9, 0.95),
            fused=enabled,
        )
    learner.opt_v = torch.optim.Adam(learner.critic.parameters(), lr=learner.cfg.lr, fused=enabled)
    return {
        "fused_adam_requested": settings.fused_adam,
        "fused_adam_effective": enabled,
    }


def configure_determinism(settings: BenchmarkSettings) -> dict[str, Any]:
    torch.manual_seed(settings.seed)
    try:
        torch.use_deterministic_algorithms(mode=settings.deterministic)
        enabled = settings.deterministic
    except Exception as error:  # A backend may reject determinism for an op in this model.
        torch.use_deterministic_algorithms(mode=False)
        enabled = False
        return {"deterministic_algorithms": False, "determinism_error": str(error)}
    return {"deterministic_algorithms": enabled}


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}

    def percentile(value: float) -> float:
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * value)))
        return ordered[index]

    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p05": percentile(0.05),
        "p95": percentile(0.95),
    }


def _summarize_samples(samples: Sequence[HardwareSample]) -> dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "cpu_util_percent": None,
            "gpu_util_percent": None,
            "gpu_memory_used_mib": None,
            "source": "disabled-or-no-samples",
        }
    gpu_utils = [
        sample.gpu_util_percent for sample in samples if sample.gpu_util_percent is not None
    ]
    gpu_memory = [sample.gpu_memory_mib for sample in samples if sample.gpu_memory_mib is not None]
    cpu = [sample.cpu_percent for sample in samples if sample.cpu_percent is not None]
    sources = sorted({sample.source for sample in samples})
    return {
        "sample_count": len(samples),
        "cpu_util_percent": _percentiles(cpu) if cpu else None,
        "gpu_util_percent": _percentiles(gpu_utils) if gpu_utils else None,
        "gpu_memory_used_mib": _percentiles(gpu_memory) if gpu_memory else None,
        "source": sources,
    }


def _torch_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "hip_build": getattr(torch.version, "hip", None),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    if cuda_available:
        try:
            current = torch.cuda.current_device()
            result.update(
                {
                    "current_device": current,
                    "device_name": torch.cuda.get_device_name(current),
                    "device_capability": ".".join(
                        map(str, torch.cuda.get_device_capability(current))
                    ),
                    "torch_cuda_version": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                }
            )
        except Exception as error:  # Enumeration can fail on restricted ROCm containers.
            result["device_query_error"] = str(error)
    return result


def _memory_snapshot(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "allocated_mib": None,
            "reserved_mib": None,
            "max_allocated_mib": None,
            "max_reserved_mib": None,
        }
    synchronize = getattr(torch.cuda, "synchronize", None)
    if callable(synchronize):
        synchronize(device)
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "max_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def _cuda_graph_scope_supported(settings: BenchmarkSettings, device: torch.device) -> None:
    if settings.cuda_graph == "off":
        return
    if device.type != "cuda" or not hasattr(torch.cuda, "CUDAGraph"):
        raise BenchmarkError("CUDA graphs require PyTorch on a CUDA device")
    if settings.cuda_graph not in ("off", "update", "imag-step"):
        raise BenchmarkError(f"unknown CUDA graph scope {settings.cuda_graph!r}")


def _make_update_callable(
    learner: ImaginedSuperMimicLearner,
    settings: BenchmarkSettings,
    device: torch.device,
) -> Callable[[], tuple[Any, ...]]:
    _cuda_graph_scope_supported(settings, device)
    if settings.cuda_graph == "off":
        return lambda: tuple(learner.update())

    if settings.cuda_graph == "imag-step":
        # This graph covers the repeated frozen-WM MLP transition, which is the
        # graph-safe inner loop. The reported metric remains learner.update().
        static_args = learner._imag_step_fn(
            tuple(item.clone() for item in learner._state),  # type: ignore[attr-defined]
            learner._obs_hist.clone(),  # type: ignore[attr-defined]
            learner._previous_action.clone(),  # type: ignore[attr-defined]
            learner._frames.clone(),  # type: ignore[attr-defined]
            learner._context.clone(),  # type: ignore[attr-defined]
            learner._normalized_context.clone(),  # type: ignore[attr-defined]
            learner.mdp.goal(learner._frames, settings.goal_frames),  # type: ignore[attr-defined]
        )
        graph = torch.cuda.CUDAGraph()
        learner.synchronize = lambda: torch.cuda.synchronize(device)  # type: ignore[method-assign]
        learner._cuda_graph = graph  # type: ignore[attr-defined]
        with torch.cuda.graph(graph):
            learner._imag_step_fn(*static_args)  # type: ignore[attr-defined]

        def replay() -> tuple[Any, ...]:
            graph.replay()
            return tuple(learner.update())

        return replay

    # Full-update capture is intentionally attempted rather than silently
    # reporting eager numbers. Current reset/count logic contains host syncs, so
    # capture failure is useful diagnostic information rather than a fallback.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = tuple(learner.update())

    def replay() -> tuple[Any, ...]:
        graph.replay()
        return result

    return replay


def _export_profile(
    profiler: torch.profiler.profile,
    settings: BenchmarkSettings,
    output_dir: Path,
) -> dict[str, str]:
    exported: dict[str, str] = {}
    trace = Path(settings.trace_path or output_dir / "trace.json").expanduser()
    trace.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace))
    exported["chrome_trace"] = str(trace)
    table_path = Path(settings.profile_table_path or output_dir / "profile.txt").expanduser()
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table = profiler.key_averages().table(
        sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
        row_limit=settings.profile_row_limit,
    )
    table_path.write_text(table, encoding="utf-8")
    exported["profiler_table"] = str(table_path)
    return exported


def run_benchmark(settings: BenchmarkSettings, output_dir: str | Path) -> dict[str, Any]:
    """Run one reproducible benchmark and return a JSON-serializable report."""
    if settings.iterations <= 0 or settings.warmup < 0:
        raise BenchmarkError("iterations must be positive and warmup cannot be negative")
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    device = resolve_device(settings.device)
    if device.type == "cuda":
        # ImaginedSuperMimicLearner compares torch.device objects, and plain
        # "cuda" is deliberately unequal to the task's concrete "cuda:N".
        device = torch.device("cuda", torch.cuda.current_device())
    task_builder = build_real_task if settings.task == "real" else build_fake_task
    if settings.task not in ("fake", "real"):
        raise BenchmarkError(f"unknown task mode {settings.task!r}")

    os.environ.setdefault("PYTHONHASHSEED", str(settings.seed))
    precision = configure_precision(settings, device)
    determinism = configure_determinism(settings)
    torch.manual_seed(settings.seed)
    task = task_builder(settings, device)
    learner_cfg = build_learner_settings(settings)
    learner = ImaginedSuperMimicLearner(task, learner_cfg, output_path, device)
    if settings.task == "fake":
        # The tiny synthetic motion deliberately uses arbitrary joint counts. The
        # production reset path has temporary 29-joint slices, so reset all rows
        # together after an update instead of indexing partial reference states.
        original_reset_batch = learner._reset_batch

        def fake_reset_batch(indices=None):
            return original_reset_batch(None)

        learner._reset_batch = fake_reset_batch
    optimizer = configure_optimizer(learner, settings, device)
    networks_module.AMP_DTYPE = torch.bfloat16 if settings.amp == "bf16" else None
    update = _make_update_callable(learner, settings, device)
    synchronize = getattr(torch.cuda, "synchronize", None)

    for _ in range(settings.warmup):
        update()
    if callable(synchronize) and device.type == "cuda":
        synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before_memory = _memory_snapshot(device)
    times: list[float] = []
    cuda_times: list[float] = []
    monitor = HardwareMonitor(settings.utilization_interval, settings.utilization)
    monitor.start()
    started = time.perf_counter()

    profiler: torch.profiler.profile | None = None
    if settings.profile != "off":
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.__enter__()

    try:
        for _ in range(settings.iterations):
            start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if start_event is not None and end_event is not None:
                start_event.record()
            start = time.perf_counter()
            update()
            elapsed = time.perf_counter() - start
            if start_event is not None and end_event is not None:
                end_event.record()
                torch.cuda.current_stream(device).synchronize()
                cuda_times.append(start_event.elapsed_time(end_event) / 1000.0)
            times.append(elapsed)
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
        if callable(synchronize) and device.type == "cuda":
            synchronize(device)
        wall = time.perf_counter() - started
        samples = monitor.stop()
    after_memory = _memory_snapshot(device)

    timing = _percentiles(times)
    cuda_timing = _percentiles(cuda_times)
    measured_seconds = sum(times)
    imagined_steps = settings.iterations * settings.batch * settings.horizon
    report: dict[str, Any] = {
        "schema_version": 1,
        "label": settings.label,
        "settings": asdict(settings),
        "environment": _torch_environment(),
        "precision": {
            **precision,
            **determinism,
            "amp": settings.amp,
            "compile": settings.compile,
            "cuda_graph": settings.cuda_graph,
            **optimizer,
        },
        "timing": {
            "iterations": settings.iterations,
            "warmup": settings.warmup,
            "wall_seconds": wall,
            "measured_seconds": measured_seconds,
            "seconds_per_iteration": timing,
            "cuda_seconds_per_iteration": cuda_timing if cuda_timing else None,
            "iterations_per_second": settings.iterations / measured_seconds
            if measured_seconds
            else None,
            "imagined_steps_per_second": imagined_steps / measured_seconds
            if measured_seconds
            else None,
            "imagined_steps_total": imagined_steps,
        },
        "memory": {
            "before": before_memory,
            "after": after_memory,
            "peak_growth": {
                "max_allocated_mib": max(
                    0.0, after_memory["max_allocated_mib"] - before_memory["max_allocated_mib"]
                )
                if after_memory["max_allocated_mib"] is not None
                and before_memory["max_allocated_mib"] is not None
                else None,
                "max_reserved_mib": max(
                    0.0, after_memory["max_reserved_mib"] - before_memory["max_reserved_mib"]
                )
                if after_memory["max_reserved_mib"] is not None
                and before_memory["max_reserved_mib"] is not None
                else None,
            },
        },
        "utilization": _summarize_samples(samples),
    }
    if profiler is not None:
        report["profile"] = _export_profile(profiler, settings, output_path)

    result_path = output_path / "benchmark.json"
    report["report_path"] = str(result_path)
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark imagined frozen-world-model learner updates reproducibly."
    )
    parser.add_argument("--task", choices=("fake", "real"), default="fake")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or hip:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--goal-frames", type=int, default=2)
    parser.add_argument("--hist", type=int, default=2)
    parser.add_argument("--z-dim", type=int, default=8)
    parser.add_argument("--policy-hidden", type=parse_hidden, default=(32, 32, 32))
    parser.add_argument("--critic-hidden", type=parse_hidden, default=(16,))
    parser.add_argument("--wm-hidden", type=parse_hidden, default=(16,))
    parser.add_argument("--num-joints", type=int, default=4)
    parser.add_argument("--context-dim", type=int, default=33)
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tf32", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--pi-opt", choices=("adam", "muon"), default="muon")
    parser.add_argument("--fused-adam", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--cuda-graph",
        choices=("off", "update", "imag-step"),
        default="off",
        help="off, full learner update capture, or the graph-safe imagination step",
    )
    parser.add_argument("--profile", choices=("off", "torch"), default="off")
    parser.add_argument("--trace-path")
    parser.add_argument("--profile-table-path")
    parser.add_argument("--profile-row-limit", type=int, default=40)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--critic-minibatches", type=int, default=2)
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--no-sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pi-ema", type=float, default=0.0)
    parser.add_argument("--utilization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--utilization-interval", type=float, default=0.2)
    parser.add_argument("--label", default="imagined")
    parser.add_argument("--output", default="benchmarks/imagined")
    parser.add_argument("--checkpoint", help="frozen WM checkpoint; required for --task real")
    parser.add_argument("--motion-file", action="append", default=[], dest="motion_files")
    parser.add_argument("--mjcf-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = BenchmarkSettings(
        task=args.task,
        device=args.device,
        batch=args.batch,
        horizon=args.horizon,
        goal_frames=args.goal_frames,
        hist=args.hist,
        z_dim=args.z_dim,
        policy_hidden=args.policy_hidden,
        critic_hidden=args.critic_hidden,
        wm_hidden=args.wm_hidden,
        num_joints=args.num_joints,
        context_dim=args.context_dim,
        num_frames=args.num_frames,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        amp=args.amp,
        compile=args.compile,
        tf32=args.tf32,
        pi_opt=args.pi_opt,
        fused_adam=args.fused_adam,
        cuda_graph=args.cuda_graph,
        profile=args.profile,
        trace_path=args.trace_path,
        profile_table_path=args.profile_table_path,
        profile_row_limit=args.profile_row_limit,
        deterministic=args.deterministic,
        critic_epochs=args.critic_epochs,
        critic_minibatches=args.critic_minibatches,
        no_critic=args.no_critic,
        no_sync=args.no_sync,
        pi_ema=args.pi_ema,
        utilization=args.utilization,
        utilization_interval=args.utilization_interval,
        label=args.label,
        checkpoint=args.checkpoint,
        motion_files=tuple(args.motion_files),
        mjcf_path=args.mjcf_path,
    )
    report = run_benchmark(settings, args.output)
    timing = report["timing"]
    print(
        f"{settings.label}: {timing['iterations_per_second']:.3f} it/s, "
        f"{timing['imagined_steps_per_second']:.1f} imagined steps/s, "
        f"{timing['seconds_per_iteration']['median'] * 1e3:.3f} ms/update"
    )
    print(f"report: {report['report_path']}")
    if "profile" in report:
        print(f"trace: {report['profile'].get('chrome_trace')}")
        print(f"table: {report['profile'].get('profiler_table')}")
    return 0


__all__ = [
    "BenchmarkError",
    "BenchmarkSettings",
    "HardwareMonitor",
    "build_parser",
    "main",
    "run_benchmark",
]
