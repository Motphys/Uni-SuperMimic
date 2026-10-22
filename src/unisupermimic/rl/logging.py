# Adapted from UniLab (Apache-2.0) unilab/rl/logging/offpolicy.py
"""SuperMimic training logger: Rich live panel plus TensorBoard scalars."""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

_METRIC_LABELS = (
    ("wm_loss", "wm_loss"),
    ("wm_err8", "wm_err8"),
    ("pi_loss", "pi_loss"),
    ("v_loss", "v_loss"),
    ("imag_rew", "imag_rew"),
    ("imag_survival", "imag_survival"),
    ("pi_gradnorm", "pi_gradnorm"),
    ("lr", "lr"),
)

_SPARKLINE_TICKS = "▁▂▃▄▅▆▇█"
_SPARKLINE_ASCII_TICKS = " .:-=+*#%@"
_RATE_WINDOW = 50


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _fmt_number(value: float) -> str:
    if abs(value) == 0:
        return "0"
    if abs(value) >= 1e6:
        return f"{value:.2e}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    if abs(value) >= 0.001:
        return f"{value:.4f}"
    return f"{value:.2e}"


def _console_supports_unicode() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "⏱ ▲ ▼ ━ ▁█".encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def _tb_proto_modules() -> tuple[Any, Any]:
    """Import tensorboard's protobuf modules as ``Any``.

    The generated modules define their messages through reflection, which
    static checkers cannot see; returning them as ``Any`` keeps attribute
    access clean without per-line suppressions.
    """
    from tensorboard.compat.proto import event_pb2, summary_pb2

    return event_pb2, summary_pb2


class SuperMimicLogger:
    """Rich live panel and TensorBoard backend with SuperMimic metric names."""

    def __init__(
        self,
        *,
        env_name: str,
        max_iterations: int,
        num_envs: int,
        log_dir: str,
        log_backend: str = "tensorboard",
        refresh_per_second: int = 1,
    ) -> None:
        backend = log_backend.lower()
        if backend not in ("tensorboard", "none", "no_print"):
            raise ValueError(
                f"log_backend must be 'tensorboard', 'none', or 'no_print', got {log_backend!r}"
            )
        self.algo_name = "SuperMimic"
        self.env_name = env_name
        self.max_iterations = max_iterations
        self.num_envs = num_envs

        self._no_print = backend == "no_print"
        self._unicode_console = _console_supports_unicode()
        self._console = Console(force_terminal=False if not self._unicode_console else None)
        self._interactive = self._console.is_terminal and not self._no_print
        self._refresh_rate = refresh_per_second
        self._live: Live | None = None
        self._last_live_refresh_time: float | None = None

        self._start_time = 0.0
        self._iteration = 0
        self._status = ""
        self._metrics: dict[str, float | None] = {key: None for key, _ in _METRIC_LABELS}
        self._reward_history: deque[float] = deque(maxlen=200)
        self._ep_len: float | None = None
        self._ep_ret_norm: float | None = None
        self._term_frac: float | None = None
        self._collect_time = 0.0
        self._learn_time = 0.0
        self._iteration_time = 0.0
        self._iteration_times: deque[float] = deque(maxlen=_RATE_WINDOW)
        self._iteration_steps: deque[int] = deque(maxlen=_RATE_WINDOW)
        self._last_save = ""

        self._progress = Progress(
            TextColumn("[yellow]iter {task.completed}/{task.total}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        self._progress_task = self._progress.add_task("iter", total=max_iterations)

        self._tb_writer: Any | None = None
        self._finished = False
        self._closed = False
        if backend == "tensorboard" and log_dir:
            self._init_tensorboard(log_dir)

    def _init_tensorboard(self, log_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            if not self._no_print:
                self._console.print("[yellow]tensorboard not installed[/]")
            return
        os.makedirs(log_dir, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=log_dir)
        if not self._no_print:
            self._console.print(f"[dim]TensorBoard logging to: {log_dir}[/]")

    def start(self, *, status: str = "Warming up...") -> None:
        self._start_time = time.time()
        self._status = status
        if self._no_print:
            return
        if self._interactive:
            if self._live is None:
                self._live = Live(
                    self._build_display(),
                    console=self._console,
                    auto_refresh=False,
                    transient=False,
                )
                self._live.start(refresh=True)
        else:
            self._console.print(status, style="dim")

    def start_training_timer(self) -> float:
        """Restart the measured training window after warm-up collection."""
        self._start_time = time.time()
        return self._start_time

    def log_status(self, status: str) -> None:
        self._status = status
        if self._no_print:
            return
        if not self._interactive:
            self._console.print(status, style="dim")
        self._refresh()

    def log_buffer_fill(self, current: int, target: int) -> None:
        percent = current / max(target, 1) * 100
        self.log_status(f"Replay buffer: {current:,}/{target:,} steps ({percent:.0f}%)")

    def log_save(self, path: str) -> None:
        self._last_save = path
        self.log_status(f"Saved checkpoint: {path}")

    def log_step(
        self,
        iteration: int,
        *,
        wm_loss: float,
        wm_err8: float,
        lr: float,
        pi_loss: float | None = None,
        v_loss: float | None = None,
        imag_rew: float | None = None,
        imag_survival: float | None = None,
        pi_gradnorm: float | None = None,
        sim_rew: float | None = None,
        ep_len: float | None = None,
        ep_ret_norm: float | None = None,
        term_frac: float | None = None,
        iter_sim_steps: int = 0,
        collect_time: float = 0.0,
        learn_time: float = 0.0,
        iteration_time: float = 0.0,
    ) -> None:
        self._iteration = iteration
        self._metrics["wm_loss"] = wm_loss
        self._metrics["wm_err8"] = wm_err8
        self._metrics["lr"] = lr
        if pi_loss is not None:
            self._metrics["pi_loss"] = pi_loss
        if v_loss is not None:
            self._metrics["v_loss"] = v_loss
        if imag_rew is not None:
            self._metrics["imag_rew"] = imag_rew
        if imag_survival is not None:
            self._metrics["imag_survival"] = imag_survival
        if pi_gradnorm is not None:
            self._metrics["pi_gradnorm"] = pi_gradnorm
        if sim_rew is not None:
            self._reward_history.append(sim_rew)
        if ep_len is not None:
            self._ep_len = ep_len
        if ep_ret_norm is not None:
            self._ep_ret_norm = ep_ret_norm
        if term_frac is not None:
            self._term_frac = term_frac
        self._collect_time = collect_time
        self._learn_time = learn_time
        self._iteration_time = iteration_time
        if iteration_time > 0:
            self._iteration_times.append(iteration_time)
            self._iteration_steps.append(iter_sim_steps)
        self._status = "Training"
        self._progress.update(self._progress_task, completed=iteration)

        scalars: dict[str, float] = {
            "Loss/world_model": wm_loss,
            "Loss/world_model_err8": wm_err8,
            "Train/learning_rate": lr,
            "Perf/iteration_time": iteration_time,
            "Perf/collect_time": collect_time,
            "Perf/learning_time": learn_time,
            "Perf/it_per_s": self._it_per_s,
            "Perf/sim_steps_per_s": self._sim_steps_per_s,
        }
        if pi_loss is not None:
            scalars["Loss/policy"] = pi_loss
        if v_loss is not None:
            scalars["Loss/value"] = v_loss
        if imag_rew is not None:
            scalars["Loss/imagined_reward"] = imag_rew
        if imag_survival is not None:
            scalars["Loss/imagined_survival"] = imag_survival
        if pi_gradnorm is not None:
            scalars["Train/grad_norm"] = pi_gradnorm
        if sim_rew is not None:
            scalars["Reward/sim_step_norm"] = sim_rew
        if ep_len is not None:
            scalars["Episode/length"] = ep_len
        if ep_ret_norm is not None:
            scalars["Episode/return_norm"] = ep_ret_norm
        if term_frac is not None:
            scalars["Episode/terminated_frac"] = term_frac
        self._write_tb_scalars(list(scalars.items()), iteration)
        self._refresh()

    @property
    def _it_per_s(self) -> float:
        total = sum(self._iteration_times)
        return len(self._iteration_times) / total if total > 0 else 0.0

    @property
    def _sim_steps_per_s(self) -> float:
        total = sum(self._iteration_times)
        return sum(self._iteration_steps) / total if total > 0 else 0.0

    def _estimate_eta(self) -> str:
        if self._iteration <= 0 or not self._iteration_times:
            return ""
        mean_iter = sum(self._iteration_times) / len(self._iteration_times)
        remaining = max(self.max_iterations - self._iteration, 0)
        return _fmt_time(mean_iter * remaining)

    def _write_tb_scalars(self, scalars: list[tuple[str, Any]], global_step: int) -> None:
        """Write one iteration's scalars as a single event record.

        Per-scalar ``add_scalar`` costs the writer thread one open/write/close
        per record; batching keeps it to one record per iteration. Falls back
        to per-scalar writes if the batched path is unavailable.
        """
        writer = self._tb_writer
        if writer is None or not scalars:
            return
        try:
            event_pb2, summary_pb2 = _tb_proto_modules()
            event = event_pb2.Event(
                wall_time=time.time(),
                step=global_step,
                summary=summary_pb2.Summary(
                    value=[
                        summary_pb2.Summary.Value(tag=tag, simple_value=float(value))
                        for tag, value in scalars
                    ]
                ),
            )
            writer.file_writer.add_event(event)
        except (ImportError, AttributeError, TypeError, ValueError):
            for tag, value in scalars:
                writer.add_scalar(tag, value, global_step)

    def _refresh(self, *, force: bool = False) -> None:
        if self._live is None:
            return
        now = time.time()
        if (
            not force
            and self._refresh_rate > 0
            and self._last_live_refresh_time is not None
            and now - self._last_live_refresh_time < 1.0 / self._refresh_rate
        ):
            return
        self._last_live_refresh_time = now
        self._live.update(self._build_display(), refresh=True)

    def _sparkline(self, width: int = 32) -> str:
        points = list(self._reward_history)[-width:]
        if not points:
            return ""
        ticks = _SPARKLINE_TICKS if self._unicode_console else _SPARKLINE_ASCII_TICKS
        low, high = min(points), max(points)
        if high - low < 1e-12:
            return ticks[0] * len(points)
        scale = (len(ticks) - 1) / (high - low)
        return "".join(ticks[int((point - low) * scale)] for point in points)

    def _build_header(self) -> Text:
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        fields: list[tuple[str, str]] = [
            (f"{'⏱' if self._unicode_console else 'time'} {_fmt_time(elapsed)}", "green"),
        ]
        eta = self._estimate_eta()
        if eta:
            fields.append((f"ETA {eta}", "bold magenta"))
        if self._it_per_s > 0:
            fields.append((f"{self._it_per_s:.2f} it/s", "bold green"))
        if self._sim_steps_per_s > 0:
            fields.append((f"{self._sim_steps_per_s:,.0f} steps/s", "bold cyan"))
        if self._status:
            fields.append((self._status, "dim italic"))
        header = Text(no_wrap=True, overflow="ellipsis", justify="center")
        for index, (text, style) in enumerate(fields):
            if index > 0:
                header.append(" | ", style="dim")
            header.append(text, style=style)
        return header

    def _build_metrics_table(self) -> Table:
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold cyan",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Metrics", style="white", no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", width=13, no_wrap=True)
        for key, label in _METRIC_LABELS:
            value = self._metrics[key]
            table.add_row(label, "[dim]—[/]" if value is None else _fmt_number(value))
        return table

    def _build_reward_table(self) -> Table:
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold green",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Reward / Episode", style="white", no_wrap=True)
        table.add_column("Value", justify="right", no_wrap=True)
        recent = list(self._reward_history)
        if recent:
            mean_rew = sum(recent[-50:]) / len(recent[-50:])
            peak_rew = max(recent)
            trend = ""
            if len(recent) >= 10:
                old = sum(recent[-20:-10]) / 10
                new = sum(recent[-10:]) / 10
                if new > old * 1.05:
                    trend = "[green]▲[/]" if self._unicode_console else "[green]+[/]"
                elif new < old * 0.95:
                    trend = "[red]▼[/]" if self._unicode_console else "[red]-[/]"
                else:
                    trend = "[yellow]━[/]" if self._unicode_console else "[yellow]=[/]"
            table.add_row(
                f"[bold]sim_rew[/] {trend}",
                f"Mean [bold green]{mean_rew:.4f}[/] / Peak [dim]{peak_rew:.4f}[/]",
            )
            table.add_row("", f"[green]{self._sparkline()}[/]")
        else:
            table.add_row("[dim]Waiting for data...[/]", "")
        table.add_row("ep_len", "[dim]—[/]" if self._ep_len is None else f"{self._ep_len:.1f}")
        table.add_row(
            "ep_ret_norm",
            "[dim]—[/]" if self._ep_ret_norm is None else _fmt_number(self._ep_ret_norm),
        )
        table.add_row(
            "term_frac",
            "[dim]—[/]" if self._term_frac is None else f"{self._term_frac * 100:.1f}%",
        )
        return table

    def _build_timing_table(self) -> Table:
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold blue",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Timing (per iteration)", style="white", no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", width=16, no_wrap=True)

        def _add_phase(label: str, seconds: float) -> None:
            ms = seconds * 1000
            pct = seconds / self._iteration_time * 100.0 if self._iteration_time > 0 else 0.0
            table.add_row(label, f"{ms:>7.1f}ms  {pct:>3.0f}%")

        _add_phase("Collect", self._collect_time)
        _add_phase("Learn", self._learn_time)
        table.add_row("Total", f"{self._iteration_time * 1000:>7.1f}ms  100%")
        return table

    def _build_display(self) -> Panel:
        if self._console.width >= 96:
            grid = Table.grid(expand=True)
            grid.add_column(ratio=1)
            grid.add_column(width=2)
            grid.add_column(ratio=1)
            grid.add_row(self._build_metrics_table(), "", self._build_reward_table())
        else:
            grid = Table.grid(expand=True)
            grid.add_column()
            grid.add_row(self._build_metrics_table())
            grid.add_row(Text(""))
            grid.add_row(self._build_reward_table())
        title = Text()
        if self._unicode_console:
            title.append(" 🚀")
        title.append(f" {self.algo_name} ", style="bold cyan")
        title.append("|", style="dim")
        title.append(f" {self.env_name} ", style="bold white")
        return Panel(
            Group(
                self._progress,
                self._build_header(),
                Text(""),
                grid,
                Text(""),
                self._build_timing_table(),
            ),
            title=title,
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _stop_live(self) -> None:
        live = self._live
        if live is None:
            return
        self._live = None
        self._last_live_refresh_time = None
        try:
            live.update(self._build_display(), refresh=False)
        finally:
            try:
                live.stop()
            finally:
                self._console.show_cursor(True)

    def _close_writer(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

    def finish(self, *, title: str = "Training Summary") -> None:
        if self._finished:
            return
        self._finished = True
        if self._live is not None:
            self._stop_live()
        elif not self._no_print:
            self._console.print(self._build_display())
        if not self._no_print:
            elapsed = time.time() - self._start_time if self._start_time else 0.0
            lines = [
                "[bold green]Training complete[/]",
                f"  Algo: [cyan]{self.algo_name}[/] | Env: [cyan]{self.env_name}[/]",
                f"  Iterations: [yellow]{self._iteration}[/]/{self.max_iterations}",
                f"  Wall time: [yellow]{_fmt_time(elapsed)}[/]",
            ]
            final = ", ".join(
                f"{label}={_fmt_number(value)}"
                for key, label in _METRIC_LABELS
                if (value := self._metrics[key]) is not None
            )
            if final:
                lines.append(f"  Final: [yellow]{final}[/]")
            if self._last_save:
                lines.append(f"  Last checkpoint: [dim]{self._last_save}[/]")
            self._console.print()
            self._console.print(
                Panel("\n".join(lines), title=f"[bold]{title}[/]", border_style="green")
            )
        self._close_writer()
        self._closed = True

    def close(self) -> None:
        """Release live terminal state and the TB writer; no-op after finish()."""
        if self._closed:
            return
        try:
            self._stop_live()
        finally:
            try:
                self._close_writer()
            finally:
                self._closed = True


__all__ = ["SuperMimicLogger"]
