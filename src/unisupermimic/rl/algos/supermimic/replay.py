"""Fixed-size device-resident replay laid out as ``(time, environment)``."""

from __future__ import annotations

import torch


def _make_fields(
    num_joints: int,
    context_dim: int,
) -> dict[str, tuple[int | tuple[()], torch.dtype]]:
    state_dim = 13 + 2 * num_joints
    return {
        "s": (state_dim, torch.float32),
        "s1": (state_dim, torch.float32),
        "a": (num_joints, torch.float32),
        "a_prev": (num_joints, torch.float32),
        "xi": (context_dim, torch.float32),
        "r": ((), torch.float32),
        "f": ((), torch.long),
        "ep_step": ((), torch.int32),
        "done": ((), torch.bool),
    }


FIELDS = _make_fields(29, 33)


class ReplayBuffer:
    def __init__(
        self,
        T: int,
        N: int,
        device: str | torch.device,
        num_joints: int = 29,
        context_dim: int = 33,
    ):
        self.T = T
        self.N = N
        self.device = torch.device(device)
        self.fields = _make_fields(num_joints, context_dim)
        self.data = {}
        for key, (shape, dtype) in self.fields.items():
            field_shape = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[key] = torch.zeros(field_shape, dtype=dtype, device=self.device)
        self.ptr = 0
        self.size = 0

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        self.add_staged(transition)
        self.commit(1)

    def add_staged(self, transition: dict[str, torch.Tensor]) -> None:
        """Write a row without making it visible to samplers (see ``commit``)."""
        for key in self.fields:
            self.data[key][self.ptr] = transition[key]
        self.ptr = (self.ptr + 1) % self.T

    def commit(self, n: int) -> None:
        """Publish the ``n`` most recently staged rows to samplers."""
        self.size = min(self.size + n, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time indices (zero is oldest) to storage rows."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int) -> dict[str, torch.Tensor]:
        """Sample ``k``-step windows and mask transitions after episode ends."""
        if not 1 <= k <= self.size:
            raise ValueError(f"window length must be in [1, {self.size}], got {k}")
        start = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = start.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        envs = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, envs] for key in ("s", "s1", "a", "xi", "done")}
        done = out["done"].float()
        mask = torch.cumprod(
            torch.cat((torch.ones(B, 1, device=self.device), 1 - done[:, :-1]), -1),
            -1,
        )
        out["mask"] = mask
        return out

    def sample_starts(
        self,
        B: int,
        hist: int,
        exam_rows: int = 0,
        exam_prob: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample imagination starts and oldest-first, boundary-padded histories.

        With ``exam_rows`` > 0 and ``exam_prob`` > 0, each start is drawn from
        the last ``exam_rows`` environment rows (the exam worlds) with
        probability ``exam_prob`` and uniformly from the remaining practice
        rows otherwise — BlinkMimic2's ``--start_prac_frac`` split, so policy
        imagination concentrates on full-clip exam transitions.
        """
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if hist < 1:
            raise ValueError(f"history length must be positive, got {hist}")
        if exam_rows < 0 or exam_rows > self.N:
            raise ValueError(f"exam_rows must be in [0, {self.N}], got {exam_rows}")
        if not 0.0 <= exam_prob <= 1.0:
            raise ValueError(f"exam_prob must be in [0, 1], got {exam_prob}")
        age = torch.randint(0, self.size, (B,), device=self.device)
        if 0 < exam_rows < self.N and exam_prob > 0.0:
            practice = torch.randint(0, self.N - exam_rows, (B,), device=self.device)
            exam = torch.randint(0, exam_rows, (B,), device=self.device) + (self.N - exam_rows)
            pick_exam = torch.rand(B, device=self.device) < exam_prob
            env = torch.where(pick_exam, exam, practice)
        else:
            env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
        back = torch.minimum(back, age.unsqueeze(-1))
        history_rows = self._row(age.unsqueeze(-1) - back)
        history_envs = env.unsqueeze(-1).expand_as(history_rows)
        start = {key: self.data[key][row, env] for key in ("s", "a_prev", "xi", "f")}
        history = {
            key: self.data[key][history_rows, history_envs] for key in ("s", "a_prev", "xi", "f")
        }
        return start, history


__all__ = ["FIELDS", "ReplayBuffer"]
