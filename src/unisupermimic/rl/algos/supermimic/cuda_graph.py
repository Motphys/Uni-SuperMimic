"""Explicit CUDA graph replay for the fixed-shape SuperMimic learner updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

Inputs = Mapping[str, Any]  # tensors, or nested mappings of tensors


def _flatten(value: Inputs) -> list[torch.Tensor]:
    return [
        leaf
        for item in value.values()
        for leaf in (_flatten(item) if isinstance(item, Mapping) else [item])
    ]


def _map_leaves(value: Inputs, function: Callable[[str, torch.Tensor], Any]) -> dict[str, Any]:
    return {
        key: _map_leaves(item, function) if isinstance(item, Mapping) else function(key, item)
        for key, item in value.items()
    }


def _reverse_cumsum(values: torch.Tensor) -> torch.Tensor:
    return values.flip(0).cumsum(0).flip(0)


class _CumprodNoHostSync(torch.autograd.Function):
    """``torch.cumprod(values, 0)`` whose backward never synchronizes with the host.

    ATen's cumprod backward reads the number of zeros in the input on the host to choose
    between two formulas, which CUDA graph capture forbids. Without zeros the VJP is
    ``reverse_cumsum(grad * result) / values``, op for op what ATen computes. With the
    first zero at ``z``: the same expression for positions before ``z``,
    ``exclusive_cumprod[z] * reverse_cumsum(grad * tail)[z]`` at ``z`` (``tail`` is the
    cumprod of ``values`` with positions up to ``z`` replaced by one) and zero after it.
    """

    @staticmethod
    def forward(ctx, values: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(values, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        values, result = ctx.saved_tensors
        out = _reverse_cumsum(grad_output * result) / values
        zero = values == 0
        has_zero = zero.any(0, keepdim=True)
        first = torch.argmax(zero.to(torch.uint8), 0, keepdim=True)
        position = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
        exclusive = torch.cat((torch.ones_like(result[:1]), result[:-1]), 0)
        tail = torch.cumprod(torch.where(position <= first, torch.ones_like(values), values), 0)
        at_zero = (
            exclusive.gather(0, first) * _reverse_cumsum(grad_output * tail).gather(0, first)
        ).expand_as(out)
        patched = torch.where(
            position == first, at_zero, torch.where(position < first, out, torch.zeros_like(out))
        )
        return torch.where(has_zero, patched, out), None


def cumprod_without_host_sync(values: torch.Tensor) -> torch.Tensor:
    """``torch.cumprod(values, 0)`` for a body that is captured in a CUDA graph."""
    return _CumprodNoHostSync.apply(values, torch.cumprod(values, 0))


def snapshot_tensors(tensors: list[torch.Tensor]) -> Callable[[], None]:
    """Copy ``tensors`` now; return what writes the copies back.

    The ``snapshot`` of a ``CapturedUpdate`` whose body only mutates tensors at fixed
    addresses (normalizers, histories, optimizer state). Call it at the capture, so the
    copies are of the state right before the capture's warm-ups run the body for real.
    """
    saved = [tensor.detach().clone() for tensor in tensors]

    def restore() -> None:
        with torch.no_grad():
            for tensor, copy in zip(tensors, saved, strict=True):
                tensor.copy_(copy)

    return restore


class CapturedUpdate:
    """Run a fixed-shape learner update step, replayed from a CUDA graph after a warmup.

    :meth:`step` behaves like ``body(inputs)``:

    * the first ``warmup`` calls run ``body`` eagerly, reading the persistent input
      buffers, so lazy ``torch.compile`` work finishes before anything is captured;
    * the next call snapshots the CUDA RNG state, runs ``body`` three more times on a
      side stream, captures it into a CUDA graph, restores the RNG state and replays
      the graph, so this call — like every later one — performs exactly one update;
    * afterwards every call copies ``inputs`` into the persistent buffers and replays.

    ``body`` must be a pure function of its inputs and of state that lives at a fixed
    address (parameters, buffers, gradients): it must not synchronize with the host and
    must not read Python values that change between calls, because the graph freezes
    them. Random draws go through the default CUDA generator, which PyTorch registers
    with the graph so each replay consumes fresh offsets, and a replay consumes exactly
    what the eager body would.

    The returned tensors live in the graph's private memory pool and are overwritten by
    the next replay, so they may only be consumed before it. Positions listed in
    ``clone_outputs`` are cloned after every replay for tensors that outlive the call.

    ``snapshot`` is for bodies that mutate state (an optimizer step): it is called before
    the capture, because the capture runs the body three times on a side stream for real,
    and what it returns is called once the graph exists, so that the replay below is the
    single update this call performs.
    """

    def __init__(
        self,
        warmup: int = 3,
        clone_outputs: tuple[int, ...] = (),
        snapshot: Callable[[], Callable[[], None]] | None = None,
    ) -> None:
        if warmup < 1:
            raise ValueError(f"CapturedUpdate warmup must be positive, got {warmup}")
        self.warmup = warmup
        self.clone_outputs = clone_outputs
        self.snapshot = snapshot
        self.calls = 0
        self.inputs: dict[str, Any] | None = None
        self.body: Callable[[Any], Any] | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.outputs: Any = None

    @property
    def replaying(self) -> bool:
        return self.graph is not None

    @property
    def warming_up(self) -> bool:
        """Whether the next call still runs ``body`` in Python (before the capture).

        A body may keep host-side state of its own (a call counter); that advances only
        while the Python runs, never on a replay.
        """
        return self.graph is None and self.calls < self.warmup

    def reset(self) -> None:
        """Drop the graph; the next call runs ``warmup`` eager steps and captures again.

        For a body bound to optimizer state that ``load_state_dict`` may have moved.
        """
        self.calls = 0
        self.inputs = None
        self.body = None
        self.graph = None
        self.outputs = None

    def step(self, body: Callable[[Any], Any], inputs: Inputs) -> Any:
        if self.inputs is None:
            self.inputs = _map_leaves(inputs, lambda _key, value: torch.empty_like(value))
            self.body = body
        elif body != self.body:  # bound methods compare by instance and function
            raise ValueError("CapturedUpdate cannot switch bodies after its first call")
        targets = _flatten(self.inputs)
        for target, source in zip(targets, _flatten(inputs), strict=True):
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError(
                    f"CapturedUpdate input changed shape or dtype: "
                    f"{tuple(target.shape)}/{target.dtype} -> "
                    f"{tuple(source.shape)}/{source.dtype}"
                )
            target.copy_(source)
        if self.graph is None:
            self.calls += 1
            if self.calls <= self.warmup:
                return self.body(self.inputs)
            self._capture()
        graph = self.graph
        assert graph is not None
        graph.replay()
        if not self.clone_outputs:
            return self.outputs
        outputs = list(self.outputs)
        for index in self.clone_outputs:
            outputs[index] = outputs[index].clone()
        return tuple(outputs)

    def _capture(self) -> None:
        assert self.inputs is not None and self.body is not None
        flattened = _flatten(self.inputs)
        device = (
            flattened[0].device if flattened else torch.device("cuda", torch.cuda.current_device())
        )
        rng_state = torch.cuda.get_rng_state(device)
        restore = self.snapshot() if self.snapshot is not None else None
        try:
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            # Drop the previous outputs first: their autograd graph keeps the parameters'
            # AccumulateGrad nodes bound to the default stream, which the backward below
            # runs on other streams (PyTorch warns and inserts stream synchronizations).
            self.outputs = None
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.outputs = None
                    self.outputs = self.body(self.inputs)
                self.outputs = None
            torch.cuda.current_stream(device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.outputs = self.body(self.inputs)
        finally:
            if restore is not None:
                restore()
        # The side-stream warmup and the capture consumed random numbers for steps that
        # the replay below redoes: rewind so the replayed update sees the eager sequence.
        torch.cuda.set_rng_state(rng_state, device)
        self.graph = graph


__all__ = ["CapturedUpdate", "snapshot_tensors"]
