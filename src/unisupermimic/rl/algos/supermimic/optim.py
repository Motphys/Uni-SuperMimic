"""Mixed Muon/Adam optimizers for SuperMimic networks."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


class MuonAdam:
    """Use Muon for hidden weight matrices and Adam for every other parameter.

    This follows the split used by InstantTrack: Muon is intended for
    two-dimensional hidden-layer weight matrices, while first/last layers, biases,
    normalizers, and other vector parameters remain on Adam. Muon uses
    ``match_rms_adamw`` scaling so that it can reuse the existing Adam learning
    rate and schedule without a separate learning-rate search.

    ``capturable`` is for callers that capture the optimizer step in a CUDA graph.
    The learning rates live in device tensors (``learning_rates``, written by
    :meth:`set_lr`) and the Muon update is spelled with explicit ops, because the
    stock step reads a Python float. Both paths update bit-identically: the Muon
    step below is the same fused multiply-add that
    ``param.add_(update, alpha=-lr * ratio)`` performs, and the capturable fused
    Adam matched the stock fused Adam bit for bit over 300 steps with a changing
    learning rate.
    """

    def __init__(
        self,
        model: nn.Module,
        mlps: list[nn.Module],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        fused: bool = True,
        capturable: bool = False,
    ) -> None:
        hidden = {
            id(layer.weight) for mlp_module in mlps for layer in _linear_layers(mlp_module)[1:-1]
        }
        muon_parameters = [parameter for parameter in model.parameters() if id(parameter) in hidden]
        adam_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in hidden and parameter.requires_grad
        ]
        if not muon_parameters:
            raise ValueError("MuonAdam requires at least one hidden Linear weight matrix")

        self.capturable = capturable
        if capturable:
            device = muon_parameters[0].device
            # float64 for Muon: the stock step scales by the Python float ``lr * ratio``
            # (double), so the device product has to be rounded from double as well.
            # The capturable fused Adam takes a float32 rate.
            self.muon_lr = torch.tensor(float(lr), dtype=torch.float64, device=device)
            self.adam_lr = torch.tensor(float(lr), dtype=torch.float32, device=device)
            self.muon = torch.optim.Muon(
                muon_parameters,
                lr=cast(float, self.muon_lr),
                weight_decay=0.0,
                adjust_lr_fn="match_rms_adamw",
            )
            self.adam = torch.optim.Adam(
                adam_parameters,
                lr=self.adam_lr,
                betas=betas,
                fused=True,
                capturable=True,
            )
        else:
            self.muon = torch.optim.Muon(
                muon_parameters,
                lr=lr,
                weight_decay=0.0,
                adjust_lr_fn="match_rms_adamw",
            )
            self.adam = torch.optim.Adam(adam_parameters, lr=lr, betas=betas, fused=fused)

    @property
    def param_groups(self) -> list[dict]:
        """Return live parameter groups from both wrapped optimizers."""

        return cast(list[dict], self.muon.param_groups + self.adam.param_groups)

    @property
    def learning_rates(self) -> list[torch.Tensor] | None:
        """The device learning-rate tensors when ``capturable``, else ``None``.

        ``train`` writes the iteration's learning rate into them (``fill_``); a captured step
        reads them from device memory.
        """
        if not self.capturable:
            return None
        return [self.muon_lr, self.adam_lr]

    def set_lr(self, learning_rate: float) -> None:
        """Re-bind the learning rate of both wrapped optimizers.

        ``capturable`` fills the device tensors in place (a captured step reads them);
        otherwise the float is re-bound on the param groups.
        """
        if self.capturable:
            self.muon_lr.fill_(learning_rate)
            self.adam_lr.fill_(learning_rate)
            return
        for group in self.param_groups:
            group["lr"] = learning_rate

    def state_tensors(self) -> list[torch.Tensor]:
        """The optimizer state a step reads and writes: Muon momentum, Adam moments and step
        counters, plus the device learning rates when ``capturable``.

        For a caller that snapshots an optimizer around a CUDA graph capture.
        """
        tensors = [
            value
            for optimizer in (self.muon, self.adam)
            for state in optimizer.state.values()
            for value in state.values()
        ]
        return tensors + (self.learning_rates or [])

    @property
    def defaults(self) -> dict:
        """Expose defaults for lightweight optimizer introspection."""

        return {**self.muon.defaults, **self.adam.defaults}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _muon_step_tensor_lr(self, ok: torch.Tensor | None = None) -> None:
        from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

        for group in self.muon.param_groups:
            if group["weight_decay"] != 0.0:
                raise ValueError("MuonAdam does not support Muon weight decay")
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.muon.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        parameter.grad, memory_format=torch.preserve_format
                    )
                buffer = state["momentum_buffer"]
                if ok is None:
                    buffer.lerp_(parameter.grad, 1 - group["momentum"])
                else:
                    buffer.copy_(
                        torch.where(
                            ok,
                            buffer.lerp(parameter.grad, 1 - group["momentum"]),
                            buffer,
                        )
                    )
                update = (
                    parameter.grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                )
                update = _zeropower_via_newtonschulz(
                    update, group["ns_coefficients"], group["ns_steps"], group["eps"]
                )
                ratio = _adjust_lr(1.0, group["adjust_lr_fn"], parameter.shape)
                # ``p.add_(update, alpha=a)`` is one fused multiply-add on the device
                # (rounding once); ``p + update * a`` as two ops rounds twice and drifts
                # by one ulp on ~0.3% of the elements. ``addcmul_`` is the same FMA with
                # a scale read from device memory, and the float64 product rounds to the
                # float the stock step converts its Python scalar to.
                scale = (group["lr"] * -ratio).to(parameter.dtype)
                if ok is None:
                    parameter.addcmul_(update.to(parameter.dtype), scale)
                else:
                    parameter.copy_(
                        torch.where(
                            ok,
                            torch.addcmul(parameter, update.to(parameter.dtype), scale),
                            parameter,
                        )
                    )

    @torch.no_grad()
    def step(self, ok: torch.Tensor | None = None) -> None:
        """Apply both optimizer steps, optionally gated by a device bool."""

        if ok is not None:
            if self.capturable:
                self._muon_step_tensor_lr(ok)
            else:
                parameters = [
                    parameter
                    for group in self.muon.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
                for parameter in parameters:
                    self.muon.state[parameter].setdefault(
                        "momentum_buffer",
                        torch.zeros_like(parameter.grad, memory_format=torch.preserve_format),
                    )
                buffers = [
                    self.muon.state[parameter]["momentum_buffer"] for parameter in parameters
                ]
                before = [tensor.clone() for tensor in parameters + buffers]
                self.muon.step()
                for tensor, original in zip(parameters + buffers, before, strict=True):
                    tensor.copy_(torch.where(ok, tensor, original))
            if not self.adam.defaults["fused"]:
                raise ValueError("A gated MuonAdam step requires fused Adam")
            cast(Any, self.adam).found_inf = (~ok).float()
            try:
                self.adam.step()
            finally:
                cast(Any, self.adam).found_inf = None
            return

        if self.capturable:
            self._muon_step_tensor_lr()
        else:
            self.muon.step()
        self.adam.step()

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])
        # ``Optimizer.load_state_dict`` installs the saved param groups wholesale, so the
        # checkpoint's ``capturable`` flag and kind of learning rate would turn this
        # optimizer into the one that saved it. A captured step needs it to stay
        # ``capturable`` (and the stock fused kernel would read a float as a pointer the
        # other way round), so both are put back after every load.
        for optimizer in (self.muon, self.adam):
            for group in optimizer.param_groups:
                if "capturable" in group:
                    group["capturable"] = self.capturable
                value = group["lr"]
                if self.capturable:
                    learning_rate = self.muon_lr if optimizer is self.muon else self.adam_lr
                    learning_rate.fill_(value.item() if isinstance(value, torch.Tensor) else value)
                    group["lr"] = learning_rate
                elif isinstance(value, torch.Tensor):
                    group["lr"] = value.item()


__all__ = ["MuonAdam"]
