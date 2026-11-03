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

    ``capturable`` is kept for callers that later capture the optimizer in a CUDA
    graph. It stores the learning rate in a device tensor and performs the Muon
    update with the same arithmetic as ``torch.optim.Muon``; the stock optimizer
    otherwise reads a tensor learning rate on the host.
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
            learning_rate = cast(float, torch.tensor(float(lr), device=device))
            self.muon = torch.optim.Muon(
                muon_parameters,
                lr=learning_rate,
                weight_decay=0.0,
                adjust_lr_fn="match_rms_adamw",
            )
            self.adam = torch.optim.Adam(
                adam_parameters,
                lr=learning_rate,
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
                delta = update.to(parameter.dtype) * (group["lr"] * -ratio)
                parameter.add_(delta if ok is None else torch.where(ok, delta, 0.0))

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
        learning_rates = [group["lr"] for group in self.param_groups]
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])
        if self.capturable:
            for group, learning_rate in zip(self.param_groups, learning_rates, strict=True):
                learning_rate.fill_(float(group["lr"]))
                group["lr"] = learning_rate


__all__ = ["MuonAdam"]
