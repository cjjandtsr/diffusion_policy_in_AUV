"""Auditable LoRA wrappers for the existing E5 Linear and Conv1d modules."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterator, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class LoRAContractError(RuntimeError):
    """Raised before a partial or ambiguous adapter installation is returned."""


# P5 inventories the complete E5 policy as 16 Linear + 43 Conv1d.  Formal RL
# keeps the observation encoder frozen, so the actor allowlist is the denoiser
# subset: (16 - 2) Linear and (43 - 10) Conv1d.
E5_DENOISER_LINEAR_TARGETS = 14
E5_DENOISER_CONV1D_TARGETS = 33
E5_FROZEN_ENCODER_LINEAR_MODULES = 2
E5_FROZEN_ENCODER_CONV1D_MODULES = 10


class _Gate:
    def __init__(self) -> None:
        self.enabled = False


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, gate: _Gate):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.__dict__["_gate"] = gate
        parameter_options = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, base.in_features, **parameter_options)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, self.rank, **parameter_options)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value):
        result = self.base(value)
        if self.__dict__["_gate"].enabled:
            result = result + F.linear(
                F.linear(value, self.lora_a), self.lora_b
            ) * self.scale
        return result


class LoRAConv1d(nn.Module):
    def __init__(self, base: nn.Conv1d, rank: int, alpha: float, gate: _Gate):
        super().__init__()
        if base.groups != 1:
            raise LoRAContractError("grouped Conv1d is not an approved E5 LoRA target")
        if base.padding_mode != "zeros":
            raise LoRAContractError("non-zero Conv1d padding mode is not supported")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.__dict__["_gate"] = gate
        kernel = base.kernel_size[0]
        parameter_options = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_a = nn.Parameter(
            torch.empty(
                self.rank, base.in_channels, kernel, **parameter_options
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_channels, self.rank, 1, **parameter_options
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value):
        result = self.base(value)
        if self.__dict__["_gate"].enabled:
            low_rank = F.conv1d(
                value,
                self.lora_a,
                stride=self.base.stride,
                padding=self.base.padding,
                dilation=self.base.dilation,
            )
            result = result + F.conv1d(low_rank, self.lora_b) * self.scale
        return result


@dataclass(frozen=True)
class LoRAIdentity:
    target_names: Tuple[str, ...]
    rank: int
    alpha: float
    trainable_parameters: int

    @property
    def content_hash(self) -> str:
        payload = {
            "alpha": self.alpha,
            "rank": self.rank,
            "target_names": self.target_names,
            "trainable_parameters": self.trainable_parameters,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class LoRABundle:
    """Shared gate and exact target registry; parameters live in the E5 model."""

    def __init__(self, model: nn.Module, gate: _Gate, identity: LoRAIdentity):
        self.model = model
        self._gate = gate
        self.identity = identity

    @property
    def target_names(self) -> Tuple[str, ...]:
        return self.identity.target_names

    @contextmanager
    def mode(self, enabled: bool) -> Iterator[None]:
        previous = self._gate.enabled
        self._gate.enabled = bool(enabled)
        try:
            yield
        finally:
            self._gate.enabled = previous

    def named_adapter_parameters(self):
        for name, parameter in self.model.named_parameters():
            if name.endswith(("lora_a", "lora_b")):
                yield name, parameter

    def adapter_state_dict(self):
        return {
            name: parameter.detach().clone()
            for name, parameter in self.named_adapter_parameters()
        }

    def load_adapter_state_dict(self, state):
        current = dict(self.named_adapter_parameters())
        if set(state) != set(current):
            raise LoRAContractError("LoRA checkpoint target set mismatch")
        with torch.no_grad():
            for name, parameter in current.items():
                value = state[name]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise LoRAContractError("LoRA checkpoint shape mismatch: {}".format(name))
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _replace_module(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def install_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    expected_linear: int = E5_DENOISER_LINEAR_TARGETS,
    expected_conv1d: int = E5_DENOISER_CONV1D_TARGETS,
) -> LoRABundle:
    """Wrap every exact Linear/Conv1d in the native E5 denoiser only.

    Callers must pass ``policy.model`` rather than the complete policy.  The
    native observation encoder is deliberately excluded and remains frozen.
    """
    if rank <= 0 or alpha <= 0:
        raise LoRAContractError("rank and alpha must be positive")
    candidates: List[Tuple[str, nn.Module]] = [
        (name, module)
        for name, module in model.named_modules()
        if name and type(module) in (nn.Linear, nn.Conv1d)
    ]
    linear_count = sum(type(module) is nn.Linear for _, module in candidates)
    conv_count = sum(type(module) is nn.Conv1d for _, module in candidates)
    if (linear_count, conv_count) != (expected_linear, expected_conv1d):
        raise LoRAContractError(
            "E5 LoRA inventory mismatch: expected ({}, {}), got ({}, {})".format(
                expected_linear, expected_conv1d, linear_count, conv_count
            )
        )

    model.requires_grad_(False)
    gate = _Gate()
    for name, module in candidates:
        if type(module) is nn.Linear:
            replacement = LoRALinear(module, rank, alpha, gate)
        else:
            replacement = LoRAConv1d(module, rank, alpha, gate)
        _replace_module(model, name, replacement)

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(
        not name.endswith(("lora_a", "lora_b")) for name, _ in trainable
    ):
        raise LoRAContractError("trainable parameter boundary is not LoRA-only")
    identity = LoRAIdentity(
        target_names=tuple(sorted(name for name, _ in candidates)),
        rank=int(rank),
        alpha=float(alpha),
        trainable_parameters=sum(parameter.numel() for _, parameter in trainable),
    )
    return LoRABundle(model, gate, identity)
