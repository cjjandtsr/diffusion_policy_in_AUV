"""Atomic rollback boundary for one ReinFlow actor/critic update."""

from __future__ import annotations

import copy
import random
from typing import MutableMapping, Sequence

import numpy as np
import torch


class TransactionError(RuntimeError):
    """Raised when a transaction is used out of order."""


class UpdateTransaction:
    def __init__(
        self,
        *,
        actor,
        noise,
        critic,
        optimizers: Sequence,
        counters: MutableMapping,
    ) -> None:
        self.modules = (actor, noise, critic)
        self.optimizers = tuple(optimizers)
        self.counters = counters
        self._captured = False
        self._committed = False

    def capture(self):
        if self._captured:
            raise TransactionError("transaction is already captured")
        self.module_states = tuple(
            copy.deepcopy(module.state_dict()) for module in self.modules
        )
        self.module_modes = tuple(module.training for module in self.modules)
        self.gradient_states = tuple(
            tuple(
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in module.parameters()
            )
            for module in self.modules
        )
        self.optimizer_states = tuple(
            copy.deepcopy(optimizer.state_dict()) for optimizer in self.optimizers
        )
        self.counter_state = copy.deepcopy(dict(self.counters))
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_cpu_rng = torch.get_rng_state().clone()
        self.torch_cuda_rng = (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else None
        )
        self._captured = True
        return self

    def commit(self) -> None:
        if not self._captured:
            raise TransactionError("capture transaction before commit")
        self._committed = True

    def rollback(self) -> None:
        if not self._captured:
            raise TransactionError("capture transaction before rollback")
        for module, state, training in zip(
            self.modules, self.module_states, self.module_modes
        ):
            module.load_state_dict(state, strict=True)
            module.train(training)
        for module, gradients in zip(self.modules, self.gradient_states):
            for parameter, gradient in zip(module.parameters(), gradients):
                parameter.grad = None if gradient is None else gradient.clone().to(parameter.device)
        for optimizer, state in zip(self.optimizers, self.optimizer_states):
            optimizer.load_state_dict(state)
        self.counters.clear()
        self.counters.update(copy.deepcopy(self.counter_state))
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_cpu_rng)
        if self.torch_cuda_rng is not None:
            torch.cuda.set_rng_state_all(list(self.torch_cuda_rng))

    def __enter__(self):
        return self.capture()

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None or not self._committed:
            self.rollback()
        return False


__all__ = ["TransactionError", "UpdateTransaction"]
