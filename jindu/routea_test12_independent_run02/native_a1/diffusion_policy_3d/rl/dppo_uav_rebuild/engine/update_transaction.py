"""Rollback boundary for one DPPO optimizer update."""

from __future__ import annotations

import copy
import random

import numpy as np

import torch


class UpdateRejected(RuntimeError):
    def __init__(self, message, *, reason="unspecified", recoverable=False):
        super().__init__(message)
        self.reason = str(reason)
        self.recoverable = bool(recoverable)


class UpdateTransaction:
    def __init__(self, *, trainable_parameters, critic, optimizers, scaler=None,
                 counters=None, torch_generators=(), numpy_generators=()):
        self.parameters = tuple(trainable_parameters)
        self.critic = critic
        self.optimizers = tuple(optimizers)
        self.scaler = scaler
        self.counters = counters
        self.torch_generators = tuple(torch_generators)
        self.numpy_generators = tuple(numpy_generators)
        self._committed = False

    def __enter__(self):
        self.parameter_state = [value.detach().clone() for value in self.parameters]
        self.gradient_state = [
            None if value.grad is None else value.grad.detach().clone()
            for value in self.parameters
        ]
        self.critic_state = {
            key: value.detach().clone() for key, value in self.critic.state_dict().items()
        }
        self.optimizer_state = [
            copy.deepcopy(optimizer.state_dict()) for optimizer in self.optimizers
        ]
        self.scaler_state = copy.deepcopy(self.scaler.state_dict()) if self.scaler is not None else None
        self.counter_state = copy.deepcopy(vars(self.counters)) if self.counters is not None else None
        self.python_rng = random.getstate()
        self.numpy_legacy_rng = np.random.get_state()
        self.torch_generator_states = [generator.get_state().clone() for generator in self.torch_generators]
        self.numpy_generator_states = [copy.deepcopy(generator.bit_generator.state) for generator in self.numpy_generators]
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return self

    def commit(self):
        self._committed = True

    def rollback(self):
        with torch.no_grad():
            for parameter, saved, saved_gradient in zip(
                self.parameters, self.parameter_state, self.gradient_state
            ):
                parameter.copy_(saved)
                parameter.grad = (
                    None if saved_gradient is None else saved_gradient.clone()
                )
        self.critic.load_state_dict(self.critic_state, strict=True)
        for optimizer, state in zip(self.optimizers, self.optimizer_state):
            optimizer.load_state_dict(state)
        if self.scaler is not None:
            self.scaler.load_state_dict(self.scaler_state)
        if self.counters is not None:
            vars(self.counters).clear()
            vars(self.counters).update(copy.deepcopy(self.counter_state))
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_legacy_rng)
        for generator, state in zip(self.torch_generators, self.torch_generator_states):
            generator.set_state(state)
        for generator, state in zip(self.numpy_generators, self.numpy_generator_states):
            generator.bit_generator.state = copy.deepcopy(state)
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None or not self._committed:
            self.rollback()
        return False
