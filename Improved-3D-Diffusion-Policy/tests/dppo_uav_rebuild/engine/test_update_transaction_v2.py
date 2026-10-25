"""Rollback covers parameters, optimizer, counters, and all RNG families."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.update_transaction import UpdateTransaction


def test_uncommitted_transaction_restores_rng_and_counter_state():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    critic = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    counters = SimpleNamespace(value=3)
    torch_gen = torch.Generator().manual_seed(7)
    numpy_gen = np.random.default_rng(9)
    random.seed(11); np.random.seed(13); torch.manual_seed(15)
    expected_python = random.getstate(); expected_numpy = np.random.get_state()
    expected_torch = torch.get_rng_state().clone(); expected_gen = torch_gen.get_state().clone()
    expected_np_gen = numpy_gen.bit_generator.state
    with pytest.raises(RuntimeError):
        with UpdateTransaction(
            trainable_parameters=(parameter,), critic=critic, optimizers=(optimizer,),
            counters=counters, torch_generators=(torch_gen,), numpy_generators=(numpy_gen,),
        ):
            with torch.no_grad(): parameter.add_(5)
            counters.value = 99
            random.random(); np.random.random(); torch.rand(1)
            torch.rand(1, generator=torch_gen); numpy_gen.random()
            raise RuntimeError("inject")
    assert float(parameter) == 1.0 and counters.value == 3
    assert random.getstate() == expected_python
    assert all(np.array_equal(a, b) if hasattr(a, "shape") else a == b for a, b in zip(np.random.get_state(), expected_numpy))
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert torch.equal(torch_gen.get_state(), expected_gen)
    assert numpy_gen.bit_generator.state == expected_np_gen
