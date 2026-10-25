"""Read-only fixed-condition R1 action probe with isolated random streams."""

from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def _probe_rng(seed, device):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def sample_probe_action(sampler, condition, *, seed, device):
    """The actor sees the original E5 condition and returns its policy 8x11."""
    if set(condition) != {"state", "global_cond"}:
        raise ValueError("R1 probe condition must contain actor inputs only")
    with _probe_rng(int(seed), device), torch.no_grad():
        action = sampler.sample(condition, deterministic=False).action.policy
    if tuple(action.shape) != (1, 8, 11) or not torch.isfinite(action).all():
        raise ValueError("R1 probe action must be finite (1,8,11)")
    return action.detach().cpu().clone()


def compare_probe_actions(initial, current):
    initial = torch.as_tensor(initial)
    current = torch.as_tensor(current)
    if tuple(initial.shape) != (1, 8, 11) or current.shape != initial.shape:
        raise ValueError("R1 probe requires paired 8x11 policy actions")
    difference = current - initial
    return {
        "prefix_mean_abs_delta": float(difference[:, :4].abs().mean()),
        "tail_mean_abs_delta": float(difference[:, 4:].abs().mean()),
        "prefix_l2_delta": float(difference[:, :4].square().sum().sqrt()),
        "tail_l2_delta": float(difference[:, 4:].square().sum().sqrt()),
    }
