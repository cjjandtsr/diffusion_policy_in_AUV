"""Observation adapter — the single entry point for observation-frame selection.

Design: RouteA_RL_Training_System_A_OfficialCore_Design_v1_0 §4.4.
The old code sliced `x[:, :n_obs_steps]` (OLDEST frames); the contract is
the LATEST obs_steps frames of the history buffer.
All callers (rollout, predict/eval, old/new logprob recompute, value critic,
CFG uncond, BC anchor) must route through select_latest_observations();
copying this slicing anywhere else is forbidden (§6.7).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract


@dataclass(frozen=True)
class ObservationSelection:
    """Selection metadata for debug trace (design §4.4 [JUDGMENT])."""

    history_indices: List[int]  # e.g. [5, 6, 7] for T_h=8, T_o=3
    history_steps: int
    obs_steps: int


def select_latest_observations(
    obs: Dict[str, torch.Tensor],
    obs_steps: int,
) -> Tuple[Dict[str, torch.Tensor], ObservationSelection]:
    """Select the latest obs_steps frames from each (B, T_history, ...) tensor.

    Args:
        obs: chronological observation dict, every tensor (B, T_history, ...).
        obs_steps: number of most recent frames to keep.

    Returns:
        (selected dict (B, obs_steps, ...), selection metadata).

    Raises:
        AssertionError: obs_steps out of range or inconsistent history length.
    """
    assert 0 < obs_steps, f"obs_steps={obs_steps} must be >= 1"
    assert obs, "observation dict must be non-empty"

    # History length: taken from the first tensor, all keys must agree.
    first_key = next(iter(obs))
    history_steps = obs[first_key].shape[1]
    assert (
        obs_steps <= history_steps
    ), f"obs_steps={obs_steps} exceeds history_steps={history_steps}"
    for key, value in obs.items():
        assert value.shape[1] == history_steps, (
            f"key {key} has history {value.shape[1]}, expected {history_steps}"
        )

    start = history_steps - obs_steps
    history_indices = list(range(start, history_steps))
    out = {key: value[:, start:, ...] for key, value in obs.items()}
    meta = ObservationSelection(
        history_indices=history_indices,
        history_steps=history_steps,
        obs_steps=obs_steps,
    )
    return out, meta
