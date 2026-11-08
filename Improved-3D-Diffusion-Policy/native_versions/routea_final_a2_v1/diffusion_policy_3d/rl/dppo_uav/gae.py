"""GAE with per-transition next-values and episode boundaries.

Semantics (design §3.5):

    terminated: MDP terminal (reach/collision/irrecoverable hard fault) —
                masks the bootstrap
    truncated:  time limit or external sampling cut — resets the env but
                is NOT a terminal state for value bootstrapping

    bootstrap_mask_t = 1.0 - float(terminated[t])
    delta_t = rewards[t] + gamma * bootstrap_mask_t * next_values[t]
              - values[t]
    same_episode_next = t is not the last transition of its episode
    (R5.8-B: per-episode grouping — interleaved vector-env buffers
    recurse along each episode's own time series)
    continuation_t = float(
        same_episode_next and not terminated[t] and not truncated[t]
    )
    adv[t] = delta_t + gamma * gae_lambda * continuation_t * adv[t + 1]
    returns_t = adv[t] + values[t]

[PROVEN] treating time-limit truncation as terminated zeroes V(s_{t+1})
and introduces a negative terminal bias — truncated never masks bootstrap.
The episode_id check is the last-resort boundary: even if terminated or
truncated were wrongly False, a changed episode id still stops the chain.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def compute_gae(
    rewards: np.ndarray,       # (T,)
    values: np.ndarray,        # (T,) V(s_t)
    next_values: np.ndarray,   # (T,) V(next_obs_t) — per transition
    terminated: np.ndarray,    # (T,) bool — MDP terminal (masks bootstrap)
    truncated: np.ndarray,     # (T,) bool — time limit (does NOT mask)
    episode_ids: Sequence[str],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (advantages (T,), returns (T,)).

    Every transition bootstraps from its own ``next_values[t]`` (critic of
    the next observation) — the scalar final-step ``next_value`` branch is
    gone. The chain stops at episode boundaries via the
    continuation mask (terminated OR truncated OR episode-id change).

    Raises:
        ValueError: empty buffer; mismatched array lengths; empty or
            mismatched episode_ids; non-finite rewards/values/next_values.
    """
    t_len = len(rewards)
    if t_len == 0:
        raise ValueError("compute_gae on an empty buffer")
    for name, arr in (
        ("values", values),
        ("next_values", next_values),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if len(arr) != t_len:
            raise ValueError(
                f"{name} len {len(arr)} != rewards len {t_len}"
            )
    if len(episode_ids) != t_len or not all(
        isinstance(eid, str) and eid for eid in episode_ids
    ):
        raise ValueError(
            "episode_ids must be non-empty strings, one per transition"
        )
    for name, arr in (
        ("rewards", rewards),
        ("values", values),
        ("next_values", next_values),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf")

    advantages = np.zeros(t_len, dtype=np.float64)
    # R5.8-B (N3): recurse per EPISODE (slot time series), not along the
    # flattened buffer. An interleaved vector-env buffer (A0,B0,A1,B1,…)
    # must not degrade to one-step TD — each episode's own transitions
    # form its recursion chain regardless of flat adjacency.
    groups: Dict[str, List[int]] = {}
    for t in range(t_len):
        groups.setdefault(episode_ids[t], []).append(t)
    for idxs in groups.values():
        lastgaelam = 0.0
        for pos in range(len(idxs) - 1, -1, -1):
            t = idxs[pos]
            bootstrap_mask = 1.0 - float(terminated[t])
            delta = (
                rewards[t]
                + gamma * bootstrap_mask * next_values[t]
                - values[t]
            )
            same_episode_next = pos + 1 < len(idxs)
            continuation = float(
                same_episode_next
                and not terminated[t]
                and not truncated[t]
            )
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * continuation * lastgaelam
            )
    returns = advantages + values
    return advantages, returns
