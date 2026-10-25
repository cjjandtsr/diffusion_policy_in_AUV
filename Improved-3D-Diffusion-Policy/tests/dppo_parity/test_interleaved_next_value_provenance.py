"""RED #3 (R7-NEXT-VALUE): per-transition next value from own next_obs.

Current state (measured 2026-08-22, re-verified 2026-08-23): the
successor-value shortcut in ``fill_buffer_next_values``
(rollout_buffer.py:194-233) fills non-final transitions with the
SUCCESSOR transition's ``value_old`` and never forwards a critic over
the transition's own next_obs. An interleaved vector-env buffer
A0,B0,A1,B1 with A0.next_obs != A1.obs exposes the defect: A0's
next-value would be A1's critic value even though A1.obs is not the
observation A0 leads into.

R7-NEXT-VALUE contract (guide R7-4): every transition's next-value must
come from a chunked critic forward over ITS OWN next_obs
(``evaluate_transition_next_values(next_obs_seq, critic, chunk_size)``
on the rollout buffer module); GAE dual-mask preserved.

The counterexample follows the guide's Step 1 fixture verbatim:
A0.next_value must be critic(A0.next_obs) = 3.0, never A1.value_old
= 10.0. A1.obs != A0.next_obs (7.0 vs 3.0 marker).

These tests MUST FAIL on the successor shortcut and go green with the
R7-4 implementation.
"""

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
    RolloutBuffer,
    fill_buffer_next_values,
)


def _agent_pos_marker(v: float) -> np.ndarray:
    """(3, 9) agent_pos with the value marker at [0, 0]."""
    s = np.zeros((3, 9), dtype=np.float32)
    s[0, 0] = v
    return s


def _make_interleaved_buffer() -> RolloutBuffer:
    """Flat buffer A0,B0,A1,B1 — the interleaved counterexample.

    Episodes: A = [A0, A1] (A0.next_obs != A1.obs), B = [B0, B1].
    value_old: A0=1.0, B0=10.0, A1=10.0, B1=20.0 — A1.value_old=10.0
    is the guide's poisoned successor value A0 must never receive.
    next_obs markers: A0.next_obs=3.0, B0.next_obs=4.0, A1.next_obs=5.0,
    B1.next_obs=6.0 — critic(A0.next_obs) must yield 3.0.
    A1 and B1 are truncated finals (bootstrap candidates).
    """
    buffer = RolloutBuffer(contract=object())
    buffer.episode_ids[:] = ["A", "B", "A", "B"]
    buffer.values[:] = [1.0, 10.0, 10.0, 20.0]
    buffer.terminated[:] = [False, False, False, False]
    buffer.truncated[:] = [False, False, True, True]
    buffer.obs[:] = [
        {"agent_pos": _agent_pos_marker(1.0)},
        {"agent_pos": _agent_pos_marker(10.0)},
        {"agent_pos": _agent_pos_marker(7.0)},  # A1.obs — NOT A0's successor obs
        {"agent_pos": _agent_pos_marker(20.0)},
    ]
    buffer.next_obs[:] = [
        {"agent_pos": _agent_pos_marker(3.0)},  # A0.next_obs — its own
        {"agent_pos": _agent_pos_marker(4.0)},
        {"agent_pos": _agent_pos_marker(5.0)},
        {"agent_pos": _agent_pos_marker(6.0)},
    ]
    return buffer


def test_interleaved_next_value_must_come_from_own_next_obs() -> None:
    """A0.next_value must equal critic(A0.next_obs)=3.0, never 10.0.

    The successor shortcut would fill A0.next_value with A1's value_old
    (10.0) even though A1.obs != A0.next_obs. The critic maps the obs
    marker at agent_pos[0,0] to the value — the only way A0 gets 3.0 is
    a forward over A0's own next_obs.
    """
    buffer = _make_interleaved_buffer()

    def critic(obs_b):
        return obs_b["agent_pos"][:, 0, 0].float()

    fill_buffer_next_values(buffer, critic, chunk_size=4)

    assert buffer.next_values[0] == 3.0, (
        f"A0.next_value = {buffer.next_values[0]} — expected "
        "critic(A0.next_obs)=3.0; the successor-value shortcut reused "
        "A1.value_old=10.0 instead of forwarding over A0's own next_obs"
    )
    assert buffer.next_values[0] != 10.0, (
        "A0.next_value leaked A1.value_old (10.0) — successor shortcut"
    )


def test_evaluate_transition_next_values_api_exists() -> None:
    """R7-4 introduces the per-transition critic-based evaluator.

    ``evaluate_transition_next_values(transitions, critic, chunk_size)``
    must exist on the rollout buffer module.
    """
    import importlib

    mod = importlib.import_module(
        "diffusion_policy_3d.rl.dppo_uav.rollout_buffer"
    )
    assert hasattr(mod, "evaluate_transition_next_values"), (
        "evaluate_transition_next_values missing — per-transition next"
        " value evaluation is not implemented"
    )
