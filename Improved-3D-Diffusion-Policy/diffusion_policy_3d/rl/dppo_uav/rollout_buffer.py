"""Rollout buffer (design §4.11).

RolloutTransition mirrors the §4.11 dataclass; RolloutBuffer enforces the
6 invariants at add() time:

1. len(obs) == len(next_obs) == len(reward) == len(terminated) (all lists)
2. episode_step strictly increases within episode
3. next_obs[t] equals obs[t+1] when no reset
4. chains_policy.shape == (K_f+1, H_a, 11)
5. old_logprobs.shape == (K_f, H_a, 11)
6. all observation_valid == True for trainable transitions

[JUDGMENT] next_obs is written in the same add() call — there is no
cross-iteration prev-frame auto-fill parameter (the old batched path
risk, train script 611-659, is closed by API shape).
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract


@dataclass(frozen=True)
class RolloutTransition:
    """One environment transition (design §4.11).

    Every transition carries BOTH hashes — manifest_entry_hash (the
    task spec's manifest source hash) and episode_config_hash (the sampled
    runtime episode hash from the env's reset info) — so the buffer trace
    is bound to the manifest and to the exact episode it was rolled out on.
    """

    obs: Dict[str, np.ndarray]  # latest T_o raw/normalized frames
    next_obs: Dict[str, np.ndarray]
    chains_policy: np.ndarray  # (K_f+1, H_a, 11) denoising chain samples
    old_logprobs: np.ndarray  # (K_f, H_a, 11)
    reward: float
    terminated: bool
    truncated: bool
    value_old: float
    episode_id: str
    episode_step: int
    world_uid: str
    role: str
    manifest_entry_hash: str
    episode_config_hash: str
    observation_valid: bool
    env_id: int = 0  # the env slot this transition was rolled out on
    # (R5.8-B: explicit per-slot axis for vector-env GAE time-series)
    # R5.8-C (N5/N9): outcome-derived truth per transition — the
    # TERMINAL transition of an episode carries the episode's real
    # outcome (success/collision/done_reason from the env's
    # EpisodeOutcome); non-terminal transitions are all-False/"none".
    done_reason: str = "none"
    success: bool = False
    collision: bool = False
    # R6-C (§11.4.2): the transition whose truncated=True is only the
    # GAE bootstrap of a budget-bounded ACTIVE slot (train_loop R5.3
    # flip). A rollout cut NEVER completes the episode — distinct from
    # a loop-level max_ep_len truncation, which does. Tracked
    # explicitly so the statistic layer can tell them apart without
    # re-reading truncated+done_reason (both are "none").
    rollout_cut: bool = False
    # R7-5 dual hashes: source_manifest_hash (manifest identity) +
    # runtime_episode_hash (env's ACTUAL post-reset state). None on mock
    # paths (add() skips the hard-fail); populated transitions are
    # hard-checked in add(): empty / non-64-hex / source !=
    # manifest_entry_hash / runtime != episode_config_hash all raise.
    source_manifest_hash: Optional[str] = None
    runtime_episode_hash: Optional[str] = None


def _is_64_hex(value: object) -> bool:
    """True iff value is a lowercase 64-char hex string (canonical_sha256
    digest format used for both dual hashes)."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


class RolloutBuffer:
    """Flat transition buffer with add()-time invariant enforcement."""

    def __init__(self, contract: RLSystemContract) -> None:
        self.contract = contract
        self.obs: List[Dict[str, np.ndarray]] = []
        self.next_obs: List[Dict[str, np.ndarray]] = []
        self.chains: List[np.ndarray] = []
        self.logprobs: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminated: List[bool] = []
        self.truncated: List[bool] = []
        self.values: List[float] = []
        self.episode_ids: List[str] = []
        self.episode_steps: List[int] = []
        self.world_uids: List[str] = []
        self.roles: List[str] = []
        self.manifest_entry_hashes: List[str] = []
        self.episode_config_hashes: List[str] = []
        # R7-5 dual hashes: per-transition copies of
        # RolloutTransition.source_manifest_hash / runtime_episode_hash.
        # None on mock paths; populated transitions were hard-checked in
        # add() (format + equality with the recorded hashes above).
        self.source_manifest_hashes: List[Optional[str]] = []
        self.runtime_episode_hashes: List[Optional[str]] = []
        self.env_ids: List[int] = []
        self.done_reasons: List[str] = []
        self.successes: List[bool] = []
        self.collisions: List[bool] = []
        # R6-C (§11.4.2): per-transition rollout-cut marker (see
        # RolloutTransition.rollout_cut) — statistics must distinguish a
        # budget cut from a loop-level truncation.
        self.rollout_cut: List[bool] = []
        # R5.8-B (B-1): per-transition NEXT values from the ROLLOUT
        # critic — post-filled by fill_buffer_next_values() after
        # collection (None placeholders until then; a gap is a bug).
        self.next_values: List[Optional[float]] = []
        # R7-4: provenance record of the last fill (source, counts,
        # resolved chunk size, obs/value hashes, episode groups) —
        # persisted in update metrics / checkpoints.
        self.next_value_provenance: Optional[Dict[str, Any]] = None
        self._last_step: Dict[str, int] = {}  # episode_id -> last step

    def add(self, t: RolloutTransition) -> None:
        """Append one transition, enforcing the §4.11 invariants."""
        # invariants 4+5: chain and logprob shapes
        k_f = self.contract.ft_denoising_steps
        h_a = self.contract.action_horizon
        d_a = self.contract.action_dim
        assert t.chains_policy.shape == (k_f + 1, h_a, d_a), (
            f"chains_policy must be ({k_f + 1}, {h_a}, {d_a}), got"
            f" {t.chains_policy.shape}"
        )
        assert t.old_logprobs.shape == (k_f, h_a, d_a), (
            f"old_logprobs must be ({k_f}, {h_a}, {d_a}), got"
            f" {t.old_logprobs.shape}"
        )
        # obs/next_obs key sets must agree
        assert set(t.obs) == set(t.next_obs), (
            f"obs keys {set(t.obs)} != next_obs keys {set(t.next_obs)}"
        )
        # invariant 6: trainable transitions are observation-valid only
        assert t.observation_valid, (
            "invalid observation transitions never enter the PPO buffer"
            " (§4.7 rule 2 — the gate routes them away before add())"
        )
        # invariant 2: episode_step strictly increases within episode
        last = self._last_step.get(t.episode_id)
        if last is not None:
            assert t.episode_step > last, (
                f"episode {t.episode_id}: step {t.episode_step} must be >"
                f" previous {last}"
            )
        # invariant 3: next_obs[t-1] equals obs[t] when no reset
        if self.episode_ids and self.episode_ids[-1] == t.episode_id:
            prev_next = self.next_obs[-1]
            for key in prev_next:
                assert np.array_equal(prev_next[key], t.obs[key]), (
                    f"invariant 3 broken at {t.episode_id} step"
                    f" {t.episode_step}: next_obs[{len(self.obs) - 1}][{key}]"
                    " != obs[current]"
                )
        # R7-5 hard-fail: dual hashes, when populated, must be present
        # together, well-formed 64-hex, and identical to the transition's
        # recorded manifest_entry_hash / episode_config_hash. The buffer
        # trace is bound to the manifest and the exact runtime episode —
        # any gap or mismatch is a provenance corruption, not a warning.
        if (
            t.source_manifest_hash is not None
            or t.runtime_episode_hash is not None
        ):
            if (
                t.source_manifest_hash is None
                or t.runtime_episode_hash is None
            ):
                raise ValueError(
                    "source_manifest_hash and runtime_episode_hash must be"
                    " populated together (R7-5 hard fail)"
                )
            if not _is_64_hex(t.source_manifest_hash):
                raise ValueError(
                    "source_manifest_hash must be 64 lowercase hex chars,"
                    f" got {t.source_manifest_hash!r} (R7-5 hard fail)"
                )
            if not _is_64_hex(t.runtime_episode_hash):
                raise ValueError(
                    "runtime_episode_hash must be 64 lowercase hex chars,"
                    f" got {t.runtime_episode_hash!r} (R7-5 hard fail)"
                )
            if t.source_manifest_hash != t.manifest_entry_hash:
                raise ValueError(
                    "source_manifest_hash"
                    f" {t.source_manifest_hash[:16]} != manifest_entry_hash"
                    f" {t.manifest_entry_hash[:16]} (R7-5 hard fail)"
                )
            if t.runtime_episode_hash != t.episode_config_hash:
                raise ValueError(
                    "runtime_episode_hash"
                    f" {t.runtime_episode_hash[:16]} != episode_config_hash"
                    f" {t.episode_config_hash[:16]} (R7-5 hard fail)"
                )
        self.obs.append(t.obs)
        self.next_obs.append(t.next_obs)
        self.chains.append(t.chains_policy)
        self.logprobs.append(t.old_logprobs)
        self.rewards.append(t.reward)
        self.terminated.append(t.terminated)
        self.truncated.append(t.truncated)
        self.values.append(t.value_old)
        self.episode_ids.append(t.episode_id)
        self.episode_steps.append(t.episode_step)
        self.world_uids.append(t.world_uid)
        self.roles.append(t.role)
        self.manifest_entry_hashes.append(t.manifest_entry_hash)
        self.episode_config_hashes.append(t.episode_config_hash)
        self.source_manifest_hashes.append(t.source_manifest_hash)
        self.runtime_episode_hashes.append(t.runtime_episode_hash)
        self.env_ids.append(t.env_id)
        self.done_reasons.append(t.done_reason)
        self.successes.append(t.success)
        self.collisions.append(t.collision)
        self.rollout_cut.append(t.rollout_cut)
        self.next_values.append(None)  # post-filled after collection
        self._last_step[t.episode_id] = t.episode_step
        # invariant 1: equal lengths hold by construction; assert once
        assert len(self.obs) == len(self.next_obs) == len(self.rewards) == len(
            self.terminated
        ) == len(self.truncated) == len(self.next_values)

    def clear(self) -> None:
        self.obs.clear()
        self.next_obs.clear()
        self.chains.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.values.clear()
        self.episode_ids.clear()
        self.episode_steps.clear()
        self.world_uids.clear()
        self.roles.clear()
        self.manifest_entry_hashes.clear()
        self.episode_config_hashes.clear()
        self.source_manifest_hashes.clear()
        self.runtime_episode_hashes.clear()
        self.env_ids.clear()
        self.done_reasons.clear()
        self.successes.clear()
        self.collisions.clear()
        self.rollout_cut.clear()
        self.next_values.clear()
        self.next_value_provenance = None
        self._last_step.clear()

    def __len__(self) -> int:
        return len(self.obs)


def _stack_next_obs(
    obs_list: List[Dict[str, np.ndarray]],
    device: str,
) -> Dict[str, torch.Tensor]:
    """Stack per-transition next_obs dicts into per-key tensors (N, ...).

    Mirrors ``trainer._stack_obs`` exactly (same np.stack -> float ->
    device collate) so the chunked fill produces the same per-key
    tensors the update-time obs stack produces.
    """
    first_keys = set(obs_list[0])
    for i, t in enumerate(obs_list):
        if set(t) != first_keys:
            raise ValueError(
                f"next_obs key mismatch at transition {i}: "
                f"{set(t)} != {first_keys}"
            )
    return {
        key: torch.from_numpy(np.stack([t[key] for t in obs_list]))
        .float()
        .to(device)
        for key in obs_list[0]
    }


def _next_obs_hash(
    next_obs_seq: List[Dict[str, np.ndarray]],
) -> str:
    """First 16 hex chars of sha256 over per-key stacked bytes.

    Keys hashed in fixed sorted order — deterministic regardless of
    dict ordering.
    """
    h = hashlib.sha256()
    for key in sorted(next_obs_seq[0]):
        stacked = np.stack([t[key] for t in next_obs_seq])
        h.update(np.ascontiguousarray(stacked).tobytes())
    return h.hexdigest()[:16]


def evaluate_transition_next_values(
    next_obs_seq: List[Dict[str, np.ndarray]],
    critic,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """R7-4: chunked critic forward over EACH transition's own next_obs.

    Returns the (N,) FP32 value tensor on the critic's parameter device
    (CPU for plain callables). Transitions are evaluated in fixed
    contiguous order in chunks of ``chunk_size`` (None = single
    forward — the small-fixture path). ALL transitions are evaluated,
    terminated finals included: GAE's bootstrap_mask decides their
    contribution, not the fill.

    Raises:
        ValueError: empty sequence, chunk_size <= 0, key mismatch,
            critic output count mismatch.
        FloatingPointError: non-finite critic output.
    """
    if not next_obs_seq:
        raise ValueError("evaluate_transition_next_values on empty sequence")
    n = len(next_obs_seq)
    for i, t in enumerate(next_obs_seq):
        if t is None:
            raise ValueError(f"next_obs missing at transition {i}")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be None or > 0, got {chunk_size}"
        )
    try:
        device = str(next(critic.parameters()).device)
    except (AttributeError, StopIteration):
        device = "cpu"  # plain callables (test fixtures)
    resolved = n if chunk_size is None else min(chunk_size, n)
    out = torch.empty(n, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, n, resolved):
            stop = min(start + resolved, n)
            batch = _stack_next_obs(next_obs_seq[start:stop], device)
            flat = critic(batch).reshape(-1)
            if flat.shape[0] != stop - start:
                raise ValueError(
                    f"critic returned {flat.shape[0]} values for chunk"
                    f" [{start}:{stop}] of {stop - start} transitions"
                )
            out[start:stop] = flat
    if not torch.isfinite(out).all():
        raise FloatingPointError(
            "critic produced NaN/Inf next values (R7-4 hard fail)"
        )
    return out


def fill_buffer_next_values(
    buffer: "RolloutBuffer",
    critic,
    chunk_size: Optional[int] = 64,
) -> None:
    """R7-4: every transition's next value from ITS OWN next_obs.

    The successor-value shortcut (``next_values[t] = values[t+1]`` —
    wrong for interleaved vector-env buffers and for any buffer where
    next_obs[t] != obs[t+1]) is deleted. Each transition gets exactly
    one critic forward over its own next_obs, chunked in contiguous
    transition order via ``evaluate_transition_next_values``; GAE's
    dual-mask semantics (bootstrap_mask = 1 - terminated; continuation
    = same episode AND not terminated AND not truncated) are unchanged
    and keep the terminated finals' values out of the bootstrap. Still
    runs before any optimizer step.

    ``chunk_size`` bounds peak memory on real rollouts (~900
    transitions of three-frame point clouds); None = single forward.

    Provenance is recorded on ``buffer.next_value_provenance``; a
    count gap or non-finite value raises.
    """
    n = len(buffer)
    values = evaluate_transition_next_values(
        buffer.next_obs, critic, chunk_size=chunk_size
    )
    if values.shape[0] != n:
        raise RuntimeError(
            f"fill count mismatch: {values.shape[0]} values for {n}"
            " transitions (R7-4 hard fail)"
        )
    next_values = [float(values[i].item()) for i in range(n)]
    # Episode groups: distinct episodes in the flat (possibly
    # interleaved) buffer — the provenance's interleaved-episode count.
    groups: Dict[str, List[int]] = {}
    for i in range(n):
        groups.setdefault(buffer.episode_ids[i], []).append(i)
    resolved = n if chunk_size is None else min(chunk_size, n)
    buffer.next_values[:] = next_values
    buffer.next_value_provenance = {
        "next_value_source": "critic(each_transition.next_obs)",
        "next_obs_count": n,
        "next_value_count": n,
        "chunk_size": resolved,
        "next_obs_hash": _next_obs_hash(buffer.next_obs),
        "next_value_hash": hashlib.sha256(
            np.ascontiguousarray(
                np.asarray(next_values, dtype=np.float64)
            ).tobytes()
        ).hexdigest()[:16],
        "interleaved_episode_count": len(groups),
    }
