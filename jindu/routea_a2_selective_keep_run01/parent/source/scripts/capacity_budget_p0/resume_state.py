#!/usr/bin/env python3
"""resume_state.py — Full-state checkpoint + resume audit for P0 capacity budget.

P0 Task 4: Full-state checkpoint audit with field-by-field verification,
round-trip determinism test, and old-checkpoint residue audit.

Outputs ``resume_state_audit.json`` (sub-report for the P0 aggregator).

Exports
-------
FullStateCheckpoint
    Save/load utility with required-field validation.  Backward-compatible
    READING of the existing a2_actor_best.pt / a2_critic_best.pt format.

run_resume_state_audit(config) -> dict
    Main entry point: audits old checkpoint, runs round-trip, writes JSON.

Full-state field checklist (§4.2)
----------------------------------
actor; online twin critic; target twin critic; actor optimizer; critic
optimizer; replay content + role/age/frontier metadata; Python RNG; NumPy RNG;
PyTorch CPU RNG; PyTorch CUDA RNG; actor/critic/phase counters;
reward-normalization state (or immutable hash).
"""

import copy
import json
import math
import os
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.capacity_budget_p0 import SCHEMA_VERSION, GATE_PASS_KEY

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

BATCH_SIZE = 32  # small for CPU round-trip speed
ACTOR_LR = 1e-5
CRITIC_LR = 1e-5
N_REPLAY_SUBSET = 100  # number of transitions to load for round-trip
GAMMA = 0.99

# §4.2 full-state field checklist (every field is critical for continuation)
FULL_STATE_FIELDS: List[str] = [
    "actor_state_dict",
    "critic_state_dict",
    "target_critic_state_dict",
    "actor_optimizer_state_dict",
    "critic_optimizer_state_dict",
    "replay_content",
    "replay_role_metadata",
    "replay_age_metadata",
    "replay_frontier_metadata",
    "python_rng_state",
    "numpy_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_state",
    "actor_update_counter",
    "critic_update_counter",
    "phase_counter",
    "reward_norm_state",
]

CRITICAL_FIELDS: set = set(FULL_STATE_FIELDS)

DECISION_FULL_STATE_CONTINUE = "FULL_STATE_CONTINUE"
DECISION_RESTART_FROM_COMMON_B1 = "RESTART_FROM_COMMON_B1"

# Path to the real B0 replay for loading transition data
_B0_REPLAY_PATH = (
    "/home/cjj/AUV/program/shiyan/results_routeA_RL_b0b1_rerun_v13_20260710/"
    "b0_collect_rolefix/replay.pt"
)
_OLD_CKPT_DIR = (
    "/home/cjj/AUV/program/shiyan/results_routeA_RL_B10_rank16_200upd_20260708/"
    "b10_rank16_200upd_seed20260824"
)
_REWARD_NORM_PATH = (
    "/home/cjj/AUV/program/shiyan/results_routeA_RL_b0b1_rerun_v13_20260710/"
    "reward_norm_stats.json"
)


# ──────────────────────────────────────────────────────────────────────
# Surrogate actor for round-trip (small MLP, not the real FM policy)
# ──────────────────────────────────────────────────────────────────────


class SurrogateActor(nn.Module):
    """Minimal deterministic actor for round-trip save/load mechanics.

    Takes (B, 1536) global_cond and produces (B, 8, 11) action prediction.
    Documented choice: this is a **training-step surrogate** — the round-trip
    tests the full-state save/load of all §4.2 fields including model weights,
    optimizer states, and gradient computation, not the FM policy's specific
    architecture.
    """

    def __init__(self, state_dim: int = 1536, hidden: int = 256,
                 horizon: int = 8, action_dim: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * action_dim),
        )
        self.horizon = horizon
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        out = self.net(state)
        return out.view(B, self.horizon, self.action_dim)


# ──────────────────────────────────────────────────────────────────────
# FullStateCheckpoint — save / load with validation
# ──────────────────────────────────────────────────────────────────────


class FullStateCheckpoint:
    """Full-state checkpoint with required-field validation.

    Designed as a standalone utility that does NOT modify the training
    script's existing ``CheckpointWriter``.

    Key behaviours
    --------------
    * ``save(path, full_state_dict)`` — writes the full state as a
      single ``torch.save`` dict.
    * ``load(path, require_continue=False)`` — reads a checkpoint dict and
      returns ``(state, report)`` where *report* contains field-by-field
      presence and the resume decision.
    * When *require_continue* is ``True`` and ANY critical field is absent,
      the decision is ``RESTART_FROM_COMMON_B1``.
    * Loads old-format a2_actor_best.pt / a2_critic_best.pt checkpoints
      without error (missing fields are recorded as absent).
    """

    SAVE_FILENAME = "full_state_ckpt_roundtrip.pt"

    @staticmethod
    def save(path: Path, state: dict) -> None:
        """Save a full-state checkpoint dict to *path*.

        Parameters
        ----------
        path : Path
            Destination file path.
        state : dict
            Must contain at minimum the SCHEMA_VERSION and run_id.
        """
        state["_schema_version"] = state.get("_schema_version", SCHEMA_VERSION)
        # Serialize replay transitions (RouteATransition objects) via pickle
        # so they survive the torch.save round-trip
        torch.save(state, path)

    @staticmethod
    def load(path: Path, require_continue: bool = False) -> Tuple[dict, dict]:
        """Load a checkpoint and produce a field-by-field audit report.

        Parameters
        ----------
        path : Path
            Path to the checkpoint file.
        require_continue : bool
            When True, the report's *decision* is ``RESTART_FROM_COMMON_B1``
            if ANY field in CRITICAL_FIELDS is absent.

        Returns
        -------
        state : dict
            The full loaded dict.  Missing critical fields are set to None.
        report : dict
            Field-level audit with keys: ``present_fields``, ``missing_critical``,
            ``decision``, ``decision_rationale``.
        """
        if not path.is_file():
            return {}, {
                "present_fields": [],
                "missing_critical": list(CRITICAL_FIELDS),
                "decision": DECISION_RESTART_FROM_COMMON_B1,
                "decision_rationale": f"File not found: {path}",
            }

        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            return {}, {
                "present_fields": [],
                "missing_critical": list(CRITICAL_FIELDS),
                "decision": DECISION_RESTART_FROM_COMMON_B1,
                "decision_rationale": f"Unloadable: {exc}",
            }

        if not isinstance(state, dict):
            return {}, {
                "present_fields": [],
                "missing_critical": list(CRITICAL_FIELDS),
                "decision": DECISION_RESTART_FROM_COMMON_B1,
                "decision_rationale": "Loaded object is not a dict",
            }

        present_fields: List[str] = []
        missing_critical: List[str] = []

        for field in FULL_STATE_FIELDS:
            if field in state and state[field] is not None:
                present_fields.append(field)
            else:
                missing_critical.append(field)

        if require_continue and missing_critical:
            decision = DECISION_RESTART_FROM_COMMON_B1
            rationale = (
                f"Missing {len(missing_critical)} critical field(s): "
                f"{missing_critical}"
            )
        elif require_continue:
            decision = DECISION_FULL_STATE_CONTINUE
            rationale = f"All {len(FULL_STATE_FIELDS)} §4.2 fields present."
        else:
            decision = None
            rationale = "Informational load (require_continue=False)."

        report = {
            "present_fields": present_fields,
            "missing_critical": missing_critical,
            "decision": decision,
            "decision_rationale": rationale,
        }

        return state, report

    @staticmethod
    def audit_old_format(actor_path: Path, critic_path: Path) -> dict:
        """Audit old-format a2_actor_best.pt / a2_critic_best.pt checkpoints.

        Returns a field-by-field audit dict without fabricating missing fields.
        """
        # ── Load actor ──
        actor_state = {}
        actor_fields: List[dict] = []
        if actor_path.is_file():
            try:
                ad = torch.load(actor_path, map_location="cpu", weights_only=False)
                if isinstance(ad, dict):
                    actor_state = ad
            except Exception:
                pass

        # Define fields for the actor checkpoint
        for field in FULL_STATE_FIELDS:
            present = field in actor_state and actor_state[field] is not None
            evidence = ""
            if field == "actor_state_dict":
                evidence = "Top-level key in a2_actor_best.pt" if present else "Absent from a2_actor_best.pt"
            elif field in ("critic_state_dict", "target_critic_state_dict",
                           "critic_optimizer_state_dict"):
                evidence = "Not stored in actor checkpoint"
            elif field == "actor_optimizer_state_dict":
                evidence = "a2_actor_best.pt only has state_dict, no optimizer" if not present else "Present"
            elif field in ("replay_content", "replay_role_metadata",
                           "replay_age_metadata", "replay_frontier_metadata"):
                evidence = "Never saved by train_a2_10k_capability.py CheckpointWriter"
            elif "rng" in field:
                evidence = "RNG state never captured by CheckpointWriter"
            elif field in ("actor_update_counter", "critic_update_counter",
                           "phase_counter"):
                evidence = "Counters never saved by CheckpointWriter"
            elif field == "reward_norm_state":
                evidence = "Reward norm stored separately in reward_norm_stats.json, not in checkpoint"
            else:
                evidence = "Present" if present else "Absent"
            actor_fields.append({
                "field": field,
                "present": present,
                "evidence": evidence,
            })

        # ── Load critic ──
        critic_fields: List[dict] = []
        if critic_path.is_file():
            try:
                cd = torch.load(critic_path, map_location="cpu", weights_only=False)
            except Exception:
                cd = {}
        else:
            cd = {}

        for field in FULL_STATE_FIELDS:
            present = field in cd and cd[field] is not None
            evidence = ""
            if field == "critic_state_dict":
                evidence = "Top-level key in a2_critic_best.pt" if present else "Absent from a2_critic_best.pt"
            elif field == "target_critic_state_dict":
                evidence = "Top-level key in a2_critic_best.pt" if present else "Absent from a2_critic_best.pt"
            elif field == "actor_state_dict":
                evidence = "Not stored in critic checkpoint"
            elif field == "actor_optimizer_state_dict":
                evidence = "Not stored in critic checkpoint"
            elif field == "critic_optimizer_state_dict":
                evidence = "a2_critic_best.pt only has state_dict, no optimizer" if not present else "Present"
            elif field in ("replay_content", "replay_role_metadata",
                           "replay_age_metadata", "replay_frontier_metadata"):
                evidence = "Never saved by train_a2_10k_capability.py CheckpointWriter"
            elif "rng" in field:
                evidence = "RNG state never captured by CheckpointWriter"
            elif field in ("actor_update_counter", "critic_update_counter",
                           "phase_counter"):
                evidence = "Counters never saved by CheckpointWriter"
            elif field == "reward_norm_state":
                evidence = "Reward norm stored separately in reward_norm_stats.json, not in checkpoint"
            else:
                evidence = "Present" if present else "Absent"
            critic_fields.append({
                "field": field,
                "present": present,
                "evidence": evidence,
            })

        # ── Combined verdict ──
        # Fields present if they exist in either checkpoint
        combined_present = set()
        for f_list in [actor_fields, critic_fields]:
            for f in f_list:
                if f["present"]:
                    combined_present.add(f["field"])

        actor_has_actor_sd = any(f["present"] and f["field"] == "actor_state_dict" for f in actor_fields)
        critic_has_critic_sd = any(f["present"] and f["field"] == "critic_state_dict" for f in critic_fields)
        critic_has_target_sd = any(f["present"] and f["field"] == "target_critic_state_dict" for f in critic_fields)
        has_any_optimizer = any("optimizer" in f["field"] and f["present"] for f in actor_fields + critic_fields)

        missing_critical = [f for f in FULL_STATE_FIELDS if f not in combined_present]

        # Decision: missing ANY critical field → RESTART_FROM_COMMON_B1
        if missing_critical:
            decision = DECISION_RESTART_FROM_COMMON_B1
            rationale = (
                f"Old checkpoint format missing {len(missing_critical)} critical "
                f"field(s): {missing_critical}. Present fields: "
                f"actor_state_dict={actor_has_actor_sd}, "
                f"critic_state_dict={critic_has_critic_sd}, "
                f"target_critic_state_dict={critic_has_target_sd}, "
                f"any_optimizer={has_any_optimizer}"
            )
        else:
            decision = DECISION_FULL_STATE_CONTINUE
            rationale = "All §4.2 fields present (unexpected for old format)"

        # Build the combined field table
        combined_field_table = {}
        for field in FULL_STATE_FIELDS:
            a_present = any(f["present"] and f["field"] == field for f in actor_fields)
            c_present = any(f["present"] and f["field"] == field for f in critic_fields)
            # Pick evidence from the checkpoint where the field IS present;
            # fall back to the actor entry if neither has it.
            evidence = ""
            for src in [actor_fields, critic_fields]:
                for f in src:
                    if f["field"] == field and f["present"]:
                        evidence = f["evidence"]
                        break
                if evidence:
                    break
            if not evidence:
                evidence = next(
                    (f["evidence"] for f in actor_fields + critic_fields
                     if f["field"] == field),
                    "",
                )
            combined_field_table[field] = {
                "present": a_present or c_present,
                "in_actor": a_present,
                "in_critic": c_present,
                "evidence": evidence,
            }

        return {
            "actor_path": str(actor_path),
            "critic_path": str(critic_path),
            "fields": combined_field_table,
            "missing_critical": missing_critical,
            "resume_decision": decision,
            "rationale": rationale,
        }


# ──────────────────────────────────────────────────────────────────────
# Round-trip harness
# ──────────────────────────────────────────────────────────────────────


def _capture_rng() -> dict:
    """Capture all four RNG states (CUDA guard)."""
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng(rng_state: dict) -> None:
    """Restore all four RNG states from a captured dict."""
    if "python_rng_state" in rng_state:
        random.setstate(rng_state["python_rng_state"])
    if "numpy_rng_state" in rng_state:
        np.random.set_state(rng_state["numpy_rng_state"])
    if "torch_cpu_rng_state" in rng_state:
        torch.set_rng_state(rng_state["torch_cpu_rng_state"])
    cuda_state = rng_state.get("torch_cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _load_replay_subset(path: str, n: int = N_REPLAY_SUBSET) -> list:
    """Load the first *n* transitions from a saved replay.pt file.

    Returns a list of RouteATransition objects with all metadata intact.
    """
    with open(path, "rb") as f:
        replay_data = pickle.load(f)
    transitions = replay_data["transitions"]
    return transitions[:n]


def _serialize_transition(trans) -> dict:
    """Serialize a RouteATransition to a plain dict for checkpoint storage."""
    d = {
        "obs_feature_t": trans.obs_feature_t,
        "action_chunk": trans.action_chunk,
        "action_chunk_tp1": trans.action_chunk_tp1,
        "reward_total": trans.reward_total,
        "reward_terms": trans.reward_terms,
        "obs_feature_tp1": trans.obs_feature_tp1,
        "done": trans.done,
        "role_active": trans.role_active,
        "replay_age_steps": trans.replay_age_steps,
        "frontier_score": trans.frontier_score,
    }
    return d


def _deserialize_transition(d: dict):
    """Reconstruct a RouteATransition from a plain dict."""
    from diffusion_policy_3d.rl.routea_replay_buffer import RouteATransition
    t = RouteATransition(
        obs_feature_t=d["obs_feature_t"],
        action_chunk=d["action_chunk"],
        action_chunk_tp1=d["action_chunk_tp1"],
        reward_total=d["reward_total"],
        reward_terms=d["reward_terms"],
        obs_feature_tp1=d["obs_feature_tp1"],
        done=d["done"],
    )
    # Restore metadata
    for key in ("role_active", "replay_age_steps", "frontier_score"):
        if key in d:
            setattr(t, key, d[key])
    return t


class RoundTripHarness:
    """Minimal but complete training-step surrogate for round-trip testing.

    Exercises all §4.2 fields in a single deterministic step:
      1. Sample replay batch (records indices)
      2. Compute actor FM-like loss (MSE between pred and target action)
      3. Compute twin critic TD loss
      4. Step both optimizers
      5. Increment counters
      6. Capture post-step RNG states
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.actor = SurrogateActor().to(device)
        self.actor.train()

        from diffusion_policy_3d.model.rl.chunk_critic import ChunkTwinCritic, hard_update
        self.critic = ChunkTwinCritic().to(device)
        self.target_critic = ChunkTwinCritic().to(device)
        hard_update(self.target_critic, self.critic)
        self.critic.train()
        self.target_critic.train()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay_transitions: list = []
        self.actor_update_counter = 0
        self.critic_update_counter = 0
        self.phase_counter = 0
        self.reward_norm_state = {"mean": 0.0, "std": 1.0}

    def load_replay_subset(self, path: str = _B0_REPLAY_PATH,
                           n: int = N_REPLAY_SUBSET) -> None:
        """Load transitions from a saved replay buffer."""
        self.replay_transitions = _load_replay_subset(path, n)

    def set_reward_norm(self, path: str = _REWARD_NORM_PATH) -> None:
        """Load reward normalization stats from JSON."""
        try:
            with open(path, "r") as f:
                self.reward_norm_state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def deterministic_step(self) -> dict:
        """Run one deterministic training step.

        Returns a dict with all tracking values for comparison.
        """
        from diffusion_policy_3d.model.rl.chunk_critic import compute_critic_loss
        from diffusion_policy_3d.rl.routea_replay_buffer import (
            collate_to_critic,
            collate_to_critic_device,
        )

        B = min(BATCH_SIZE, len(self.replay_transitions))
        indices = random.sample(range(len(self.replay_transitions)), B)
        sampled = [self.replay_transitions[i] for i in indices]
        raw_batch = collate_to_critic(sampled)
        batch = collate_to_critic_device(raw_batch, torch.device(self.device))

        # ── Actor forward / loss ──
        action_pred = self.actor(batch["state"])
        actor_loss = F.mse_loss(action_pred, batch["action"])

        # ── Critic forward / loss ──
        critic_loss, critic_log = compute_critic_loss(
            critic=self.critic,
            target_critic=self.target_critic,
            state=batch["state"],
            action=batch["action"],
            reward_total=batch["reward_total"],
            reward_heads=batch["reward_heads"],
            next_state=batch["next_state"],
            next_action=batch["action_tp1"],
            done=batch["done"],
            gamma=GAMMA,
        )

        # ── Step actor ──
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()
        self.actor_update_counter += 1

        # ── Step critic ──
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()
        self.critic_update_counter += 1

        # Increment phase counter each call (simplified)
        self.phase_counter += 1

        # Post-step RNG
        post_rng = _capture_rng()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "sample_indices": indices,
            "actor_update_counter": self.actor_update_counter,
            "critic_update_counter": self.critic_update_counter,
            "phase_counter": self.phase_counter,
            "post_rng": post_rng,
        }

    def capture_full_state(self) -> dict:
        """Produce a full-state dict with all §4.2 fields."""
        # Serialize transitions as plain dicts for robustness
        replay_dicts = [_serialize_transition(t) for t in self.replay_transitions]
        return {
            "_schema_version": SCHEMA_VERSION,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optim.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            "replay_content": replay_dicts,
            "replay_role_metadata": [t.role_active for t in self.replay_transitions],
            "replay_age_metadata": [t.replay_age_steps for t in self.replay_transitions],
            "replay_frontier_metadata": [t.frontier_score for t in self.replay_transitions],
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "actor_update_counter": self.actor_update_counter,
            "critic_update_counter": self.critic_update_counter,
            "phase_counter": self.phase_counter,
            "reward_norm_state": self.reward_norm_state,
        }

    def restore_full_state(self, state: dict) -> None:
        """Restore all §4.2 fields from a captured state dict."""
        _ = state.get("_schema_version", SCHEMA_VERSION)

        self.actor.load_state_dict(state["actor_state_dict"])
        self.critic.load_state_dict(state["critic_state_dict"])
        self.target_critic.load_state_dict(state["target_critic_state_dict"])
        # NOTE: do NOT hard_update here — the target critic must remain at its
        # SAVED value (which may differ from the online critic after multiple
        # Polyak updates).  A hard_update would defeat full-state resume.
        self.actor_optim.load_state_dict(state["actor_optimizer_state_dict"])
        self.critic_optim.load_state_dict(state["critic_optimizer_state_dict"])

        # Deserialize replay
        replay_dicts = state["replay_content"]
        self.replay_transitions = [_deserialize_transition(d) for d in replay_dicts]

        # Restore metadata (already in transitions, but verify)
        meta_role = state.get("replay_role_metadata", [])
        meta_age = state.get("replay_age_metadata", [])
        meta_frontier = state.get("replay_frontier_metadata", [])
        for i, t in enumerate(self.replay_transitions):
            if i < len(meta_role):
                t.role_active = meta_role[i]
            if i < len(meta_age):
                t.replay_age_steps = meta_age[i]
            if i < len(meta_frontier):
                t.frontier_score = meta_frontier[i]

        # Restore RNG
        _restore_rng(state)

        self.actor_update_counter = state.get("actor_update_counter", 0)
        self.critic_update_counter = state.get("critic_update_counter", 0)
        self.phase_counter = state.get("phase_counter", 0)
        self.reward_norm_state = state.get("reward_norm_state", {"mean": 0.0, "std": 1.0})


# ──────────────────────────────────────────────────────────────────────
# Entry point: run_resume_state_audit
# ──────────────────────────────────────────────────────────────────────


def run_resume_state_audit(config: dict) -> dict:
    """Run the full-state checkpoint + resume audit.

    Parameters
    ----------
    config : dict
        Must contain at least ``output_dir`` (str).  Optional keys:
        ``run_id`` (str), ``device`` (str, default ``'cpu'``).

    Returns
    -------
    dict
        The full sub-report dict (also written to
        ``<output_dir>/resume_state_audit.json``).
    """
    output_dir = Path(config["output_dir"])
    run_id = config.get("run_id", output_dir.name)
    device_str = config.get("device", "cpu")
    seed = config.get("seed", 42)

    # Resolve device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"

    print(f"[resume_state] device={device_str}, seed={seed}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # 1. Old checkpoint audit
    # ──────────────────────────────────────────────────────────────────
    old_ckpt_dir = Path(config.get("old_ckpt_dir", _OLD_CKPT_DIR))
    old_actor_path = old_ckpt_dir / "a2_actor_best.pt"
    old_critic_path = old_ckpt_dir / "a2_critic_best.pt"

    print(f"[resume_state] Auditing old checkpoints:")
    print(f"  Actor: {old_actor_path}")
    print(f"  Critic: {old_critic_path}")

    old_audit = FullStateCheckpoint.audit_old_format(old_actor_path, old_critic_path)
    old_decision = old_audit["resume_decision"]

    print(f"  Old-ckpt fields present: "
          f"{sum(1 for f in old_audit['fields'].values() if f['present'])}"
          f"/{len(FULL_STATE_FIELDS)}")
    print(f"  Missing critical: {old_audit['missing_critical']}")
    print(f"  Old-ckpt decision: {old_decision}")

    # ──────────────────────────────────────────────────────────────────
    # 2. Round-trip determinism test (full-state capable writer)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[resume_state] Running round-trip determinism test...")
    round_trip = {"passed": False, "actor_loss_match": False,
                  "critic_loss_match": False, "sample_indices_match": False,
                  "counters_match": False, "rng_match": False}

    # Build harness (CPU for determinism)
    harness = RoundTripHarness(device=device_str)
    harness.load_replay_subset()
    harness.set_reward_norm()

    # ── Path A: seed → save → step (reference) ──
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Save full state before step
    tmp_ckpt_path = output_dir / FullStateCheckpoint.SAVE_FILENAME
    pre_step_state = harness.capture_full_state()
    FullStateCheckpoint.save(tmp_ckpt_path, pre_step_state)

    # Step A (reference)
    result_a = harness.deterministic_step()

    # ── Path B: re-seed → load → step ──
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create a fresh harness, load full state
    harness_b = RoundTripHarness(device=device_str)
    loaded_state, load_report = FullStateCheckpoint.load(
        tmp_ckpt_path, require_continue=True
    )
    if load_report["decision"] == DECISION_RESTART_FROM_COMMON_B1:
        print(f"  [FAIL] Load rejected: {load_report['decision_rationale']}")
    else:
        harness_b.restore_full_state(loaded_state)

    # Step B
    result_b = harness_b.deterministic_step()

    # ── Compare ──
    # Loss match (tolerance for floating-point on CPU)
    atol = 1e-6
    actor_loss_match = math.isclose(result_a["actor_loss"],
                                    result_b["actor_loss"], rel_tol=atol, abs_tol=atol)
    critic_loss_match = math.isclose(result_a["critic_loss"],
                                     result_b["critic_loss"], rel_tol=atol, abs_tol=atol)
    sample_indices_match = result_a["sample_indices"] == result_b["sample_indices"]
    counters_match = (
        result_a["actor_update_counter"] == result_b["actor_update_counter"]
        and result_a["critic_update_counter"] == result_b["critic_update_counter"]
        and result_a["phase_counter"] == result_b["phase_counter"]
    )

    # RNG match: compare serialized forms
    rng_a = result_a["post_rng"]
    rng_b = result_b["post_rng"]
    rng_match = _compare_rng_states(rng_a, rng_b)

    round_trip = {
        "passed": actor_loss_match and critic_loss_match and sample_indices_match
                  and counters_match and rng_match,
        "actor_loss_match": actor_loss_match,
        "critic_loss_match": critic_loss_match,
        "sample_indices_match": sample_indices_match,
        "counters_match": counters_match,
        "rng_match": rng_match,
    }

    print(f"  Round-trip: {'PASS' if round_trip['passed'] else 'FAIL'}")
    print(f"    actor_loss_match: {actor_loss_match} "
          f"({result_a['actor_loss']:.8f} vs {result_b['actor_loss']:.8f})")
    print(f"    critic_loss_match: {critic_loss_match} "
          f"({result_a['critic_loss']:.8f} vs {result_b['critic_loss']:.8f})")
    print(f"    sample_indices_match: {sample_indices_match}")
    print(f"    counters_match: {counters_match}")
    print(f"    rng_match: {rng_match}")

    # ──────────────────────────────────────────────────────────────────
    # 3. Gate
    # ──────────────────────────────────────────────────────────────────
    # The resume audit gate PASSES when a full-state-capable writer
    # round-trips AND a unique verdict is produced — even if that verdict
    # is RESTART_FROM_COMMON_B1 (the expected honest outcome for old ckpts).
    # Gate pass = round_trip passed AND old_audit has exactly one decision.
    unique_decision_produced = old_decision in (DECISION_FULL_STATE_CONTINUE,
                                                DECISION_RESTART_FROM_COMMON_B1)
    gate_pass = round_trip["passed"] and unique_decision_produced

    # Cleanup
    if tmp_ckpt_path.exists():
        tmp_ckpt_path.unlink()

    # ──────────────────────────────────────────────────────────────────
    # 4. Build output
    # ──────────────────────────────────────────────────────────────────
    # Format old-ckpt field table
    field_table = []
    for field in FULL_STATE_FIELDS:
        info = old_audit["fields"].get(field, {"present": False, "evidence": ""})
        field_table.append({
            "field": field,
            "present": info["present"],
            "evidence": info.get("evidence", ""),
        })

    output = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        GATE_PASS_KEY: gate_pass,
        "device_used": device_str,
        "full_state_fields": FULL_STATE_FIELDS,
        "round_trip": round_trip,
        "old_checkpoint_audit": {
            "actor_path": str(old_actor_path),
            "critic_path": str(old_critic_path),
            "fields": field_table,
            "missing_critical": old_audit["missing_critical"],
        },
        "resume_decision": old_decision,
        "rationale": old_audit["rationale"],
        "notes": [
            "Full-state capable writer (resume_state.py RoundTripHarness) "
            "round-trips all 17 §4.2 fields.",
            "Old CheckpointWriter (train_a2_10k_capability.py) lacks: "
            "optimizer states, replay content + metadata, all RNG states, "
            "update counters, and reward-norm state.",
            "Surrogate actor (SurrogateActor MLP) used for round-trip; "
            "real ChunkTwinCritic used for critic. This exercises the "
            "full-state save/load of all fields including gradient "
            "computation.",
            "No fields were fabricated during old-checkpoint audit.",
            f"decision_rationale: {old_audit['rationale']}",
        ],
    }

    # ──────────────────────────────────────────────────────────────────
    # 5. Write output
    # ──────────────────────────────────────────────────────────────────
    output_path = output_dir / "resume_state_audit.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[resume_state] Written to {output_path}")

    print(f"\n{'=' * 60}")
    print(f"Resume State Audit Summary")
    print(f"{'=' * 60}")
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")
    print(f"  Round-trip: {'PASS' if round_trip['passed'] else 'FAIL'}")
    print(f"  Old-ckpt resume_decision: {old_decision}")
    print(f"  Missing critical fields: {old_audit['missing_critical']}")
    print(f"{'=' * 60}")

    return output


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _compare_rng_states(a: dict, b: dict) -> bool:
    """Compare two captured RNG state dicts for equality.

    Handles None CUDA states and numpy/Tensor comparison.
    """
    for key in ("python_rng_state", "numpy_rng_state", "torch_cpu_rng_state",
                "torch_cuda_rng_state"):
        sa = a.get(key)
        sb = b.get(key)
        if sa is None and sb is None:
            continue
        if sa is None or sb is None:
            return False
        if isinstance(sa, torch.Tensor) and isinstance(sb, torch.Tensor):
            if not torch.equal(sa, sb):
                return False
        elif isinstance(sa, tuple) and isinstance(sb, tuple):
            # numpy get_state returns tuple
            if len(sa) != len(sb):
                return False
            for ia, ib in zip(sa, sb):
                if isinstance(ia, np.ndarray) and isinstance(ib, np.ndarray):
                    if not np.array_equal(ia, ib):
                        return False
                elif ia != ib:
                    return False
        elif isinstance(sa, list) and isinstance(sb, list):
            # CUDA rng state is list of tensors
            if len(sa) != len(sb):
                return False
            for ta, tb in zip(sa, sb):
                if not torch.equal(ta, tb):
                    return False
        elif sa != sb:
            return False
    return True
