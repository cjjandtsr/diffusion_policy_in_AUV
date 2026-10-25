"""Episode-level paired evaluation artifact (design §4.17, RTM EVAL-AUDIT).

§4.17 requires per artifact:
  - episode-level BC/RL paired records, not only aggregates;
  - episode config, seed, world, role, start, goal, disturbance;
  - BC/RL success, collision, timeout, return, steps;
  - three capability metrics (anticipatory avoidance, full-space support
    expansion, dense-obstacle ceiling);
  - checkpoint/config/contract/manifest/code hash;
  - reward version and exec steps;
  - observation invalid count;
  - missing/unexpected weight keys;
  - LoRA parameter checksum;
  - evaluation timestamp and device.

Paired-eval extension (§4.17):
  - the frozen paired bank (PairedBankEntry + freeze/JSON round-trip):
    each entry carries world/role/start/goal/yaw/seed, both hashes,
    disturbance and admission metrics; the eval replays the bank exactly
    (no runtime task re-selection) and hard-fails when the runtime
    episode config hash differs from the frozen one;
  - per-episode per-step trace (raw series, never aggregate-only);
  - the unique canonical artifact filename + atomic no-overwrite write.

[JUDGMENT] formal eval MUST hard-fail on missing LoRA, critical key
missing, contract mismatch, or invalid point cloud — never warn-and-
continue into a PPO==BC result.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    CAPABILITY_METRIC_KEYS,
    FIVE_PIECE_KEYS,
    GROUP_OF_ROLE,
    CapabilityOpportunity,
    OfflineDensity,
    opportunity_from_dict,
    opportunity_to_dict,
)
from diffusion_policy_3d.rl.dppo_uav.checkpoint import EMPTY_LORA_CHECKSUM
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    ROLE_NAMES,
    SPLIT_NAMES,
    EpisodeTaskSpec,
    canonical_sha256,
)

CAPABILITY_METRICS = (
    "anticipatory_avoidance",
    "fullspace_support_expansion",
    "dense_obstacle_ceiling",
)


class EvalArtifactError(Exception):
    """Hard-fail: artifact cannot represent a valid paired evaluation."""


@dataclass(frozen=True)
class EpisodePairedRecord:
    """One episode's BC/RL paired outcome."""

    episode_config: Dict[str, Any]
    seed: int
    world: str
    role: str
    start: List[float]
    goal: List[float]
    disturbance: Dict[str, Any]
    bc: Dict[str, Any]  # success/collision/timeout/return/steps
    rl: Dict[str, Any]
    # the RAW per-step trace for both sides — {"bc": [...], "rl":
    # [...]}, each entry {step, elapsed, pos, z, yaw, goal_dist,
    # clearance, hit_count, action summary, done flags}. Never
    # aggregate-only.
    trace: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # R5.4 paired policy RNG provenance: the derived diffusion-noise seed
    # BOTH sides drew from, the replica index and the seeding contract.
    policy_noise_seed: int = 0
    replica_id: int = 0
    rng_contract_version: str = ""
    # R6-EVAL-1 (§11.5.1): the policy checkpoint that produced this
    # record, mirrored from artifact-level provenance so the pure-list
    # statistics layer can deduplicate BC records across RL artifacts
    # without losing which policy they came from (18/24 counterexample).
    policy_checkpoint_hash: str = ""
    # review partB P2-1: per-side policy sampler call counts (E3).
    sampler_call_counts: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedBankEntry:
    """One FROZEN eval episode: the bank IS the eval episode list.

    Everything needed to replay the episode exactly — spec identity,
    sampled geometry, seed, both hashes, disturbance and the admission
    metrics the record carried at freeze time. (spec, seed) determines
    the sampled config deterministically, so the eval-time freeze check
    (runtime episode_config_hash == entry hash) passes by construction
    and fails loudly on bank/manifest/world drift.
    """

    world_uid: str
    role: str
    split: str
    admission_id: str
    manifest_entry_hash: str
    obstacle_json: str
    generator_version: str
    seed: int
    start: List[float]
    goal: List[float]
    start_yaw: float
    episode_config_hash: str
    disturbance: Dict[str, Any]
    admission: Dict[str, Any]
    # R6-EVAL-2 (§11.5.3): the frozen capability opportunity for this
    # entry (nominal-path geometry / world topology / offline density —
    # policy-independent facts). None when the bank carries no
    # opportunity block (v3 banks stay readable, never silently
    # defaulted — the gate then reports NOT_INFORMATIVE).
    opportunity: Optional[CapabilityOpportunity] = None
    # Guide R7-8 Step 1: the trace contract the R7-9 BC replay (v2
    # rebuild) must produce — declared per record, defaulted to the
    # canonical per-step key set. The lambda defers the name lookup
    # past the dataclass definition (TRACE_STEP_REQUIRED_KEYS lives
    # below); missing keys in old banks fall back to the same constant
    # in _entries_from_dicts, so pre-R7-8 banks stay readable.
    required_trace_fields: List[str] = field(
        default_factory=lambda: list(TRACE_STEP_REQUIRED_KEYS)
    )

    def to_spec(self) -> EpisodeTaskSpec:
        """Reconstruct the reset spec from the frozen fields — fully
        deterministic, never derived from a live registry."""
        return EpisodeTaskSpec(
            world_uid=self.world_uid,
            obstacle_json=self.obstacle_json,
            role=self.role,
            split=self.split,
            admission_id=self.admission_id,
            generator_version=self.generator_version,
            manifest_entry_hash=self.manifest_entry_hash,
        )


# The frozen bank schema — every field is REQUIRED: the
# entry must be fully replayable and freeze-verifiable.
REQUIRED_BANK_ENTRY_FIELDS = (
    "world_uid", "role", "split", "admission_id", "manifest_entry_hash",
    "obstacle_json", "generator_version",
    "seed", "start", "goal", "start_yaw", "episode_config_hash",
    "disturbance", "admission",
    # Guide R7-8 Step 1: every frozen record declares the trace contract
    # the replay must produce — a bank without it is not admission-ready.
    "required_trace_fields",
)

BANK_VERSION = "3"  # bank_id binding added (R3 rework); was "2" pre-binding

# Every per-step trace entry must carry the full step observation — the
# trace is the RAW series, never aggregate-only. Keys match
# _eval_episode's per-step record in train_loop.py.
TRACE_STEP_REQUIRED_KEYS = (
    "step", "elapsed",
    "pos", "z", "yaw", "goal_dist", "clearance", "hit_count",
    "action_norm", "action_max_abs",
    "done_reason", "terminated", "truncated",
)

# The numeric trace axes the artifact validator checks for finiteness
# (P1-4: NaN/Inf in a required trace field is never a valid metric).
# done_reason/terminated/truncated are str/bool — excluded. "pos" is the
# 3-value position list, checked component-wise.
NUMERIC_TRACE_KEYS = (
    "step", "elapsed", "pos", "z", "yaw", "goal_dist", "clearance",
    "hit_count", "action_norm", "action_max_abs",
)


@dataclass(frozen=True)
class EvalArtifact:
    """Full §4.17 artifact (mutated via dataclasses.replace in tests only).

    history_indices (observation-selection metadata) is part of the schema.
    §7.4: use_obstacle_projection must be False — projection-enabled
    evaluation is invalid.
    bank_id + offline_density_sha256 (R3 rework, P1-3): the artifact
    records WHICH bank and WHICH offline density produced it — provenance
    missing from the artifact is a hard fail (the bank/density the numbers
    came from must be auditable).
    """

    episodes: List[EpisodePairedRecord]
    checkpoint_hash: str
    config_hash: str
    contract_hash: str
    manifest_hash: str
    code_hash: str
    reward_version: str
    exec_steps: int
    observation_invalid_count: int
    lora_checksum: str
    timestamp: str
    device: str
    capability_metrics: Dict[str, Dict[str, float]]
    history_indices: List[int]
    use_obstacle_projection: bool
    # R6-EVAL-2 (§11.5.3): the five-piece outcome contract
    # {opportunity_count, exposure_count, event_count, value|null,
    #  validity_reason} per capability group — the capability gate is
    # driven by opportunity presence, never by the presence of the
    # three-group numerics block alone. Schema-validated on build like
    # capability_metrics; empty is a hard fail (never a silent default).
    capability_outcomes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    missing_weight_keys: List[str] = field(default_factory=list)
    unexpected_weight_keys: List[str] = field(default_factory=list)
    bank_id: str = ""
    offline_density_sha256: str = ""
    # Gate D reopening (2026-08-22): the aggregate hash over the external
    # world assets (scene + obstacle files) the evaluated checkpoint
    # trained under — a paired artifact must be attributable to the
    # exact world files, never to a hand-copied claims string.
    world_assets_hash: str = ""
    # R5.9-C2: eval-side identity provenance. identity_mode is "strict"
    # (all 8 axes validated) or "commit-drift-explicit" (eval-only
    # escape hatch: 2 git axes skipped, source-file axis pinned to the
    # reviewed delta); source_file_delta lists the files whose runtime
    # hash differs from the checkpoint's training-tree claim;
    # project_git_commit is the ACTUAL runtime commit (never silently
    # re-attributing the checkpoint to a different tree).
    identity_mode: str = "strict"
    source_file_delta: List[str] = field(default_factory=list)
    project_git_commit: str = ""
    # R6-E §11.6.1 (F-06 → R6-LOAD-1): the load contract recorded at
    # checkpoint load time ({"bc_checkpoint_hash", "loader_schema_version",
    # "compatibility_manifest"}). REQUIRED whenever unexpected_weight_keys
    # is non-empty — the recorded unexpected keys must self-explain via
    # the contract (anchor-carrying registered BC checkpoint); an
    # artifact that cannot prove they are the allowlisted anchors rather
    # than poisoned extra keys is rejected.
    load_contract: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "episodes": [
                {
                    "episode_config": ep.episode_config,
                    "seed": ep.seed,
                    "world": ep.world,
                    "role": ep.role,
                    "start": ep.start,
                    "goal": ep.goal,
                    "disturbance": ep.disturbance,
                    "bc": ep.bc,
                    "rl": ep.rl,
                    # the raw per-step series travels with the record.
                    "trace": ep.trace,
                    "policy_noise_seed": ep.policy_noise_seed,
                    "replica_id": ep.replica_id,
                    "rng_contract_version": ep.rng_contract_version,
                    "policy_checkpoint_hash": ep.policy_checkpoint_hash,
                    "sampler_call_counts": ep.sampler_call_counts,
                }
                for ep in self.episodes
            ],
            "checkpoint_hash": self.checkpoint_hash,
            "config_hash": self.config_hash,
            "contract_hash": self.contract_hash,
            "manifest_hash": self.manifest_hash,
            "code_hash": self.code_hash,
            "reward_version": self.reward_version,
            "exec_steps": self.exec_steps,
            "observation_invalid_count": self.observation_invalid_count,
            "missing_weight_keys": self.missing_weight_keys,
            "unexpected_weight_keys": self.unexpected_weight_keys,
            "lora_checksum": self.lora_checksum,
            "timestamp": self.timestamp,
            "device": self.device,
            "capability_metrics": self.capability_metrics,
            "capability_outcomes": self.capability_outcomes,
            "history_indices": self.history_indices,
            "use_obstacle_projection": self.use_obstacle_projection,
            "bank_id": self.bank_id,
            "offline_density_sha256": self.offline_density_sha256,
            "world_assets_hash": self.world_assets_hash,
            "identity_mode": self.identity_mode,
            "source_file_delta": self.source_file_delta,
            "project_git_commit": self.project_git_commit,
            "load_contract": self.load_contract,
        }
        return d


def build_eval_artifact(
    episodes: List[EpisodePairedRecord],
    checkpoint_hash: str,
    config_hash: str,
    contract_hash: str,
    manifest_hash: str,
    world_assets_hash: str,
    code_hash: str,
    reward_version: str,
    exec_steps: int,
    observation_invalid_count: int,
    missing_weight_keys: List[str],
    unexpected_weight_keys: List[str],
    lora_checksum: str,
    timestamp: str,
    device: str,
    capability_metrics: Dict[str, Dict[str, float]],
    history_indices: List[int],
    use_obstacle_projection: bool,
    capability_outcomes: Optional[Dict[str, Dict[str, Any]]] = None,
    expected_contract_hash: str = "contract789",
    bank_id: str = "",
    offline_density: Optional[OfflineDensity] = None,
    identity_mode: str = "strict",
    source_file_delta: Optional[List[str]] = None,
    project_git_commit: str = "",
    load_contract: Optional[Dict[str, Any]] = None,
) -> EvalArtifact:
    """Construct and validate a §4.17 artifact (validates on build).

    expected_contract_hash = runtime RLSystemContract hash at the call site.
    bank_id + offline_density (R3 rework, P1-3): provenance recorded as
    the bank id and the canonical sha256 over the offline density's full
    content — any density byte change changes the recorded binding.
    identity_mode/source_file_delta/project_git_commit (R5.9-C2): the
    eval-side identity provenance — strict vs commit-drift-explicit, the
    observed source-file drift, and the ACTUAL runtime commit.
    """
    artifact = EvalArtifact(
        episodes=episodes,
        checkpoint_hash=checkpoint_hash,
        config_hash=config_hash,
        contract_hash=contract_hash,
        manifest_hash=manifest_hash,
        code_hash=code_hash,
        reward_version=reward_version,
        world_assets_hash=world_assets_hash,
        exec_steps=exec_steps,
        observation_invalid_count=observation_invalid_count,
        missing_weight_keys=missing_weight_keys,
        unexpected_weight_keys=unexpected_weight_keys,
        lora_checksum=lora_checksum,
        timestamp=timestamp,
        device=device,
        capability_metrics=capability_metrics,
        capability_outcomes=capability_outcomes or {},
        history_indices=history_indices,
        use_obstacle_projection=use_obstacle_projection,
        bank_id=bank_id,
        offline_density_sha256=(
            canonical_sha256(offline_density.to_dict())
            if offline_density is not None else ""
        ),
        identity_mode=identity_mode,
        source_file_delta=source_file_delta or [],
        project_git_commit=project_git_commit,
        load_contract=load_contract,
    )
    validate_eval_artifact(
        artifact,
        expected_exec_steps=exec_steps,
        expected_contract_hash=expected_contract_hash,
    )
    return artifact


def validate_eval_artifact(
    artifact: EvalArtifact,
    expected_exec_steps: int = 4,
    expected_contract_hash: str = "contract789",
) -> None:
    """§4.17 + [JUDGMENT] hard-fail rules. Raises EvalArtifactError.

    expected_contract_hash comes from the runtime RLSystemContract hash
    (contract serialization + hash, §4.16) at the eval call site.
    """
    if not artifact.episodes:
        raise EvalArtifactError("artifact has zero episodes — aggregate-only is §4.17-forbidden")
    if not artifact.lora_checksum:
        raise EvalArtifactError("LoRA parameter checksum missing — hard fail (§4.17)")
    # sha256(b"") is non-empty yet means NO LoRA parameters — the old
    # non-empty check let PPO==BC (empty adaptor set) through as a valid
    # result. The empty-LoRA case is rejected EXPLICITLY (P0-1).
    if artifact.lora_checksum == EMPTY_LORA_CHECKSUM:
        raise EvalArtifactError(
            "empty LoRA parameter set (checksum == sha256(b\"\")) — the "
            "BC baseline would be reported as the RL result; hard fail, "
            "never warn-and-continue into PPO==BC"
        )
    if not artifact.bank_id or not artifact.offline_density_sha256:
        raise EvalArtifactError(
            "bank/density provenance missing — the artifact cannot say "
            "which bank or which offline density produced it; not "
            "auditable, hard fail (P1-3)"
        )
    if artifact.missing_weight_keys:
        raise EvalArtifactError(
            "critical weight keys missing: "
            f"{artifact.missing_weight_keys} — hard fail (§4.17)"
        )
    # R6-E §11.6.1 (F-06 → R6-LOAD-1): unexpected weight keys must
    # self-explain via the load contract recorded at checkpoint load time
    # — the SAME function the loader uses. An artifact that records
    # unexpected keys without the contract cannot prove they are the
    # allowlisted keys rather than a poisoned extra key (schema v2:
    # allowlist is empty, so any unexpected key fails the exact-set
    # check inside validate_policy_load_contract).
    if artifact.unexpected_weight_keys:
        lc = artifact.load_contract
        if not lc or not isinstance(lc, dict):
            raise EvalArtifactError(
                "unexpected_weight_keys recorded without the §11.6.1 "
                "load_contract — they must self-explain via the frozen "
                "allowlist contract"
            )
        for key in ("bc_checkpoint_hash", "loader_schema_version",
                    "compatibility_manifest"):
            if key not in lc:
                raise EvalArtifactError(
                    f"load_contract missing key {key!r}"
                )
        from diffusion_policy_3d.rl.dppo_uav.policy_loader import (
            validate_policy_load_contract,
        )

        try:
            validate_policy_load_contract(
                bc_checkpoint_hash=lc["bc_checkpoint_hash"],
                missing_keys=[],
                unexpected_keys=artifact.unexpected_weight_keys,
                compatibility_manifest=lc["compatibility_manifest"],
            )
        except ValueError as e:
            raise EvalArtifactError(
                f"load contract invalid: {e}"
            ) from e
    if artifact.use_obstacle_projection:
        raise EvalArtifactError(
            "use_obstacle_projection=True (§7.4): obstacle projection is "
            "policy-external — formal eval hard-fails on True"
        )
    if not artifact.history_indices:
        raise EvalArtifactError(
            "history_indices missing (A5): observation-selection metadata "
            "is required"
        )
    if artifact.contract_hash != expected_contract_hash:
        raise EvalArtifactError(
            f"contract mismatch ({artifact.contract_hash} != "
            f"{expected_contract_hash}) — hard fail (§4.17)"
        )
    if artifact.observation_invalid_count:
        raise EvalArtifactError(
            "invalid point-cloud observations counted — hard fail (§4.17)"
        )
    if artifact.exec_steps != expected_exec_steps:
        raise EvalArtifactError(
            f"exec_steps {artifact.exec_steps} != expected {expected_exec_steps}"
        )
    if set(artifact.capability_metrics) != set(CAPABILITY_METRICS):
        raise EvalArtifactError(
            f"capability metrics must be exactly {CAPABILITY_METRICS}, got "
            f"{set(artifact.capability_metrics)}"
        )
    for group, required in CAPABILITY_METRIC_KEYS.items():
        group_metrics = artifact.capability_metrics[group]
        if not isinstance(group_metrics, dict) or not group_metrics:
            raise EvalArtifactError(
                f"capability metrics {group} must be a non-empty object"
            )
        missing = [key for key in required if key not in group_metrics]
        if missing:
            raise EvalArtifactError(
                f"capability metrics {group} missing keys: {missing}"
            )
        for key, value in group_metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvalArtifactError(
                    f"capability metrics {group}.{key} must be a number"
                )
            if not np.isfinite(float(value)):
                raise EvalArtifactError(
                    f"capability metrics {group}.{key} must be finite"
                )
        role = GROUP_OF_ROLE[group]
        expected_n = sum(
            1 for ep in artifact.episodes if ep.role == role
        )
        if group_metrics["n_episodes"] != expected_n:
            raise EvalArtifactError(
                f"capability metrics {group}.n_episodes "
                f"{group_metrics['n_episodes']} != {expected_n} episodes "
                f"with role {role!r} — role alignment"
            )
        for key in ("sr_bc", "sr_rl", "cr_bc", "cr_rl"):
            if not 0.0 <= group_metrics[key] <= 1.0:
                raise EvalArtifactError(
                    f"capability metrics {group}.{key} must be in [0, 1]"
                )
    # R6-EVAL-2 (§11.5.3): the five-piece outcome contract — REQUIRED
    # (an artifact without it is schema-invalid, never a silent default),
    # exactly the three groups, exact piece keys, honest null semantics
    # (event_count==0 -> value null + validity_reason; event_count>0 ->
    # finite value + no reason). Gate B: the capability gate must be
    # driven by opportunity evidence, so the artifact MUST carry it.
    if set(artifact.capability_outcomes) != set(CAPABILITY_METRICS):
        raise EvalArtifactError(
            f"capability_outcomes must cover exactly {CAPABILITY_METRICS}, "
            f"got {set(artifact.capability_outcomes)}"
        )
    for group, pieces in artifact.capability_outcomes.items():
        if not isinstance(pieces, dict) or set(pieces) != set(FIVE_PIECE_KEYS):
            raise EvalArtifactError(
                f"capability_outcomes {group} must be exactly the five "
                f"pieces {FIVE_PIECE_KEYS}, got {set(pieces) if isinstance(pieces, dict) else type(pieces).__name__}"
            )
        for key in ("opportunity_count", "exposure_count", "event_count"):
            count = pieces[key]
            if isinstance(count, bool) or not isinstance(count, int) \
                    or count < 0:
                raise EvalArtifactError(
                    f"capability_outcomes {group}.{key} must be a "
                    f"non-negative int, got {count!r}"
                )
        if pieces["exposure_count"] > pieces["opportunity_count"]:
            raise EvalArtifactError(
                f"capability_outcomes {group}.exposure_count "
                f"{pieces['exposure_count']} > opportunity_count "
                f"{pieces['opportunity_count']}"
            )
        if pieces["event_count"] > pieces["exposure_count"]:
            raise EvalArtifactError(
                f"capability_outcomes {group}.event_count "
                f"{pieces['event_count']} > exposure_count "
                f"{pieces['exposure_count']}"
            )
        value = pieces["value"]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
        ):
            raise EvalArtifactError(
                f"capability_outcomes {group}.value must be null or a "
                f"finite number, got {value!r}"
            )
        reason = pieces["validity_reason"]
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise EvalArtifactError(
                f"capability_outcomes {group}.validity_reason must be null "
                f"or a non-empty string, got {reason!r}"
            )
        # §11.5.5: event_count==0 -> value null + explicit validity_reason
        # (null, never a 0 score); event_count>0 -> finite value and no
        # reason — the honest two-state contract, no third state.
        if pieces["event_count"] == 0:
            if value is not None or reason is None:
                raise EvalArtifactError(
                    f"capability_outcomes {group} event_count=0 requires "
                    "value null and a validity_reason (honest empty "
                    "state), got value="
                    f"{value!r} reason={reason!r}"
                )
        else:
            if value is None or reason is not None:
                raise EvalArtifactError(
                    f"capability_outcomes {group} event_count>0 requires "
                    f"a finite value and no validity_reason, got "
                    f"value={value!r} reason={reason!r}"
                )
    # R5.8-E (N4): every recorded policy-noise seed must re-derive from
    # the artifact's OWN provenance (bank_id + episode_config_hash +
    # replica) — "the field exists" is not enough; a seed recorded under
    # a different bank id is a silent provenance break.
    from diffusion_policy_3d.rl.dppo_uav.paired_rng import policy_noise_seed

    for ep in artifact.episodes:
        expected_seed = policy_noise_seed(
            artifact.bank_id,
            ep.episode_config.get("episode_config_hash", ""),
            replica_id=ep.replica_id,
        )
        if ep.policy_noise_seed != expected_seed:
            raise EvalArtifactError(
                f"policy_noise_seed {ep.policy_noise_seed} does not "
                f"re-derive from (bank_id, episode_config_hash, replica) "
                f"= {expected_seed} — the recorded noise stream is not "
                "reproducible from the artifact provenance"
            )
    for ep in artifact.episodes:
        for tag, d in (("bc", ep.bc), ("rl", ep.rl)):
            # R5.9-C4 (chain-E review P1-1): done_reason and the
            # early-death flags join the required set — an artifact that
            # cannot say WHY an episode ended (reached vs collision vs
            # out_of_bounds vs altitude vs timeout) is rejected.
            for key in ("success", "collision", "timeout", "done_reason",
                        "out_of_bounds", "altitude_exceed",
                        "terminated", "truncated", "terminal_category",
                        "return", "steps"):
                if key not in d:
                    raise EvalArtifactError(
                        f"episode {ep.seed} {tag} record missing {key}"
                    )
            # Rework #2 (P1-2): keys present is not values correct — the
            # outcome layer gets the same type/finite/range treatment the
            # trace layer already had (存而不验 anti-pattern).
            for key in ("success", "collision", "timeout",
                        "out_of_bounds", "altitude_exceed",
                        "terminated", "truncated"):
                if not isinstance(d[key], bool):
                    raise EvalArtifactError(
                        f"episode {ep.seed} {tag} {key} must be a bool, "
                        f"got {d[key]!r}"
                    )
            if not isinstance(d["done_reason"], str) or not d["done_reason"]:
                raise EvalArtifactError(
                    f"episode {ep.seed} {tag} done_reason must be a "
                    f"non-empty str, got {d['done_reason']!r}"
                )
            # R6-E §11.6.4 (F-10): the mutually exclusive terminal
            # category + the raw event flags ride every record, and the
            # category must RE-DERIVE from the raw flags — a record that
            # claims one ending on contradictory raw events is an
            # unmappable state and hard-fails (no priority-silent
            # reclassification).
            from diffusion_policy_3d.env.episode_outcome import (
                NON_ENV_CATEGORIES,
                TERMINAL_CATEGORY_ENUM,
                terminal_category,
            )

            stored = d["terminal_category"]
            if not isinstance(stored, str) or not stored:
                raise EvalArtifactError(
                    f"episode {ep.seed} {tag} terminal_category must be a "
                    f"non-empty str, got {stored!r}"
                )
            if stored not in TERMINAL_CATEGORY_ENUM:
                raise EvalArtifactError(
                    f"episode {ep.seed} {tag} terminal_category {stored!r} "
                    f"not in the fixed enum {sorted(TERMINAL_CATEGORY_ENUM)}"
                )
            # timeout counts ONLY when done_reason=="timeout" — the old
            # bool(truncated or loop_timeout) semantic residue is rejected.
            if d["timeout"] != (d["done_reason"] == "timeout"):
                raise EvalArtifactError(
                    f"episode {ep.seed} {tag} timeout flag "
                    f"{d['timeout']!r} must equal (done_reason == "
                    f"'timeout') with done_reason={d['done_reason']!r}"
                )
            # Re-derivation: a non-env category (observation_fault /
            # rollout_cut / other_terminated) replays as other=<stored> —
            # they require a clean non-terminated env state, which the
            # mapper enforces.
            other = stored if stored in NON_ENV_CATEGORIES else "none"
            try:
                expected = terminal_category(
                    terminated=d["terminated"],
                    truncated=d["truncated"],
                    success=d["success"],
                    collision=d["collision"],
                    done_reason=d["done_reason"],
                    out_of_bounds=d["out_of_bounds"],
                    altitude_exceed=d["altitude_exceed"],
                    other=other,
                )
            except ValueError as e:
                raise EvalArtifactError(
                    f"episode {ep.seed} {tag} terminal_category "
                    f"{stored!r} contradicts the raw terminal events: {e}"
                ) from e
            if stored != expected:
                raise EvalArtifactError(
                    f"episode {ep.seed} {tag} terminal_category "
                    f"{stored!r} does not re-derive from the raw flags "
                    f"(expected {expected!r})"
                )
            ret = d["return"]
            if isinstance(ret, bool) or not isinstance(ret, (int, float)):
                raise EvalArtifactError(
                    f"episode {ep.seed} {tag} return must be a number, "
                    f"got {ret!r}"
                )
            # int 数学上恒有限（任意精度），float(ret) 对超大 int 反而会
            # OverflowError — 有限性判定只对 float 走转换（P2-3）。
            if not isinstance(ret, int) and not np.isfinite(float(ret)):
                raise EvalArtifactError(
                    f"episode {ep.seed} {tag} return must be finite, "
                    f"got {ret!r}"
                )
            steps = d["steps"]
            if isinstance(steps, bool) or not isinstance(steps, int):
                raise EvalArtifactError(
                    f"episode {ep.seed} {tag} steps must be an int, "
                    f"got {steps!r}"
                )
            if steps < 0:
                raise EvalArtifactError(
                    f"episode {ep.seed} {tag} steps must be >= 0, "
                    f"got {steps!r}"
                )
        for side in ("bc", "rl"):
            steps = ep.trace.get(side)
            if not steps:
                raise EvalArtifactError(
                    f"episode {ep.seed} trace[{side}] missing — the raw "
                    "per-step series is required, aggregate-only "
                    "is forbidden"
                )
            for k, step in enumerate(steps):
                if not isinstance(step, dict):
                    raise EvalArtifactError(
                        f"episode {ep.seed} trace[{side}][{k}] is not an object"
                    )
                missing = [
                    key for key in TRACE_STEP_REQUIRED_KEYS if key not in step
                ]
                if missing:
                    raise EvalArtifactError(
                        f"episode {ep.seed} trace[{side}][{k}] missing "
                        f"step fields: {missing}"
                    )
                # P1-4: a NaN/Inf in ANY required trace field poisons
                # every downstream metric — refuse, never mask (R3.5:
                # NaN is never a valid metric).
                for key in NUMERIC_TRACE_KEYS:
                    value = step[key]
                    values = value if key == "pos" else [value]
                    for v in values:
                        if not np.isfinite(float(v)):
                            raise EvalArtifactError(
                                f"episode {ep.seed} trace[{side}][{k}]."
                                f"{key} is non-finite ({v!r}) — a required "
                                "trace field must be finite"
                            )


def freeze_paired_bank(
    env: Any, specs: Sequence[EpisodeTaskSpec], seeds: Sequence[int],
) -> List[PairedBankEntry]:
    """Freeze one bank entry per (spec, seed).

    Probes each spec on the DEDICATED single-slot env (reset -> read the
    runtime config -> abort_episode to close the probe slot), so every
    entry carries the ACTUAL sampled start/goal/yaw and the frozen
    episode_config_hash. (spec, seed) determines the sampled config
    deterministically on both the real adapter and the mock, so replaying
    the bank at eval time reproduces the same hash — the freeze contract.

    Only the env duck-type surface is used (reset/current_episode_config/
    abort_episode/n_envs), so the same freeze works against the real
    adapter and the mock. n_envs != 1 hard-fails: the probe closes slot 0.
    """
    if getattr(env, "n_envs", 1) != 1:
        raise RuntimeError(
            "freeze_paired_bank probes the single-slot env surface; got "
            f"n_envs={env.n_envs} — freeze the bank on a dedicated "
            "n_envs=1 env (the probe closes slot 0 after every reset)"
        )
    if len(specs) != len(seeds):
        raise ValueError(
            "freeze_paired_bank needs one seed per spec — got "
            f"{len(specs)} specs and {len(seeds)} seeds"
        )
    entries: List[PairedBankEntry] = []
    for spec, seed in zip(specs, seeds):
        seed = int(seed)
        env.reset([spec], [seed])
        cfg = env.current_episode_config()
        env.abort_episode([0])
        if cfg["world"] != spec.world_uid or cfg["role"] != spec.role:
            raise ValueError(
                "freeze probe mismatch: runtime world/role "
                f"({cfg['world']}/{cfg['role']}) != spec "
                f"({spec.world_uid}/{spec.role}) — the environment and "
                "the spec disagree"
            )
        if int(cfg["seed"]) != seed:
            raise ValueError(
                f"freeze probe mismatch: runtime seed {cfg['seed']} != "
                f"requested {seed} for {spec.world_uid}"
            )
        entries.append(PairedBankEntry(
            world_uid=spec.world_uid,
            role=spec.role,
            split=spec.split,
            admission_id=spec.admission_id,
            manifest_entry_hash=spec.manifest_entry_hash,
            obstacle_json=spec.obstacle_json,
            generator_version=spec.generator_version,
            seed=seed,
            start=[float(v) for v in cfg["start"]],
            goal=[float(v) for v in cfg["goal"]],
            start_yaw=float(cfg["start_yaw"]),
            episode_config_hash=str(cfg["episode_config_hash"]),
            # The adapter's reset always builds disturb_events=[]; the
            # entry records what the environment will actually inject.
            disturbance={"events": list(cfg.get("disturb_events", []))},
            admission={
                "start_clearance": float(cfg["start_clearance"]),
                "goal_clearance": float(cfg["goal_clearance"]),
                "reachability_pass": bool(cfg["reachability_pass"]),
                "path_length": float(cfg["path_length"]),
                "path_stretch": float(cfg["path_stretch"]),
            },
        ))
    return entries


def _bank_content(bank: Dict[str, Any]) -> Dict[str, Any]:
    """The bank content the bank_id binds — version, provenance and every
    entry byte, EXCLUDING the injected bank_id/bank_version fields
    themselves (a self-referencing hash could never go stale)."""
    return {
        "bank_version": bank["bank_version"],
        "source_hash": bank["source_hash"],
        "creation_commit": bank["creation_commit"],
        "entries": [
            {k: v for k, v in e.items()
             if k not in ("bank_id", "bank_version")}
            for e in bank["entries"]
        ],
    }


def _entries_content(
    entries: Sequence[PairedBankEntry],
) -> List[Dict[str, Any]]:
    """The canonical per-entry content dict (single schema source).

    The R6-EVAL-2 opportunity block rides along ONLY when the entry
    carries one — an entry without opportunity emits the identical
    content as before, so its bank_id binding is unchanged (v3 banks
    stay readable and hash-stable).
    """
    out: List[Dict[str, Any]] = []
    for e in entries:
        entry: Dict[str, Any] = {
            "world_uid": e.world_uid,
            "role": e.role,
            "split": e.split,
            "admission_id": e.admission_id,
            "manifest_entry_hash": e.manifest_entry_hash,
            "obstacle_json": e.obstacle_json,
            "generator_version": e.generator_version,
            "seed": e.seed,
            "start": list(e.start),
            "goal": list(e.goal),
            "start_yaw": float(e.start_yaw),
            "episode_config_hash": e.episode_config_hash,
            "disturbance": dict(e.disturbance),
            "admission": dict(e.admission),
            "required_trace_fields": list(e.required_trace_fields),
        }
        if e.opportunity is not None:
            entry["opportunity"] = opportunity_to_dict(e.opportunity)
        out.append(entry)
    return out


def bank_id_from_entries(entries: Sequence[PairedBankEntry]) -> str:
    """R5.8-E (N4): the canonical content hash for entry objects —
    run_e2e_batch derives it when the caller passes no bank_id, so every
    paired eval binds the bank's content even without the JSON round
    trip (the file-level bank_id adds the bank headers on top)."""
    return canonical_sha256(_bank_content({"entries": _entries_content(entries)}))


def paired_bank_to_dict(
    entries: Sequence[PairedBankEntry], source_hash: str, creation_commit: str,
) -> Dict[str, Any]:
    """The frozen bank as JSON: bank_version + the bank's source
    hash + the creation commit, then the entries (single schema source —
    REQUIRED_BANK_ENTRY_FIELDS).

    bank_id (P1-3, bank v3): canonical sha256 over the FULL content —
    every entry byte (seed/start/goal/disturbance/admission/...). The id
    is injected top-level AND per-entry, so any later byte change breaks
    the binding and paired_bank_from_dict refuses the bank.
    """
    bank = {
        "bank_version": BANK_VERSION,
        "source_hash": source_hash,
        "creation_commit": creation_commit,
        "entries": _entries_content(entries),
    }
    bank["bank_id"] = canonical_sha256(_bank_content(bank))
    for entry in bank["entries"]:
        entry["bank_id"] = bank["bank_id"]
        entry["bank_version"] = BANK_VERSION
    return bank


def paired_bank_from_dict(bank: Dict[str, Any]) -> List[PairedBankEntry]:
    """Schema validation of a frozen bank JSON — the single
    schema source for BOTH entrypoints. Raises ValueError with a precise
    message on any structural problem (CLI level, before preflight)."""
    if not isinstance(bank, dict) or "bank_version" not in bank:
        raise ValueError(
            "paired bank must be a JSON object with a bank_version field"
        )
    # Schema gates FIRST (precise messages), bank_id binding LAST: the
    # binding is the catch-all for schema-valid tampering, so it must not
    # shadow a specific schema violation (e.g. a stale v1/v2 bank or a
    # dropped source_hash would otherwise report "binding broken").
    if bank["bank_version"] != BANK_VERSION:
        raise ValueError(
            f"paired bank schema mismatch: bank_version {bank['bank_version']!r}"
            f" != {BANK_VERSION!r} — regenerate the bank with "
            "freeze_paired_bank"
        )
    for key in ("source_hash", "creation_commit"):
        if not isinstance(bank.get(key), str) or not bank[key]:
            raise ValueError(f"paired bank missing {key}")
    entries = bank.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("paired bank must carry a non-empty entries list")
    out: List[PairedBankEntry] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"paired bank entry {i} is not a JSON object")
        missing = [k for k in REQUIRED_BANK_ENTRY_FIELDS if k not in entry]
        if missing:
            raise ValueError(f"paired bank entry {i} missing fields: {missing}")
        if entry["role"] not in ROLE_NAMES:
            raise ValueError(
                f"paired bank entry {i} invalid role {entry['role']!r}: "
                f"expected one of {ROLE_NAMES}"
            )
        if entry["split"] not in SPLIT_NAMES:
            raise ValueError(
                f"paired bank entry {i} invalid split {entry['split']!r}: "
                f"expected one of {SPLIT_NAMES}"
            )
        # R6-EVAL-2: the opportunity block is OPTIONAL — a v3 bank
        # without it stays readable (None, never a fabricated default);
        # a present block is validated strictly and an invalid one
        # (e.g. policy-dependent provenance) hard-fails the bank load.
        opportunity = None
        if entry.get("opportunity") is not None:
            opportunity = opportunity_from_dict(entry["opportunity"])
        out.append(PairedBankEntry(
            world_uid=str(entry["world_uid"]),
            role=str(entry["role"]),
            split=str(entry["split"]),
            admission_id=str(entry["admission_id"]),
            manifest_entry_hash=str(entry["manifest_entry_hash"]),
            obstacle_json=str(entry["obstacle_json"]),
            generator_version=str(entry["generator_version"]),
            seed=int(entry["seed"]),
            start=[float(v) for v in entry["start"]],
            goal=[float(v) for v in entry["goal"]],
            start_yaw=float(entry["start_yaw"]),
            episode_config_hash=str(entry["episode_config_hash"]),
            disturbance=dict(entry["disturbance"]),
            admission=dict(entry["admission"]),
            opportunity=opportunity,
            required_trace_fields=[
                str(v) for v in entry.get(
                    "required_trace_fields", list(TRACE_STEP_REQUIRED_KEYS)
                )
            ],
        ))
    # Binding LAST: any schema-valid byte change (a reseeded entry, an
    # edited disturbance) is caught here — the artifact cannot say which
    # bank produced it when the recorded bank_id no longer matches.
    if not bank.get("bank_id"):
        raise ValueError(
            "paired bank missing bank_id — a bank without its content "
            "binding is not auditable (bank v3; regenerate with "
            "freeze_paired_bank)"
        )
    if canonical_sha256(_bank_content(bank)) != bank["bank_id"]:
        raise ValueError(
            "paired bank bank_id binding broken: content does not hash to "
            "the recorded bank_id — the bank was edited after freeze; "
            "refusing to load"
        )
    return out


# ═══════════════════════════════════════════════════════════════════
# R7-8: the r7-paired-bank-v1 frozen artifact (guide Task R7-8,
# Step 1/6/7) — v3 bank content + the admission-facing top-level
# contract (bank_size / membership_hash / aggregate / content_hash).
# ═══════════════════════════════════════════════════════════════════

R7_BANK_SCHEMA_VERSION = "r7-paired-bank-v1"
R7_BANK_EVAL_SPLITS = ("rl_val", "rl_test")


def _r7_content_hashable(bank: Dict[str, Any]) -> Dict[str, Any]:
    """The content content_hash binds — everything except itself and the
    injected top-level bank_id. Per-entry bank_id/bank_version stay in
    (they are bound by bank_id already; there is no cycle because
    bank_id never hashes content_hash), so the same helper keeps the
    writer and the validator symmetric."""
    return {
        k: v for k, v in bank.items() if k not in ("content_hash", "bank_id")
    }


def r7_bank_to_dict(
    entries: Sequence[PairedBankEntry],
    *,
    source_manifest_hash: str,
    bc_checkpoint_hash: str,
    builder_source_hash: str,
    creation_commit: str,
    aggregate: Dict[str, Any],
) -> Dict[str, Any]:
    """The R7-8 frozen bank as JSON — v3 entry content plus the
    admission-facing top-level contract.

    Three independent integrity layers, so any byte change is caught:
      - membership_hash binds the entry list (recomputed by
        r7_bank_validate and consumed by admission_parsers);
      - content_hash binds EVERYTHING except itself (top-level tamper
        such as an edited aggregate);
      - bank_id (v3) additionally binds version + provenance + entries
        (paired_bank_from_dict keeps accepting the file unchanged).

    ``aggregate`` is the Step-1 aggregate (geometry candidates /
    opportunity counts; exposure NOT_YET_MEASURED in v1 — the two-phase
    freeze ruling: R7-9 measures it and rebuilds v2).
    """
    bank: Dict[str, Any] = {
        "schema_version": R7_BANK_SCHEMA_VERSION,
        "bank_version": BANK_VERSION,
        "bank_size": len(entries),
        "membership_hash": canonical_sha256(_entries_content(entries)),
        "aggregate": dict(aggregate),
        "source_hash": source_manifest_hash,
        "bc_checkpoint_hash": bc_checkpoint_hash,
        "builder_source_hash": builder_source_hash,
        "creation_commit": creation_commit,
        "entries": _entries_content(entries),
    }
    bank["bank_id"] = canonical_sha256(_bank_content(bank))
    for entry in bank["entries"]:
        entry["bank_id"] = bank["bank_id"]
        entry["bank_version"] = BANK_VERSION
    bank["content_hash"] = canonical_sha256(_r7_content_hashable(bank))
    return bank


def r7_bank_validate(
    bank: Dict[str, Any],
    train_world_uids: Optional[Sequence[str]] = None,
) -> None:
    """Fail-closed validation of the r7-paired-bank-v1 artifact.

    Raises ValueError with a precise message on ANY of the Step-7
    rejection axes: wrong schema, bank_size mismatch, membership /
    content / bank_id tamper, duplicate episode_config_hash, a train
    split entry, aggregate.opportunity inconsistent with the entries,
    or (when the train world set is given) train leakage.
    """
    if bank.get("schema_version") != R7_BANK_SCHEMA_VERSION:
        raise ValueError(
            f"r7 bank schema mismatch: "
            f"{bank.get('schema_version')!r} != "
            f"{R7_BANK_SCHEMA_VERSION!r}"
        )
    entries = bank.get("entries")
    if not isinstance(entries, list):
        raise ValueError("r7 bank must carry an entries list")
    if bank.get("bank_size") != len(entries):
        raise ValueError(
            f"r7 bank bank_size {bank.get('bank_size')} != entries "
            f"{len(entries)}"
        )
    membership = canonical_sha256(
        _entries_content(
            _entries_from_dicts(entries)
        )
    )
    if bank.get("membership_hash") != membership:
        raise ValueError(
            "r7 bank membership_hash binding broken: entries do not "
            "hash to the recorded membership_hash — the bank was "
            "edited after freeze"
        )
    content = _r7_content_hashable(bank)
    if canonical_sha256(content) != bank.get("content_hash"):
        raise ValueError(
            "r7 bank content_hash binding broken: content does not hash "
            "to the recorded content_hash — the bank was edited after "
            "freeze"
        )
    if canonical_sha256(_bank_content(bank)) != bank.get("bank_id"):
        raise ValueError(
            "r7 bank bank_id binding broken: content does not hash to "
            "the recorded bank_id"
        )
    seen = set()
    for i, entry in enumerate(entries):
        cfg_hash = entry.get("episode_config_hash")
        if not isinstance(cfg_hash, str) or not cfg_hash:
            raise ValueError(
                f"r7 bank entry {i} missing episode_config_hash"
            )
        if cfg_hash in seen:
            raise ValueError(
                f"r7 bank duplicate episode_config_hash {cfg_hash!r}"
            )
        seen.add(cfg_hash)
        if entry.get("split") not in R7_BANK_EVAL_SPLITS:
            raise ValueError(
                f"r7 bank entry {i} split {entry.get('split')!r} not in "
                f"{R7_BANK_EVAL_SPLITS} — the bank is eval-only (Step 3)"
            )
    if train_world_uids is not None:
        train_set = set(train_world_uids)
        leaked = sorted({
            e["world_uid"] for e in entries
            if e.get("world_uid") in train_set
        })
        if leaked:
            raise ValueError(
                "r7 bank train leakage: bank worlds present in the "
                f"train manifest: {leaked}"
            )
    agg = bank.get("aggregate")
    if not isinstance(agg, dict) or "opportunity" not in agg:
        raise ValueError("r7 bank aggregate must carry opportunity")
    recomputed = _r7_opportunity_counts(entries)
    for cap in ("anticipatory", "full_space", "dense_recovery"):
        stored = agg["opportunity"].get(cap)
        if stored != recomputed[cap]:
            raise ValueError(
                f"r7 bank aggregate.opportunity.{cap} {stored} != "
                f"recomputed {recomputed[cap]} — the aggregate was "
                "edited after freeze"
            )


def _entries_from_dicts(entries: Sequence[Dict[str, Any]]
                        ) -> Sequence[PairedBankEntry]:
    """Reconstruct PairedBankEntry objects from entry dicts (the
    canonical entry content already round-trips losslessly)."""
    out: List[PairedBankEntry] = []
    for entry in entries:
        opportunity = None
        if entry.get("opportunity") is not None:
            opportunity = opportunity_from_dict(entry["opportunity"])
        out.append(PairedBankEntry(
            world_uid=str(entry["world_uid"]),
            role=str(entry["role"]),
            split=str(entry["split"]),
            admission_id=str(entry["admission_id"]),
            manifest_entry_hash=str(entry["manifest_entry_hash"]),
            obstacle_json=str(entry["obstacle_json"]),
            generator_version=str(entry["generator_version"]),
            seed=int(entry["seed"]),
            start=[float(v) for v in entry["start"]],
            goal=[float(v) for v in entry["goal"]],
            start_yaw=float(entry["start_yaw"]),
            episode_config_hash=str(entry["episode_config_hash"]),
            disturbance=dict(entry["disturbance"]),
            admission=dict(entry["admission"]),
            opportunity=opportunity,
            required_trace_fields=[
                str(v) for v in entry.get(
                    "required_trace_fields", list(TRACE_STEP_REQUIRED_KEYS)
                )
            ],
        ))
    return out


def _r7_opportunity_counts(
    entries: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    """Recompute the Step-2 opportunity counts from the entries' frozen
    opportunity blocks — the aggregate must equal this or the artifact
    is a tamper symptom."""
    counts = {"anticipatory": 0, "full_space": 0, "dense_recovery": 0}
    for entry in entries:
        op = entry.get("opportunity")
        if op is None or not op.get("valid", False):
            continue
        if op.get("risk_shell_crossing"):
            counts["anticipatory"] += 1
        # dense opportunity (user ruling 2026-08-24): trap ∪ cluttered —
        # must mirror r7_bank_builder.compute_aggregate exactly.
        if (
            op.get("trap_opportunity_count", 0) > 0
            or op.get("topology_class") == "cluttered"
        ):
            counts["dense_recovery"] += 1
        if op.get("support_stratum"):
            counts["full_space"] += 1
    return counts


def read_paired_bank_file(path: str) -> List[PairedBankEntry]:
    """Read + schema-validate a frozen bank file — the shared
    loader for both entrypoints. Raises ValueError with a precise message
    on any structural problem (CLI level, before preflight)."""
    bank_path = Path(path)
    if not bank_path.is_file():
        raise ValueError(f"paired bank not found: {bank_path}")
    try:
        with open(bank_path, "r") as f:
            bank = json.load(f)
    except ValueError as e:
        raise ValueError(
            f"paired bank {bank_path} is not valid JSON: {e}"
        ) from e
    return paired_bank_from_dict(bank)


def paired_eval_artifact_name(
    env_steps: int, ppo_update: int, checkpoint_sha12: str,
) -> str:
    """The UNIQUE canonical artifact filename: env_steps and
    ppo_update are the checkpoint's own cumulative counters (a resumed
    checkpoint keeps counting), so two eval runs of the SAME checkpoint
    share the name only if they would produce identical content."""
    return (
        f"paired_eval_envstep_{int(env_steps):09d}"
        f"_ppoupd_{int(ppo_update):06d}_{checkpoint_sha12}.json"
    )


def write_paired_eval_json(artifact: EvalArtifact, path: str,
                           *, expected_contract_hash: str) -> None:
    """Atomic no-overwrite persist of the §4.17 artifact.

    Serializes, writes a tmp sibling, then os.replace — a reader never
    sees a partial file. An existing target with DIFFERENT content
    hard-fails (the canonical name must stay unique per outcome); identical
    content is a no-op (resume-friendly). Re-validates the artifact once
    more before writing — against the caller's RUNTIME contract hash
    (contract_hash_of(contract)), never the artifact's own recorded value.
    """
    validate_eval_artifact(
        artifact,
        expected_exec_steps=artifact.exec_steps,
        expected_contract_hash=expected_contract_hash,
    )
    payload = json.dumps(artifact.to_dict(), indent=2, default=str).encode()
    target = Path(path)
    if target.is_dir():
        raise RuntimeError(
            f"eval artifact target is a directory: {target}"
        )
    if target.exists():
        existing = target.read_bytes()
        if hashlib.sha256(existing).digest() == hashlib.sha256(payload).digest():
            return  # identical content — no-op
        raise RuntimeError(
            "refusing to overwrite existing eval artifact "
            f"{target}: different content under the same canonical name —"
            " the artifact filename must stay unique per outcome"
        )
    tmp = target.with_name(target.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
