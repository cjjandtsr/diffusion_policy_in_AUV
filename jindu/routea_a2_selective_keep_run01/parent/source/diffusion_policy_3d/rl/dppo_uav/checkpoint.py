"""Checkpoint schema, naming and save/load (design §4.16, RTM CKPT-AUDIT).

§4.16 field checklist (all required):
  schema_version, official_dppo_commit, project_git_commit/tree_hash,
  source_file_hashes, contract serialization + hash, manifest hash,
  BC checkpoint hash, config hash, policy/value state, LoRA state,
  actor/value optimizer state, LR scheduler state, sampling/logprob std
  state, reward-normalization state, RNG states (python/numpy/torch/cuda/
  env), sampler state (EpisodeTaskSampler: schema/base_seed/RNG/cursors/
  attempt/episode/completed counters + spec-set hash — SEED-RESUME
  production path), numerical mode state, outer_iteration, ppo_update_count,
  optimizer_step_count, environment_transition_count,
  completed_episode_count, wall-clock timestamp, exact resolved config.

[JUDGMENT] naming uses real counters:
  dppo_uav_envstep_<envstep:09d>_ppoupd_<ppoupd:06d>.pt
[JUDGMENT] final checkpoint must never write step as 0.

Schema v2 (R3.2): formal resume validates the FULL runtime identity —
official DPPO commit, project git commit + tree hash, contract hash,
manifest hash, BC checkpoint hash, config hash, source file hashes —
via validate_resume_identity() BEFORE any state_dict load. Schema-v1
checkpoints are read-only audit/export only; the formal loader reports
unsupported resume schema v1 instead of guessing the missing
sampler/hash state.

Schema v3 (R3 rework, 2026-08-11): the LoRA configuration identity
(lora_identity: rank/alpha/n_adaptors + parameter checksum) and the
environment-RNG identity (environment_rng_state) are REQUIRED fields,
and the exact hashed config layer persists inside exact_resolved_config
(config_layer) so the eval side can recompute the config hash instead
of trusting the saved value. Schema-v2 checkpoints are explicit
audit-only: the formal loader reports unsupported resume schema v2
(no lora_identity/environment_rng_state to verify).
"""

import base64
import hashlib
import json
import os
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np
import torch

# v3: lora_identity + environment_rng_state + config_layer added; the
# eval side recomputes config_hash from config_layer (P1-2), the restore
# side validates LoRA/sampler/structured/env-RNG identity BEFORE any
# tensor load (P0-1/P0-2/C1/C2); v2 checkpoints are read-only
# audit/export only (audit-only message, never formal resume).
# v4 (external-review #3, RULING-7): lora_identity gains the
# n_lora_param_tensors axis — the lora_state key set is the parameter
# TENSOR count (n adaptors -> 2n tensors, A+B), independent of
# n_adaptors (module count); validate_checkpoint_lora_usable compares
# against it (P0-1). environment_rng_state restorable=True semantics:
# with hash(str(world_id)) removed from the envs and a fixed
# PYTHONHASHSEED preflight on the formal entrypoints, the episode
# sequence is cross-process reproducible, so the builder records
# restorable=True and the formal restore refuses restorable=False
# (P0-2). v3 checkpoints are read-only audit/export only.
SCHEMA_VERSION = "6"  # v6: + quiescent-boundary/phase/vendor/RNG-owner/provenance axes (R7-6)

# sha256(b"") is non-empty yet means NO LoRA parameters — the eval-side
# "non-empty checksum" check used to let PPO==BC (empty adaptor set)
# through as a valid result. The empty-LoRA case is rejected EXPLICITLY.
EMPTY_LORA_CHECKSUM = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class RuntimeIdentity:
    """The runtime identity a checkpoint must match to be formally resumed.

    Every hash is the RUNNING side's value at resume time — if any differs
    from what the checkpoint recorded when it was saved, the resume would
    silently continue training under a different contract/manifest/config,
    so the mismatch is a hard error raised before any state_dict load.
    """

    official_dppo_commit: str
    project_git_commit: str
    project_tree_hash: str
    contract_hash: str
    manifest_hash: str
    bc_checkpoint_hash: str
    world_assets_hash: str  # external world assets (scene + obstacles)
    config_hash: str
    source_file_hashes: Dict[str, str]


class CheckpointError(Exception):
    """Raised on schema violation, missing file, or bad payload."""


@dataclass(frozen=True)
class CheckpointData:
    """§4.16 full checkpoint payload (30 fields: schema_version + 29).

    The §4.16 checklist names 21 categories; several categories expand to
    two fields here (policy/value state, actor/value optimizer state,
    project git commit + tree hash), plus sampler_state, config_hash and
    numerical_mode_state (schema v2), and lora_identity +
    environment_rng_state (schema v3), giving 29 + schema_version = 30.
    """

    schema_version: str
    official_dppo_commit: str
    project_git_commit: str
    project_tree_hash: str
    source_file_hashes: Dict[str, str]
    contract_serialized: bytes  # utf-8 JSON of RLSystemContract
    contract_hash: str
    manifest_hash: str
    bc_checkpoint_hash: str
    config_hash: str  # sha256 over the reproducible config layer
    policy_state: Dict[str, Any]
    value_state: Dict[str, Any]
    lora_state: Dict[str, Any]
    lora_identity: Dict[str, Any]  # rank/alpha/n_adaptors/n_lora_param_tensors + checksum (v4)
    actor_optimizer_state: Dict[str, Any]
    value_optimizer_state: Dict[str, Any]
    lr_scheduler_state: Dict[str, Any]
    sampling_std_state: Dict[str, Any]
    reward_norm_state: Dict[str, Any]
    rng_states: Dict[str, str]  # python/numpy/torch/cuda/env
    numerical_mode_state: Dict[str, Any]  # resolved TF32/determinism
    environment_rng_state: Dict[str, Any]  # env-RNG identity (v3, C1)
    sampler_state: Dict[str, Any]  # EpisodeTaskSampler.state_dict()
    outer_iteration: int
    ppo_update_count: int
    optimizer_step_count: int
    environment_transition_count: int
    completed_episode_count: int
    wall_clock_timestamp: str
    exact_resolved_config: Dict[str, Any]
    # R5.3: per-slot full state of ACTIVE (mid-episode) slots at a
    # transition-budget cut — the next update restores and continues them
    # exactly (empty for episode-based batches). Placed last: dataclass
    # default fields must follow every non-default field.
    active_slot_states: Dict[str, Any] = field(default_factory=dict)
    # R5.5: "resume" checkpoints store the MUTABLE state only (base UNet
    # stripped from policy_state — the frozen base is re-materialized from
    # the BC checkpoint, bound by bc_checkpoint_hash identity); "release"
    # checkpoints materialize the FULL policy for deployment/eval.
    checkpoint_kind: str = "resume"

    # ── R7-6 schema v6 axes (guide R7-6 Step 1) ─────────────────────────
    # All carry defaults so v5-era call sites keep constructing (the
    # formal save path always fills them; validate_v6_contract hard-fails
    # on any non-quiescent value before a tensor load).

    # LoRA mapping axes: the saved state's ordered parameter names and
    # the actor optimizer's ordered parameter names (validated against
    # the live wiring BEFORE any tensor load — a drift is a hard error).
    lora_name_order: List[str] = field(default_factory=list)
    actor_optimizer_param_names: List[str] = field(default_factory=list)
    # Quiescent-boundary assertion: a formal checkpoint is ONLY written at
    # the update boundary (never mid-rollout, never with pending async
    # work) — the resume side refuses anything else.
    checkpoint_phase: str = "quiescent_post_update"
    pending_async_work: bool = False
    # Provenance axes: the policy-observation window the checkpoint's
    # world was computed under (R7-5 window gate) and the next-value
    # derivation (R7-4 own-next-observation) — audit trails that pin what
    # the saved tensors actually saw.
    observation_window_provenance: Dict[str, Any] = field(
        default_factory=dict)
    next_value_provenance: Dict[str, Any] = field(default_factory=dict)
    # Vendor binding (R7-1): the frozen official runtime the run executed
    # under — upstream commit + the 401-file SHA256SUMS manifest hash.
    official_vendor_commit: str = ""
    vendor_manifest_hash: str = ""
    # Double axis of source_file_hashes: the runtime source file set is
    # recorded twice (validate_v6_contract demands equality) so a drift
    # between the two recorded copies is caught, never silently resumed.
    official_runtime_source_files: Dict[str, str] = field(
        default_factory=dict)
    # RNG owners beyond the globals: explicitly-owned persistent
    # Generators (name -> state hex). The production path currently has
    # none (the synth far-dome generator is created per-call with
    # manual_seed — stateless); a non-empty set must be restorable.
    owned_generators: Dict[str, str] = field(default_factory=dict)
    # Metrics writer cursor: how many metrics.jsonl records were
    # committed at save time — the resumed process continues numbering
    # after it (a cursor drift is a hard error).
    metrics_writer_cursor: int = 0
    # Voided (observation-fault) episodes counter — the completed-count
    # twin that never counts toward completed_episode_count.
    voided_episode_count: int = 0

    def __post_init__(self) -> None:
        if self.checkpoint_kind not in ("resume", "release"):
            raise CheckpointError(
                f"checkpoint_kind must be 'resume'|'release', got "
                f"{self.checkpoint_kind!r} (a typo must never silently "
                "materialize a full checkpoint)"
            )


def checkpoint_filename(env_step: int, ppo_update: int) -> str:
    """§4.16 [JUDGMENT] naming with real counters, zero env-step rejected."""
    if env_step <= 0:
        raise CheckpointError(
            f"final checkpoint must never write step 0 (got env_step={env_step})"
        )
    return f"dppo_uav_envstep_{env_step:09d}_ppoupd_{ppo_update:06d}.pt"


class CheckpointBuilder:
    """Serialize/deserialize CheckpointData with schema validation."""

    @staticmethod
    def _encode_active_slots(
        states: Dict[str, Any],
    ) -> Dict[str, str]:
        """R4-rework review partB P0-2: the per-slot states contain
        numpy arrays (kinematics/history/RNG). torch.load(weights_only=True)
        rejects numpy globals, so the saved payload carries base64-pickled
        strings; load() decodes them back. Keys are str(env_id)."""
        import base64 as _b64
        import pickle as _pickle

        return {
            str(env_id): _b64.b64encode(
                _pickle.dumps(state, protocol=4)
            ).decode("ascii")
            for env_id, state in states.items()
        }

    @staticmethod
    def _decode_active_slots(
        encoded: Dict[str, Any],
    ) -> Dict[int, Any]:
        """Inverse of _encode_active_slots; tolerant of already-decoded
        dicts (round-trip idempotence)."""
        import base64 as _b64
        import pickle as _pickle

        out: Dict[int, Any] = {}
        for key, value in encoded.items():
            if isinstance(value, str):
                out[int(key)] = _pickle.loads(_b64.b64decode(value))
            else:
                out[int(key)] = value
        return out

    @staticmethod
    def build(data: CheckpointData) -> Dict[str, Any]:
        """dict form used for torch.save; schema_version must match."""
        if data.schema_version != SCHEMA_VERSION:
            raise CheckpointError(
                f"schema_version {data.schema_version} != {SCHEMA_VERSION}"
            )
        d = asdict(data)
        if d.get("active_slot_states"):
            d["active_slot_states"] = CheckpointBuilder._encode_active_slots(
                d["active_slot_states"]
            )
        # ensure the §4.16 checklist cannot silently drop fields
        missing = [f.name for f in fields(CheckpointData) if f.name not in d]
        if missing:
            raise CheckpointError(f"missing checkpoint fields: {missing}")
        # raw bytes are rejected by torch.load(weights_only=True) (torch 2.4:
        # _codecs.encode is not in the allowlist) — store base64 str instead
        d["contract_serialized"] = base64.b64encode(
            data.contract_serialized
        ).decode("ascii")
        return d

    @staticmethod
    def save(data: CheckpointData, path: Path) -> None:
        """Atomic save (S2): write a tmp sibling, fsync, os.replace —
        a reader never sees a partial checkpoint and a simulated crash
        leaves the previous file intact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(CheckpointBuilder.build(data), tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @staticmethod
    def load(path: Path) -> CheckpointData:
        """Formal load: the current schema ONLY (v6 for R7-6+). A v5 (or
        older) checkpoint is refused — formal resume is v6-only (guide
        R7-6 Step 2); read-only audit/export goes through load_for_audit."""
        return CheckpointBuilder._load_payload(path, formal=True)

    @staticmethod
    def load_for_audit(path: Path) -> CheckpointData:
        """Read-only audit/export load (guide R7-6 Step 2): accepts v5 and
        v6 checkpoints, filling v6 defaults when the file predates them.
        NEVER used by the formal resume path — restore_training_state uses
        load(), so a v5 file cannot silently continue training."""
        return CheckpointBuilder._load_payload(path, formal=False)

    @staticmethod
    def _load_payload(path: Path, *, formal: bool) -> CheckpointData:
        if not Path(path).exists():
            raise CheckpointError(f"checkpoint file not found: {path}")
        # R7-6 Step 9: a leftover .tmp sibling means the atomic save
        # (tmp write -> fsync -> os.replace) was interrupted — the final
        # file may be stale or partial. The formal path refuses the
        # ambiguous state BEFORE any read/load; the read-only audit path
        # inspects whatever exists.
        if formal and Path(str(path) + ".tmp").exists():
            raise CheckpointError(
                f"checkpoint {path} has a leftover .tmp sibling — an "
                "interrupted atomic save leaves the final file's "
                "freshness unverifiable; refusing to resume from an "
                "ambiguous state"
            )
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # corrupt file / bad payload
            raise CheckpointError(f"cannot read checkpoint {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CheckpointError(
                f"checkpoint {path} payload is {type(payload).__name__}, "
                "not a dict — unreadable/invalid checkpoint"
            )
        if payload.get("active_slot_states"):
            payload["active_slot_states"] = (
                CheckpointBuilder._decode_active_slots(
                    payload["active_slot_states"]
                )
            )
        schema = payload.get("schema_version")
        if formal and schema != SCHEMA_VERSION:
            raise CheckpointError(
                f"unsupported resume schema {schema!r}: "
                f"expected {SCHEMA_VERSION} — formal resume is v6-only "
                "(guide R7-6 Step 2); older checkpoints are read-only "
                "audit/export only, never formal resume"
            )
        # drop unknown keys from the frozen dataclass; required ones checked
        known = {f.name for f in fields(CheckpointData)}
        payload = {k: v for k, v in payload.items() if k in known}
        if formal:
            required = [f.name for f in fields(CheckpointData)
                        if f.name not in payload]
        else:
            # audit path: only the non-default fields are required (a v5
            # file legitimately lacks the v6 default-carrying fields).
            required = [
                f.name for f in fields(CheckpointData)
                if f.name not in payload and f.default is MISSING
            ]
        if required:
            raise CheckpointError(f"checkpoint missing fields: {required}")
        # base64 str -> bytes (inverse of build(); see comment there)
        if isinstance(payload.get("contract_serialized"), str):
            payload["contract_serialized"] = base64.b64decode(
                payload["contract_serialized"].encode("ascii")
            )
        return CheckpointData(**payload)


def contract_to_checkpoint_bytes(contract: Any) -> bytes:
    """§4.16 contract serialization: frozen dataclass -> utf-8 JSON bytes."""
    return json.dumps(
        asdict(contract) if hasattr(contract, "__dataclass_fields__") else contract,
        default=str,
    ).encode("utf-8")


def contract_hash_of(contract: Any) -> str:
    """Runtime contract hash for the eval call site.

    sha256 hexdigest over the §4.16 contract serialization
    (contract_to_checkpoint_bytes), so the checkpoint hash and the eval
    artifact's expected_contract_hash always agree.
    """
    return hashlib.sha256(contract_to_checkpoint_bytes(contract)).hexdigest()


def config_hash_of(config_layer: Dict[str, Any]) -> str:
    """sha256 over the reproducible config layer (sorted JSON, default=str).

    The single config-hash implementation — the checkpoint save side, the
    runtime identity, and the eval-side recompute all agree by
    construction (P1-2: the eval side recomputes from config_layer, never
    trusts the saved value against itself).
    """
    return hashlib.sha256(
        json.dumps(config_layer, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def verify_config_layer_identity(
    exact_resolved_config: Dict[str, Any], saved_hash: str
) -> None:
    """Eval-side gate: recompute the config hash from the persisted
    config_layer and compare with the saved config_hash.

    The old eval flow compared data.config_hash against itself
    (tautology) — a checkpoint whose saved hash was stale/wrong still
    "verified". The persisted config_layer is the EXACT dict that was
    hashed at save time, so recomputing over it is an independent check:
    any tamper breaks the binding.
    """
    layer = exact_resolved_config.get("config_layer")
    if layer is None:
        raise CheckpointError(
            "checkpoint missing config_layer in exact_resolved_config — "
            "cannot recompute the config identity (pre-v3 audit-only "
            "checkpoint?)"
        )
    if config_hash_of(layer) != saved_hash:
        raise CheckpointError(
            "config layer tamper: recomputed config_hash != saved "
            f"config_hash {saved_hash!r}; refusing to evaluate under a "
            "different config"
        )


def validate_v6_contract(data: CheckpointData) -> None:
    """R7-6 Step 1/5: the v6 quiescent-boundary contract, checked BEFORE
    any tensor load.

    A formal v6 checkpoint records the state of a QUIESCENT world: the
    save happened exactly at the update boundary (checkpoint_phase) with
    no pending async work, and the official runtime source file set is
    recorded twice (official_runtime_source_files == source_file_hashes)
    so a drift between the two recorded copies is caught. Any violation
    raises CheckpointError naming the axis — never a silent continuation.
    """
    if data.schema_version != SCHEMA_VERSION:
        raise CheckpointError(
            f"v6 contract requires schema_version {SCHEMA_VERSION}, got "
            f"{data.schema_version!r} — formal resume is v6-only"
        )
    if data.checkpoint_phase != "quiescent_post_update":
        raise CheckpointError(
            f"checkpoint_phase {data.checkpoint_phase!r} != "
            "'quiescent_post_update' — a checkpoint saved mid-rollout "
            "cannot resume deterministically"
        )
    if data.pending_async_work:
        raise CheckpointError(
            "checkpoint pending_async_work=True — a checkpoint with "
            "pending async work cannot resume deterministically"
        )
    if data.official_runtime_source_files != data.source_file_hashes:
        raise CheckpointError(
            "official_runtime_source_files != source_file_hashes — the "
            "double-recorded runtime source set drifted; refusing to "
            "resume under an inconsistent source identity"
        )


def validate_vendor_identity(
    data: CheckpointData,
    runtime_vendor: Optional[Dict[str, str]],
) -> None:
    """R7-6 Step 1/5: the frozen official vendor binding (R7-1), checked
    BEFORE any tensor load.

    ``runtime_vendor`` carries the RUNNING side's
    {official_vendor_commit, vendor_manifest_hash} (built by the launcher
    from the vendor snapshot). A checkpoint that CLAIMS a vendor binding
    must match the running side exactly; a checkpoint that claims one
    while the running side supplies NO basis is refused (fail-closed —
    never a silent unverified continuation). Only a checkpoint with no
    vendor claim at all passes without a basis (test/fixture world).
    """
    claimed_commit = data.official_vendor_commit
    claimed_manifest = data.vendor_manifest_hash
    if runtime_vendor is None:
        if claimed_commit or claimed_manifest:
            raise CheckpointError(
                "checkpoint claims a vendor binding "
                f"(official_vendor_commit={claimed_commit!r}) but the "
                "running side supplied no vendor identity basis — "
                "refusing an unverified resume"
            )
        return
    expected_commit = runtime_vendor.get("official_vendor_commit", "")
    expected_manifest = runtime_vendor.get("vendor_manifest_hash", "")
    if claimed_commit != expected_commit:
        raise CheckpointError(
            f"official_vendor_commit mismatch: checkpoint "
            f"{claimed_commit!r} != runtime {expected_commit!r} — the "
            "frozen official runtime the checkpoint ran under is not "
            "the one the resume would run under"
        )
    if claimed_manifest != expected_manifest:
        raise CheckpointError(
            f"vendor_manifest_hash mismatch: checkpoint "
            f"{claimed_manifest!r} != runtime {expected_manifest!r} — "
            "the 401-file vendor manifest drifted between save and resume"
        )


def _source_delta_files(
        saved: Dict[str, str], runtime: Dict[str, str]) -> Set[str]:
    """Files whose hash differs between the checkpoint claim and runtime.

    Both directions count: a file new in runtime (absent from the
    checkpoint's saved tree) and a file deleted since training (present
    in saved, absent at runtime).
    """
    return {
        f for f in set(saved) | set(runtime)
        if saved.get(f) != runtime.get(f)
    }


def validate_resume_identity(
        data: CheckpointData, runtime: RuntimeIdentity,
        skip_commit_axes: bool = False,
        allowed_source_delta: Optional[Set[str]] = None) -> None:
    """Raise CheckpointError before any state_dict load on mismatch.

    Every identity field of the checkpoint (the saved side) must equal the
    running side's value. The source_file_hashes comparison normalizes both
    sides to sorted JSON so identical dicts compare equal regardless of key
    order. Raises on the FIRST mismatch so the caller sees exactly which
    identity axis diverged.

    ``skip_commit_axes`` is the EXPLICIT eval-only escape hatch (R5.9-C2,
    chain-E review P1-1): cold-evaluating a final release checkpoint with
    an artifact-layer fix whose training semantics are unchanged (the
    reward/outcome/rollout math is byte-identical; only the info-wrapping
    and artifact serialization changed). It skips ONLY the two git-derived
    axes (project_git_commit/project_tree_hash). The functional axes stay
    hard — in particular source_file_hashes is NOT skipped outright:
    ``allowed_source_delta`` (when given) pins the EXACT reviewed file set
    whose runtime hash may differ from the checkpoint claim; any file
    differing outside that set raises. The eval entry MUST record the
    actual runtime commit AND the observed source-file delta in the
    artifact — callers never get to silently re-attribute a checkpoint to
    a different tree. Training resume NEVER passes either parameter.
    """
    checks = (
        ("official_dppo_commit",
         data.official_dppo_commit, runtime.official_dppo_commit),
        ("project_git_commit",
         data.project_git_commit, runtime.project_git_commit),
        ("project_tree_hash",
         data.project_tree_hash, runtime.project_tree_hash),
        ("contract_hash", data.contract_hash, runtime.contract_hash),
        ("manifest_hash", data.manifest_hash, runtime.manifest_hash),
        ("bc_checkpoint_hash",
         data.bc_checkpoint_hash, runtime.bc_checkpoint_hash),
        # Gate D reopening (2026-08-22): the saved side is the
        # config_layer's world_assets_hash (hashed at save time); a
        # pre-world-hash checkpoint (config_layer without the key) reads
        # None and is refused — an unbound checkpoint must never resume
        # under a world-binding contract.
        ("world_assets_hash",
         (data.exact_resolved_config or {}).get("config_layer", {}).get(
             "world_assets_hash"),
         runtime.world_assets_hash),
        ("config_hash", data.config_hash, runtime.config_hash),
        (
            "source_file_hashes",
            json.dumps(data.source_file_hashes, sort_keys=True),
            json.dumps(runtime.source_file_hashes, sort_keys=True),
        ),
    )
    for name, saved, expected in checks:
        if skip_commit_axes and name in (
                "project_git_commit", "project_tree_hash"):
            continue
        if name == "source_file_hashes" and allowed_source_delta is not None:
            unexpected = sorted(
                _source_delta_files(
                    data.source_file_hashes, runtime.source_file_hashes
                ) - set(allowed_source_delta)
            )
            if unexpected:
                raise CheckpointError(
                    f"resume identity mismatch: source_file_hashes — "
                    f"{len(unexpected)} file(s) differ outside the pinned "
                    f"artifact-layer delta: {unexpected[:8]}; refusing to "
                    "load any state (the escape hatch pins the EXACT "
                    "reviewed delta, never an open whitelist)"
                )
            continue
        if saved != expected:
            raise CheckpointError(
                f"resume identity mismatch: {name} — checkpoint "
                f"{saved!r} != runtime {expected!r}; refusing to load any "
                "state (a resumed run must reproduce the saved "
                "contract/manifest/config, never silently continue under "
                "different hashes)"
            )


def lora_checksum_of(lora_state: Dict[str, Any]) -> str:
    """Deterministic sha256 over the LoRA state (empty state -> sha256(b""))."""
    h = hashlib.sha256()
    for key in sorted(lora_state.keys()):
        h.update(key.encode("utf-8"))
        value = lora_state[key]
        arr = (value.detach().cpu().numpy() if torch.is_tensor(value)
               else np.asarray(value))
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _live_lora_axes(lora_info: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the live LoRA axes.

    insert_lora returns the axes nested under ``report`` for a wired
    model; plain dicts (tests, callers that only know rank/alpha) carry
    them top-level. Both forms resolve to the same axis dict, with
    absent axes defaulting to the no-LoRA case (0 / 0.0 / 0 / 0 for
    rank / alpha / n_adaptors / n_lora_param_tensors).
    n_lora_param_tensors is the parameter-TENSOR count of the live wiring
    (n adaptors -> 2n tensors, A+B); it lives top-level in the insert_lora
    return dict (lora_params), not in report (P0-1, rework #3).
    """
    report = lora_info.get("report", {}) if isinstance(lora_info, dict) else {}
    return {
        "lora_rank": lora_info.get("lora_rank", report.get("lora_rank", 0)),
        "lora_alpha": lora_info.get("lora_alpha", report.get("lora_alpha", 0.0)),
        "n_adaptors": lora_info.get("n_adaptors", report.get("n_adaptors", 0)),
        "n_lora_param_tensors": len(
            lora_info.get("lora_params", report.get("lora_params", []))
        ),
    }


def build_lora_identity(
    lora_state: Dict[str, Any], lora_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Record the LoRA configuration identity at save time.

    Axes come from the LIVE wiring (rank/alpha/n_adaptors +
    n_lora_param_tensors); the checksum covers the SAVED LoRA parameters.
    An untrained world records the zero identity with the EMPTY checksum —
    which the eval validator rejects explicitly (sha256(b"") is non-empty
    but means NO LoRA). n_lora_param_tensors must equal len(lora_state):
    lora_state is keyed per parameter TENSOR (P0-1, rework #3).
    """
    axes = _live_lora_axes(lora_info or {})
    identity = {
        "lora_rank": axes["lora_rank"],
        "lora_alpha": axes["lora_alpha"],
        "n_adaptors": axes["n_adaptors"],
        "n_lora_param_tensors": axes["n_lora_param_tensors"],
        "lora_checksum": lora_checksum_of(lora_state or {}),
    }
    if identity["n_lora_param_tensors"] != len(lora_state or {}):
        raise CheckpointError(
            f"lora_state key count {len(lora_state or {})} != live "
            f"n_lora_param_tensors {identity['n_lora_param_tensors']} — "
            "the save-side snapshot does not match the live wiring"
        )
    return identity


def validate_lora_identity(data: CheckpointData) -> None:
    """Structure gate: the saved lora_identity must carry all five axes.

    A checkpoint whose lora_identity cannot answer "which LoRA was this
    trained with" is not formally resumable (pre-v3 audit-only). The
    n_lora_param_tensors axis (v4, RULING-7 P0-1) is required: without it
    the parameter-TENSOR count of lora_state cannot be verified against
    the identity.
    """
    required = (
        "lora_rank", "lora_alpha", "n_adaptors",
        "n_lora_param_tensors", "lora_checksum",
    )
    missing = [k for k in required if k not in data.lora_identity]
    if missing:
        raise CheckpointError(
            f"checkpoint lora_identity missing {missing} — cannot verify "
            "the LoRA configuration identity (pre-v4 audit-only "
            "checkpoint?)"
        )


def validate_checkpoint_lora_usable(
    data: CheckpointData, lora_info: Dict[str, Any]
) -> None:
    """Hard-fail before ANY state load when the live LoRA wiring does not
    match the checkpoint's recorded identity (P0-1).

    A checkpoint trained WITH LoRA resumed WITHOUT wiring would silently
    run the BC baseline (PPO==BC); live LoRA over an empty-state
    checkpoint leaves the adaptor tensors random; a checkpoint whose
    lora_state holds tensors but whose identity declares rank 0 is
    internally inconsistent. All three are hard errors — never a
    silently different actor.
    """
    if data.lora_state and data.lora_identity.get("lora_rank", 0) <= 0:
        raise CheckpointError(
            "LoRA wiring inconsistency: checkpoint holds LoRA state but "
            "its lora_identity declares rank 0 — refusing to load any "
            "state (cannot verify which LoRA this was trained with)"
        )
    live = _live_lora_axes(lora_info or {})
    saved = data.lora_identity
    for axis in ("lora_rank", "lora_alpha", "n_adaptors"):
        if live.get(axis, 0) != saved.get(axis, 0):
            raise CheckpointError(
                f"LoRA identity mismatch: {axis} — checkpoint "
                f"{saved.get(axis, 0)} != live {live.get(axis, 0)}; "
                "refusing to load any state (a differently-scaled LoRA "
                "is a different policy)"
            )
    # Rework #2 (P0-1b): the checkpoint's OWN identity must not declare a
    # LoRA it does not carry — wired identity + empty state is an internal
    # inconsistency (the reverse of the rank-0 check above).
    if saved.get("lora_rank", 0) > 0 and not data.lora_state:
        raise CheckpointError(
            "LoRA wiring inconsistency: lora_identity declares wired LoRA "
            f"(rank={saved.get('lora_rank')}) but lora_state is empty — "
            "refusing to load any state (the identity claims a LoRA the "
            "checkpoint does not carry)"
        )
    # Rework #2 (P0-1a): the declared checksum must equal the checksum
    # RECOMPUTED over the actual lora_state tensors — a saved string is
    # never proof of the parameters it claims to bind.
    actual = lora_checksum_of(data.lora_state)
    declared = saved.get("lora_checksum")
    if declared is not None and actual != declared:
        raise CheckpointError(
            f"LoRA checksum mismatch: recomputed {actual[:12]}... != "
            f"declared {declared[:12]}... — the identity does not bind "
            "the actual LoRA parameters; refusing to load any state"
        )
    # Rework #2 (P0-1c) + rework #3 (P0-1): key set / parameter count /
    # tensor shape. The state keys must follow the lora_<i> naming
    # contract with exactly n_lora_param_tensors entries — the lora_state
    # key set is the parameter-TENSOR count (n adaptors -> 2n tensors,
    # A+B), NOT the module count; comparing against n_adaptors rejected
    # every real LoRA checkpoint (RULING-7 P0-1). When the live wiring
    # exposes its parameters, each saved tensor must match the live
    # parameter's shape (rank determines shape — a differently-shaped
    # LoRA is a different policy). Rework #4 (external-review #4 P1)
    # closes the third leg of the three-way equality: the LIVE
    # parameter-tensor count must equal the declared count before the
    # shape walk (the walk iterates live params, so extra saved tensors
    # would otherwise escape unverified).
    if data.lora_state:
        state_keys = list(data.lora_state.keys())
        if state_keys != [f"lora_{i}" for i in range(len(state_keys))]:
            raise CheckpointError(
                f"lora_state keys {state_keys} do not follow the lora_<i> "
                "naming contract — refusing to load any state"
            )
        declared_tensors = saved.get("n_lora_param_tensors")
        if declared_tensors is None:
            raise CheckpointError(
                "checkpoint lora_identity missing n_lora_param_tensors — "
                "the parameter-tensor count cannot be verified (pre-v4 "
                "audit-only checkpoint?)"
            )
        if len(state_keys) != declared_tensors:
            raise CheckpointError(
                f"lora_state key count {len(state_keys)} != declared "
                f"n_lora_param_tensors {declared_tensors} — the identity "
                "does not match the carried parameters"
            )
        live_params = (lora_info or {}).get("lora_params")
        if live_params is None:
            raise CheckpointError(
                "live LoRA wiring exposes no lora_params — the tensor "
                "shapes cannot be verified; refusing to load any state "
                "(a shape-unverifiable restore could silently run a "
                "differently-shaped LoRA)"
            )
        # Rework #4 (external-review #4 P1): close the RULING-7 three-way
        # equality — the LIVE parameter-tensor count must equal the
        # checkpoint-declared count BEFORE the per-tensor shape walk.
        # Without this, saved=4/live=2 passes the walk (it only iterates
        # the live params) and the failure surfaces later in
        # restore_lora_state — after policy_state was already loaded
        # (partial mutation, validation-first contract violated). 2n is
        # only the current production fact, never a restore formula.
        live_tensor_count = len(live_params)
        if live_tensor_count != declared_tensors:
            raise CheckpointError(
                f"live n_lora_param_tensors {live_tensor_count} != "
                f"checkpoint declared {declared_tensors} — the live wiring "
                "exposes a different parameter set than the checkpoint "
                "identity; refusing to load any state (three-way equality: "
                "len(lora_state) == declared == len(live lora_params))"
            )
        for i, p in enumerate(live_params):
            saved_shape = tuple(data.lora_state[f"lora_{i}"].shape)
            if saved_shape != tuple(p.shape):
                raise CheckpointError(
                    f"lora_state lora_{i} shape {saved_shape} != live "
                    f"parameter shape {tuple(p.shape)} — refusing to "
                    "load any state (a differently-shaped LoRA is a "
                    "different policy)"
                )


def build_environment_rng_state() -> Dict[str, Any]:
    """Environment-sequence RNG identity (C1, rework #3 restorable=True).

    The environment episode sequence is driven by the EpisodeTaskSampler
    state (SEED-RESUME production path), so that is the recorded seed
    source. With hash(str(world_id)) removed from the envs (RULING-7 P0-2)
    the episode RNG derives from a stable blake2b of the task payload —
    cross-process reproducible when PYTHONHASHSEED is fixed. The builder
    therefore records restorable=True for a fixed non-empty hash seed,
    and restorable=False (with an empty recorded seed) when unset or
    "random" — such a checkpoint is refused by
    validate_environment_rng_state, never silently resumed. The formal
    entrypoints preflight PYTHONHASHSEED (require_fixed_python_hashseed)
    before any save, so the False branch is a defensive fallback.
    """
    hash_seed = os.environ.get("PYTHONHASHSEED", "")
    restorable = bool(hash_seed) and hash_seed != "random"
    return {
        "seed_source": "sampler_state",
        "python_hash_seed": hash_seed if restorable else "",
        "restorable": restorable,
    }


_ENVIRONMENT_RNG_STATE_KEYS = ("seed_source", "python_hash_seed", "restorable")

# The only legal seed source today: the EpisodeTaskSampler state drives
# the episode sequence (SEED-RESUME production path).
_ENVIRONMENT_RNG_SEED_SOURCES = ("sampler_state",)


def validate_environment_rng_state(data: CheckpointData) -> None:
    """C1: environment_rng_state is a required, VALUE-VALIDATED field.

    Rework #2 (P0-2): keys present is not values correct — seed_source
    must be a known enumeration, restorable must be a real bool, and the
    recorded PYTHONHASHSEED must match the running process character for
    character (a cross-process resume under a different hash seed would
    not reproduce the point-cloud RNG on the multi-world path). Any
    divergence is refused before any state load (存而不验 anti-pattern).
    """
    missing = [
        k for k in _ENVIRONMENT_RNG_STATE_KEYS
        if k not in data.environment_rng_state
    ]
    if missing:
        raise CheckpointError(
            f"checkpoint environment_rng_state missing {missing} — the "
            "environment-sequence RNG identity must be verifiable"
        )
    state = data.environment_rng_state
    if state.get("seed_source") not in _ENVIRONMENT_RNG_SEED_SOURCES:
        raise CheckpointError(
            f"checkpoint environment_rng_state seed_source "
            f"{state.get('seed_source')!r} not in "
            f"{_ENVIRONMENT_RNG_SEED_SOURCES} — refusing to load any "
            "state (the episode-sequence seed source must be verifiable)"
        )
    if not isinstance(state.get("restorable"), bool):
        raise CheckpointError(
            "checkpoint environment_rng_state restorable must be a bool, "
            f"got {state.get('restorable')!r} — refusing to load any "
            "state"
        )
    if state.get("restorable") is not True:
        raise CheckpointError(
            "checkpoint environment_rng_state restorable=False — the "
            "recorded episode sequence is not cross-process reproducible "
            "(unset/random PYTHONHASHSEED at save time); refusing to "
            "resume a run whose point-cloud RNG cannot be reproduced"
        )
    live_hash_seed = os.environ.get("PYTHONHASHSEED", "")
    if state.get("python_hash_seed") != live_hash_seed:
        raise CheckpointError(
            "checkpoint environment_rng_state python_hash_seed "
            f"{state.get('python_hash_seed')!r} != live "
            f"{live_hash_seed!r} — refusing to load any state (a "
            "cross-process resume under a different PYTHONHASHSEED "
            "would not reproduce the point-cloud RNG)"
        )
