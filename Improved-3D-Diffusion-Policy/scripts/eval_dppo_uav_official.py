#!/usr/bin/env python
"""Canonical paired evaluation entry (A-Official-Core, R3.3).

The ONLY formal eval entry: evaluates a trained RL checkpoint against
its BC baseline on the frozen paired bank. Accepts exactly three inputs
(guide R3.3): --checkpoint, --paired_bank, and the resolved config —
the latter is read from the checkpoint's own exact_resolved_config (the
training config is a saved fact, not independently reconstructible at
eval time).

Identity (R3.3): the checkpoint is validated against the eval-side
runtime on 7 code/asset axes — official DPPO commit, project git
commit/tree, contract, manifest, BC checkpoint, source file hashes. The
config-hash axis passes the checkpoint's own saved value (a training
config cannot be recomputed from the eval side) [JUDGMENT]. The preflight
(git clean/commit, vendor hashes, BC/manifest/contract hashes,
projection=False, train/eval split, numerical mode) runs before ANY
policy/env construction.

The paired bank (R3.3 contract; frozen by freeze_paired_bank): a
bank_version=3 JSON frozen by freeze_paired_bank — every entry carries
world/role/split/admission_id/manifest_entry_hash + the frozen
seed/start/goal/start_yaw/episode_config_hash/disturbance/admission, so
the eval replays the bank exactly and hard-fails when the runtime
episode config hash differs from the frozen one. The schema (bank
version, required fields, role/split enums, bank_id/density
provenance) is validated by paired_bank_from_dict at CLI level (exit
2), before preflight; the world must be reachable in the production
pool — the canonical eval pool is the manifest's three literal splits
rl_train ∪ rl_val ∪ rl_test (R7-8 ruling 2026-08-24), so eval-only
worlds w4/w8/w11/w12/w18 ARE reachable by the canonical eval — and the
entry's manifest_entry_hash must equal the hash recomputed from the
manifest (a gate that compared the bank's
hash to itself would be a tautology).

Usage:
  python scripts/eval_dppo_uav_official.py --checkpoint <ckpt.pt> \
      --paired_bank <bank.json> --output_dir <dir> --device cuda:0
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from diffusion_policy_3d.policy.lora_adapters import insert_lora

DEFAULT_BC_CKPT = (
    "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5"
    "/checkpoints/latest.ckpt"
)
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS_TRAIN = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
)
DEFAULT_WORLDS_VAL = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/val"
)
DEFAULT_WORLDS_TEST = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/test"
)

from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    EMPTY_LORA_CHECKSUM,
    CheckpointBuilder,
    CheckpointError,
    RuntimeIdentity,
    config_hash_of,
    validate_resume_identity,
    verify_config_layer_identity,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (
    file_sha256,
    validate_official_environment,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    AdmissionExhausted,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    EvalArtifactError,
    paired_eval_artifact_name,
    read_paired_bank_file,
    write_paired_eval_json,
)
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    OfflineDensity,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_eval_pool,
    entry_metadata_hash,
    manifest_eval_entries,
)
from diffusion_policy_3d.rl.dppo_uav.policy_loader import (
    load_ddpm_policy,
    restore_lora_state,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    LoopConfig,
    _missing_weight_keys,
    _unexpected_weight_keys,
    build_runtime_eval_artifact,
    lora_checksum_of,
    run_paired_eval,
)


# R5.9-C2: the EXACT reviewed artifact-layer delta between the training
# tree (fd3ddda, chain E) and the fix tree (HEAD). Under
# --allow-identity-commit-drift the source-file identity axis accepts
# drift ONLY within this set (C1 outcome forwarding in the adapter,
# C4 eval side-dict/record fields, C2 identity machinery, and the tests
# pinning them); any other differing file hard-fails.
R59_ARTIFACT_DELTA = frozenset({
    "diffusion_policy_3d/rl/dppo_uav/checkpoint.py",
    "diffusion_policy_3d/rl/dppo_uav/environment_adapter.py",
    "diffusion_policy_3d/rl/dppo_uav/evaluator.py",
    "diffusion_policy_3d/rl/dppo_uav/train_loop.py",
    "scripts/eval_dppo_uav_official.py",
    "tests/dppo_parity/test_checkpoint.py",
    "tests/dppo_parity/test_eval_artifact.py",
    "tests/dppo_parity/test_p5_contracts.py",
    "tests/dppo_parity/test_paired_bank.py",
})


def _root_abs(p: str) -> str:
    """Resolve a possibly-relative path against the project root.

    The preflight and the policy loader resolve against the repo root, so
    the entry does too — a relative path must never silently depend on CWD.
    """
    path = Path(p)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _run_replica_evals(eval_env: Any, bc_policy: Any, rl_policy: Any,
                       contract: Any, loop_config: Any, bank: Any,
                       bank_id: str, replicas: int = 1,
                       policy_checkpoint_hash: str = "",
                       ) -> Tuple[List[Any], int]:
    """R6-G (§11.8.2): run the frozen bank ``replicas`` times with
    distinct R5.4 policy-noise streams (replica_id 0..N-1) and merge the
    records into one eval matrix (12 entries x 5 replicas = 60/side).

    Each replica is a real rollout over the SAME frozen bank entry with
    a DIFFERENT policy-noise seed — the R6-D 12x5 matrix validator's
    production path (run_paired_eval used to hard-code replica 0, so a
    multi-replica artifact could never be produced). replicas=1 keeps the
    pre-R6-G single-replica eval bitwise identical.
    """
    records: List[Any] = []
    eval_invalid = 0
    for rep in range(replicas):
        rep_records, rep_invalid = run_paired_eval(
            eval_env, bc_policy, rl_policy, contract, loop_config, bank,
            bank_id=bank_id, replica_id=rep,
            # R6-D §11.5.1 (R6-EVAL-1): every record carries the evaluated
            # policy's checkpoint hash — the SAME hash the artifact embeds
            # (ckpt_hash), so dedup across replicas clusters by policy and
            # cross-artifact BC records deduplicate by BC identity.
            policy_checkpoint_hash=policy_checkpoint_hash,
        )
        records.extend(rep_records)
        eval_invalid += rep_invalid
    return records, eval_invalid


def main() -> int:
    p = argparse.ArgumentParser(
        description="Canonical paired evaluation (A-Official-Core)."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--paired_bank", required=True)
    p.add_argument("--offline_density", required=True)
    p.add_argument("--bc_ckpt", default=DEFAULT_BC_CKPT)
    p.add_argument("--admission_manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--worlds_train", default=DEFAULT_WORLDS_TRAIN)
    p.add_argument("--worlds_val", default=DEFAULT_WORLDS_VAL)
    p.add_argument("--worlds_test", default=DEFAULT_WORLDS_TEST)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument(
        "--replicas", type=int, default=1,
        help="R6-G (§11.8.2): policy-noise replica count per bank entry — "
        "12 entries x 5 replicas = 60/side for R6.4/G8. Each replica runs "
        "the SAME frozen bank entry with a DIFFERENT R5.4 policy-noise "
        "seed (replica_id 0..N-1); records merge into one artifact, whose "
        "12x5 matrix the R6-D validator can check. Default 1 preserves "
        "the pre-R6-G single-replica eval exactly.",
    )
    p.add_argument(
        "--allow-identity-commit-drift", action="store_true",
        help="R5.9-C2 eval-only: cold-evaluate a final release checkpoint "
        "whose training commit differs from the current HEAD because an "
        "ARTIFACT-LAYER fix landed after training (training semantics "
        "byte-identical). Skips ONLY the two git-derived identity axes and "
        "pins the source-file axis to the exact reviewed artifact-layer "
        "delta (R59_ARTIFACT_DELTA); any file differing outside that set "
        "hard-fails, and contract/manifest/bc/config axes stay hard. The "
        "artifact records identity_mode=commit-drift-explicit plus the "
        "actual runtime commit and the observed source-file delta. NEVER "
        "used for training resume.",
    )
    args = p.parse_args()
    if args.replicas < 1:
        print(
            "CLI error: --replicas must be >= 1 "
            f"(got {args.replicas}) — 0 replicas would produce an "
            "empty eval artifact",
            file=sys.stderr,
        )
        return 2
    args.checkpoint = _root_abs(args.checkpoint)
    args.paired_bank = _root_abs(args.paired_bank)
    args.offline_density = _root_abs(args.offline_density)
    args.bc_ckpt = _root_abs(args.bc_ckpt)
    args.admission_manifest = _root_abs(args.admission_manifest)
    args.worlds_train = _root_abs(args.worlds_train)
    args.worlds_val = _root_abs(args.worlds_val)
    args.worlds_test = _root_abs(args.worlds_test)
    args.output_dir = _root_abs(args.output_dir)

    # Bank checks are CLI-level (before preflight): schema, pool
    # reachability, manifest hash truth.
    try:
        bank = read_paired_bank_file(args.paired_bank)
    except ValueError as e:
        print(f"CLI error: {e}", file=sys.stderr)
        return 2
    # The bank's own bank_id binds the eval artifact to the exact frozen
    # bank (bank v3 provenance). An old bank without one is a clean CLI
    # failure — the artifact validator refuses empty provenance anyway.
    try:
        with open(args.paired_bank, "r", encoding="utf-8") as f:
            bank_id = json.load(f).get("bank_id", "")
    except (OSError, ValueError) as e:
        print(
            f"CLI error: cannot read bank_id from {args.paired_bank}: {e}",
            file=sys.stderr,
        )
        return 2
    if not bank_id:
        print(
            "CLI error: paired bank has no bank_id (bank v3 provenance "
            "required by the eval artifact)",
            file=sys.stderr,
        )
        return 2
    # Offline density is CLI-level too: the frozen visit-count field
    # (schema_version 1) drives the capability metrics — a missing or
    # malformed field exits 2 before any rollout.
    try:
        offline_density = OfflineDensity.read_offline_density_file(
            args.offline_density
        )
    except ValueError as e:
        print(f"CLI error: {e}", file=sys.stderr)
        return 2
    # Pool reachability: the canonical eval pool is the manifest's
    # three literal splits rl_train ∪ rl_val ∪ rl_test, each resolved
    # per-split from worlds/{train,val,test} (R7-8 ruling 2026-08-24:
    # "rl_eval" has no literal manifest split) — eval-only worlds
    # w4/w8/w11/w12/w18 ARE reachable here. A bank world outside the
    # eval pool would traceback deep in env construction.
    try:
        pool = build_routea_eval_pool(
            args.admission_manifest, args.worlds_train,
            args.worlds_val, args.worlds_test,
        )
        pool_uids = {uid for uid, _ in pool}
    except (RuntimeError, OSError, ValueError) as e:
        print(f"CLI error: cannot read the eval pool: {e}", file=sys.stderr)
        return 2
    for i, entry in enumerate(bank):
        if entry.world_uid not in pool_uids:
            print(
                f"CLI error: paired bank entry {i} world_uid "
                f"{entry.world_uid} is not in the admission-manifest "
                "eval pool (rl_train ∪ rl_val ∪ rl_test)",
                file=sys.stderr,
            )
            return 2
    # Hash truth: the bank's manifest_entry_hash must equal the hash
    # recomputed from the manifest itself — an admission gate that
    # compared the bank's hash to itself would be a tautology (rule-6
    # admission is hash-proxy over the manifest).
    try:
        manifest_entries = manifest_eval_entries(args.admission_manifest)
        # admission_id is the registry's resolved composite key (a manifest
        # entry may lack the raw field — TaskSpecRegistry falls back).
        manifest_hashes = {
            (
                e["world_uid"], e["role"],
                e.get("admission_id")
                or f"{e['world_uid']}__{e['role']}__{e['split']}",
            ): entry_metadata_hash(e)
            for e in manifest_entries
        }
        for i, entry in enumerate(bank):
            key = (
                entry.world_uid, entry.role, entry.admission_id,
            )
            real = manifest_hashes.get(key)
            if real is None:
                raise ValueError(
                    f"paired bank entry {i} "
                    f"{entry.world_uid}/{entry.role}/"
                    f"{entry.admission_id} is not an admitted "
                    "rl_train ∪ rl_val ∪ rl_test manifest entry"
                )
            if entry.manifest_entry_hash != real:
                raise ValueError(
                    f"paired bank entry {i} manifest_entry_hash mismatch: "
                    f"bank {entry.manifest_entry_hash[:12]}... != "
                    f"manifest {real[:12]}..."
                )
    except ValueError as e:
        print(f"CLI error: {e}", file=sys.stderr)
        return 2

    # Preflight BEFORE any policy/env construction.
    try:
        facts = validate_official_environment(
            manifest_path=args.admission_manifest,
            bc_checkpoint_path=args.bc_ckpt,
            worlds_train_dir=args.worlds_train,
        )
    except RuntimeError as e:
        print(f"preflight FAIL: {e}", file=sys.stderr)
        return 1

    contract = RLSystemContract()
    # Checkpoint + identity: 7 code/asset axes validated against the
    # eval-side runtime. config_hash is NOT the checkpoint's own saved
    # value against itself (tautology): the persisted config_layer is
    # the exact dict hashed at save time — verify_config_layer_identity
    # recomputes over it, then the runtime identity binds the recomputed
    # hash (P1-2).
    try:
        data = CheckpointBuilder.load(args.checkpoint)
        erc = data.exact_resolved_config
        verify_config_layer_identity(erc, data.config_hash)
        validate_resume_identity(
            data, RuntimeIdentity(
                official_dppo_commit=facts["official_dppo_commit"],
                project_git_commit=facts["project_git_commit"],
                project_tree_hash=facts["project_tree_hash"],
                contract_hash=facts["contract_hash"],
                manifest_hash=facts["manifest_hash"],
                bc_checkpoint_hash=facts["bc_checkpoint_hash"],
                world_assets_hash=facts["world_assets_hash"],
                config_hash=config_hash_of(erc["config_layer"]),
                source_file_hashes=facts["source_file_hashes"],
            ),
            # R5.9-C2 eval-only escape hatch: skip ONLY the two git-derived
            # axes and pin the source-file axis to the exact reviewed
            # artifact-layer delta when evaluating a final release
            # checkpoint whose training commit predates an artifact-layer
            # fix (training semantics byte-identical). Training resume
            # never passes either parameter.
            skip_commit_axes=args.allow_identity_commit_drift,
            allowed_source_delta=(
                R59_ARTIFACT_DELTA
                if args.allow_identity_commit_drift else None
            ),
        )
    except CheckpointError as e:
        print(f"checkpoint identity FAIL: {e}", file=sys.stderr)
        return 1

    # The RL policy: BC baseline + the checkpoint's live LoRA + value head.
    # Empty LoRA hard-fails BEFORE policy construction: sha256(b"") means
    # NO LoRA parameters — an untrained checkpoint must never evaluate.
    if lora_checksum_of(data.lora_state) == EMPTY_LORA_CHECKSUM:
        print(
            "checkpoint identity FAIL: empty LoRA parameter set in the "
            "checkpoint (sha256(b'') = EMPTY_LORA_CHECKSUM) — refusing to "
            "evaluate an untrained LoRA",
            file=sys.stderr,
        )
        return 1
    # The CLI rank/alpha must equal the checkpoint's recorded LoRA
    # identity — silently re-wiring a different rank would evaluate a
    # different model than was trained.
    lora_identity = data.lora_identity
    if (lora_identity["lora_rank"] != args.lora_rank
            or lora_identity["lora_alpha"] != args.lora_alpha):
        print(
            "checkpoint identity FAIL: checkpoint LoRA rank/alpha "
            f"({lora_identity['lora_rank']}/{lora_identity['lora_alpha']}) "
            f"!= --lora_rank/--lora_alpha ({args.lora_rank}/"
            f"{args.lora_alpha}) — refusing to evaluate under a "
            "different LoRA wiring",
            file=sys.stderr,
        )
        return 1
    rl_policy = load_ddpm_policy(args.bc_ckpt, device=args.device)
    lora_info = insert_lora(
        rl_policy.model, r=args.lora_rank, alpha=args.lora_alpha,
        verbose=False,
    )
    # The live wiring must answer the same n_adaptors question as the
    # checkpoint's identity (insert_lora nests the axes under report).
    live_n_adaptors = (
        lora_info.get("n_adaptors")
        or lora_info.get("report", {}).get("n_adaptors")
        or 0
    )
    if live_n_adaptors != lora_identity["n_adaptors"]:
        print(
            "checkpoint identity FAIL: live LoRA n_adaptors "
            f"{live_n_adaptors} != checkpoint identity "
            f"{lora_identity['n_adaptors']} — the model wiring does not "
            "match the trained checkpoint",
            file=sys.stderr,
        )
        return 1
    critic = ValueCritic(contract, PolicyObsEncoder(rl_policy)).to(args.device)
    # R5.8-D (D-3): the canonical eval cold-loads RELEASE checkpoints
    # only — a resume-kind checkpoint (lightweight, base UNet stripped)
    # must fail with the KIND named, never a misleading "state does not
    # fit" message.
    if data.checkpoint_kind != "release":
        print(
            "checkpoint identity FAIL: checkpoint_kind is "
            f"{data.checkpoint_kind!r} — the canonical eval only "
            "cold-loads RELEASE checkpoints (full policy materialized); "
            "a resume-kind checkpoint is for training continuation, not "
            "evaluation",
            file=sys.stderr,
        )
        return 1
    try:
        rl_policy.load_state_dict(data.policy_state)
    except (RuntimeError, ValueError) as e:
        print(
            "checkpoint identity FAIL: policy state does not fit the BC "
            f"model: {e}",
            file=sys.stderr,
        )
        return 1
    if data.lora_state:
        try:
            restore_lora_state(lora_info, data.lora_state)
        except RuntimeError as e:
            print(
                "checkpoint identity FAIL: LoRA state does not fit "
                f"--lora_rank={args.lora_rank}/--lora_alpha={args.lora_alpha} "
                f"(the checkpoint was trained with different ranks): {e}",
                file=sys.stderr,
            )
            return 1
    try:
        critic.load_value_state_dict(data.value_state)
    except (RuntimeError, ValueError) as e:
        print(
            "checkpoint identity FAIL: value state does not fit the "
            f"critic: {e}",
            file=sys.stderr,
        )
        return 1
    model = getattr(rl_policy, "model", None)
    anchor = next(model.parameters(), None) if model is not None else None
    if anchor is not None:
        rl_policy.to(anchor.device)
        for module, name in ((rl_policy, "policy"), (critic, "critic")):
            for p in module.parameters():
                assert p.dtype == torch.float32, (
                    f"{name} param not FP32 after eval restore: {p.dtype}"
                )
                assert p.device == anchor.device, (
                    f"{name} param on {p.device} != policy anchor "
                    f"{anchor.device} after eval restore"
                )
    # The BC baseline is the fresh frozen checkpoint — no LoRA, no update.
    bc_policy = load_ddpm_policy(args.bc_ckpt, device=args.device)

    # Eval env: dedicated single-slot adapter over the production pool.
    specs = [e.to_spec() for e in bank]
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs
    }
    # Resolved config comes from the checkpoint (a saved fact); the eval
    # rollout params mirror the training batch's. An older checkpoint
    # without the full config layer is a clean hard failure, never a
    # late KeyError traceback.
    try:
        max_ep_len = int(erc["max_ep_len"])
        reward_version = erc["reward_version"]
        gate_phase = erc.get("gate_phase", "formal")
        min_sampling_std = float(erc["min_sampling_denoising_std"])
        min_logprob_std = float(erc["min_logprob_denoising_std"])
    except (KeyError, TypeError, AttributeError) as e:
        print(
            "checkpoint identity FAIL: exact_resolved_config lacks the "
            f"eval-relevant keys: {e}",
            file=sys.stderr,
        )
        return 1

    # R5.8-C (E-4): the eval env must run the checkpoint's OWN reward
    # version — the artifact's reward_version label must describe the
    # reward that actually produced the recorded returns. Any version
    # outside the supported set is a clean hard failure, never a silent
    # fallback to a legacy reward (which would also misclassify
    # collisions).
    if reward_version not in ("v14", "v16"):
        print(
            "CLI error: checkpoint reward_version "
            f"{reward_version!r} is not supported by the canonical eval "
            "(v14 | v16) — refusing to evaluate under a mismatched "
            "reward function",
            file=sys.stderr,
        )
        return 2
    eval_env = build_adapter(
        contract,
        worlds_dir=args.worlds_train,
        # The canonical eval hands its OWN pool (rl_train ∪ rl_val ∪
        # rl_test, R7-8 ruling 2026-08-24) — eval-only worlds must be
        # reachable here, and the pool was already validated CLI-level
        # against every bank world.
        pool=pool,
        manifest_records=records,
        device=args.device,
        max_episode_steps=max_ep_len,
        n_envs=1,
        reward_version=reward_version,
        # R5.8-E (P1-3): the canonical eval replays the frozen bank
        # EXACTLY — random disturbance is a stochastic term the bank
        # contract does not record, so eval runs disturbance OFF (the
        # old eval convention). Training keeps disturbance ON.
        disturb_enable=False,
    )
    loop_config = LoopConfig(
        mode="one_update", n_episodes=1,
        max_ep_len=max_ep_len,
        device=args.device, output_dir=args.output_dir,
        eval_n_episodes=len(bank),
        gate_phase=gate_phase,
        min_sampling_denoising_std=min_sampling_std,
        min_logprob_denoising_std=min_logprob_std,
    )
    # R6-D §11.5.1 (R6-EVAL-1): the evaluated policy's checkpoint hash —
    # computed BEFORE the eval so run_paired_eval can mirror it onto every
    # record (the same hash the artifact embeds at build time below).
    ckpt_hash = file_sha256(Path(args.checkpoint))
    try:
        # R5.8-E (N4): the canonical eval derives the paired policy-noise
        # seeds from the REAL bank identity — the artifact's recorded
        # seed must re-derive from its own provenance.
        # R6-G (§11.8.2): --replicas N runs the same bank N times with
        # distinct R5.4 policy-noise streams (replica_id 0..N-1) so the
        # R6-D 12x5=60/side matrix has a production path; records merge
        # into one artifact and the 12x5 matrix validator checks it.
        eval_records, eval_invalid = _run_replica_evals(
            eval_env, bc_policy, rl_policy, contract, loop_config, bank,
            bank_id=bank_id, replicas=args.replicas,
            policy_checkpoint_hash=ckpt_hash,
        )
    except AdmissionExhausted as e:
        print(
            f"eval FAIL: admission exhausted for a bank world ({e}) — the "
            "paired bank and the production pool disagree",
            file=sys.stderr,
        )
        return 1
    except (RuntimeError, ValueError) as e:
        print(f"eval FAIL: {e}", file=sys.stderr)
        return 1
    if eval_invalid > 0:
        print(
            f"eval gate: {eval_invalid}/{len(bank)} paired episodes "
            "voided on invalid point-cloud frames — refusing to persist "
            "a polluted paired eval",
            file=sys.stderr,
        )
        return 1

    runtime = {
        "device": args.device,
        "reward_version": reward_version,
        "manifest_hash": facts["manifest_hash"],
        "world_assets_hash": facts["world_assets_hash"],
        "project_git_commit": facts["project_git_commit"],
        "project_tree_hash": facts["project_tree_hash"],
        # R5.9-C2: the artifact must make the identity mode explicit so a
        # downstream reader can distinguish a strict identity check from
        # the eval-only commit-drift bypass, and must record the observed
        # source-file delta (files whose runtime hash differs from the
        # checkpoint's training-tree claim) for audit.
        "identity_mode": (
            "commit-drift-explicit"
            if args.allow_identity_commit_drift else "strict"
        ),
        "source_file_delta": sorted(
            f for f in set(data.source_file_hashes) | set(facts["source_file_hashes"])
            if data.source_file_hashes.get(f) != facts["source_file_hashes"].get(f)
        ),
        "source_file_hashes": facts["source_file_hashes"],
        "bc_checkpoint_hash": facts["bc_checkpoint_hash"],
        # The eval artifact gets its own wall-clock — the checkpoint's
        # recorded timestamp stays in the checkpoint.
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        artifact = build_runtime_eval_artifact(
            records=eval_records,
            checkpoint_hash=ckpt_hash,
            config_dict=erc,
            contract=contract,
            runtime=runtime,
            lora_checksum=lora_checksum_of(data.lora_state),
            observation_invalid_count=eval_invalid,
            missing_weight_keys=_missing_weight_keys(rl_policy),
            unexpected_weight_keys=_unexpected_weight_keys(rl_policy),
            history_indices=list(
                range(
                    contract.history_steps - contract.obs_steps,
                    contract.history_steps,
                )
            ),
            offline_density=offline_density,
            bank_id=bank_id,
            # R6-E §11.6.1 (F-06): the RL policy loads the SAME
            # registered BC checkpoint — its unexpected keys are exactly
            # the 7 allowlisted anchors; the artifact carries the frozen
            # weight-load contract that proves it (validation hard-fails
            # on unexpected keys without a valid contract).
            load_contract=getattr(rl_policy, "_dppo_load_contract", None),
            # R6-EVAL-2 (§11.5.3): the frozen capability opportunity
            # blocks ride the bank entries — the artifact's five-piece
            # outcomes are computed from them (bank_id -> frozen bank ->
            # re-derivable).
            opportunity_by_entry={
                e.episode_config_hash: e.opportunity
                for e in bank if e.opportunity is not None
            },
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / paired_eval_artifact_name(
            data.environment_transition_count, data.ppo_update_count,
            ckpt_hash[:12],
        )
        write_paired_eval_json(
            artifact, str(out_path),
            expected_contract_hash=facts["contract_hash"],
        )
    except (RuntimeError, ValueError, EvalArtifactError) as e:
        print(f"eval FAIL: {e}", file=sys.stderr)
        return 1

    print(
        f"EVAL COMPLETE checkpoint={args.checkpoint} "
        f"paired={len(eval_records)} artifact={out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
