"""g7_gate_report.py — the G7 gate report (design §5.9, Guide R4.4).

Consumes the pre-registered G7 artifacts per seed:
  --metrics-jsonl  output_dir/metrics.jsonl (per-batch §4.15 stream
                   persisted by the canonical entry)
  --eval-artifact  paired_eval_*.json from the official eval entry
  --final-ckpt     the seed's final training checkpoint
  --bc-ckpt        the BC baseline checkpoint (base-UNet frozen check)
  --seed           the seed label (report key)

Gates (design §5.9; R6-C §11.4.4 three-way layering):
  ENGINE_HARD_GATE:
    strict JSON + no unexpected non-finite (OptionalStatistic
    two-state contract re-verified from the JSONL — no NaN sentinels)
    per-denoise 5-index live (every fine-tune index carries samples)
    outcome invariants (conditioned sample_count links to counts)
    observation/task/metadata fault = 0 (routing sums zero)
    base update = 0             (final ckpt base UNet bitwise == BC ckpt)
    unexecuted PPO gradient <= 1e-8   (per batch)
    checkpoint/resume           (ppo_update_count == N; env_steps
                               strictly increasing across the chain)
    paired BC regression CR != 1.0
    artifacts audit PASS        (bank_id/checkpoint_hash/LoRA/projection)
  UPDATE_HEALTH_GATE:
    actor update nonzero rate = 100%
    per-denoise KL/clipfrac structured output (per batch)
    action variance >= 10% of BC      (FROZEN-BANK BC baseline via
                               --bc-variance-json)
  CAPABILITY_INFORMATION_GATE:
    conditioned-advantage opportunity (success/collision-conditioned
    sample_count accumulated; insufficient -> BLOCKED, does NOT
    invalidate the engine verdict)

Run (per seed):
  python scripts/g7_gate_report.py --seed 101 \
      --metrics-jsonl data/outputs/g7_microtrain_seed101/metrics.jsonl \
      --eval-artifact <dir>/paired_eval_*.json \
      --final-ckpt <dir>/dppo_uav_*.pt --bc-ckpt <BC ckpt> --out ...
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BC_CKPT = (
    "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5"
    "/checkpoints/latest.ckpt"
)

# R6-C (§11.4.1): frozen OptionalStatistic invalid-reason enum (must
# mirror diffusion_policy_3d.rl.dppo_uav.metrics.OPTIONAL_INVALID_REASONS
# — the gate re-validates the contract independently).
OPTIONAL_INVALID_REASONS = frozenset({
    "no_success_terminal", "no_collision_terminal",
    "constant_value_target", "zero_executed_gradient_denominator",
    "capability_opportunity_absent",
})
OPTIONAL_STATISTIC_FIELDS = frozenset({
    "sample_count", "value", "valid", "invalid_reason",
})
# R6-C rework (P1-3): reader-side strict schema allowlist — only THIS
# version is gateable current input; any other non-empty schema_version
# is a hard-fail (never silently read under the wrong rules). Mirrors
# the writer's constant (scripts/train_dppo_uav_official.SCHEMA_VERSION)
# deliberately: like OPTIONAL_INVALID_REASONS, the gate re-validates the
# contract independently. Version migration must be implemented
# explicitly in this reader, never silently tolerated.
GATE_SCHEMA_VERSION = "routeA_rl_env_v1_r6c"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_optional_statistic(value: Any, where: str,
                                problems: List[str]) -> None:
    """§11.4.1 two-state contract — the gate re-verifies it from the
    JSONL. Any third state (count>0 with value None/NaN, count==0 with
    a numeric value, missing fields, non-finite value, free-form
    invalid_reason) is a gate FAIL."""
    if not isinstance(value, dict):
        problems.append(f"{where} is not an OptionalStatistic dict: {value!r}")
        return
    if set(value) != OPTIONAL_STATISTIC_FIELDS:
        problems.append(
            f"{where} has fields {sorted(value)} — expected "
            f"{sorted(OPTIONAL_STATISTIC_FIELDS)}"
        )
        return
    count = value["sample_count"]
    val = value["value"]
    valid = value["valid"]
    reason = value["invalid_reason"]
    if not isinstance(count, int) or count < 0:
        problems.append(f"{where}.sample_count invalid: {count!r}")
        return
    if count > 0:
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            problems.append(
                f"{where}: count={count} but value {val!r} is not a "
                "finite float (forbidden third state)"
            )
        if valid is not True or reason is not None:
            problems.append(
                f"{where}: count={count} requires valid=True and "
                f"invalid_reason=None, got valid={valid!r} reason={reason!r}"
            )
    else:
        if val is not None or valid is not False:
            problems.append(
                f"{where}: count=0 requires value=None and valid=False, "
                f"got value={val!r} valid={valid!r} (forbidden third "
                "state: count=0 with a numeric value)"
            )
        if reason not in OPTIONAL_INVALID_REASONS:
            problems.append(
                f"{where}: invalid_reason {reason!r} not in the frozen "
                f"enum {sorted(OPTIONAL_INVALID_REASONS)}"
            )


def check_outcome_invariants(record: Dict[str, Any],
                             problems: List[str]) -> None:
    """§11.4.2/§11.4.5 (R6-C rework P1-1/P1-2): outcome counts
    re-verified from the FULL record (cross-layer closure needs
    record.completed_episodes, not just metrics).

    P1-1: conditioned-statistic linkage is UNCONDITIONAL and
    two-way — sample_count must equal the event count whether the
    count is 0 or positive (a phantom count==0/sample_count>0 state
    is a FAIL, not a skip). Event counts must be plain non-negative
    ints (bool is an int subclass and is rejected; missing field is a
    FAIL, never silently skipped).

    P1-2: completed-episode closure — terminal_transition_count must
    equal record.completed_episodes; per-role completed maps must sum
    to it; void/fault/rollout_cut are listed separately and must never
    appear in the completed map. Any violation is an ENGINE gate FAIL.
    """
    m = record.get("metrics")
    if not isinstance(m, dict):
        problems.append("record.metrics missing")
        return
    counts = m.get("outcome_counts")
    if not isinstance(counts, dict):
        problems.append("metrics.outcome_counts missing")
        return
    succ_events = counts.get("success_event_count")
    coll_events = counts.get("collision_event_count")
    per_role = counts.get("per_role_transition_count")
    total = counts.get("transition_count_total")
    if isinstance(per_role, dict) and isinstance(total, int):
        if sum(int(v) for v in per_role.values()) != total:
            problems.append(
                f"outcome invariant: per-role transition counts "
                f"{per_role} do not sum to total {total}"
            )
    elif not isinstance(per_role, dict) or not isinstance(total, int):
        problems.append(
            "outcome_counts.per_role_transition_count / "
            f"transition_count_total broken: {per_role!r} / {total!r}"
        )
    for field, events in (("success_conditioned_advantage", succ_events),
                          ("collision_conditioned_advantage", coll_events)):
        stat = m.get(field)
        validate_optional_statistic(stat, field, problems)
        if not isinstance(events, int) or isinstance(events, bool):
            problems.append(
                f"outcome invariant: {field} event count {events!r} "
                "missing or not a plain int"
            )
            continue
        if events < 0:
            problems.append(
                f"outcome invariant: {field} event count {events} < 0"
            )
            continue
        if not isinstance(stat, dict):
            continue  # structure already flagged by validate_optional_statistic
        if stat.get("sample_count") != events:
            problems.append(
                f"outcome invariant: {field}.sample_count "
                f"{stat.get('sample_count')} != event count {events} "
                "(unconditional, two-way)"
            )
        if events > 0 and stat.get("valid") is not True:
            problems.append(
                f"outcome invariant: {events} {field} events but the "
                "statistic is not valid"
            )
    # ── P1-2: completed-episode cross-layer closure ────────────────
    completed = record.get("completed_episodes")
    if not isinstance(completed, int) or isinstance(completed, bool):
        problems.append(
            f"record.completed_episodes {completed!r} not a plain int"
        )
    term = counts.get("terminal_transition_count")
    if not isinstance(term, int) or isinstance(term, bool):
        problems.append(
            f"outcome_counts.terminal_transition_count {term!r} "
            "not a plain int"
        )
    elif isinstance(completed, int) and not isinstance(completed, bool):
        if term != completed:
            problems.append(
                f"cross-layer invariant: terminal_transition_count "
                f"{term} != record.completed_episodes {completed}"
            )
    per_role_comp = counts.get("per_role_completed_episode_count")
    if not isinstance(per_role_comp, dict):
        problems.append(
            "outcome_counts.per_role_completed_episode_count missing"
        )
    elif isinstance(term, int) and not isinstance(term, bool):
        if any(isinstance(v, bool) for v in per_role_comp.values()):
            problems.append(
                f"per-role completed counts contain bool: "
                f"{per_role_comp}"
            )
        if sum(int(v) for v in per_role_comp.values()) != term:
            problems.append(
                f"outcome invariant: per-role completed {per_role_comp} "
                f"sum {sum(int(v) for v in per_role_comp.values())} "
                f"!= terminal_transition_count {term}"
            )
        for role in ("void", "fault", "rollout_cut"):
            if role in per_role_comp:
                problems.append(
                    f"outcome invariant: role {role!r} leaked into "
                    "per_role_completed_episode_count (must be listed "
                    "separately, never a completed episode)"
                )
    cut = counts.get("rollout_cut_count")
    if not isinstance(cut, int) or isinstance(cut, bool) or cut < 0:
        problems.append(
            f"outcome_counts.rollout_cut_count {cut!r} invalid"
        )
    voided = record.get("voided_episodes")
    if not isinstance(voided, int) or isinstance(voided, bool) \
            or voided < 0:
        problems.append(f"record.voided_episodes {voided!r} invalid")


def check_per_denoise(m: Dict[str, Any], ft_steps: int,
                      problems: List[str]) -> None:
    """§11.4.1/§11.4.4: per-denoise means are NOT optional statistics —
    every fine-tune denoise index 0..ft_steps-1 must be present with a
    finite value in both streams; any missing index or non-finite value
    is an ENGINE hard-fail."""
    for field in ("per_denoise_approx_kl", "per_denoise_clipfrac"):
        stream = m.get(field)
        if not isinstance(stream, dict):
            problems.append(f"{field} missing")
            continue
        for k in range(int(ft_steps)):
            key = str(k)
            if key not in stream:
                problems.append(
                    f"{field}: denoise index {k} missing (ENGINE_HARD_GATE)"
                )
                continue
            v = stream[key]
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                problems.append(f"{field}[{k}] not finite: {v!r}")


def _check_finite(value: Any, where: str, problems: List[str]) -> None:
    if isinstance(value, dict):
        # OptionalStatistic containers are validated structurally (their
        # value may legitimately be None) — descend into everything else.
        if set(value) == OPTIONAL_STATISTIC_FIELDS:
            validate_optional_statistic(value, where, problems)
            return
        for k, v in value.items():
            _check_finite(v, f"{where}.{k}", problems)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _check_finite(v, f"{where}[{i}]", problems)
    elif isinstance(value, float) and not math.isfinite(value):
        problems.append(f"non-finite metric {where}={value}")


def _load_metrics(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load the JSONL. R6-C (§11.4.3): a record WITHOUT schema_version
    is legacy — readable as historical evidence but marked as such and
    never gated as current input. Strict-JSON reader side: a NaN/Inf
    literal (non-standard JSON that json.loads would silently parse)
    and any non-finite value or OptionalStatistic contract violation is
    a gate problem."""
    problems: List[str] = []
    records: List[Dict[str, Any]] = []
    if not path.is_file():
        return records, [f"metrics.jsonl missing: {path}"]
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # R5.8-D (H-4): a torn trailing line from a crash mid-write
            # is reported as such, not a fatal crash — the records
            # before it are intact.
            if i == len(lines):
                problems.append(
                    f"metrics.jsonl line {i} truncated (torn write) — "
                    "using the {i-1} intact records"
                )
            else:
                raise
            continue
        schema = record.get("schema_version")
        if not schema:
            # §11.4.3: legacy schema — historical evidence only, never
            # current-gate input (the writer only produces the new
            # schema).
            problems.append(
                "legacy metrics record without schema_version (line "
                f"{i}) — historical evidence only, not current-gate "
                "input"
            )
            records.append(record)
            continue
        if schema != GATE_SCHEMA_VERSION:
            # R6-C rework (P1-3): strict allowlist — an unknown/old/
            # forged version must never be read under the current
            # rules; hard-fail. Version migration must be implemented
            # explicitly in this reader, never silently tolerated.
            problems.append(
                f"unknown metrics schema_version {schema!r} (line {i}) "
                f"— only {GATE_SCHEMA_VERSION!r} is gateable input"
            )
            records.append(record)
            continue
        records.append(record)
        _check_finite(record["metrics"], "metrics", problems)
    return records, problems


def _base_unet_diff(final_ckpt: Path, bc_ckpt: Path,
                    device: str) -> Tuple[float, List[str]]:
    """Bitwise base-UNet comparison: checkpoint policy_state model.* vs
    the freshly loaded BC policy's model state dict.

    Key normalization: the checkpoint's policy_state is the FULL policy
    state dict (keys prefixed ``model.``/``obs_encoder.``/...), while
    ``policy.model.state_dict()`` is relative to the UNet submodule
    (UNPREFIXED) — the saved keys must be stripped of the ``model.``
    prefix before comparison (review_R4_part1 P1-1).
    """
    import torch

    from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder
    from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy

    data = CheckpointBuilder.load(final_ckpt)
    bc_policy = load_ddpm_policy(str(bc_ckpt), device=device)
    base_live = bc_policy.model.state_dict()
    prefix = "model."
    base_saved = {k[len(prefix):]: v for k, v in data.policy_state.items()
                  if k.startswith(prefix)}
    problems: List[str] = []
    total = 0.0
    for k, v in base_live.items():
        got = base_saved.get(k)
        if got is None:
            problems.append(f"base key missing from policy_state: {k}")
            continue
        g = np.asarray(got, dtype=np.float64)
        w = np.asarray(v.detach().cpu(), dtype=np.float64)
        total += float(np.sum((g - w) ** 2))
    extra = set(base_saved) - set(base_live)
    if extra:
        problems.append(f"unexpected policy_state model keys: {sorted(extra)[:5]}")
    return math.sqrt(total), problems


def _ckpt_chain_checks(final_ckpt: Path,
                       expected_updates: int) -> Tuple[Dict[str, Any],
                                                        List[str]]:
    """Final checkpoint counters + env-step accumulation across the
    checkpoint chain in the same directory."""
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder

    problems: List[str] = []
    data = CheckpointBuilder.load(final_ckpt)
    counts = {
        "ppo_update_count": data.ppo_update_count,
        "environment_transition_count": data.environment_transition_count,
        "outer_iteration": data.outer_iteration,
    }
    if data.ppo_update_count != expected_updates:
        problems.append(
            f"final ppo_update_count {data.ppo_update_count} != "
            f"{expected_updates}"
        )
    chain = sorted(final_ckpt.parent.glob("dppo_uav_*.pt"))
    env_steps = [
        CheckpointBuilder.load(p).environment_transition_count for p in chain
    ]
    strictly_increasing = all(
        b > a for a, b in zip(env_steps, env_steps[1:])
    )
    if not strictly_increasing or len(chain) < 2:
        problems.append(
            f"checkpoint chain env_steps not strictly increasing: "
            f"{env_steps[:5]}… ({len(chain)} checkpoints)"
        )
    return counts, problems


def _eval_side_summary(artifact: Dict[str, Any], side: str) -> Dict[str, float]:
    """R6-EVAL-1 (§11.5.1): the artifact's episodes are reconstructed
    into paired records and aggregated through the statistics layer with
    the artifact's OWN provenance — dedup-honest sr/cr over the unique
    units, the R5.4 RNG-provenance re-derivation live, and the §11.5.5
    three-n report (raw/unique/cluster) instead of the old raw len(eps)
    pooling that double-counted duplicate BC units. The contract keys
    sr/cr/timeout_rate/n_episodes are preserved; n_episodes is the
    effective (unique) n, never the raw count.
    """
    # Function-local imports (file convention: the gate defers heavy
    # imports to the call site — `import torch` happens inside main()).
    from diffusion_policy_3d.rl.dppo_uav.eval_statistics import (
        aggregate_binary_outcomes,
    )
    from diffusion_policy_3d.rl.dppo_uav.evaluator import (
        EpisodePairedRecord,
    )

    bank_id = artifact.get("bank_id")
    if not bank_id:
        raise ValueError(
            "eval artifact without a bank_id cannot feed the statistics "
            "layer — the R5.4 RNG-provenance re-derivation needs the "
            "bank identity"
        )
    records = [EpisodePairedRecord(**ep) for ep in artifact["episodes"]]
    agg = aggregate_binary_outcomes(
        records,
        bank_id=bank_id,
        side=side,
        policy_checkpoint_hash=artifact.get("checkpoint_hash"),
        # The BC side's unit identity is the shared BC reference from the
        # §11.6.1 load contract — records from every RL artifact collapse
        # onto one unit per (episode, replica); ignored for side="rl".
        bc_checkpoint_hash=(
            artifact.get("load_contract", {}).get("bc_checkpoint_hash")
            if side == "bc" else None
        ),
    )
    n = agg["effective_n"]
    return {
        "sr": agg["sr"],
        "cr": agg["cr"],
        "timeout_rate": float(agg["raw_events"]["timeout"]) / n,
        "n_episodes": n,
        # §11.5.5: the full three-n report — never a single n.
        "raw_n": agg["raw_n"],
        "unique_n": agg["unique_n"],
        "cluster_n": agg["cluster_n"],
        "policy_clusters": agg["policy_clusters"],
    }


def _capability_outcome_verdict(
    outcomes: Any,
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """R6-EVAL-2 (§11.5.3): the capability audit is driven by the
    artifact's five-piece outcomes via capability_informativeness — a
    bank without opportunity evidence reports NOT_INFORMATIVE (BLOCKED
    channel, §11.4.4), NEVER the old presence-only PASS that let a real
    frozen bank (0 opportunity) through. Returns (per-group verdicts,
    blocked entries); a missing/malformed outcomes block raises
    ValueError (structural defect — the schema requires it), which the
    caller folds into ``problems``.
    """
    from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
        capability_informativeness,
    )
    from diffusion_policy_3d.rl.dppo_uav.evaluator import (
        CAPABILITY_METRICS,
    )

    if not isinstance(outcomes, dict) or set(outcomes) != set(CAPABILITY_METRICS):
        raise ValueError(
            "artifact capability_outcomes must cover exactly the three "
            "capability groups — the five-piece contract (§11.5.3) is "
            "required, never a silent default"
        )
    verdicts = capability_informativeness(outcomes)
    blocked: List[str] = []
    for group in sorted(verdicts):
        verdict = verdicts[group]
        if verdict["status"] != "INFORMATIVE":
            blocked.append(
                f"capability {group}: {verdict['status']} — "
                f"{verdict['reason']}"
            )
    return verdicts, blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--eval-artifact", required=True)
    parser.add_argument("--final-ckpt", required=True)
    parser.add_argument("--bc-ckpt", default=DEFAULT_BC_CKPT)
    parser.add_argument("--bc-variance-json", default=None,
        help="BC action-variance baseline measured on the FROZEN bank "
        "via the E1-paired pipeline (review F6: the old 'first RL batch "
        "variance' baseline was not a BC measurement). REQUIRED for the "
        "action-variance gate; without it the gate is BLOCKED, never "
        "silently satisfied",
    )
    parser.add_argument("--expected-updates", type=int, default=30)
    parser.add_argument("--ft-steps", type=int, default=5,
        help="fine-tune denoise step count (every index must carry a "
        "live KL stream — R5.6 all-k gate)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (base-UNet comparison loads the BC ckpt)")

    problems: List[str] = []
    records, m_problems = _load_metrics(Path(args.metrics_jsonl))
    problems += m_problems
    if len(records) != args.expected_updates:
        problems.append(
            f"metrics.jsonl has {len(records)} batches, expected "
            f"{args.expected_updates}"
        )

    # ── per-batch gate stream ─────────────────────────────────────
    actor_nonzero = 0
    unexec_max = 0.0
    kl_clip_structured = 0
    routing_faults = 0
    variance_mins: List[float] = []
    variance_first: List[float] = []
    per_k_kl: Dict[int, List[float]] = {}
    # R6-C (§11.4.4): CAPABILITY_INFORMATION opportunity — batches that
    # actually carry a success/collision-conditioned sample.
    conditioned_opportunity_batches = 0
    per_denoise_index_ok = True
    outcome_invariants_ok = True
    for rec in records:
        m = rec["metrics"]
        # R6-C (§11.4.1/§11.4.2): ENGINE-side re-verification of the
        # per-denoise 5-index contract and the outcome invariants —
        # current-schema records only (legacy records are historical
        # evidence and already marked as such).
        if rec.get("schema_version") == GATE_SCHEMA_VERSION:
            _before = len(problems)
            check_per_denoise(m, args.ft_steps, problems)
            if len(problems) > _before:
                per_denoise_index_ok = False
            _before = len(problems)
            check_outcome_invariants(rec, problems)
            if len(problems) > _before:
                outcome_invariants_ok = False
            for field in ("success_conditioned_advantage",
                          "collision_conditioned_advantage"):
                stat = m.get(field)
                if (isinstance(stat, dict)
                        and stat.get("sample_count", 0) > 0):
                    conditioned_opportunity_batches += 1
        if m.get("actor_grad_norm", 0.0) > 0 and m.get(
            "parameter_update_norm", 0.0
        ) > 0:
            actor_nonzero += 1
        unex = m.get("executed_vs_unexecuted_grad_norm", {}).get(
            "unexecuted", 0.0
        )
        unexec_max = max(unexec_max, float(unex))
        kl = m.get("per_denoise_approx_kl", {})
        clip = m.get("per_denoise_clipfrac", {})
        kl_vals = [v for v in kl.values() if v == v]
        clip_vals = [v for v in clip.values() if v == v]
        if kl_vals and clip_vals:
            kl_clip_structured += 1
        # R5.6 (review F6): per-index KL stream — a denoise index whose
        # KL is EXACTLY zero in every batch is a dead transition (the
        # F1 signature), never a "stable" observation.
        for k, v in kl.items():
            if v == v:  # finite
                per_k_kl.setdefault(int(k), []).append(float(v))
        routing = rec.get("routing", {})
        routing_faults += sum(
            int(routing.get(k, 0))
            for k in ("REJECT_EPISODE", "SAFE_END_EPISODE",
                      "EXCLUDE_TRANSITION", "HARD_FAILURE")
        )
        var = m.get("action_variance_by_dim", [])
        if var:
            variance_mins.append(float(np.mean(var)))
            if len(variance_first) == 0:
                variance_first.append(float(np.mean(var)))

    if actor_nonzero != len(records):
        problems.append(
            f"actor update nonzero rate {actor_nonzero}/{len(records)}"
        )
    if unexec_max > 1e-8:
        problems.append(f"unexecuted PPO grad max {unexec_max} > 1e-8")
    if kl_clip_structured != len(records):
        problems.append(
            f"KL/clipfrac structured output {kl_clip_structured}/"
            f"{len(records)} batches"
        )
    # R5.6: every FT index must have moved at least once — an all-zero
    # per-index KL stream is the F1 dead-transition signature.
    dead_k = [k for k, vals in sorted(per_k_kl.items())
              if all(v == 0.0 for v in vals)]
    if dead_k:
        problems.append(
            f"denoise indices with KL identically zero across all "
            f"batches: {dead_k} (dead transitions — F1 signature)"
        )
    missing_k = [k for k in range(int(args.ft_steps))
                 if k not in per_k_kl]
    if missing_k:
        problems.append(
            f"denoise indices with no KL stream at all: {missing_k}"
        )
    if routing_faults != 0:
        problems.append(f"routing faults sum {routing_faults} != 0")

    variance_ok = True
    bc_var = None
    if args.bc_variance_json:
        bc_doc = json.loads(Path(args.bc_variance_json).read_text())
        bc_var = float(bc_doc["action_variance_mean"])
        if variance_mins:
            min_frac = min(variance_mins) / bc_var if bc_var > 0 else 0.0
            if min_frac < 0.10:
                problems.append(
                    f"action variance min {min_frac:.3f} of frozen-BC "
                    "baseline < 0.10"
                )
                variance_ok = False
        else:
            problems.append("action variance stream empty")
            variance_ok = False
    else:
        # R5.6 (review F6): without a frozen-bank BC measurement the
        # variance gate is BLOCKED — the old 'first RL batch variance'
        # baseline is NOT a BC baseline and must never gate.
        problems.append(
            "action-variance gate BLOCKED: --bc-variance-json missing "
            "(the BC baseline must come from a frozen-bank paired BC run)"
        )
        variance_ok = False

    # ── checkpoint chain ───────────────────────────────────────────
    counts, ckpt_problems = _ckpt_chain_checks(
        Path(args.final_ckpt), args.expected_updates
    )
    problems += ckpt_problems

    # ── base UNet frozen ───────────────────────────────────────────
    base_norm, base_problems = _base_unet_diff(
        Path(args.final_ckpt), Path(args.bc_ckpt), args.device
    )
    problems += base_problems
    if base_norm != 0.0:
        problems.append(f"base UNet update norm {base_norm} != 0")

    # ── eval artifact ──────────────────────────────────────────────
    artifact_path = Path(args.eval_artifact)
    artifact = json.loads(artifact_path.read_text())
    # R6-EVAL-1: the statistics layer can hard-fail on a corrupt artifact
    # (missing bank_id, tampered RNG provenance, conflicting duplicates,
    # empty records) — that IS a gate failure; record it as a problem and
    # carry NaN placeholders so the report stays complete instead of
    # crashing mid-gate.
    try:
        bc_summary = _eval_side_summary(artifact, "bc")
        rl_summary = _eval_side_summary(artifact, "rl")
    except ValueError as e:
        problems.append(f"eval artifact statistics: {e}")
        bc_summary = {"sr": float("nan"), "cr": float("nan"),
                      "timeout_rate": float("nan"), "n_episodes": 0}
        rl_summary = dict(bc_summary)
    # R6-EVAL-2 (§11.5.3): the capability audit is driven by the
    # five-piece outcomes — NOT the presence of the numerics block.
    # A missing/malformed outcomes block is a STRUCTURAL defect (the
    # schema requires it) and lands in ``problems``; a well-formed
    # block with no opportunity evidence lands in the BLOCKED channel.
    try:
        capability_verdicts, capability_blocked_outcome = \
            _capability_outcome_verdict(artifact.get("capability_outcomes"))
    except ValueError as e:
        problems.append(f"artifact capability_outcomes: {e}")
        capability_verdicts = {}
        capability_blocked_outcome = []
    if not artifact.get("bank_id"):
        problems.append("artifact bank_id missing")
    if artifact.get("use_obstacle_projection") is not False:
        problems.append("artifact projection not False")
    if not artifact.get("lora_checksum"):
        problems.append("artifact LoRA checksum missing")
    if artifact.get("checkpoint_hash") != _file_sha256(
        Path(args.final_ckpt)
    ):
        problems.append("artifact checkpoint_hash != final ckpt file hash")
    if not artifact.get("capability_metrics"):
        problems.append("artifact capability_metrics empty")
    if bc_summary["cr"] >= 1.0:
        problems.append(
            f"BC side CR {bc_summary['cr']} == 1.0 (E040-mode regression)"
        )
    admission_blocked = [
        rec.get("admission_blocked_episodes", 0) for rec in records
    ]
    if any(admission_blocked):
        problems.append(
            f"admission-blocked episodes present: {admission_blocked}"
        )

    # R6-C (§11.4.4): three-way gate layering — ENGINE (stop training on
    # any failure), UPDATE HEALTH (tunability calibration; thresholds
    # frozen BEFORE G8), CAPABILITY INFORMATION (opportunity sufficiency;
    # BLOCKED never invalidates the engine verdict).
    engine_hard = {
        "strict_json_no_unexpected_nonfinite": not m_problems,
        "fault_routing_eq_0": routing_faults == 0,
        "per_denoise_5_index_live": per_denoise_index_ok,
        "outcome_invariants": outcome_invariants_ok,
        "base_update_eq_0": base_norm == 0.0 and not base_problems,
        "unexecuted_ppo_grad_leq_1e-8": unexec_max <= 1e-8,
        "checkpoint_resume_ok": not ckpt_problems,
        "bc_cr_lt_1": bc_summary["cr"] < 1.0,
        "artifact_audit_pass": (
            bool(artifact.get("bank_id"))
            and artifact.get("use_obstacle_projection") is False
            and bool(artifact.get("lora_checksum"))
            and artifact.get("checkpoint_hash")
            == _file_sha256(Path(args.final_ckpt))
            and bool(artifact.get("capability_metrics"))
            # R6-EVAL-2: the five-piece outcome contract is part of the
            # artifact schema — structural audit requires it (capability
            # audit separately BLOCKED on NOT_INFORMATIVE, §11.5.3).
            and bool(artifact.get("capability_outcomes"))
        ),
    }
    update_health = {
        "actor_update_nonzero_rate_100": actor_nonzero == len(records),
        "kl_clipfrac_structured": kl_clip_structured == len(records),
        "action_variance_geq_10pct_bc": variance_ok,
    }
    capability_info = {
        "conditioned_advantage_opportunity": conditioned_opportunity_batches > 0,
        # R6-EVAL-2: per-group capability_informativeness verdicts —
        # the gate's PASS requires opportunity evidence, never artifact
        # field presence (Gate B: a real frozen bank with 0 opportunity
        # reported PASS under the old presence check).
        "capability_opportunity_present": bool(capability_verdicts) and any(
            v["status"] == "INFORMATIVE"
            for v in capability_verdicts.values()
        ),
        "capability_informativeness": {
            group: v["status"]
            for group, v in sorted(capability_verdicts.items())
        },
    }
    # §11.4.4: capability insufficiency is reported as BLOCKED, never as
    # an engine failure — it must not erase the engine/health verdict.
    # BLOCKED lives in its own channel ("capability_blocked"), outside
    # ``problems``, so ``status`` stays driven by engine+health alone.
    capability_blocked: List[str] = []
    if not capability_info["conditioned_advantage_opportunity"]:
        capability_blocked.append(
            "conditioned advantage opportunity: no batch carried a "
            "success/collision-conditioned advantage sample"
        )
    capability_blocked.extend(capability_blocked_outcome)
    # "capability" = ENGINE_OK + HEALTH_OK (+ structural problems absent)
    # is still a valid engine verdict; BLOCKED is an honest information
    # statement, not a training failure (G8 engine 否/G9 是 per §12.1).
    capability_verdict = (
        "BLOCKED" if capability_blocked else "PASS"
    )
    gates = {
        "ENGINE_HARD_GATE": engine_hard,
        "UPDATE_HEALTH_GATE": update_health,
        "CAPABILITY_INFORMATION_GATE": capability_info,
    }
    engine_ok = all(engine_hard.values())
    health_ok = all(update_health.values())
    report = {
        "gate": "G7",
        "seed": args.seed,
        "gates": gates,
        "counters": counts,
        "base_unet_update_norm": base_norm,
        "unexecuted_ppo_grad_max": unexec_max,
        "action_variance_min_frac_of_bc": (
            min(variance_mins) / bc_var
            if bc_var is not None and bc_var > 0 and variance_mins
            else None
        ),
        "bc_variance_source": (
            args.bc_variance_json or "BLOCKED (no frozen-bank BC baseline)"
        ),
        "initial_rl_stream_variance_first_batch": (
            variance_first[0] if variance_first else None
        ),
        "per_index_kl_all_zero_somewhere": {
            str(k): all(v == 0.0 for v in vals)
            for k, vals in sorted(per_k_kl.items())
        },
        "routing_fault_sum": routing_faults,
        "admission_blocked_per_batch": admission_blocked,
        "capability_blocked": capability_blocked,
        "capability_verdict": capability_verdict,
        "capability_verdicts": capability_verdicts,
        "eval": {
            "bc": bc_summary,
            "rl": rl_summary,
            "delta_sr": rl_summary["sr"] - bc_summary["sr"],
            "delta_cr": rl_summary["cr"] - bc_summary["cr"],
            "capability_metrics": artifact.get("capability_metrics"),
            # R6-EVAL-2: the five pieces ride the report so the reader
            # sees opportunity/exposure/event counts + null value, not a
            # number with hidden provenance.
            "capability_outcomes": artifact.get("capability_outcomes"),
            "bank_id": artifact.get("bank_id"),
            "checkpoint_hash": artifact.get("checkpoint_hash"),
        },
        "problems": problems,
        "status": "PASS" if not problems and engine_ok and health_ok else "FAIL",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
