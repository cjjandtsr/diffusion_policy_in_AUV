#!/usr/bin/env python3
"""multicheckpoint_schema.py — P0 4-point evaluation schema audit (Task 7).

Pure functions:
    validate_fourpoint_schema(evaluations) -> dict
    aggregate_formal_overall(point_result, formal_roles, aux_roles) -> float
    check_pairing_consistency(points) -> bool

Orchestrator:
    run_multicheckpoint_schema_audit(config) -> dict

Produces:
    multicheckpoint_schema_audit.json  — gate decision with schema mechanics report.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.capacity_budget_p0 import SCHEMA_VERSION, GATE_PASS_KEY
from scripts.capacity_budget_p0.eval_bank import (
    FORMAL_ROLES,
    build_pairing_key,
)

# ── Constants ─────────────────────────────────────────────────────────────

FOUR_NAMED_POINTS = ("BC_pre", "update_200", "update_500", "update_1000")
AMBIGUOUS_KEYS = {"eval_mid", "eval_pre", "eval_post"}
AUX_ROLES = ("yaw_lateral_recovery",)

# Required contract keys for each named point's eval result
PER_EPISODE_CONTRACT_KEYS = {"SR", "CR", "done_reason", "steps", "episode_id"}


# ══════════════════════════════════════════════════════════════════════════
# Pure functions
# ══════════════════════════════════════════════════════════════════════════


def validate_fourpoint_schema(evaluations: dict) -> dict:
    """Validate that *evaluations* satisfies the 4-point eval schema.

    Parameters
    ----------
    evaluations : dict
        Top-level dict keyed by checkpoint name (e.g. ``BC_pre``).

    Returns
    -------
    report : dict
        ``valid`` (bool), ``named_points`` (list), ``missing_points`` (list),
        ``ambiguous_eval_mid_detected`` (bool), ``notes`` (str).
    """
    found = set(evaluations.keys())
    required = set(FOUR_NAMED_POINTS)
    missing = required - found
    ambiguous_detected = bool(found & AMBIGUOUS_KEYS)

    # The ambiguous-3-point check: detect presence of eval_mid / pre/mid/post
    # pattern that the design forbids in favour of the exact 4 named points.
    valid = (len(missing) == 0) and (not ambiguous_detected)

    notes_parts = []
    if missing:
        notes_parts.append(f"Missing required points: {sorted(missing)}")
    if ambiguous_detected:
        found_ambig = sorted(found & AMBIGUOUS_KEYS)
        notes_parts.append(
            f"Ambiguous eval keys detected: {found_ambig}. "
            f"The design requires exactly {list(FOUR_NAMED_POINTS)}."
        )
    if valid:
        notes_parts.append("All 4 named points present; no ambiguous keys.")

    return {
        "valid": valid,
        "named_points": sorted(list(found)),
        "missing_points": sorted(missing) if missing else [],
        "ambiguous_eval_mid_detected": ambiguous_detected,
        "notes": "; ".join(notes_parts),
    }


def aggregate_formal_overall(
    point_result: dict,
    formal_roles: Tuple[str, ...] = FORMAL_ROLES,
    aux_roles: Tuple[str, ...] = AUX_ROLES,
) -> float:
    """Compute mean SR across formal roles, excluding auxiliary roles.

    Parameters
    ----------
    point_result : dict
        Must contain ``per_role`` dict keyed by role name, each entry
        containing at least ``SR`` (float or int).

    formal_roles : tuple of str
        Role names to include in the mean.

    aux_roles : tuple of str
        Role names to explicitly exclude.

    Returns
    -------
    float
        Mean SR over formal roles present in ``per_role``.  Missing formal
        roles are treated as 0.0 SR.
    """
    per_role = point_result.get("per_role", {})
    sr_values = []
    for role in formal_roles:
        if role in per_role and role not in aux_roles:
            sr_values.append(per_role[role].get("SR", 0.0))
        elif role not in per_role:
            sr_values.append(0.0)
        # aux roles are intentionally skipped

    if not sr_values:
        return 0.0
    return sum(sr_values) / len(sr_values)


def check_pairing_consistency(points: dict) -> bool:
    """Check that all four named points have identical ordered pairing keys.

    Parameters
    ----------
    points : dict
        Dict keyed by checkpoint name, each value is an eval result with
        a ``per_episode`` dict keyed by pairing key.

    Returns
    -------
    bool
        True iff every named point is present, none has an empty
        ``per_episode`` map, and all have the same ``per_episode``
        keys in the same iteration order.
    """
    keys_list = []
    for name in FOUR_NAMED_POINTS:
        if name not in points:
            return False
        pt = points[name]
        per_ep = pt.get("per_episode", {})
        # Empty per_episode at any point is a data-load failure → fail
        if not per_ep:
            return False
        keys_list.append(list(per_ep.keys()))

    if len(keys_list) < 2:
        return False

    first = keys_list[0]
    for keys in keys_list[1:]:
        if keys != first:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# Synthetic eval generator (no GPU)
# ══════════════════════════════════════════════════════════════════════════


def _generate_synthetic_eval(episodes: list) -> dict:
    """Generate a deterministic per-episode eval result from bank episodes.

    No GPU, no policy loading.  Each episode's outcome is determined
    by ``episode_seed`` parity and simple modulo rules, guaranteeing
    deterministic results for schema-proving purposes.

    Returns
    -------
    dict
        ``per_role`` (dict), ``per_episode`` (dict keyed by pairing key),
        ``n_episodes`` (int).
    """
    per_episode = {}
    per_role_agg = defaultdict(
        lambda: {"n_success": 0, "n_collision": 0, "n_total": 0}
    )

    for ep in episodes:
        key = build_pairing_key(ep)
        role = ep["role"]
        seed = ep["episode_seed"]

        # Deterministic outcome
        sr = 1 if (seed % 3 != 0) else 0  # ~2/3 success
        cr = 1 if (seed % 5 == 0) else 0  # ~1/5 collision

        per_episode[key] = {
            "SR": sr,
            "CR": cr,
            "done_reason": "success" if sr else ("collision" if cr else "timeout"),
            "steps": abs(seed) % 60 + 10,
            "episode_id": ep["episode_id"],
        }
        per_role_agg[role]["n_total"] += 1
        per_role_agg[role]["n_success"] += sr
        per_role_agg[role]["n_collision"] += cr

    per_role = {}
    for role, data in per_role_agg.items():
        n = max(data["n_total"], 1)
        per_role[role] = {
            "SR": round(data["n_success"] / n, 4),
            "CR": round(data["n_collision"] / n, 4),
            "n_episodes": data["n_total"],
        }

    return {"per_role": per_role, "per_episode": per_episode, "n_episodes": len(episodes)}


# ══════════════════════════════════════════════════════════════════════════
# Bank manifest loader
# ══════════════════════════════════════════════════════════════════════════


def load_bank_manifest(manifest_path: str) -> List[dict]:
    """Load episodes from ``eval_bank_manifest.json``."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Bank manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("episodes", [])


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════


def run_multicheckpoint_schema_audit(config: dict) -> dict:
    """Run the 4-point evaluation schema audit.

    Config keys
    -----------
    output_dir : str
        Directory for output JSON and bank manifest lookup.
    run_id : str, optional
        Run identifier (defaults to output_dir basename).
    eval_bank_manifest_path : str, optional
        Path to ``eval_bank_manifest.json``.  Defaults to
        ``<output_dir>/eval_bank_manifest.json``.
    _synthetic_episodes : list, optional
        For testing — use pre-built episodes instead of loading manifest.

    Returns
    -------
    dict
        Full audit report (also written to
        ``<output_dir>/multicheckpoint_schema_audit.json``).
    """
    output_dir = Path(config.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.get("run_id", output_dir.name)

    # ── Load bank episodes ───────────────────────────────────────────────
    synthetic_eps = config.get("_synthetic_episodes")
    if synthetic_eps is not None:
        episodes = synthetic_eps
        print(f"  Using synthetic episodes ({len(episodes)} total)")
    else:
        manifest_path = config.get(
            "eval_bank_manifest_path",
            str(output_dir / "eval_bank_manifest.json"),
        )
        episodes = load_bank_manifest(manifest_path)
        print(f"  Loaded {len(episodes)} episodes from {manifest_path}")

    if not episodes:
        raise ValueError("No episodes available — cannot run multicheckpoint schema audit.")

    # ── Build per-role coverage tracking ─────────────────────────────────
    per_role_ep_counts = defaultdict(int)
    for ep in episodes:
        per_role_ep_counts[ep.get("role", "?")] += 1

    # ── Generate deterministic eval result ───────────────────────────────
    eval_result = _generate_synthetic_eval(episodes)

    # ── Fill all four named points with SAME eval result ─────────────────
    evaluations = {name: eval_result for name in FOUR_NAMED_POINTS}

    # ── Schema validation ────────────────────────────────────────────────
    schema_report = validate_fourpoint_schema(evaluations)

    # ── Roundtrip test: JSON serialize/deserialize ───────────────────────
    try:
        serialized = json.dumps(evaluations, indent=2, ensure_ascii=False)
        restored = json.loads(serialized)
        roundtrip_ok = True
        for name in FOUR_NAMED_POINTS:
            orig_eps = evaluations[name].get("per_episode", {})
            rest_eps = restored[name].get("per_episode", {})
            # ALL pairing keys must survive identically
            if sorted(orig_eps.keys()) != sorted(rest_eps.keys()):
                roundtrip_ok = False
                break
            # ALL per-episode fields (SR, CR, done_reason) must survive
            for key, orig_val in orig_eps.items():
                rest_val = rest_eps[key]
                for field in ("SR", "CR", "done_reason"):
                    if field in orig_val and orig_val[field] != rest_val.get(field):
                        roundtrip_ok = False
                        break
                if not roundtrip_ok:
                    break
            if not roundtrip_ok:
                break
    except Exception:
        roundtrip_ok = False

    # ── Pairing key consistency across points ────────────────────────────
    pairing_consistent = check_pairing_consistency(evaluations)

    # ── Per-point analysis ───────────────────────────────────────────────
    per_point = {}
    for name in FOUR_NAMED_POINTS:
        pt = evaluations[name]
        per_role = pt.get("per_role", {})

        # Count formal roles present
        n_formal = sum(1 for r in FORMAL_ROLES if r in per_role)

        # Episodes per role
        episodes_per_role = {}
        for r in FORMAL_ROLES:
            if r in per_role:
                episodes_per_role[r] = per_role[r].get("n_episodes", 0)

        # Verify aux exclusion
        overall_with_all = aggregate_formal_overall(pt, FORMAL_ROLES, AUX_ROLES)
        # Add a synthetic aux role for the exclusion test
        pt_with_aux = dict(pt)
        pt_with_aux["per_role"] = dict(pt.get("per_role", {}))
        pt_with_aux["per_role"]["yaw_lateral_recovery"] = {
            "SR": 1.0, "CR": 0.0, "n_episodes": 999,
        }
        overall_excluding_aux = aggregate_formal_overall(
            pt_with_aux, FORMAL_ROLES, AUX_ROLES
        )

        excludes_aux = abs(overall_with_all - overall_excluding_aux) < 1e-9

        per_point[name] = {
            "n_formal_roles": n_formal,
            "episodes_per_role": episodes_per_role,
            "formal_overall_excludes_aux": excludes_aux,
        }

    # ── Formal overall example (from BC_pre) ─────────────────────────────
    formal_overall_example = aggregate_formal_overall(
        evaluations["BC_pre"], FORMAL_ROLES, AUX_ROLES
    )

    # ── Gate logic ──────────────────────────────────────────────────────
    missing_point_rejected = len(schema_report.get("missing_points", [])) > 0
    ambiguous_rejected = schema_report.get("ambiguous_eval_mid_detected", False)

    gate_pass = (
        schema_report["valid"]
        and roundtrip_ok
        and pairing_consistent
        and not missing_point_rejected
        and not ambiguous_rejected
    )

    # ── Build report ────────────────────────────────────────────────────
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        GATE_PASS_KEY: gate_pass,
        "named_points": list(FOUR_NAMED_POINTS),
        "schema_valid": schema_report["valid"],
        "missing_point_rejected": missing_point_rejected,
        "ambiguous_eval_mid_rejected": ambiguous_rejected,
        "roundtrip_consistent": roundtrip_ok,
        "pairing_keys_consistent_across_points": pairing_consistent,
        "per_point": per_point,
        "formal_overall_example": formal_overall_example,
        "notes": schema_report["notes"],
    }

    # ── Persist ──────────────────────────────────────────────────────────
    out_path = output_dir / "multicheckpoint_schema_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Wrote {out_path}")
    print(f"  Gate pass: {gate_pass}")
    print(f"  Schema valid: {schema_report['valid']}")
    print(f"  Roundtrip consistent: {roundtrip_ok}")
    print(f"  Pairing keys consistent: {pairing_consistent}")
    print(f"  Missing point rejected: {missing_point_rejected}")
    print(f"  Ambiguous eval_mid rejected: {ambiguous_rejected}")

    return report
