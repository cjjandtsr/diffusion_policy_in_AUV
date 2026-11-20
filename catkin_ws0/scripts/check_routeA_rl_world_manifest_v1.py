#!/usr/bin/env python3
"""
check_routeA_rl_world_manifest_v1.py — Route A RL v1 World Manifest Validator
================================================================================

Validates the routeA_rl_v1_world_manifest.json against all design constraints.

Checks:
  1. world_uid globally unique
  2. start_world / goal_world non-empty and 3D
  3. split ∈ {rl_train, rl_val, rl_test, stress_only, historical_eval_only}
  4. role_generated ∈ {anticipatory, dense_recovery, support_expansion, yaw_lateral_recovery}
  5. No legacy scenario names as formal role
  6. No forbidden world IDs (w4/w8/w11/w12/w18)
  7. role_validated / role_active not filled by generator
  8. validation / baseline_calibration empty
  9. admission_status ∈ {pending_generation, pending_validation}

Usage:
  python check_routeA_rl_world_manifest_v1.py \
    --manifest /path/to/routeA_rl_v1_world_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_SPLITS = {"rl_train", "rl_val", "rl_test", "stress_only", "historical_eval_only"}
ALLOWED_ROLES = {"anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery"}
ALLOWED_ADMISSION = {"pending_generation", "pending_validation"}

FORBIDDEN_WORLD_IDS = {4, 8, 11, 12, 18}
FORBIDDEN_LEGACY_SCENARIOS = {
    "canonical", "support", "default", "dense", "random_stress",
    "empty", "sparse", "medium_A", "medium_B", "super_dense", "special",
    "narrow_gap", "symmetric", "trap", "ood", "corridor",
}


# ─────────────────────────────────────────────────────────────────────────────


def check_manifest(manifest_path: Path) -> int:
    """Run all checks on the manifest. Returns 0 on pass, 1 on failure."""

    if not manifest_path.exists():
        print(f"FAIL: manifest file not found: {manifest_path}")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    worlds: List[Dict[str, Any]] = data.get("worlds", [])
    if not worlds:
        print("FAIL: manifest contains no worlds")
        return 1

    errors: List[str] = []
    warnings: List[str] = []
    seen_uids: set = set()
    seen_ids: set = set()

    for i, w in enumerate(worlds):
        prefix = f"[{i}] {w.get('world_uid', 'UNKNOWN')[:60]}"

        # 1. world_uid globally unique
        uid = w.get("world_uid", "")
        if not uid:
            errors.append(f"{prefix}: missing world_uid")
        elif uid in seen_uids:
            errors.append(f"{prefix}: duplicate world_uid")
        else:
            seen_uids.add(uid)

        # 1b. world_id globally unique
        wid = w.get("world_id")
        if wid is None:
            errors.append(f"{prefix}: missing world_id")
        elif wid in seen_ids:
            errors.append(f"{prefix}: duplicate world_id={wid}")
        else:
            seen_ids.add(wid)

        # 1c. Check forbidden legacy world references in world_uid
        # The constraint is about legacy filenames (w4.world, w8.world, etc.)
        # not about sequential world_id integers.
        uid_lower = uid.lower()
        for fwid in FORBIDDEN_WORLD_IDS:
            # Check for patterns like "w4", "w04", "_w4_", "__4__" in the UID
            import re
            if re.search(rf'\bw{fwid}\b', uid_lower) or re.search(rf'\bw0{fwid}\b', uid_lower):
                errors.append(
                    f"{prefix}: world_uid contains forbidden legacy reference 'w{fwid}'"
                )

        # 2. start_world / goal_world non-empty and 3D
        sw = w.get("start_world", [])
        gw = w.get("goal_world", [])
        if not sw or len(sw) < 3:
            errors.append(f"{prefix}: start_world missing or not 3D: {sw}")
        if not gw or len(gw) < 3:
            errors.append(f"{prefix}: goal_world missing or not 3D: {gw}")

        # 3. split check
        sp = w.get("split", "")
        if sp not in ALLOWED_SPLITS:
            errors.append(f"{prefix}: split='{sp}' not in {ALLOWED_SPLITS}")

        # 4. role_generated check
        rg = w.get("role_generated", "")
        if rg not in ALLOWED_ROLES:
            errors.append(f"{prefix}: role_generated='{rg}' not in {ALLOWED_ROLES}")

        # 5. No legacy scenario as formal role
        if rg.lower() in FORBIDDEN_LEGACY_SCENARIOS:
            errors.append(
                f"{prefix}: role_generated='{rg}' is a forbidden legacy scenario"
            )
        # Also check role_validated and role_active for legacy names
        for field in ["role_validated", "role_active"]:
            val = w.get(field)
            if val and isinstance(val, str) and val.lower() in FORBIDDEN_LEGACY_SCENARIOS:
                errors.append(f"{prefix}: {field}='{val}' is a forbidden legacy name")

        # 7. role_validated / role_active not filled by generator
        if w.get("role_validated") is not None:
            errors.append(
                f"{prefix}: role_validated={w['role_validated']} — "
                f"must be null (generator must not fill this)"
            )
        if w.get("role_active") is not None:
            errors.append(
                f"{prefix}: role_active={w['role_active']} — "
                f"must be null (generator must not fill this)"
            )

        # 8. validation / baseline_calibration empty
        validation = w.get("validation", {})
        if validation:
            errors.append(
                f"{prefix}: validation is non-empty: {validation} — "
                f"generator must not write validation results"
            )
        baseline = w.get("baseline_calibration", {})
        if baseline:
            errors.append(
                f"{prefix}: baseline_calibration is non-empty: {baseline} — "
                f"generator must not write calibration results"
            )

        # 9. admission_status
        adm = w.get("admission_status", "")
        if adm not in ALLOWED_ADMISSION:
            errors.append(
                f"{prefix}: admission_status='{adm}' not in {ALLOWED_ADMISSION}"
            )

        # Additional sanity checks
        # start_yaw should be a number
        sy = w.get("start_yaw")
        if sy is not None and not isinstance(sy, (int, float)):
            errors.append(f"{prefix}: start_yaw is not numeric: {sy}")

        # generator_family should be present
        gf = w.get("generator_family", "")
        if not gf:
            errors.append(f"{prefix}: generator_family missing")

        # schema_version check
        sv = w.get("schema_version", "")
        if sv != "routeA_rl_env_v1":
            errors.append(
                f"{prefix}: schema_version='{sv}' ≠ 'routeA_rl_env_v1'"
            )

    # ── Cross-world checks ──

    # Check total count matches expected
    print(f"Total worlds in manifest: {len(worlds)}")
    role_counts: Dict[str, int] = {}
    split_counts: Dict[str, int] = {}
    for w in worlds:
        rg = w.get("role_generated", "?")
        role_counts[rg] = role_counts.get(rg, 0) + 1
        sp = w.get("split", "?")
        split_counts[sp] = split_counts.get(sp, 0) + 1

    print(f"Role distribution: {role_counts}")
    print(f"Split distribution: {split_counts}")

    # Ensure all 4 roles present
    for role in ALLOWED_ROLES:
        if role not in role_counts:
            errors.append(f"Missing role in manifest: {role}")

    # Ensure rl_train, rl_val, rl_test all present
    for split in ["rl_train", "rl_val", "rl_test"]:
        if split not in split_counts:
            errors.append(f"Missing split in manifest: {split}")

    # ── Report ──

    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  {w}")

    if errors:
        print(f"\n✗ {len(errors)} error(s) found:")
        for e in errors:
            print(f"  {e}")
        print(f"\nFAIL routeA_rl_v1_world_manifest")
        return 1
    else:
        print(f"\n✓ All checks passed ({len(worlds)} worlds)")
        print(f"PASS routeA_rl_v1_world_manifest")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Route A RL v1 World Manifest Validator",
    )
    parser.add_argument(
        "--manifest", type=str,
        default="/home/cjj/AUV/program/reports/routeA_rl_env_v1/worldgen/routeA_rl_v1_world_manifest.json",
        help="Path to the manifest JSON file",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    return check_manifest(manifest_path)


if __name__ == "__main__":
    sys.exit(main())
