#!/usr/bin/env python3
"""
routea_rl_env_adapter_v1.py — Route A RL v1 Environment Adapter

Connects the new role-driven world pool (admission_manifest.json) to Route A
A0/A1/A2 training. Provides:

  - Admission manifest loading and validation (Step 1)
  - Role allowlists (Step 2)
  - Historical contamination guard (Step 3)
  - Legacy scenario adapter (Step 4)
  - Role-balanced sampler (Step 5)
  - Dry-run admission check (Step 13)

Design: zhidao/Guide_RouteA_RL_Training_On_NewEnv_v1.md
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Role allowlists
# ═══════════════════════════════════════════════════════════════════════════════

FORMAL_CAPABILITY_ROLES: Tuple[str, ...] = (
    "anticipatory", "dense_recovery", "support_expansion"
)
SUPPORT_ROLES: Tuple[str, ...] = ("yaw_lateral_recovery",)
NON_TRAIN_ROLES: Tuple[str, ...] = (
    "historical_eval_only", "stress_only", "guard_eval_only"
)
ALL_ROUTEA_RL_ROLES: Set[str] = set(
    FORMAL_CAPABILITY_ROLES + SUPPORT_ROLES + NON_TRAIN_ROLES
)

TRAINABLE_ROLES: Tuple[str, ...] = (
    "anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Historical contamination guard
# ═══════════════════════════════════════════════════════════════════════════════

# Historical world identifiers — only the w-prefixed forms (structural matching).
# Raw digits ("4", "8", etc.) are NOT used because they match seed suffixes (__000004 etc.).
HISTORICAL_EVAL_WORLD_IDS: Set[str] = {"w4", "w8", "w11", "w12", "w18"}

# Regex patterns that match old historical world identifiers.
# Requires structural boundaries: w4, w8, w11, w12, w18 surrounded by _ - or string boundaries.
HISTORICAL_UID_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?:^|[_-])w(?:4|8|11|12|18)(?:[_-]|$)", re.IGNORECASE),
]


def check_historical_contamination(
    world_uid: str,
    world_file: str,
    metadata_file: str,
    split: str,
) -> Optional[str]:
    """Check if a world is a historical eval world that shouldn't enter train/val.

    Uses structural matching only (w4, w8, w11, w12, w18 as word components).
    Raw digits are NOT used to avoid false positives on seed suffixes (e.g. __000004).

    Args:
        world_uid: unique world identifier
        world_file: path to .world file
        metadata_file: path to metadata file
        split: rl_train / rl_val / rl_test / stress_only / historical_eval_only

    Returns:
        None if clean, or a string describing the contamination if detected.
    """
    if split not in ("rl_train", "rl_val"):
        return None  # rl_test / stress_only / historical_eval_only are allowed

    # Check world_uid against historical w-prefixed IDs (structural)
    for hid in HISTORICAL_EVAL_WORLD_IDS:
        if hid in world_uid:
            return f"historical_eval_world_id '{hid}' found in world_uid '{world_uid}'"

    # Check world_uid against historical regex patterns
    for pattern in HISTORICAL_UID_PATTERNS:
        if pattern.search(world_uid):
            return f"historical pattern '{pattern.pattern}' matched in world_uid '{world_uid}'"

    # Check world_file path for historical w-prefixed identifiers
    for hid in HISTORICAL_EVAL_WORLD_IDS:
        if hid in world_file:
            return f"historical_eval_world_id '{hid}' found in world_file '{world_file}'"

    # Check metadata_file path for historical w-prefixed identifiers
    for hid in HISTORICAL_EVAL_WORLD_IDS:
        if hid in metadata_file:
            return f"historical_eval_world_id '{hid}' found in metadata_file '{metadata_file}'"

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Legacy scenario adapter
# ═══════════════════════════════════════════════════════════════════════════════

LEGACY_SCENARIO_TO_ROUTEA_ROLE: Dict[str, str] = {
    "canonical": "historical_anchor_eval",
    "support": "support_expansion_guard_eval",
    "default": "mixed_capability_eval_only",
    "dense": "historical_dense_eval_only",
    "random_stress": "stress_eval_only",
}

# Reverse: which legacy scenarios should NEVER be mapped to trainable roles
LEGACY_FORBIDDEN_TRAIN_MAPPINGS: Dict[str, str] = {
    "legacy_default": "dense_recovery_train",
    "legacy_dense": "dense_recovery_train",
}


def map_legacy_scenario_to_role(legacy_scenario: str) -> str:
    """Map a legacy scenario name to its Route A RL v1 role.

    This adapter is ONLY for legacy log compatibility, historical eval,
    and sanity reference. Actor update must NOT sample by legacy scenario.
    """
    return LEGACY_SCENARIO_TO_ROUTEA_ROLE.get(
        legacy_scenario, f"unknown_legacy_{legacy_scenario}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Admission manifest loader
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RouteARLWorldEntry:
    """Parsed entry for one world in the admission manifest."""
    world_uid: str
    split: str                      # rl_train / rl_val / rl_test / stress_only / historical_eval_only
    role_generated: str
    role_validated: Optional[str]
    role_active: Optional[str]
    admission_status: str           # e.g. "admitted_anticipatory" / "needs_more_eval" / "reject_geometry"
    generator_family: str = ""
    paired_seed_group: str = ""
    world_file: str = ""
    metadata_file: str = ""
    # Calibration metrics
    bc_fm_sr: float = 0.0
    bc_fm_sr_ci_low: float = 0.0
    bc_fm_sr_ci_high: float = 0.0
    bc_fm_cr: float = 0.0
    ego_available: bool = False
    beyond_boundary: bool = False
    # Extra
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteARLAdmissionManifest:
    """Parsed Route A RL v1 admission manifest."""
    schema_version: str
    calibration_date: str
    mode: str
    stats: Dict[str, Any]
    worlds: List[RouteARLWorldEntry]

    def get_worlds_by_split(self, split: str) -> List[RouteARLWorldEntry]:
        return [w for w in self.worlds if w.split == split]

    def get_admitted_worlds(self, split: Optional[str] = None) -> List[RouteARLWorldEntry]:
        result = [w for w in self.worlds if w.admission_status.startswith("admitted")]
        if split is not None:
            result = [w for w in result if w.split == split]
        return result

    def get_worlds_by_role(self, role: str, split: Optional[str] = None) -> List[RouteARLWorldEntry]:
        result = []
        for w in self.worlds:
            effective_role = w.role_active or w.role_validated
            if effective_role == role:
                if split is None or w.split == split:
                    result.append(w)
        return result


def load_routea_rl_admission_manifest(path: str) -> RouteARLAdmissionManifest:
    """Load and validate the Route A RL v1 admission manifest.

    Args:
        path: Path to admission_manifest.json

    Returns:
        RouteARLAdmissionManifest with parsed and validated entries.

    Raises:
        FileNotFoundError: if manifest is missing.
        ValueError: if manifest fails validation orrl_train formal role buckets are empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RouteA RL admission manifest is required before new-env training. "
            f"Expected at: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    schema_version = raw.get("schema_version", "unknown")
    calibration_date = raw.get("calibration_date", "unknown")
    mode = raw.get("mode", "unknown")
    stats = raw.get("stats", {})

    worlds: List[RouteARLWorldEntry] = []
    errors: List[str] = []

    for i, item in enumerate(raw.get("results", [])):
        world_uid = item.get("world_uid", "")
        split = item.get("split", "")
        role_generated = item.get("role_generated", "")
        role_validated = item.get("role_validated")
        role_active = item.get("role_active")
        admission_status = item.get("admission_status", "")

        # Validate required fields
        if not world_uid:
            errors.append(f"world[{i}]: missing world_uid")
            continue
        # rl_eval (design v1.3 '只评估不训练'): the canonical eval pool's
        # eval-only worlds — admitted, never trained (P1-1).
        if split not in ("rl_train", "rl_val", "rl_test", "rl_eval",
                         "stress_only", "historical_eval_only"):
            errors.append(f"world[{i}] {world_uid}: unknown split '{split}'")
        if not role_generated:
            errors.append(f"world[{i}] {world_uid}: missing role_generated")

        # Derive world file path from world_uid
        world_file = f"routeA_rl_v1__{world_uid}.world" if not world_uid.endswith(".world") else world_uid
        metadata_file = world_file.replace(".world", ".metadata.json")

        # Extract generator_family from world_uid: routeA_rl_v1__{role}__{split}__{family}__{seed}
        parts = world_uid.split("__")
        generator_family = parts[3] if len(parts) >= 4 else ""
        paired_seed_group = parts[4] if len(parts) >= 5 else ""

        entry = RouteARLWorldEntry(
            world_uid=world_uid,
            split=split,
            role_generated=role_generated,
            role_validated=role_validated,
            role_active=role_active,
            admission_status=admission_status,
            generator_family=generator_family,
            paired_seed_group=paired_seed_group,
            world_file=world_file,
            metadata_file=metadata_file,
            bc_fm_sr=item.get("bc_fm_sr", 0.0),
            bc_fm_sr_ci_low=item.get("bc_fm_sr_ci_low", 0.0),
            bc_fm_sr_ci_high=item.get("bc_fm_sr_ci_high", 0.0),
            bc_fm_cr=item.get("bc_fm_cr", 0.0),
            ego_available=item.get("ego_available", False),
            beyond_boundary=item.get("beyond_boundary", False),
            raw=item,
        )
        worlds.append(entry)

    if errors:
        raise ValueError(
            f"RouteA RL admission manifest validation failed with {len(errors)} errors:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    manifest = RouteARLAdmissionManifest(
        schema_version=schema_version,
        calibration_date=calibration_date,
        mode=mode,
        stats=stats,
        worlds=worlds,
    )

    return manifest


def validate_rl_train_admission(manifest: RouteARLAdmissionManifest) -> Dict[str, Any]:
    """Validate that rl_train has admitted worlds for all formal capability roles.

    Returns:
        Dict with keys: pass_schema, pass_train_admission, formal_role_buckets,
        support_bucket, historical_train_worlds, stress_train_worlds, errors.
    """
    result: Dict[str, Any] = {
        "pass_schema": True,
        "pass_train_admission": False,
        "formal_role_buckets": {},
        "support_bucket": 0,
        "historical_train_worlds": 0,
        "stress_train_worlds": 0,
        "errors": [],
    }

    # Count admitted worlds per role in rl_train
    train_worlds = manifest.get_worlds_by_split("rl_train")
    admitted_train = [w for w in train_worlds if w.admission_status.startswith("admitted")]

    for role in FORMAL_CAPABILITY_ROLES:
        count = sum(
            1 for w in admitted_train
            if (w.role_active or w.role_validated) == role
        )
        result["formal_role_buckets"][role] = count

    result["support_bucket"] = sum(
        1 for w in admitted_train
        if (w.role_active or w.role_validated) in SUPPORT_ROLES
    )

    # Check for historical contamination in train
    contamination_errors = []
    for w in train_worlds:
        err = check_historical_contamination(
            w.world_uid, w.world_file, w.metadata_file, w.split,
        )
        if err:
            contamination_errors.append(err)

    result["historical_train_worlds"] = len(contamination_errors)
    if contamination_errors:
        result["errors"].extend(contamination_errors)

    # Check stress_only in train
    result["stress_train_worlds"] = sum(
        1 for w in train_worlds if w.admission_status == "stress_only"
    )

    # Determine pass
    formal_empty = any(count == 0 for count in result["formal_role_buckets"].values())
    result["pass_train_admission"] = (
        not formal_empty
        and result["historical_train_worlds"] == 0
    )

    if formal_empty:
        result["errors"].append(
            f"admitted rl_train formal role bucket is empty: "
            + ", ".join(f"{r}={c}" for r, c in result["formal_role_buckets"].items())
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Role-balanced sampler
# ═══════════════════════════════════════════════════════════════════════════════


def build_role_buckets(
    manifest: RouteARLAdmissionManifest,
    split: str,
) -> Dict[str, List[RouteARLWorldEntry]]:
    """Build per-role world buckets from admission manifest for a given split.

    Args:
        manifest: parsed admission manifest
        split: rl_train / rl_val / rl_test

    Returns:
        Dict mapping role → list of admitted worlds for that role+split.
    """
    buckets: Dict[str, List[RouteARLWorldEntry]] = {
        role: [] for role in TRAINABLE_ROLES
    }
    for item in manifest.worlds:
        if item.split != split:
            continue
        role = item.role_active or item.role_validated
        if role in TRAINABLE_ROLES and item.admission_status.startswith("admitted"):
            buckets[role].append(item)
    return buckets


def validate_role_buckets_for_training(
    buckets: Dict[str, List[RouteARLWorldEntry]],
) -> Tuple[bool, str]:
    """Check that all formal capability role buckets are non-empty.

    Returns:
        (pass, reason) — pass=True if all formal roles have at least one world.
    """
    for role in FORMAL_CAPABILITY_ROLES:
        if len(buckets.get(role, [])) == 0:
            return False, (
                f"RouteA RL train requires non-empty formal capability role buckets. "
                f"Role '{role}' has 0 admitted worlds."
            )
    return True, "all formal role buckets non-empty"


def classify_world_difficulty(
    entry: RouteARLWorldEntry,
    bc_fm_sr_median: float = 0.45,
) -> str:
    """Classify a world as easier / frontier / harder based on BC/FM SR.

    Args:
        entry: world entry from admission manifest
        bc_fm_sr_median: median BC/FM SR across the pool

    Returns:
        "easier" | "frontier" | "harder"
    """
    sr = entry.bc_fm_sr
    if sr >= bc_fm_sr_median + 0.15:
        return "easier"
    elif sr <= bc_fm_sr_median - 0.15:
        return "harder"
    else:
        return "frontier"


def sample_role_balanced_worlds(
    buckets: Dict[str, List[RouteARLWorldEntry]],
    n_per_role: int = 1,
    yaw_lateral_ratio: float = 0.2,
    seed: int = 20260703,
) -> List[RouteARLWorldEntry]:
    """Sample worlds with role-balanced distribution.

    Sampling rules:
      - Each training step ensures all 3 formal capability roles appear.
      - yaw_lateral_recovery injected at fixed ratio.
      - Within each role: 25% easier + 50% frontier + 25% harder.

    Args:
        buckets: per-role world lists from build_role_buckets()
        n_per_role: target worlds per role per step
        yaw_lateral_ratio: fraction of yaw_lateral_recovery worlds
        seed: deterministic seed

    Returns:
        List of selected world entries.
    """
    import random
    rng = random.Random(seed)

    selected: List[RouteARLWorldEntry] = []

    # Compute per-role BC/FM SR median for difficulty classification
    all_sr = []
    for role_list in buckets.values():
        for w in role_list:
            if w.bc_fm_sr > 0:
                all_sr.append(w.bc_fm_sr)
    sr_median = sorted(all_sr)[len(all_sr) // 2] if all_sr else 0.45

    for role in FORMAL_CAPABILITY_ROLES:
        role_worlds = buckets.get(role, [])
        if not role_worlds:
            continue

        # Classify by difficulty
        easier = [w for w in role_worlds if classify_world_difficulty(w, sr_median) == "easier"]
        frontier = [w for w in role_worlds if classify_world_difficulty(w, sr_median) == "frontier"]
        harder = [w for w in role_worlds if classify_world_difficulty(w, sr_median) == "harder"]

        # If classification yields empty categories, use all worlds
        if not easier and not harder:
            # All frontier — sample uniformly
            for _ in range(n_per_role):
                if role_worlds:
                    selected.append(rng.choice(role_worlds))
            continue

        n_easier = max(1, int(n_per_role * 0.25))
        n_harder = max(1, int(n_per_role * 0.25))
        n_frontier = n_per_role - n_easier - n_harder

        for _ in range(n_easier):
            pool = easier if easier else role_worlds
            selected.append(rng.choice(pool))
        for _ in range(n_frontier):
            pool = frontier if frontier else role_worlds
            selected.append(rng.choice(pool))
        for _ in range(n_harder):
            pool = harder if harder else role_worlds
            selected.append(rng.choice(pool))

    # Inject yaw_lateral_recovery at fixed ratio
    yaw_worlds = buckets.get("yaw_lateral_recovery", [])
    if yaw_worlds:
        n_yaw = max(1, int(len(selected) * yaw_lateral_ratio / (1.0 - yaw_lateral_ratio)))
        for _ in range(n_yaw):
            selected.append(rng.choice(yaw_worlds))

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Step 13: Dry-run admission check
# ═══════════════════════════════════════════════════════════════════════════════


def dry_run_newenv_loader(
    manifest_path: str,
) -> Dict[str, Any]:
    """Run dry-run validation of the new-env loader without starting training.

    Expected output with current (2026-07-03) smoke manifest:
      PASS routeA_rl_newenv_loader_schema
      BLOCKED routeA_rl_newenv_train_admission
      reason: admitted rl_train formal role bucket is empty
      formal_role_buckets: anticipatory=0 dense_recovery=0 support_expansion=0

    Returns:
        Dict with keys: schema_pass, train_blocked, formal_role_buckets,
        support_bucket, historical_train_worlds, stress_train_worlds,
        errors, summary_lines.
    """
    result: Dict[str, Any] = {
        "schema_pass": False,
        "train_blocked": True,
        "formal_role_buckets": {},
        "support_bucket": 0,
        "historical_train_worlds": 0,
        "stress_train_worlds": 0,
        "errors": [],
        "summary_lines": [],
    }

    # Load manifest (schema check)
    try:
        manifest = load_routea_rl_admission_manifest(manifest_path)
        result["schema_pass"] = True
        result["summary_lines"].append("PASS routeA_rl_newenv_loader_schema")
    except Exception as e:
        result["errors"].append(f"Schema load failed: {e}")
        result["summary_lines"].append(f"FAIL routeA_rl_newenv_loader_schema: {e}")
        return result

    # Train admission check
    admission = validate_rl_train_admission(manifest)
    result["formal_role_buckets"] = admission["formal_role_buckets"]
    result["support_bucket"] = admission["support_bucket"]
    result["historical_train_worlds"] = admission["historical_train_worlds"]
    result["stress_train_worlds"] = admission["stress_train_worlds"]
    result["errors"].extend(admission["errors"])

    if admission["pass_train_admission"]:
        result["train_blocked"] = False
        result["summary_lines"].append("PASS routeA_rl_newenv_loader")
        result["summary_lines"].append(
            f"formal_role_buckets: " +
            ", ".join(f"{r}={c}" for r, c in admission["formal_role_buckets"].items())
        )
        result["summary_lines"].append(
            f"historical_train_worlds: {admission['historical_train_worlds']}"
        )
        result["summary_lines"].append(
            f"stress_train_worlds: {admission['stress_train_worlds']}"
        )
    else:
        result["train_blocked"] = True
        result["summary_lines"].append("BLOCKED routeA_rl_newenv_train_admission")
        result["summary_lines"].append(
            f"reason: admitted rl_train formal role bucket is empty"
        )
        result["summary_lines"].append(
            f"formal_role_buckets: " +
            ", ".join(f"{r}={c}" for r, c in admission["formal_role_buckets"].items())
        )
        result["summary_lines"].append(
            f"support_bucket: yaw_lateral_recovery={admission['support_bucket']}"
        )
        result["summary_lines"].append(
            f"historical_train_worlds: {admission['historical_train_worlds']}"
        )
        result["summary_lines"].append(
            f"stress_train_worlds: {admission['stress_train_worlds']}"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7-9: A0/A1/A2 gate rules for new-env training
# ═══════════════════════════════════════════════════════════════════════════════


def get_newenv_gate_config() -> Dict[str, Any]:
    """Return gate configuration for new-env A0/A1/A2 training.

    These gates are applied IN ADDITION to the existing A0/A1/A2 gates.
    """
    return {
        "a0_newenv": {
            "description": "A0 critic warmup on new env pool",
            "require_formal_roles_nonempty": True,
            "require_nan_inf_zero": True,
            "require_replay_metadata_complete": True,
            "require_historical_contamination_zero": True,
            "forbid_actor_update_on_historical_eval_only": True,
        },
        "a1_newenv": {
            "description": "A1 stability on new env pool",
            "require_no_role_nan_inf": True,
            "require_no_role_collision_explosion": True,
            "require_support_expansion_sr_above_bc_ci_low": True,
            "require_basic_guard_eval_cr_below_bc_ci_high": True,
            "require_no_historical_in_replay": True,
            "forbid_stress_only_in_hard_gate": True,
        },
        "a2_newenv": {
            "description": "A2 capability on new env pool",
            "require_at_least_one_formal_role_pass": True,
            "anticipatory_gate": {
                "onset_distance_improvement_pct": 10.0,
                "future_clearance_improvement": True,
                "cr_not_increased": True,
            },
            "dense_recovery_gate": {
                "trap_recovery_improvement_pct": 10.0,
                "or_deadend_dwell_reduction_pct": 10.0,
                "dense_cr_not_increased": True,
            },
            "support_expansion_gate": {
                "low_density_coverage_improvement_pct": 10.0,
                "support_sr_not_below_bc_ci_low": True,
            },
            "yaw_lateral_recovery_gate": {
                "recovery_sr_improvement_pp": 10.0,
                "or_recovery_time_reduction_pct": 10.0,
                "post_disturb_cr_not_increased": True,
                "note": "support bucket only — not a fourth top-level capability claim",
            },
            "ci_rules": {
                "wilson_overlap_large": "NEED_MORE_EVAL",
                "bootstrap_ci_contains_zero": "no_main_capability_evidence",
                "rl_mean_worse_than_bc_beyond_ci": "fail_hard_veto",
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════


def _self_test():
    """Minimal self-test of the adapter module."""
    import tempfile

    print("=" * 60)
    print("routea_rl_env_adapter_v1 — self-test")
    print("=" * 60)

    # Test role allowlists
    assert "anticipatory" in FORMAL_CAPABILITY_ROLES
    assert "dense_recovery" in FORMAL_CAPABILITY_ROLES
    assert "support_expansion" in FORMAL_CAPABILITY_ROLES
    assert "yaw_lateral_recovery" in SUPPORT_ROLES
    assert "historical_eval_only" in NON_TRAIN_ROLES
    assert FORMAL_CAPABILITY_ROLES + SUPPORT_ROLES == TRAINABLE_ROLES
    print("  ✓ Role allowlists correct")

    # Test historical contamination guard
    err = check_historical_contamination(
        world_uid="routeA_rl_v1__anticipatory__rl_train__front_offset__000000",
        world_file="routeA_rl_v1__anticipatory__rl_train__front_offset__000000.world",
        metadata_file="routeA_rl_v1__anticipatory__rl_train__front_offset__000000.metadata.json",
        split="rl_train",
    )
    assert err is None, f"Clean UID should pass: {err}"
    print("  ✓ Clean world passes contamination check")

    err_bad = check_historical_contamination(
        world_uid="some_w4_world",
        world_file="some_w4_world.world",
        metadata_file="some_w4_world.metadata.json",
        split="rl_train",
    )
    assert err_bad is not None, "w4 should be flagged"
    assert "w4" in err_bad
    print("  ✓ Historical w4 detected in world_uid")

    # Test that rl_test allows historical IDs (eval-only)
    err_test = check_historical_contamination(
        world_uid="some_w4_world",
        world_file="some_w4_world.world",
        metadata_file="some_w4_world.metadata.json",
        split="rl_test",
    )
    assert err_test is None, f"rl_test should allow historical: {err_test}"
    print("  ✓ Historical world allowed in rl_test")

    # Test legacy adapter
    assert map_legacy_scenario_to_role("canonical") == "historical_anchor_eval"
    assert map_legacy_scenario_to_role("default") == "mixed_capability_eval_only"
    assert map_legacy_scenario_to_role("dense") == "historical_dense_eval_only"
    print("  ✓ Legacy scenario adapter correct")

    # Test manifest loading with mock data
    mock_manifest = {
        "schema_version": "routeA_rl_env_v1",
        "calibration_date": "2026-07-03",
        "mode": "real",
        "stats": {
            "total": 4,
            "admitted": 4,
            "anticipatory_admitted": 2,
            "dense_recovery_admitted": 2,
        },
        "results": [
            {
                "world_uid": "routeA_rl_v1__anticipatory__rl_train__front_offset__000000",
                "split": "rl_train",
                "role_generated": "anticipatory",
                "role_validated": "anticipatory",
                "role_active": "anticipatory",
                "admission_status": "admitted_anticipatory",
                "bc_fm_sr": 0.55, "bc_fm_sr_ci_low": 0.40, "bc_fm_sr_ci_high": 0.70,
                "bc_fm_cr": 0.10, "ego_available": True, "beyond_boundary": False,
            },
            {
                "world_uid": "routeA_rl_v1__dense_recovery__rl_train__bottleneck__000001",
                "split": "rl_train",
                "role_generated": "dense_recovery",
                "role_validated": "dense_recovery",
                "role_active": "dense_recovery",
                "admission_status": "admitted_dense_recovery",
                "bc_fm_sr": 0.35, "bc_fm_sr_ci_low": 0.20, "bc_fm_sr_ci_high": 0.50,
                "bc_fm_cr": 0.25, "ego_available": False, "beyond_boundary": False,
            },
            {
                "world_uid": "routeA_rl_v1__support_expansion__rl_train__mixed_3d__000002",
                "split": "rl_train",
                "role_generated": "support_expansion",
                "role_validated": "support_expansion",
                "role_active": "support_expansion",
                "admission_status": "admitted_support_expansion",
                "bc_fm_sr": 0.40, "bc_fm_sr_ci_low": 0.25, "bc_fm_sr_ci_high": 0.55,
                "bc_fm_cr": 0.20, "ego_available": False, "beyond_boundary": False,
            },
            {
                "world_uid": "routeA_rl_v1__yaw_lateral_recovery__rl_train__disturb__000003",
                "split": "rl_train",
                "role_generated": "yaw_lateral_recovery",
                "role_validated": "yaw_lateral_recovery",
                "role_active": "yaw_lateral_recovery",
                "admission_status": "admitted_yaw_lateral_recovery",
                "bc_fm_sr": 0.45, "bc_fm_sr_ci_low": 0.30, "bc_fm_sr_ci_high": 0.60,
                "bc_fm_cr": 0.15, "ego_available": False, "beyond_boundary": False,
            },
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mock_manifest, f)
        tmp_path = f.name

    try:
        manifest = load_routea_rl_admission_manifest(tmp_path)
        assert manifest.schema_version == "routeA_rl_env_v1"
        assert len(manifest.worlds) == 4
        print("  ✓ Manifest loading with mock data")

        # Test get_worlds_by_role
        anti_worlds = manifest.get_worlds_by_role("anticipatory", "rl_train")
        assert len(anti_worlds) == 1
        print("  ✓ get_worlds_by_role filtering")

        # Test build_role_buckets
        buckets = build_role_buckets(manifest, "rl_train")
        assert len(buckets["anticipatory"]) == 1
        assert len(buckets["dense_recovery"]) == 1
        assert len(buckets["support_expansion"]) == 1
        assert len(buckets["yaw_lateral_recovery"]) == 1
        print("  ✓ build_role_buckets")

        # Test validate_role_buckets
        passed, reason = validate_role_buckets_for_training(buckets)
        assert passed, f"Buckets should pass: {reason}"
        print("  ✓ validate_role_buckets_for_training passes")

        # Test with empty bucket
        empty_buckets = {role: [] for role in TRAINABLE_ROLES}
        empty_buckets["anticipatory"] = []
        passed_empty, reason_empty = validate_role_buckets_for_training(empty_buckets)
        assert not passed_empty
        assert "has 0 admitted worlds" in reason_empty, \
            f"Expected empty bucket message, got: {reason_empty}"
        print("  ✓ Empty bucket detected")

        # Test sample_role_balanced_worlds
        sampled = sample_role_balanced_worlds(buckets, n_per_role=2, seed=42)
        assert len(sampled) > 0
        print(f"  ✓ Role-balanced sampling: {len(sampled)} worlds")

        # Test validate_rl_train_admission
        admission_check = validate_rl_train_admission(manifest)
        assert admission_check["pass_train_admission"], \
            f"Should pass: {admission_check['errors']}"
        print("  ✓ validate_rl_train_admission passes with mock data")

    finally:
        os.unlink(tmp_path)

    # Test dry-run with the actual smoke manifest path (if it exists)
    actual_manifest = "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest.json"
    if os.path.exists(actual_manifest):
        dry_run = dry_run_newenv_loader(actual_manifest)
        assert dry_run["schema_pass"], "Schema should pass for real manifest"
        assert dry_run["train_blocked"], "Train should be blocked for smoke manifest"
        print(f"  ✓ Dry-run on real manifest: schema=PASS, train=BLOCKED")
        for line in dry_run["summary_lines"]:
            print(f"    {line}")

    print(f"\n  ✅ All adapter self-tests passed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _self_test()
