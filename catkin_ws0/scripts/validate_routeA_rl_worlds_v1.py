#!/usr/bin/env python3
"""
validate_routeA_rl_worlds_v1.py — Route A RL v1 Three-Level World Validator CLI
================================================================================
Runs Level 1/2/3 validation on all worlds in a manifest, writes admission decisions.

Usage:
  # Smoke mode
  python validate_routeA_rl_worlds_v1.py --smoke

  # Full validation
  python validate_routeA_rl_worlds_v1.py \
    --manifest /path/to/routeA_rl_v1_world_manifest.json \
    --world_dir /path/to/worlds \
    --output_dir /path/to/validation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from routea_rl_env_validation import (
    parse_world_obstacles,
    validate_level1_geometry,
    validate_level2_reachability,
    validate_level3_semantic,
    compute_admission,
    ALLOWED_ROLES,
    ALLOWED_SPLITS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/worldgen/"
    "routeA_rl_v1_world_manifest.json"
)
DEFAULT_WORLD_DIR = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds"
DEFAULT_OUTPUT_DIR = "/home/cjj/AUV/program/reports/routeA_rl_env_v1/validation"

FORBIDDEN_WORLD_IDS = {4, 8, 11, 12, 18}
FORBIDDEN_LEGACY_SCENARIOS = {
    "canonical", "support", "default", "dense", "random_stress",
    "empty", "sparse", "medium_A", "medium_B", "super_dense", "special",
}


# ─────────────────────────────────────────────────────────────────────────────


def validate_all_worlds(
    manifest_path: Path,
    world_dir: Path,
    output_dir: Path,
    smoke: bool = False,
) -> int:
    """Run full validation pipeline on all worlds in manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    worlds = manifest_data.get("worlds", [])
    if not worlds:
        print("ERROR: manifest contains no worlds")
        return 1

    # Smoke: only first 4 worlds
    if smoke:
        worlds = worlds[:4]
        print(f"=== SMOKE MODE: {len(worlds)} worlds ===")

    print(f"Validating {len(worlds)} worlds...")
    print(f"World dir:  {world_dir}")
    print(f"Output dir: {output_dir}")
    print()

    results: List[Dict[str, Any]] = []
    stats = {
        "total": len(worlds),
        "l1_pass": 0,
        "l1_fail": 0,
        "l2_reachable": 0,
        "l2_unreachable": 0,
        "l3_pass": 0,
        "l3_fail": 0,
        "l3_relabel": 0,
        "accepted": 0,
        "reject_geometry": 0,
        "reject_unreachable": 0,
        "reject_semantic": 0,
        "pending_calibration": 0,
        "pending_calibration_relabelled": 0,
        "pending_density": 0,
    }

    for i, world_meta in enumerate(worlds):
        world_uid = world_meta.get("world_uid", f"unknown_{i}")
        world_filename = f"{world_uid}.world"
        world_path = world_dir / world_filename

        print(f"[{i+1:04d}/{len(worlds)}] {world_uid[:70]}")

        # Check forbidden legacy references
        uid_lower = world_uid.lower()
        has_forbidden = False
        for fwid in FORBIDDEN_WORLD_IDS:
            if f"w{fwid}" in uid_lower:
                print(f"  ✗ Forbidden legacy reference: w{fwid}")
                has_forbidden = True
        if has_forbidden:
            results.append({
                "world_uid": world_uid,
                "split": world_meta.get("split", ""),
                "role_generated": world_meta.get("role_generated", ""),
                "role_validated": None,
                "role_active": None,
                "level1_passed": False,
                "level2_reachable_3d": False,
                "level3_passed": False,
                "admission_status": "reject_forbidden_legacy",
                "fail_reasons": ["Contains forbidden legacy world reference"],
                "path_stretch": 0.0,
                "bottleneck_clearance_surface": 0.0,
            })
            continue

        # Parse obstacles from .world file
        if not world_path.exists():
            print(f"  ✗ World file not found: {world_path}")
            results.append({
                "world_uid": world_uid,
                "split": world_meta.get("split", ""),
                "role_generated": world_meta.get("role_generated", ""),
                "role_validated": None,
                "role_active": None,
                "level1_passed": False,
                "level2_reachable_3d": False,
                "level3_passed": False,
                "admission_status": "reject_missing_file",
                "fail_reasons": [f"World file not found: {world_path}"],
                "path_stretch": 0.0,
                "bottleneck_clearance_surface": 0.0,
            })
            continue

        obstacles = parse_world_obstacles(world_path)

        # Level 1: Geometry
        l1_report = validate_level1_geometry(world_meta, obstacles)
        l1_pass = l1_report["passed"]
        if l1_pass:
            stats["l1_pass"] += 1
        else:
            stats["l1_fail"] += 1

        status_l1 = "✓" if l1_pass else f"✗ ({len(l1_report.get('fail_reasons', []))})"
        print(f"  L1 geometry:     {status_l1}  "
              f"spawn_clr={l1_report.get('spawn_clearance_surface', '?'):.3f}  "
              f"goal_clr={l1_report.get('goal_clearance_surface', '?'):.3f}")

        # Level 2: Reachability
        l2_report = validate_level2_reachability(world_meta, obstacles)
        l2_reachable = l2_report["reachable_3d"]
        if l2_reachable:
            stats["l2_reachable"] += 1
        else:
            stats["l2_unreachable"] += 1

        status_l2 = "✓" if l2_reachable else f"✗ ({l2_report.get('reachability_label', '?')})"
        print(f"  L2 reachability: {status_l2}  "
              f"stretch={l2_report.get('path_stretch', '?'):.2f}  "
              f"bn_clr={l2_report.get('bottleneck_clearance_surface', '?'):.2f}")

        # Level 3: Semantic
        l3_report = validate_level3_semantic(world_meta, obstacles, l2_report)
        l3_matches = l3_report.get("matches_generated_role", False)
        l3_relabel = l3_report.get("needs_relabel", False)

        if l3_matches:
            stats["l3_pass"] += 1
        elif l3_relabel:
            stats["l3_relabel"] += 1
        else:
            stats["l3_fail"] += 1

        status_l3 = (
            "✓" if l3_matches
            else f"↻→{l3_report.get('best_matching_role', '?')}" if l3_relabel
            else "✗"
        )
        print(f"  L3 semantic:     {status_l3}  "
              f"semantic_status={l3_report.get('semantic_status', 'ok')}")

        # Admission decision
        admission = compute_admission(l1_report, l2_report, l3_report)
        adm_status = admission["admission_status"]

        # Update stats
        stat_map = {
            "reject_geometry": "reject_geometry",
            "reject_unreachable_or_stress": "reject_unreachable",
            "reject_semantic": "reject_semantic",
            "pending_calibration": "pending_calibration",
            "pending_calibration_relabelled": "pending_calibration_relabelled",
            "pending_offline_density": "pending_density",
        }
        for key, stat_key in stat_map.items():
            if adm_status == key:
                stats[stat_key] += 1
                break

        if adm_status.startswith("pending_calibration"):
            stats["accepted"] += 1

        print(f"  → {adm_status}  "
              f"role_validated={admission.get('role_validated')}  "
              f"role_active={admission.get('role_active')}")

        # Collect all fail reasons
        all_fail_reasons = []
        if l1_report.get("fail_reasons"):
            all_fail_reasons.extend(l1_report["fail_reasons"])
        if admission.get("fail_reasons"):
            all_fail_reasons.extend(admission["fail_reasons"])

        # Build result row
        result = {
            "world_uid": world_uid,
            "split": world_meta.get("split", ""),
            "role_generated": world_meta.get("role_generated", ""),
            "role_validated": admission.get("role_validated"),
            "role_active": admission.get("role_active"),
            "level1_passed": l1_pass,
            "level1_spawn_clearance": l1_report.get("spawn_clearance_surface"),
            "level1_goal_clearance": l1_report.get("goal_clearance_surface"),
            "level2_reachable_2d": l2_report.get("reachable_2d"),
            "level2_reachable_3d": l2_reachable,
            "level2_path_stretch": l2_report.get("path_stretch"),
            "level2_bottleneck_clearance": l2_report.get("bottleneck_clearance_surface"),
            "level2_z_span_oracle": str(l2_report.get("z_span_oracle", [])),
            "level2_reachability_label": l2_report.get("reachability_label"),
            "level3_passed": l3_matches or l3_relabel,
            "level3_best_role": l3_report.get("best_matching_role"),
            "level3_semantic_status": l3_report.get("semantic_status", "ok"),
            "admission_status": adm_status,
            "fail_reasons": " | ".join(all_fail_reasons) if all_fail_reasons else "",
            "path_stretch": l2_report.get("path_stretch", 0.0),
            "bottleneck_clearance_surface": l2_report.get("bottleneck_clearance_surface", 0.0),
        }
        results.append(result)

        # Write per-world validation JSON
        val_path = output_dir / f"{world_uid}.validation.json"
        per_world = {
            "world_uid": world_uid,
            "level1": l1_report,
            "level2": l2_report,
            "level3": l3_report,
            "admission": admission,
        }
        val_path.parent.mkdir(parents=True, exist_ok=True)
        val_path.write_text(
            json.dumps(per_world, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Write outputs ──

    # Validation manifest
    val_manifest_path = output_dir / "validation_manifest.json"
    val_manifest = {
        "schema_version": "routeA_rl_env_v1",
        "validation_date": "2026-07-03",
        "stats": stats,
        "results": results,
    }
    val_manifest_path.write_text(
        json.dumps(val_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # CSV summary
    csv_path = output_dir / "validation_summary.csv"
    csv_fields = [
        "world_uid", "split", "role_generated", "role_validated", "role_active",
        "level1_passed", "level2_reachable_3d", "level3_passed",
        "admission_status", "fail_reasons", "path_stretch",
        "bottleneck_clearance_surface",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Markdown summary
    md_path = output_dir / "validation_summary.md"
    md_lines = [
        "# Route A RL v1 Validation Summary",
        "",
        f"- **Date**: 2026-07-03",
        f"- **Total worlds validated**: {stats['total']}",
        "",
        "## Level 1: Geometry",
        f"- Pass: {stats['l1_pass']}",
        f"- Fail: {stats['l1_fail']}",
        "",
        "## Level 2: Reachability",
        f"- Reachable (3D): {stats['l2_reachable']}",
        f"- Unreachable: {stats['l2_unreachable']}",
        "",
        "## Level 3: Semantic",
        f"- Pass (matches generated role): {stats['l3_pass']}",
        f"- Relabelled to different role: {stats['l3_relabel']}",
        f"- Fail: {stats['l3_fail']}",
        "",
        "## Admission Summary",
        f"- Accepted (pending_calibration): {stats['accepted']}",
        f"- Reject geometry: {stats['reject_geometry']}",
        f"- Reject unreachable: {stats['reject_unreachable']}",
        f"- Reject semantic: {stats['reject_semantic']}",
        f"- Pending calibration (relabelled): {stats['pending_calibration_relabelled']}",
        f"- Pending offline density: {stats['pending_density']}",
        "",
        "## Constraints",
        "- A*/BFS path NOT written to actor input or training replay",
        "- role_active set only for stress_only downgrades",
        "- role_validated filled by validation, not generator",
        "",
        "## Per-World Details",
        "",
        "| # | world_uid | L1 | L2 | L3 | admission | role_validated |",
        "|---|-----------|----|----|----|-----------|----------------|",
    ]

    for i, r in enumerate(results):
        l1 = "✓" if r["level1_passed"] else "✗"
        l2 = "✓" if r["level2_reachable_3d"] else "✗"
        l3 = "✓" if r["level3_passed"] else "✗"
        uid_short = r["world_uid"][:50]
        md_lines.append(
            f"| {i+1} | {uid_short} | {l1} | {l2} | {l3} | "
            f"{r['admission_status']} | {r.get('role_validated') or '-'} |"
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # ── Final summary ──
    print()
    print(f"{'=' * 60}")
    print(f"Validation complete: {stats['total']} worlds")
    print(f"  L1 pass: {stats['l1_pass']}/{stats['total']}")
    print(f"  L2 reachable: {stats['l2_reachable']}/{stats['total']}")
    print(f"  L3 semantic pass: {stats['l3_pass']} + relabel: {stats['l3_relabel']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Rejected: {stats['reject_geometry'] + stats['reject_unreachable'] + stats['reject_semantic']}")
    print(f"  Pending density: {stats['pending_density']}")
    print()
    print(f"Outputs:")
    print(f"  {val_manifest_path}")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print()

    if stats["accepted"] > 0:
        print(f"✓ {stats['accepted']} worlds ready for calibration")
    else:
        print("⚠ No worlds accepted for calibration")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Route A RL v1 Three-Level World Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=str, default=DEFAULT_MANIFEST,
        help="Path to world manifest JSON",
    )
    parser.add_argument(
        "--world_dir", type=str, default=DEFAULT_WORLD_DIR,
        help="Directory containing .world files",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for validation reports",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Validate only first 4 worlds for quick check",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    world_dir = Path(args.world_dir)
    output_dir = Path(args.output_dir)

    return validate_all_worlds(
        manifest_path=manifest_path,
        world_dir=world_dir,
        output_dir=output_dir,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    sys.exit(main())
