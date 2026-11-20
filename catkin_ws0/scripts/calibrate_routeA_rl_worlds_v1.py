#!/usr/bin/env python3
"""
calibrate_routeA_rl_worlds_v1.py — Route A RL v1 Baseline Calibration CLI
==========================================================================
Two-pass adaptive calibration: Pass-1 20ep screening → Pass-2 boundary 50-100ep.
Writes admission manifest with CI-aware role_active decisions.

Baseline runners:
  frozen_bc_fm      — current BC/FM policy (required)
  ego_or_expert     — EGO planner upper bound (if available)
  random_or_noop    — sanity floor
  feasibility_checker — A*/BFS reachability from Level 2

Smoke mode uses synthetic episodes for pipeline testing without Gazebo/SITL.

Usage:
  # Smoke (synthetic data)
  python calibrate_routeA_rl_worlds_v1.py --smoke

  # Real calibration
  python calibrate_routeA_rl_worlds_v1.py \
    --validation_manifest /path/to/validation_manifest.json \
    --episodes_pass1 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add scripts directory for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routea_rl_calibration_stats import (
    PairedEpisodeResult,
    WorldCalibrationResult,
    wilson_ci,
    wilson_ci_halfwidth,
    compute_binary_metric_ci,
    ci_aware_gate,
    determine_role_active,
    generate_synthetic_episodes,
    bootstrap_mean_ci,
    ADMISSION_GATES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_VALIDATION_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/validation/"
    "validation_manifest.json"
)
DEFAULT_WORLD_DIR = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds"
DEFAULT_OUTPUT_DIR = "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"

PASS1_DEFAULT_EPISODES = 20
PASS2_DEFAULT_EPISODES = 50
PASS2_EXPANDED_EPISODES = 100

BASELINE_NAMES = ["frozen_bc_fm", "ego_or_expert", "random_or_noop", "feasibility_checker"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Offline density check
# ─────────────────────────────────────────────────────────────────────────────


def check_offline_density_available() -> Tuple[bool, str]:
    """Check if offline density tools/data exist."""
    density_paths = [
        "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/data/offline_density.json",
        "/home/cjj/AUV/data/offline_density.json",
    ]
    for p in density_paths:
        if os.path.exists(p):
            return True, p
    return False, "offline_density.json not found (P0-1 pending)"


# ─────────────────────────────────────────────────────────────────────────────
# Baseline runner abstraction
# ─────────────────────────────────────────────────────────────────────────────


class BaselineRunner:
    """Abstract baseline runner. Real implementation calls Gazebo/SITL."""

    def __init__(self, smoke: bool = False, seed: int = 20260703):
        self.smoke = smoke
        self.seed = seed

    def run_episodes(
        self,
        world_uid: str,
        role_validated: str,
        split: str,
        baseline_name: str,
        n_episodes: int,
        start_world: Optional[List[float]] = None,
        goal_world: Optional[List[float]] = None,
        start_yaw: float = 0.0,
        world_file: Optional[str] = None,
    ) -> List[PairedEpisodeResult]:
        """Run n_episodes of a baseline on one world.

        In smoke mode: generates synthetic data.
        In real mode: calls Gazebo/SITL rollout (not yet implemented).
        """
        if self.smoke:
            return generate_synthetic_episodes(
                world_uid=world_uid,
                role_validated=role_validated,
                split=split,
                baseline_name=baseline_name,
                n_episodes=n_episodes,
                seed=self.seed,
            )
        else:
            # Real rollout would go here — call GazeboUAVEnv / rollout_gazebo_eval
            raise NotImplementedError(
                "Real Gazebo rollout not yet implemented. "
                "Use --smoke for pipeline testing with synthetic data."
            )

    def is_ego_available(self) -> Tuple[bool, str]:
        """Check if EGO/expert planner is available."""
        if self.smoke:
            return True, "synthetic"
        # Check for EGO planner
        ego_worlds_dir = os.path.expanduser(
            "~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds"
        )
        ego_script = os.path.expanduser(
            "~/AUV/catkin_ws0/scripts/ego_gazebo.sh"
        )
        if os.path.exists(ego_script):
            return True, "EGO planner available"
        return False, (
            "EGO planner not available: ego_gazebo.sh not found or "
            "EGO not configured for Route A RL worlds"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pass-1 screening
# ─────────────────────────────────────────────────────────────────────────────


def run_pass1(
    runner: BaselineRunner,
    world_meta: Dict[str, Any],
    n_episodes: int,
) -> Tuple[List[PairedEpisodeResult], Dict[str, Any]]:
    """Run Pass-1 screening for one world.

    Returns:
        (all_episodes, pass1_summary)
    """
    world_uid = world_meta["world_uid"]
    role_validated = world_meta.get("role_validated") or world_meta.get("role_generated", "")
    split = world_meta.get("split", "rl_train")

    all_episodes: List[PairedEpisodeResult] = []

    # Run all baselines
    ego_avail, ego_reason = runner.is_ego_available()
    baselines_to_run = ["frozen_bc_fm", "random_or_noop"]
    if ego_avail:
        baselines_to_run.append("ego_or_expert")

    for baseline in baselines_to_run:
        eps = runner.run_episodes(
            world_uid=world_uid,
            role_validated=role_validated,
            split=split,
            baseline_name=baseline,
            n_episodes=n_episodes,
        )
        all_episodes.extend(eps)

    # Compute metrics
    bc_eps = [e for e in all_episodes if e.baseline_name == "frozen_bc_fm"]
    ego_eps = [e for e in all_episodes if e.baseline_name == "ego_or_expert"]
    random_eps = [e for e in all_episodes if e.baseline_name == "random_or_noop"]

    bc_sr_ci = compute_binary_metric_ci(bc_eps, lambda e: e.success)
    bc_cr_ci = compute_binary_metric_ci(bc_eps, lambda e: e.collision)

    path_lengths = [e.path_length for e in bc_eps]
    future_clrs = [e.future_clearance_softmin for e in bc_eps
                   if e.future_clearance_softmin is not None]

    summary = {
        "world_uid": world_uid,
        "role_validated": role_validated,
        "split": split,
        "bc_fm_sr": bc_sr_ci,
        "bc_fm_cr": bc_cr_ci,
        "bc_fm_path_length": bootstrap_mean_ci(path_lengths),
        "bc_fm_future_clr": bootstrap_mean_ci(future_clrs),
        "n_episodes_pass1": n_episodes,
    }

    if ego_eps:
        ego_sr_ci = compute_binary_metric_ci(ego_eps, lambda e: e.success)
        summary["ego_sr"] = ego_sr_ci
        summary["ego_available"] = True
    else:
        summary["ego_sr"] = None
        summary["ego_available"] = ego_avail
        summary["ego_unavailable_reason"] = ego_reason

    if random_eps:
        random_sr_ci = compute_binary_metric_ci(random_eps, lambda e: e.success)
        random_cr_ci = compute_binary_metric_ci(random_eps, lambda e: e.collision)
        summary["random_sr"] = random_sr_ci
        summary["random_cr"] = random_cr_ci

    # Feasibility from validation Level 2
    l2_label = world_meta.get("level2_reachability_label", "unknown")
    l2_reachable = world_meta.get("level2_reachable_3d", False)
    l2_stretch = world_meta.get("level2_path_stretch", 0.0)
    summary["feasibility_reachable"] = l2_reachable
    summary["feasibility_path_stretch"] = l2_stretch

    # Pass-1 screening decision
    needs_pass2, trigger_reason = _screen_pass1(summary)

    summary["needs_pass2"] = needs_pass2
    summary["pass2_trigger_reason"] = trigger_reason

    return all_episodes, summary


def _screen_pass1(summary: Dict[str, Any]) -> Tuple[bool, str]:
    """Determine if a world needs Pass-2 based on Pass-1 results."""
    bc_sr = summary["bc_fm_sr"]
    sr_rate = bc_sr.get("rate", float("nan"))
    ci_hw = bc_sr.get("ci_halfwidth", float("nan"))
    cr_rate = summary["bc_fm_cr"].get("rate", float("nan"))

    triggers = []

    # CI too wide
    if ci_hw > 0.15:
        triggers.append(f"CI half-width {ci_hw:.3f} > 0.15")

    # Near admission boundary
    role = summary.get("role_validated", "")
    gate = ADMISSION_GATES.get(role, {})
    min_sr = gate.get("min_sr", 0.0)
    max_sr = gate.get("max_sr", 1.0)
    if abs(sr_rate - min_sr) < 0.10:
        triggers.append(f"SR {sr_rate:.3f} near min boundary {min_sr}")
    if abs(sr_rate - max_sr) < 0.10:
        triggers.append(f"SR {sr_rate:.3f} near max boundary {max_sr}")

    # High CR
    if cr_rate > 0.6:
        triggers.append(f"CR {cr_rate:.3f} > 0.6")

    if triggers:
        return True, "; ".join(triggers)
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Pass-2 boundary expansion
# ─────────────────────────────────────────────────────────────────────────────


def run_pass2(
    runner: BaselineRunner,
    world_meta: Dict[str, Any],
    pass1_summary: Dict[str, Any],
    existing_episodes: List[PairedEpisodeResult],
    n_episodes_final: int,
) -> Tuple[List[PairedEpisodeResult], Dict[str, Any]]:
    """Run Pass-2 boundary expansion, appending episodes."""
    world_uid = world_meta["world_uid"]
    role_validated = pass1_summary.get("role_validated", "")
    split = pass1_summary.get("split", "rl_train")

    # Count existing BC episodes
    existing_bc = sum(1 for e in existing_episodes if e.baseline_name == "frozen_bc_fm")
    additional = max(0, n_episodes_final - existing_bc)

    all_episodes = list(existing_episodes)

    if additional > 0:
        # Run additional BC episodes
        extra_eps = runner.run_episodes(
            world_uid=world_uid,
            role_validated=role_validated,
            split=split,
            baseline_name="frozen_bc_fm",
            n_episodes=additional,
        )
        all_episodes.extend(extra_eps)

        # If EGO available, run matching additional
        if pass1_summary.get("ego_available"):
            extra_ego = runner.run_episodes(
                world_uid=world_uid,
                role_validated=role_validated,
                split=split,
                baseline_name="ego_or_expert",
                n_episodes=additional,
            )
            all_episodes.extend(extra_ego)

    # Recompute metrics
    bc_eps = [e for e in all_episodes if e.baseline_name == "frozen_bc_fm"]
    ego_eps = [e for e in all_episodes if e.baseline_name == "ego_or_expert"]
    random_eps = [e for e in all_episodes if e.baseline_name == "random_or_noop"]

    bc_sr_ci = compute_binary_metric_ci(bc_eps, lambda e: e.success)
    bc_cr_ci = compute_binary_metric_ci(bc_eps, lambda e: e.collision)

    path_lengths = [e.path_length for e in bc_eps]
    future_clrs = [e.future_clearance_softmin for e in bc_eps
                   if e.future_clearance_softmin is not None]

    # Trap/dense semantic metrics
    n_trap_active = sum(1 for e in bc_eps if e.trap_active)
    n_trap_recovered = sum(1 for e in bc_eps if e.trap_recovered)
    n_deadend = sum(1 for e in bc_eps if e.deadend_dwell_steps > 2)

    low_density_coverages = [
        e.successful_low_density_coverage for e in bc_eps
        if e.successful_low_density_coverage is not None
    ]

    yaw_errors = [
        e.yaw_error_integral for e in bc_eps
        if e.yaw_error_integral is not None
    ]

    semantic_metrics = {
        "trap_active_rate": round(n_trap_active / max(len(bc_eps), 1), 4),
        "trap_recovered_rate": round(n_trap_recovered / max(len(bc_eps), 1), 4),
        "deadend_dwell_rate": round(n_deadend / max(len(bc_eps), 1), 4),
        "low_density_coverage": bootstrap_mean_ci(low_density_coverages),
        "yaw_error_integral": bootstrap_mean_ci(yaw_errors),
    }

    summary = {
        **pass1_summary,
        "bc_fm_sr": bc_sr_ci,
        "bc_fm_cr": bc_cr_ci,
        "bc_fm_path_length": bootstrap_mean_ci(path_lengths),
        "bc_fm_future_clr": bootstrap_mean_ci(future_clrs),
        "semantic_metrics": semantic_metrics,
        "n_episodes_final": len(bc_eps),
        "n_episodes_total": len(bc_eps),
    }

    if ego_eps:
        ego_sr_ci = compute_binary_metric_ci(ego_eps, lambda e: e.success)
        summary["ego_sr"] = ego_sr_ci

    if random_eps:
        summary["random_sr"] = compute_binary_metric_ci(random_eps, lambda e: e.success)
        summary["random_cr"] = compute_binary_metric_ci(random_eps, lambda e: e.collision)

    return all_episodes, summary


# ─────────────────────────────────────────────────────────────────────────────
# Admission decision
# ─────────────────────────────────────────────────────────────────────────────


def make_admission_decision(
    world_meta: Dict[str, Any],
    final_summary: Dict[str, Any],
) -> WorldCalibrationResult:
    """Apply admission rules to produce final WorldCalibrationResult."""
    world_uid = final_summary["world_uid"]
    role_validated = final_summary.get("role_validated", "")
    role_generated = world_meta.get("role_generated", "")
    split = final_summary.get("split", "rl_train")

    bc_sr_ci = final_summary["bc_fm_sr"]
    bc_cr_ci = final_summary["bc_fm_cr"]
    ego_sr_ci = final_summary.get("ego_sr")
    eg_avail = final_summary.get("ego_available", False)
    eg_reason = final_summary.get("ego_unavailable_reason", "")
    rand_sr = final_summary.get("random_sr", {})
    rand_cr = final_summary.get("random_cr", {})
    sem_metrics = final_summary.get("semantic_metrics", {})

    # Determine role_active
    bc_metrics = {
        "sr_rate": bc_sr_ci.get("rate", 0.0),
        "sr_ci_low": bc_sr_ci.get("ci_low", 0.0),
        "sr_ci_high": bc_sr_ci.get("ci_high", 0.0),
        "cr_rate": bc_cr_ci.get("rate", 0.0),
        "sr_ci_halfwidth": bc_sr_ci.get("ci_halfwidth", 0.0),
    }
    ego_metrics = None
    if ego_sr_ci:
        ego_metrics = {
            "sr_rate": ego_sr_ci.get("rate", 0.0),
            "sr_ci_low": ego_sr_ci.get("ci_low", 0.0),
            "sr_ci_high": ego_sr_ci.get("ci_high", 0.0),
        }

    role_active, admission_status, beyond_boundary = determine_role_active(
        role_validated=role_validated or role_generated,
        bc_fm_metrics=bc_metrics,
        ego_metrics=ego_metrics,
        semantic_metrics=sem_metrics,
    )

    # Override for support_expansion without density
    if role_validated == "support_expansion":
        density_avail, _ = check_offline_density_available()
        if not density_avail:
            role_active = None
            admission_status = "blocked_support_offline_density"
            beyond_boundary = False

    # CI-aware downgrade
    ci_hw = bc_sr_ci.get("ci_halfwidth", 0.0)
    gate_status, gate_reason = ci_aware_gate(bc_sr_ci)
    if gate_status == "needs_more_eval":
        admission_status = "needs_more_eval"

    stop_reason = ""
    if admission_status == "needs_more_eval":
        stop_reason = f"CI half-width {ci_hw:.4f} — more episodes needed"
    elif admission_status == "beyond_BC_EGO_boundary":
        stop_reason = "BC/FM and EGO both below threshold"
    elif admission_status.startswith("admitted"):
        stop_reason = f"Admitted as {role_active}"

    return WorldCalibrationResult(
        world_uid=world_uid,
        split=split,
        role_generated=role_generated,
        role_validated=role_validated,
        role_active=role_active,
        admission_status=admission_status,
        n_episodes_pass1=final_summary.get("n_episodes_pass1", 0),
        n_episodes_final=final_summary.get("n_episodes_final", 0),
        bc_fm_sr=bc_sr_ci.get("rate", float("nan")),
        bc_fm_sr_ci_low=bc_sr_ci.get("ci_low", float("nan")),
        bc_fm_sr_ci_high=bc_sr_ci.get("ci_high", float("nan")),
        bc_fm_cr=bc_cr_ci.get("rate", float("nan")),
        bc_fm_cr_ci_low=bc_cr_ci.get("ci_low", float("nan")),
        bc_fm_cr_ci_high=bc_cr_ci.get("ci_high", float("nan")),
        bc_fm_path_length_mean=final_summary.get("bc_fm_path_length", {}).get("mean", float("nan")),
        bc_fm_future_clr_softmin_mean=final_summary.get("bc_fm_future_clr", {}).get("mean", float("nan")),
        ego_sr=ego_sr_ci.get("rate") if ego_sr_ci else None,
        ego_sr_ci_low=ego_sr_ci.get("ci_low") if ego_sr_ci else None,
        ego_sr_ci_high=ego_sr_ci.get("ci_high") if ego_sr_ci else None,
        ego_available=eg_avail,
        ego_unavailable_reason=eg_reason,
        random_sr=rand_sr.get("rate") if rand_sr else None,
        random_cr=rand_cr.get("rate") if rand_cr else None,
        feasibility_reachable=final_summary.get("feasibility_reachable", False),
        feasibility_path_stretch=final_summary.get("feasibility_path_stretch"),
        needs_pass2=final_summary.get("needs_pass2", False),
        pass2_trigger_reason=final_summary.get("pass2_trigger_reason", ""),
        beyond_boundary=beyond_boundary,
        stop_reason=stop_reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────


def write_admission_manifest(
    results: List[WorldCalibrationResult],
    output_dir: Path,
) -> Path:
    """Write admission manifest JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "admission_manifest.json"

    stats = {
        "total": len(results),
        "admitted": sum(1 for r in results if r.admission_status.startswith("admitted")),
        "rejected": sum(1 for r in results if r.admission_status.startswith("reject")),
        "needs_more_eval": sum(1 for r in results if r.admission_status == "needs_more_eval"),
        "blocked_density": sum(1 for r in results if "density" in r.admission_status),
        "beyond_boundary": sum(1 for r in results if r.beyond_boundary),
        "stress_only": sum(1 for r in results if r.role_active == "stress_only"),
        "ego_available": all(r.ego_available for r in results),
    }
    # Per-role breakdown
    for role in ["anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery"]:
        role_results = [r for r in results if r.role_validated == role
                        or r.role_generated == role]
        if role_results:
            stats[f"{role}_admitted"] = sum(
                1 for r in role_results if r.admission_status.startswith("admitted")
            )
            stats[f"{role}_total"] = len(role_results)

    manifest = {
        "schema_version": "routeA_rl_env_v1",
        "calibration_date": "2026-07-03",
        "mode": "smoke_synthetic" if any(True for _ in []) else "real",
        "stats": stats,
        "results": [asdict(r) for r in results],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def write_admission_csv(
    results: List[WorldCalibrationResult],
    output_dir: Path,
) -> Path:
    """Write admission summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "admission_summary.csv"

    fields = [
        "world_uid", "split", "role_generated", "role_validated", "role_active",
        "admission_status", "n_episodes_pass1", "n_episodes_final",
        "bc_fm_sr", "bc_fm_sr_ci_low", "bc_fm_sr_ci_high",
        "bc_fm_cr", "bc_fm_cr_ci_low", "bc_fm_cr_ci_high",
        "ego_sr", "ego_available", "needs_pass2", "pass2_trigger_reason",
        "beyond_boundary", "stop_reason",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            d = asdict(r)
            # Flatten ego_available
            writer.writerow(d)

    return path


def write_calibration_report_md(
    results: List[WorldCalibrationResult],
    output_dir: Path,
    ego_available: bool,
    ego_reason: str,
) -> Path:
    """Write human-readable calibration report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "baseline_calibration_report.md"

    admitted = [r for r in results if r.admission_status.startswith("admitted")]
    rejected = [r for r in results if r.admission_status.startswith("reject")]
    needs_more = [r for r in results if r.admission_status == "needs_more_eval"]
    blocked = [r for r in results if "density" in r.admission_status]

    lines = [
        "# Route A RL v1 Baseline Calibration Report",
        "",
        f"- **Date**: 2026-07-03",
        f"- **Total worlds calibrated**: {len(results)}",
        f"- **Admitted**: {len(admitted)}",
        f"- **Rejected**: {len(rejected)}",
        f"- **Needs more eval**: {len(needs_more)}",
        f"- **Blocked (offline density)**: {len(blocked)}",
        "",
        "## Baseline Availability",
        "",
        f"- **frozen_bc_fm**: ✅ available",
        f"- **ego_or_expert**: {'✅' if ego_available else '❌'} {ego_reason}",
        f"- **random_or_noop**: ✅ synthetic",
        f"- **feasibility_checker**: ✅ from Level 2 validation",
        "",
        "## Admission Summary by Role",
        "",
        "| Role | Total | Admitted | Rejected | Needs More | Blocked |",
        "|------|-------|----------|----------|------------|---------|",
    ]

    for role in ["anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery"]:
        role_res = [r for r in results
                    if r.role_validated == role or r.role_generated == role]
        if not role_res:
            continue
        n_adm = sum(1 for r in role_res if r.admission_status.startswith("admitted"))
        n_rej = sum(1 for r in role_res if r.admission_status.startswith("reject"))
        n_nme = sum(1 for r in role_res if r.admission_status == "needs_more_eval")
        n_blk = sum(1 for r in role_res if "density" in r.admission_status)
        lines.append(
            f"| {role} | {len(role_res)} | {n_adm} | {n_rej} | {n_nme} | {n_blk} |"
        )

    lines += [
        "",
        "## Admitted Worlds",
        "",
        "| # | world_uid | role_active | split | BC/FM SR [CI] | EGO SR | Status |",
        "|---|-----------|-------------|-------|---------------|--------|--------|",
    ]
    for i, r in enumerate(admitted):
        uid_short = r.world_uid[:55]
        sr_str = f"{r.bc_fm_sr:.3f} [{r.bc_fm_sr_ci_low:.3f}, {r.bc_fm_sr_ci_high:.3f}]"
        ego_str = f"{r.ego_sr:.3f}" if r.ego_sr is not None else "N/A"
        lines.append(
            f"| {i+1} | {uid_short} | {r.role_active} | {r.split} | "
            f"{sr_str} | {ego_str} | {r.admission_status} |"
        )

    if rejected:
        lines += [
            "",
            "## Rejected Worlds",
            "",
            "| # | world_uid | Reason |",
            "|---|-----------|--------|",
        ]
        for i, r in enumerate(rejected):
            lines.append(f"| {i+1} | {r.world_uid[:55]} | {r.stop_reason} |")

    if needs_more:
        lines += [
            "",
            "## Needs More Evaluation",
            "",
            "| # | world_uid | CI half-width | Reason |",
            "|---|-----------|---------------|--------|",
        ]
        for i, r in enumerate(needs_more):
            ci_hw = (r.bc_fm_sr_ci_high - r.bc_fm_sr_ci_low) / 2.0
            lines.append(f"| {i+1} | {r.world_uid[:55]} | {ci_hw:.4f} | {r.stop_reason} |")

    lines += [
        "",
        "## CI-Aware Gates",
        "",
        "| Role | SR Gate | Status |",
        "|------|---------|--------|",
    ]
    for role, gate in ADMISSION_GATES.items():
        role_res = [r for r in results
                    if r.role_validated == role or r.role_generated == role]
        if not role_res:
            continue
        n_pass = sum(1 for r in role_res
                     if gate["min_sr"] <= r.bc_fm_sr_ci_low
                     and r.bc_fm_sr_ci_high <= gate["max_sr"])
        lines.append(
            f"| {role} | [{gate['min_sr']:.2f}, {gate['max_sr']:.2f}] | "
            f"{n_pass}/{len(role_res)} pass |"
        )

    lines += [
        "",
        "## Constraints",
        "",
        "- Actor/critic NOT updated during calibration",
        "- Replay NOT written",
        "- Reward math NOT changed",
        "- EGO/A*/BFS NOT used as inference-time planner",
        "- `beyond_BC_EGO_boundary` worlds NOT in `rl_train`",
        "- CI-wide worlds NOT hard-passed",
    ]

    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_calibration_pipeline(
    validation_manifest_path: Path,
    output_dir: Path,
    world_dir: Path,
    episodes_pass1: int,
    episodes_pass2: int,
    smoke: bool = False,
    seed: int = 20260703,
) -> int:
    """Run the full two-pass calibration pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load validation manifest
    with open(validation_manifest_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    val_results = val_data.get("results", [])
    if not val_results:
        print("ERROR: validation manifest contains no results")
        return 1

    # Filter to calibratable worlds
    calibratable = [
        r for r in val_results
        if r.get("admission_status", "").startswith("pending_calibration")
        or r.get("admission_status", "") == "pending_offline_density"
    ]

    if not calibratable:
        # If no pending_calibration, use all validated worlds
        calibratable = [
            r for r in val_results
            if r.get("level1_passed") and r.get("level2_reachable_3d")
        ]

    print(f"Calibration pipeline")
    print(f"{'=' * 60}")
    print(f"Validation manifest: {validation_manifest_path}")
    print(f"Output dir:          {output_dir}")
    print(f"World dir:           {world_dir}")
    print(f"Episodes pass-1:     {episodes_pass1}")
    print(f"Episodes pass-2:     {episodes_pass2}")
    print(f"Smoke mode:          {smoke}")
    print(f"Worlds to calibrate: {len(calibratable)}")
    print()

    # Step 0: Offline density check
    density_avail, density_path = check_offline_density_available()
    if density_avail:
        print(f"Offline density: ✅ available at {density_path}")
    else:
        print(f"Offline density: ❌ {density_path}")
        print("  → support_expansion worlds will be blocked")
    print()

    # Baseline runner
    runner = BaselineRunner(smoke=smoke, seed=seed)
    ego_avail, ego_reason = runner.is_ego_available()
    print(f"EGO/expert baseline: {'✅' if ego_avail else '❌'} {ego_reason}")
    print()

    # Process each world
    results: List[WorldCalibrationResult] = []

    for i, world_meta in enumerate(calibratable):
        world_uid = world_meta.get("world_uid", f"unknown_{i}")
        adm_status = world_meta.get("admission_status", "")
        role_val = world_meta.get("role_validated") or world_meta.get("role_generated", "")

        print(f"[{i+1:04d}/{len(calibratable)}] {world_uid[:65]}")
        print(f"  role_validated={role_val}  admission={adm_status}")

        # Handle blocked support_expansion — only block if density truly unavailable
        if role_val == "support_expansion" and not density_avail:
            result = WorldCalibrationResult(
                world_uid=world_uid,
                split=world_meta.get("split", ""),
                role_generated=world_meta.get("role_generated", ""),
                role_validated=role_val,
                role_active=None,
                admission_status="blocked_support_offline_density",
                n_episodes_pass1=0,
                n_episodes_final=0,
                bc_fm_sr=float("nan"), bc_fm_sr_ci_low=float("nan"), bc_fm_sr_ci_high=float("nan"),
                bc_fm_cr=float("nan"), bc_fm_cr_ci_low=float("nan"), bc_fm_cr_ci_high=float("nan"),
                bc_fm_path_length_mean=float("nan"),
                bc_fm_future_clr_softmin_mean=float("nan"),
                ego_sr=None, ego_sr_ci_low=None, ego_sr_ci_high=None,
                ego_available=ego_avail, ego_unavailable_reason=ego_reason,
                random_sr=None, random_cr=None,
                feasibility_reachable=world_meta.get("level2_reachable_3d", False),
                feasibility_path_stretch=world_meta.get("level2_path_stretch"),
                needs_pass2=False, pass2_trigger_reason="",
                beyond_boundary=False,
                stop_reason="Offline density data not available (P0-1 pending)",
            )
            results.append(result)
            print(f"  → blocked_support_offline_density")
            continue

        # If support_expansion with density available: proceed with calibration
        if role_val == "support_expansion" and density_avail:
            print(f"  density data available → proceeding with calibration")

        # Pass-1
        all_eps, pass1_summary = run_pass1(runner, world_meta, episodes_pass1)

        sr_rate = pass1_summary["bc_fm_sr"].get("rate", float("nan"))
        cr_rate = pass1_summary["bc_fm_cr"].get("rate", float("nan"))
        needs_p2 = pass1_summary["needs_pass2"]
        p2_reason = pass1_summary["pass2_trigger_reason"]

        print(f"  Pass-1: SR={sr_rate:.3f} CR={cr_rate:.3f}  "
              f"needs_pass2={needs_p2}  {p2_reason}")

        # Pass-2 if needed
        if needs_p2:
            n_pass2 = episodes_pass2
            # Expand for val/test worlds
            split = world_meta.get("split", "")
            if split in ("rl_val", "rl_test"):
                n_pass2 = max(n_pass2, 100)

            all_eps, final_summary = run_pass2(
                runner, world_meta, pass1_summary, all_eps, n_pass2,
            )
            sr_rate = final_summary["bc_fm_sr"].get("rate", float("nan"))
            cr_rate = final_summary["bc_fm_cr"].get("rate", float("nan"))
            print(f"  Pass-2: SR={sr_rate:.3f} CR={cr_rate:.3f}  "
                  f"n_eps={final_summary.get('n_episodes_final', 0)}")
        else:
            pass1_summary["n_episodes_final"] = episodes_pass1
            final_summary = pass1_summary

        # Admission decision
        result = make_admission_decision(world_meta, final_summary)
        result.episodes = [asdict(e) for e in all_eps]

        results.append(result)

        print(f"  → {result.admission_status}  "
              f"role_active={result.role_active}  "
              f"beyond_boundary={result.beyond_boundary}")

    # ── Write outputs ──
    manifest_path = write_admission_manifest(results, output_dir)
    csv_path = write_admission_csv(results, output_dir)
    md_path = write_calibration_report_md(results, output_dir, ego_avail, ego_reason)

    # ── Summary ──
    admitted = [r for r in results if r.admission_status.startswith("admitted")]
    rejected = [r for r in results if r.admission_status.startswith("reject")]
    blocked = [r for r in results if "density" in r.admission_status]
    needs_more = [r for r in results if r.admission_status == "needs_more_eval"]

    print()
    print(f"{'=' * 60}")
    print(f"Calibration complete: {len(results)} worlds")
    print(f"  Admitted:         {len(admitted)}")
    print(f"  Rejected:         {len(rejected)}")
    print(f"  Needs more eval:  {len(needs_more)}")
    print(f"  Blocked (density): {len(blocked)}")
    print()
    print(f"Outputs:")
    print(f"  {manifest_path}")
    print(f"  {csv_path}")
    print(f"  {md_path}")

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Route A RL v1 Baseline Calibration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--validation_manifest", type=str,
        default=DEFAULT_VALIDATION_MANIFEST,
        help="Path to validation manifest JSON",
    )
    parser.add_argument(
        "--world_dir", type=str,
        default=DEFAULT_WORLD_DIR,
        help="Directory containing .world files",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for calibration reports",
    )
    parser.add_argument(
        "--episodes_pass1", type=int,
        default=PASS1_DEFAULT_EPISODES,
        help="Episodes per world for Pass-1 screening",
    )
    parser.add_argument(
        "--episodes_pass2", type=int,
        default=PASS2_DEFAULT_EPISODES,
        help="Episodes per world for Pass-2 boundary expansion",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use synthetic data for pipeline testing (no Gazebo needed)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260703,
        help="Random seed",
    )
    args = parser.parse_args()

    return run_calibration_pipeline(
        validation_manifest_path=Path(args.validation_manifest),
        output_dir=Path(args.output_dir),
        world_dir=Path(args.world_dir),
        episodes_pass1=args.episodes_pass1,
        episodes_pass2=args.episodes_pass2,
        smoke=args.smoke,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
