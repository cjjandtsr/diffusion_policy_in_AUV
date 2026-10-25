#!/usr/bin/env python3
"""
p0_7_preflight_summary.py — P0-7: Preflight summary (§7)

聚合所有 P0 前置任务结果，生成 preflight_summary.json + 可读报告。

检查:
  P0-1: offline density
  P0-2: episode-local RNG
  P0-3: guard/loop audit
  P0-4: instrumentation interface
  P0-5: stochastic path toy test
  P0-6: BC variance calibration

用法:
  python tools/p0_7_preflight_summary.py
"""

import json, os, sys, time
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# ═══════════════════════════════════════════════════════════════════
# Path resolution helpers
# ═══════════════════════════════════════════════════════════════════

SHIYAN_DIR = PROJECT_DIR / "program" / "shiyan"
DATA_DIR = PROJECT_DIR / "data"

def find_latest(pattern):
    """Find latest file matching glob pattern."""
    matches = sorted(SHIYAN_DIR.glob(pattern))
    return str(matches[-1]) if matches else None


def find_latest_dir(prefix):
    """Find latest directory with given prefix."""
    dirs = sorted([d for d in SHIYAN_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)])
    return str(dirs[-1]) if dirs else None


# ═══════════════════════════════════════════════════════════════════
# Check functions
# ═══════════════════════════════════════════════════════════════════

def check_p0_1_offline_density():
    """Check offline density exists and has reasonable structure."""
    density_json = DATA_DIR / "offline_density.json"
    report_json = DATA_DIR / "offline_density_report.json"
    if not density_json.exists():
        return {"status": "NOT_FOUND", "path": str(density_json)}

    d = json.load(open(density_json))
    n_cells = len(d.get("cell_counts", {}))
    total = d.get("total", 0)

    report = {}
    if report_json.exists():
        report = json.load(open(report_json))

    return OrderedDict([
        ("status", "OK"),
        ("path", str(density_json)),
        ("total_frames", total),
        ("n_cells", n_cells),
        ("p20_threshold", d.get("p20_threshold")),
        ("p50_threshold", d.get("p50_threshold")),
        ("in_support_density", report.get("in_support_density")),
        ("low_density", report.get("low_density")),
        ("pass", total > 0 and n_cells > 0),
    ])


def check_p0_2_episode_rng():
    """Check that synth_pointcloud.py has rng_seed parameter."""
    synth_pc_file = PROJECT_DIR / "diffusion_policy_3d" / "model" / "common" / "synth_pointcloud.py"
    if not synth_pc_file.exists():
        return {"status": "NOT_FOUND", "path": str(synth_pc_file)}

    content = synth_pc_file.read_text()
    has_rng_seed = "rng_seed" in content
    has_local_generator = "torch.Generator" in content
    has_np_rng = "np.random.default_rng(" in content

    # Check env wiring
    env_files = [
        PROJECT_DIR / "diffusion_policy_3d" / "env" / "kinematic_uav_env.py",
        PROJECT_DIR / "diffusion_policy_3d" / "env" / "kinematic_uav_env_v2.py",
        PROJECT_DIR / "diffusion_policy_3d" / "env" / "gazebo_uav_env.py",
    ]
    env_wired = []
    for ef in env_files:
        if ef.exists():
            ec = ef.read_text()
            wired = "_episode_rng_seed" in ec and "rng_seed=" in ec
            env_wired.append({"file": ef.name, "wired": wired})

    return OrderedDict([
        ("status", "OK" if has_rng_seed else "MISSING"),
        ("synth_pc_has_rng_seed", has_rng_seed),
        ("synth_pc_has_local_generator", has_local_generator),
        ("synth_pc_has_np_rng", has_np_rng),
        ("env_wiring", env_wired),
        ("pass", has_rng_seed and has_local_generator and has_np_rng
                 and all(e["wired"] for e in env_wired)),
    ])


def check_p0_3_guard_loop_audit():
    """Check latest guard/loop audit results."""
    audit_dirs = sorted(SHIYAN_DIR.glob("guard_loop_audit_*"))
    if not audit_dirs:
        return {"status": "NOT_FOUND"}

    latest = audit_dirs[-1]
    audit_json = latest / "guard_loop_audit.json"
    if not audit_json.exists():
        return {"status": "NOT_FOUND", "path": str(audit_json)}

    audit = json.load(open(audit_json))
    fp_rate = audit.get("false_positive_analysis", {}).get("false_positive_rate", 0)
    loop_pass = audit.get("gates", {}).get("false_loop_on_successful_recovery", {}).get("pass", False)
    guard_mean = audit.get("guard_penalty_breakdown", {}).get("mean_guard_total", 0)

    # Check if fixes were applied
    reward_v13 = PROJECT_DIR / "diffusion_policy_3d" / "env" / "reward_v13.py"
    reward_content = reward_v13.read_text()
    fixes_applied = ("loop_sign_changes" in reward_content
                     and "guard_d_soft_threshold" in reward_content)

    return OrderedDict([
        ("status", "OK"),
        ("audit_path", str(audit_json)),
        ("false_positive_rate", round(fp_rate, 4)),
        ("false_loop_on_successful_recovery_pass", loop_pass),
        ("mean_guard_penalty", round(guard_mean, 2)),
        ("fixes_applied_to_reward_v13", fixes_applied),
        ("pass", fixes_applied),
    ])


def check_p0_4_instrumentation():
    """Check instrumentation module exists."""
    instr_file = PROJECT_DIR / "diffusion_policy_3d" / "training" / "p0_4_instrumentation.py"
    if not instr_file.exists():
        return {"status": "NOT_FOUND", "path": str(instr_file)}

    content = instr_file.read_text()
    return OrderedDict([
        ("status", "OK"),
        ("path", str(instr_file)),
        ("has_RouteATrainingLogger", "class RouteATrainingLogger" in content),
        ("has_required_keys", all(k in content for k in [
            "critic_loss_total", "actor_loss_pg",
            "reward_terms_episode_sum", "loop_detected_rate",
        ])),
    ])


def check_p0_5_toy_test():
    """Check latest stochastic path toy test results."""
    test_dirs = sorted(SHIYAN_DIR.glob("stochastic_path_toy_test_*"))
    if not test_dirs:
        return {"status": "NOT_FOUND"}

    latest = test_dirs[-1]
    result_json = latest / "stochastic_path_toy_test.json"
    if not result_json.exists():
        return {"status": "NOT_FOUND", "path": str(result_json)}

    r = json.load(open(result_json))
    gates = r.get("gates", {})
    return OrderedDict([
        ("status", "OK"),
        ("path", str(result_json)),
        ("gates", gates),
        ("pass", gates.get("pass", False)),
    ])


def check_p0_6_bc_variance():
    """Check latest BC variance calibration results."""
    var_dirs = sorted(SHIYAN_DIR.glob("bc_variance_calibration_*"))
    if not var_dirs:
        return {"status": "NOT_FOUND"}

    latest = var_dirs[-1]
    summary_json = latest / "bc_variance_summary.json"
    if not summary_json.exists():
        return {"status": "NOT_FOUND", "path": str(summary_json)}

    summary = json.load(open(summary_json))
    # Extract worst CI width
    max_ci_width = 0.0
    for sname, sdata in summary.items():
        sr_width = sdata.get("SR", {}).get("ci_width_pp", 0)
        max_ci_width = max(max_ci_width, sr_width)

    return OrderedDict([
        ("status", "OK"),
        ("path", str(summary_json)),
        ("n_scenarios", len(summary)),
        ("max_SR_CI_width_pp", round(max_ci_width, 2)),
        ("recommended_A2_threshold", max(5.0, round(max_ci_width * 2, 1))),
    ])


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    output_dir = SHIYAN_DIR / f"preflight_summary_{time.strftime('%Y%m%d_%H%M')}"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("P0-7: Preflight Summary")
    print("=" * 60)

    checks = OrderedDict([
        ("P0-1_offline_density", check_p0_1_offline_density()),
        ("P0-2_episode_rng", check_p0_2_episode_rng()),
        ("P0-3_guard_loop_audit", check_p0_3_guard_loop_audit()),
        ("P0-4_instrumentation", check_p0_4_instrumentation()),
        ("P0-5_stochastic_path_toy_test", check_p0_5_toy_test()),
        ("P0-6_bc_variance", check_p0_6_bc_variance()),
    ])

    # Determine overall readiness
    all_pass = True
    for name, result in checks.items():
        p = result.get("pass", result.get("status") != "NOT_FOUND")
        if not p:
            all_pass = False
        status_icon = "✅" if p else "❌"
        print(f"  {status_icon} {name}: {result.get('status', '?')}"
              f"{' (pass)' if p else ' (fail)'}")

    # Build summary
    preflight = OrderedDict([
        ("meta", OrderedDict([
            ("task", "P0 Preflight Summary — Route A-RF"),
            ("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
            ("description", "Aggregated P0 checks for Route A-RF short-train readiness"),
        ])),
        ("overall_ready", all_pass),
        ("checks", checks),
        ("recommendations", []),
    ])

    # Generate recommendations
    recs = []
    if not all_pass:
        recs.append("❌ Some P0 checks failed — address before Route A-RF short-train")
    else:
        recs.append("✅ All P0 checks pass — ready for Route A-RF short-train")

    # P0-3 specific
    p0_3 = checks.get("P0-3_guard_loop_audit", {})
    if p0_3.get("status") == "OK":
        if p0_3.get("false_loop_on_successful_recovery_pass"):
            recs.append("✅ Guard/loop: false positive rate within threshold")
        else:
            recs.append("⚠️ Guard/loop: false loop on success still high — monitor during RL")

        fp_rate = p0_3.get("false_positive_rate", 0)
        if fp_rate > 0.05:
            recs.append("⚠️ Guard loop FP rate > 5% — increases were applied to reward_v13.py")

    # P0-6 specific
    p0_6 = checks.get("P0-6_bc_variance", {})
    if p0_6.get("status") == "OK":
        ci_w = p0_6.get("max_SR_CI_width_pp", 0)
        recs.append(f"📊 BC baseline SR CI width: {ci_w:.1f}pp — "
                    f"recommended A2 delta threshold: {p0_6.get('recommended_A2_threshold', 5):.0f}pp")

    preflight["recommendations"] = recs

    # Save
    out_path = output_dir / "preflight_summary.json"
    with open(out_path, "w") as f:
        json.dump(preflight, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Overall readiness: {'✅ READY' if all_pass else '❌ NOT READY'}")
    print(f"Summary: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
