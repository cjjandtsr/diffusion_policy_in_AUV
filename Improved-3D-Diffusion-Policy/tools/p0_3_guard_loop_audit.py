#!/usr/bin/env python3
"""
p0_3_guard_loop_audit.py — P0-3: Guard / loop false-positive audit (§8.3)

目标: 确保 guard 不会压制真实 early avoidance / dense recovery。

抽查集:
  - canonical successful detours
  - dense trap recovery episodes
  - loop_detected=True successes
  - guard penalty high but no collision episodes

验收:
  | 指标 | 阈值 |
  |------|------|
  | false loop on successful recovery | <5% |
  | guard changes success ranking | no (SR order preserved) |
  | path_length_ratio >1.2 + success+clearance improved | manually reviewed |

输出:
  - guard_loop_audit.json       (aggregated audit result)
  - false_positive_cases.jsonl  (each FP case with per-step breakdown)
  - audit_report.log            (human-readable report)
"""

import json, os, sys, math, time, argparse
from collections import defaultdict, OrderedDict
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch, dill
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
from diffusion_policy_3d.env.kinematic_uav_env_v2 import WORLDS_DIR, WORLD_ENV_MAP
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from diffusion_policy_3d.policy.phase1a_policy_fm_rl import Phase1aPolicyFM
from diffusion_policy_3d.policy.fm_base_policy import FMBasePolicy

# ═══════════════════════════════════════════════════════════════════
# Constants (mirrors eval_reward_v13_short.py)
# ═══════════════════════════════════════════════════════════════════

OOD_WORLDS = [4, 8, 11, 12, 18]
MAX_STEPS = 60
REWARD_TERM_KEYS = [
    "base", "pbrs", "onset", "future_clearance",
    "trap", "flip", "support", "guard",
]

# ── Classification rules (from design doc §7.3) ──
# False positive categories
FP_SUCCESSFUL_DETOUR = "successful_detour_misidentified"
FP_DENSE_RECOVERY = "dense_trap_recovery_overpenalized"
FP_LOW_RISK_DETOUR = "low_risk_detour_penalty"
TP_REAL_OSCILLATION = "real_oscillation"


# ═══════════════════════════════════════════════════════════════════
# Policy loading (reuses eval_reward_v13_short.py logic)
# ═══════════════════════════════════════════════════════════════════

def load_bc_policy(ckpt_path, device="cuda:0"):
    ckpt = torch.load(ckpt_path, map_location=device, pickle_module=dill)
    cfg = ckpt["cfg"]
    sd = ckpt["state_dicts"]["ema_model"]
    pc = cfg["policy"]

    fm = Phase1aPolicyFM(
        shape_meta=cfg["shape_meta"],
        noise_scheduler=DDIMScheduler(
            num_train_timesteps=50, beta_start=0.0001, beta_end=0.02,
            beta_schedule="squaredcos_cap_v2", clip_sample=True,
            set_alpha_to_one=True, steps_offset=0, prediction_type="sample",
        ),
        horizon=cfg["horizon"],
        n_action_steps=cfg["n_action_steps"],
        n_obs_steps=cfg["n_obs_steps"],
        obs_as_global_cond=pc.get("obs_as_global_cond", True),
        diffusion_step_embed_dim=pc.get("diffusion_step_embed_dim", 128),
        down_dims=pc.get("down_dims", (256, 512, 1024)),
        kernel_size=pc.get("kernel_size", 5),
        n_groups=pc.get("n_groups", 8),
        anchor_bank_path=None,
        cfg_lambda=1.5, num_fm_steps=4, sigma_min=0.005, fm_t_scale=50,
        solver="euler", fm_anchor_t_start=0.95, use_pc_color=False,
        pointnet_type="multi_stage_pointnet",
        pointcloud_encoder_cfg=pc.get("pointcloud_encoder_cfg", None),
        point_downsample=pc.get("point_downsample", True),
    )
    fm.load_state_dict(sd, strict=False)
    fm.to(device).eval()
    base_policy = FMBasePolicy(fm, device)
    # Warm-up
    dummy_z = torch.zeros(1, 176, device=device)
    dummy_obs = {
        "point_cloud": np.zeros((8, 4096, 3), dtype=np.float32),
        "agent_pos": np.zeros((8, 9), dtype=np.float32),
    }
    with torch.no_grad():
        _ = base_policy(dummy_obs, dummy_z)
    return base_policy


def create_env(world_id, z_min=1.0, z_max=5.0, disturb=False, seed=42):
    wname = WORLD_ENV_MAP[world_id]
    obs_json = str(WORLDS_DIR / f"{wname}.obstacles.json")
    synth_pc = SynthPointCloud(obs_json, device="cpu")
    env = KinematicUAVEnv(
        world_id=world_id, synth_pc=synth_pc,
        rl_action_steps=8, h_min=0.3,
        z_min=z_min, z_max=z_max,
        disturb_enable=disturb,
        disturb_prob=0.2, disturb_pos_scale=0.3,
        mid_state_init_ratio=0.0,
        seed=seed,
        reward_version="v13",
    )
    return env


def run_episode(env, base_policy, device, max_steps=MAX_STEPS):
    """Run one BC episode, return trajectory + guard/loop data per step."""
    obs, info = env.reset()
    z = np.zeros(176, dtype=np.float32)
    z_tensor = torch.from_numpy(z).unsqueeze(0).to(device)

    steps = []
    done_reason = "truncated"
    traj_positions = [env._drone_pos.copy()]
    traj_yaws = [float(env._drone_yaw)]

    for step_i in range(max_steps):
        raw_obs = env._get_obs_dict()
        raw_obs["point_cloud"] = raw_obs["point_cloud"][..., :3]
        with torch.no_grad():
            real_action = base_policy(raw_obs, z_tensor)

        obs, reward, terminated, truncated, info = env.step(real_action[0])

        # ── Step data: guard/loop focus ──
        metrics = info.get("reward_metrics", {})
        terms = info.get("reward_terms", {})

        step_data = {
            "step": step_i,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "guard": float(terms.get("guard", 0.0)),
            "loop_detected": bool(metrics.get("loop_detected", False)),
            "loop_detected_rate": float(metrics.get("loop_detected_rate", 0.0)),
            "risk_gate": float(metrics.get("risk_gate", 0.0)),
            "trap_active": bool(metrics.get("trap_active", False)),
            "trap_recovered": bool(metrics.get("trap_recovered", False)),
            "onset_active": bool(metrics.get("onset_active", False)),
            "d_soft_h": float(metrics.get("d_soft_h", -1.0)),
            "side_flip_count": float(metrics.get("side_flip_count", 0)),
        }
        steps.append(step_data)
        traj_positions.append(env._drone_pos.copy())
        traj_yaws.append(float(env._drone_yaw))

        if terminated or truncated:
            done_reason = info.get("done_reason", "unknown")
            break

    reached = (done_reason == "reached")
    collided = ("collision" in done_reason or "penetrat" in done_reason
                or "crash" in done_reason)
    traj = np.array(traj_positions)
    path_len = float(np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=1)))
    straight_dist = float(np.linalg.norm(traj[-1] - traj[0]))
    path_ratio = straight_dist / path_len if path_len > 0 else 1.0

    return {
        "reached": reached,
        "collided": collided,
        "done_reason": done_reason,
        "reward_total": round(float(np.sum([s["reward"] for s in steps])), 4),
        "length": len(steps),
        "path_length_ratio": round(path_ratio, 4),
        "guard_total": round(float(np.sum([s["guard"] for s in steps])), 4),
        "loop_count": sum(1 for s in steps if s["loop_detected"]),
        "loop_detected_rate": float(np.mean([1.0 if s["loop_detected"] else 0.0 for s in steps])),
        "trap_episode": any(s["trap_active"] for s in steps),
        "trap_recovered": any(s["trap_recovered"] for s in steps),
        "steps": steps,
    }


# ═══════════════════════════════════════════════════════════════════
# Classification logic
# ═══════════════════════════════════════════════════════════════════

def classify_loop_false_positive(ep_data):
    """Classify an episode's loop_detected signals.

    Returns list of (step_idx, classification, reason) for each loop event.
    """
    fp_events = []

    for s in ep_data["steps"]:
        if not s["loop_detected"]:
            continue

        step_idx = s["step"]

        # Case 1: Successful detour → FP
        # If this is a successful episode, any loop detection is likely false
        if ep_data["reached"]:
            fp_events.append((step_idx, FP_SUCCESSFUL_DETOUR,
                              f"successful episode (reached), loop at step {step_idx}"))
            continue

        # Case 2: Dense trap recovery → FP
        # Loop detected during or near trap recovery
        if s["trap_recovered"] or s["trap_active"]:
            fp_events.append((step_idx, FP_DENSE_RECOVERY,
                              f"trap recovery context at step {step_idx}"))
            continue

        # Case 3: Low risk detour → FP
        # If risk_gate is very low (< guard_risk_threshold=0.2)
        # AND there's no obstacle interaction (d_soft_h high)
        if s["risk_gate"] < 0.2 and s["d_soft_h"] > 0.75:
            fp_events.append((step_idx, FP_LOW_RISK_DETOUR,
                              f"low risk (rg={s['risk_gate']:.3f}) high clearance (d={s['d_soft_h']:.2f})"))
            continue

        # Otherwise: True positive (real oscillation)
        fp_events.append((step_idx, TP_REAL_OSCILLATION,
                          f"real oscillation at step {step_idx}, rg={s['risk_gate']:.3f}, d={s['d_soft_h']:.2f}"))

    return fp_events


def compute_detour_penalty_analysis(ep_data):
    """Analyze detour penalty (non-loop guard) for low-risk detour false positive.

    Returns dict with penalty breakdown.
    """
    # Guard penalty from detour (not loop)
    detour_penalty_steps = [s for s in ep_data["steps"]
                           if s["guard"] < 0 and not s["loop_detected"]
                           and s["risk_gate"] < 0.2]

    detour_penalty = sum(s["guard"] for s in detour_penalty_steps) if detour_penalty_steps else 0.0
    return {
        "detour_penalty_total": round(detour_penalty, 4),
        "detour_penalty_steps": len(detour_penalty_steps),
        "low_risk_penalty_contrib": round(detour_penalty / max(abs(ep_data["guard_total"]), 1e-6), 4),
    }


# ═══════════════════════════════════════════════════════════════════
# Main audit
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=40,
                        help="Total episodes across OOD worlds")
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = str(PROJECT_DIR / "data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt")
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M")
        args.output_dir = str(PROJECT_DIR / "program" / "shiyan" /
                              f"guard_loop_audit_{ts}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Logging ──
    log_path = os.path.join(args.output_dir, "audit_report.log")
    log_file = open(log_path, "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 60)
    log("P0-3: Guard / Loop False-Positive Audit")
    log(f"Checkpoint: {args.ckpt}")
    log(f"Episodes: {args.n_episodes}")
    log(f"Output: {args.output_dir}")
    log("=" * 60)

    # ── Load policy ──
    log("\nLoading BC policy...")
    try:
        base_policy = load_bc_policy(args.ckpt, device=args.device)
        log("BC policy loaded OK")
    except Exception as e:
        log(f"FAILED: {e}")
        import traceback; traceback.print_exc(file=log_file)
        return

    # ── Run episodes ──
    all_episodes = []
    eps_per_world = max(1, args.n_episodes // len(OOD_WORLDS))
    remaining = args.n_episodes

    log(f"\nRunning {args.n_episodes} episodes across {len(OOD_WORLDS)} worlds...")
    for world_id in OOD_WORLDS:
        n_this = min(eps_per_world, remaining)
        for ep in range(n_this):
            seed = args.seed + hash(f"guard_audit_{world_id}_{ep}") % (2**31 - 1)
            env = create_env(world_id, seed=seed, disturb=True)
            ep_data = run_episode(env, base_policy, args.device)
            ep_data["world_id"] = world_id
            ep_data["episode_seed"] = seed
            all_episodes.append(ep_data)
            env.close()
            remaining -= 1
        if remaining <= 0:
            break

    log(f"  Ran {len(all_episodes)} episodes")

    # ── Classify each episode ──
    all_fp_events = []
    per_ep_classification = []

    for ep_idx, ep_data in enumerate(all_episodes):
        fp_events = classify_loop_false_positive(ep_data)
        detour = compute_detour_penalty_analysis(ep_data)

        n_fp = sum(1 for _, cat, _ in fp_events
                   if cat in (FP_SUCCESSFUL_DETOUR, FP_DENSE_RECOVERY, FP_LOW_RISK_DETOUR))
        n_tp = sum(1 for _, cat, _ in fp_events if cat == TP_REAL_OSCILLATION)

        entry = {
            "episode": ep_idx,
            "world_id": ep_data["world_id"],
            "reached": ep_data["reached"],
            "collided": ep_data["collided"],
            "done_reason": ep_data["done_reason"],
            "guard_total": ep_data["guard_total"],
            "loop_count": ep_data["loop_count"],
            "loop_detected_rate": ep_data["loop_detected_rate"],
            "trap_episode": ep_data["trap_episode"],
            "trap_recovered": ep_data["trap_recovered"],
            "path_length_ratio": ep_data["path_length_ratio"],
            "reward_total": ep_data["reward_total"],
            "n_fp_events": n_fp,
            "n_tp_events": n_tp,
            "detour_penalty": detour,
        }
        per_ep_classification.append(entry)
        all_fp_events.extend([
            {"episode": ep_idx, "step": s, "category": cat, "reason": reason}
            for s, cat, reason in fp_events
        ])

    # ── Aggregate statistics ──
    n_total = len(all_episodes)

    # Loop detection stats
    eps_with_loop = [e for e in per_ep_classification if e["loop_count"] > 0]
    n_loop_episodes = len(eps_with_loop)
    total_loop_events = sum(e["loop_count"] for e in per_ep_classification)

    # FP classification
    fp_by_category = defaultdict(int)
    for ev in all_fp_events:
        fp_by_category[ev["category"]] += 1

    n_fp_total = sum(1 for ev in all_fp_events
                     if ev["category"] in (FP_SUCCESSFUL_DETOUR, FP_DENSE_RECOVERY, FP_LOW_RISK_DETOUR))
    n_tp_total = sum(1 for ev in all_fp_events if ev["category"] == TP_REAL_OSCILLATION)
    fp_rate = n_fp_total / max(total_loop_events, 1)

    # False loop on successful recovery
    success_with_loop = [e for e in per_ep_classification
                         if e["reached"] and e["loop_count"] > 0]
    false_loop_on_success = len(success_with_loop)
    total_success = sum(1 for e in per_ep_classification if e["reached"])
    false_loop_on_success_rate = false_loop_on_success / max(total_success, 1)

    # High guard episodes
    high_guard_eps = [e for e in per_ep_classification if e["guard_total"] < -30]
    high_guard_no_collision = [e for e in high_guard_eps if not e["collided"]]
    high_guard_success = [e for e in high_guard_eps if e["reached"]]

    # Trap recovery analysis
    trap_episodes = [e for e in per_ep_classification if e["trap_episode"]]
    trap_with_loop = [e for e in trap_episodes if e["loop_count"] > 0]

    # Path length analysis
    high_plr = [e for e in per_ep_classification
                if e["path_length_ratio"] > 1.2 and e["reached"]]

    # ── Build results ──
    aggregate = OrderedDict([
        ("meta", OrderedDict([
            ("task", "P0-3 guard/loop false-positive audit"),
            ("checkpoint", args.ckpt),
            ("n_episodes", n_total),
            ("eval_time", time.strftime("%Y-%m-%dT%H:%M:%S")),
        ])),
        ("overview", OrderedDict([
            ("n_total", n_total),
            ("n_success", total_success),
            ("n_collision", sum(1 for e in per_ep_classification if e["collided"])),
            ("SR", round(total_success / max(n_total, 1), 4)),
            ("CR", round(sum(1 for e in per_ep_classification if e["collided"]) / max(n_total, 1), 4)),
        ])),
        ("loop_detection", OrderedDict([
            ("n_episodes_with_loop", n_loop_episodes),
            ("total_loop_events", total_loop_events),
            ("mean_loop_detected_rate", round(
                np.mean([e["loop_detected_rate"] for e in per_ep_classification]), 4)),
            ("median_loop_detected_rate", round(
                np.median([e["loop_detected_rate"] for e in per_ep_classification]), 4)),
        ])),
        ("false_positive_analysis", OrderedDict([
            ("total_events_classified", len(all_fp_events)),
            ("false_positive_events", n_fp_total),
            ("true_positive_events", n_tp_total),
            ("false_positive_rate", round(fp_rate, 4)),
            ("by_category", {
                "successful_detour": fp_by_category.get(FP_SUCCESSFUL_DETOUR, 0),
                "dense_trap_recovery": fp_by_category.get(FP_DENSE_RECOVERY, 0),
                "low_risk_detour": fp_by_category.get(FP_LOW_RISK_DETOUR, 0),
                "real_oscillation": fp_by_category.get(TP_REAL_OSCILLATION, 0),
            }),
        ])),
        ("gates", OrderedDict([
            ("false_loop_on_successful_recovery", OrderedDict([
                ("n_cases", false_loop_on_success),
                ("rate", round(false_loop_on_success_rate, 4)),
                ("threshold", 0.05),
                ("pass", false_loop_on_success_rate < 0.05),
            ])),
            ("guard_changes_success_ranking", OrderedDict([
                ("guard_penalty_high_but_success", len(high_guard_success)),
                ("high_guard_no_collision", len(high_guard_no_collision)),
                ("note", "manual review needed if SR correlates negatively with guard magnitude"),
            ])),
            ("path_length_ratio_review", OrderedDict([
                ("high_plr_success_episodes", len(high_plr)),
                ("episodes", [{"world": e["world_id"], "plr": e["path_length_ratio"],
                               "guard_total": e["guard_total"], "reward_total": e["reward_total"]}
                              for e in high_plr]),
                ("note", "manual review recommended"),
            ])),
        ])),
        ("guard_penalty_breakdown", OrderedDict([
            ("mean_guard_total", round(np.mean([e["guard_total"] for e in per_ep_classification]), 4)),
            ("median_guard_total", round(np.median([e["guard_total"] for e in per_ep_classification]), 4)),
            ("guard_quantiles", {
                "p25": round(float(np.percentile([e["guard_total"] for e in per_ep_classification], 25)), 4),
                "p50": round(float(np.percentile([e["guard_total"] for e in per_ep_classification], 50)), 4),
                "p75": round(float(np.percentile([e["guard_total"] for e in per_ep_classification], 75)), 4),
            }),
            ("detour_penalty_share",
             round(np.mean([e["detour_penalty"]["low_risk_penalty_contrib"]
                           for e in per_ep_classification]), 4)),
        ])),
        ("recommendations", []),
    ])

    # ── Generate recommendations ──
    recs = []

    if not aggregate["gates"]["false_loop_on_successful_recovery"]["pass"]:
        recs.append(
            "🔴 false_loop_on_successful_recovery > 5%: "
            "Increase sign_change threshold from 3 → 4 (§7.3 table)"
        )

    if aggregate["false_positive_analysis"]["by_category"]["dense_trap_recovery"] > 0:
        recs.append(
            "🟡 dense trap recovery over-penalized: "
            "Gate guard on risk_gate < 0.2 AND no obstacle interaction (§7.3 table)"
        )

    if len(high_plr) > 0:
        plr_worlds = [(e["world_id"], e["plr"]) for e in high_plr]
        recs.append(
            f"🟡 {len(high_plr)} success episodes with path_length_ratio > 1.2 "
            f"(worlds: {plr_worlds}): manual review needed — "
            f"if SR/clearance/onset all improved, this is necessary detour, not hack"
        )

    if aggregate["guard_penalty_breakdown"]["mean_guard_total"] < -30:
        recs.append(
            f"🟡 Mean guard penalty {aggregate['guard_penalty_breakdown']['mean_guard_total']:.1f} is large. "
            f"Check if this is suppressing early avoidance — consider reducing w_eff "
            f"if onset_distance / trap_recovery_rate are below expectations"
        )

    # Build audit file path
    if n_fp_total == 0 and n_tp_total == 0:
        recs.append("🟢 No loop events detected — guard/loop not actionable in this sample")
    elif fp_rate < 0.05:
        recs.append(f"✅ FP rate {fp_rate:.2%} < 5% — guard/loop classification is reliable")
    else:
        recs.append(f"🔴 FP rate {fp_rate:.2%} ≥ 5% — guard is suppressing valid behavior")

    aggregate["recommendations"] = recs

    # ── Write outputs ──
    # Per-episode summary
    with open(os.path.join(args.output_dir, "per_episode.json"), "w") as f:
        json.dump(per_ep_classification, f, indent=2, default=str)

    # FP cases JSONL
    with open(os.path.join(args.output_dir, "false_positive_cases.jsonl"), "w") as f:
        for ev in all_fp_events:
            f.write(json.dumps(ev, default=str) + "\n")

    # Aggregate
    with open(os.path.join(args.output_dir, "guard_loop_audit.json"), "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    # ── Log report ──
    log("\n" + "=" * 60)
    log("GUARD / LOOP AUDIT REPORT")
    log("=" * 60)

    log(f"\n📊 Overview: {n_total} episodes, SR={aggregate['overview']['SR']:.3f}, "
        f"CR={aggregate['overview']['CR']:.3f}")

    log(f"\n🔄 Loop detection:")
    log(f"  Episodes with loop: {n_loop_episodes}/{n_total}")
    log(f"  Total loop events: {total_loop_events}")
    log(f"  Mean loop_detected_rate: {aggregate['loop_detection']['mean_loop_detected_rate']:.4f}")
    log(f"  Median loop_detected_rate: {aggregate['loop_detection']['median_loop_detected_rate']:.4f}")

    log(f"\n🎯 False positive analysis:")
    log(f"  Total classified events: {len(all_fp_events)}")
    log(f"  FP: {n_fp_total} ({fp_rate:.2%})")
    log(f"  TP: {n_tp_total} ({n_tp_total/max(len(all_fp_events),1):.2%})")
    log(f"  By category: {json.dumps(aggregate['false_positive_analysis']['by_category'])}")

    log(f"\n✅ Gate: false_loop_on_successful_recovery = "
        f"{false_loop_on_success}/{total_success} "
        f"({false_loop_on_success_rate:.2%}) "
        f"{'PASS' if aggregate['gates']['false_loop_on_successful_recovery']['pass'] else 'FAIL'}")

    log(f"\n💰 Guard penalty:")
    log(f"  Mean: {aggregate['guard_penalty_breakdown']['mean_guard_total']:.1f}")
    log(f"  Median: {aggregate['guard_penalty_breakdown']['median_guard_total']:.1f}")
    log(f"  Detour penalty share: {aggregate['guard_penalty_breakdown']['detour_penalty_share']:.2%}")
    log(f"  High guard (no collision): {len(high_guard_no_collision)} cases")
    log(f"  High guard (reached): {len(high_guard_success)} cases")

    log(f"\n📏 Path length review:")
    log(f"  High PLR + success: {len(high_plr)} episodes")

    log(f"\n💡 Recommendations:")
    for r in recs:
        log(f"  {r}")

    log(f"\n{'=' * 60}")
    log(f"Results: {args.output_dir}")
    log(f"{'=' * 60}")

    log_file.close()
    print(f"\nReport: {log_path}")


if __name__ == "__main__":
    main()
