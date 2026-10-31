#!/usr/bin/env python3
"""
C2-OOC Expanded Paired Eval
============================
Fixed episode-bank (32 eps/role, 96 total), rl_val worlds only.
Evaluates Pre (BC baseline) / Mid (Phase 10) / Post (Phase 20) against the same bank.
Outputs: per-role SR/CR, Wilson 95% CI, paired switches, per-role fine-grained deltas,
adjudication report.

Usage:
  python scripts/run_c2_ooc_expanded_eval.py \
      --admission_manifest /path/to/admission_manifest.json \
      --bc_ckpt data/outputs/.../latest.ckpt \
      --mid_ckpt /path/to/mid_checkpoints/actor_phase10.pt \
      --post_ckpt /path/to/a2_actor_best.pt \
      --post_critic_ckpt /path/to/a2_critic_best.pt \
      --b1_critic_ckpt /path/to/b1_critic_rolefix/a2_critic_best.pt \
      --output_dir /path/to/expanded_eval_output \
      --episodes_per_role 32 \
      --seed 42
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc
import torch

import numpy as np
import torch

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
    RouteARLAdmissionManifest, load_routea_rl_admission_manifest,
)
from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% confidence interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def generate_fixed_episode_bank(
    val_pool: List[Tuple[str, str]],
    world_meta_lookup: dict,
    role_to_configs: Dict[str, int],
    rng: np.random.Generator,
    curriculum_stage: str = "C2",
) -> List[dict]:
    """Generate deterministic episode configs for paired eval.

    Returns list of dicts: {world_name, start_pos, goal_pos, start_yaw, episode_seed, role}
    """
    from diffusion_policy_3d.env.collision_geom import load_obstacles, compute_obstacle_clearance

    _ROLE_PARAMS = {
        'anticipatory':         {'d_xy': (10, 24), 'start_cl': 0.60, 'goal_cl': 2.50},
        'dense_recovery':       {'d_xy': (5, 18),  'start_cl': 0.60, 'goal_cl': 1.50},
        'support_expansion':    {'d_xy': (8, 26),  'start_cl': 0.80, 'goal_cl': 2.00},
        'yaw_lateral_recovery': {'d_xy': (6, 18),  'start_cl': 0.80, 'goal_cl': 2.00},
    }

    configs = []
    for role, n_eps in role_to_configs.items():
        rp = _ROLE_PARAMS.get(role, _ROLE_PARAMS['anticipatory'])
        d_xy_min, d_xy_max = rp['d_xy']
        start_cl, goal_cl = rp['start_cl'], rp['goal_cl']
        x_min, x_max, y_min, y_max = -14.0, 14.0, -14.0, 14.0

        # Find val pool worlds matching this role
        role_worlds = []
        for idx, (wname, obs_json) in enumerate(val_pool):
            meta = world_meta_lookup.get(wname)
            wrole = (meta.role_active or meta.role_validated) if meta else 'anticipatory'
            if wrole == role:
                role_worlds.append((wname, obs_json, idx))

        if not role_worlds:
            print(f"  [WARN] No val worlds for role '{role}', using all val worlds")
            role_worlds = [(wname, obs_json, i) for i, (wname, obs_json) in enumerate(val_pool)]

        for ep_i in range(n_eps):
            wname, obs_json, _ = role_worlds[ep_i % len(role_worlds)]
            obstacles = load_obstacles(obs_json)
            ep_seed = int(rng.integers(0, 2**31))
            ep_rng = np.random.default_rng(ep_seed)

            found = False
            sx = sy = sz = gx = gy = gz = 0.0
            for _attempt in range(100):
                sx = float(ep_rng.uniform(x_min, x_max))
                sy = float(ep_rng.uniform(y_min, y_max))
                sz = float(ep_rng.uniform(1.3, 4.7))
                gx = float(ep_rng.uniform(x_min, x_max))
                gy = float(ep_rng.uniform(y_min, y_max))
                gz = float(ep_rng.uniform(1.5, 4.5))

                d_xy = math.hypot(gx - sx, gy - sy)
                if d_xy < d_xy_min or d_xy > d_xy_max:
                    continue
                if math.hypot(d_xy, gz - sz) < 4.0:
                    continue
                start_clearance = compute_obstacle_clearance(obstacles, (sx, sy, sz))
                goal_clearance = compute_obstacle_clearance(obstacles, (gx, gy, gz))
                if start_clearance < start_cl or goal_clearance < goal_cl:
                    continue
                found = True
                break

            if not found:
                print(f"  [WARN] Could not sample start/goal for {role} ep {ep_i} on {wname}, using fallback")
                sx, sy, sz = float(ep_rng.uniform(x_min, x_max)), float(ep_rng.uniform(y_min, y_max)), float(ep_rng.uniform(1.3, 4.7))
                gx, gy, gz = float(ep_rng.uniform(x_min, x_max)), float(ep_rng.uniform(y_min, y_max)), float(ep_rng.uniform(1.5, 4.5))

            start_yaw = float(ep_rng.uniform(-math.pi, math.pi))
            configs.append({
                "world_name": wname,
                "start_pos": [sx, sy, sz],
                "goal_pos": [gx, gy, gz],
                "start_yaw": start_yaw,
                "episode_seed": ep_seed,
                "episode_id": f"{role}_{ep_i:03d}",
                "role": role,
                "val_pool_index": next(i for i, (w, _) in enumerate(val_pool) if w == wname),
            })

    return configs


def rollout_single_episode(
    actor, base_policy, env, device, episode_seed: int, deterministic: bool = False
) -> dict:
    """Run a single episode and return per-step metrics."""
    from scripts.train_a2_10k_capability import rollout_episode_a2
    transitions, obs_buf, ep_sum = rollout_episode_a2(
        actor, base_policy, env, device,
        episode_seed=episode_seed, world_id=-1,
        deterministic=deterministic, lazy_teacher=True,
    )

    # Extract SR and CR from episode summary (matching rollout_episode_a2 field names)
    sr = int(ep_sum.get("reached", 0))
    cr = int(ep_sum.get("collided", 0))
    reward = float(ep_sum.get("reward_total", 0.0))
    path_length = float(ep_sum.get("path_length", 0.0))
    episode_length = int(ep_sum.get("length", 0))
    min_clearance = float(ep_sum.get("min_clearance", float('nan')))
    onset_time = ep_sum.get("onset_time")  # can be None
    future_clearance_softmin = float(ep_sum.get("future_clearance_softmin", float('nan')))
    trap_recovery = int(ep_sum.get("trap_recovered", 0))
    low_density_coverage = float(ep_sum.get("low_density_coverage", 0.0))
    progress_to_goal = float(ep_sum.get("progress_to_goal", float('nan')))

    return {
        "SR": sr, "CR": cr, "reward": reward, "path_length": path_length,
        "episode_length": episode_length, "min_clearance": min_clearance,
        "onset_time": onset_time, "future_clearance_softmin": future_clearance_softmin,
        "trap_recovery": trap_recovery, "low_density_coverage": low_density_coverage,
        "progress_to_goal": progress_to_goal,
    }


def run_eval_on_bank(
    actor, base_policy, device, val_pool, world_meta_lookup,
    episode_bank: List[dict], curriculum_stage: str = "C2",
) -> List[dict]:
    """Evaluate all episodes in the fixed bank."""
    results = []
    from collections import defaultdict
    by_world = defaultdict(list)
    for i, cfg in enumerate(episode_bank):
        by_world[cfg["world_name"]].append((i, cfg))

    for wname, episodes in by_world.items():
        obs_json = None
        for w, oj in val_pool:
            if w == wname:
                obs_json = oj
                break
        if obs_json is None:
            print(f"  [WARN] World {wname} not found in val_pool, skipping {len(episodes)} episodes")
            continue

        for ep_idx, cfg in episodes:
            try:
                env = KinematicUAVEnv(
                    world_id=-1,  # use world_pool (not default world)
                    world_pool=[(wname, obs_json)],
                    synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                    rl_action_steps=8, h_min=0.3, z_min=1.0, z_max=5.0,
                    disturb_enable=False,
                    mid_state_init_ratio=0.0,
                    seed=cfg["episode_seed"],
                    reward_version="v13",
                    curriculum_stage=curriculum_stage,
                )
                # Force multi_world mode even with single world
                env._multi_world = True
                # Monkey-patch _sample_start_goal to return fixed bank values
                _fixed_start = np.array(cfg["start_pos"], dtype=np.float32)
                _fixed_goal = np.array(cfg["goal_pos"], dtype=np.float32)
                _fixed_yaw = float(cfg["start_yaw"])
                env._sample_start_goal = lambda role=None: (_fixed_start.copy(), _fixed_yaw, _fixed_goal.copy())

                result = rollout_single_episode(
                    actor, base_policy, env, device,
                    episode_seed=cfg["episode_seed"], deterministic=False,
                )
                result["episode_id"] = cfg["episode_id"]
                result["role"] = cfg["role"]
                result["world_name"] = wname
                results.append(result)
                env.close()
                # Free per-episode GPU tensors (ODE forward + synth pointcloud)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [ERROR] Episode {cfg['episode_id']} on {wname}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "episode_id": cfg["episode_id"], "role": cfg["role"],
                    "world_name": wname, "SR": 0, "CR": 0, "error": str(e),
                })

    return results


def compute_paired_switches(pre_results: dict, post_results: dict) -> dict:
    """Compute per-episode paired switches between two eval runs."""
    switches = {"success_to_failure": [], "failure_to_success": [], "unchanged_success": [],
                "unchanged_failure": [], "role_switches": defaultdict(lambda: {"s2f": 0, "f2s": 0, "us": 0, "uf": 0})}

    pre_lookup = {r["episode_id"]: r for r in pre_results}
    post_lookup = {r["episode_id"]: r for r in post_results}

    for eid in pre_lookup:
        if eid not in post_lookup:
            continue
        pre_sr = pre_lookup[eid]["SR"]
        post_sr = post_lookup[eid]["SR"]
        role = pre_lookup[eid].get("role", "unknown")

        if pre_sr and not post_sr:
            switches["success_to_failure"].append(eid)
            switches["role_switches"][role]["s2f"] += 1
        elif not pre_sr and post_sr:
            switches["failure_to_success"].append(eid)
            switches["role_switches"][role]["f2s"] += 1
        elif pre_sr and post_sr:
            switches["unchanged_success"].append(eid)
            switches["role_switches"][role]["us"] += 1
        else:
            switches["unchanged_failure"].append(eid)
            switches["role_switches"][role]["uf"] += 1

    return switches


def compute_role_summary(results: List[dict], role: str, n_eps: int) -> dict:
    """Compute per-role summary statistics."""
    role_results = [r for r in results if r.get("role") == role]
    n = len(role_results)
    sr_count = sum(r["SR"] for r in role_results)
    cr_count = sum(r["CR"] for r in role_results)
    sr = sr_count / max(n, 1)
    cr = cr_count / max(n, 1)
    ci_low, ci_high = wilson_ci(sr_count, n)

    # Fine-grained metrics
    onset_vals = [r["onset_time"] for r in role_results if r.get("onset_time") is not None]
    fc_vals = [r["future_clearance_softmin"] for r in role_results
               if r.get("future_clearance_softmin") is not None and not math.isnan(r["future_clearance_softmin"])]
    clearance_vals = [r["min_clearance"] for r in role_results
                      if r.get("min_clearance") is not None and not math.isnan(r["min_clearance"])]
    progress_vals = [r["progress_to_goal"] for r in role_results
                     if r.get("progress_to_goal") is not None and not math.isnan(r["progress_to_goal"])]
    lc_vals = [r["low_density_coverage"] for r in role_results
               if r.get("low_density_coverage") is not None]
    trap_count = sum(r.get("trap_recovery", 0) for r in role_results)

    return {
        "role": role, "n_episodes": n,
        "SR": sr, "CR": cr, "SR_count": sr_count, "CR_count": cr_count,
        "SR_Wilson_95CI": [ci_low, ci_high],
        "onset_time_mean": float(np.mean(onset_vals)) if onset_vals else None,
        "onset_time_active_frac": len(onset_vals) / max(n, 1),
        "future_clearance_softmin_mean": float(np.mean(fc_vals)) if fc_vals else None,
        "min_clearance_mean": float(np.mean(clearance_vals)) if clearance_vals else None,
        "progress_to_goal_mean": float(np.mean(progress_vals)) if progress_vals else None,
        "low_density_coverage_mean": float(np.mean(lc_vals)) if lc_vals else None,
        "trap_recovery_total": trap_count,
        "episode_length_mean": float(np.mean([r.get("episode_length", 0) for r in role_results])),
    }


def compute_paired_delta(pre_summary: dict, post_summary: dict) -> dict:
    """Compute delta between two eval summaries."""
    delta = {}
    for key in ["SR", "CR", "onset_time_mean", "future_clearance_softmin_mean",
                "min_clearance_mean", "progress_to_goal_mean", "low_density_coverage_mean",
                "trap_recovery_total", "episode_length_mean"]:
        pre_val = pre_summary.get(key)
        post_val = post_summary.get(key)
        if pre_val is not None and post_val is not None:
            delta[key] = post_val - pre_val
        else:
            delta[key] = None
    return delta


def load_actor_from_ckpt(base_policy, ckpt_path: str, device: str, lora_rank: int = 16):
    """Load LoRA actor weights from a checkpoint."""
    from scripts.train_a2_10k_capability import create_actor_wrapper
    actor = create_actor_wrapper(base_policy, base_policy, device, lora_rank=lora_rank)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
    else:
        actor.load_state_dict(ckpt, strict=False)
    actor.eval()
    return actor


def adjudicate(results: dict) -> str:
    """Apply adjudication rules and return verdict."""
    lines = ["# C2-OOC Expanded Eval Adjudication", "", f"**Timestamp:** {datetime.now().isoformat()}", ""]

    roles = ["anticipatory", "dense_recovery", "support_expansion"]
    verdicts = {}

    for role in roles:
        pre = results["per_role"][role]["pre"]
        mid = results["per_role"][role].get("mid")
        post = results["per_role"][role]["post"]
        switches = results["switches"].get(f"pre_to_post", {}).get("role_switches", {}).get(role, {})
        delta = results["deltas"].get(f"pre_to_post", {}).get(role, {})

        lines.append(f"## {role}")
        lines.append(f"")
        lines.append(f"| Metric | Pre | Mid | Post | Δ(pre→post) |")
        lines.append(f"|--------|-----|-----|------|-------------|")
        mid_sr_str = f"{mid['SR']:.3f}" if mid else "N/A"
        lines.append(f"| SR | {pre['SR']:.3f} | {mid_sr_str} | {post['SR']:.3f} | {delta.get('SR', 0):+.3f} |")
        _mid_cr = mid.get('CR', 0) if mid else 0
        _mid_cr_str = f"{_mid_cr:.3f}" if isinstance(_mid_cr, (int, float)) else "N/A"
        _delta_cr = delta.get('CR', 0)
        _delta_cr = _delta_cr if isinstance(_delta_cr, (int, float)) else 0
        lines.append(f"| CR | {pre['CR']:.3f} | {_mid_cr_str} | {post['CR']:.3f} | {_delta_cr:+.3f} |")
        lines.append(f"| SR Wilson 95% CI | [{pre['SR_Wilson_95CI'][0]:.3f}, {pre['SR_Wilson_95CI'][1]:.3f}] | — | [{post['SR_Wilson_95CI'][0]:.3f}, {post['SR_Wilson_95CI'][1]:.3f}] | — |")
        lines.append(f"")

        # Paired switches
        n = pre.get("n_episodes", 0)
        s2f = switches.get("s2f", 0)
        f2s = switches.get("f2s", 0)
        us = switches.get("us", 0)
        uf = switches.get("uf", 0)
        lines.append(f"**Paired switches:** {s2f} success→failure, {f2s} failure→success, "
                     f"{us} unchanged-success, {uf} unchanged-failure")
        lines.append(f"")

        # Fine-grained deltas
        for metric, label in [("onset_time_mean", "Onset time"),
                               ("future_clearance_softmin_mean", "Future clearance (softmin)"),
                               ("min_clearance_mean", "Min clearance"),
                               ("progress_to_goal_mean", "Progress to goal")]:
            pre_v = pre.get(metric)
            post_v = post.get(metric)
            d = delta.get(metric)
            pre_s = f"{pre_v:.3f}" if pre_v is not None else "N/A"
            post_s = f"{post_v:.3f}" if post_v is not None else "N/A"
            d_s = f"{d:+.3f}" if d is not None else "N/A"
            lines.append(f"- {label}: {pre_s} → {post_s} (Δ={d_s})")

        lines.append(f"")

        # Adjudication
        ci_overlap = (pre["SR_Wilson_95CI"][0] <= post["SR_Wilson_95CI"][1] and
                      post["SR_Wilson_95CI"][0] <= pre["SR_Wilson_95CI"][1])

        if role == "anticipatory":
            onset_delta = delta.get("onset_time_mean")
            fc_delta = delta.get("future_clearance_softmin_mean")
            if delta.get("SR", 0) < -0.05 and not ci_overlap:
                if onset_delta is not None and onset_delta > 0 and fc_delta is not None and fc_delta < 0:
                    verdicts[role] = "REAL_REGRESSION — stop at Phase 10, root-cause audit"
                else:
                    verdicts[role] = "SR_DECLINE_BUT_FINE_GRAINED_UNCLEAR — review manually"
            elif ci_overlap:
                verdicts[role] = "SMALL_SAMPLE_NOISE — CI overlap, cannot conclude regression"
            else:
                verdicts[role] = "INCONCLUSIVE — review paired switches"
        elif role == "support_expansion":
            if abs(delta.get("SR", 0)) < 0.03 and ci_overlap:
                verdicts[role] = "NO_EVIDENCE_OF_CHANGE — 80 updates insufficient for support, not 'cannot learn'"
            elif delta.get("SR", 0) > 0.05 and not ci_overlap:
                verdicts[role] = "IMPROVEMENT_DETECTED"
            else:
                verdicts[role] = "INCONCLUSIVE"
        elif role == "dense_recovery":
            if delta.get("SR", 0) > 0.05 and not ci_overlap:
                verdicts[role] = "IMPROVEMENT_DETECTED"
            elif ci_overlap:
                verdicts[role] = "DIRECTIONAL_SIGNAL_BUT_CI_OVERLAP — larger N needed"
            else:
                verdicts[role] = "INCONCLUSIVE"

        lines.append(f"**Verdict:** {verdicts[role]}")
        lines.append(f"")

    # Overall verdict
    lines.append("## Overall")
    if any("REGRESSION" in v for v in verdicts.values()):
        lines.append("**STOP — at least one role shows real regression. Root-cause audit needed.**")
    elif all("NO_EVIDENCE" in v or "IMPROVEMENT" in v or "SMALL_SAMPLE" in v for v in verdicts.values()):
        lines.append("**PROCEED — no confirmed regression. B10 single cell stability can continue evaluation.**")
    else:
        lines.append("**MIXED — review per-role verdicts manually.**")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="C2-OOC Expanded Paired Eval")
    parser.add_argument("--admission_manifest", type=str, required=True)
    parser.add_argument("--bc_ckpt", type=str, required=True)
    parser.add_argument("--mid_ckpt", type=str, default=None, help="Phase 10 actor checkpoint")
    parser.add_argument("--post_ckpt", type=str, required=True, help="Phase 20 actor checkpoint")
    parser.add_argument("--post_critic_ckpt", type=str, default=None)
    parser.add_argument("--b1_critic_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--episodes_per_role", type=int, default=32)
    parser.add_argument("--curriculum_stage", type=str, default="C2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank of the actor checkpoint (must match ckpt). "
                             "Capacity-budget matrix: rank4->4, rank16->16, rank32->32")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = args.device

    print("=" * 60)
    print("  C2-OOC Expanded Paired Eval")
    print(f"  Episodes per role: {args.episodes_per_role}")
    print(f"  Total episodes: {args.episodes_per_role * 3}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Load admission manifest
    print("\n[1/5] Loading admission manifest...")
    from scripts.train_a2_10k_capability import _make_newenv_world_pool
    manifest = load_routea_rl_admission_manifest(args.admission_manifest)
    world_dir = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds"
    val_pool = _make_newenv_world_pool(manifest, "rl_val", world_dir)

    world_meta_lookup = {}
    for w in manifest.get_admitted_worlds("rl_val"):
        wname = w.world_uid
        if wname.endswith(".world"):
            wname = wname[:-6]
        world_meta_lookup[wname] = w

    val_roles = set()
    for w in manifest.get_admitted_worlds("rl_val"):
        val_roles.add(w.role_active or w.role_validated)
    print(f"  Val pool: {len(val_pool)} worlds, roles: {val_roles}")

    # Generate fixed episode-bank
    print(f"\n[2/5] Generating fixed episode-bank ({args.episodes_per_role} eps/role)...")
    from scripts.train_a2_10k_capability import _generate_eval_episode_configs

    # Generate per-world configs using the training code's existing logic
    world_configs = _generate_eval_episode_configs(
        val_pool, world_meta_lookup, rng,
        n_configs_per_world=args.episodes_per_role,  # over-generate, then filter by role
    )

    # Build flat episode bank with role annotations
    all_configs = []
    for wname, cfgs in world_configs.items():
        meta = world_meta_lookup.get(wname)
        if meta is None:
            continue
        role = meta.role_active or meta.role_validated or "anticipatory"
        for i, cfg in enumerate(cfgs):
            cfg["role"] = role
            cfg["episode_id"] = f"{role}_{i:03d}"
            # Find val_pool index for this world
            cfg["val_pool_index"] = next((idx for idx, (w, _) in enumerate(val_pool) if w == wname), 0)
            all_configs.append(cfg)

    # Downsample to args.episodes_per_role per role, add episode_seed
    episode_bank = []
    for role in ["anticipatory", "dense_recovery", "support_expansion"]:
        role_cfgs = [c for c in all_configs if c.get("role") == role]
        if len(role_cfgs) >= args.episodes_per_role:
            selected = role_cfgs[:args.episodes_per_role]
        else:
            selected = role_cfgs
            print(f"  [WARN] Only {len(role_cfgs)} configs for {role}, wanted {args.episodes_per_role}")
        for i, cfg in enumerate(selected):
            cfg["episode_seed"] = abs(hash(f"{cfg.get('episode_id', role)}_{i}_{args.seed}")) % (2**31)
        episode_bank.extend(selected)
    print(f"  Generated {len(episode_bank)} episode configs")
    # Save bank
    with open(output_dir / "episode_bank.json", "w") as f:
        json.dump(episode_bank, f, indent=2)

    # Load BC base policy
    print(f"\n[3/5] Loading BC/FM base policy...")
    from scripts.train_a2_10k_capability import load_v9_2_policy, create_actor_wrapper
    base_policy = load_v9_2_policy(args.bc_ckpt, device)
    base_policy.eval()

    # Create actor wrappers for each checkpoint
    print(f"\n[4/5] Creating actor wrappers...")
    # Pre: zero-init LoRA (= BC behavior). Pass base_policy as teacher (unused during eval)
    actor_pre = create_actor_wrapper(base_policy, base_policy, device, lora_rank=args.lora_rank)

    # Post: load from Phase 20 checkpoint
    actor_post = load_actor_from_ckpt(base_policy, args.post_ckpt, device, lora_rank=args.lora_rank)

    # Mid: load from Phase 10 checkpoint (if provided)
    actor_mid = None
    if args.mid_ckpt and os.path.exists(args.mid_ckpt):
        actor_mid = load_actor_from_ckpt(base_policy, args.mid_ckpt, device, lora_rank=args.lora_rank)
    else:
        print("  [WARN] No mid checkpoint provided, skipping mid eval")

    # Run evals
    print(f"\n[5/5] Running expanded evals...")
    t0 = time.time()

    print(f"\n  --- Pre eval (BC baseline) ---")
    results_pre = run_eval_on_bank(
        actor_pre, base_policy, device, val_pool, world_meta_lookup,
        episode_bank, args.curriculum_stage,
    )
    # Free pre/post-actor GPU memory at transition points (critical for 8GB GPU)
    _has_mid = actor_mid is not None
    del actor_pre
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results_mid = None
    if _has_mid:
        print(f"\n  --- Mid eval (Phase 10) ---")
        results_mid = run_eval_on_bank(
            actor_mid, base_policy, device, val_pool, world_meta_lookup,
            episode_bank, args.curriculum_stage,
        )

    print(f"\n  --- Post eval (Phase 20) ---")
    if torch.cuda.is_available():
        print(f"  [GPU before post eval] used={torch.cuda.memory_allocated()/1e9:.2f}GB reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
    results_post = run_eval_on_bank(
        actor_post, base_policy, device, val_pool, world_meta_lookup,
        episode_bank, args.curriculum_stage,
    )
    # Free post-actor GPU memory
    del actor_post
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  All evals complete in {elapsed / 60:.1f} min")

    # Compute summaries
    print(f"\n[6/6] Computing summaries...")
    roles = ["anticipatory", "dense_recovery", "support_expansion"]
    per_role = {}
    for role in roles:
        per_role[role] = {
            "pre": compute_role_summary(results_pre, role, args.episodes_per_role),
        }
        if results_mid:
            per_role[role]["mid"] = compute_role_summary(results_mid, role, args.episodes_per_role)
        per_role[role]["post"] = compute_role_summary(results_post, role, args.episodes_per_role)

    # Paired switches
    switches = {}
    switches["pre_to_post"] = compute_paired_switches(results_pre, results_post)
    if results_mid:
        switches["pre_to_mid"] = compute_paired_switches(results_pre, results_mid)
        switches["mid_to_post"] = compute_paired_switches(results_mid, results_post)

    # Deltas
    deltas = {}
    for pair_name, pre_key, post_key in [("pre_to_post", "pre", "post")]:
        deltas[pair_name] = {}
        for role in roles:
            deltas[pair_name][role] = compute_paired_delta(
                per_role[role][pre_key], per_role[role][post_key]
            )

    # Compile results
    all_results = {
        "config": {
            "episodes_per_role": args.episodes_per_role,
            "total_episodes": len(episode_bank),
            "roles": roles,
            "curriculum_stage": args.curriculum_stage,
            "seed": args.seed,
        },
        "per_role": per_role,
        "switches": {
            k: {"success_to_failure": len(v["success_to_failure"]),
                "failure_to_success": len(v["failure_to_success"]),
                "unchanged_success": len(v["unchanged_success"]),
                "unchanged_failure": len(v["unchanged_failure"]),
                "role_switches": dict(v["role_switches"])}
            for k, v in switches.items()
        },
        "switches_detail": {
            k: {"success_to_failure": v["success_to_failure"],
                "failure_to_success": v["failure_to_success"]}
            for k, v in switches.items()
        },
        "deltas": deltas,
    }

    # Save
    with open(output_dir / "expanded_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Adjudication
    adjudication = adjudicate(all_results)
    with open(output_dir / "adjudication.md", "w", encoding="utf-8") as f:
        f.write(adjudication)

    # Print summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for role in roles:
        pre = per_role[role]["pre"]
        post = per_role[role]["post"]
        mid = per_role[role].get("mid")
        ci = pre["SR_Wilson_95CI"]
        ci_post = post["SR_Wilson_95CI"]
        mid_str = f" → Mid: {mid['SR']:.3f}" if mid else ""
        print(f"  {role}:")
        print(f"    Pre:  SR={pre['SR']:.3f}, CR={pre['CR']:.3f}, CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
        if mid:
            print(f"    Mid:  SR={mid['SR']:.3f}, CR={mid['CR']:.3f}")
        print(f"    Post: SR={post['SR']:.3f}, CR={post['CR']:.3f}, CI=[{ci_post[0]:.3f}, {ci_post[1]:.3f}]")
        sw = switches.get("pre_to_post", {}).get("role_switches", {}).get(role, {})
        print(f"    Paired: {sw.get('f2s', 0)} fail→success, {sw.get('s2f', 0)} success→fail, "
              f"{sw.get('us', 0)} unchanged-S, {sw.get('uf', 0)} unchanged-F")

    print(f"\n  Adjudication: {output_dir / 'adjudication.md'}")
    print(f"  Full results: {output_dir / 'expanded_eval_results.json'}")


if __name__ == "__main__":
    main()
