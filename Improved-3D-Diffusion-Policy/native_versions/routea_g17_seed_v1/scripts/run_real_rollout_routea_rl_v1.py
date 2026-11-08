#!/usr/bin/env python3
"""
Route A RL v1 — Real Rollout Runner (Two-Stage, Incremental Flush)
===================================================================
Stage 1: Small dry-run → gate check
Stage 2: Full 1080-episode paired rollout (gated by dry-run PASS)

Key features:
- Per-world incremental flush (JSONL + CSV) — no data loss on crash
- Progress logging every world
- Resumable: skips already-completed episodes if output files exist
"""

import argparse, csv, json, math, os, resource, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

ROOT = Path("/home/cjj/AUV")
PROJ = ROOT / "Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
PREP_DIR = ROOT / "program/shiyan/results_routeA_RL_newenv_real_eval_prep_20260704"
CKPT_DIR = ROOT / "program/shiyan/results_routeA_RL_newenv_A2_checkpoint_persist_20260704"
MANIFEST_PATH = ROOT / "program/reports/routeA_rl_env_v1/calibration/admission_manifest.json"
PX4_WORLDS_DIR = ROOT / "PX4-Autopilot/Tools/sitl_gazebo/worlds"
FORMAL_ROLES = ["anticipatory", "dense_recovery", "support_expansion"]

sys.path.insert(0, str(PROJ / "scripts"))
from train_a2_10k_capability import (
    load_v9_2_policy, create_actor_wrapper,
    _make_newenv_world_pool, _rollout_bc_fm_episode, rollout_episode_a2,
    KinematicUAVEnv, SynthPointCloud, REWARD_TERM_NAMES,
)
from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
    load_routea_rl_admission_manifest, FORMAL_CAPABILITY_ROLES,
)


def _wilson_ci(success, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = success / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / d
    return (p, max(0, c-m), min(1, c+m))


def _bootstrap_ci(values, B=2000):
    arr = np.array([v for v in values if np.isfinite(v)])
    if len(arr) < 2:
        m = float(np.mean(arr)) if len(arr) > 0 else 0.0
        return (m, m, m)
    rng = np.random.default_rng(42)
    boots = sorted(float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(B))
    return (float(np.mean(arr)), boots[int(0.025*B)], boots[int(0.975*B)])


def _now():
    return datetime.now().strftime("%H:%M:%S")


def _write_jsonl_append(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, float)):
                    clean[k] = float(v) if np.isfinite(v) else None
                elif isinstance(v, (np.bool_, bool)):
                    clean[k] = bool(v)
                elif isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                else:
                    clean[k] = v
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def _write_csv_append(path, rows, fieldnames):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)


CSV_FIELDS = ["episode_id", "world_uid", "role_active", "policy_name", "episode_seed",
              "success", "collision", "done_reason", "n_steps", "reward_total",
              "min_clearance", "onset_active_steps", "trap_active_steps",
              "trap_recovered", "low_density_entries", "split"]


def run_one_episode(policy_name, policy_obj, base_policy, env, device,
                    ep_seed, wname, role_active, split):
    """Run one episode and collect per-step metrics."""
    if policy_name == "frozen_bc_fm":
        transitions, summary = _rollout_bc_fm_episode(
            base_policy, env, device, ep_seed, wname, role_active, role_active, split)
    else:  # routeA_rl_a2
        transitions, _, summary = rollout_episode_a2(
            policy_obj, base_policy, env, device,
            episode_seed=ep_seed, world_id=-1, deterministic=True)
        for t in transitions:
            t.world_uid = wname; t.split = split
            t.role_validated = role_active; t.role_active = role_active

    records = []
    for step_i, t in enumerate(transitions):
        rm = t.reward_metrics or {}
        done = t.done
        records.append({
            "episode_id": f"{policy_name}_{wname}_s{ep_seed}",
            "timestep": step_i, "world_uid": wname, "split": split,
            "role_active": role_active, "policy_name": policy_name,
            "episode_seed": ep_seed,
            "pos_world_x": 0.0, "pos_world_y": 0.0, "pos_world_z": 0.0,
            "collision_flag": bool("collision" in (t.done_reason or "")),
            "goal_distance": -1.0,
            "success_flag": bool(summary.get("reached", False)),
            "done": done, "done_reason": t.done_reason or "",
            "reward_total": float(sum(rm.get(k, 0.0) for k in REWARD_TERM_NAMES)) if isinstance(rm, dict) else 0.0,
            "d_soft_h": float(rm.get("d_soft_h", -1.0)) if isinstance(rm, dict) else -1.0,
            "onset_active": bool(rm.get("onset_active", False)) if isinstance(rm, dict) else False,
            "trap_active": bool(rm.get("trap_active", False)) if isinstance(rm, dict) else False,
            "trap_recovered": bool(rm.get("trap_recovered", False)) if isinstance(rm, dict) else False,
            "low_density_cell": bool(rm.get("low_density_cell", False)) if isinstance(rm, dict) else False,
            "memory_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        })

    return records, {
        "episode_id": f"{policy_name}_{wname}_s{ep_seed}",
        "world_uid": wname, "role_active": role_active, "policy_name": policy_name,
        "episode_seed": ep_seed, "split": split,
        "success": summary.get("reached", False),
        "collision": summary.get("collided", False),
        "done_reason": summary.get("done_reason", ""),
        "n_steps": len(transitions),
        "reward_total": round(summary.get("reward_total", 0), 4),
        "min_clearance": None,
        "onset_active_steps": sum(1 for r in records if r["onset_active"]),
        "trap_active_steps": sum(1 for r in records if r["trap_active"]),
        "trap_recovered": any(r["trap_recovered"] for r in records),
        "low_density_entries": sum(1 for r in records if r["low_density_cell"]),
    }


def run_stage(prefix, out_dir, world_items, policies, n_eps_per, device, rng,
              actor, base_policy, append_mode=False, seed_offset=0,
              existing_csv_path=None):
    """Run a stage with incremental per-world flushing.

    Args:
        n_eps_per: Default episodes per policy. If None, reads from
                   item["episode_count_per_policy"] (plan-driven).
                   If int, uses that count for all items.
        append_mode: If True, don't wipe existing files; append only.
        seed_offset: Starting episode index offset (for resume: skip first N).
        existing_csv_path: If set, skip episodes already in this CSV (resume dedup).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{prefix}_episode_records.jsonl"
    csv_path = out_dir / f"{prefix}_episode_summary.csv"
    if not append_mode:
        if jsonl_path.exists(): jsonl_path.unlink()
        if csv_path.exists(): csv_path.unlink()

    # --- Resume dedup: load already-completed episode IDs ---
    completed_ids = set()
    if existing_csv_path and os.path.exists(existing_csv_path):
        with open(existing_csv_path, "r") as f:
            for row in csv.DictReader(f):
                completed_ids.add(row.get("episode_id", ""))

    # --- Compute per-world n_eps_per ---
    def _get_n_eps(item):
        if n_eps_per is not None:
            return n_eps_per
        return item.get("episode_count_per_policy", 20)

    # Compute total
    n_total = sum(len(policies) * _get_n_eps(item) for item in world_items)
    if seed_offset > 0:
        n_total = sum(len(policies) * max(0, _get_n_eps(item) - seed_offset)
                      for item in world_items)

    all_summaries = []
    n_done = 0; n_skipped = 0
    t_start = time.time()

    for wi, item in enumerate(world_items):
        wname = item["world_uid"]
        role = item.get("role_active") or item.get("role_validated", "")
        split = item.get("split", "")
        _n_eps = _get_n_eps(item)
        n_to_run = max(0, _n_eps - seed_offset)

        obs_json = str(PX4_WORLDS_DIR / f"{wname}.world.obstacles.json")
        if not os.path.exists(obs_json):
            print(f"  [{_now()}] SKIP {wname}: no obstacles.json")
            n_done += len(policies) * n_to_run
            continue

        try:
            env = KinematicUAVEnv(
                world_id=4, world_pool=[(wname, obs_json)],
                synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                rl_action_steps=8, h_min=0.3, z_min=1.0, z_max=5.0,
                disturb_enable=False, mid_state_init_ratio=0.0,
                seed=int(rng.integers(0, 2**31)), reward_version="v13",
            )
        except Exception as e:
            print(f"  [{_now()}] ERR env {wname}: {e}")
            n_done += len(policies) * n_to_run
            continue

        seeds = item.get("episode_seed_list", [n_done])[:_n_eps]
        world_records = []; world_ep_summaries = []

        for pi, policy_name in enumerate(policies):
            for ei in range(seed_offset, _n_eps):
                ep_seed = seeds[ei % len(seeds)] + ei * 100 + pi * 1000
                ep_id = f"{policy_name}_{wname}_s{ep_seed}"
                if ep_id in completed_ids:
                    n_skipped += 1; n_done += 1
                    continue
                try:
                    recs, ep_sum = run_one_episode(
                        policy_name, actor, base_policy, env, device,
                        ep_seed, wname, role, split)
                    world_records.extend(recs)
                    world_ep_summaries.append(ep_sum)
                    all_summaries.append(ep_sum)
                except Exception as e:
                    print(f"  [{_now()}] ERR {policy_name}/{wname}/s{ep_seed}: {e}")
                n_done += 1

        env.close()

        # Incremental flush per world
        if world_records:
            _write_jsonl_append(jsonl_path, world_records)
            _write_csv_append(csv_path, world_ep_summaries, CSV_FIELDS)

        elapsed = time.time() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (n_total - n_done) / rate if rate > 0 else 0
        print(f"  [{_now()}] {prefix} [{wi+1}/{len(world_items)}] {role}/{wname[-20:]} "
              f"n={len(world_ep_summaries)}ep total={n_done}/{n_total} "
              f"skipped={n_skipped} rate={rate:.1f}ep/s ETA={eta/60:.0f}min")

    return all_summaries, 0, 0  # summaries, AU, CU


def compute_role_metrics(summaries, role):
    eps = [s for s in summaries if s.get("role_active") == role]
    n = len(eps)
    n_s = sum(1 for s in eps if s.get("success"))
    n_c = sum(1 for s in eps if s.get("collision"))
    sr, sr_l, sr_h = _wilson_ci(n_s, n)
    cr, cr_l, cr_h = _wilson_ci(n_c, n)
    return {"role": role, "n_episodes": n,
            "success_rate": [sr_l, sr, sr_h],
            "collision_rate": [cr_l, cr, cr_h]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["dryrun","full","both"], default="both")
    p.add_argument("--dryrun_eps", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=20260704)
    p.add_argument("--resume", action="store_true",
                   help="Resume: run missing test-world episodes only (seed_offset=20), then merge")
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    with open(PREP_DIR / "checkpoint_resolution.json") as f: ckpt = json.load(f)
    assert ckpt["status"] == "resolved" and not ckpt["blocked"]
    with open(PREP_DIR / "real_eval_world_plan.json") as f: plan = json.load(f)
    manifest = load_routea_rl_admission_manifest(str(MANIFEST_PATH))

    # Load policies
    print(f"[{_now()}] Loading policies...")
    base = load_v9_2_policy(ckpt["base_bc_fm_checkpoint"], device)
    base.eval()
    teacher = load_v9_2_policy(ckpt["base_bc_fm_checkpoint"], device)
    teacher.eval(); teacher.requires_grad_(False)
    actor = create_actor_wrapper(base, teacher, device, lora_rank=4)
    a2_ckpt = torch.load(ckpt["routea_rl_a2_checkpoint"], map_location=device)
    actor_sd = a2_ckpt["actor_state_dict"]
    for name, param in actor.named_parameters():
        if param.requires_grad and name in actor_sd:
            param.data.copy_(actor_sd[name].to(device))
    actor.eval()
    print(f"[{_now()}] Policies loaded. {len(manifest.worlds)} worlds in manifest")

    # Stage 1: Dry-run (skip if resume mode)
    dryrun_dir = Path("/home/cjj/AUV/program/shiyan/results_routeA_RL_newenv_real_rollout_dryrun_20260704")
    if not args.resume and args.stage in ("dryrun", "both"):
        dryrun_items = []
        for role in FORMAL_ROLES:
            items = [it for it in plan["eval_items"] if it["role_active"] == role]
            if items: dryrun_items.append(items[0])
        yaw = [it for it in plan["eval_items"] if it["role_active"] == "yaw_lateral_recovery"]
        if yaw: dryrun_items.append(yaw[0])

        print(f"\n{'='*60}\n  Stage 1: Dry-Run ({len(dryrun_items)}w x 2p x {args.dryrun_eps}ep)\n{'='*60}")
        sums, au, cu = run_stage("dryrun", dryrun_dir, dryrun_items,
                                 ["frozen_bc_fm", "routeA_rl_a2"], args.dryrun_eps, device, rng,
                                 actor, base)

        role_metrics = {r: compute_role_metrics(sums, r) for r in FORMAL_ROLES}
        nan = sum(1 for s in sums if not np.isfinite(s.get("reward_total", 0)))
        formal_ok = all(any(s.get("role_active") == r for s in sums) for r in FORMAL_ROLES)
        pol_ok = all(any(s.get("policy_name") == p for s in sums) for p in ["frozen_bc_fm", "routeA_rl_a2"])

        gate = all([len(sums) > 0, formal_ok, pol_ok, au == 0, cu == 0, nan == 0,
                    all(role_metrics[r]["n_episodes"] > 0 for r in FORMAL_ROLES)])
        print(f"  Dry-run GATE: {'PASS' if gate else 'FAIL'} | eps={len(sums)} roles={formal_ok} pols={pol_ok} NaN={nan}")

        with open(dryrun_dir / "dryrun_gate_report.json", "w") as f:
            json.dump({"gate_pass": gate, "next": "full_real_rollout" if gate else "blocked",
                       "episodes": len(sums), "timestamp": datetime.now().isoformat()}, f, indent=2)
        if not gate:
            print("[BLOCKED] Dry-run failed"); return 1

    # Stage 2: Full rollout (skip if resume mode)
    full_dir = Path("/home/cjj/AUV/program/shiyan/results_routeA_RL_newenv_real_rollout_full_20260704")
    if not args.resume and args.stage in ("full", "both"):
        print(f"\n{'='*60}\n  Stage 2: Full Rollout ({len(plan['eval_items'])}w x {len(plan['policies'])}p)\n{'='*60}")
        t0 = time.time()
        sums, au, cu = run_stage("full", full_dir, plan["eval_items"],
                                 plan["policies"], None, device, rng,
                                 actor, base)
        elapsed = time.time() - t0

        # Per-role metrics
        role_metrics = {}
        for role in FORMAL_ROLES + ["yaw_lateral_recovery"]:
            role_metrics[role] = compute_role_metrics(sums, role)

        # Paired comparison
        pairs = {}
        for role in FORMAL_ROLES:
            bc = [s for s in sums if s["role_active"] == role and s["policy_name"] == "frozen_bc_fm"]
            rl = [s for s in sums if s["role_active"] == role and s["policy_name"] == "routeA_rl_a2"]
            bsr = sum(1 for s in bc if s["success"]) / max(len(bc), 1)
            rsr = sum(1 for s in rl if s["success"]) / max(len(rl), 1)
            bcr = sum(1 for s in bc if s["collision"]) / max(len(bc), 1)
            rcr = sum(1 for s in rl if s["collision"]) / max(len(rl), 1)
            pairs[role] = {"BC_FM_SR": bsr, "RL_A2_SR": rsr, "delta_SR": rsr - bsr,
                           "BC_FM_CR": bcr, "RL_A2_CR": rcr, "delta_CR": rcr - bcr,
                           "n_bc": len(bc), "n_rl": len(rl)}

        summary = {
            "status": "complete" if len(sums) >= 1000 else "partial",
            "total_episodes": len(sums),
            "elapsed_min": round(elapsed / 60, 1),
            "nan": sum(1 for s in sums if not np.isfinite(s.get("reward_total", 0))),
            "actor_updates": au, "critic_updates": cu,
            "role_metrics": role_metrics,
            "paired_comparison": pairs,
            "claim_allowed": False,
            "timestamp": datetime.now().isoformat(),
        }
        with open(full_dir / "full_real_rollout_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Summary MD
        lines = ["# Full Real Rollout Summary", "",
                 f"**Episodes**: {len(sums)} | **Elapsed**: {elapsed/60:.0f}min",
                 f"**NaN**: {summary['nan']} | **AU/CU**: {au}/{cu}", "",
                 "## Paired Comparison", ""]
        for role, p in pairs.items():
            lines.append(f"### {role}")
            lines.append(f"- BC/FM: SR={p['BC_FM_SR']:.3f} CR={p['BC_FM_CR']:.3f} (n={p['n_bc']})")
            lines.append(f"- RL A2: SR={p['RL_A2_SR']:.3f} CR={p['RL_A2_CR']:.3f} (n={p['n_rl']})")
            lines.append(f"- dSR={p['delta_SR']:+.3f} dCR={p['delta_CR']:+.3f}")
            lines.append("")
        with open(full_dir / "full_real_rollout_summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Manifest
        mf = {}
        for fn in os.listdir(full_dir):
            mf[fn] = os.path.getsize(full_dir / fn)
        with open(full_dir / "artifact_manifest.json", "w") as f:
            json.dump({"files": mf, "total_episodes": len(sums)}, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Full rollout: {len(sums)} episodes in {elapsed/60:.0f}min")
        for role in FORMAL_ROLES:
            p = pairs.get(role, {})
            print(f"  {role}: dSR={p.get('delta_SR', 0):+.3f} dCR={p.get('delta_CR', 0):+.3f}")
        print(f"  Output: {full_dir}")
        print(f"{'='*60}")

    # --- Resume mode: run missing test-world episodes, then merge ---
    if args.resume:
        resume_dir = Path("/home/cjj/AUV/program/shiyan/"
                          "results_routeA_RL_newenv_real_rollout_resume_120_20260704")
        full_dir = Path("/home/cjj/AUV/program/shiyan/"
                        "results_routeA_RL_newenv_real_rollout_full_20260704")

        # Identify test worlds and missing counts
        test_items = [it for it in plan["eval_items"] if it.get("split") == "rl_test"]
        expected_test = sum(it.get("episode_count_per_policy", 30) for it in test_items)
        print(f"\n{'='*60}")
        print(f"  Resume: {len(test_items)} test worlds, "
              f"expected {expected_test}/policy, running extra with seed_offset=20")
        print(f"{'='*60}")

        # Run missing episodes (seed_offset=20 — skip already-completed 20/policy)
        t0 = time.time()
        sums_new, au, cu = run_stage(
            "resume_120", resume_dir, test_items,
            plan["policies"], None, device, rng,
            actor, base, append_mode=True, seed_offset=20,
            existing_csv_path=str(full_dir / "full_episode_summary.csv"))
        elapsed_resume = time.time() - t0
        print(f"\n  Resume done: {len(sums_new)} new episodes in {elapsed_resume/60:.0f}min")

        # --- Merge: 960 existing + 120 new → 1080 ---
        print(f"\n  Merging 960 + {len(sums_new)} → 1080...")

        # Merge CSV
        merged_csv = full_dir / "full_episode_summary.csv"
        resume_csv = resume_dir / "resume_120_episode_summary.csv"
        with open(merged_csv, "r") as f:
            existing_csv_lines = f.read().rstrip().split("\n")
        header = existing_csv_lines[0]
        with open(merged_csv, "w") as f:
            f.write(header + "\n")
            f.write("\n".join(existing_csv_lines[1:]) + "\n")
            if resume_csv.exists():
                with open(resume_csv, "r") as rf:
                    resume_lines = rf.read().rstrip().split("\n")
                    if resume_lines[0] == header:
                        f.write("\n".join(resume_lines[1:]) + "\n")
                    else:
                        f.write("\n".join(resume_lines) + "\n")

        # Merge JSONL
        merged_jsonl = full_dir / "full_episode_records.jsonl"
        resume_jsonl = resume_dir / "resume_120_episode_records.jsonl"
        with open(merged_jsonl, "a") as f:
            if resume_jsonl.exists():
                with open(resume_jsonl, "r") as rf:
                    f.write(rf.read())

        # Load merged summaries for recomputation
        all_sums = []
        for row in csv.DictReader(open(merged_csv, "r")):
            row["success"] = row.get("success", "False") in ("True", "true", "1")
            row["collision"] = row.get("collision", "False") in ("True", "true", "1")
            try: row["reward_total"] = float(row.get("reward_total", 0))
            except: row["reward_total"] = 0.0
            all_sums.append(row)

        print(f"  Merged: {len(all_sums)} total episodes")

        # --- Regenerate full summary ---
        role_metrics = {}
        for role in FORMAL_ROLES + ["yaw_lateral_recovery"]:
            role_metrics[role] = compute_role_metrics(all_sums, role)

        pairs = {}
        for role in FORMAL_ROLES:
            bc = [s for s in all_sums if s["role_active"] == role and s["policy_name"] == "frozen_bc_fm"]
            rl = [s for s in all_sums if s["role_active"] == role and s["policy_name"] == "routeA_rl_a2"]
            bsr = sum(1 for s in bc if s["success"]) / max(len(bc), 1)
            rsr = sum(1 for s in rl if s["success"]) / max(len(rl), 1)
            bcr = sum(1 for s in bc if s["collision"]) / max(len(bc), 1)
            rcr = sum(1 for s in rl if s["collision"]) / max(len(rl), 1)
            pairs[role] = {"BC_FM_SR": bsr, "RL_A2_SR": rsr, "delta_SR": rsr - bsr,
                           "BC_FM_CR": bcr, "RL_A2_CR": rcr, "delta_CR": rcr - bcr,
                           "n_bc": len(bc), "n_rl": len(rl)}

        n_correct = sum(1 for it in test_items
                        for pn in plan["policies"]
                        if sum(1 for s in all_sums
                               if s["world_uid"] == it["world_uid"] and s["policy_name"] == pn)
                        == it.get("episode_count_per_policy", 30))

        summary = {
            "status": "complete" if len(all_sums) >= 1080 else "partial",
            "total_episodes": len(all_sums),
            "elapsed_min": round((elapsed_resume) / 60, 1),
            "nan": sum(1 for s in all_sums if not np.isfinite(float(s.get("reward_total", 0)))),
            "actor_updates": 0, "critic_updates": 0,
            "resume_episodes_added": len(sums_new),
            "test_world_ep_per_policy_ok": n_correct == len(test_items) * len(plan["policies"]),
            "role_metrics": role_metrics,
            "paired_comparison": pairs,
            "claim_allowed": False,
            "timestamp": datetime.now().isoformat(),
        }
        with open(full_dir / "full_real_rollout_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Summary MD
        lines = ["# Full Real Rollout Summary (1080)",
                 f"**Episodes**: {len(all_sums)} | **Resume added**: {len(sums_new)}",
                 f"**NaN**: {summary['nan']} | **Test ep/policy OK**: {summary['test_world_ep_per_policy_ok']}",
                 "", "## Paired Comparison", ""]
        for role, p in pairs.items():
            lines.append(f"### {role}")
            lines.append(f"- BC/FM: SR={p['BC_FM_SR']:.3f} CR={p['BC_FM_CR']:.3f} (n={p['n_bc']})")
            lines.append(f"- RL A2: SR={p['RL_A2_SR']:.3f} CR={p['RL_A2_CR']:.3f} (n={p['n_rl']})")
            lines.append(f"- dSR={p['delta_SR']:+.3f} dCR={p['delta_CR']:+.3f}")
            lines.append("")
        lines.append("## Per-Role CI (Wilson 95%)")
        lines.append("")
        for role in FORMAL_ROLES + ["yaw_lateral_recovery"]:
            rm = role_metrics.get(role, {})
            sr = rm.get("success_rate", [0, 0, 0])
            cr = rm.get("collision_rate", [0, 0, 0])
            lines.append(f"- **{role}**: SR={sr[1]:.3f} [{sr[0]:.3f}, {sr[2]:.3f}]  "
                         f"CR={cr[1]:.3f} [{cr[0]:.3f}, {cr[2]:.3f}]  n={rm.get('n_episodes', 0)}")
        with open(full_dir / "full_real_rollout_summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Updated manifest
        mf = {}
        for fn in os.listdir(full_dir):
            mf[fn] = os.path.getsize(full_dir / fn)
        with open(full_dir / "artifact_manifest.json", "w") as f:
            json.dump({"files": mf, "total_episodes": len(all_sums),
                       "resume_added": len(sums_new)}, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Merged summary: {len(all_sums)} episodes")
        for role in FORMAL_ROLES:
            p = pairs.get(role, {})
            print(f"  {role}: dSR={p.get('delta_SR', 0):+.3f} dCR={p.get('delta_CR', 0):+.3f}")
        print(f"  Output: {full_dir}")
        print(f"{'='*60}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
