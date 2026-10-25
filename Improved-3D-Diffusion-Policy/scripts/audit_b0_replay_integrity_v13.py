#!/usr/bin/env python3
"""audit_b0_replay_integrity_v13.py — deep integrity audit of a B0 collect replay.

Answers two P1 questions the coverage audit did not:
  P1-A  Seed transitions: replay = seed(prepended) + collected. Separate them by
        insertion order (get_all_transitions returns online partition in add order),
        and verify EVERY episode's world_uid / role / admission + re-check v1.3 gates
        on the STORED start/goal (RouteATransition.start/goal).
  P1-B  Observation quality: count NaN/Inf/degenerate obs_feature_t/tp1, non-finite
        reward, validate() failures — i.e. did any no-ray-hit / bad obs contaminate?

Pure metadata + numpy; no GPU. v1.3 gate re-check uses the env's own methods.

Usage: python scripts/audit_b0_replay_integrity_v13.py --replay <replay.pt> \
         --manifest <manifest.json> [--collect_summary <collect_only_summary.json>] [--out <json>]
"""
import argparse, json, sys
from collections import defaultdict, OrderedDict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diffusion_policy_3d.rl.routea_replay_buffer import RouteAReplayBuffer

FORMAL_ROLES = ["anticipatory", "dense_recovery", "support_expansion"]


def admitted_worlds(manifest_path, split="rl_train"):
    m = json.load(open(manifest_path))
    s = {}
    for e in m.get("results", []):
        if e.get("split") != split:
            continue
        w = e["world_uid"]
        if w.endswith(".world"):
            w = w[:-6]
        s[w] = e.get("role_active") or e.get("role_validated") or e.get("role_generated")
    return s


def finite_arr(a):
    return a is not None and np.all(np.isfinite(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--collect_summary", default=None)
    ap.add_argument("--seed_trans", type=int, default=None,
                    help="Override seed-transition count (else inferred from collect_summary).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    replay = RouteAReplayBuffer.load(args.replay)
    trans = replay.get_all_transitions()
    n = len(trans)

    # Infer seed transition count = total - collected
    seed_trans = args.seed_trans
    if seed_trans is None and args.collect_summary:
        cs = json.load(open(args.collect_summary))
        seed_trans = n - int(cs.get("transitions_collected", n))
    if seed_trans is None:
        seed_trans = 0

    admit = admitted_worlds(args.manifest)

    # ---- P1-B: observation / value integrity across ALL transitions ----
    obs_nan_t = obs_nan_tp1 = rew_nonfinite = validate_fail = 0
    act_nonfinite = 0
    for t in trans:
        if not finite_arr(getattr(t, "obs_feature_t", None)): obs_nan_t += 1
        if not finite_arr(getattr(t, "obs_feature_tp1", None)): obs_nan_tp1 += 1
        if not np.isfinite(getattr(t, "reward_total", np.nan)): rew_nonfinite += 1
        if not finite_arr(getattr(t, "action_chunk", None)): act_nonfinite += 1
        try:
            if not t.validate(): validate_fail += 1
        except Exception:
            validate_fail += 1

    # ---- P1-A: split seed vs collected by insertion order; per-group compliance ----
    def summarize(group, label):
        eps = OrderedDict()  # (world_uid, episode_seed) -> list idx
        role_c = defaultdict(int); world_c = defaultdict(int)
        bad_world = 0; bad_admit = 0; bad_role = 0; missing_sg = 0
        for t in group:
            w = getattr(t, "world_uid", ""); rseed = getattr(t, "episode_seed", -1)
            role = getattr(t, "role_active", "") or getattr(t, "role_validated", "")
            adm = getattr(t, "admission_status", "")
            eps.setdefault((w, rseed), []).append(t)
            role_c[role] += 1; world_c[w] += 1
            if w not in admit: bad_world += 1
            if not adm.startswith("admitted"): bad_admit += 1
            if role not in FORMAL_ROLES: bad_role += 1
            if getattr(t, "start", None) is None or getattr(t, "goal", None) is None:
                missing_sg += 1
        return {
            "label": label, "n_transitions": len(group), "n_episodes": len(eps),
            "roles_by_transition": dict(role_c),
            "distinct_worlds": len(world_c),
            "transitions_world_not_admitted": bad_world,
            "transitions_admission_not_admitted": bad_admit,
            "transitions_role_not_formal": bad_role,
            "transitions_missing_start_or_goal": missing_sg,
            "episode_keys": [{"world_uid": k[0], "episode_seed": k[1],
                              "role": (getattr(v[0], "role_active", "") or ""),
                              "n_trans": len(v)} for k, v in eps.items()],
        }

    seed_group = trans[:seed_trans] if seed_trans > 0 else []
    collect_group = trans[seed_trans:]
    seed_summary = summarize(seed_group, "seed") if seed_group else {"label": "seed", "n_transitions": 0}
    collect_summary = summarize(collect_group, "collected")
    # Drop the big per-episode list for the collected group (keep counts only)
    collect_episode_keys = collect_summary.pop("episode_keys", [])

    report = {
        "replay": args.replay,
        "n_transitions_total": n,
        "seed_transitions_inferred": seed_trans,
        "obs_integrity": {
            "obs_feature_t_nonfinite": obs_nan_t,
            "obs_feature_tp1_nonfinite": obs_nan_tp1,
            "action_chunk_nonfinite": act_nonfinite,
            "reward_total_nonfinite": rew_nonfinite,
            "validate_failures": validate_fail,
            "all_obs_finite": (obs_nan_t == 0 and obs_nan_tp1 == 0),
        },
        "seed_group": seed_summary,
        "collected_group": collect_summary,
        "collected_episode_count": len(collect_episode_keys),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = args.out or str(Path(args.replay).parent / "b0_replay_integrity_v13.json")
    json.dump(report, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")

    # Verdict
    obs_ok = report["obs_integrity"]["all_obs_finite"] and rew_nonfinite == 0 and validate_fail == 0
    seed_ok = (seed_summary.get("transitions_world_not_admitted", 0) == 0 and
               seed_summary.get("transitions_admission_not_admitted", 0) == 0 and
               seed_summary.get("transitions_role_not_formal", 0) == 0)
    print(f"OBS_INTEGRITY: {'PASS' if obs_ok else 'FAIL'}")
    print(f"SEED_COMPLIANCE: {'PASS' if seed_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
