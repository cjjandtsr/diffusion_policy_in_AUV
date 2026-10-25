#!/usr/bin/env python3
"""audit_b0_replay_v13.py — verify a B0 collect replay's world coverage + role balance.

Loads a RouteAReplayBuffer .pt and tallies, per role_active:
  - distinct world_uid visited (coverage)
  - distinct (world_uid, episode_seed) episodes
  - transition counts
Compares distinct-world coverage against the admission manifest's rl_train buckets.

Usage: python scripts/audit_b0_replay_v13.py --replay <replay.pt> --manifest <manifest.json> [--out <audit.json>]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diffusion_policy_3d.rl.routea_replay_buffer import RouteAReplayBuffer

FORMAL_ROLES = ["anticipatory", "dense_recovery", "support_expansion"]


def expected_worlds_from_manifest(manifest_path, split="rl_train"):
    with open(manifest_path) as f:
        m = json.load(f)
    exp = defaultdict(set)
    for e in m.get("results", []):
        if e.get("split") != split:
            continue
        role = e.get("role_active") or e.get("role_validated") or e.get("role_generated")
        wname = e["world_uid"]
        if wname.endswith(".world"):
            wname = wname[:-6]
        exp[role].add(wname)
    return {r: sorted(v) for r, v in exp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min_eps_per_role", type=int, default=190)
    args = ap.parse_args()

    replay = RouteAReplayBuffer.load(args.replay)
    trans = replay.get_all_transitions()

    worlds_by_role = defaultdict(set)
    eps_by_role = defaultdict(set)
    trans_by_role = defaultdict(int)
    for t in trans:
        role = getattr(t, "role_active", "") or getattr(t, "role_validated", "") or "unknown"
        wuid = getattr(t, "world_uid", "")
        seed = getattr(t, "episode_seed", -1)
        worlds_by_role[role].add(wuid)
        eps_by_role[role].add((wuid, seed))
        trans_by_role[role] += 1

    expected = expected_worlds_from_manifest(args.manifest)

    per_role = {}
    coverage_ok = True
    balance_ok = True
    for role in FORMAL_ROLES:
        got = sorted(w for w in worlds_by_role.get(role, set()) if w)
        exp = expected.get(role, [])
        n_eps = len(eps_by_role.get(role, set()))
        role_cov_ok = (len(got) == len(exp)) and len(exp) > 0
        role_bal_ok = n_eps >= args.min_eps_per_role
        coverage_ok = coverage_ok and role_cov_ok
        balance_ok = balance_ok and role_bal_ok
        per_role[role] = {
            "worlds_visited": len(got),
            "worlds_expected": len(exp),
            "coverage_ok": role_cov_ok,
            "missing_worlds": sorted(set(exp) - set(got)),
            "n_episodes": n_eps,
            "n_transitions": trans_by_role.get(role, 0),
            "balance_ok": role_bal_ok,
        }

    report = {
        "replay": args.replay,
        "n_transitions_total": len(trans),
        "per_role": per_role,
        "coverage_ok": coverage_ok,
        "balance_ok": balance_ok,
        "all_pass": coverage_ok and balance_ok,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = args.out or str(Path(args.replay).parent / "b0_replay_audit_v13.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
    print(f"VERDICT: {'PASS' if report['all_pass'] else 'FAIL'}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
