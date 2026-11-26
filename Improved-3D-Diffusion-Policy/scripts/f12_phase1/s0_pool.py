#!/usr/bin/env python3
"""S0 task pool: training-domain blocked/passable fixed start-goal tickets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.f12_phase1 import f12_common as C


def task_id(payload):
    body = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(body).hexdigest()


def ticket(world, role, start, yaw, goal, clearance, kind, z_range, idx):
    payload = C.make_payload(world['world_id'], role, start, yaw, goal,
                             episode_id=f'f12s0-{kind}-{idx:04d}', z_range=z_range)
    return dict(task_id=task_id(payload), world_id=world['world_id'], role=role,
                kind=kind, straight_clearance_m=float(clearance), payload=payload,
                repeats=[dict(repeat=r, reset_seed=C.episode_seed(f'{task_id(payload)}:{r}'),
                              action_seeds=[C.action_seed(task_id(payload), r, d) for d in range(60)])
                         for r in range(1, 6)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(C.OUT / 's0' / 's0_task_pool.json'))
    ap.add_argument('--blocked', type=int, default=120)
    ap.add_argument('--passable', type=int, default=120)
    ap.add_argument('--draws-per-world', type=int, default=80)
    args = ap.parse_args()
    system, config, header, _ = C.load_system()
    bank = C.bank_worlds()
    ex = C.excluded_triples()
    worlds = list(bank['worlds'].values())
    rng = np.random.default_rng(20261126)
    blocked, passable = [], []
    seen = set()
    excluded_hits = 0
    # Seed from the existing 85 training tickets first.
    for t in bank['tasks']:
        w = bank['worlds'][t['world_id']]
        p = t['payload']
        from diffusion_policy_3d.env.collision_geom import load_obstacles
        obs = load_obstacles(w['obstacle_path'])
        cl = C.straight_clearance(p['start_pos'], p['goal_pos'], obs)
        kind = 'blocked' if cl < C.BLOCKED_CLEARANCE_M else 'passable'
        if C.is_excluded(w['world_id'], p['start_pos'], p['goal_pos'], ex):
            continue
        key = (w['world_id'], tuple(np.round(p['start_pos'], 6)), tuple(np.round(p['goal_pos'], 6)))
        if key in seen:
            continue
        seen.add(key)
        row = ticket(w, t['role'], p['start_pos'], p['start_yaw'], p['goal_pos'], cl, kind, p['z_range'],
                     len(blocked) + len(passable))
        (blocked if kind == 'blocked' else passable).append(row)
    print('seed tickets', len(blocked), len(passable), flush=True)
    # Extra draws, balanced across all training worlds.
    need_blocked = max(0, args.blocked - len(blocked))
    need_passable = max(0, args.passable - len(passable))
    per_world_blocked = int(np.ceil(need_blocked / max(1, len(worlds))))
    per_world_passable = int(np.ceil(need_passable / max(1, len(worlds))))
    for wi, w in enumerate(worlds):
        ad = C.adapter_for(system, config, w, w['role'])
        env = ad.environment
        env._rng = np.random.default_rng(20261126 + wi)
        added_b = added_p = 0
        try:
            for _ in range(args.draws_per_world):
                if added_b >= per_world_blocked and added_p >= per_world_passable:
                    break
                start, yaw, goal = env._draw_start_goal_candidate(w['role'])
                reasons = env._candidate_rejection_reasons(start, goal, w['role'])
                if reasons:
                    continue
                from diffusion_policy_3d.env.collision_geom import load_obstacles
                obs = load_obstacles(w['obstacle_path'])
                cl = C.straight_clearance(start, goal, obs)
                kind = 'blocked' if cl < C.BLOCKED_CLEARANCE_M else 'passable'
                if kind == 'blocked' and (len(blocked) >= args.blocked or added_b >= per_world_blocked):
                    continue
                if kind == 'passable' and (len(passable) >= args.passable or added_p >= per_world_passable):
                    continue
                if C.is_excluded(w['world_id'], start, goal, ex):
                    excluded_hits += 1
                    continue
                key = (w['world_id'], tuple(np.round(start, 6)), tuple(np.round(goal, 6)))
                if key in seen:
                    continue
                seen.add(key)
                row = ticket(w, w['role'], start, yaw, goal, cl, kind, [1., 5.],
                             len(blocked) + len(passable))
                (blocked if kind == 'blocked' else passable).append(row)
                if kind == 'blocked':
                    added_b += 1
                else:
                    added_p += 1
        finally:
            env.close()
        print('world', wi, w['world_id'], 'blocked', len(blocked), 'passable', len(passable), flush=True)
    result = dict(schema='f12_s0_pool_v1', seed=20261126,
                  source_bank=str(C.TRAIN_BANK), source_bank_sha256=C.sha256(C.TRAIN_BANK),
                  excluded_counts={k: len(v) for k, v in ex.items()},
                  excluded_applied=excluded_hits,
                  rule='blocked: straight start-goal minimum clearance (0.05 m grid) < 0.3 m; '
                       'passable: >= 0.3 m; dev420/val20/test12 triples excluded',
                  blocked=blocked, passable=passable,
                  counts=dict(blocked=len(blocked), passable=len(passable)))
    C.write_json(args.output, result)
    print('WROTE', args.output, result['counts'])


if __name__ == '__main__':
    main()
