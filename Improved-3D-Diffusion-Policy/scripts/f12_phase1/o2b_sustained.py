#!/usr/bin/env python3
"""O2b sustained deviation (phase 1b, Claude 2026-11-27).

From each of the 200 O2 A-states: replay the S0 prefix, then fly P6 #E2 deterministic to the end of
the episode, rewriting the goal input (state dims 3,4 of all 3 history frames) for the first k
decisions.  Conditions x 4 repeats per state; one batched P6 call per decision (24 slots, fixed
order, finished slots padded) with torch seed = f(state, decision) -> reproducible.

  python scripts/f12_phase1/o2b_sustained.py --selftest
  python scripts/f12_phase1/o2b_sustained.py --out <dir> [--limit-states N]
"""
import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONDITIONS = (('orig', 0., 0), ('feas30_k1', 30., 1), ('feas30_k3', 30., 3), ('feas30_k5', 30., 5),
              ('away30_k5', -30., 5), ('feas60_k3', 60., 3))
REPEATS = 4
MAX_DECISIONS = 80


def rotate_goal_frames(agent, theta):
    """agent (3,9); rotate goal_body (dims 3,4) of every frame by theta (rad, +left)."""
    a = np.asarray(agent, np.float32).copy()
    c, s = math.cos(theta), math.sin(theta)
    gx, gy = a[:, 3].copy(), a[:, 4].copy()
    a[:, 3], a[:, 4] = c * gx - s * gy, s * gx + c * gy
    return a


def signed_theta(deg, feasible_side):
    """+deg toward the feasible side (left = +1), negative deg = toward the opposite side."""
    return math.radians(deg) * float(feasible_side)


def seed_of(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def first_entry_distance(path, goal):
    """Minimum distance from goal to the executed polyline (continuous, segment-wise)."""
    p = np.asarray(path, float); g = np.asarray(goal, float)
    if len(p) == 1:
        return float(np.linalg.norm(p[0] - g))
    a, b = p[:-1], p[1:]
    d = b - a
    L = np.maximum((d * d).sum(1), 1e-12)
    t = np.clip(((g - a) * d).sum(1) / L, 0., 1.)
    return float(np.min(np.linalg.norm(a + t[:, None] * d - g, axis=1)))


def selftest():
    a = np.zeros((3, 9), np.float32); a[:, 3] = 5.
    r = rotate_goal_frames(a, signed_theta(30., +1))
    assert np.allclose(r[:, 3], 4.330127, atol=1e-5) and np.allclose(r[:, 4], 2.5, atol=1e-6), r[0]
    r2 = rotate_goal_frames(a, signed_theta(30., -1))
    assert np.allclose(r2[:, 4], -2.5, atol=1e-6)
    r0 = rotate_goal_frames(a, 0.)
    assert np.array_equal(r0, a)
    assert abs(first_entry_distance([[0, 0, 0], [2, 0, 0]], [1, .5, 0]) - .5) < 1e-12
    for name, deg, k in CONDITIONS:
        for d in range(8):
            active = d < k
            assert (name == 'orig') <= (not active)
    print('O2B_SELFTEST ok: (5,0) -> (4.330, 2.500) left, (4.330, -2.500) right; '
          'rotation applied only while decision < k; segment distance 0.5')


def main():
    from scripts.f12_phase1 import f12_common as C
    import torch
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle
    from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import encode_e5_condition_with_normalized
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', default=str(C.OUT / 's0' / 's0_states.npz'))
    ap.add_argument('--pool', default=str(C.OUT / 's0' / 's0_task_pool.json'))
    ap.add_argument('--traces', default=str(C.OUT / 's0' / 'traces'))
    ap.add_argument('--o2-states', default=str(C.OUT / 'o2' / 'o2_states.csv'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit-states', type=int, default=0)
    ap.add_argument('--probe-repeat', action='store_true',
                    help='run the first 2 states twice and assert identical paths (probe 2)')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.states, allow_pickle=False)
    pool = C.read_json(args.pool)
    tasks = {t['task_id']: t for kind in ('blocked', 'passable') for t in pool[kind]}
    worlds = {w['world_id']: w for w in C.bank_worlds()['worlds'].values()}
    states = list(csv.DictReader(open(args.o2_states, encoding='utf-8-sig')))
    if args.limit_states:
        states = states[:args.limit_states]
    system, config, header, _ = C.load_system()
    dev = system.bc_runtime.device
    spec = system.observation_adapter.spec
    norm = system.bc_runtime.policy.normalizer['action']
    slots = [(c, r) for c in CONDITIONS for r in range(REPEATS)]

    def run_state(st):
        i = int(st['state_index'])
        side = float(st['feasible_side'])
        task = tasks[str(z['task_id'][i])]
        world = worlds[str(z['world_id'][i])]
        rep = int(z['repeat'][i]); decision0 = int(z['decision'][i])
        trace = Path(args.traces) / ('%s_%d.jsonl.gz' % (task['task_id'], rep))
        prefix = []
        with gzip.open(trace, 'rt', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('type') == 'transition' and rec['decision'] < decision0:
                    prefix.append(np.asarray(rec['full_action_16x11'], np.float32))
        reset_seed = next(int(r['reset_seed']) for r in task['repeats'] if int(r['repeat']) == rep)
        ad0 = C.adapter_for(system, config, world, task['role'])
        try:
            C.reset(ad0, task['payload'], reset_seed)
            for full in prefix:
                _, _, term, trunc, _ = ad0.step(ActionBundle.from_full(torch.as_tensor(full)))
                if term or trunc:
                    raise RuntimeError('prefix terminated before branch state')
            base_env, base_rs = copy.deepcopy(ad0.environment), copy.deepcopy(ad0.reward_state)
        finally:
            ad0.environment.close()
        ads = []
        for _ in slots:
            a = copy.copy(ad0); a.environment = copy.deepcopy(base_env); a.reward_state = copy.deepcopy(base_rs)
            ads.append(a)
        paths = [[np.asarray(base_env._drone_pos, float).copy()] for _ in slots]
        rewrite_paths = [[] for _ in slots]
        active = [True] * len(slots)
        steps = [0] * len(slots)
        goal_true = np.asarray(task['payload']['goal_pos'], float)
        probe_rows = []
        for d in range(MAX_DECISIONS):
            live = [j for j in range(len(slots)) if active[j]]
            if not live:
                break
            agents, pcs = [], []
            for j in range(len(slots)):
                jj = j if active[j] else live[0]
                o = ads[jj].environment._get_obs_dict()
                agent = np.asarray(o['agent_pos'][-3:], np.float32)
                (name, deg, k), _r = slots[j]
                if d < k:
                    agent_used = rotate_goal_frames(agent, signed_theta(deg, side))
                else:
                    agent_used = agent
                if active[j] and name == 'feas30_k3' and _r == 0 and d in (2, 3):
                    probe_rows.append(dict(decision=d, rewritten=bool(d < k),
                                           identical_to_true=bool(np.array_equal(agent_used, agent))))
                agents.append(agent_used)
                pcs.append(np.asarray(o['point_cloud'][-3:, :, :3], np.float32))
            with torch.no_grad():
                cond, _ = encode_e5_condition_with_normalized(
                    system.bc_runtime.policy,
                    {'agent_pos': torch.as_tensor(np.stack(agents), device=dev),
                     'point_cloud': torch.as_tensor(np.stack(pcs), device=dev)}, spec)
            torch.manual_seed(seed_of('o2b', st['state_rank'], d))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_of('o2b', st['state_rank'], d))
            with torch.no_grad(), C.protocol(system, deterministic=True, min_std=.1, clip=2.0):
                s = system.official_runtime(cond=cond, deterministic=True, return_chain=True, use_base_policy=False)
                full = norm.unnormalize(s.trajectories).cpu().numpy()
            for j in live:
                (name, deg, k), _r = slots[j]
                _, _, term, trunc, ctx = ads[j].step(ActionBundle.from_full(torch.as_tensor(full[j])))
                steps[j] += 1
                seg = np.asarray(ctx.get('executed_path'), float)
                paths[j].extend(seg[1:])
                if d < k:
                    rewrite_paths[j].extend(seg[1:].round(3).tolist())
                if term or trunc:
                    active[j] = False
        rows = []
        for j, ((name, deg, k), r) in enumerate(slots):
            env = ads[j].environment
            reason = str(env._done_reason or 'none')
            path = np.asarray(paths[j], float)
            mind = first_entry_distance(path, goal_true)
            # success = executed path enters 1 m before any collision; the path ends at the event
            rows.append(dict(state_rank=int(st['state_rank']), state_index=i, condition=name,
                             rotation_deg=deg, k=k, repeat=r, feasible_side=side,
                             outcome=reason, reached_1m=bool(reason == 'reached' or mind <= 1.0),
                             min_goal_distance_m=mind, decisions_after_branch=steps[j],
                             collision_xyz=json.dumps(path[-1].round(3).tolist()) if reason in ('collision', 'ground_crash') else '',
                             rewrite_path=json.dumps(rewrite_paths[j]),
                             world_id=str(z['world_id'][i]), task_id=task['task_id'],
                             free_front=float(st['free_front'])))
            env.close()
        return rows, probe_rows

    all_rows, probes = [], dict(rewrite_only_first_k=[], repeat_identical=[])
    for n, st in enumerate(states):
        if abs(float(st['feasible_side'])) < .5:
            print('O2B_SKIP feasible_side=0', st['state_rank'], flush=True)
            continue
        rows, pr = run_state(st)
        if pr:
            probes['rewrite_only_first_k'].append(dict(state_rank=int(st['state_rank']), rows=pr))
        if args.probe_repeat and n < 2:
            rows2, _ = run_state(st)
            same = all(a['min_goal_distance_m'] == b['min_goal_distance_m'] and a['outcome'] == b['outcome']
                       and a['rewrite_path'] == b['rewrite_path'] for a, b in zip(rows, rows2))
            probes['repeat_identical'].append(dict(state_rank=int(st['state_rank']), identical=bool(same)))
        all_rows.extend(rows)
        print('O2B_STATE', n + 1, '/', len(states), flush=True)
    with open(out / 'o2b_rollouts.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    p1 = all(r['identical_to_true'] == (not r['rewritten']) for g in probes['rewrite_only_first_k'] for r in g['rows'])
    p2 = all(x['identical'] for x in probes['repeat_identical']) if probes['repeat_identical'] else None
    C.write_json(out / 'o2b_probes.json', dict(probes, probe1_pass=p1, probe2_pass=p2))
    summary = {}
    for name, _, _ in CONDITIONS:
        rr = [r for r in all_rows if r['condition'] == name]
        summary[name] = dict(n=len(rr), reached_1m=sum(r['reached_1m'] for r in rr),
                             collision=sum(r['outcome'] in ('collision', 'ground_crash') for r in rr))
    C.write_json(out / 'o2b_summary.json', dict(schema='f12_o2b_v1', conditions=summary,
        protocol='P6 #E2 deterministic, clip 2.0; goal dims 3,4 of 3 frames rotated for decisions < k; '
                 'feasible side from O2 geodesic first step (left +1)'))
    print('O2B_DONE', len(all_rows), 'probe1', p1, 'probe2', p2, json.dumps(summary), flush=True)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        main()
