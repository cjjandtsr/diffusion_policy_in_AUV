#!/usr/bin/env python3
"""S0: run P6 #E2 on the training-domain ticket pool and select O2/O3/O5 states.

A: blocked episode, front free distance along goal azimuth in [3,6] m.
B: blocked collision episode, decisions n-5..n-3 (0-based) before the terminal collision.
C: passable episode, random decision (p=0.2 per decision, capped later).
"""
import argparse
from collections import deque
import csv
import gzip
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.f12_phase1 import f12_common as C


def clearance_from_path(path, obstacles):
    return C.min_clearance_path(path, obstacles, step_m=.1)


def state_record(obs, context, pre, full, plan, task, world, kind, decision, labels):
    return dict(kind=kind, task_id=task['task_id'], world_id=world['world_id'], role=task['role'],
                episode_id=task['payload']['episode_id'], repeat=int(task['repeat']),
                decision=int(decision), labels=labels,
                point_cloud=np.asarray(obs['point_cloud'][-3:], np.float16),
                agent_pos=np.asarray(obs['agent_pos'][-3:], np.float32),
                pre_position=np.asarray(pre['position'], np.float32),
                pre_velocity=np.asarray(pre['velocity'], np.float32),
                pre_yaw=np.float32(pre['yaw']),
                full_action=np.asarray(full, np.float32),
                plan_path=np.asarray(plan, np.float32),
                goal=np.asarray(task['payload']['goal_pos'], np.float32),
                payload=task['payload'])


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    fields = list(rows[0])
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default=str(C.OUT / 's0' / 's0_task_pool.json'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=str(C.OUT / 's0'))
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'traces').mkdir(parents=True, exist_ok=True)
    pool = C.read_json(args.pool)
    tasks = []
    for kind in ('blocked', 'passable'):
        for t in pool[kind]:
            for r in t['repeats']:
                tasks.append(dict(t, kind=kind, repeat=r['repeat'], reset_seed=r['reset_seed'],
                                  action_seeds=r['action_seeds']))
    if args.limit:
        tasks = tasks[:args.limit]
    system, config, header, _ = C.load_system()
    from diffusion_policy_3d.env.collision_geom import load_obstacles
    bank = C.bank_worlds()
    pol = C.Policy(system, header)
    worlds = {}
    obstacles = {}
    for w in bank['worlds'].values():
        worlds[w['world_id']] = w
        obstacles[w['world_id']] = load_obstacles(w['obstacle_path'])
    states = []
    episodes = []
    started = time.time()
    with C.protocol(system, deterministic=True, min_std=.1, clip=2.0):
        for ti, task in enumerate(tasks, 1):
            world = worlds[task['world_id']]
            obs_list = []
            ad = C.adapter_for(system, config, world, task['role'])
            obs, ctx = C.reset(ad, task['payload'], task['reset_seed'])
            rows = []
            snapshots = []
            try:
                for decision in range(60):
                    pre = {'position': np.asarray(ad.environment._drone_pos, np.float32).copy(),
                           'velocity': np.asarray(ad.environment._drone_vel, np.float32).copy(),
                           'yaw': float(ad.environment._drone_yaw)}
                    bundle = pol.predict(obs, ctx, seed=task['action_seeds'][decision])
                    full = bundle.full[0].detach().cpu().numpy()
                    plan = C.world_positions(full, pre['position'], pre['velocity'], pre['yaw'])
                    obs2, reward, term, trunc, ctx2 = C.step(ad, bundle)
                    rows.append(dict(decision=decision, pre=pre, full=full, plan=plan,
                                     post=np.asarray(ad.environment._drone_pos, np.float32).copy(),
                                     executed=np.asarray(ctx2.get('executed_path'), np.float32),
                                     reason=str(ctx2.get('done_reason', 'none')),
                                     reward=float(reward)))
                    snapshots.append((obs, ctx, pre, full, plan))
                    obs, ctx = obs2, ctx2
                    if term or trunc:
                        break
            finally:
                ad.environment.close()
            n = len(rows)
            reason = rows[-1]['reason'] if rows else 'invalid'
            goal = np.asarray(task['payload']['goal_pos'], float)
            obs = obstacles[task['world_id']]
            # cumulative executed path and endpoints at each decision
            path = [np.asarray(task['payload']['start_pos'], np.float32)]
            ends = []
            for r in rows:
                path.extend(np.asarray(r['executed'], np.float32)[1:])
                ends.append(len(path) - 1)
            path = np.asarray(path, np.float32)
            # labels per decision
            labels = []
            for d, r in enumerate(rows):
                p = np.asarray(r['pre']['position'], float)
                theta = C.goal_azimuth(p, goal)
                f = C.free_distance(p, [math.cos(theta), math.sin(theta), 0.], obs)
                fl = C.free_distance(p, [math.cos(theta + math.pi / 4), math.sin(theta + math.pi / 4), 0.], obs)
                fr = C.free_distance(p, [math.cos(theta - math.pi / 4), math.sin(theta - math.pi / 4), 0.], obs)
                # future minimum clearance over next 2 / 4 decision segments
                def fut(k):
                    end = ends[min(len(ends) - 1, d + k - 1)]
                    return clearance_from_path(path[ends[d] if d else 0:end + 1], obs)
                labels.append(dict(free_front=float(f), free_left45=float(fl), free_right45=float(fr),
                                    future_clear_2=float(fut(2)), future_clear_4=float(fut(4)),
                                    blocked=bool(C.straight_clearance(p, goal, obs) < C.BLOCKED_CLEARANCE_M),
                                    goal_azimuth=float(theta)))
            # trace (compact; metadata kept as hashes)
            trace_path = out / 'traces' / f"{task['task_id']}_{task['repeat']}.jsonl.gz"
            with gzip.open(trace_path, 'wt', encoding='utf-8') as f:
                f.write(json.dumps(dict(type='episode_start', task_id=task['task_id'],
                                        world_id=task['world_id'], role=task['role'],
                                        repeat=task['repeat'], reset_seed=task['reset_seed'],
                                        payload=task['payload'], kind=task['kind'],
                                        policy='P6', sampling='deterministic',
                                        clip=2.0, min_std=.1), sort_keys=True) + '\n')
                for d, r in enumerate(rows):
                    rec = dict(type='transition', decision=d,
                               pre_state={k: (v.tolist() if hasattr(v, 'tolist') else v)
                                          for k, v in r['pre'].items()},
                               full_action_16x11=r['full'].astype(float).tolist(),
                               plan_path_world=r['plan'].astype(float).tolist(),
                               executed_path_world=r['executed'].astype(float).tolist(),
                               post_position=r['post'].astype(float).tolist(),
                               done_reason=r['reason'], reward=r['reward'])
                    f.write(json.dumps(rec, sort_keys=True) + '\n')
                f.write(json.dumps(dict(type='episode_end', done_reason=reason, decisions=n,
                                        trajectory_metrics=dict(min_goal_distance_m=float(np.min(
                                            np.linalg.norm(path - goal, axis=1))))),
                                   sort_keys=True) + '\n')
            # A states: up to 4 per blocked episode
            a_taken = 0
            if task['kind'] == 'blocked':
                for d, r in enumerate(rows):
                    if a_taken >= 4:
                        break
                    if 3.0 <= labels[d]['free_front'] <= 6.0:
                        o, cx, pre, full, plan = snapshots[d]
                        states.append(state_record(o, cx, pre, full, plan, task, world, 'A', d, labels[d]))
                        a_taken += 1
            # B states: collision and last 3..5 decisions
            if task['kind'] == 'blocked' and reason in ('collision', 'crash_ground', 'ground_crash'):
                for idx in range(max(0, n - 5), max(0, n - 3)):
                    o, cx, pre, full, plan = snapshots[idx]
                    states.append(state_record(o, cx, pre, full, plan, task, world, 'B', idx, labels[idx]))
            # C states: random decision in passable episode
            if task['kind'] == 'passable':
                for d, r in enumerate(rows):
                    if np.random.default_rng(C.episode_seed(f"{task['task_id']}:{task['repeat']}:{d}")).random() < .2:
                        o, cx, pre, full, plan = snapshots[d]
                        states.append(state_record(o, cx, pre, full, plan, task, world, 'C', d, labels[d]))
            episodes.append(dict(task_id=task['task_id'], world_id=task['world_id'], role=task['role'],
                                 kind=task['kind'], repeat=task['repeat'], reset_seed=task['reset_seed'],
                                 outcome=reason, decisions=n,
                                 min_goal_distance_m=float(np.min(np.linalg.norm(path - goal, axis=1))),
                                 trace=str(trace_path.relative_to(out)),
                                 trace_sha256=C.sha256(trace_path),
                                 state_counts={k: sum(1 for s in states if s['task_id'] == task['task_id']
                                                      and s['repeat'] == task['repeat'] and s['kind'] == k)
                                               for k in ('A', 'B', 'C')}))
            if ti % 10 == 0 or ti == len(tasks):
                dt = time.time() - started
                print(f'S0 {ti}/{len(tasks)} states={len(states)} elapsed={dt:.0f}s '
                      f'eta={dt / ti * (len(tasks) - ti):.0f}s', flush=True)
    # save states in one compressed npz
    A = [s for s in states if s['kind'] == 'A']
    B = [s for s in states if s['kind'] == 'B']
    Cst = [s for s in states if s['kind'] == 'C']
    # cap C deterministically to 2200, keep all A/B
    if len(Cst) > 2200:
        idx = np.random.default_rng(20261126).choice(len(Cst), 2200, replace=False)
        Cst = [Cst[i] for i in sorted(idx)]
    chosen = A + B + Cst
    arrays = dict(
        kind=np.asarray([s['kind'] for s in chosen]),
        task_id=np.asarray([s['task_id'] for s in chosen]),
        world_id=np.asarray([s['world_id'] for s in chosen]),
        role=np.asarray([s['role'] for s in chosen]),
        episode_id=np.asarray([s['episode_id'] for s in chosen]),
        repeat=np.asarray([s['repeat'] for s in chosen], np.int32),
        decision=np.asarray([s['decision'] for s in chosen], np.int32),
        point_cloud=np.stack([s['point_cloud'] for s in chosen]) if chosen else np.zeros((0, 3, 4096, 3), np.float16),
        agent_pos=np.stack([s['agent_pos'] for s in chosen]) if chosen else np.zeros((0, 3, 9), np.float32),
        pre_position=np.stack([s['pre_position'] for s in chosen]) if chosen else np.zeros((0, 3), np.float32),
        pre_velocity=np.stack([s['pre_velocity'] for s in chosen]) if chosen else np.zeros((0, 3), np.float32),
        pre_yaw=np.asarray([s['pre_yaw'] for s in chosen], np.float32),
        full_action=np.stack([s['full_action'] for s in chosen]) if chosen else np.zeros((0, 16, 11), np.float32),
        plan_path=np.stack([s['plan_path'] for s in chosen]) if chosen else np.zeros((0, 15, 3), np.float32),
        goal=np.stack([s['goal'] for s in chosen]) if chosen else np.zeros((0, 3), np.float32),
    )
    for key in ('free_front', 'free_left45', 'free_right45', 'future_clear_2', 'future_clear_4',
                'blocked', 'goal_azimuth'):
        arrays['label_' + key] = np.asarray([s['labels'][key] for s in chosen])
    np.savez_compressed(out / 's0_states.npz', **arrays)
    write_csv(out / 's0_states.csv', [dict(kind=s['kind'], task_id=s['task_id'], world_id=s['world_id'],
                                           role=s['role'], episode_id=s['episode_id'], repeat=s['repeat'],
                                           decision=s['decision'], **s['labels']) for s in chosen])
    write_csv(out / 's0_episodes.csv', episodes)
    C.write_json(out / 's0_selection_rules.json', dict(
        schema='f12_s0_selection_v1', pool=str(Path(args.pool).resolve()),
        counts=dict(episodes=len(episodes), A=len(A), B=len(B), C=len(Cst), total=len(chosen)),
        rules=dict(A='blocked task; 3 m <= geometric free distance along goal azimuth at current height < 6 m; '
                     'max 4 per episode, latest decisions only',
                   B='blocked collision episode; 0-based decisions max(0,n-5)..max(0,n-4), i.e. 3rd-5th before terminal',
                   C='passable task; per-decision deterministic random draw p=0.2, capped at 2200 uniformly'),
        outcome_counts={k: sum(e['outcome'] == k for e in episodes) for k in
                        sorted({e['outcome'] for e in episodes})},
        state_kind_counts={k: sum(1 for s in chosen if s['kind'] == k) for k in ('A', 'B', 'C')}))
    print('S0_DONE', json.dumps(dict(states=len(chosen), A=len(A), B=len(B), C=len(Cst))), flush=True)


if __name__ == '__main__':
    main()
