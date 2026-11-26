#!/usr/bin/env python3
"""O2: candidate yield at selected blocked A states (7 conditions x 32 candidates)."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.f12_phase1 import f12_common as C
from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import encode_e5_condition_with_normalized


def make_condition(policy, agent_pos, point_cloud, theta=0.0):
    a = np.asarray(agent_pos, np.float32).copy()
    if theta:
        c, s = math.cos(theta), math.sin(theta)
        for t in range(a.shape[0]):
            gx, gy = float(a[t, 3]), float(a[t, 4])
            a[t, 3] = c * gx - s * gy
            a[t, 4] = s * gx + c * gy
    po = {'agent_pos': torch.as_tensor(a[None], dtype=torch.float32, device=policy.system.bc_runtime.device),
          'point_cloud': torch.as_tensor(np.asarray(point_cloud, np.float32)[None],
                                         dtype=torch.float32, device=policy.system.bc_runtime.device)}
    cond, _ = encode_e5_condition_with_normalized(policy.system.bc_runtime.policy, po,
                                                  policy.system.observation_adapter.spec)
    return cond


def mirror_action(action, yaw):
    a = np.asarray(action, np.float32).copy()
    c, s = math.cos(yaw), math.sin(yaw)
    # world -> body, flip lateral y, body -> world
    for t in range(a.shape[0]):
        x, y = float(a[t, 0]), float(a[t, 1])
        bx, by = c * x + s * y, -s * x + c * y
        by = -by
        a[t, 0] = c * bx - s * by
        a[t, 1] = s * bx + c * by
        a[t, 9] = -a[t, 9]
        a[t, 10] = -a[t, 10]
    return a


def geometry(plan, pre_pos, goal, obstacles, field):
    mc = C.min_clearance_path(plan, obstacles, step_m=.1)
    coll = bool(mc < .1 or np.min(plan[:, 2]) < .1)
    end = plan[-1]
    end_free = C.free_distance(end, C.plan_direction(plan), obstacles)
    dec = field.decrease(pre_pos, goal, end)
    lat = C.signed_lateral(pre_pos, goal, end)
    jag = C.jaggedness(plan)
    score = -10. * float(coll) + float(dec) + 2. * min(float(end_free), 6.) / 6.
    return dict(min_clearance=float(mc), collision=coll, end_free=float(end_free),
                geo_decrease=float(dec), end_lateral=float(lat), jaggedness=float(jag),
                score=float(score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', default=str(C.OUT / 's0' / 's0_states.npz'))
    ap.add_argument('--pool', default=str(C.OUT / 's0' / 's0_task_pool.json'))
    ap.add_argument('--out', default=str(C.OUT / 'o2'))
    ap.add_argument('--n-states', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20261126)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.states, allow_pickle=False)
    kinds = z['kind'].astype(str)
    idx = np.where(kinds == 'A')[0]
    if not len(idx):
        raise SystemExit('no A states')
    rng = np.random.default_rng(args.seed)
    if len(idx) > args.n_states:
        idx = np.sort(rng.choice(idx, args.n_states, replace=False))
    print('A states', len(idx), flush=True)
    system, config, header, _ = C.load_system()
    pol = C.Policy(system, header)
    from diffusion_policy_3d.env.collision_geom import load_obstacles
    bank = C.bank_worlds()
    worlds = {w['world_id']: w for w in bank['worlds'].values()}
    obs_cache = {}

    def obstacles(world_id):
        if world_id not in obs_cache:
            obs_cache[world_id] = load_obstacles(worlds[world_id]['obstacle_path'])
        return obs_cache[world_id]

    fields = {}
    groups = ['G-orig', 'G-rot30', 'G-rot60', 'G-mirror', 'G-sigma0.2', 'G-sigma0.3', 'G-early']
    rows, action_rows, plan_rows = [], [], []
    with C.protocol(system, deterministic=True, min_std=.1, clip=2.0):
        for si, i in enumerate(idx, 1):
            world_id = str(z['world_id'][i]); goal = z['goal'][i]
            agent = z['agent_pos'][i].astype(np.float32)
            pc = z['point_cloud'][i].astype(np.float32)
            pre = z['pre_position'][i].astype(np.float32)
            vel = z['pre_velocity'][i].astype(np.float32)
            yaw = float(z['pre_yaw'][i])
            obs = obstacles(world_id)
            if world_id not in fields:
                fields[world_id] = C.GeoField(worlds[world_id])
            field = fields[world_id]
            for gi, group in enumerate(groups):
                seed = int(rng.integers(0, 2 ** 31 - 1))
                if group == 'G-orig':
                    cond = make_condition(pol, agent, pc)
                    full = pol.sample_batch([cond] * 32, seed=seed, min_std=.1)
                    thetas = [0.] * 32
                elif group in ('G-rot30', 'G-rot60'):
                    th = 30. if group == 'G-rot30' else 60.
                    conds, thetas = [], []
                    for k in range(32):
                        t = math.radians(th if k < 16 else -th)
                        conds.append(make_condition(pol, agent, pc, t))
                        thetas.append(float(t))
                    full = pol.sample_batch(conds, seed=seed, min_std=.1)
                elif group == 'G-mirror':
                    cond = make_condition(pol, agent, pc)
                    full = pol.sample_batch([cond] * 32, seed=seed, min_std=.1)
                    full = np.stack([mirror_action(a, yaw) for a in full])
                    thetas = [0.] * 32
                elif group == 'G-sigma0.2':
                    cond = make_condition(pol, agent, pc)
                    full = pol.sample_batch([cond] * 32, seed=seed, min_std=.2)
                    thetas = [0.] * 32
                elif group == 'G-sigma0.3':
                    cond = make_condition(pol, agent, pc)
                    full = pol.sample_batch([cond] * 32, seed=seed, min_std=.3)
                    thetas = [0.] * 32
                else:  # G-early: current runtime has ft_denoising_steps == ddim_steps == 10
                    cond = make_condition(pol, agent, pc)
                    full = pol.sample_batch([cond] * 32, seed=seed, min_std=.1)
                    thetas = [0.] * 32
                for j in range(32):
                    plan = C.world_positions(full[j], pre, vel, yaw)
                    g = geometry(plan, pre, goal, obs, field)
                    rows.append(dict(state_index=int(i), state_rank=si, group=group, candidate=j,
                                     theta_deg=math.degrees(thetas[j]), sample_seed=seed,
                                     **g))
                    action_rows.append(np.asarray(full[j], np.float32))
                    plan_rows.append(plan.astype(np.float32))
                print(f'O2 {si}/{len(idx)} {group}', flush=True)
    np.savez_compressed(out / 'o2_candidate_arrays.npz',
                        action=np.stack(action_rows), plan=np.stack(plan_rows))
    fields_out = list(rows[0])
    with (out / 'o2_candidates.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields_out); w.writeheader(); w.writerows(rows)
    # choose top-4 and random-4 per state/group
    picks = []
    for si in range(1, len(idx) + 1):
        for group in groups:
            cand = [r for r in rows if r['state_rank'] == si and r['group'] == group]
            top = sorted(cand, key=lambda r: (-r['score'], r['candidate']))[:4]
            rr = np.random.default_rng(C.episode_seed(f'o2:{si}:{group}')).choice(32, 4, replace=False)
            picks += [dict(state_rank=si, group=group, reason='best', candidate=int(r['candidate']))
                      for r in top]
            picks += [dict(state_rank=si, group=group, reason='random', candidate=int(c))
                      for c in sorted(set(int(x) for x in rr) - {int(r['candidate']) for r in top})]
    with (out / 'o2_rollout_plan.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(picks[0])); w.writeheader(); w.writerows(picks)
    state_rows = []
    for si, i in enumerate(idx, 1):
        w = str(z['world_id'][i])
        f = fields[w]
        state_rows.append(dict(state_rank=si, state_index=int(i), world_id=w,
                               task_id=str(z['task_id'][i]), episode_id=str(z['episode_id'][i]),
                               repeat=int(z['repeat'][i]), decision=int(z['decision'][i]),
                               goal_azimuth=float(z['label_goal_azimuth'][i]),
                               free_front=float(z['label_free_front'][i]),
                               feasible_side=float(f.feasible_side(z['pre_position'][i], z['goal'][i]))))
    with (out / 'o2_states.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(state_rows[0])); w.writeheader(); w.writerows(state_rows)
    C.write_json(out / 'o2_selection.json', dict(states=len(idx), groups=groups,
        candidates=len(rows), rollouts=len(picks), seed=args.seed,
        source_states=str(Path(args.states).resolve()),
        notes=['G-early is degenerate under the loaded runtime: ft_denoising_steps == ddim_steps == 10, '
               'so all denoising steps already use the fine-tuned actor and the same floor.']))
    print('O2_CANDIDATES_DONE', len(rows), 'picks', len(picks), flush=True)


if __name__ == '__main__':
    main()
