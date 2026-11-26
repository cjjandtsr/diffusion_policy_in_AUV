#!/usr/bin/env python3
"""O5: counterfactual sensitivity at B states (original / obstacle removed / goal +2 m left / both)."""
import argparse
import csv
import gzip
import hashlib
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


def aim_shift(agent, dy):
    a = np.asarray(agent, np.float32).copy()
    a[:, 4] += float(dy)
    return a


def body_positions(full, yaw):
    d = np.asarray(full, np.float32)[:, :3]
    c, s = math.cos(yaw), math.sin(yaw)
    out, p = [], np.zeros(3, np.float32)
    for k in range(len(d)):
        bx = c * d[k, 0] + s * d[k, 1]
        by = -s * d[k, 0] + c * d[k, 1]
        p = p + np.array([bx, by, d[k, 2]], np.float32)
        out.append(p.copy())
    return np.asarray(out)


def nearest_obstacle(point, obstacles):
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance
    c, name, _ = compute_obstacle_clearance(float(point[0]), float(point[1]), float(point[2]), obstacles)
    return name, c


def episode_seed(payload, reset_seed):
    data = {'world_id': payload['world_name'],
            'goal_w': [round(float(v), 4) for v in payload['goal_pos']],
            'drone_pos': [round(float(v), 4) for v in payload['start_pos']],
            'seed': int(reset_seed)}
    digest = hashlib.blake2b(json.dumps(data, sort_keys=True).encode('utf-8'), digest_size=4).digest()
    return int.from_bytes(digest, 'little')


def synth_cloud(obstacle_json, pos, yaw, seed, device):
    from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
    sp = SynthPointCloud(str(obstacle_json), device=device)
    pc, _ = sp.generate_with_metadata(np.asarray(pos, float), float(yaw), num_points=4096,
                                      add_noise=True, rng_seed=int(seed), episode_identity_hash='0' * 64,
                                      decision_index=0)
    return pc.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', default=str(C.OUT / 's0' / 's0_states.npz'))
    ap.add_argument('--pool', default=str(C.OUT / 's0' / 's0_task_pool.json'))
    ap.add_argument('--traces', default=str(C.OUT / 's0' / 'traces'))
    ap.add_argument('--out', default=str(C.OUT / 'o5'))
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--seed', type=int, default=20261126)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.states, allow_pickle=False)
    idx = np.where(z['kind'].astype(str) == 'B')[0]
    if not len(idx):
        raise SystemExit('no B states')
    rng = np.random.default_rng(args.seed)
    if len(idx) > args.n:
        idx = np.sort(rng.choice(idx, args.n, replace=False))
    print('B states', len(idx), flush=True)
    pool = C.read_json(args.pool)
    tasks = {}
    for kind in ('blocked', 'passable'):
        for t in pool[kind]:
            tasks[t['task_id']] = t
    bank = C.bank_worlds()
    worlds = {w['world_id']: w for w in bank['worlds'].values()}
    from diffusion_policy_3d.env.collision_geom import load_obstacles
    system, config, header, _ = C.load_system()
    pol = C.Policy(system, header)
    device = str(system.bc_runtime.device)
    rows, plans = [], []
    with C.protocol(system, deterministic=True, min_std=.1, clip=2.0), torch.no_grad():
        for si, i in enumerate(idx, 1):
            world_id = str(z['world_id'][i])
            task = tasks[str(z['task_id'][i])]
            world = worlds[world_id]
            obstacles = load_obstacles(world['obstacle_path'])
            # terminal point from the S0 trace
            trace = Path(args.traces) / f"{task['task_id']}_{int(z['repeat'][i])}.jsonl.gz"
            terminal = None
            with gzip.open(trace, 'rt', encoding='utf-8') as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get('type') == 'transition':
                        terminal = np.asarray(rec['executed_path_world'][-1], float)
            obs_name, _ = nearest_obstacle(terminal, obstacles)
            geometry = json.loads(Path(world['obstacle_path']).read_text())
            geometry['obstacles'] = [o for o in geometry['obstacles'] if o.get('name') != obs_name]
            temp = out / f"no_{obs_name}_{world_id}.json"
            temp.write_text(json.dumps(geometry))
            agent0 = z['agent_pos'][i].astype(np.float32)
            pc0 = z['point_cloud'][i].astype(np.float32)
            pre = z['pre_position'][i].astype(np.float32)
            yaw = float(z['pre_yaw'][i])
            reset_seed = next(int(r['reset_seed']) for r in task['repeats']
                              if int(r['repeat']) == int(z['repeat'][i]))
            seed = episode_seed(task['payload'], reset_seed)
            # re-synthesize the three saved history frames without the collision obstacle
            pcs = []
            for t in range(3):
                pos_odom = agent0[t, :3]
                spawn = np.asarray(world.get('spawn_offset', [0, 0, 0]), np.float32)
                pos_world = pos_odom + spawn
                y = math.atan2(float(agent0[t, 7]), float(agent0[t, 8]))
                pcs.append(synth_cloud(temp, pos_world, y, seed, device))
            pc_removed = np.stack(pcs)
            scenarios = [('orig', agent0, pc0), ('no_obstacle', agent0, pc_removed),
                         ('goal_left2', aim_shift(agent0, 2.0), pc0),
                         ('both', aim_shift(agent0, 2.0), pc_removed)]
            for name, agent, pc in scenarios:
                a = torch.as_tensor(agent[None], dtype=torch.float32, device=device)
                p = torch.as_tensor(pc[None], dtype=torch.float32, device=device)
                cond, _ = encode_e5_condition_with_normalized(system.bc_runtime.policy,
                                                               {'agent_pos': a, 'point_cloud': p},
                                                               system.observation_adapter.spec)
                samples = system.official_runtime(cond=cond, deterministic=True, return_chain=True,
                                                  use_base_policy=False)
                full = system.bc_runtime.policy.normalizer['action'].unnormalize(samples.trajectories)
                full = full[0].detach().cpu().numpy()
                bp = body_positions(full[2:16], yaw)
                plans.append(bp)
                rows.append(dict(state_rank=si, kind='B', world_id=world_id, task_id=task['task_id'],
                                 repeat=int(z['repeat'][i]), decision=int(z['decision'][i]),
                                 collision_obstacle=str(obs_name), scenario=name,
                                 endpoint_lateral_m=float(bp[-1, 1]), endpoint_forward_m=float(bp[-1, 0]),
                                 endpoint_x_m=float(bp[-1, 0]), endpoint_y_m=float(bp[-1, 1]),
                                 endpoint_z_m=float(bp[-1, 2])))
                # geometry-free probe: non-ground points in the removed world
                if name == 'no_obstacle':
                    spawn = np.asarray(world.get('spawn_offset', [0, 0, 0]), np.float32)
                    world_z = pc_removed[..., 2] + agent0[:, None, 2] + spawn[2]
                    rows[-1]['removed_non_ground_points'] = int(np.sum(world_z > .15))
            print('O5', si, '/', len(idx), flush=True)
    np.savez_compressed(out / 'o5_plans_body.npz', plans=np.stack(plans))
    fields = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with (out / 'o5_counterfactual.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    C.write_json(out / 'o5_selection.json', dict(states=len(idx), scenarios=4,
        source_states=str(Path(args.states).resolve()), seed=args.seed,
        note='removed obstacle = closest obstacle at the terminal collision point; clouds re-synthesized '
             'with the episode seed and the same 3 historical poses'))
    print('O5_DONE', len(rows), flush=True)


if __name__ == '__main__':
    main()
