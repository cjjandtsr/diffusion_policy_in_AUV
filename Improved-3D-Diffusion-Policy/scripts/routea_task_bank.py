"""Candidate training-task bank on the 21 training worlds, labelled by difficulty tier.

Candidates follow the current training sampler (KinematicUAVEnv, curriculum C2):
  start/goal xy ~ U(-14, 14); start z ~ U(1.3, 4.7); goal z ~ U(1.5, 4.5);
  start_yaw = atan2(goal - start) + U(-0.3, 0.3);
  admission: |S - G| > 0.5 m, compute_obstacle_clearance(start) >= start_cl, (goal) >= goal_cl,
  with the per-role start_cl / goal_cl of _ROLE_SAMPLING_PARAMS (role = the world's role).
Derived "near-obstacle start" candidates: for a blocked candidate, a start placed 3.0 m before the
first obstacle hit along the straight S-G line (same goal), admitted with the same rules.
Candidates whose start AND goal are both within 1.0 m of a dev420 task in the same world are dropped
(dev420 uses these worlds for evaluation).
Each admitted candidate gets the metrics of metrics.py (route-band b_band, kappa, path_zmax, blocked)
and outcome_change.band_density (route band +-3 m, 3 m beyond goal). No policy outcome is used.
Usage: python task_bank.py <repo_root> <out.json> [per_world] [workers]
"""
import json
import math
import random
import sys
from multiprocessing import Pool

import numpy as np
import argparse
import hashlib
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROLE_CL = {'anticipatory': (0.60, 2.50), 'dense_recovery': (0.60, 1.50),
           'support_expansion': (0.80, 2.00), 'yaw_lateral_recovery': (0.80, 2.00)}
FIELDS = ('name', 'shape', 'x', 'y', 'z', 'yaw', 'radius', 'height', 'size_x', 'size_y', 'size_z')
REPO = None


def init(repo):
    global REPO
    REPO = repo
    sys.path.insert(0, repo + '/Improved-3D-Diffusion-Policy')


def admit(start, goal, obstacles, role):
    from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
    env = object.__new__(KinematicUAVEnv)
    env.curriculum_stage, env.z_max, env.reward_version = 'C2', 5.0, 'v2'
    env.obstacles = obstacles
    return not env._candidate_rejection_reasons(np.asarray(start,dtype=np.float32),np.asarray(goal,dtype=np.float32),role)


def near_obstacle_start(start, goal, obstacles, back=3.0, n=400):
    """First hit parameter along S->G by dense sampling with segment checks; start 3 m before it."""
    from diffusion_policy_3d.env.collision_geom import segment_obstacle_collision
    L = math.dist(start, goal)
    prev = start
    for i in range(1, n + 1):
        t = i / n
        p = [start[k] + t * (goal[k] - start[k]) for k in range(3)]
        if segment_obstacle_collision(prev, p, obstacles, 0.1) is not None:
            d = max(t * L - back, 0.0)
            if d < 1.0:
                return None
            return [start[k] + d / L * (goal[k] - start[k]) for k in range(3)]
        prev = p
    return None


def work(job):
    from routea_paired_support import task_geometry_metrics as M
    from routea_paired_support.task_geometry_metrics import band_density
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec, segment_obstacle_collision
    world, role, geometry, dev_pairs, n, seed = job
    obstacles = [ObstacleSpec(**{k: o[k] for k in FIELDS}) for o in geometry]
    rng = random.Random(seed)
    rows, tries = [], 0
    while len(rows) < n and tries < 50 * n:
        tries += 1
        s = [rng.uniform(-14, 14), rng.uniform(-14, 14), rng.uniform(1.3, 4.7)]
        g = [rng.uniform(-14, 14), rng.uniform(-14, 14), rng.uniform(1.5, 4.5)]
        if not admit(s, g, obstacles, role):
            continue
        cands = [('random', s)]
        if segment_obstacle_collision(s, g, obstacles, 0.1) is not None:
            ns = near_obstacle_start(s, g, obstacles)
            if ns is not None and admit(ns, g, obstacles, role):
                cands.append(('near_obstacle_start', ns))
        for kind, st in cands:
            if len(rows) == n:
                break
            if any(math.dist(st, a) < 1.0 and math.dist(g, b) < 1.0 for a, b in dev_pairs):
                continue
            yaw = math.atan2(g[1] - st[1], g[0] - st[0]) + rng.uniform(-0.3, 0.3)
            task = dict(task_id='%s__bank%05d_%s' % (world, len(rows), kind[0]), split='bank', role=role,
                        world=world, start=st, goal=g, geometry=geometry)
            m = M.compute(task)
            if m['flags']:
                continue
            m.pop('path', None)
            m['tier'] = M.tier(dict(m, band_rho=band_density(task)))
            m.update(kind=kind, start_yaw=yaw, band_rho=band_density(task), n_world_obstacles=len(geometry))
            rows.append(m)
    if len(rows) != n:
        raise RuntimeError('candidate budget exhausted for '+world)
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest','worlds-train','dev-bank','output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--per-world',type=int,default=400)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    from diffusion_policy_3d.rl.dppo_uav.integration import manifest_train_entries
    from diffusion_policy_3d.env.collision_geom import load_obstacles
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    dev = json.loads(Path(args.dev_bank).read_text(encoding='utf-8'))
    worlds, jobs = {}, []
    entries = sorted(manifest_train_entries(args.manifest),key=lambda e:e['world_uid'])
    if len(entries)!=21 or len({e['world_uid'] for e in entries})!=21:
        raise ValueError('expected 21 admitted training worlds')
    for i,e in enumerate(entries):
        world,role = e['world_uid'],e.get('role') or e['role_active']
        path = Path(args.worlds_train)/(world+'.world.obstacles.json')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if dev['worlds'][world]['obstacle_sha256']!=sha:
            raise ValueError('dev/training geometry mismatch: '+world)
        geometry = [asdict(o) for o in load_obstacles(str(path))]
        worlds[world] = dict(role=role,obstacle_sha256=sha,n_obstacles=len(geometry))
        # Execution-side fix (2026-11-01): the frozen dev420 bank stores task geometry under
        # the task 'payload' (`schema=routea_train_domain_paired_dev_v1`), not at top level.
        pairs = [(t['payload']['start_pos'],t['payload']['goal_pos'])
                 for t in dev['tasks'] if t['world_id']==world]
        if len(pairs)!=4:
            raise ValueError('expected four dev tasks per world')
        jobs.append((world,role,geometry,pairs,args.per_world,20261031+i))
    with Pool(args.workers) as pool:
        rows = [r for part in pool.map(work,jobs) for r in part]
    payload = dict(schema='routea_training_task_bank_v1',worlds=worlds,tasks=rows,
                   per_world=args.per_world,generation_seed=20261031,
                   manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                   dev_bank_sha256=hashlib.sha256(Path(args.dev_bank).read_bytes()).hexdigest())
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,sort_keys=True,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(tasks=len(rows),worlds=len(worlds),
                         sha256=hashlib.sha256(out.read_bytes()).hexdigest())))

if __name__ == '__main__':
    main()

