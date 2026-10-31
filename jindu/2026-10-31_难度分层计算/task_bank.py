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

ROLE_CL = {'anticipatory': (0.60, 2.50), 'dense_recovery': (0.60, 1.50),
           'support_expansion': (0.80, 2.00), 'yaw_lateral_recovery': (0.80, 2.00)}
FIELDS = ('name', 'shape', 'x', 'y', 'z', 'yaw', 'radius', 'height', 'size_x', 'size_y', 'size_z')
REPO = None


def init(repo):
    global REPO
    REPO = repo
    sys.path.insert(0, repo + '/Improved-3D-Diffusion-Policy')


def admit(start, goal, obstacles, role):
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance
    s_cl, g_cl = ROLE_CL[role]
    for p, lo_z, hi_z in ((start, 1.0, 5.0), (goal, 1.0, 5.0)):
        if not (-14 <= p[0] <= 14 and -14 <= p[1] <= 14 and lo_z <= p[2] <= hi_z):
            return False
    if math.dist(start, goal) <= 0.5:
        return False
    if compute_obstacle_clearance(*start, obstacles)[0] < s_cl:
        return False
    if compute_obstacle_clearance(*goal, obstacles)[0] < g_cl:
        return False
    return True


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
    import metrics as M
    from outcome_change import band_density
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec, segment_obstacle_collision
    M._init(REPO)
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
            if any(math.dist(st, a) < 1.0 and math.dist(g, b) < 1.0 for a, b in dev_pairs):
                continue
            yaw = math.atan2(g[1] - st[1], g[0] - st[0]) + rng.uniform(-0.3, 0.3)
            task = dict(task_id='%s__bank%05d_%s' % (world, len(rows), kind[0]), split='bank', role=role,
                        world=world, start=st, goal=g, geometry=geometry)
            m = M.compute(task)
            m.pop('path', None)
            m.update(kind=kind, start_yaw=yaw, band_rho=band_density(task), n_world_obstacles=len(geometry))
            rows.append(m)
    return rows


def main(repo, out_path, per_world='300', workers='7'):
    geo = json.load(open('task_geometry.json'))
    worlds = {}
    for t in geo:
        if t['split'] != 'dev420':
            continue
        w = worlds.setdefault(t['world'], dict(role=t['role'], geometry=t['geometry'], dev=[]))
        w['dev'].append((t['start'], t['goal']))
    jobs = [(w, v['role'], v['geometry'], v['dev'], int(per_world), 20261031 + i)
            for i, (w, v) in enumerate(sorted(worlds.items()))]
    with Pool(int(workers), initializer=init, initargs=(repo,)) as pool:
        rows = [r for part in pool.map(work, jobs) for r in part]
    json.dump(rows, open(out_path, 'w'))
    print('candidates', len(rows), 'worlds', len(jobs))


if __name__ == '__main__':
    main(*sys.argv[1:])
