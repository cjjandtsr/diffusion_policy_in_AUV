"""Shared, read-only F12 Phase-1 helpers.

All inference uses the frozen P6@u256 runtime and the native E2 environment.
Training/evaluation source bytes are verified before any episode is run.
"""
from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/home/cjj/AUV/program/results/routea_f12_phase1_run01')
F9_PLAN = Path('/home/cjj/AUV/program/results/routea_f9_run01/plan.json')
TRAIN_BANK = Path('/home/cjj/AUV/program/results/routea_f8_diag_run03/bank.json')
DEV_BANK = Path('/home/cjj/AUV/program/results/routea_p6_u256_e2_dev420_run01_restored/bank.json')
VAL_BANK = Path('/home/cjj/AUV/program/results/routea_p6_u256_e2_val20_run01/bank.json')
TEST12_BANK = Path('/home/cjj/AUV/program/results/routea_test12_independent_run02/bank/bank.json')
WORLDS_TRAIN = Path('/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train')
P6_STATE_SHA = 'a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43'
RUNTIME_ID = 'd987235a0cbb5ad395bbd6e7979bdb8d7e0cff1303b0611cb462c7d554bbd61f'
STRAIGHT_STEP_M = .05
BLOCKED_CLEARANCE_M = .3
FREE_MAX_M = 10.0
PLAN_START = 2
PLAN_END = 16


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n',
                   encoding='utf-8')
    tmp.replace(path)


def load_plan():
    plan = read_json(F9_PLAN)
    assert plan['P6']['state_sha256'] == P6_STATE_SHA
    assert plan['P6']['runtime_identity_hash'] == RUNTIME_ID
    return plan


def load_system():
    """Strict native restore; returns (system, config, header, overlays)."""
    import torch
    # Any project module imported by a caller before this point must not force
    # the restored runtime to bind the work-tree copy instead of native_a2.
    for name in list(sys.modules):
        if name == 'diffusion_policy_3d' or name.startswith('diffusion_policy_3d.'):
            del sys.modules[name]
    from scripts.run_routea_f10_eval import overlay_modules
    from scripts.run_routea_xk_evaluation import native_system, verify_source
    plan = load_plan()
    s = plan['P6']
    verify_source(s['source_root'], dict(files=s['sources']))
    system, config, header = native_system(SimpleNamespace(
        source_root=s['source_root'], repo=plan['repo'], checkpoint=s['checkpoint'],
        expected_update=256, policy='P6', evaluation_reproduction=False))
    assert header['runtime_identity_hash'] == RUNTIME_ID
    overlays = overlay_modules()
    return system, config, header, overlays


def protocol(system, *, deterministic=True, min_std=.1, clip=2.0):
    from scripts.run_routea_xk_evaluation import load_override
    return load_override(ROOT)(system.official_runtime, denoised_clip_value=clip,
                               min_std_vector=[float(min_std)] * 11,
                               sampling_mode='deterministic' if deterministic else 'random')


class Policy:
    """Deterministic P6 / BC prediction with the exact E2 protocol."""

    def __init__(self, system, header, *, base=False):
        from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy
        self.system = system
        self.use_base_policy = bool(base)
        self.identity_hash = (system.bc_runtime.identity.content_hash if base
                              else header['runtime_identity_hash'])
        self.policy = _E5Policy(system, use_base_policy=base, identity_hash=self.identity_hash)

    def predict(self, observation, context, seed):
        return self.policy.predict(observation, context, action_seed=int(seed))

    def encode(self, observation, context):
        return self.system.observation_adapter.encode(
            observation, context['pointcloud_meta_history'],
            episode_identity_hash=context['episode_identity_hash'],
            decision_index=context['decision_index'])

    def sample_batch(self, conditions, *, seed, min_std=.1):
        """Stochastic DDIM, one batched native forward; returns unnormalized (B,16,11)."""
        import torch
        cond = {k: torch.cat([c[k] for c in conditions], dim=0) for k in conditions[0]}
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad(), protocol(self.system, deterministic=False, min_std=min_std):
            sample = self.system.official_runtime(
                cond=cond, deterministic=False, return_chain=True,
                use_base_policy=self.use_base_policy)
            full = self.system.bc_runtime.policy.normalizer['action'].unnormalize(sample.trajectories)
        return full.detach().cpu().numpy()


def bank_worlds():
    b = read_json(TRAIN_BANK)
    for w in b['worlds'].values():
        if not Path(w['obstacle_path']).is_file():
            raise FileNotFoundError(w['obstacle_path'])
        assert sha256(w['obstacle_path']) == w['obstacle_sha256']
    return b


def excluded_triples():
    """(world, rounded start, rounded goal) from dev/val/test12 evaluation banks."""
    out = {}
    for name, path in (('dev', DEV_BANK), ('val', VAL_BANK), ('test12', TEST12_BANK)):
        if not Path(path).is_file():
            out[name] = []
            continue
        b = read_json(path)
        rows = []
        for t in b['tasks']:
            p = t['payload']
            rows.append((str(t['world_id']),
                         tuple(round(float(v), 6) for v in p['start_pos']),
                         tuple(round(float(v), 6) for v in p['goal_pos'])))
        out[name] = rows
    return out


def is_excluded(world_id, start, goal, exclusions):
    key = (str(world_id), tuple(round(float(v), 6) for v in start),
           tuple(round(float(v), 6) for v in goal))
    for rows in exclusions.values():
        if key in rows:
            return True
    return False


def make_payload(world_id, role, start, yaw, goal, episode_id, z_range=(1.0, 5.0)):
    return dict(episode_id=str(episode_id), world_name=str(world_id),
                start_pos=[float(v) for v in start], start_yaw=float(yaw),
                goal_pos=[float(v) for v in goal], z_range=[float(z_range[0]), float(z_range[1])],
                disturb_enable=False, role=str(role))


def adapter_for(system, config, world, role):
    from scripts.routea_paired_support.runtime import make_adapter
    return make_adapter(system, config, world, role, evaluation_protocol='E2')


def reset(adapter, payload, seed):
    from scripts.evaluate_dppo_uav_formal_endpoint import reset_fixed_episode
    return reset_fixed_episode(adapter.environment, payload, seed=int(seed),
                               reset_callable=adapter.reset)


def step(adapter, bundle):
    return adapter.step(bundle)


def bundle_from_full(full):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle
    import torch
    if not torch.is_tensor(full):
        full = torch.as_tensor(np.asarray(full, np.float32))
    return ActionBundle.from_full(full)


def world_positions(full, pre_pos, pre_vel, pre_yaw):
    """F10 guidance conversion: full[2:16] deltas from current world pose."""
    from diffusion_policy_3d.env.traj_utils import action_to_world, POS
    full = np.asarray(full, np.float32)
    if full.ndim == 3:
        full = full[0]
    wp = action_to_world(full[PLAN_START:PLAN_END], np.asarray(pre_pos, np.float32),
                         np.asarray(pre_vel, np.float32), float(pre_yaw))
    return np.vstack((np.asarray(pre_pos, np.float32), wp[:, POS]))


def plan_direction(path):
    d = np.asarray(path[-1], float) - np.asarray(path[-2], float)
    d[2] = 0.
    n = np.linalg.norm(d[:2])
    return d / n if n > 1e-9 else np.array([1., 0., 0.])


def min_clearance_path(path, obstacles, step_m=.1, ground_z=.1):
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance
    path = np.asarray(path, float)
    best = float('inf')
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step_m) + 1)
        for s in np.linspace(0, 1, n + 1):
            p = a + s * (b - a)
            if p[2] < ground_z:
                return float(min(best, -1.0))
            c, _, _ = compute_obstacle_clearance(float(p[0]), float(p[1]), float(p[2]), obstacles)
            if c < best:
                best = c
    return float(best)


def free_distance(pos, direction, obstacles, max_m=FREE_MAX_M, step_m=.05, ground_z=.1):
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance
    pos = np.asarray(pos, float)
    d = np.asarray(direction, float)
    d[2] = 0.
    n = np.linalg.norm(d[:2])
    if n < 1e-9:
        return 0.
    d = d / n
    for r in np.arange(0.0, max_m + 1e-9, step_m):
        p = pos + d * r
        if p[2] < ground_z:
            return float(r)
        c, _, _ = compute_obstacle_clearance(float(p[0]), float(p[1]), float(p[2]), obstacles)
        if c <= BLOCKED_CLEARANCE_M + 1e-9:
            return float(r)
    return float(max_m)


def straight_clearance(start, goal, obstacles, step_m=STRAIGHT_STEP_M):
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    n = max(1, int(np.ceil(np.linalg.norm(goal - start) / step_m)))
    best = float('inf')
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance
    for i in range(n + 1):
        p = start + (goal - start) * (i / n)
        c, _, _ = compute_obstacle_clearance(float(p[0]), float(p[1]), float(p[2]), obstacles)
        if c < best:
            best = c
    return float(best)


def goal_azimuth(pos, goal):
    d = np.asarray(goal, float)[:2] - np.asarray(pos, float)[:2]
    return math.atan2(d[1], d[0])


def signed_lateral(pos, goal, point):
    d = np.asarray(goal, float)[:2] - np.asarray(pos, float)[:2]
    n = np.linalg.norm(d)
    if n < 1e-9:
        return 0.
    d = d / n
    v = np.asarray(point, float)[:2] - np.asarray(pos, float)[:2]
    return float(d[0] * v[1] - d[1] * v[0])


def jaggedness(path):
    p = np.asarray(path, float)
    if len(p) < 3:
        return 0.
    dd = p[2:] - 2 * p[1:-1] + p[:-2]
    return float(np.mean(np.linalg.norm(dd, axis=1)))


class GeoField:
    def __init__(self, world):
        from scripts.routea_paired_support.f8_geo_field import World
        geom = world['geometry']
        pts = []
        for o in geom:
            pts.append([o['x'], o['y'], o['z']])
            if o.get('shape') == 'cylinder':
                r = float(o.get('radius', 0.))
                pts.append([o['x'] + r, o['y'] + r, o['z']])
            else:
                pts.append([o['x'] + float(o.get('size_x', 0.)) / 2,
                            o['y'] + float(o.get('size_y', 0.)) / 2, o['z']])
        pts = np.asarray(pts, float) if pts else np.zeros((1, 3))
        lo = np.minimum(pts.min(0) - 5., [0., 0., 0.])
        hi = np.maximum(pts.max(0) + 5., [np.max(pts[:, 0]) + 5, np.max(pts[:, 1]) + 5, 8.])
        lo[2] = 0.
        self.world = World(geom, lo, hi)
        self._cache = {}

    def field(self, goal):
        key = tuple(np.round(np.asarray(goal, float), 6))
        if key not in self._cache:
            self._cache[key] = self.world.field(np.asarray(goal, float))
        return self._cache[key]

    def decrease(self, start, goal, end):
        f = self.field(goal)
        return float(f.dist(start) - f.dist(end))

    def feasible_side(self, pos, goal):
        """Signed side of the first geodesic step, left positive."""
        f = self.field(goal)
        pts = f.path(np.asarray(pos, float), 2.0)
        if len(pts) < 2:
            return 0.
        p = np.asarray(pts[1], float)
        d = np.asarray(goal, float)[:2] - np.asarray(pos, float)[:2]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return 0.
        d = d / n
        v = p[:2] - np.asarray(pos, float)[:2]
        c = d[0] * v[1] - d[1] * v[0]
        return float(np.sign(c))


def episode_seed(episode_id):
    return int(hashlib.sha256(str(episode_id).encode()).hexdigest()[:8], 16)


def action_seed(episode_id, repeat, decision):
    return int(hashlib.sha256(f'{episode_id}:{repeat}:{decision}'.encode()).hexdigest()[:8], 16)


def run_episode(system, config, world, task, *, policy=None, capture=False,
                branch_prefix=None, branch_action=None, max_decisions=60):
    """Run one deterministic E2 episode and return transitions.

    branch_prefix: optional list of full actions to replay before switching to policy.
    branch_action: optional full action to execute as the first action.
    """
    if policy is None:
        pol = Policy(system, {'runtime_identity_hash': RUNTIME_ID})
    else:
        pol = policy
    adapter = adapter_for(system, config, world, task['role'])
    obs, ctx = reset(adapter, task['payload'], task['reset_seed'])
    rows = []
    try:
        for decision in range(max_decisions):
            pre = {'position': np.asarray(adapter.environment._drone_pos, np.float32).copy(),
                   'velocity': np.asarray(adapter.environment._drone_vel, np.float32).copy(),
                   'yaw': float(adapter.environment._drone_yaw)}
            if branch_action is not None and decision == 0:
                full = np.asarray(branch_action, np.float32)
            elif branch_prefix is not None and decision < len(branch_prefix):
                full = np.asarray(branch_prefix[decision], np.float32)
            else:
                bundle = pol.predict(obs, ctx, action_seed(task['reset_seed'] + decision)
                                     if branch_prefix is None and branch_action is None
                                     else action_seed(task['task_id'], task['repeat'], decision))
                full = bundle.full[0].detach().cpu().numpy()
            bundle = bundle_from_full(full)
            obs_next, reward, term, trunc, ctx_next = step(adapter, bundle)
            rows.append(dict(decision=decision, pre_state=pre,
                             full_action=full.astype(np.float32),
                             post_position=np.asarray(adapter.environment._drone_pos, np.float32).copy(),
                             reward=float(reward), done_reason=str(ctx_next.get('done_reason', 'none')),
                             executed_path=np.asarray(ctx_next.get('executed_path'), np.float32),
                             observation=copy.deepcopy(obs) if capture else None,
                             context=copy.deepcopy(ctx) if capture else None))
            obs, ctx = obs_next, ctx_next
            if term or trunc:
                break
    finally:
        adapter.environment.close()
    return rows
