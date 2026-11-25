#!/usr/bin/env python3
"""F10: evaluation-only denoising-time collision guidance (P6/BC x none/g1/G0).

Never trains, never edits the native sources, never rewrites the guidance module:
the guidance is loaded from scripts/routea_denoising_guidance.py as-is.

Per decision this entry records, in `probe_metrics`:
  guided            any guided update changed the x0 reconstruction (delta_exec > 0)
  delta_exec        F9 execution_delta(base, guided) on the 4 executed waypoints
  denoise_seconds   DDIM loop wall time (torch.cuda.synchronize around the sampler)
  inference_seconds whole predict() wall time (encode + denoise + unnormalize)
  cloud_points      number of points used by the guidance cost (z >= 0.1, world frame)
"""
import argparse
from collections import Counter
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.routea_paired_support.core import file_hash, read_json, write_json

POLICY_BASE = {'P6': False, 'BC': True}
GUIDANCE = {'none': None,
            'g1': dict(d_safe=.6, eta=.4, M=4, consistency=0),
            'G0': dict(d_safe=.6, eta=.2, M=2, consistency=0)}
P6_SHA = 'a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43'
OVERLAYS = ['diffusion_policy_3d.env.reward_v2_support.environment',
            'diffusion_policy_3d.env.kinematic_uav_env',
            'scripts.evaluate_dppo_uav_formal_endpoint',
            'scripts.routea_paired_support.core', 'scripts.routea_paired_support.runtime',
            'scripts.routea_paired_support.episodes']
SOURCES = [name.replace('.', '/') + '.py' for name in OVERLAYS] + [
    'scripts/run_routea_f10_eval.py', 'scripts/routea_denoising_guidance.py',
    'scripts/run_routea_xk_evaluation.py', 'scripts/f7_analysis.py',
    'diffusion_policy_3d/rl/dppo_uav_rebuild/engine/contracts.py',
    'diffusion_policy_3d/rl/dppo_uav_rebuild/engine/f9_online.py',
    'diffusion_policy_3d/env/traj_utils.py',
    'diffusion_policy_3d/env/reward_v2_support/first_event.py',
    'third_party/dppo_official_cc7234a/uav_idp3/e5_runtime.py',
    'third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py']


def csv_write(path, rows):
    import csv
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def execution_delta(base, guided, position, velocity, yaw):
    from diffusion_policy_3d.env.traj_utils import action_to_world
    a = action_to_world(np.asarray(base, np.float32)[2:6], np.asarray(position, np.float32),
                        np.asarray(velocity, np.float32), yaw)[:, :3]
    b = action_to_world(np.asarray(guided, np.float32)[2:6], np.asarray(position, np.float32),
                        np.asarray(velocity, np.float32), yaw)[:, :3]
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=1).max())


def delta_matches_f9():
    import ast
    import inspect
    import textwrap
    path = ROOT / 'diffusion_policy_3d/rl/dppo_uav_rebuild/engine/f9_online.py'
    text = path.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == 'execution_delta')
    upstream = textwrap.dedent(ast.get_source_segment(text, node)).strip()
    local = textwrap.dedent(inspect.getsource(execution_delta)).strip()
    assert upstream == local, 'execution_delta differs from F9 f9_online.py'
    return True


def source_identity():
    return {name: file_hash(ROOT / name) for name in SOURCES}


def load_plan(output):
    path = Path(output).resolve()
    plan = read_json(path / 'request.json')
    assert plan['evaluation_sources'] == source_identity(), 'evaluation code changed after prepare'
    delta_matches_f9()
    for name, item in plan['banks'].items():
        assert file_hash(item['path']) == item['sha256'], name + ' bank changed'
    return path, plan


def prepare(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = read_json(Path(args.weights_verified))['specs']['P6']
    assert receipt['state_sha256'] == P6_SHA and receipt['update'] == 256
    banks = {}
    for split, path in [('dev', Path(args.dev_bank)), ('val', Path(args.val_bank))]:
        bank = read_json(path)
        expected = 420 if split == 'dev' else 80
        assert len(bank['episodes']) == expected, (split, len(bank['episodes']))
        assert len({(e['task_id'], e['repeat']) for e in bank['episodes']}) == expected
        banks[split] = dict(path=str(path.resolve()), sha256=file_hash(path),
                            content_hash=bank['content_hash'], episodes=expected)
    write_json(out / 'request.json', dict(
        schema='routea_f10_v1', p6=receipt, banks=banks,
        evaluation_sources=source_identity(), native_source=str(Path(args.source_root).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()), repo=str(Path(args.repo).resolve()),
        settings=GUIDANCE, policies=POLICY_BASE,
        outcome='outcome_1m (paper arrival_1m, radius 1.0) plus raw done_reason',
        full_run_requires_user_schedule=True))
    print('F10_PREPARED', {k: v['episodes'] for k, v in banks.items()},
          'banks', {k: v['content_hash'][:8] for k, v in banks.items()})


def overlay_modules():
    identities = {}
    for name in OVERLAYS:
        path = ROOT / (name.replace('.', '/') + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        parent, leaf = name.rsplit('.', 1)
        setattr(sys.modules[parent], leaf, module)
        identities[name] = dict(file=str(path), sha256=file_hash(path))
    return identities


def runtime(args, plan):
    from scripts.run_routea_xk_evaluation import native_system, verify_source
    verify_source(plan['native_source'], dict(files=plan['p6']['sources']))
    assert file_hash(Path(plan['checkpoint']) / 'state.pt') == P6_SHA
    system, config, header = native_system(SimpleNamespace(
        source_root=plan['native_source'], repo=plan['repo'], checkpoint=plan['checkpoint'],
        expected_update=256, policy='P6', evaluation_reproduction=False))
    assert header['runtime_identity_hash'] == plan['p6']['runtime_identity_hash']
    return system, config, header, overlay_modules()


def bank_maps(plan, split):
    item = plan['banks'][split]
    bank = read_json(item['path'])
    for world in bank['worlds'].values():
        if not Path(world['obstacle_path']).is_file():
            world['obstacle_path'] = str(Path(item['path']).parent / world['obstacle_copy'])
        assert file_hash(world['obstacle_path']) == world['obstacle_sha256']
    return bank, {t['task_id']: t for t in bank['tasks']}


class TimedForward:
    """Time the DDIM loop only; the restored runtime keeps every other operator."""
    def __init__(self, forward):
        self.forward = forward
        self.last_seconds = None

    def __call__(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        value = self.forward(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_seconds = time.perf_counter() - start
        return value


class F10Policy:
    """Formal policy path unchanged; guidance + measurement layered on top."""

    def __init__(self, policy, env, *, guide, timed, runtime, restored_mean, guided_mean):
        self.policy, self.env, self.guide, self.timed = policy, env, guide, timed
        self.runtime = runtime
        self.restored_mean, self.guided_mean = restored_mean, guided_mean

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def begin_episode(self):
        if self.guide is not None:
            self.guide.previous = None

    def predict(self, observation, context, *, action_seed):
        device = self.policy.system.bc_runtime.device
        cloud_points = 0
        if self.guide is not None:
            cloud = torch.as_tensor(observation['point_cloud'][-1], device=device, dtype=torch.float32)
            self.guide.begin(cloud, self.env._drone_pos, float(self.env._drone_yaw))
            cloud_points = int(self.guide.cloud.shape[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        bundle = self.policy.predict(observation, context, action_seed=action_seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start
        denoise_seconds = self.timed.last_seconds
        metrics = dict(inference_seconds=inference_seconds, denoise_seconds=denoise_seconds,
                       guided=False, delta_exec=0., triggered_denoise_steps=[], update_count=0,
                       cloud_points=cloud_points, cloud_source='observation.point_cloud[-1] (z>=0.1, world)')
        if self.guide is not None:
            metrics['triggered_denoise_steps'] = list(self.guide.triggered_steps)
            metrics['update_count'] = int(self.guide.update_count)
            if self.guide.triggered_steps:
                # Base chain re-run with the same action seed: identical RNG stream, guidance off.
                self.runtime.p_mean_var = self.restored_mean
                try:
                    base = self.policy.predict(observation, context, action_seed=action_seed)
                finally:
                    self.runtime.p_mean_var = self.guided_mean
                delta = execution_delta(base.full[0].detach().cpu().numpy(),
                                        bundle.full[0].detach().cpu().numpy(),
                                        self.env._drone_pos, self.env._drone_vel, float(self.env._drone_yaw))
                metrics['delta_exec'] = float(delta)
                metrics['guided'] = bool(delta > 0.)
            # The memory is the FINAL output, not a denoising intermediate.
            position = torch.as_tensor(self.env._drone_pos, device=bundle.full.device, dtype=torch.float32)
            positions = []
            for delta in bundle.full[0, 2:16, :3]:
                position = position + delta
                positions.append(position)
            self.guide.previous = torch.stack(positions).detach()
        self.last_probe_metrics = metrics
        return bundle


def worker(args):
    out, plan = load_plan(args.output)
    if not args.limit:
        assert args.full_run_approved, 'full runs require scheduling approval'
    assert args.guidance in GUIDANCE and args.policy in POLICY_BASE
    system, config, header, overlays = runtime(args, plan)
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy
    from scripts.routea_paired_support.runtime import make_adapter
    from scripts.routea_paired_support.episodes import record_episode
    from scripts.run_routea_xk_evaluation import load_override
    from scripts.routea_denoising_guidance import Guidance, install
    bank, tasks = bank_maps(plan, args.split)
    if bank.get('schema') == 'routea_fixed_split_v1' and bank.get('repeats') == 1:
        from scripts import evaluate_dppo_uav_formal_endpoint as endpoint
        from scripts.routea_paired_support import episodes
        episodes.action_seed = lambda task_id, repeat, decision: endpoint._action_seed(task_id, decision)
    setting = dict(GUIDANCE[args.guidance] or {})
    if args.eta is not None:
        assert setting, 'eta override requires a guidance arm'
        setting['eta'] = float(args.eta)
    if args.consistency is not None:
        assert setting, 'consistency override requires a guidance arm'
        setting['consistency'] = int(args.consistency)
    label = '%s_%s' % (args.policy, args.guidance)
    name = args.tag or ('%s_%s' % (label, args.split))
    job = out / ('probes' if args.limit else 'runs') / name
    job.mkdir(parents=True, exist_ok=False)
    runtime_file = Path(inspect.getfile(type(system.official_runtime))).resolve()
    policy = _E5Policy(system, use_base_policy=POLICY_BASE[args.policy],
                       identity_hash=header['runtime_identity_hash'])
    identity = dict(type=type(system.official_runtime).__qualname__,
                    module=type(system.official_runtime).__module__, file=str(runtime_file),
                    source_sha256=file_hash(runtime_file), actual_use_base_policy=POLICY_BASE[args.policy],
                    state_sha256=file_hash(Path(plan['checkpoint']) / 'state.pt'),
                    native_runtime_identity=header['runtime_identity_hash'], overlays=overlays,
                    bank_sha256=plan['banks'][args.split]['sha256'], label=label, policy=args.policy,
                    guidance=args.guidance, setting=setting, split=args.split,
                    limit=args.limit, evaluation_sources=source_identity())
    scope = load_override(ROOT)(system.official_runtime, denoised_clip_value=2.,
                                min_std_vector=[.1] * 11, sampling_mode='deterministic')
    with scope as protocol:
        identity['sampling_protocol'] = protocol
        real = system.official_runtime
        forward_file = Path(inspect.getfile(real.forward)).resolve()
        identity['bound_forward'] = dict(file=str(forward_file), sha256=file_hash(forward_file))
        guide = None
        restored_mean = real.p_mean_var
        guided_mean = restored_mean
        if setting:
            guide = Guidance(system.bc_runtime.policy.normalizer['action'].unnormalize,
                             setting['eta'], setting['consistency'], setting['d_safe'], setting['M'])
            identity['guidance'] = install(real, guide)
            guided_mean = real.p_mean_var
        timed = TimedForward(real.forward)
        real.forward = timed
        wrapper = F10Policy(policy, None, guide=guide, timed=timed, runtime=real,
                            restored_mean=restored_mean, guided_mean=guided_mean)
        write_json(job / 'identity.json', identity)
        print('F10_RUNTIME', json.dumps(identity, ensure_ascii=False), flush=True)
        for i, entry in enumerate(bank['episodes'][:args.limit or None]):
            task = tasks[entry['task_id']]
            adapter = make_adapter(system, config, bank['worlds'][task['world_id']], task['role'],
                                   evaluation_protocol='E2')
            assert adapter.environment.exec_noise_scale == 0 and not adapter.environment.disturb_enable
            wrapper.env = adapter.environment
            try:
                record_episode(job, label, wrapper, adapter, task, entry,
                               bank['worlds'][task['world_id']], bank['content_hash'])
            finally:
                adapter.environment.close()
            print('F10_EPISODE', name, i + 1, flush=True)
    rows = collect(job, bank)
    csv_write(job / 'conditions.csv', rows)
    counts = Counter(r['outcome_1m'] for r in rows)
    raw = Counter(r['raw_outcome'] for r in rows)
    write_json(job / 'summary.json', dict(conditions=len(rows), label=label,
                                          outcome_1m=dict(counts), raw_done_reason=dict(raw),
                                          reached_1m=counts['reached']))
    print('F10_SUMMARY', json.dumps(dict(conditions=len(rows), outcome_1m=dict(counts),
                                         raw_done_reason=dict(raw)), ensure_ascii=False))


def _decision_stats(metrics):
    denoise = np.asarray([m['denoise_seconds'] for m in metrics], float)
    guided = np.asarray([bool(m['guided']) for m in metrics])
    deltas = np.asarray([m['delta_exec'] for m in metrics], float)
    out = dict(decisions=len(metrics), guided_decisions=int(guided.sum()),
               triggered_decisions=sum(bool(m['triggered_denoise_steps']) for m in metrics),
               delta_exec_max=float(deltas.max()) if len(deltas) else 0.,
               delta_exec_sum=float(deltas.sum()),
               denoise_median_ms=float(np.median(denoise) * 1e3) if len(denoise) else None,
               denoise_p95_ms=float(np.quantile(denoise, .95) * 1e3) if len(denoise) else None)
    if guided.any():
        out['denoise_median_ms_guided'] = float(np.median(denoise[guided]) * 1e3)
        out['denoise_median_ms_unguided'] = (float(np.median(denoise[~guided]) * 1e3)
                                             if (~guided).any() else None)
    return out


def collect(job, bank):
    from scripts.routea_paired_support.episodes import load_commit
    from scripts.f7_analysis import outcome_1m
    job = Path(job)
    identity = read_json(job / 'identity.json')
    tasks = {t['task_id']: t for t in bank['tasks']}
    commits = {p.stem: p for p in (job / 'commits').glob('*.json')}
    expected = {e['key'] for e in bank['episodes']}
    assert set(commits) <= expected
    rows = []
    for e in bank['episodes']:
        if e['key'] not in commits:
            continue
        h, trace, end = load_commit(job, commits[e['key']])
        assert h['policy'] == identity['label'] and h['policy_identity_hash'] == identity['native_runtime_identity']
        assert h['task_id'] == e['task_id'] and h['repeat'] == e['repeat']
        assert h['initial_hash'] == e['initial_hash']
        assert h['use_base_policy'] == identity['actual_use_base_policy']
        assert h['bank_hash'] == bank['content_hash'] and h['task'] == tasks[e['task_id']]['payload']
        world = bank['worlds'][tasks[e['task_id']]['world_id']]
        assert h['geometry'] == world['geometry'] and h['obstacle_sha256'] == world['obstacle_sha256']
        assert h['reset_seed'] == e['reset_seed']
        metrics = [t['probe_metrics'] for t in trace]
        goal = np.asarray(h['task']['goal_pos'], np.float32)
        outcome = outcome_1m(trace, goal, end['done_reason'])
        receipt = read_json(commits[e['key']])
        rows.append(dict(task_id=e['task_id'], repeat=e['repeat'], outcome_1m=outcome,
                         raw_outcome=end['done_reason'], decisions_total=end['decisions'],
                         min_goal_distance_m=end['trajectory_metrics']['min_goal_distance_m'],
                         trace=str(job / receipt['trace']), trace_sha256=receipt['sha256'],
                         **_decision_stats(metrics)))
    return rows


def collect_job(args):
    """Recompute a job's condition table without touching its traces."""
    out, plan = load_plan(args.output)
    bank, _ = bank_maps(plan, args.split)
    rows = collect(Path(args.job), bank)
    counts = Counter(r['outcome_1m'] for r in rows)
    raw = Counter(r['raw_outcome'] for r in rows)
    print(json.dumps(dict(job=str(args.job), conditions=len(rows), outcome_1m=dict(counts),
                          raw_done_reason=dict(raw)), ensure_ascii=False, indent=1))
    if args.write:
        write_json(Path(args.write), dict(conditions=len(rows), rows=rows))


def rejudge(args):
    """T7 helper: 1 m re-judgement of an existing job (no identity.json required)."""
    from scripts.routea_paired_support.episodes import load_commit
    from scripts.f7_analysis import outcome_1m
    job = Path(args.job).resolve()
    bank = read_json(args.bank)
    commits = {p.stem: p for p in (job / 'commits').glob('*.json')}
    rows = []
    for e in bank['episodes']:
        if e['key'] not in commits:
            continue
        h, trace, end = load_commit(job, commits[e['key']])
        goal = np.asarray(h['task']['goal_pos'], np.float32)
        rows.append(dict(task_id=e['task_id'], repeat=e['repeat'],
                         outcome_1m=outcome_1m(trace, goal, end['done_reason']),
                         raw_outcome=end['done_reason'],
                         min_goal_distance_m=end['trajectory_metrics']['min_goal_distance_m']))
    rows.sort(key=lambda r: (r['task_id'], r['repeat']))
    entered_late = [r for r in rows
                    if r['raw_outcome'] in ('collision', 'ground_crash', 'crash_ground')
                    and r['outcome_1m'] == 'reached']
    collision_before_entry = [r for r in rows
                              if r['raw_outcome'] in ('collision', 'ground_crash', 'crash_ground')
                              and r['outcome_1m'] != 'reached']
    raw_reached = [r for r in rows if r['raw_outcome'] == 'reached' and r['outcome_1m'] != 'reached']
    report = dict(job=str(job), conditions=len(rows),
                  reached_1m=sum(r['outcome_1m'] == 'reached' for r in rows),
                  reached_raw=sum(r['raw_outcome'] == 'reached' for r in rows),
                  raw_reached_outside_1m=raw_reached[:3],
                  entered_1m_then_collided=entered_late[:3],
                  collision_before_1m_examples=collision_before_entry[:3],
                  collision_before_1m=len(collision_before_entry))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.write:
        write_json(Path(args.write), dict(report=report, rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('output', 'weights-verified', 'dev-bank', 'val-bank', 'source-root', 'checkpoint', 'repo'):
        p.add_argument('--' + name, required=True)
    p = sub.add_parser('worker')
    p.add_argument('--output', required=True)
    p.add_argument('--policy', choices=sorted(POLICY_BASE), required=True)
    p.add_argument('--guidance', choices=sorted(GUIDANCE), required=True)
    p.add_argument('--split', choices=['dev', 'val'], required=True)
    p.add_argument('--eta', type=float)
    p.add_argument('--consistency', type=int)
    p.add_argument('--tag')
    p.add_argument('--limit', type=int)
    p.add_argument('--full-run-approved', action='store_true')
    p = sub.add_parser('collect')
    p.add_argument('--output', required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--split', choices=['dev', 'val'], required=True)
    p.add_argument('--write')
    p = sub.add_parser('rejudge')
    p.add_argument('--job', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--write')
    args = parser.parse_args()
    with torch.jit.optimized_execution(False):
        globals()[args.command](args)


if __name__ == '__main__':
    main()
