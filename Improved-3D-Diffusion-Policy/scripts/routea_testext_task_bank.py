"""Paper C1-C8: geometry-only selection, frozen observations and append provenance."""
import argparse
import copy
from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.routea_paired_support.core import canonical_hash, file_hash, read_json, write_json

CATEGORIES = ('zero', 'low', 'high')


def episode_payload(row):
    return dict(episode_id='%s__ext%04d' % (row['world'], row['candidate']),
                world_name=row['world'], start_pos=row['start'], goal_pos=row['goal'],
                start_yaw=row['start_yaw'], z_range=[1.0, 5.0],
                disturb_enable=False, role=row['role'])


def category_of(task, world):
    from scripts.routea_paired_support.task_geometry_metrics import corridor_density
    from scripts.analyze_routea_f5 import labels
    rho = corridor_density(dict(start=task['payload']['start_pos'],
                                goal=task['payload']['goal_pos'], geometry=world['geometry']), 1.)['rho']
    return labels(dict(**{'rho_1.0': rho, 'rho_2.0': 0.}, b_band_m=None,
                       kappa=None, band_rho=0., blocked=False))['w1']


def bank_check(path):
    bank = read_json(path)
    if canonical_hash({k: v for k, v in bank.items() if k != 'content_hash'}) != bank['content_hash']:
        raise ValueError('bank content changed: '+str(path))
    return bank


def balanced_selection(candidate, test, old):
    """C6/C7: use only the already frozen 400/world pool and geometry."""
    if len(candidate['worlds']) != 30 or len(candidate['candidates']) != 12000:
        raise ValueError('this delivery is for the verified C1 30-world/12000 pool')
    if {canonical_hash(episode_payload(r)) for r in candidate['selected']} != {t['task_id'] for t in old['tasks']}:
        raise ValueError('C1 selection is not the height-corrected old bank')
    baseline, occupied = Counter(), {}
    for bank in (test, old):
        for task in bank['tasks']:
            cat = category_of(task, bank['worlds'][task['world_id']])
            baseline[cat] += 1
            for field in ('start_pos', 'goal_pos'):
                occupied.setdefault(task['world_id'], []).append(task['payload'][field])
    if dict(baseline) != dict(zero=59, low=25, high=24):
        raise ValueError('original O category counts differ from the registered 59/25/24')
    old_ids = {(r['world'], r['candidate']) for r in candidate['selected']}
    eligible, excluded = [], []
    for row in candidate['candidates']:
        reason = None
        if row['category'] == 'zero':
            reason = 'sparse_not_supplemented'
        elif (row['world'], row['candidate']) in old_ids:
            reason = 'already_selected_run02'
        elif any(math.dist(point, prior) < 1. for point in (row['start'], row['goal'])
                 for prior in occupied.get(row['world'], [])):
            reason = 'endpoint_within_1m_of_original_O'
        if reason:
            excluded.append(dict(world=row['world'], candidate=row['candidate'], reason=reason))
        else:
            eligible.append(row)
    counts = Counter((t['world_id'], t['category']) for t in old['tasks'])
    available = Counter((r['world'], r['category']) for r in eligible)
    feasibility = [dict(world=w, category=c, candidates=available[w,c],
                        retained=counts[w,c], capacity=max(0, 5-counts[w,c]),
                        usable=min(available[w,c], max(0, 5-counts[w,c])))
                   for w in sorted(candidate['worlds']) for c in ('low', 'high')]
    selected = []
    shortfall = {}
    for cat in ('low', 'high'):
        worlds = sorted(candidate['worlds'])
        random.Random(2026110502).shuffle(worlds)
        pools = {w: sorted((r for r in eligible if r['world'] == w and r['category'] == cat),
                           key=lambda r: r['candidate']) for w in worlds}
        offsets = Counter()
        needed = 70-baseline[cat]
        added = 0
        while added < needed:
            progressed = False
            for world in worlds:
                if added == needed:
                    break
                if counts[world, cat] >= 5 or offsets[world] >= len(pools[world]):
                    continue
                row = pools[world][offsets[world]]
                selected.append(row)
                offsets[world] += 1
                counts[world, cat] += 1
                added += 1
                progressed = True
            if not progressed:
                break
        if added != needed:
            shortfall[cat] = dict(needed=needed, selected=added)
    # Recompute labels only for the selected rows; no policy outcome is an input.
    for row in selected:
        task = dict(payload=episode_payload(row))
        if category_of(task, candidate['worlds'][row['world']]) != row['category']:
            raise ValueError('selected category does not reproduce C1')
    payload = dict(schema='routea_testext_candidates_v2_additions',
        candidate_source_hash=candidate['content_hash'], original_O_counts=dict(baseline),
        worlds=candidate['worlds'], candidates=eligible, selected=selected,
        excluded=excluded, counts=dict(Counter(r['category'] for r in selected)),
        feasibility=feasibility,
        shortfall=shortfall, sampling='reuse C1 1--400 only; no continuation needed for this pool',
        exclusion='each new endpoint >=1m from every original O endpoint in the same world',
        selection_seed=2026110502, per_world_class_cap_including_old=5)
    payload['content_hash'] = canonical_hash(payload)
    return payload


def balance_inputs(plan):
    loaded = {}
    for name, spec in plan['inputs'].items():
        choices = [Path(p) for p in spec['paths'] if Path(p).is_file()]
        path = next((p for p in choices if file_hash(p) == spec['file_sha256']), None)
        if path is None:
            raise ValueError('missing exact input: '+name)
        value = bank_check(path) if name != 'review' else read_json(path)
        if name != 'review' and value['content_hash'] != spec['content_hash']:
            raise ValueError('wrong semantic input: '+name)
        loaded[name] = (path, value)
    candidate = loaded['candidates'][1]
    review = loaded['review'][1]
    expected = {w['world_id'] for w in review['worlds'] if w['eligible']}
    if expected != set(candidate['worlds']) or candidate['review_sha256'] != file_hash(loaded['review'][0]):
        raise ValueError('candidate world pool does not match C0 admission')
    return loaded


def balance(args):
    plan = read_json(args.plan)
    loaded = balance_inputs(plan)
    payload = balanced_selection(loaded['candidates'][1], loaded['test'][1], loaded['old'][1])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'additions.json', payload)
    write_json(out/'inputs.json', {k: dict(path=str(p), sha256=file_hash(p)) for k, (p, _) in loaded.items()})
    if payload['shortfall']:
        raise ValueError('C7 quota shortfall; preserve evidence, do not render or relax constraints')
    print('C6_C7_PASS', len(payload['selected']), payload['counts'], payload['content_hash'])


def merge_banks(old, added):
    bank = copy.deepcopy(old)
    for uid, world in added['worlds'].items():
        if uid in bank['worlds']:
            keep = lambda w: {k: v for k, v in w.items() if k != 'obstacle_path'}
            if keep(bank['worlds'][uid]) != keep(world):
                raise ValueError('shared world changed: '+uid)
        else:
            bank['worlds'][uid] = copy.deepcopy(world)
    for field, key in [('tasks', 'task_id'), ('episodes', 'key')]:
        before = {r[key] for r in bank[field]}
        extra = {r[key] for r in added[field]}
        if len(before) != len(bank[field]) or len(extra) != len(added[field]) or before & extra:
            raise ValueError('duplicate retained/additional '+field)
        bank[field].extend(copy.deepcopy(added[field]))
    bank['candidate_content_hash'] = added['candidate_content_hash']
    bank['component_bank_hashes'] = dict(retained=old['content_hash'], added=added['content_hash'])
    bank['version'] = 2
    bank.pop('content_hash')
    bank['content_hash'] = canonical_hash(bank)
    return bank


def draw_candidate(rng):
    start = [rng.uniform(-14, 14), rng.uniform(-14, 14), rng.uniform(1.3, 4.7)]
    goal = [rng.uniform(-14, 14), rng.uniform(-14, 14), rng.uniform(1.5, 4.5)]
    yaw = math.atan2(goal[1] - start[1], goal[0] - start[0]) + rng.uniform(-.3, .3)
    return start, goal, yaw


def select_tasks(rows):
    worlds = sorted({r['world'] for r in rows})
    random.Random(2026110302).shuffle(worlds)
    selected, counts, totals = [], Counter(), Counter()
    for round_index in range(2):
        for world in worlds:
            for category in CATEGORIES:
                pool = [r for r in rows if r['world'] == world and r['category'] == category]
                if round_index < len(pool) and counts[category] < 20 and totals[world] < 4:
                    selected.append(pool[round_index])
                    counts[category] += 1
                    totals[world] += 1
    return selected


def candidates(args):
    from scripts.routea_task_bank import admit, ROLE_CL
    from scripts.routea_paired_support.task_geometry_metrics import corridor_density
    from scripts.analyze_routea_f5 import labels
    from diffusion_policy_3d.env.collision_geom import load_obstacles
    review = read_json(args.review)
    worlds = sorted((w for w in review['worlds'] if w['eligible']), key=lambda w: w['world_id'])
    if len(worlds) < 20 or len({w['world_id'] for w in worlds}) != len(worlds):
        raise ValueError('C0 requires at least 20 distinct reviewed eligible worlds')
    rows, rejected, assets = [], [], {}
    for index, world in enumerate(worlds):
        if not world.get('evidence') or world['role'] not in ROLE_CL:
            raise ValueError('missing H10 evidence or valid role: ' + world['world_id'])
        source = Path(world['obstacle_path']).resolve()
        obstacles = load_obstacles(str(source))
        geometry = [asdict(o) for o in obstacles]
        assets[world['world_id']] = dict(world, obstacle_path=str(source),
                                         obstacle_sha256=file_hash(source), geometry=geometry)
        rng = random.Random(20261103 + index)
        accepted, number = 0, 0
        while accepted < 400 and number < 50 * 400:
            number += 1
            start, goal, yaw = draw_candidate(rng)
            row = dict(world=world['world_id'], role=world['role'], candidate=number,
                       start=start, goal=goal, start_yaw=yaw, sampling_seed=20261103 + index)
            if not admit(start, goal, obstacles, world['role']):
                rejected.append(row)
                continue
            task = dict(row, geometry=geometry)
            rho1, rho2 = (corridor_density(task, width)['rho'] for width in (1., 2.))
            category = labels(dict(**{'rho_1.0': rho1, 'rho_2.0': rho2},
                                   b_band_m=None, kappa=None, band_rho=0., blocked=False))['w1']
            rows.append(dict(row, category=category, rho_1_0=rho1, rho_2_0=rho2))
            accepted += 1
        if accepted != 400:
            raise ValueError('candidate budget exhausted for ' + world['world_id'])
    selected = select_tasks(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    payload = dict(schema='routea_testext_candidates_v1', review_sha256=file_hash(args.review),
                   admitted_candidates_per_world=400, max_attempts_per_world=20000,
                   index_origin=0, worlds=assets,
                   candidates=rows, rejected=rejected, selected=selected,
                   counts=dict(Counter(r['category'] for r in selected)))
    payload['content_hash'] = canonical_hash(payload)
    write_json(out / 'candidates.json', payload)
    print('CANDIDATES', len(selected), payload['counts'], payload['content_hash'])


def freeze(args):
    from scripts.run_routea_xk_evaluation import native_system
    # Strict native restore happens before importing the environment-dependent helpers.
    system, config, header = native_system(args)
    from scripts.evaluate_dppo_uav_formal_endpoint import canonical_episode_config, reset_fixed_episode
    from scripts.routea_paired_support.runtime import make_adapter, metadata
    from scripts.routea_paired_support.core import observation_hash, write_npz
    from scripts.routea_paired_support.bank import episode_key
    payload = read_json(args.candidates)
    if canonical_hash({k: v for k, v in payload.items() if k != 'content_hash'}) != payload['content_hash']:
        raise ValueError('candidate content changed')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    bank = dict(schema='routea_fixed_split_v1', split='rl_test_ext', repeats=5,
                worlds={}, tasks=[], episodes=[], candidate_content_hash=payload['content_hash'])
    for row in payload['selected']:
        uid, role = row['world'], row['role']
        if uid not in bank['worlds']:
            asset = payload['worlds'][uid]
            if file_hash(asset['obstacle_path']) != asset['obstacle_sha256']:
                raise ValueError('world changed after C0/C1: ' + uid)
            copied = out / 'obstacles' / (uid + '.json')
            copied.parent.mkdir(exist_ok=True)
            shutil.copy2(asset['obstacle_path'], copied)
            bank['worlds'][uid] = dict(world_id=uid, role=role, geometry=asset['geometry'],
                obstacle_path=str(copied.resolve()), obstacle_copy=copied.relative_to(out).as_posix(),
                obstacle_sha256=file_hash(copied))
        world = bank['worlds'][uid]
        fixed = canonical_episode_config(episode_payload(row))
        bank['tasks'].append(dict(task_id=fixed.config_hash, world_id=uid, role=role,
                                 payload=fixed.payload, category=row['category']))
        adapter = make_adapter(system, config, world, role)
        try:
            adapter.environment._set_world(uid, world['obstacle_path'])
            world['spawn_offset'] = adapter.environment._spawn_offset.tolist()
            for repeat in range(1, 6):
                seed = (fixed.seed if repeat == 1 else int(canonical_hash(
                    ['xk-fixed-reset-v1', fixed.config_hash, repeat])[:8], 16))
                obs, ctx = reset_fixed_episode(adapter.environment, fixed.payload, seed=seed,
                                                reset_callable=adapter.reset)
                obs = {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}
                key = episode_key(fixed.config_hash, repeat)
                path = out / 'initial' / (key + '.npz')
                write_npz(path, obs)
                bank['episodes'].append(dict(key=key, task_id=fixed.config_hash, repeat=repeat,
                    reset_seed=seed, observation_hash=observation_hash(obs),
                    initial_hash=canonical_hash([fixed.payload, seed, observation_hash(obs)]),
                    initial_npz=path.relative_to(out).as_posix(), initial_npz_sha256=file_hash(path),
                    initial_pointcloud_metadata=metadata(ctx)))
        finally:
            adapter.environment.close()
    bank['content_hash'] = canonical_hash(bank)
    write_json(out / 'bank.json', bank)
    print('BANK_FROZEN', len(bank['tasks']), len(bank['episodes']), bank['content_hash'])
    return dict(runtime_identity_hash=header['runtime_identity_hash'],
                state_sha256=file_hash(Path(args.checkpoint)/'state.pt'),
                render_device=str(system.bc_runtime.device), policy_evaluations=0)


def check_scripts(plan):
    for relative, expected in plan['scripts'].items():
        if file_hash(Path(plan['repo'])/relative) != expected:
            raise ValueError('existing helper changed; do not overwrite it: '+relative)


def copy_bank_assets(bank, source, target):
    paths = [(e['initial_npz'], e['initial_npz_sha256']) for e in bank['episodes']]
    paths += [(w['obstacle_copy'], w['obstacle_sha256']) for w in bank['worlds'].values()]
    for relative, expected in paths:
        original, dest = Path(source)/relative, Path(target)/relative
        if file_hash(original) != expected:
            raise ValueError('frozen asset changed: '+str(original))
        if dest.exists():
            if file_hash(dest) != expected:
                raise ValueError('asset collision: '+str(dest))
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, dest)
        if file_hash(dest) != expected:
            raise ValueError('asset copy differs: '+str(dest))


def verify_arrays(bank, folder):
    from scripts.routea_paired_support.core import observation_hash
    tasks = {t['task_id']: t for t in bank['tasks']}
    pairs = {(e['task_id'], e['repeat']) for e in bank['episodes']}
    if len(tasks) != len(bank['tasks']) or len(pairs) != len(bank['episodes']):
        raise ValueError('duplicate task or episode')
    if pairs != {(t, r) for t in tasks for r in range(1, 6)}:
        raise ValueError('repeat product differs from 1--5')
    for e in bank['episodes']:
        if e['key'] != canonical_hash([e['task_id'], e['repeat']])[:24]:
            raise ValueError('episode key differs')
        expected_seed = (int(e['task_id'][:8], 16) if e['repeat'] == 1 else int(canonical_hash(
            ['xk-fixed-reset-v1', e['task_id'], e['repeat']])[:8], 16))
        if e['reset_seed'] != expected_seed:
            raise ValueError('episode seed differs')
        path = Path(folder)/e['initial_npz']
        if file_hash(path) != e['initial_npz_sha256']:
            raise ValueError('initial NPZ bytes changed')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'point_cloud', 'agent_pos'}:
                raise ValueError('unexpected initial arrays')
            obs = {k: archive[k] for k in archive.files}
        for key, shape in [('point_cloud', (8, 4096, 3)), ('agent_pos', (8, 9))]:
            if obs[key].dtype != np.float32 or obs[key].shape != shape or not np.isfinite(obs[key]).all():
                raise ValueError('initial array contract differs: '+key)
        if observation_hash(obs) != e['observation_hash'] or canonical_hash([
                tasks[e['task_id']]['payload'], e['reset_seed'], e['observation_hash']]) != e['initial_hash']:
            raise ValueError('observation/initial identity differs')
        if len(e['initial_pointcloud_metadata']) != 8:
            raise ValueError('expected eight observation-history metadata frames')
    for world in bank['worlds'].values():
        if file_hash(Path(folder)/world['obstacle_copy']) != world['obstacle_sha256']:
            raise ValueError('copied obstacle changed')


def append(args):
    from scripts.routea_paired_support.core import verify_checkpoint
    plan = read_json(args.plan)
    check_scripts(plan)
    loaded = balance_inputs(plan)
    selection = bank_check(args.selection)
    expected = balanced_selection(loaded['candidates'][1], loaded['test'][1], loaded['old'][1])
    if selection != expected or selection['shortfall'] or selection['counts'] != dict(low=45, high=46):
        raise ValueError('C7 selection differs from deterministic replay')
    model = plan['model']
    source = Path(model['source_root'])
    for row in plan['source_files']:
        if file_hash(source/row['relative_path']) != row['sha256']:
            raise ValueError('P6 native rendering source changed: '+row['relative_path'])
    # H3 pins the P6 state/update/runtime. Also verify the checkpoint's committed header digest.
    header = Path(model['checkpoint'])/'header.json'
    verify_checkpoint(model['checkpoint'], dict(model, header_sha256=file_hash(header)))
    for world in loaded['candidates'][1]['worlds'].values():
        if file_hash(world['obstacle_path']) != world['obstacle_sha256']:
            raise ValueError('C0 live obstacle changed: '+world['world_id'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'inputs_before.json', {k: dict(path=str(p), sha256=file_hash(p)) for k, (p, _) in loaded.items()})
    shutil.copy2(args.selection, out/'additions.json')
    rendering = freeze(argparse.Namespace(repo=plan['repo'], source_root=model['source_root'],
        checkpoint=model['checkpoint'], expected_update=256, candidates=str(out/'additions.json'),
        output=str(out/'incremental')))
    if not rendering['render_device'].startswith('cuda'):
        raise ValueError('use the original CUDA rendering backend; no CPU/GPU hash substitution')
    write_json(out/'render_provenance.json', rendering)
    old, added = loaded['old'][1], bank_check(out/'incremental/bank.json')
    bank = merge_banks(old, added)
    full = out/'bank'
    full.mkdir()
    copy_bank_assets(old, loaded['old'][0].parent, full)
    copy_bank_assets(added, out/'incremental', full)
    write_json(full/'bank.json', bank)
    sources = []
    for name, path, component, public_set in [
            ('test12', loaded['test'][0], loaded['test'][1], 'test12'),
            ('retained_run02', loaded['old'][0], old, 'extension_v2'),
            ('added', out/'incremental/bank.json', added, 'extension_v2')]:
        for e in component['episodes']:
            sources.append(dict(dataset=public_set, key=e['key'], task_id=e['task_id'],
                repeat=e['repeat'], initial_hash=e['initial_hash'], component=name,
                source_bank_path=str(path), source_bank_hash=component['content_hash']))
    manifest = dict(schema='routea_O_prime_source_index_v1', test_bank_hash=loaded['test'][1]['content_hash'],
                    extension_v2_hash=bank['content_hash'], conditions=sources,
                    note='Old raw traces retain their original bank_hash; append is not a rewrite of old evidence.')
    manifest['content_hash'] = canonical_hash(manifest)
    write_json(out/'O_prime_index.json', manifest)
    check_scripts(plan)
    for row in plan['source_files']:
        if file_hash(source/row['relative_path']) != row['sha256']:
            raise ValueError('native source changed while rendering')
    verify_append(args, plan=plan)
    print('C8_APPEND_FROZEN', len(bank['tasks']), len(bank['episodes']), bank['content_hash'])


def verify_append(args, plan=None):
    plan = plan or read_json(args.plan)
    loaded = balance_inputs(plan)
    out = Path(args.output)
    old, added, full = loaded['old'][1], bank_check(out/'incremental/bank.json'), bank_check(out/'bank/bank.json')
    replay = balanced_selection(loaded['candidates'][1], loaded['test'][1], old)
    if replay != bank_check(out/'additions.json') or full != merge_banks(old, added):
        raise ValueError('selection or full bank differs from replay')
    if full['tasks'][:60] != old['tasks'] or full['episodes'][:300] != old['episodes']:
        raise ValueError('old conditions changed')
    if len(added['tasks']) != 91 or len(added['episodes']) != 455:
        raise ValueError('addition denominator differs')
    if len(full['tasks']) != 151 or len(full['episodes']) != 755:
        raise ValueError('extension v2 denominator differs')
    for bank, folder in [(added, out/'incremental'), (full, out/'bank')]:
        verify_arrays(bank, folder)
    verify_arrays(old, loaded['old'][0].parent)
    counts = Counter(category_of(t, loaded['test'][1]['worlds'][t['world_id']]) for t in loaded['test'][1]['tasks'])
    counts.update(category_of(t, full['worlds'][t['world_id']]) for t in full['tasks'])
    if dict(counts) != dict(zero=59, low=70, high=70):
        raise ValueError('O-prime class quotas differ')
    caps = Counter((t['world_id'], category_of(t, full['worlds'][t['world_id']])) for t in full['tasks'])
    if any(n > 5 for (_, cat), n in caps.items() if cat in ('low', 'high')):
        raise ValueError('extension per-world/class cap exceeded')
    index = bank_check(out/'O_prime_index.json')
    expected = {(e['key'], e['initial_hash'], b['content_hash']) for b in (loaded['test'][1], old, added)
                for e in b['episodes']}
    actual = {(e['key'], e['initial_hash'], e['source_bank_hash']) for e in index['conditions']}
    if (expected != actual or len(index['conditions']) != 995
            or len({e['key'] for e in index['conditions']}) != 995):
        raise ValueError('O-prime source provenance differs')
    report = dict(status='passed', old_tasks_unchanged=60, old_episodes_unchanged=300,
        added_tasks=91, added_episodes=455, extension_tasks=151, extension_episodes=755,
        O_prime_tasks=199, O_prime_episodes=995, task_categories=dict(counts),
        extension_v2_hash=full['content_hash'], incremental_hash=added['content_hash'],
        O_prime_index_hash=index['content_hash'], review_required=True, policies_evaluated=0)
    report_path = out/'verification.json'
    if report_path.exists():
        if read_json(report_path) != report:
            raise ValueError('stored verification changed')
    else:
        write_json(report_path, report)
    print('C8_VERIFY_PASS', 995, dict(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    candidate = sub.add_parser('candidates')
    candidate.add_argument('--review', required=True)
    candidate.add_argument('--output', required=True)
    candidate.set_defaults(func=candidates)
    frozen = sub.add_parser('freeze')
    for key in ('candidates', 'repo', 'source-root', 'checkpoint', 'output'):
        frozen.add_argument('--' + key, required=True)
    frozen.add_argument('--expected-update', type=int, required=True)
    frozen.set_defaults(func=freeze)
    for name, function in [('balance', balance), ('append', append), ('verify-append', verify_append)]:
        command = sub.add_parser(name)
        command.add_argument('--plan', required=True)
        command.add_argument('--output', required=True)
        if name == 'append':
            command.add_argument('--selection', required=True)
        command.set_defaults(func=function)
    args = parser.parse_args()
    if args.command in ('freeze', 'append'):
        import torch
        # Match the common worker before native loading or the first scripted ray call.
        with torch.jit.optimized_execution(False):
            args.func(args)
    else:
        args.func(args)


if __name__ == '__main__':
    main()
