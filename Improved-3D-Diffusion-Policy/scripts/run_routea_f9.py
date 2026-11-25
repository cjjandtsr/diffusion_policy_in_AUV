"""F9 v1.3 stages; reuse canonical PPO training and formal episode recording."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.routea_paired_support.core import canonical_hash, file_hash, read_json, write_json, write_npz

SETTINGS = [dict(d_safe=d, eta=e, M=m) for d in (.6, .9, 1.2) for e, m in ((.2, 2), (.4, 4))]
SEEDS = (202611081, 202611082, 202611213)
# Stage-A and restart attempts have their OWN policy identity; P24-27 retain the proposed B numbering.
RUNS = {'P28': ('A', SEEDS[0], True, 1), 'P29': ('A', SEEDS[0], False, 1),
        'P24': ('B', SEEDS[0], True, 1), 'P25': ('B', SEEDS[1], True, 1),
        'P26': ('B', SEEDS[2], True, 1), 'P27': ('B', SEEDS[2], False, 1),
        'P30': ('B', SEEDS[1], False, 1), 'P31': ('A', SEEDS[0], True, 2),
        'P32': ('B', SEEDS[0], True, 2), 'P33': ('B', SEEDS[1], True, 2),
        'P34': ('B', SEEDS[2], True, 2),
        # Stage A2: same seed/budget as P28, teacher episodes drawn from the blocked sampler.
        'P35': ('A', SEEDS[0], True, 1),
        # Stage A2 second (final) attempt: full restart from P6 u256, coefficient x3, no recalibration.
        'P36': ('A', SEEDS[0], True, 2)}


def csv_write(path, rows):
    import csv
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sources():
    # Do not import current model packages before a strict NATIVE restore.
    paths = []
    for directory in ('diffusion_policy_3d/rl/dppo_uav_rebuild', 'diffusion_policy_3d/env',
        'third_party/dppo_official_cc7234a/uav_idp3', 'third_party/dppo_official_cc7234a/model',
        'scripts/routea_paired_support'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py','.yaml')]
    paths += [ROOT/p for p in ('scripts/run_routea_f9.py', 'scripts/routea_denoising_guidance.py',
        'scripts/run_routea_xk_evaluation.py', 'scripts/run_routea_a2.py', 'scripts/f7_analysis.py',
        'scripts/run_routea_e3g_probe.py', 'scripts/evaluate_dppo_uav_formal_endpoint.py',
        'diffusion_policy_3d/model/common/synth_pointcloud.py')]
    return {p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(set(paths))}


def evaluation_overlay():
    from scripts.run_routea_e3g_probe import overlay_modules
    overlay_modules()
    prefix = 'diffusion_policy_3d.rl.dppo_uav_rebuild.engine.'
    for name in ('selective_keep', 'f9_online'):
        full_name = prefix + name
        path = ROOT/(full_name.replace('.', '/') + '.py')
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)


def inputs(args):
    out = Path(args.run).resolve()
    p = read_json(out/'plan.json')
    assert p['source_files'] == sources(), 'F9 sources changed since prepare'
    for s, item in p['banks'].items():
        assert file_hash(item['path']) == item['sha256'], s
    return out, p


def bank(plan, split):
    b = read_json(plan['banks'][split]['path'])
    for world in b['worlds'].values():
        if not Path(world['obstacle_path']).is_file():
            world['obstacle_path'] = str(Path(plan['banks'][split]['path']).parent/world['obstacle_copy'])
        assert file_hash(world['obstacle_path']) == world['obstacle_sha256']
    return b


def native_p6(plan):
    from scripts.run_routea_xk_evaluation import native_system, verify_source
    s = plan['P6']
    verify_source(s['source_root'], dict(files=s['sources']))
    assert file_hash(Path(s['checkpoint'])/'state.pt') == s['state_sha256']
    return native_system(SimpleNamespace(source_root=s['source_root'], repo=plan['repo'],
        checkpoint=s['checkpoint'], expected_update=256, policy='P6', evaluation_reproduction=False))


class PairPolicy:
    def __init__(self, system, env, identity, setting, *, guided, shadow=True):
        self.system, self.env, self.identity_hash = system, env, identity
        self.setting, self.guided, self.shadow = setting, guided, shadow
        self.use_base_policy = False

    def predict(self, observation, context, *, action_seed):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.f9_online import paired_sample
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle
        if not self.shadow:
            from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy
            from scripts.run_routea_xk_evaluation import load_override
            policy = _E5Policy(self.system,use_base_policy=False,identity_hash=self.identity_hash)
            with load_override(ROOT)(self.system.official_runtime,denoised_clip_value=2.,
                                     min_std_vector=[.1]*11,sampling_mode='deterministic'):
                bundle = policy.predict(observation,context,action_seed=action_seed)
            self.last_provenance = policy.last_provenance
            return bundle
        torch.manual_seed(action_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(action_seed)
        encoded = self.system.observation_adapter.encode(observation, context['pointcloud_meta_history'],
            episode_identity_hash=context['episode_identity_hash'], decision_index=context['decision_index'])
        self.system.observation_adapter.verify(encoded)
        self.last_provenance = encoded.provenance
        start = time.perf_counter()
        cloud = torch.as_tensor(observation['point_cloud'][-1], device=self.system.bc_runtime.device, dtype=torch.float32)
        pair = paired_sample(self.system.official_runtime, encoded.condition,
            self.system.bc_runtime.policy.normalizer['action'].unnormalize, cloud,
            self.env._drone_pos, self.env._drone_vel, float(self.env._drone_yaw), self.setting,
            deterministic=True)
        self.last_probe_metrics = dict(delta_exec=pair['delta_exec'], effective=pair['effective'],
            guided=pair['triggered'], inference_seconds=time.perf_counter()-start)
        return ActionBundle.from_full(pair['full'] if self.guided else pair['base_full'])


def prepare(args):
    out = Path(args.run).resolve(); out.mkdir(parents=True, exist_ok=False)
    diag, f8, probe = Path(args.diag_run).resolve(), Path(args.f8_run).resolve(), Path(args.probe_run).resolve()
    p6 = read_json(diag/'weights_verified.json')['specs']['P6']
    assert p6['state_sha256']=='a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43'
    reference=torch.load(args.reference,map_location='cpu',weights_only=True)
    assert len(reference['actions'])==549 and len(set(reference['task_ids']))==47
    assert file_hash(args.reference)=='8759fc40c20a20c3bd08a5568965f5562c508f3577d8c82fa4c646d6a080ce95'
    banks = {}
    for s, p in [('train', diag/'bank.json'), ('dev', Path(args.dev_bank)), ('val', Path(args.val_bank))]:
        b = read_json(p)
        assert canonical_hash({k:v for k,v in b.items() if k!='content_hash'}) == b['content_hash']
        assert len(b['episodes']) == dict(train=425, dev=420, val=80)[s]
        banks[s] = dict(path=str(p.resolve()), sha256=file_hash(p))
    optional = dict(included=[], reason='P18/P19 excluded: no exact F7/F8 source proof supplied')
    from omegaconf import OmegaConf
    old_config = OmegaConf.load(f8/'prepared/P22.yaml')
    old_checkpoint = Path(args.p22_checkpoint) if args.p22_checkpoint else Path(old_config.output_dir)/'checkpoint_update_000257'
    old_header = read_json(old_checkpoint/'header.json')
    assert old_header['update_index']==257
    assert old_header['runtime_identity_hash']==read_json(f8/'training/P22/fork_lineage.json')['child_runtime_identity_hash']
    if args.f7_prepared:
        a = read_json(args.f7_prepared)['source_files']; b = read_json(f8/'prepared/prepared.json')['source_files']
        a = {r['relative_path']:r['sha256'] for r in a}; b = {r['relative_path']:r['sha256'] for r in b}
        diff = sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
        optional = dict(included=[] if diff else ['P18','P19'], source_differences=diff,
                        f7_prepared=str(Path(args.f7_prepared).resolve()))
    write_json(out/'plan.json', dict(schema='f9_v1_3', source_files=sources(), banks=banks,
        P6=p6, repo=str(Path(args.repo).resolve()), f8=str(f8), diag=str(diag), probe=str(probe),
        reference=str(Path(args.reference).resolve()), weights=str(Path(args.weights).resolve()),
        optional_controls=optional, settings=SETTINGS, policy_runs=RUNS,
        g0_checkpoint=dict(path=str(old_checkpoint.resolve()), state_sha256=file_hash(old_checkpoint/'state.pt'),
                           header_sha256=file_hash(old_checkpoint/'header.json')),
        baseline_failures='F8 D1 P6, same 425-condition bank; reached_1m',
        decisions=dict(prefill_cap='finish current episode, never start another after 4096',
                       missing_r='pending_r; no automatic extension of eight-round stage A')))
    print('F9_PREPARED', {s:len(read_json(x['path'])['episodes']) for s,x in banks.items()})


def step0(args):
    from scripts.routea_paired_support.episodes import record_episode, load_commit
    from scripts.f7_analysis import outcome_1m
    out, plan = inputs(args)
    system, config, header = native_p6(plan)
    evaluation_overlay()
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.f9_online import eligible_indices
    from scripts.routea_paired_support.runtime import make_adapter
    b = bank(plan, 'train'); tasks = {t['task_id']:t for t in b['tasks']}
    group = out/('step0_smoke' if args.limit else 'step0')/('g%d' % args.group)
    group.mkdir(parents=True, exist_ok=False)
    setting = SETTINGS[args.group]
    rows = []
    for e in b['episodes'][:args.limit or None]:
        t = tasks[e['task_id']]; w = b['worlds'][t['world_id']]
        adapter = make_adapter(system, config, w, t['role'])
        policy = PairPolicy(system, adapter.environment, header['runtime_identity_hash'], setting, guided=True)
        if args.group == 0:
            old = Path(plan['probe'])/'runs/grid_c0_eta0.2'
            h, trace, end = load_commit(old, old/'commits'/(e['key']+'.json'))
            from scripts.evaluate_dppo_uav_formal_endpoint import reset_fixed_episode
            obs, context = reset_fixed_episode(adapter.environment, t['payload'], seed=e['reset_seed'], reset_callable=adapter.reset)
            from scripts.routea_paired_support.core import observation_hash
            assert observation_hash(obs)==e['observation_hash']
            metrics = []
            for r in trace:
                sampled = policy.predict(obs, context, action_seed=r['action_seed'])
                assert policy.last_provenance.policy_input_hash==r['observation_provenance']['policy_input_hash']
                assert np.array_equal(sampled.full[0].cpu().numpy(), np.asarray(r['full_action_16x11'],np.float32))
                obs, _, _, _, context = adapter.step(sampled)
                assert np.array_equal(context['executed_path'], np.asarray(r['executed_path'],np.float32))
                metrics.append(policy.last_probe_metrics)
            write_json(group/(e['key']+'.json'), dict(old_commit_sha256=file_hash(old/'commits'/(e['key']+'.json')), metrics=metrics))
            deltas = [m['delta_exec'] for m in metrics]
            source_trace = str(old/read_json(old/'commits'/(e['key']+'.json'))['trace'])
        else:
            record_episode(group, 'F9_g%d' % args.group, policy, adapter, t, e, w, b['content_hash'])
            h, trace, end = load_commit(group, group/'commits'/(e['key']+'.json'))
            deltas = [r['probe_metrics']['delta_exec'] for r in trace]
            source_trace = str(group/read_json(group/'commits'/(e['key']+'.json'))['trace'])
        adapter.environment.close()
        paths = [r['executed_path'] for r in trace]
        ids = eligible_indices(paths, t['payload']['goal_pos'], deltas, [r['done_reason'] for r in trace])
        old_d1 = Path(plan['diag'])/'d1/P6'
        bh, bt, be = load_commit(old_d1, old_d1/'commits'/(e['key']+'.json'))
        assert bh['initial_hash']==e['initial_hash'] and bh['task']==t['payload']
        baseline_failed = outcome_1m(bt, np.asarray(t['payload']['goal_pos']), be['done_reason']) != 'reached'
        metrics = metrics if args.group == 0 else [r['probe_metrics'] for r in trace]
        rows.append(dict(task_id=e['task_id'], repeat=e['repeat'], eligible=len(ids),
            baseline_failed=baseline_failed, first_interventions=int(bool(ids)),
            eligible_decisions=[dict(decision=d,intervention='first_intervention' if j==0 else 'after_intervention',
                P6_condition_failed=baseline_failed,delta_exec=deltas[d]) for j,d in enumerate(ids)],
            reached=outcome_1m(trace,np.asarray(t['payload']['goal_pos']),end['done_reason'])=='reached',
            triggered=sum(m['guided'] for m in metrics), effective=sum(d>=.05 for d in deltas),
            delta_exec=deltas, source_trace=source_trace, source_trace_sha256=file_hash(source_trace)))
        print('F9_STEP0', args.group, len(rows), flush=True)
    write_json(group/'result.json', dict(setting=setting, smoke=bool(args.limit), conditions=len(rows),
        eligible_failed=sum(r['eligible'] for r in rows if r['baseline_failed']),
        eligible_all=sum(r['eligible'] for r in rows), rows=rows))


def select(args):
    out, plan = inputs(args)
    groups = [read_json(out/'step0'/('g%d' % i)/'result.json') for i in range(6)]
    assert all(g['conditions']==425 and not g['smoke'] for g in groups)
    winner = min(groups, key=lambda g:(-g['eligible_failed'], -g['eligible_all'],
        g['setting']['d_safe'], g['setting']['M']))
    write_json(out/'selection.json', dict(passed=winner['eligible_failed']>=100, setting=winner['setting'],
        winner=winner, group_sha256=[file_hash(out/'step0'/('g%d' % i)/'result.json') for i in range(6)]))


def configure(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.a2_fork import validate_child_config
    out, plan = inputs(args)
    selection = read_json(out/'selection.json'); assert selection['passed']
    stage, seed, treatment, attempt = RUNS[args.policy]
    if stage == 'B':
        # Stage A is a separate run, never the start checkpoint of B.
        gates = list((out/'training').glob('P*/gate.json'))
        assert any(read_json(p)['decision']=='pass' and read_json(p)['stage']=='A' for p in gates)
    parent = read_json(Path(plan['f8'])/'P6_receipt.json')
    config = json.loads(json.dumps(parent['training_config']))
    old_reference = config['a2']['parent_receipt_sha256']
    config['run'].update(seed=seed, updates=320, checkpoint_every_updates=8,
        checkpoint_updates=sorted({257}|set(range(0,321,8))|set(config['run'].get('checkpoint_updates',[]))))
    config['optimizer'].update(actor_lr=1e-5, actor_lr_end=1e-5, critic_lr=1e-3)
    config['a2'].update(parent_checkpoint=plan['P6']['checkpoint'],
        parent_receipt=str(Path(plan['f8'])/'P6_receipt.json'),
        parent_receipt_sha256=file_hash(Path(plan['f8'])/'P6_receipt.json'),
        parent_source_root=plan['P6']['source_root'], reference_bank=plan['reference'],
        reference_sha256=file_hash(plan['reference']))
    config['f8'] = dict(run_id=args.policy, bank_path='', bank_sha256='', new_seed=seed+8000,
        keep_coefficient=.01061006086489362, reference_parent_receipt_sha256=old_reference)
    trial = out/'training'/args.policy
    trial.mkdir(parents=True, exist_ok=False)
    config['output_dir'] = str(Path(plan['weights'])/args.policy)
    config['result_dir'] = str(trial)
    override = 0.
    restart_reference = None
    if attempt == 2:
        original = {'P31':'P28','P32':'P24','P33':'P25','P34':'P26','P36':'P35'}[args.policy]
        gate = read_json(out/'training'/original/'gate.json'); assert gate['decision']=='restart'
        calibration = out/'training'/original/'f9_calibration.json'
        override = read_json(calibration)['coefficient'] * 3.
        restart_reference = dict(policy=original, calibration_sha256=file_hash(calibration),
                                 gate_sha256=file_hash(out/'training'/original/'gate.json'))
    config['f9'] = dict(enabled=treatment, stage=stage, attempt=attempt, trial_id=args.policy,
        teacher_task_sampler=('blocked' if args.policy in ('P35','P36') else 'default'),
        **selection['setting'], capacity=4096, batch_size=32, target_ratio=1., effective_threshold=.05,
        coefficient_override=override, restart_reference=restart_reference,
        gate_path=str(trial/'gate.json'), selection_path=str(out/'selection.json'),
        selection_sha256=file_hash(out/'selection.json'), experiment_root=str(out))
    validate_child_config(parent['training_config'], config)
    path = trial/'config.yaml'; OmegaConf.save(OmegaConf.create(config),path)
    write_json(trial/'configured.json', dict(config_sha256=file_hash(path), source_files=plan['source_files'],
        policy=args.policy, stage=stage, attempt=attempt, seed=seed, treatment=treatment))


def train(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import run
    out, plan = inputs(args)
    assert args.approved, 'GPU execution must be arranged by the user'
    trial = out/'training'/args.policy
    assert file_hash(trial/'config.yaml') == read_json(trial/'configured.json')['config_sha256']
    stage, seed, treatment, attempt = RUNS[args.policy]
    if treatment:
        # G-0 is recorded, not gating, for stage A (2026-11-22 ruling). It must exist and be
        # explicitly non-blocking; a missing/blocking record still stops treatment arms.
        g0_record = read_json(out/'g0.json')
        assert g0_record.get('recorded') is True and g0_record.get('blocking') is False
        assert read_json(out/'pair_probe/proof.json')['all_draws_bitwise_equal']
        assert read_json(out/'shadow_probe/passed.json')['passed']
    if args.policy == 'P30': assert args.stop == 264
    if stage == 'A': assert args.stop in (257,264)
    if args.stop == 320 and treatment: assert args.resume
    run(OmegaConf.load(trial/'config.yaml'), ROOT,
        resume_from=Path(args.resume) if args.resume else None, stop_after_update=args.stop)


def _same_values(a,b,path='',ignored=()):
    if isinstance(a,dict):
        assert set(a)-set(ignored) == set(b)-set(ignored), path
        for k in a:
            if k not in ignored: _same_values(a[k],b[k],path+'/'+str(k),ignored)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)): _same_values(x,y,path+'/'+str(i),ignored)
    elif isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),path
    else:
        assert a==b,path


_INDEX = re.compile(r'/\d+(?=/|$)')


def _max_abs(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape:
        return float((a.double()-b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a)-float(b))
    return None


def _collect_diffs(a, b, path='', ignored=(), acc=None):
    """Aggregate differences by index-normalised path; never raises."""
    if acc is None: acc = {}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a)|set(b)):
            if k in ignored: continue
            if k not in a or k not in b:
                e = acc.setdefault(_INDEX.sub('/*', path+'/'+str(k)), dict(kind='missing', count=0))
                e['count'] += 1
                continue
            _collect_diffs(a[k], b[k], path+'/'+str(k), ignored, acc)
        return acc
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            e = acc.setdefault(_INDEX.sub('/*', path), dict(kind='length', count=0))
            e['count'] += 1; e['a'] = len(a); e['b'] = len(b)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _collect_diffs(x, y, path+'/'+str(i), ignored, acc)
        return acc
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        key = _INDEX.sub('/*', path)
        if a.dtype != b.dtype or a.shape != b.shape:
            e = acc.setdefault(key, dict(kind='tensor_shape', count=0)); e['count'] += 1
        elif not torch.equal(a, b):
            e = acc.setdefault(key, dict(kind='tensor', count=0, max_abs=0.))
            e['count'] += 1; e['max_abs'] = max(e['max_abs'], _max_abs(a, b))
        return acc
    if a != b:
        d = _max_abs(a, b)
        e = acc.setdefault(_INDEX.sub('/*', path), dict(kind='value', count=0))
        e['count'] += 1
        if d is not None:
            e['max_abs'] = max(e.get('max_abs') or 0., d)
        else:
            e.setdefault('example', [repr(a)[:120], repr(b)[:120]])
    return acc


def _aggregate_max(acc):
    values = [e['max_abs'] for e in acc.values() if isinstance(e.get('max_abs'), (int, float))]
    return max(values) if values else (0. if not acc else None)


def g0(args):
    """G-0 record: compare P29@u257 against the historical P22@u257 and record, never assert.

    Bitwise equality is not expected for F9: the actor update merges the PPO and keep
    gradients in two autograd passes while P22 used one backward, so only the float
    accumulation order differs. Per the 2026-11-22 ruling the record is not blocking for
    stage A; it does not authorise stage B reuse of P22/P23.
    """
    from omegaconf import OmegaConf
    out, plan = inputs(args)
    assert file_hash(Path(args.p22_checkpoint)/'state.pt')==plan['g0_checkpoint']['state_sha256']
    assert file_hash(Path(args.p22_checkpoint)/'header.json')==plan['g0_checkpoint']['header_sha256']
    current = read_json(out/'training/P29/metrics_update_000257.json')
    original = read_json(Path(plan['f8'])/'training/P22/metrics_update_000257.json')
    ignored = ('config_hash','runtime_identity_hash','source_set_hash')
    a,b = current['rollout_audit']['transition_records'],original['rollout_audit']['transition_records']
    assert len(a)==len(b)==1024
    transition = _collect_diffs(a, b, ignored=ignored)
    differing_records = sum(1 for x,y in zip(a,b) if _collect_diffs(x, y, ignored=ignored))
    assert len(current['updates'])==len(original['updates'])==20
    updates = _collect_diffs(current['updates'], original['updates'])
    config = OmegaConf.load(out/'training/P29/config.yaml')
    now = torch.load(Path(config.output_dir)/'checkpoint_update_000257/state.pt',map_location='cpu',weights_only=True)
    old = torch.load(Path(args.p22_checkpoint)/'state.pt',map_location='cpu',weights_only=True)
    fields = ('lora','critic','actor_optimizer','critic_optimizer','counters')
    state = {f: _collect_diffs(now[f], old[f]) for f in fields}
    state_equal = {f: not state[f] for f in fields}
    passed = (not transition) and (not updates) and all(state_equal.values())
    write_json(out/'g0.json', dict(
        recorded=True, passed=passed, blocking=False,
        compared_against=str(Path(args.p22_checkpoint)),
        old_state_sha256=file_hash(Path(args.p22_checkpoint)/'state.pt'),
        transitions=1024, commits=20,
        compared_state_fields=list(fields), expected_identity_differences=list(ignored),
        transition_equal=not transition, update_records_equal=not updates, state_equal=state_equal,
        differing_transition_records=differing_records,
        max_abs_difference=dict(transitions=_aggregate_max(transition),
                                updates=_aggregate_max(updates),
                                **{f: _aggregate_max(state[f]) for f in fields}),
        transition_differences=transition, update_differences=updates, state_differences=state,
        ruling='g0_not_blocking_stage_a_2026-11-22',
        note=('Two-pass actor gradient (PPO backward then keep autograd.grad) vs P22 single backward: '
              'same mathematics, different float accumulation order. Recorded, not gate-relaxed.')))
    print('F9_G0', dict(passed=passed, differing_transition_records=differing_records,
        transition_fields=sorted(transition)[:10], state_equal=state_equal), flush=True)


def isolation(args):
    out,plan=inputs(args)
    t=read_json(out/'training'/args.policy/'metrics_update_000257.json')
    c=read_json(out/'training'/args.control/'metrics_update_000257.json')
    ti=read_json(out/'training'/args.policy/'configured.json')
    ci=read_json(out/'training'/args.control/'configured.json')
    assert ti['seed']==ci['seed'] and ti['treatment'] and not ci['treatment']
    # Post-update gradient/credit diagnostics SHOULD differ; they are not collected transitions.
    ignored=('runtime_identity_hash','source_set_hash','config_hash','gradient_diagnostics','dppo_credit')
    a,b=t['rollout_audit']['transition_records'],c['rollout_audit']['transition_records']
    assert len(a)==len(b)==1024
    _same_values(a,b,ignored=ignored)
    write_json(out/('random_isolation_'+args.policy+'.json'),dict(passed=True,transitions=1024,
        treatment=args.policy,control=args.control,excluded_fields=list(ignored),
        prefill=read_json(out/'training'/args.policy/'f9_prefill.json')))


def pair_probe(args):
    out,plan=inputs(args);system,config,header=native_p6(plan)
    evaluation_overlay()
    from scripts.routea_paired_support.runtime import make_adapter
    from scripts.evaluate_dppo_uav_formal_endpoint import reset_fixed_episode
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.f9_online import paired_sample,rng_scope
    b=bank(plan,'train');e=b['episodes'][0];t=next(t for t in b['tasks'] if t['task_id']==e['task_id'])
    adapter=make_adapter(system,config,b['worlds'][t['world_id']],t['role'])
    obs,ctx=reset_fixed_episode(adapter.environment,t['payload'],seed=e['reset_seed'],reset_callable=adapter.reset)
    encoded=system.observation_adapter.encode(obs,ctx['pointcloud_meta_history'],
        episode_identity_hash=ctx['episode_identity_hash'],decision_index=ctx['decision_index'])
    env=adapter.environment
    with rng_scope(seed=202611213) as local:
        pair=paired_sample(system.official_runtime,encoded.condition,
            system.bc_runtime.policy.normalizer['action'].unnormalize,
            torch.as_tensor(obs['point_cloud'][-1],dtype=torch.float32,device=system.bc_runtime.device),
            env._drone_pos,env._drone_vel,float(env._drone_yaw),SETTINGS[0],
            deterministic=False,audit_noise=True)
    target=out/'pair_probe';target.mkdir(exist_ok=False)
    torch.save(pair['noise'],target/'noise.pt')
    write_json(target/'proof.json',dict(noise_sha256=file_hash(target/'noise.pt'),
        draws=len(pair['noise'][0]),all_draws_bitwise_equal=True,delta_exec=pair['delta_exec'],
        base_full=pair['base_full'][0].cpu().tolist(),guided_full=pair['full'][0].cpu().tolist(),
        position=env._drone_pos.tolist(),velocity=env._drone_vel.tolist(),yaw=float(env._drone_yaw),
        guidance_identity=pair['guidance_identity']))
    adapter.environment.close()


def shadow_probe(args):
    out,plan=inputs(args)
    paths=[]
    for enabled in (False,True):
        job=out/'shadow_probe'/('on' if enabled else 'off')
        command=[sys.executable,str(Path(__file__).resolve()),'eval-worker','--run',str(out),
            '--policy','P6','--update','256','--split','train','--job',str(job),'--limit','1']
        if not enabled:command+=['--shadow-off']
        run_logged(out,'shadow_probe_'+str(enabled),command)
        paths.append(job)
    entry=read_json(plan['banks']['train']['path'])['episodes'][0]
    traces=[]
    for job in paths:
        receipt=read_json(job/'commits'/(entry['key']+'.json'))
        traces.append([json.loads(s) for s in (job/receipt['trace']).read_text(encoding='utf-8').splitlines()])
    _same_values(traces[0],traces[1],ignored=('probe_metrics',))
    write_json(out/'shadow_probe/passed.json',dict(passed=True,conditions=1,
        comparison='all episode fields except the intentionally different shadow measurements'))


def run_logged(out, name, command):
    directory = Path(out)/'execution'; directory.mkdir(exist_ok=True)
    write_json(directory/(name+'.command.json'),dict(argv=command))
    with (directory/(name+'.log')).open('x',encoding='utf-8') as stream:
        result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    with (directory/(name+'.rc')).open('x',encoding='utf-8') as stream: stream.write(str(result.returncode)+'\n')
    assert result.returncode==0, name


def load_actor(plan, policy, update):
    if policy=='P6': return native_p6(plan)
    from scripts.run_routea_xk_evaluation import native_system
    return native_system(SimpleNamespace(source_root=str(ROOT),repo=plan['repo'],
        checkpoint=str(Path(plan['weights'])/policy/('checkpoint_update_%06d'%update)),
        expected_update=update,policy=policy,evaluation_reproduction=False))


def eval_worker(args):
    out, plan = inputs(args)
    system, config, h = load_actor(plan,args.policy,args.update)
    evaluation_overlay()
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.f9_online import demand_counts
    from scripts.routea_paired_support.runtime import make_adapter
    from scripts.routea_paired_support.episodes import record_episode,load_commit
    from scripts.f7_analysis import first_strict_entry,outcome_1m
    from scripts import routea_paired_support
    if args.split=='val':
        from scripts import evaluate_dppo_uav_formal_endpoint as endpoint
        from scripts.routea_paired_support import episodes
        episodes.action_seed=lambda tid,rep,dec:endpoint._action_seed(tid,dec)
    b=bank(plan,args.split); tasks={t['task_id']:t for t in b['tasks']}
    job=Path(args.job);job.mkdir(parents=True,exist_ok=False)
    setting=read_json(out/'selection.json')['setting']
    counts=Counter(); numerator=denominator=0; table=[]
    for e in b['episodes'][:args.limit or None]:
        t=tasks[e['task_id']]; w=b['worlds'][t['world_id']]
        adapter=make_adapter(system,config,w,t['role'])
        policy=PairPolicy(system,adapter.environment,h['runtime_identity_hash'],setting,
                          guided=args.guided,shadow=not args.shadow_off)
        record_episode(job,args.policy,policy,adapter,t,e,w,b['content_hash'])
        header,trace,end=load_commit(job,job/'commits'/(e['key']+'.json'))
        adapter.environment.close()
        goal=np.asarray(t['payload']['goal_pos'],np.float32)
        n,den,dstar=demand_counts([r['executed_path'] for r in trace],goal,
                                 [r.get('probe_metrics',{}).get('delta_exec',0.) for r in trace])
        numerator+=n;denominator+=den
        result=outcome_1m(trace,goal,end['done_reason']);counts[result]+=1
        commit=read_json(job/'commits'/(e['key']+'.json'))
        table.append(dict(task_id=e['task_id'],repeat=e['repeat'],outcome=result,D_num=n,D_den=den,
            dstar=dstar,delta_exec=[r.get('probe_metrics',{}).get('delta_exec') for r in trace],
            trace=commit['trace'],trace_sha256=commit['sha256']))
        print('F9_EVAL',args.policy,len(table),flush=True)
    write_json(job/'result.json',dict(policy=args.policy,update=args.update,split=args.split,
        conditions=len(table),D_num=numerator,D_den=denominator,D=None if not denominator else numerator/denominator,
        outcomes=dict(counts),guided=args.guided,shadow_off=args.shadow_off,smoke=bool(args.limit),rows=table,
        state_sha256=file_hash((Path(plan['P6']['checkpoint']) if args.policy=='P6' else Path(plan['weights'])/args.policy/('checkpoint_update_%06d'%args.update))/'state.pt'),
        bank_sha256=plan['banks'][args.split]['sha256'],source_files=plan['source_files']))


def freeze_holdout(args):
    out, plan=inputs(args)
    state=torch.load(Path(plan['weights'])/args.policy/('checkpoint_update_%06d'%args.update)/'state.pt',
                     map_location='cpu',weights_only=True)
    schedule=state['schedules']['f9_session']; snapshot=Path(schedule['snapshot'])
    assert file_hash(snapshot)==schedule['snapshot_sha256']
    data=torch.load(snapshot,map_location='cpu',weights_only=True)
    records=data['heldout']
    assert all(r['episode_number']%10==0 for r in records)
    assert not {r['sample_id'] for r in records}&{r['sample_id'] for r in data['records']}
    target=out/'measure'/args.policy/('u%06d'%args.update)/'offline';target.mkdir(parents=True,exist_ok=False)
    torch.save(records,target/'records.pt')
    grid=read_json(Path(plan['diag'])/'grid.json')
    write_json(target/'grid.json',grid)
    gen=torch.Generator().manual_seed(2026111901)
    torch.save(torch.randn((len(records),len(grid['timesteps']),16,11),generator=gen,dtype=torch.float32),target/'noise.pt')
    write_json(target/'frozen.json',dict(decisions=len(records),adequate=len(records)>=20,
        records_sha256=file_hash(target/'records.pt'),noise_sha256=file_hash(target/'noise.pt'),
        grid_sha256=file_hash(target/'grid.json'),snapshot_sha256=file_hash(snapshot)))


def prefix_worker(args):
    out,plan=inputs(args)
    target=out/'measure'/args.treatment/('u%06d'%args.update)/'offline'
    frozen=read_json(target/'frozen.json')
    for key,name in [('records','records.pt'),('noise','noise.pt'),('grid','grid.json')]:
        assert file_hash(target/name)==frozen[key+'_sha256']
    system,config,header=load_actor(plan,args.policy,256 if args.policy=='P6' else args.update)
    evaluation_overlay()
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.selective_keep import prefix_x0_loss,executed_weights
    from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import encode_e5_condition_with_normalized
    records=torch.load(target/'records.pt',map_location='cpu',weights_only=True)
    noise=torch.load(target/'noise.pt',map_location='cpu',weights_only=True)
    grid=read_json(target/'grid.json'); runtime=system.official_runtime;device=system.bc_runtime.device
    assert runtime.ddim_t[-runtime.ft_denoising_steps:].cpu().tolist()==grid['timesteps']
    assert torch.equal(runtime.ddim_alphas[-runtime.ft_denoising_steps:].cpu(),torch.tensor(grid['alphas']))
    table=[]
    with torch.no_grad():
        for i,r in enumerate(records):
            assert file_hash(r['observation_path'])==r['observation_sha256']
            with np.load(r['observation_path']) as z:raw={k:z[k][None] for k in z.files}
            bridge=system.observation_adapter.bridge.to_policy_observation(raw,device)
            cond,_=encode_e5_condition_with_normalized(system.observation_adapter.policy,bridge,system.observation_adapter.spec)
            condition=cond['global_cond'].float()
            diff=float((condition.cpu()[0]-r['condition']).abs().max())
            x0=r['action'].unsqueeze(0).to(device); w=executed_weights('none').unsqueeze(0).to(device)
            values=[]
            for j,t in enumerate(grid['timesteps']):
                value=prefix_x0_loss(runtime,x0,condition,w,torch.tensor([t],device=device),
                    torch.tensor([grid['alphas'][j]],device=device),noise[i,j].unsqueeze(0).to(device))
                values.append(float(value))
            assert np.isfinite(values).all()
            table.append(dict(sample_id=r['sample_id'],intervention=r['intervention'],loss=float(np.mean(values)),
                              timestep_losses=values,condition_max_difference=diff))
    means={kind:float(np.mean([r['loss'] for r in table if kind=='all' or r['intervention']==kind]))
           for kind in ('all','first_intervention','after_intervention')
           if any(kind=='all' or r['intervention']==kind for r in table)}
    write_json(target/(args.policy+'.json'),dict(policy=args.policy,means=means,rows=table,
        noise_sha256=frozen['noise_sha256'],records_sha256=frozen['records_sha256'],
        grid_sha256=frozen['grid_sha256']))


def task_strata(treatment, control, parent):
    """Task solved = >=3/5; strata are frozen by the unguided P6 outcome."""
    maps=[]
    for result in (treatment,control,parent):
        rows=result['rows'];keys=[(r['task_id'],r['repeat']) for r in rows]
        assert len(keys)==425 and len(set(keys))==425
        by_task=defaultdict(list)
        for r in rows:by_task[r['task_id']].append(r)
        assert len(by_task)==85 and all(len(v)==5 for v in by_task.values())
        maps.append({tid:sum(r['outcome']=='reached' for r in rs)>=3 for tid,rs in by_task.items()})
    t,c,p=maps;assert set(t)==set(c)==set(p)
    assert sum(p.values())==49
    output={}
    for solved in (True,False):
        ids=[tid for tid in p if p[tid]==solved]
        output['P6_solved_49' if solved else 'P6_unsolved_36']=dict(tasks=len(ids),
            solved_tasks=dict(T=sum(t[i] for i in ids),C=sum(c[i] for i in ids),P6=sum(p[i] for i in ids)),
            T_vs_C=dict(gained=[i for i in ids if t[i] and not c[i]],lost=[i for i in ids if c[i] and not t[i]]),
            T_vs_P6=dict(gained=[i for i in ids if t[i] and not p[i]],lost=[i for i in ids if p[i] and not t[i]]))
    return output


# Source paths changed by F9 stage A2 (teacher blocked sampler). Only these may differ when a
# stage-A measurement is reused by the control/parent arms; any other difference must stop the run.
A2_REUSE_PATHS = frozenset({
    'diffusion_policy_3d/rl/dppo_uav_rebuild/entrypoints/f9_blocked_sampler.py',
    'diffusion_policy_3d/rl/dppo_uav_rebuild/entrypoints/f9_support.py',
    'diffusion_policy_3d/rl/dppo_uav_rebuild/checkpointing/a2_fork.py',
    'scripts/run_routea_f9.py',
})


def measure(args):
    out,plan=inputs(args)
    assert RUNS[args.policy][2]
    expected_control={SEEDS[0]:'P29',SEEDS[1]:'P30',SEEDS[2]:'P27'}[RUNS[args.policy][1]]
    assert args.control==expected_control
    target=out/'measure'/args.policy/('u%06d'%args.update)
    references={}
    for label,policy in [('T',args.policy),('C',args.control),('P6','P6')]:
        update=args.update if policy!='P6' else 256
        job=out/'evaluations'/policy/('u%06d'%update)/'train_shadow'
        command=[sys.executable,str(Path(__file__).resolve()),'eval-worker','--run',str(out),
            '--policy',policy,'--update',str(update),'--split','train','--job',str(job)]
        if not (job/'result.json').exists():
            run_logged(out,'measure_%s_%s_train'%(policy,update),command)
        result=read_json(job/'result.json')
        checkpoint=(Path(plan['P6']['checkpoint']) if policy=='P6' else Path(plan['weights'])/policy/('checkpoint_update_%06d'%update))
        # 2026-11-24 (Claude-approved): the A2 sampler changes the source identity, so the arms
        # measured in stage A cannot be re-evaluated under the new code (their checkpoints raise
        # IdentityMismatch). Control P29@u264 and parent P6@u256 may therefore reuse their stage-A
        # measurement, but only if the weights are identical and the whole source diff is inside
        # A2_REUSE_PATHS. The treatment arm must still match the current plan exactly.
        state_matches = result['state_sha256']==file_hash(checkpoint/'state.pt')
        identical = result['source_files']==plan['source_files']
        differing = sorted(k for k in set(result['source_files'])|set(plan['source_files'])
            if result['source_files'].get(k)!=plan['source_files'].get(k))
        reused = (label in ('C','P6')) and state_matches and not identical and set(differing)<=A2_REUSE_PATHS
        assert (identical or reused) and state_matches, 'measure identity mismatch: '+label
        references[label]=dict(path=str(job/'result.json'),sha256=file_hash(job/'result.json'),
            source_identity=('current' if identical else 'stageA_215'),
            record_source_files=int(len(result['source_files'])),
            current_source_files=int(len(plan['source_files'])),
            differing_sources=([] if identical else differing))
    target.mkdir(parents=True,exist_ok=True)
    write_json(target/'eval_refs.json',references)
    write_json(target/'task_strata.json',task_strata(*[read_json(references[k]['path']) for k in ('T','C','P6')]))
    freeze_holdout(args)
    offline=target/'offline'
    if read_json(offline/'frozen.json')['adequate']:
        for policy in (args.policy,'P6'):
            command=[sys.executable,str(Path(__file__).resolve()),'prefix-worker','--run',str(out),
                '--policy',policy,'--treatment',args.policy,'--update',str(args.update)]
            run_logged(out,'offline_%s_%s_%s'%(args.policy,args.update,policy),command)


def gate(args):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.f9_online import stage_gate
    out,plan=inputs(args);target=out/'measure'/args.policy/('u%06d'%args.update)
    refs=read_json(target/'eval_refs.json')
    assert all(file_hash(item['path'])==item['sha256'] for item in refs.values())
    t,c,p=[read_json(refs[label]['path']) for label in ('T','C','P6')]
    assert t['policy']==args.policy and p['policy']=='P6'
    assert RUNS[c['policy']][1]==RUNS[args.policy][1] and not RUNS[c['policy']][2]
    assert c['policy']=={SEEDS[0]:'P29',SEEDS[1]:'P30',SEEDS[2]:'P27'}[RUNS[args.policy][1]]
    assert all(x['update']==264 for x in (t,c)) and p['update']==256
    assert all(r['conditions']==425 and not r['guided'] and not r['smoke'] and not r['shadow_off'] for r in (t,c,p))
    assert t['D'] is not None and c['D'] is not None
    r=None;by_kind={}
    if read_json(target/'offline/frozen.json')['adequate']:
        student=read_json(target/'offline'/(args.policy+'.json'))
        parent=read_json(target/'offline/P6.json')
        assert student['noise_sha256']==parent['noise_sha256']
        for kind,value in student['means'].items():
            den=parent['means'][kind]
            if den>0:by_kind[kind]=value/den
        r=by_kind.get('all')
    stage,seed,treatment,attempt=RUNS[args.policy]
    decision=stage_gate((t['D_num'],t['D_den']),(c['D_num'],c['D_den']),t['outcomes'].get('reached',0),c['outcomes'].get('reached',0),r,restarted=attempt==2)
    write_json(out/'training'/args.policy/'gate.json',dict(decision=decision,stage=stage,policy=args.policy,
        r=r,r_by_intervention=by_kind,D_T=t['D'],D_C=c['D'],D_P6=p['D'],
        T_arrivals=t['outcomes'].get('reached',0),C_arrivals=c['outcomes'].get('reached',0),
        T_state_sha256=t['state_sha256'],control_policy=c['policy'],control_state_sha256=c['state_sha256'],
        control_D_zero=c['D_num']==0,
        offline_behavior_disagreement=decision=='pass' and r is not None and r>.5))


def endpoints(args):
    out,plan=inputs(args)
    assert RUNS[args.policy][0]=='B' and args.policy!='P30'
    for split in ('dev','val'):
        for guided in ((False,True) if RUNS[args.policy][2] else (False,)):
            job=out/'final'/args.policy/(split+('_guided' if guided else ''))
            command=[sys.executable,str(Path(__file__).resolve()),'eval-worker','--run',str(out),
                '--policy',args.policy,'--update','320','--split',split,'--job',str(job)]
            if guided: command+=['--guided']
            run_logged(out,'final_%s_%s_%s'%(args.policy,split,'guided' if guided else 'student'),command)


def endpoint_counts(job,b,blocked):
    from scripts.routea_paired_support.episodes import load_commit
    from scripts.f7_analysis import outcome_1m
    tasks={t['task_id']:t for t in b['tasks']};conditions=[]
    for e in b['episodes']:
        h,trace,end=load_commit(job,Path(job)/'commits'/(e['key']+'.json'))
        assert not h['use_base_policy'] and h['initial_hash']==e['initial_hash']
        assert h['task']==tasks[e['task_id']]['payload'] and h['bank_hash']==b['content_hash']
        result=outcome_1m(trace,np.asarray(h['task']['goal_pos']),end['done_reason'])
        conditions.append(dict(task_id=e['task_id'],repeat=e['repeat'],outcome=result))
    a=Counter(r['outcome'] for r in conditions)
    solved=Counter(r['task_id'] for r in conditions if r['outcome']=='reached')
    return dict(arrivals=a['reached'],collision=a['collision'],timeout=a['time_limit'],
        blocked=sum(r['outcome']=='reached' and r['task_id'] in blocked for r in conditions),
        blocked_solved_tasks=sum(solved[tid]>=3 for tid in blocked),
        passable=sum(r['outcome']=='reached' and r['task_id'] not in blocked for r in conditions)),conditions


def analyze(args):
    out,plan=inputs(args)
    blocked=set(read_json(args.blocked))
    assert file_hash(args.blocked)=='32cbc908b61c899a9ad2d55978857de3eb6dfa752182411be7dcf7b12bd2ab3d'
    b=bank(plan,'dev');assert len(blocked)==33
    chosen=[]
    for original,retry in [('P24','P32'),('P25','P33'),('P26','P34')]:
        final=None
        for name in (original,retry):
            p=out/'training'/name/'gate.json'
            if p.exists() and read_json(p)['decision']=='pass' and (out/'final'/name/'dev/result.json').exists():
                final=name
        if final:chosen.append(final)
    if len(chosen)!=3:
        write_json(out/'analysis.json',dict(status='incomplete_three_final_runs',available=chosen,
                   note='No imputed u320 endpoint for a stopped seed.'))
        return
    controls={};records={};flat=[]
    for name in ['P22','P23','P27']+plan['optional_controls']['included']:
        job=(Path(plan['f8'])/'eval_dev'/name if name in ('P22','P23') else
             out/'final/P27/dev' if name=='P27' else Path(args.f7_jobs)/name)
        controls[name],records[name]=endpoint_counts(job,b,blocked)
    treatments={}
    for name in chosen:
        treatments[name],records[name]=endpoint_counts(out/'final'/name/'dev',b,blocked)
    c=list(controls.values());t=list(treatments.values())
    changed=(all(v['blocked']>max(x['blocked'] for x in c) for v in t)
        or all(v['blocked']<min(x['blocked'] for x in c) for v in t)
        or all(v['collision']<min(x['collision'] for x in c) for v in t)
        or all(v['collision']>max(x['collision'] for x in c) for v in t))
    adopted=(all(v['blocked']>max(x['blocked'] for x in c) for v in t)
        and all(v['collision']<min(x['collision'] for x in c) for v in t)
        and np.mean([v['passable'] for v in t])>=np.mean([v['passable'] for v in c])-10)
    verdict=('符合采纳标准' if adopted else '本预算下不符合采纳标准'
             if np.median([v['blocked'] for v in t])<=np.median([v['blocked'] for v in c]) else '不确定')
    for name,rows in records.items():flat.extend(dict(policy=name,**r) for r in rows)
    csv_write(out/'dev_conditions.csv',flat)
    write_json(out/'analysis.json',dict(status='complete',treatments=treatments,controls=controls,
        change_verdict='F9 改变了不带引导的策略' if changed else '未产生超出运行间差异的变化',
        adoption_verdict=verdict,optional_controls=plan['optional_controls']))


def audit(args):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.f9_online import validate_teacher_trace,demand_counts
    from scripts.routea_paired_support.episodes import load_commit
    from scripts.f7_analysis import outcome_1m
    out,plan=inputs(args)
    selection=read_json(out/'selection.json')
    if (out/'g0.json').exists():
        # record-only: never re-assert bitwise equality here
        assert read_json(out/'g0.json').get('recorded') is True
    analysis=(read_json(out/'analysis.json') if (out/'analysis.json').exists() else
              dict(status='stage0_stopped' if not selection['passed'] else 'partial_training_or_gate_stop'))
    step0_rows=[]
    for path in (out/'step0').glob('g*/result.json'):
        group=read_json(path)
        assert group['conditions']==425 and not group['smoke']
        for row in group['rows']:
            assert file_hash(row['source_trace'])==row['source_trace_sha256']
            for r in row['eligible_decisions']:
                step0_rows.append(dict(group=path.parent.name,task_id=row['task_id'],repeat=row['repeat'],**r))
    if step0_rows:csv_write(out/'step0_eligible_decisions.csv',step0_rows)
    # Partial/failed seeds are legitimate evidence, never relabelled complete.
    sample_rows=[];teacher_episodes=0
    for p in (out/'training').glob('P*/configured.json'):
        info=read_json(p);trial=p.parent
        assert file_hash(trial/'config.yaml')==info['config_sha256']
        if (trial/'gate.json').exists():
            gate=read_json(trial/'gate.json')
            assert gate['policy']==info['policy']
        if info['treatment']:
            for commit in (trial/'f9/episodes').glob('*/COMMITTED.json'):
                row=read_json(commit)
                assert file_hash(commit.parent/'trace.jsonl')==row['trace_sha256']
                raw=[json.loads(s) for s in (commit.parent/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
                assert raw[-1]==row['summary']
                eligible=validate_teacher_trace(raw[0],raw[1:-1],raw[-1])
                teacher_episodes+=1
                for j,index in enumerate(eligible):
                    sample_rows.append(dict(policy=info['policy'],episode=raw[0]['episode'],
                        resume_attempt=raw[0]['resume_attempt'],decision=index,
                        heldout=raw[0]['heldout'],intervention='first_intervention' if j==0 else 'after_intervention',
                        delta_exec=raw[index+1]['delta_exec'],P6_condition_failed='不适用',
                        P6_label_status='not_applicable_online_random_condition',trace_sha256=row['trace_sha256']))
            for path in (trial/'f9').glob('buffer_u*.pt'):
                data=torch.load(path,map_location='cpu',weights_only=True)
                assert len(data['records'])<=4096
                assert all(r['episode_number']%10!=0 for r in data['records'])
                assert all(r['episode_number']%10==0 for r in data['heldout'])
                for r in data['records']+data['heldout']:
                    assert file_hash(r['observation_path'])==r['observation_sha256']
    if sample_rows:csv_write(out/'teacher_samples.csv',sample_rows)
    evaluation_rows=[];prefix_rows=[]
    for path in out.rglob('result.json'):
        result=read_json(path)
        if 'state_sha256' not in result or 'split' not in result:continue
        b=bank(plan,result['split']); expected={(e['task_id'],e['repeat']) for e in b['episodes']}
        if not result['smoke']:
            actual=[(r['task_id'],r['repeat']) for r in result['rows']]
            assert len(actual)==len(expected) and set(actual)==expected
        task_by_id={t['task_id']:t for t in b['tasks']}
        entry_by_id={(e['task_id'],e['repeat']):e for e in b['episodes']}
        num=den=0
        for row in result['rows']:
            entry=entry_by_id[row['task_id'],row['repeat']]
            header,trace,end=load_commit(path.parent,path.parent/'commits'/(entry['key']+'.json'))
            assert header['initial_hash']==entry['initial_hash']
            assert file_hash(path.parent/row['trace'])==row['trace_sha256']
            goal=task_by_id[row['task_id']]['payload']['goal_pos']
            n,d,star=demand_counts([r['executed_path'] for r in trace],goal,
                [r.get('probe_metrics',{}).get('delta_exec',0.) for r in trace])
            assert (n,d,star)==(row['D_num'],row['D_den'],row['dstar'])
            assert row['outcome']==outcome_1m(trace,np.asarray(goal,np.float32),end['done_reason'])
            num+=n;den+=d
            evaluation_rows.append(dict(policy=result['policy'],update=result['update'],split=result['split'],
                guided=result['guided'],smoke=result['smoke'],job=path.parent.relative_to(out).as_posix(),
                **{k:v for k,v in row.items() if k!='delta_exec'}))
        assert (num,den)==(result['D_num'],result['D_den'])
    if evaluation_rows:csv_write(out/'all_evaluation_conditions.csv',evaluation_rows)
    for path in (out/'measure').glob('*/u*/offline/P*.json'):
        result=read_json(path)
        if 'means' not in result:continue
        for r in result['rows']:
            prefix_rows.append(dict(measurement=path.parent.relative_to(out).as_posix(),policy=result['policy'],
                **{k:v for k,v in r.items() if k!='timestep_losses'}))
    if prefix_rows:csv_write(out/'heldout_prefix_losses.csv',prefix_rows)
    rc={p.relative_to(out).as_posix():p.read_text().strip() for p in (out/'execution').glob('*.rc')}
    write_json(out/'audit.json',dict(status=analysis['status'],return_codes=rc,
        available_evidence_checks_passed=True,experiment_complete=analysis['status']=='complete',
        failed_commands=[k for k,v in rc.items() if v!='0'],raw_evidence_preserved=True,
        teacher_episodes_recomputed=teacher_episodes,teacher_samples_exported=len(sample_rows),
        evaluation_rows_recomputed=len(evaluation_rows),heldout_loss_rows=len(prefix_rows),
        pending_online_P6_labels=False,online_P6_labels='不适用',
        note='Failed attempts remain failures. No weight payload included in result manifest.'))
    files=sorted(p for p in out.rglob('*') if p.is_file())
    with (out/'MANIFEST.sha256').open('x',encoding='utf-8',newline='\n') as stream:
        for p in files:stream.write(file_hash(p)+'  '+p.relative_to(out).as_posix()+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ('run','diag-run','f8-run','probe-run','dev-bank','val-bank','repo','reference','weights'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--p22-checkpoint');p.add_argument('--f7-prepared')
    p=sub.add_parser('step0');p.add_argument('--run',required=True)
    p.add_argument('--group',type=int,choices=range(6),required=True);p.add_argument('--limit',type=int,choices=[1,20])
    for cmd in ('select','configure','train','g0','isolation','pair-probe','shadow-probe','freeze-holdout','measure','gate','endpoints','analyze','audit'):
        p=sub.add_parser(cmd);p.add_argument('--run',required=True)
        if cmd in ('configure','train','freeze-holdout','measure','gate','endpoints','isolation'):
            p.add_argument('--policy',choices=RUNS,required=True)
        if cmd=='train':
            p.add_argument('--stop',type=int,choices=[257,264,320],required=True)
            p.add_argument('--resume');p.add_argument('--approved',action='store_true')
        if cmd=='g0':p.add_argument('--p22-checkpoint',required=True)
        if cmd in ('measure','freeze-holdout','gate'):p.add_argument('--update',type=int,default=264,choices=[264])
        if cmd=='measure':p.add_argument('--control',choices=RUNS,required=True)
        if cmd=='isolation':p.add_argument('--control',choices=RUNS,required=True)
        if cmd=='analyze':p.add_argument('--blocked',required=True);p.add_argument('--f7-jobs')
    p=sub.add_parser('eval-worker');p.add_argument('--run',required=True);p.add_argument('--policy',required=True)
    p.add_argument('--update',type=int,required=True);p.add_argument('--split',choices=['train','dev','val'],required=True)
    p.add_argument('--job',required=True);p.add_argument('--guided',action='store_true')
    p.add_argument('--shadow-off',action='store_true');p.add_argument('--limit',type=int,choices=[1,20])
    p=sub.add_parser('prefix-worker');p.add_argument('--run',required=True);p.add_argument('--policy',required=True)
    p.add_argument('--treatment',required=True);p.add_argument('--update',type=int,default=264)
    args=parser.parse_args();function=globals()[args.command.replace('-','_')]
    if args.command in ('step0','eval-worker','prefix-worker','pair-probe'):
        with torch.jit.optimized_execution(False):function(args)
    else:function(args)


if __name__=='__main__':main()
