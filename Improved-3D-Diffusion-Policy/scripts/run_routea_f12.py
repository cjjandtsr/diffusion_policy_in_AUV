"""F12 phase-2 orchestration over the unchanged F7 P6-fork training pipeline (Claude 2026-11-26).

Subcommands
  prepare  write the four registered configs of one seed batch (C0, R, RN, RX)
  run      train one registered run to u257 (first update) or u320 (end)
  matched  u257 matched-collection checks: C0 vs R (reward recomputation), R vs RX (identity)
  summary  per-update training summary of one run (lightweight, from metrics files)
"""
import argparse
import copy
import math
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_a2 import read, write, digest, verify_saved_source

BATCH = {1: ('P39', 'P40', 'P41', 'P42'), 2: ('P43', 'P44', 'P45', 'P46')}
P6_STATE_SHA = 'a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43'


def make_config(receipt, receipt_path, reference, output, weights, run_id):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.f12_support import RUNS, FLOOR, RX_DEFAULTS, reward_params
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.f7_support import KEEP
    arm, seed = RUNS[run_id]
    c = copy.deepcopy(receipt['training_config'])
    original_reference_parent = c['a2']['parent_receipt_sha256']
    c['run'].update(seed=seed, updates=320, checkpoint_every_updates=8,
                    checkpoint_updates=sorted({0, 1, 129, 257} | set(range(8, 321, 8))))
    c['optimizer'].update(actor_lr=1e-5, actor_lr_end=1e-5, critic_lr=1e-3)
    c['dppo'].update(min_sampling_std=FLOOR[arm], min_logprob_std=FLOOR[arm])
    c['output_dir'] = str(Path(weights).resolve() / run_id)
    c['result_dir'] = str(Path(output).resolve() / 'training' / run_id)
    c['a2'].update(parent_checkpoint=receipt['parent_checkpoint'],
                   parent_receipt=str(Path(receipt_path).resolve()), parent_receipt_sha256=digest(receipt_path),
                   parent_source_root=receipt['source_root'], reference_bank=str(Path(reference).resolve()))
    c['reward_version'] = 'v2' if arm == 'C0' else 'v6'
    c['f7'] = dict(run_id=run_id, approach_weight=1., adaptive_lr=False, keep_coefficient=KEEP,
                   reference_parent_receipt_sha256=original_reference_parent)
    c['f12'] = dict(run_id=run_id, arm=arm, reward={} if arm == 'C0' else reward_params(),
                    rx=dict(RX_DEFAULTS, enabled=arm == 'RX'))
    return c


def sources(root):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _source_rows
    rows = _source_rows(root)
    names = ['scripts/run_routea_f12.py', 'scripts/f12_phase2/probes.py', 'scripts/f7_analysis.py']
    names += [p.relative_to(root).as_posix() for p in (root / 'scripts/routea_paired_support').glob('*.py')]
    for name in names:
        if name not in {r['relative_path'] for r in rows}:
            rows.append(dict(relative_path=name, sha256=digest(root / name)))
    return sorted(rows, key=lambda r: r['relative_path'])


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.a2_fork import validate_child_config
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.f7_support import REFERENCE_SHA
    r = read(args.receipt)
    verify_saved_source(r)
    assert r['state_sha256'] == P6_STATE_SHA, 'receipt is not the registered P6 u256'
    assert digest(Path(r['parent_checkpoint']) / 'state.pt') == r['state_sha256']
    assert digest(args.reference) == REFERENCE_SHA
    b = torch.load(args.reference, map_location='cpu', weights_only=True)
    assert len(b['actions']) == 549 and b['parent_receipt_sha256'] == r['training_config']['a2']['parent_receipt_sha256']
    out = Path(args.output) / ('batch%d' % args.batch)
    out.mkdir(parents=True, exist_ok=False)
    configs = {}
    for name in BATCH[args.batch]:
        c = make_config(r, args.receipt, args.reference, Path(args.output), args.weights, name)
        validate_child_config(r['training_config'], c)
        p = out / (name + '.yaml')
        OmegaConf.save(OmegaConf.create(c), p)
        configs[name] = dict(path=str(p.resolve()), sha256=digest(p))
    write(out / 'prepared.json', dict(batch=args.batch, configs=configs, files=sources(Path(args.repo).resolve()),
                                      receipt=str(Path(args.receipt).resolve()), receipt_sha256=digest(args.receipt)))
    print('F12_PREPARED', out)


def run(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import run as train
    prepared = read(Path(args.prepared) / 'prepared.json')
    root = Path(args.repo).resolve()
    assert sources(root) == prepared['files'], 'prepared source changed'
    item = prepared['configs'][args.run_id]
    assert digest(item['path']) == item['sha256']
    c = OmegaConf.load(item['path'])
    resume = None
    if args.stop_after_update == 320:
        resume = Path(args.resume) if args.resume else Path(c.output_dir) / 'checkpoint_update_000257'
        assert resume.resolve().parent == Path(c.output_dir).resolve()
        assert 257 <= read(resume / 'header.json')['update_index'] < 320
    elif args.resume:
        raise ValueError('u257 must fork P6')
    train(c, root, resume_from=resume, stop_after_update=args.stop_after_update)
    m = read(Path(c.result_dir) / ('metrics_update_%06d.json' % args.stop_after_update))
    assert len(m['updates']) == 20 and all(u['status'] == 'committed' for u in m['updates'])
    assert m['formal_training']['hard_stop_reason'] is None
    from scripts.routea_f6_support import verified_state
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.f7_support import KEEP
    checkpoint = Path(c.output_dir) / ('checkpoint_update_%06d' % args.stop_after_update)
    _, state = verified_state(checkpoint)
    assert state['training_config'] == OmegaConf.to_container(c, resolve=True)
    assert state['schedules']['selective_keep']['coefficient'] == KEEP
    assert state['counters']['committed_updates'] == args.stop_after_update * 20
    assert state['counters']['env_steps'] == args.stop_after_update * 1024
    if c.f12.arm == 'RX':
        assert 'f12_rx' in state['schedules']
    print('F12_TRAINING_STAGE_COMPLETE', args.run_id, args.stop_after_update)


# ------------------------------------------------------------------ matched checks
def _free_scalar(start, move, obstacles):
    """Independent re-derivation of D_f with the scalar collision_geom routine."""
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance
    n = math.sqrt(sum(x * x for x in move))
    if n < 1e-6:
        return 10.
    u = [x / n for x in move]
    for i in range(1, 101):
        t = i * .1
        p = [start[j] + t * u[j] for j in range(3)]
        if p[2] < .2 or compute_obstacle_clearance(p[0], p[1], p[2], obstacles)[0] < .3:
            return t
    return 10.


def _records(prepared, name):
    from omegaconf import OmegaConf
    item = prepared['configs'][name]
    assert digest(item['path']) == item['sha256']
    c = OmegaConf.load(item['path'])
    m = read(Path(c.result_dir) / 'metrics_update_000257.json')
    assert len(m['updates']) == 20 and all(u['status'] == 'committed' for u in m['updates'])
    rows = m['rollout_audit']['transition_records']
    data = {r['transition_id']: r for r in rows}
    assert len(rows) == len(data) == 1024
    return c, data


def matched(args):
    from scripts.f6_u129_matched_rollout import IDENT, _identical
    from diffusion_policy_3d.env.collision_geom import load_obstacles
    prepared = read(Path(args.prepared) / 'prepared.json')
    c0, r, rn, rx = BATCH[prepared['batch']]
    cc0, A = _records(prepared, c0)
    cr, B = _records(prepared, r)
    assert A.keys() == B.keys()
    worlds, report = {}, Counter()
    hist = {}
    for tid in sorted(B, key=lambda t: (B[t]['episode_id'], B[t]['decision_index'])):
        a, b = A[tid], B[tid]
        for key in IDENT + ('executed_path',):
            assert key in a and key in b and _identical(a[key], b[key]), (tid, key)
        ta, tb, mb = a['reward_terms'], b['reward_terms'], b['reward_metrics']
        assert set(tb) == set(ta) | {'stall'}, (tid, sorted(ta), sorted(tb))
        for key in set(ta) - {'newbest', 'terminal'}:
            assert _identical(ta[key], tb[key]), (tid, key)
        w = b['world_id']
        if w not in worlds:
            worlds[w] = load_obstacles(str(Path(cr.worlds_train) / (w + '.world.obstacles.json')))
        path = b['executed_path']
        move = [path[-1][j] - path[0][j] for j in range(3)]
        free = _free_scalar(path[0], move, worlds[w])
        goal_distance = math.sqrt(sum((b['goal_xyz'][j] - path[0][j]) ** 2 for j in range(3)))
        gate = 1. if goal_distance <= free else min(1., max(.2, free / 6.))
        report['goal_before_obstacle'] += goal_distance <= free < 10.
        assert abs(mb['newbest_raw'] - ta['newbest']) < 1e-12, tid
        assert abs(mb['f12_free_distance'] - free) < 1e-6, (tid, mb['f12_free_distance'], free)
        assert abs(tb['newbest'] - ta['newbest'] * gate) < 1e-9, tid
        reason = b['terminal_facts']['done_reason']
        expect_terminal = -100. if reason in ('collision', 'ground_crash') else ta['terminal']
        assert tb['terminal'] == expect_terminal, (tid, reason, tb['terminal'])
        h = hist.setdefault(b['episode_id'], [])
        h.append(math.sqrt(sum(x * x for x in move)))
        expect_stall = -.2 if len(h) >= 5 and sum(h[-5:]) < .3 and reason in ('none', 'time_limit') else 0.
        assert tb['stall'] == expect_stall, (tid, h[-5:], tb['stall'])
        assert abs(b['reward'] - sum(tb.values())) < 1e-9 and abs(a['reward'] - sum(ta.values())) < 1e-9
        report['gate<1'] += gate < 1.
        report['newbest>0'] += ta['newbest'] > 0
        report['newbest>0&gate<1'] += ta['newbest'] > 0 and gate < 1.
        report['collision_terminal'] += expect_terminal == -100.
        report['stall'] += expect_stall != 0.
    result = dict(pair_C0_R=[c0, r], transitions=len(B), counts=dict(report), passed=True)
    try:
        _, X = _records(prepared, rx)
        assert X.keys() == B.keys()
        for tid, b in B.items():
            x = X[tid]
            for key in IDENT + ('executed_path', 'reward_terms', 'reward'):
                assert _identical(b[key], x[key]), (tid, key)
        from omegaconf import OmegaConf
        crx = OmegaConf.load(prepared['configs'][rx]['path'])
        rxc = read(Path(crx.result_dir) / 'f12_rx_collect_000257.json')
        result['pair_R_RX'] = [r, rx]
        result['rx_u257'] = {k: rxc[k] for k in rxc if k != 'groups_detail'}
    except FileNotFoundError:
        result['pair_R_RX'] = 'RX u257 not present (not in this batch or not yet run)'
    write(Path(args.prepared) / 'matched_u257.json', result)
    print('F12_MATCHED_U257_PASS', result['counts'], result.get('rx_u257'))


def summary(args):
    from omegaconf import OmegaConf
    prepared = read(Path(args.prepared) / 'prepared.json')
    c = OmegaConf.load(prepared['configs'][args.run_id]['path'])
    rows = []
    for u in range(257, 321):
        p = Path(c.result_dir) / ('metrics_update_%06d.json' % u)
        if not p.exists():
            break
        m = read(p)
        recs = m['rollout_audit']['transition_records']
        ends = Counter(r['terminal_facts']['done_reason'] for r in recs if r['terminated'] or r['truncated'])
        rows.append(dict(update=u, ends=dict(ends), mean_reward=sum(r['reward'] for r in recs) / len(recs),
                         clip_fraction=m.get('optimizer', {}).get('clip_fraction')))
    write(Path(args.output), dict(run_id=args.run_id, updates=rows))
    print('F12_SUMMARY', args.run_id, len(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='command', required=True)
    a = s.add_parser('prepare')
    for k in ('repo', 'receipt', 'reference', 'output', 'weights'):
        a.add_argument('--' + k, required=True)
    a.add_argument('--batch', type=int, choices=(1, 2), required=True)
    a = s.add_parser('run')
    for k in ('repo', 'prepared', 'run-id'):
        a.add_argument('--' + k, required=True)
    a.add_argument('--stop-after-update', type=int, choices=(257, 320), required=True)
    a.add_argument('--resume')
    a = s.add_parser('matched')
    a.add_argument('--prepared', required=True)
    a = s.add_parser('summary')
    for k in ('prepared', 'run-id', 'output'):
        a.add_argument('--' + k, required=True)
    args = p.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
