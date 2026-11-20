#!/usr/bin/env python3
"""G-A4, adapted from Claude's supplied f6_u129_matched_rollout.py.

Same u128 parent, same seed, u129 before updates: rollout facts must match;
only reward-derived quantities may differ. Formula tolerance remains 1e-9.
Usage: python3 -I f6_u129_matched_rollout.py <treatment metrics> <control metrics>
Exit 0=PASS, 1=FAIL. Imported by the mandatory F6 u129/continuation gate.
"""
import json
import math
from pathlib import Path
import sys

K, R, G = 10., 2., .99
IDENT = ('episode_id', 'reset_seed', 'task_seed', 'task_id', 'world_id', 'role',
    'decision_index', 'action_chunk', 'position_before_xyz', 'position_after_xyz',
    'goal_xyz', 'goal_distance_before_3d', 'goal_distance_after_3d', 'terminal_facts',
    'terminated', 'truncated', 'collection_boundary', 'critic_episode_context',
    'next_critic_episode_context', 'value', 'next_value', 'task_bank',
    'update_index', 'policy_state_update_index', 'run_seed', 'bootstrap_allowed',
    'gae_continuation', 'task_identity_hash', 'admission_manifest_hash',
    'goal_error_xy_before', 'goal_error_z_before', 'goal_error_xy_after',
    'goal_error_z_after', 'net_goal_approach_3d', 'observation_provenance',
    'observation_provenance_after')
DERIVED = ('reward', 'return_target', 'advantage_raw', 'advantage_normalized_outer_update',
    'dppo_credit', 'reward_terms', 'reward_metrics', 'gradient_diagnostics')


def _identical(a, b):
    try:
        return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
    except ValueError:
        return False


def compare_rollouts(treatment, control):
    counts = dict(structural_mismatches=0, identity_field_mismatches=0,
                  formula_mismatches=0, reward_mismatches=0)
    examples = []
    def fail(kind, message):
        counts[kind] += 1
        if len(examples) < 10: examples.append(message)
    if len(treatment) != 1024 or len(control) != 1024:
        fail('structural_mismatches', 'requires exactly 1024 records on each side')
    t_ids = [r.get('transition_id') for r in treatment]
    c_ids = [r.get('transition_id') for r in control]
    if (None in t_ids or None in c_ids or len(set(t_ids)) != len(t_ids)
            or len(set(c_ids)) != len(c_ids) or set(t_ids) != set(c_ids)):
        fail('structural_mismatches', 'missing, duplicate or different transition_id sets')
    t = sorted(treatment, key=lambda r: str(r.get('transition_id')))
    c = sorted(control, key=lambda r: str(r.get('transition_id')))
    max_gp = max_rw = 0.
    nonzero = 0
    for a, b in zip(t, c):
        tid = a.get('transition_id')
        for name in IDENT:
            if name not in a or name not in b or not _identical(a.get(name), b.get(name)):
                fail('identity_field_mismatches', str(tid)+' field '+name+' missing/different')
        if any(r.get('update_index') != 129 or r.get('policy_state_update_index') != 128 for r in (a, b)):
            fail('structural_mismatches', str(tid)+' is not a u128-policy u129 rollout')
        ta, tb = a.get('reward_terms', {}), b.get('reward_terms', {})
        if set(ta) != set(tb) | {'goal_potential'} or 'goal_potential' in tb:
            fail('structural_mismatches', str(tid)+' reward term schema differs')
        for name, value in tb.items():
            if name not in ta or not _identical(ta[name], value):
                fail('identity_field_mismatches', str(tid)+' v2 term '+name+' differs')
        try:
            gp, rt, rc = float(ta['goal_potential']), float(a['reward']), float(b['reward'])
            before, after = float(a['goal_distance_before_3d']), float(a['goal_distance_after_3d'])
            if not all(math.isfinite(x) for x in (gp, rt, rc, before, after)):
                raise ValueError('nonfinite')
            phi = lambda d: K * (1. - min(d, R) / R)
            expected = 0. if a['terminal_facts'].get('invalid_state') else G*phi(after)-phi(before)
        except (KeyError, TypeError, ValueError, AttributeError):
            fail('formula_mismatches', str(tid)+' missing/nonfinite formula input')
            continue
        max_gp = max(max_gp, abs(gp-expected))
        max_rw = max(max_rw, abs((rt-rc)-gp))
        nonzero += gp != 0.
        if abs(gp-expected) > 1e-9:
            fail('formula_mismatches', str(tid)+' goal_potential does not match independent formula')
        if abs((rt-rc)-gp) > 1e-9:
            fail('reward_mismatches', str(tid)+' reward delta does not match goal_potential')
    return dict(passed=not any(counts.values()), records=len(treatment), control_records=len(control),
        **counts, formula_max_abs_error=max_gp, reward_delta_max_abs_error=max_rw,
        nonzero_goal_potential=nonzero, formula_tolerance=1e-9, mismatch_examples=examples,
        interpretation='G-A4 matched pre-update rollout only, not equality of updated policies')


def main():
    t = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['rollout_audit']['transition_records']
    c = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))['rollout_audit']['transition_records']
    report = compare_rollouts(t, c)
    print(json.dumps(report, indent=2, allow_nan=False))
    print('PASS' if report['passed'] else 'FAIL')
    return 0 if report['passed'] else 1


if __name__ == '__main__': sys.exit(main())
