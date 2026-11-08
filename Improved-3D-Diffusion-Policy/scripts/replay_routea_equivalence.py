"""Replay an existing equivalence plan into fresh paths and report differences.

Diagnostic only: never approves or starts the formal training stages.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_f3_resume import read, write, digest
from scripts.run_routea_stage1_seed import compare_u1_metrics


def tensor_differences(left, right):
    import torch
    rows = []
    def visit(a, b, path):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
                rows.append(dict(path=path, structure_equal=False))
                return
            equal = torch.equal(a,b)
            delta = (a.double()-b.double()).abs()
            rows.append(dict(path=path, structure_equal=True, equal=equal, elements=a.numel(),
                max_abs=float(delta.max()) if a.numel() else 0.))
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                rows.append(dict(path=path, structure_equal=False))
                return
            for key in a: visit(a[key],b[key],path+'/'+str(key))
        elif isinstance(a, (list,tuple)):
            if type(a) != type(b) or len(a) != len(b):
                rows.append(dict(path=path, structure_equal=False))
                return
            for i,(x,y) in enumerate(zip(a,b)): visit(x,y,path+'/'+str(i))
    visit(left,right,'state')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--baseline-metrics',required=True)
    p.add_argument('--baseline-checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--weights',required=True)
    args=p.parse_args()
    out,weights=Path(args.output).resolve(),Path(args.weights).resolve()
    assert not out.exists() and not weights.exists(), 'choose fresh diagnostic paths'
    plan=read(args.plan)
    # Only output routing changes; the original pinned executable and recipe remain intact.
    diagnostic=dict(plan,run=str(out/'replay'),equivalence_weights=str(weights))
    out.mkdir(parents=True)
    write(out/'plan.json',diagnostic)
    root=Path(plan['repo'])
    argv=[sys.executable,str(root/'scripts/run_routea_final_stage2.py'),'equivalence','--plan',str(out/'plan.json')]
    write(out/'command.json',dict(argv=argv,baseline_metrics_sha256=digest(args.baseline_metrics),
        original_plan_sha256=digest(args.plan),formal_training_approval=False))
    with (out/'replay.log').open('x',encoding='utf-8') as log:
        rc=subprocess.run(argv,cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                          stdout=log,stderr=subprocess.STDOUT).returncode
    (out/'replay.rc').write_text(str(rc)+'\n',encoding='utf-8')
    # A numerical mismatch can produce rc=1 only after a complete replay. Preserve it.
    result=read(out/'replay/equivalence/comparison.json')
    assert rc==0 or (rc==1 and not result['numeric_and_structure_equal'])
    import torch
    update=int(read(args.baseline_checkpoint+'/header.json')['update_index'])
    current=weights/('checkpoint_update_%06d' % update)
    from scripts.routea_paired_support.core import verify_checkpoint
    for checkpoint in (Path(args.baseline_checkpoint),current):
        header=read(checkpoint/'header.json');sums=read(checkpoint/'SHA256.json')
        verify_checkpoint(checkpoint,dict(update=update,runtime_identity_hash=header['runtime_identity_hash'],
            state_sha256=sums['state.pt'],header_sha256=sums['header.json']))
    from scripts.run_routea_final_stage2 import check_plan
    check_plan(diagnostic)
    metric=out/'replay/equivalence/training'/('metrics_update_%06d.json' % update)
    comparison=compare_u1_metrics(read(args.baseline_metrics),read(metric),out/'repeat_differences.jsonl')
    write(out/'repeat_comparison.json',comparison)
    left=torch.load(Path(args.baseline_checkpoint)/'state.pt',map_location='cpu',weights_only=True)
    right=torch.load(current/'state.pt',map_location='cpu',weights_only=True)
    rows=tensor_differences(left,right)
    write(out/'tensor_comparison.json',dict(baseline_state_sha256=digest(Path(args.baseline_checkpoint)/'state.pt'),
        replay_state_sha256=digest(current/'state.pt'),rows=rows,
        unequal_by_section=dict(Counter(r['path'].split('/')[1] if '/' in r['path'] else 'state' for r in rows
            if not r.get('structure_equal') or not r.get('equal')))))
    write(out/'completion.json',dict(diagnostic_completed=True,
        repeat_data_difference_count=comparison['data_difference_count'],
        original_equivalence_returncode=rc,formal_gate_approved=False,
        interpretation='Observed repeatability only; no tolerance change or root-cause verdict.'))
    print('REPLAY_DIAGNOSTIC_COMPLETE',comparison['data_difference_count'],flush=True)


if __name__=='__main__':
    main()
