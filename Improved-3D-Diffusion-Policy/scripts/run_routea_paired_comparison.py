#!/usr/bin/env python3
"""One paired evaluator: select models by checkpoint addresses."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
PROGRAM = Path('/home/cjj/AUV/program')
REFERENCE_NAME = 'routea_bc_a0_r1_paired_dev_20261010_py38fix_run02'
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', required=True,
                   help='compatible RL checkpoint directory; repeat for multiple models')
    p.add_argument('--reference', type=Path, help='accepted paired result directory; default is the known task bank')
    p.add_argument('--out', type=Path, help='new directory under program; default creates a timestamped run')
    p.add_argument('--resume', action='store_true', help='same code, tasks and weights only')
    p.add_argument('--plots', action='store_true', help='plot all frozen tasks at repeat 1')
    p.add_argument('--recovery-bank', type=Path, help='frozen same-state diagnostic input; does not alter training or full-evaluation denominator')
    p.add_argument('--recovery-only', action='store_true', help='run the frozen-state comparison without repeating the full task matrix')
    p.add_argument('--mechanism-baseline', type=Path,
                   help='accepted run03 recovery directory: exact diagonals, first-H4 crossover and goal response')
    p.add_argument('--paper', action='store_true',
                   help='approved BC/fresh u32 paper matrix: new 24 tasks plus full-remaining P3')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.paper:
        if a.recovery_only:
            raise ValueError('--paper includes both approved matrices')
        if a.recovery_bank is not None:
            raise ValueError('--paper uses the shipped frozen 24-state bank')
        a.recovery_bank=ROOT/'scripts/routea_paired_support/recovery_bank_v1.json'
    if a.recovery_only and not a.recovery_bank:
        raise ValueError('--recovery-only requires --recovery-bank')
    if a.mechanism_baseline and (not a.recovery_only or a.paper or a.resume):
        raise ValueError('--mechanism-baseline requires --recovery-only and a new non-paper run')
    if sys.version_info[:3] != (3, 8, 20):
        raise RuntimeError('run in the fixed idp3 Python 3.8.20 environment')
    import os
    import fcntl
    from scripts.routea_paired_support.core import write_json, file_hash
    from scripts.routea_paired_support.reference import check_reference
    from scripts.routea_paired_support.runtime import resolve_catalogue, preflight
    from scripts.routea_paired_support.workflow import save, run, finish, package_result
    from datetime import datetime
    candidates = [PROGRAM / 'results/routea_paired_evaluation' / REFERENCE_NAME,
                  ROOT / 'data/outputs/diagnostics' / REFERENCE_NAME,
                  PROGRAM / 'routea_outputs_migration_v2_run01/migrated_outputs/diagnostics' / REFERENCE_NAME]
    reference = a.reference or next((p for p in candidates if p.is_dir()), None)
    if reference is None:
        raise FileNotFoundError('accepted task bank missing; provide --reference')
    reference = reference.resolve()
    out = (a.out or PROGRAM / 'results/routea_paired_evaluation' /
           ('routea_eval_v3_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.relative_to(PROGRAM)
    if out == PROGRAM:
        raise ValueError('use an experiment subdirectory')
    if out.exists() and not a.resume:
        raise FileExistsError('use a new experiment output directory')
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib_cache')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    with (out / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        number = 1
        while (out / ('logs/attempt_%03d' % number)).exists():
            number += 1
        attempt = '%03d' % number
        logs = out / ('logs/attempt_' + attempt)
        logs.mkdir(parents=True)
        write_json(logs / 'command.json', {'python': sys.executable,
                   'argv': sys.argv if argv is None else [str(ROOT / 'scripts/run_routea_paired_comparison.py')] + argv})
        def stage(name, function):
            started = time.monotonic()
            rc, result = 0, None
            print('STAGE_START', name, flush=True)
            with (logs / (name + '.log')).open('x', encoding='utf-8') as stream:
                try:
                    with redirect_stdout(stream), redirect_stderr(stream):
                        result = function()
                except Exception:
                    traceback.print_exc(file=stream)
                    rc = 1
            (logs / (name + '.rc')).write_text(str(rc) + '\n', encoding='utf-8')
            (logs / (name + '.seconds')).write_text(str(time.monotonic()-started) + '\n', encoding='utf-8')
            if rc:
                raise RuntimeError('stage failed: ' + name + '; preserve log and return evidence')
            print('STAGE_PASS', name, flush=True)
            return result
        bank = stage('01_bank', lambda: check_reference(ROOT, reference, allow_space_expansion=True))
        cat = stage('02_weights', lambda: resolve_catalogue(ROOT, a.checkpoint))
        if a.paper:
            from scripts.routea_paired_support.paper import validate_models
            validate_models(cat)
        save(out / 'request.json', {'reference': str(reference), 'bank_hash': bank['content_hash'],
                                   'terminate_at_space_limits': False,
                                   'checkpoints': a.checkpoint, 'plots': a.plots,
                                   'recovery_only':a.recovery_only,
                                   'paper':a.paper,
                                   'mechanism_baseline':None if a.mechanism_baseline is None else str(a.mechanism_baseline.resolve()),
                                   'recovery_bank_sha256':None if a.recovery_bank is None else file_hash(a.recovery_bank)})
        save(out / 'models.json', cat)
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.jit.optimized_execution(False):
            report = stage('03_preflight', lambda: preflight(ROOT, cat))
            save(out / 'preflight.json', report)
            bank_path = reference / 'bank/bank.json'
            recovery_task_bank=bank
            if a.paper:
                from scripts.routea_paired_support.paper import prepare_bank
                bank=stage('03b_paper_bank',lambda:prepare_bank(ROOT,cat,out/'bank'))
                bank_path=out/'bank/bank.json'
            if not a.recovery_only:
                for number, mode in enumerate(('full',) if a.paper else ('smoke', 'full'), 4):
                    stage('%02d_%s' % (number, mode), lambda mode=mode: run(
                        ROOT, cat, bank_path, out / mode, mode, a.resume, reference))
                completion = stage('06_comparison', lambda: finish(ROOT, cat, bank, out, reference, a.plots))
            else:
                completion=dict(status='complete',full=0,smoke=0,comparison_full=0,anchors=0,weights_included=False)
            if a.recovery_bank:
                if a.mechanism_baseline:
                    from scripts.routea_paired_support.mechanism import run_mechanism
                    completion['mechanism']=stage('07_mechanism',lambda:run_mechanism(
                        ROOT,cat,recovery_task_bank,a.recovery_bank,a.mechanism_baseline,out/'mechanism'))
                else:
                    from scripts.routea_paired_support.recovery import run_recovery
                    completion['recovery']=stage('07_recovery',lambda:run_recovery(
                        ROOT,cat,recovery_task_bank,a.recovery_bank,out/'recovery',full_remaining=a.paper))
                if not a.recovery_only:
                    from scripts.routea_paired_support.recovery import analyze_new_evaluation
                    completion['behavior']=stage('08_behavior',lambda:analyze_new_evaluation(bank,out))
        # All stage logs have closed. One inventory includes prior failed attempts.
        package_result(out, attempt, completion)
        print('EXPERIMENT_COMPLETE', str(out), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
