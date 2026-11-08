"""Run delivered phases with commands/logs/rc and weight-free evidence archives.

No algorithm is implemented here. Run only from the established WSL idp3 env.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_f3_resume import read, write, digest


def run(args):
    root = Path(__file__).resolve().parents[1]
    plan_path = Path(args.plan) if args.plan else root/'scripts/configs/routea_final_stage2_v1.json'
    plan = read(plan_path)
    assert root == Path(plan['repo']), 'must use the delivered WSL source location'
    results = Path('/home/cjj/AUV/program/results')
    driver = results/('routea_final_driver_'+args.run_id)/args.phase
    driver.mkdir(parents=True, exist_ok=False)
    config = root/'scripts/configs'
    freeze = results/'routea_final_freeze_run01'
    selection = results/'routea_final_balance_run01/additions.json'
    out = Path(plan['run'])
    py = sys.executable
    script = lambda name: str(root/'scripts'/name)
    commands = []
    payloads = [driver]
    if args.phase == 'balance':
        commands = [[py,script('routea_testext_task_bank.py'),'balance','--plan',str(config/'routea_final_balance_v1.json'),
                     '--output',str(results/'routea_final_balance_run01')]]
        payloads += [results/'routea_final_balance_run01']
    elif args.phase == 'freeze':
        commands = [[py,script('routea_testext_task_bank.py'),'append','--plan',str(config/'routea_final_balance_v1.json'),
                     '--selection',str(selection),'--output',str(freeze)]]
        payloads += [freeze]
    elif args.phase == 'stage2':
        commands = [[py,script('run_routea_final_stage2.py'),phase,'--plan',str(plan_path)]
                    for phase in ('accept-evidence','prepare','collect','train')]
        payloads += [out]
    elif args.phase in ('existing','P15'):
        assert args.review, 'evaluation requires Claude review JSON for this actual freeze'
        evalplan = driver/'evaluation_plan.json'
        commands = [[py,script('prepare_routea_final_evaluation.py'),'--setup',str(config/'routea_final_evaluation_v1.json'),
            '--freeze',str(freeze),'--review',str(Path(args.review).resolve()),'--mode',args.phase,'--output',str(evalplan)],
            [py,script('run_routea_f3_resume.py'),'evaluate','--plan',str(evalplan),
             '--output',str(results/('routea_final_existing_e2_run01' if args.phase=='existing' else 'routea_final_P15_e2_run01'))]]
        shutil.copy2(args.review,driver/'Claude_review.json')
        payloads += [results/('routea_final_existing_e2_run01' if args.phase=='existing' else 'routea_final_P15_e2_run01')]
    elif args.phase == 'exports':
        exports = results/'routea_final_exports_run01'
        commands = [[py,script('export_routea_final_data.py'),'curves','--plan',str(config/'routea_final_exports_v1.json'),
                     '--output',str(exports)],
                    [py,script('export_routea_final_data.py'),'clearance','--plan',str(config/'routea_final_exports_v1.json'),
                     '--index',str(freeze/'O_prime_index.json'),'--output',str(exports)]]
        payloads += [exports]
    elif args.phase == 'trajectories':
        assert args.selection, 'requires Claude three-task selection JSON'
        target = results/'routea_final_trajectories_run01'
        commands = [[py,script('export_routea_final_data.py'),'trajectories','--plan',str(config/'routea_final_exports_v1.json'),
                     '--selection',str(Path(args.selection).resolve()),'--output',str(target)]]
        shutil.copy2(args.selection,driver/'Claude_selection.json')
        payloads += [target]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    write(driver/'input_plan.json',plan)
    for file in ['run_routea_final_stage2.py','prepare_routea_final_evaluation.py',
                 'export_routea_final_data.py','launch_routea_final.py','routea_testext_task_bank.py',
                 'run_routea_stage1_seed.py','run_routea_f3_resume.py',
                 'routea_paired_support/a2_reference.py']:
        target = driver/'source/scripts'/file
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'scripts'/file,target)
    shutil.copytree(config,driver/'source/scripts/configs',ignore=shutil.ignore_patterns('routea_g17_references'))
    if args.phase == 'stage2':
        for row in plan['native']['source_files']:
            target=driver/'source/native_a2'/row['relative_path']
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(Path(plan['native']['source_root'])/row['relative_path'],target)
    rc = 1
    try:
        if args.phase in ('freeze','stage2','existing','P15'):
            gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
                                 capture_output=True,text=True)
            write(driver/'gpu_before.json',dict(returncode=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr))
            assert gpu.returncode == 0 and not gpu.stdout.strip(), 'GPU busy/unavailable; do not kill other jobs'
        for i,argv in enumerate(commands):
            label = '%02d_%s' % (i,Path(argv[1]).stem)
            write(driver/(label+'.command.json'),dict(argv=argv,cwd=str(root),python=py))
            with (driver/(label+'.log')).open('x',encoding='utf-8') as log:
                code = subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            (driver/(label+'.rc')).write_text(str(code)+'\n',encoding='utf-8')
            if code:
                raise RuntimeError(label+' failed; preserve run and return evidence')
        rc = 0
    finally:
        (driver/'master.rc').write_text(str(rc)+'\n',encoding='utf-8')
        archive = results/('routea_final_'+args.phase+'_delivery_'+args.run_id+'.tar.gz')
        assert not archive.exists(), 'do not overwrite delivery'
        files = [f for folder in payloads if folder.exists() for f in folder.rglob('*')
                 if f.is_file() and f.suffix not in ('.pt','.pth','.ckpt','.pyc')
                 and '__pycache__' not in f.parts and 'unused_bytecode' not in f.parts]
        write(driver/'MANIFEST.json', {str(f.relative_to(results)):dict(sha256=digest(f),bytes=f.stat().st_size)
                                      for f in sorted(files)})
        files.append(driver/'MANIFEST.json')
        with tarfile.open(archive,'x:gz') as tar:
            for file in sorted(files):
                tar.add(file,arcname=str(file.relative_to(results)),recursive=False)
        archive.with_suffix(archive.suffix+'.sha256').write_text(digest(archive)+'  '+archive.name+'\n',encoding='utf-8')
        print('FINAL_DELIVERY',args.phase,'master.rc',rc,str(archive),flush=True)
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['balance','freeze','stage2','existing','P15','exports','trajectories'])
    p.add_argument('--review')
    p.add_argument('--selection')
    p.add_argument('--plan', help='training plan; change paths/inputs without another entry point')
    p.add_argument('--run-id', default='run01', help='fresh driver/archive suffix')
    args = p.parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
