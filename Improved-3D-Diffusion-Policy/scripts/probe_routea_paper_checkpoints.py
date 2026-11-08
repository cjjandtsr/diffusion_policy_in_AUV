"""H3 strict native checkpoint loading in isolated processes; never train or run episodes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def validate_plan(plan):
    names = [m['policy'] for m in plan['models']]
    if len(names) != 6 or set(names) != {'P3', 'P5', 'P6_u192', 'P9', 'P11', 'P6'}:
        raise ValueError('H3 requires exactly six distinct registered checkpoint groups')
    c5 = [m['policy'] for m in plan['c5_models']]
    if len(c5) != 4 or set(c5) != {'bc', 'P2', 'P4', 'P6'}:
        raise ValueError('C5 requires the registered four-policy comparison chain')


def one(args, plan):
    from types import SimpleNamespace
    from scripts.routea_paired_support.core import file_hash
    item = next(m for m in plan['models'] if m['policy'] == args.model)
    checkpoint = Path(item['checkpoint'])
    if file_hash(checkpoint / 'state.pt') != item['state_sha256']:
        raise ValueError('checkpoint state differs from accepted evidence')
    header = read(checkpoint / 'header.json')
    if header['runtime_identity_hash'] != item['runtime_identity_hash']:
        raise ValueError('checkpoint identity differs from accepted evidence')
    # Same native loader used by the existing worker; no actor-only restore or identity override.
    from scripts.run_routea_xk_evaluation import native_system
    import torch
    with torch.jit.optimized_execution(False):
        system, config, loaded = native_system(SimpleNamespace(repo=plan['repo'],
            source_root=item['source_root'], checkpoint=item['checkpoint'], expected_update=item['update']))
    from omegaconf import OmegaConf
    if loaded['runtime_identity_hash'] != item['runtime_identity_hash']:
        raise ValueError('loaded header differs')
    write(Path(args.output) / (args.model + '.json'), dict(status='strict_native_load_passed',
        model=item, resolved_training_config=OmegaConf.to_container(config, resolve=True),
        actual_bc_identity=system.bc_runtime.identity.content_hash,
        critic_parameters=sum(p.numel() for p in system.critic.parameters()),
        probe_sha256=file_hash(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = read(args.plan)
    validate_plan(plan)
    if args.model:
        one(args, plan)
        return
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for item in plan['models']:
        name = item['policy']
        command = [sys.executable, str(Path(__file__).resolve()), '--plan',
                   str(Path(args.plan).resolve()), '--output', str(out), '--model', name]
        with (out / (name + '.log')).open('x', encoding='utf-8') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        (out / (name + '.rc')).write_text(str(process.returncode) + '\n', encoding='utf-8')
        results.append(dict(model=name, rc=process.returncode, command=command))
    passed = all(r['rc'] == 0 for r in results)
    write(out / 'load_results.json', dict(status='passed' if passed else 'failed', results=results))
    if not passed:
        raise SystemExit('H3 failed: preserve every log; do not run matrix or rewrite source identities')
    settings = dict(denoised_clip_value=2., min_std_vector=[.1] * 11)
    models = plan['models']
    for name, selected in dict(D_F2_dev=[m for m in models if m['policy'] == 'P3'],
                               D_F2_val_test=[m for m in models if m['policy'] in ('P3', 'P5')],
                               D_F3_test=[m for m in models if m['policy'] in ('P6_u192', 'P9', 'P11')],
                               D_extension=[m for m in models if m['policy'] != 'P6']).items():
        write(out / (name + '.json'), dict(settings, models=selected))
    write(out / 'C5_extension.json', dict(settings, models=plan['c5_models']))
    print('H3_NATIVE_LOAD_PASS', len(results), 'matrix plans emitted; no evaluation started')


if __name__ == '__main__':
    main()
