#!/usr/bin/env python3
"""Prepare registered P8--P11; execute the existing training pipeline."""
import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_a2 import digest, read, write, verify_saved_source


RUNS = {'P8': (202610311, True), 'P9': (202610311, False),
        'P10': (202610312, True), 'P11': (202610312, False)}
MIXTURE = dict(original=.15, L0=.10, L1=.15, L23_random=.30, L23_near=.20, L4=.10)


def make_config(receipt, receipt_path, reference, bank, output, weights, run_id):
    seed, treatment = RUNS[run_id]
    c = copy.deepcopy(receipt['training_config'])
    c['run'].update(seed=seed, checkpoint_every_updates=8,
                    checkpoint_updates=sorted(set(c['run']['checkpoint_updates']) | {129}))
    c['output_dir'] = str(Path(weights).resolve()/run_id)
    c['result_dir'] = str(Path(output).resolve()/run_id)
    c['a2'] = dict(arm='T', parent_checkpoint=receipt['parent_checkpoint'],
        parent_receipt=str(Path(receipt_path).resolve()), parent_receipt_sha256=digest(receipt_path),
        parent_source_root=receipt['source_root'], reference_bank=str(Path(reference).resolve()),
        reference_sha256=digest(reference), keep_seed=2026102301)
    c['f5'] = dict(run_id=run_id, keep_coefficient=.01061006)
    if treatment:
        c['reward_version'] = 'v3'
        c['task_bank'] = dict(enabled=True, path=str(Path(bank).resolve()),
            sha256=digest(bank), seed=seed+17, mixture=dict(MIXTURE))
    return c


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _source_rows
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.a2_fork import validate_child_config
    receipt = read(args.receipt)
    verify_saved_source(receipt)
    if digest(Path(receipt['parent_checkpoint'])/'state.pt') != receipt['state_sha256']:
        raise ValueError('parent weight differs from receipt')
    reference = Path(args.reference)
    reference_manifest = read(reference.parent/'reference_manifest.json')
    if digest(reference) != reference_manifest['sha256']:
        raise ValueError('reference bank hash differs')
    ref = torch.load(reference, map_location='cpu', weights_only=True)
    if (ref.get('schema') != 'routea_a2_train_success_v1' or len(ref['actions']) != 549
            or ref.get('parent_receipt_sha256') != digest(args.receipt)):
        raise ValueError('reference provenance/size differs')
    bank = read(args.bank)
    if len(bank['worlds']) != 21 or len(bank['tasks']) != 8400:
        raise ValueError('formal bank requires 21 worlds and 400 total tasks per world')
    from collections import Counter
    if set(Counter(r['world'] for r in bank['tasks']).values()) != {400}:
        raise ValueError('formal bank per-world quota differs')
    configs = {}
    output = Path(args.output).resolve()
    weights = Path(args.weights).resolve()
    for run_id in RUNS:
        if (weights/run_id).exists():
            raise FileExistsError('new run weights already exist: '+run_id)
        c = make_config(receipt,args.receipt,reference,args.bank,output/'training',weights,run_id)
        validate_v2_config(OmegaConf.create(c))
        validate_child_config(receipt['training_config'],c)
        configs[run_id] = c
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory import build_episode_tasks
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_sampler import TaskBankMixtureSampler
    template = configs['P8']
    tasks = build_episode_tasks(template['admission_manifest'],template['worlds_train'],
                                template['run']['seed'],template['environment'])
    TaskBankMixtureSampler(tasks,args.bank,digest(args.bank),
                          template['task_bank']['seed'],MIXTURE)
    output.mkdir(parents=True,exist_ok=False)
    paths = {}
    for run_id, c in configs.items():
        path = output/(run_id+'.yaml')
        OmegaConf.save(OmegaConf.create(c),path)
        paths[run_id] = dict(path=str(path), sha256=digest(path))
    write(output/'prepared.json', dict(schema='routea_f5_prepared_v1',
        configs=paths, files=_source_rows(Path(args.repo).resolve()),
        parent_receipt_sha256=digest(args.receipt), reference_sha256=digest(reference),
        task_bank_sha256=digest(args.bank), stop_update=192,
        interpretation='paired engineering experiment, not single-factor attribution'))
    print('F5_PREPARED_P8_P9_P10_P11')


def run(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import run as train
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _source_rows
    prepared = read(Path(args.prepared)/'prepared.json')
    root = Path(args.repo).resolve()
    if _source_rows(root) != prepared['files']:
        raise ValueError('training source changed after paired configuration')
    item = prepared['configs'][args.run_id]
    if digest(item['path']) != item['sha256']:
        raise ValueError('registered configuration changed')
    config = OmegaConf.load(item['path'])
    if args.stop_after_update == 192:
        if not args.resume:
            raise ValueError('u192 continuation requires the checked u129 checkpoint')
        verify_first_update(config, Path(config.output_dir)/'checkpoint_update_000129')
    train(config, root, resume_from=args.resume, stop_after_update=args.stop_after_update)
    if args.stop_after_update == 129:
        verify_first_update(config, Path(config.output_dir)/'checkpoint_update_000129')
        print('F5_U129_CONTRACT_PASS')


def verify_first_update(config, checkpoint):
    import torch
    from scripts.routea_paired_support.core import verify_checkpoint
    header = read(checkpoint/'header.json'); sums = read(checkpoint/'SHA256.json')
    verify_checkpoint(checkpoint, dict(runtime_identity_hash=header['runtime_identity_hash'],
        header_sha256=sums['header.json'], state_sha256=sums['state.pt'], update=129))
    state = torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
    from omegaconf import OmegaConf
    if state['training_config'] != OmegaConf.to_container(config,resolve=True):
        raise ValueError('u129 configuration differs')
    if state['schedules']['selective_keep']['coefficient'] != .01061006:
        raise ValueError('fixed keep coefficient was recalibrated')
    metrics = read(Path(config.result_dir)/'metrics_update_000129.json')
    if len(metrics['updates']) != 20 or any(u['status'] != 'committed' for u in metrics['updates']):
        raise ValueError('u129 requires twenty committed updates')
    records = metrics['rollout_audit']['transition_records']
    if len(records) != 1024 or any('task_bank' not in r for r in records):
        raise ValueError('u129 transition/task-source audit missing')
    if config.reward_version == 'v3' and not any(r['task_bank']['source'] != 'original' for r in records):
        raise ValueError('smoke did not exercise task bank resets')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    a = sub.add_parser('prepare')
    for key in ('repo','receipt','reference','bank','output','weights'):
        a.add_argument('--'+key,required=True)
    a = sub.add_parser('run')
    for key in ('repo','prepared'):
        a.add_argument('--'+key,required=True)
    a.add_argument('--run-id',choices=list(RUNS),required=True)
    a.add_argument('--stop-after-update',type=int,choices=[129,192],required=True)
    a.add_argument('--resume')
    args = p.parse_args()
    sys.path.insert(0,str(Path(args.repo).resolve()))
    globals()[args.command](args)


if __name__ == '__main__':
    main()
