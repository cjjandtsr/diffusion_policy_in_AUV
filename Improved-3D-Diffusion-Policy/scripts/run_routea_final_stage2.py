"""Final paper run: native A2 equivalence, verified P14 fork and fresh reference.

Each phase is a separate process. Numerical training stays in the native pipeline.
"""
import argparse
import copy
import importlib.util
import math
from pathlib import Path
import shutil
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_f3_resume import canonical, digest, read, write
from scripts.run_routea_stage1_seed import native_pipeline, compare_u1_metrics, gate_metric


def pinned(spec):
    path = Path(spec['path'])
    if digest(path) != spec['sha256']:
        raise ValueError('input changed: '+str(path))
    return path


def check_plan(plan):
    for name, value in plan['inputs'].items():
        pinned(value)
    for relative, expected in plan['scripts'].items():
        if digest(Path(plan['repo'])/relative) != expected:
            raise ValueError('launcher/collector changed: '+relative)
    for spec in (plan['native'], plan['parent_native']):
        for row in spec['source_files']:
            if digest(Path(spec['source_root'])/row['relative_path']) != row['sha256']:
                raise ValueError('native closure changed: '+row['relative_path'])


def calibration_check(value):
    batches = value['batches']
    assert len(batches) == 4
    assert all(math.isfinite(b[k]) and b[k] > 0 for b in batches for k in ('ppo_norm', 'keep_norm'))
    expected = .1*statistics.median(b['ppo_norm'] for b in batches)/statistics.median(b['keep_norm'] for b in batches)
    assert math.isclose(value['coefficient'], expected, rel_tol=0, abs_tol=1e-16)
    return dict(expected=expected, actual=value['coefficient'], batches=batches,
                formula='0.1*median(ppo_norm)/median(keep_norm)')


def receipt(plan):
    path = Path(plan['run'])/'parent/receipt.json'
    value = read(path)
    assert value['state_sha256'] == plan['parent']['state_sha256']
    assert value['runtime_identity_hash'] == plan['parent']['runtime_identity_hash']
    assert digest(Path(value['parent_checkpoint'])/'state.pt') == value['state_sha256']
    return path, value


def accepted(plan):
    value = read(Path(plan['run'])/'a1_acceptance.json')
    assert value['status'] == 'passed' and value['contract'] == 'final_v1_3'
    assert value['source_files'] == plan['native']['source_files']
    assert value['plan_sha256'] == digest(ARGS.plan)


def accept_evidence(plan):
    out = Path(plan['run'])
    assert not out.exists(), 'choose a fresh run; retain the old failed A1'
    evidence = Path(plan['a1_evidence'])
    original = read(pinned(plan['inputs']['P6_u129']))
    actual_path = evidence/'training/metrics_update_000129.json'
    actual = read(actual_path)
    gate_metric(actual, 129)
    left, right = original['rollout_audit']['transition_records'], actual['rollout_audit']['transition_records']
    assert len(left) == len(right) == 1024
    # These fields are attached only after the optimizer loop; all other collection
    # and learning-input fields must match, apart from identity metadata.
    post_update = {'dppo_credit','gradient_diagnostics'}
    identities = {'runtime_identity_hash','config_hash','source_set_hash'}
    clean = lambda r: {k:v for k,v in r.items() if k not in post_update | identities}
    assert all(clean(a) == clean(b) for a,b in zip(left,right)), 'collection/learning input differs'
    calibration = calibration_check(read(evidence/'training/keep_calibration.json'))
    expected = read(evidence/'comparison.json')['P6_calibration']['actual']
    assert calibration['actual'] == expected == 0.01061006086489362
    stats = actual['formal_training']['optimizer_metric_summary']
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(pinned(plan['inputs']['P6_config']))
    assert stats['approx_kl']['max'] <= cfg.gates.max_approx_kl
    assert stats['ratio_mean']['min'] >= cfg.gates.ratio_mean_min
    assert stats['ratio_mean']['max'] <= cfg.gates.ratio_mean_max
    assert len(plan['source_delta']) == 1 and plan['source_delta'][0]['path'].endswith('config_schema_v2.py')
    differences = evidence/'metric_differences.jsonl'
    assert differences.is_file()
    write(out/'a1_acceptance.json',dict(status='passed',contract='final_v1_3',
        plan_sha256=digest(ARGS.plan),source_files=plan['native']['source_files'],
        evidence=str(evidence),metrics_sha256=digest(actual_path),differences_sha256=digest(differences),
        equal_collection_records=1024,calibration=calibration,
        post_update_fields=sorted(post_update),old_zero_tolerance_gate='failed; retained',
        claim='source and collection contract only; no bitwise updated-weight equivalence'))
    print('FINAL_A1_V1_3_ACCEPTED_EXISTING_EVIDENCE',flush=True)


def equivalence(plan):
    from omegaconf import OmegaConf
    out = Path(plan['run'])/'equivalence'
    weights = Path(plan['equivalence_weights'])
    assert not out.exists() and not weights.exists(), 'preserve failed run; use new plan/run'
    pipeline, _ = native_pipeline(plan['native'])
    cfg = OmegaConf.load(pinned(plan['inputs']['P6_config']))
    cfg.output_dir, cfg.result_dir = str(weights), str(out/'training')
    pipeline._validate_run_config(cfg)
    pipeline.run(cfg, Path(plan['repo']), resume_from=None, stop_after_update=129)
    reference = read(pinned(plan['inputs']['P6_u129']))
    actual = read(out/'training/metrics_update_000129.json')
    gate_metric(actual, 129)
    report = compare_u1_metrics(reference, actual, out/'metric_differences.jsonl')
    report['P6_calibration'] = calibration_check(read(out/'training/keep_calibration.json'))
    write(out/'comparison.json', report)
    assert report['numeric_and_structure_equal']
    assert report['first_five_transitions']['equal'] and report['first_five_episodes']['equal']
    check_plan(plan)
    print('FINAL_A1_U129_EQUIVALENCE_PASS', flush=True)


def prepare(plan):
    import torch
    from omegaconf import OmegaConf
    out = Path(plan['run'])
    accepted(plan)
    pipeline, identity = native_pipeline(plan['parent_native'])
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.restore import restore_checkpoint
    from scripts.routea_paired_support.core import verify_checkpoint
    parent = plan['parent']
    verify_checkpoint(parent['checkpoint'], parent)
    state = torch.load(Path(parent['checkpoint'])/'state.pt', map_location='cpu', weights_only=True)
    cfg = OmegaConf.create(state['training_config'])
    assert cfg.run.seed == 202611051 and cfg.run.updates == 256 and not cfg.get('a2')
    system = pipeline.build_system(cfg, Path(plan['repo']))
    runtime = identity.build_runtime_identity(cfg, Path(plan['repo']), system, Path(cfg.admission_manifest))
    assert runtime.content_hash == parent['runtime_identity_hash']
    header = restore_checkpoint(parent['checkpoint'], runtime_identity=runtime,
        lora_identity=system.lora_bundle.identity, lora_bundle=system.lora_bundle,
        critic=system.critic, actor_optimizer=system.actor_optimizer,
        critic_optimizer=system.critic_optimizer, counters=system.trainer.counters, map_location='cpu')
    assert header.update_index == 128
    closure = identity.build_executed_source_manifest(Path(plan['repo']), cfg, Path(cfg.admission_manifest))
    assert closure['content_hash'] == runtime.source_set_hash
    probe = read(Path(cfg.result_dir)/'r1_plan_probe.json')
    assert probe['runtime_identity_hash'] == runtime.content_hash
    r = dict(schema='a2_parent_v1', runtime_identity_hash=runtime.content_hash,
        runtime_identity=header.runtime_identity, parent_checkpoint=parent['checkpoint'],
        state_sha256=digest(Path(parent['checkpoint'])/'state.pt'),
        source_root=plan['parent_native']['source_root'], training_config=state['training_config'],
        source_manifest=closure, r1_probe=probe,
        actor_names=[n for n, _ in system.lora_bundle.named_adapter_parameters()],
        critic_names=[n for n, _ in system.critic.named_parameters()], preparation_script_sha256=digest(__file__))
    write(out/'parent/receipt.json', r)
    write(out/'parent/header.json', read(Path(parent['checkpoint'])/'header.json'))
    print('FINAL_P14_PARENT_PASS', flush=True)


def collect(plan):
    import torch
    from omegaconf import OmegaConf
    path, r = receipt(plan)
    pipeline, identity = native_pipeline(plan['native'])
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.a2_fork import fork_parent
    cfg = OmegaConf.create(r['training_config'])
    system = pipeline.build_system(cfg, Path(plan['repo']))
    cfg.a2 = dict(parent_checkpoint=r['parent_checkpoint'], parent_receipt=str(path),
        parent_receipt_sha256=digest(path), parent_source_root=r['source_root'])
    runtime = identity.build_runtime_identity(cfg, Path(plan['repo']), system, Path(cfg.admission_manifest))
    fork_parent(cfg, Path(plan['repo']), system, None, runtime)
    # Load the controlled collector explicitly: native imports remain untouched.
    name = 'scripts.routea_paired_support.a2_reference'
    spec = importlib.util.spec_from_file_location(name, Path(plan['repo'])/'scripts/routea_paired_support/a2_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inv = module.collect_reference(system, cfg, output=str(Path(plan['run'])/'reference'),
        parent_receipt_sha256=digest(path), exclude_bank=str(pinned(plan['inputs']['exclude_bank'])),
        original_tasks=str(pinned(plan['inputs']['reference_tasks'])))
    ref = Path(plan['run'])/'reference'
    bank = torch.load(ref/'reference.pt', map_location='cpu', weights_only=True)
    indices = []
    for category in ('both_success','bc_only','rl_only'):
        indices.extend([i for i,c in enumerate(bank['categories']) if c == category][:1])
    indices += [i for i in range(len(bank['task_ids'])) if i not in indices][:3-len(indices)]
    samples = [dict(index=i, task_id=bank['task_ids'][i], category=bank['categories'][i],
        source=bank['sources'][i], condition=bank['conditions'][i].tolist(),
        normalized_action=bank['actions'][i].tolist(), weights=bank['weights'][i].tolist(),
        provenance=bank['provenance'][i]) for i in indices]
    for sample in samples:
        provenance = sample['provenance']
        assert digest(ref/provenance['trace']) == provenance['trace_sha256']
        assert digest(ref/'observations'/provenance['observation']) == provenance['observation_sha256']
    write(Path(plan['run'])/'reference_samples.json', dict(samples=samples, sample_count=len(bank['task_ids']),
        reference_sha256=digest(ref/'reference.pt'), categories=inv['categories'],
        available_tasks=inv['available_tasks'], minimum_available_tasks=10,
        empty_rl_only=not bool(inv['available_tasks']['rl_only'])))
    assert sum(inv['available_tasks'].values()) >= 10, 'fewer than 10 available tasks; stop before training'
    check_plan(plan)
    print('FINAL_REFERENCE_PASS', inv['available_tasks'], flush=True)


def configuration(plan):
    path, r = receipt(plan)
    ref = Path(plan['run'])/'reference/reference.pt'
    inv = read(ref.parent/'inventory.json')
    assert sum(inv['available_tasks'].values()) >= 10
    assert digest(ref) == read(ref.parent/'reference_manifest.json')['sha256']
    c = copy.deepcopy(r['training_config'])
    c['run']['checkpoint_updates'] = sorted(set(c['run']['checkpoint_updates'])|{129})
    c['output_dir'], c['result_dir'] = plan['weights'], str(Path(plan['run'])/'training')
    c['a2'] = dict(arm='T', parent_checkpoint=r['parent_checkpoint'], parent_receipt=str(path),
        parent_receipt_sha256=digest(path), parent_source_root=r['source_root'],
        reference_bank=str(ref), reference_sha256=digest(ref), keep_seed=2026102301)
    # Do not copy old lambda; native make_keep starts at zero and calibrates at u129.
    return c


def train(plan):
    from omegaconf import OmegaConf
    from scripts.routea_paired_support.core import verify_checkpoint
    out = Path(plan['run'])
    accepted(plan)
    assert not Path(plan['weights']).exists() and not (out/'training').exists()
    config = configuration(plan)
    write(out/'resolved_config.json', config)
    pipeline, _ = native_pipeline(plan['native'])
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.a2_fork import validate_child_config
    validate_child_config(receipt(plan)[1]['training_config'], config)
    cfg = OmegaConf.create(config)
    pipeline._validate_run_config(cfg)
    publish = pipeline._write_json_once
    def save(path, value):
        publish(path, value)
        if Path(path).name.startswith('metrics_update_'):
            update = int(Path(path).stem.rsplit('_', 1)[-1])
            gate_metric(value, update)
            if update == 129:
                calibration = calibration_check(read(out/'training/keep_calibration.json'))
                write(out/'u129_gate.json', dict(status='passed', calibration=calibration,
                    parent_state_sha256=plan['parent']['state_sha256'], new_reference_sha256=cfg.a2.reference_sha256,
                    empty_rl_only_allowed=True, available_tasks=read(out/'reference/inventory.json')['available_tasks']))
                print('FINAL_P15_U129_PASS', flush=True)
    pipeline._write_json_once = save
    pipeline.run(cfg, Path(plan['repo']), resume_from=None, stop_after_update=256)
    for u in range(129, 257):
        gate_metric(read(out/'training'/('metrics_update_%06d.json' % u)), u)
    checkpoints = []
    for u in [129]+list(range(136, 257, 8)):
        ckpt = Path(plan['weights'])/('checkpoint_update_%06d' % u)
        h, sums = read(ckpt/'header.json'), read(ckpt/'SHA256.json')
        row = dict(policy='P15', checkpoint=str(ckpt), update=u,
            source_root=plan['native']['source_root'], runtime_identity_hash=h['runtime_identity_hash'],
            state_sha256=sums['state.pt'], header_sha256=sums['header.json'])
        verify_checkpoint(ckpt, row)
        assert h['runtime_identity']['config_hash'] == canonical(config)
        target = out/'weights_evidence'/ckpt.name
        target.mkdir(parents=True, exist_ok=False)
        for filename in ('header.json', 'SHA256.json', 'COMMITTED.json'):
            shutil.copy2(ckpt/filename, target/filename)
        checkpoints.append(row)
    check_plan(plan)
    write(out/'training_complete.json', dict(status='passed', updates=128, range=[129,256],
        automatic_training_stop=False, checkpoints=checkpoints, model=checkpoints[-1]))
    print('FINAL_P15_TRAIN_PASS', 128, flush=True)


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['equivalence', 'accept-evidence', 'prepare', 'collect', 'train'])
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    ARGS = args
    plan = read(args.plan)
    check_plan(plan)
    # Same JIT setting as original A2 training: no evaluation-protocol override.
    globals()[args.phase.replace('-','_')](plan)


if __name__ == '__main__':
    main()
