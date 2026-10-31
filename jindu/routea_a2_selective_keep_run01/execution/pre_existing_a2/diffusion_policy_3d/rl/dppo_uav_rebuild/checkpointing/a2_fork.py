"""Explicit A1->A2 state transfer. Ordinary resume identity rules stay strict."""
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from ..engine.contracts import RuntimeIdentity
from ..engine.rollout_buffer import OnPolicyRolloutBuffer
from ..entrypoints.identity import _canonical_hash, build_executed_source_manifest
from .restore import restore_checkpoint
from .schema import CheckpointHeader


def validate_child_config(parent, child):
    left, right = copy.deepcopy(parent), copy.deepcopy(child)
    for key in ('output_dir', 'result_dir', 'a2'):
        left.pop(key, None); right.pop(key, None)
    # Saving u129 is instrumentation for the continuity gate, not a new update.
    for value in (left, right):
        if 'checkpoint_updates' in value.get('run', {}):
            value['run']['checkpoint_updates'] = [u for u in value['run']['checkpoint_updates'] if u != 129]
    if left != right:
        raise ValueError('A2 fork changes an unapproved training field')


def verified_json(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('A2 input hash mismatch: '+str(path))
    return json.loads(raw)


def fork_parent(config, repo_root, system, task_sampler, child_identity):
    spec = config.a2
    receipt = verified_json(spec.parent_receipt, spec.parent_receipt_sha256)
    root = Path(spec.parent_checkpoint)
    header = CheckpointHeader.from_json((root/'header.json').read_text(encoding='utf-8'))
    if header.update_index != 128 or header.runtime_identity_hash != receipt['runtime_identity_hash']:
        raise ValueError('A2 must fork verified B2560 u128')
    parent_payload = dict(header.runtime_identity)
    parent_payload['caveats'] = tuple(parent_payload.get('caveats',()))
    parent = RuntimeIdentity(**parent_payload)
    if parent.content_hash != header.runtime_identity_hash:
        raise ValueError('parent runtime payload mismatch')
    if hashlib.sha256((root/'state.pt').read_bytes()).hexdigest() != receipt['state_sha256']:
        raise ValueError('parent weights differ from verified receipt')
    if _canonical_hash(receipt['training_config']) != parent.config_hash:
        raise ValueError('parent configuration mismatch')
    validate_child_config(receipt['training_config'],OmegaConf.to_container(config,resolve=True))
    closure = receipt['source_manifest']
    if (_canonical_hash({k:v for k,v in closure.items() if k!='content_hash'}) != closure['content_hash']
            or closure['content_hash'] != parent.source_set_hash):
        raise ValueError('parent source closure mismatch')
    old_root = Path(spec.parent_source_root).resolve()
    for row in closure['files']:
        path = (old_root/row['relative_path']).resolve()
        path.relative_to(old_root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('parent source changed: '+row['relative_path'])
    current = build_executed_source_manifest(repo_root,config,Path(repo_root)/config.admission_manifest)
    before = {r['relative_path']:r['sha256'] for r in closure['files']}
    after = {r['relative_path']:r['sha256'] for r in current['files']}
    changed = {p for p in set(before)|set(after) if before.get(p)!=after.get(p)}
    prefix = 'diffusion_policy_3d/rl/dppo_uav_rebuild/'
    allowed = {prefix+p for p in ('engine/trainer.py','engine/selective_keep.py',
        'entrypoints/pipeline.py','entrypoints/config_schema_v2.py','entrypoints/a2_support.py',
        'checkpointing/a2_fork.py')}
    if changed-allowed:
        raise ValueError('unapproved A2 source differences: '+repr(sorted(changed-allowed)))
    a,b = dict(closure['inputs']),dict(current['inputs'])
    a.pop('resolved_config_sha256'); b.pop('resolved_config_sha256')
    if a != b:
        raise ValueError('A2 changes parent behavior inputs')
    for key in header.runtime_identity:
        if key not in ('source_set_hash','config_hash') and _canonical_hash(getattr(child_identity,key)) != _canonical_hash(getattr(parent,key)):
            raise ValueError('A2 changes runtime contract: '+key)
    # The separate parent receipt binds optimizer ordering to the old build.
    actor_names = [n for n,_ in system.lora_bundle.named_adapter_parameters()]
    critic_names = [n for n,_ in system.critic.named_parameters()]
    if actor_names != receipt['actor_names'] or critic_names != receipt['critic_names']:
        raise ValueError('fork optimizer parameter ordering differs')
    consumed_buffer=OnPolicyRolloutBuffer(int(config.run.rollout_transitions),
        gamma=float(config.dppo.gamma_environment),gae_lambda=float(config.dppo.gae_lambda),
        outer_update_id=128,episode_context=True)
    restored = restore_checkpoint(root, runtime_identity=parent,
        lora_identity=system.lora_bundle.identity, lora_bundle=system.lora_bundle,
        critic=system.critic, actor_optimizer=system.actor_optimizer,
        critic_optimizer=system.critic_optimizer, counters=system.trainer.counters,
        task_sampler=task_sampler, rollout_buffer=consumed_buffer, map_location=system.bc_runtime.device)
    return restored, dict(parent_runtime_identity_hash=parent.content_hash,
        parent_checkpoint=str(root), child_runtime_identity_hash=child_identity.content_hash,
        parent_receipt_sha256=spec.parent_receipt_sha256,
        changed_sources=sorted(changed), initialization='explicit_A1_u128_fork',
        restored_consumed_buffer_size=len(consumed_buffer),
        consumed_buffer_disposition='u128 already optimized; u129 collects fresh on-policy data, as A1 resume does')
