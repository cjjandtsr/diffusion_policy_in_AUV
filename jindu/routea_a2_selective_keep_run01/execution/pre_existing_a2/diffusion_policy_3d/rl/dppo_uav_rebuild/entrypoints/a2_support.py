"""Small A2 configuration/auxiliary-state bridge for the existing pipeline."""
import hashlib
from pathlib import Path
import torch
from ..engine.selective_keep import SelectiveKeep


def validate_a2_config(config):
    a = config.a2
    expected = {'arm','parent_checkpoint','parent_receipt','parent_receipt_sha256',
                'parent_source_root','reference_bank','reference_sha256','keep_seed'}
    if set(a) != expected or a.arm not in ('C','T'):
        raise ValueError('invalid A2 config')
    if (int(config.run.updates) != 256 or int(config.dppo.logical_batch_pairs) != 2560
            or int(config.run.expected_optimizer_commits_per_outer_update) != 20
            or 'task_protocol' not in config):
        raise ValueError('A2 requires unchanged common B2560 protocol')
    for key in ('parent_receipt_sha256','reference_sha256'):
        value = a[key]
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid A2 digest: '+key)


def make_keep(config):
    a=config.a2
    path=Path(a.reference_bank)
    if hashlib.sha256(path.read_bytes()).hexdigest() != a.reference_sha256:
        raise ValueError('reference bank hash mismatch')
    bank=torch.load(path,map_location='cpu',weights_only=True)
    if bank.get('schema') != 'routea_a2_train_success_v1':
        raise ValueError('not a training-origin A2 bank')
    if bank.get('parent_receipt_sha256') != a.parent_receipt_sha256:
        raise ValueError('reference parent differs')
    return SelectiveKeep(bank,coefficient=0.,seed=int(a.keep_seed))
