"""Bounded eight-update success-prefix bank with checkpointed sampling RNG."""
import hashlib
import json

import torch

from .selective_keep import SelectiveKeep, executed_weights


def _content_hash(records):
    digest = hashlib.sha256()
    for record in records:
        for key in ('action', 'condition', 'weight'):
            tensor = record[key].detach().cpu().float().contiguous()
            digest.update(tensor.numpy().tobytes())
        digest.update(json.dumps({k:v for k,v in record.items()
                                  if k not in ('action','condition','weight')},
                                 sort_keys=True).encode('utf-8'))
    return digest.hexdigest()


class RollingSelfImitation(SelectiveKeep):
    def __init__(self, bank, *, coefficient, seed, batch_size=32,
                 window_updates=8, capacity=2048, initial_update=128):
        super().__init__(bank, coefficient=coefficient, seed=seed, batch_size=batch_size)
        self.window_updates = int(window_updates)
        self.capacity = int(capacity)
        if self.window_updates != 8 or self.capacity != 2048:
            raise ValueError('X self imitation requires eight updates and 2048 capacity')
        self.records = [dict(action=self.actions[i].clone(), condition=self.conditions[i].clone(),
                             weight=self.weights[i].clone(), update=int(initial_update),
                             task_id=str(bank['task_ids'][i]), source=str(bank['sources'][i]),
                             transition_id='initial:{}'.format(i), episode_id='initial')
                        for i in range(len(self.actions))]
        self._rebuild()

    def _rebuild(self):
        if not self.records:
            self.actions = torch.empty(0,16,11)
            self.conditions = torch.empty(0,576)
            self.weights = torch.empty(0,16,11)
        else:
            self.actions = torch.stack([r['action'] for r in self.records])
            self.conditions = torch.stack([r['condition'] for r in self.records])
            self.weights = torch.stack([r['weight'] for r in self.records])
        self.tasks = {}
        for i,r in enumerate(self.records):
            self.tasks.setdefault(r['task_id'],{}).setdefault(r['source'],[]).append(i)
        self.task_ids = sorted(self.tasks)

    def ingest(self, buffer, update_index):
        # Collector boundaries close a fragment. No state before that boundary is
        # fabricated or retroactively labelled from a later successful fragment.
        pending = []
        for item in buffer._items:
            pending.append(item)
            if item['outcome'] == 'reached':
                for row in pending:
                    weight = executed_weights(row['outcome'], row.get('event_segment'),
                                              row.get('event_fraction'))
                    if weight.sum() == 0:
                        continue
                    if not row.get('task_id') or not row.get('episode_id'):
                        raise ValueError('self imitation requires actual rollout provenance')
                    self.records.append(dict(action=row['chain'][-1].detach().cpu().float().clone(),
                        condition=row['condition']['global_cond'].detach().cpu().float().clone(),
                        weight=weight, update=int(update_index), task_id=str(row['task_id']),
                        source='self', transition_id=row['transition_id'], episode_id=row['episode_id']))
            if item['terminated'] or item['truncated']:
                pending = []
        self.records = [r for r in self.records
                        if r['update'] > int(update_index)-self.window_updates][-self.capacity:]
        self._rebuild()
        return dict(size=len(self.records), content_sha256=_content_hash(self.records))

    def loss(self, runtime, *, device):
        if not self.records:
            return sum(parameter.sum()*0. for parameter in runtime.actor_ft.parameters()
                       if parameter.requires_grad)
        return super().loss(runtime, device=device)

    def state_dict(self):
        state = super().state_dict()
        state.update(records=[{k:(v.clone() if isinstance(v,torch.Tensor) else v)
                               for k,v in r.items()} for r in self.records],
                     content_sha256=_content_hash(self.records),
                     window_updates=self.window_updates, capacity=self.capacity)
        return state

    def load_state_dict(self, state):
        if (state['window_updates'],state['capacity']) != (self.window_updates,self.capacity):
            raise ValueError('self imitation configuration differs')
        if _content_hash(state['records']) != state['content_sha256']:
            raise ValueError('self imitation content hash mismatch')
        super().load_state_dict(state)
        self.records = [{k:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v)
                         for k,v in r.items()} for r in state['records']]
        self._rebuild()
