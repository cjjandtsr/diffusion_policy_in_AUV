"""Training-origin executed-prefix x0 supervision; never an action-distribution KL."""
import math
from collections import Counter

import numpy as np
import torch

from .official_loss_components import AdvantageStats, compute_official_components


def executed_weights(reason, segment=None, fraction=None):
    weights = torch.zeros(16, 11, dtype=torch.float32)
    if reason == 'none':
        completed, partial = 4, 0.
    elif reason == 'reached':
        if segment not in range(4) or fraction is None or not 0 <= fraction <= 1:
            raise ValueError('invalid arrival event')
        completed, partial = int(segment), float(fraction)
        if partial == 1.:
            completed, partial = completed + 1, 0.
    else:
        raise ValueError('only safe successful-episode prefixes are eligible')
    weights[2:2+completed, [0, 1, 2, 9]] = 1.
    if partial:
        # Linear interpolation: (fraction * displacement error)^2.
        # No claim that the unexecuted endpoint or its yaw was reached.
        weights[2+completed, :3] = partial * partial
    return weights


def paired_inventory(episodes):
    tasks = {}
    for row in episodes:
        pair = tasks.setdefault(row['task_id'], {})
        if row['source'] not in ('bc', 'u128') or row['source'] in pair:
            raise ValueError('duplicate or unknown paired source')
        pair[row['source']] = row
    counts = dict.fromkeys(('both_success', 'bc_only', 'rl_only', 'neither'), 0)
    available = dict.fromkeys(('both_success', 'bc_only', 'rl_only'), 0)
    decisions = dict(available)
    categories = {}
    empty_success = 0
    for task, pair in tasks.items():
        if set(pair) != {'bc', 'u128'}:
            raise ValueError('incomplete task pairing')
        bc, rl = bool(pair['bc']['reached']), bool(pair['u128']['reached'])
        category = ('both_success' if bc and rl else 'bc_only' if bc else
                    'rl_only' if rl else 'neither')
        counts[category] += 1
        categories[task] = category
        n = sum(int(r['samples']) for r in pair.values() if r['reached'])
        empty_success += sum(r['reached'] and not r['samples'] for r in pair.values())
        if category != 'neither':
            available[category] += int(n > 0)
            decisions[category] += n
    total = sum(available.values())
    return dict(task_counts=counts, available_tasks=available, decisions=decisions,
                categories=categories, success_episodes_without_samples=empty_success,
                sampling_weights={k: v/total if total else 0. for k,v in available.items()})


class SelectiveKeep:
    def __init__(self, bank, *, coefficient, seed, batch_size=32):
        self.coefficient = float(coefficient)
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError('invalid keep coefficient')
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError('empty reference batch')
        self.generator = torch.Generator().manual_seed(int(seed))
        self.actions = torch.as_tensor(bank['actions']).detach().cpu().float()
        self.conditions = torch.as_tensor(bank['conditions']).detach().cpu().float()
        self.weights = torch.as_tensor(bank['weights']).detach().cpu().float()
        n = len(self.actions)
        if (not n or self.actions.shape != (n,16,11) or self.weights.shape != (n,16,11)
                or self.conditions.shape != (n,576)):
            raise ValueError('invalid reference tensor shapes')
        for x in (self.actions, self.conditions, self.weights):
            if not torch.isfinite(x).all(): raise ValueError('nonfinite reference')
        allowed = executed_weights('none').bool()
        if ((self.weights < 0).any() or (self.weights > 1).any()
                or self.weights[:, ~allowed].any() or (self.weights.sum((1,2)) <= 0).any()):
            raise ValueError('invalid executed supervision weights')
        if any(len(bank[k]) != n for k in ('task_ids','sources','categories')):
            raise ValueError('reference metadata count differs')
        self.tasks = {}
        for i, (task, source, category) in enumerate(zip(bank['task_ids'], bank['sources'], bank['categories'])):
            if source not in ('bc','u128') or category not in ('both_success','bc_only','rl_only'):
                raise ValueError('invalid reference source/category')
            if (category == 'bc_only' and source != 'bc') or (category == 'rl_only' and source != 'u128'):
                raise ValueError('reference source disagrees with paired category')
            self.tasks.setdefault(str(task), {}).setdefault(source, []).append(i)
        self.task_ids = sorted(self.tasks)

    def draw_indices(self):
        # Equivalent to category weight proportional to distinct available tasks,
        # then uniform task/source/state. A both-success task is not doubled.
        out = []
        for _ in range(self.batch_size):
            task = self.task_ids[int(torch.randint(len(self.task_ids),(1,),generator=self.generator))]
            sources = self.tasks[task]
            source = sorted(sources)[int(torch.randint(len(sources),(1,),generator=self.generator))]
            indices = sources[source]
            out.append(indices[int(torch.randint(len(indices),(1,),generator=self.generator))])
        return torch.tensor(out, dtype=torch.long)

    def loss(self, runtime, *, device):
        if getattr(runtime, 'predict_epsilon', None) is not False:
            raise ValueError('A2 requires actual x0 FT prediction')
        indices = self.draw_indices()
        x0 = self.actions[indices].to(device)
        cond = self.conditions[indices].to(device)
        w = self.weights[indices].to(device)
        times = runtime.ddim_t[-runtime.ft_denoising_steps:].detach().cpu().long()
        positions = torch.randint(len(times),(len(indices),),generator=self.generator)
        t = times[positions].to(device)
        noise = torch.randn(x0.shape, generator=self.generator, dtype=x0.dtype).to(device)
        # E5 exposes DDIM-position coefficients, paired with the same time table.
        alpha = runtime.ddim_alphas[-runtime.ft_denoising_steps:].to(device)[positions.to(device)].to(x0.dtype).reshape(-1,1,1)
        xt = alpha.sqrt()*x0 + (1-alpha).sqrt()*noise
        pred = runtime.actor_ft(xt, t, {'state':cond, 'global_cond':cond})
        return (((pred-x0).square()*w).sum((1,2))/16.).mean()

    def state_dict(self):
        return dict(coefficient=self.coefficient, rng=self.generator.get_state().clone())

    def load_state_dict(self, state):
        coefficient = float(state['coefficient'])
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError('invalid saved coefficient')
        self.coefficient = coefficient
        self.generator.set_state(state['rng'].detach().cpu())


def gradient_vector(gradients, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
                      for p,g in zip(parameters, gradients)])


def gradient_relation(ppo, keep, coefficient):
    pn, kn = float(ppo.norm()), float(keep.norm())
    dot = float(torch.dot(ppo,keep))
    return dict(ppo_norm=pn, keep_norm=kn,
                ratio=coefficient*kn/pn if pn > 0 else None,
                cosine=dot/(pn*kn) if pn > 0 and kn > 0 else None)


def calibrate_keep(trainer, batches, keep, *, microbatch_pairs=128, target_ratio=.1):
    from .trainer import _slice_batch
    parameters = trainer.actor_parameters
    saved = keep.state_dict()
    rows = []
    try:
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            for batch in batches:
                runtime = trainer.runtime
                stats = AdvantageStats.from_tensor(batch.advantages,
                    getattr(runtime,'clip_advantage_lower_quantile',0.),
                    getattr(runtime,'clip_advantage_upper_quantile',1.))
                n = len(batch.chains_prev)
                gp = torch.zeros(sum(p.numel() for p in parameters), device=parameters[0].device)
                for start in range(0,n,microbatch_pairs):
                    end = min(n,start+microbatch_pairs)
                    parts = compute_official_components(runtime, _slice_batch(batch,start,end), stats,
                        reward_start=trainer.config.reward_start, reward_horizon=trainer.config.reward_horizon)
                    loss = (parts.policy.mean() + trainer.config.entropy_loss_coefficient*parts.entropy.mean())*((end-start)/n)
                    gp += gradient_vector(torch.autograd.grad(loss, parameters, allow_unused=True),parameters)
                loss = keep.loss(runtime,device=parameters[0].device)
                gk = gradient_vector(torch.autograd.grad(loss,parameters,allow_unused=True),parameters)
                row = gradient_relation(gp,gk,1.)
                if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ('ppo_norm','keep_norm')):
                    raise ValueError('nonfinite or zero calibration gradient')
                rows.append(row)
        if len(rows) != 4: raise ValueError('calibration requires exactly four logical batches')
        coefficient = target_ratio * float(np.median([r['ppo_norm'] for r in rows])) / float(np.median([r['keep_norm'] for r in rows]))
        for row in rows: row['ratio'] *= coefficient
        return dict(coefficient=coefficient, target_ratio=target_ratio, batches=rows,
                    gradient_semantics='actual PPO objective before final gradient-norm clipping')
    finally:
        keep.load_state_dict(saved)
