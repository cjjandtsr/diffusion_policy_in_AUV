"""C1 compares actual frozen-parent PPO gradients; never submits an update."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..engine.official_loss_components import AdvantageStats, InertMask, compute_official_components
from ..engine.pair_schedule import iter_pair_passes
from ..engine.selective_keep import gradient_vector
from ..engine.trainer import _slice_batch


def actor_gradient(trainer, batch, mask, microbatch_pairs, final_logratios=None):
    runtime, parameters = trainer.runtime, trainer.actor_parameters
    stats = AdvantageStats.from_tensor(batch.advantages,
        getattr(runtime, 'clip_advantage_lower_quantile', 0.),
        getattr(runtime, 'clip_advantage_upper_quantile', 1.))
    n = len(batch.chains_prev)
    result = torch.zeros(sum(p.numel() for p in parameters), device=parameters[0].device)
    for start in range(0, n, microbatch_pairs):
        end = min(n, start+microbatch_pairs)
        parts = compute_official_components(runtime, _slice_batch(batch,start,end), stats,
            reward_start=trainer.config.reward_start, reward_horizon=trainer.config.reward_horizon,
            inert_mask=mask)
        if final_logratios is not None:
            selected=batch.denoising_inds[start:end]==runtime.ft_denoising_steps-1
            final_logratios.append(parts.logratio.detach()[selected].cpu())
        loss = (parts.policy.mean()+trainer.config.entropy_loss_coefficient*parts.entropy.mean())*((end-start)/n)
        result += gradient_vector(torch.autograd.grad(loss, parameters, allow_unused=True), parameters)
    if not torch.isfinite(result).all(): raise ValueError('nonfinite C1 gradient')
    return result


def lr_scale_for_ratio(ratio):
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('C1 needs finite nonzero paired gradients')
    return 1./ratio if ratio > 1.4 else 1.


def compare_c1(system, buffer, config, output):
    if buffer.outer_update_id != 129: raise ValueError('C1 requires fresh u129 buffer')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    buffer_path = output/'u129_buffer.pt'
    torch.save(buffer.state_dict(), buffer_path)
    rows = []
    batches = iter_pair_passes(buffer, mode=str(config.dppo.denoising_schedule),
        pair_passes=5, logical_batch_pairs=2560, microbatch_pairs=128,
        generator=torch.Generator().manual_seed(int(config.run.seed)+129*1009),
        device=system.bc_runtime.device)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    for batch in batches:
        gradients = {}
        quantiles = {}
        for name, mask in [('off', InertMask()), ('final_step', InertMask(True)),
                           ('all_steps', InertMask(True, scope='all_steps'))]:
            # Reset stochastic actor state identically for every comparison.
            with torch.random.fork_rng(devices=devices):
                values=[]
                gradients[name] = actor_gradient(system.trainer, batch, mask, 128,values)
                quantiles[name]=torch.quantile(torch.cat(values).float(),
                    torch.tensor([0.,.05,.5,.95,1.])).tolist()
        base = gradients['off']; norm = float(base.norm())
        if norm <= 0: raise ValueError('zero C1 baseline gradient')
        row = dict(batch=len(rows), pass_id=batch.pass_id, off_norm=norm,
            final_step_logratio_quantiles=quantiles, quantile_levels=[0.,.05,.5,.95,1.])
        for name in ('final_step', 'all_steps'):
            value = gradients[name]; size = float(value.norm())
            if size <= 0: raise ValueError('zero C1 masked gradient')
            row[name] = dict(norm=size, ratio=size/norm,
                cosine=float(torch.dot(base,value))/(norm*size))
        rows.append(row)
    if len(rows) != 20: raise ValueError('C1 requires twenty logical batches')
    ratio = float(np.median([r['final_step']['ratio'] for r in rows]))
    report = dict(schema='routea_xk_c1_v1', seed=int(config.run.seed), rows=rows,
        median_final_step_ratio=ratio, actor_lr_scale=lr_scale_for_ratio(ratio),
        buffer_sha256=hashlib.sha256(buffer_path.read_bytes()).hexdigest(),
        optimizer_submissions=0, semantics='frozen u128, fresh u129 buffer, PPO only before clipping')
    with (output/'c1.json').open('x',encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report
