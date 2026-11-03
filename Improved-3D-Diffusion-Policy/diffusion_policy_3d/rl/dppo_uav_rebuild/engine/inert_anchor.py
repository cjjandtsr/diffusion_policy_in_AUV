"""Optional actor-only inert x0 anchor on current logical-batch pairs."""
import torch
import math
import numpy as np


class InertAnchor:
    def __init__(self, *, coefficient=0., seed=0, batch_size=128):
        self.coefficient = float(coefficient)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator().manual_seed(int(seed))

    def loss(self, runtime, batch):
        if runtime.predict_epsilon is not False:
            raise ValueError('inert anchoring requires E5 x0 prediction')
        n=len(batch.denoising_inds)
        indices=torch.randperm(n,generator=self.generator)[:self.batch_size].to(batch.chains_prev.device)
        x=batch.chains_prev[indices]
        positions=batch.denoising_inds[indices]
        t=runtime.ddim_t[-runtime.ft_denoising_steps:][positions]
        condition={k:batch.condition[k][indices] for k in ('state','global_cond')}
        with torch.no_grad():
            reference=runtime.actor(x,t,condition)
        prediction=runtime.actor_ft(x,t,condition)
        return (prediction[:,2:10,[3,4,5,6,7,8,10]]-
                reference[:,2:10,[3,4,5,6,7,8,10]]).square().mean()

    def state_dict(self):
        return dict(coefficient=self.coefficient,rng=self.generator.get_state().clone())

    def load_state_dict(self,state):
        self.coefficient=float(state['coefficient'])
        self.generator.set_state(state['rng'].detach().cpu())


def calibrate_anchor(trainer,batches,anchor,*,microbatch_pairs=128,target_ratio=.05):
    from .official_loss_components import AdvantageStats,compute_official_components
    from .selective_keep import gradient_vector,gradient_relation
    from .trainer import _slice_batch
    saved=anchor.state_dict()
    rows=[]
    parameters=trainer.actor_parameters
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            for batch in batches:
                n=len(batch.denoising_inds)
                stats=AdvantageStats.from_tensor(batch.advantages,
                    getattr(trainer.runtime,'clip_advantage_lower_quantile',0.),
                    getattr(trainer.runtime,'clip_advantage_upper_quantile',1.))
                gp=torch.zeros(sum(p.numel() for p in parameters),device=parameters[0].device)
                for start in range(0,n,microbatch_pairs):
                    end=min(start+microbatch_pairs,n)
                    parts=compute_official_components(trainer.runtime,_slice_batch(batch,start,end),stats,
                        reward_start=trainer.config.reward_start,reward_horizon=trainer.config.reward_horizon,
                        inert_mask=trainer.config.inert_mask)
                    loss=(parts.policy.mean()+trainer.config.entropy_loss_coefficient*parts.entropy.mean())*((end-start)/n)
                    gp+=gradient_vector(torch.autograd.grad(loss,parameters,allow_unused=True),parameters)
                ga=gradient_vector(torch.autograd.grad(anchor.loss(trainer.runtime,batch),parameters,allow_unused=True),parameters)
                row=gradient_relation(gp,ga,1.)
                if any(not math.isfinite(row[k]) or row[k]<=0 for k in ('ppo_norm','keep_norm')):
                    raise ValueError('nonfinite or zero anchor calibration gradient')
                rows.append(row)
        if len(rows)!=4:
            raise ValueError('anchor calibration requires four logical batches')
        coefficient=target_ratio*float(np.median([r['ppo_norm'] for r in rows]))/float(np.median([r['keep_norm'] for r in rows]))
        for row in rows:
            row['ratio']*=coefficient
        return dict(coefficient=coefficient,target_ratio=target_ratio,batches=rows)
    finally:
        anchor.load_state_dict(saved)
