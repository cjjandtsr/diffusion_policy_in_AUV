"""Fixed original-protocol action drift; all probe randomness is isolated."""
import hashlib
import importlib
from pathlib import Path

import torch

from .plan_probe import _probe_rng


class DriftMonitor:
    def __init__(self, runtime, normalizer, bank_path, bank_sha256):
        path=Path(bank_path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=bank_sha256:
            raise ValueError('drift probe bank hash mismatch')
        bank=torch.load(path,map_location='cpu',weights_only=True)
        if bank.get('schema')!='routea_drift_probe_v1':
            raise ValueError('invalid drift probe bank schema')
        self.runtime=runtime
        self.normalizer=normalizer
        self.bank_sha256=bank_sha256
        self.conditions={k:torch.as_tensor(bank['conditions'][k]).detach().cpu()
                         for k in ('state','global_cond')}
        self.seeds=list(bank['seeds'])
        if len(self.seeds)!=256 or any(x.shape!=(256,576) for x in self.conditions.values()):
            raise ValueError('drift bank requires 256 fixed conditions')
        self.parent_full=(torch.as_tensor(bank['parent_full']).detach().cpu().clone()
                          if 'parent_full' in bank else None)
        if self.parent_full is not None and (self.parent_full.shape!=(256,16,11)
                                             or not torch.isfinite(self.parent_full).all()):
            raise ValueError('invalid parent physical drift baseline')

    @torch.no_grad()
    def sample(self):
        runtime=self.runtime
        device=runtime.ddim_t.device
        values=[]
        override=importlib.import_module(runtime.__class__.__module__).sampling_protocol_override
        with override(runtime,min_std_vector=[.1]*11,denoised_clip_value=1.):
            for i,seed in enumerate(self.seeds):
                condition={k:v[i:i+1].to(device) for k,v in self.conditions.items()}
                with _probe_rng(int(seed),device):
                    sample=runtime(cond=condition,deterministic=True,return_chain=True,use_base_policy=False)
                    full=self.normalizer['action'].unnormalize(sample.trajectories)
                values.append(full.detach().cpu())
        result=torch.cat(values)
        if result.shape!=(256,16,11) or not torch.isfinite(result).all():
            raise ValueError('drift probe must produce finite physical actions')
        return result

    def begin(self):
        self.start=self.sample()
        if self.parent_full is None:
            self.parent_full=self.start.clone()
        self.intra=[]
        cumulative=self._difference(self.parent_full,self.start)
        self.start_metrics=dict(step_h4_rms_m=0.,step_inert_norm=0.,
                               **{'cum_'+k:v for k,v in cumulative.items()})

    def _difference(self, before, after):
        difference=after-before
        endpoint=difference[:,2:6,:3].sum(1)
        std=self.normalizer['action'].get_input_stats()['std'].detach().cpu()
        channels=[3,4,5,6,7,8,10]
        selected=std.reshape(-1)[channels]
        if not torch.isfinite(selected).all() or (selected<=0).any():
            raise ValueError('demo std must be positive for normalized inert drift')
        return dict(h4_rms_m=float(endpoint.square().sum(1).mean().sqrt()),
                    inert_norm=float((difference[:,2:10,channels].abs()/selected).mean()))

    def measure(self, committed=None):
        current=self.sample()
        step=self._difference(self.start,current)
        cumulative=self._difference(self.parent_full,current)
        result={**{'step_'+k:v for k,v in step.items()},
                **{'cum_'+k:v for k,v in cumulative.items()},
                'bank_sha256':self.bank_sha256}
        if committed is not None:
            self.intra.append(dict(committed=committed,**result))
        return result

    def finish(self):
        return dict(self.measure(),start=self.start_metrics,intra=self.intra)

    def state_dict(self):
        return dict(bank_sha256=self.bank_sha256,parent_full=self.parent_full)

    def load_state_dict(self,state):
        if state['bank_sha256']!=self.bank_sha256:
            raise ValueError('resume drift probe bank hash differs')
        saved=state['parent_full'].detach().cpu()
        if self.parent_full is not None and not torch.equal(self.parent_full,saved):
            raise ValueError('resume drift parent baseline differs')
        self.parent_full=saved.clone()
