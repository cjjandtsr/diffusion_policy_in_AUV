"""F12 R12' facade: v2 transaction contract plus a stall term; newbest gated."""
import hashlib
import json
import math
from pathlib import Path

from ..v2.facade import RewardV2Facade
from ..v13.identity import RewardIdentity

TERMINALS = (100., 0., -50., -100.)


class RewardV6Facade(RewardV2Facade):
    version = 'v6'

    def __init__(self, space_cost_budget=None, **params):
        from diffusion_policy_3d.env.reward_v6 import DEFAULTS
        super().__init__(space_cost_budget)
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError('unknown v6 parameters: {}'.format(sorted(unknown)))
        self.params = {**DEFAULTS, **{k: type(DEFAULTS[k])(v) for k, v in params.items()}}
        self.term_names = self.term_names + ('stall',)
        source = Path(__file__).resolve().parents[4] / 'env' / 'reward_v6.py'
        h = hashlib.sha256(self.identity.source_sha256.encode('ascii') + source.read_bytes())
        h.update(json.dumps(self.params, sort_keys=True).encode('ascii'))
        self.identity = RewardIdentity(str(source), h.hexdigest(), 'v2_newbest_gate_collision100_stall')

    def accept_transition(self, total, terms, metrics, history):
        p = self.params
        nb, stall, term = float(terms['newbest']), float(terms['stall']), float(terms['terminal'])
        if not (math.isfinite(nb) and -1e-12 <= nb <= 10. + 1e-9):
            raise RuntimeError('v6 newbest bound violation')
        if stall not in (0., float(p['stall_penalty'])):
            raise RuntimeError('v6 stall value violation')
        if term not in TERMINALS:
            raise RuntimeError('v6 terminal value violation')
        return super().accept_transition(total, terms, metrics, history)
