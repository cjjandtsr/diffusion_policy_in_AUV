"""F7 reward schema and source identity."""
import hashlib
import math
from pathlib import Path
from ..v2.facade import RewardV2Facade
from ..v13.identity import RewardIdentity


class RewardV5Facade(RewardV2Facade):
    version = 'v5'

    def __init__(self, space_cost_budget=None, weight=2.):
        super().__init__(space_cost_budget)
        self.approach_weight = float(weight)
        self.term_names = self.term_names + ('approach_cbf',)
        source = Path(__file__).resolve().parents[4]/'env/reward_v5.py'
        h = hashlib.sha256(self.identity.source_sha256.encode('ascii')+source.read_bytes())
        h.update(repr(self.approach_weight).encode('ascii'))
        self.identity = RewardIdentity(str(source),h.hexdigest(),'v2_plus_approach_cbf')

    def accept_transition(self, total, terms, metrics, history):
        cost = float(terms['approach_cbf'])
        if not math.isfinite(cost) or not -self.approach_weight*2.*1.2-1e-9 <= cost <= 0.:
            raise RuntimeError('v5 approach_cbf bound violation')
        return super().accept_transition(total,terms,metrics,history)
