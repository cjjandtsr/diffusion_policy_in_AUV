"""Versioned v3 facade reusing the v2 transition contract."""

import hashlib
import math
from pathlib import Path

from ..v2.facade import RewardV2Facade
from ..v13.identity import RewardIdentity


class RewardV3Facade(RewardV2Facade):
    version = 'v3'

    def __init__(self, space_cost_budget=None):
        super().__init__(space_cost_budget)
        self.term_names = self.term_names + ('risk_close',)
        source = Path(__file__).resolve().parents[4] / 'env' / 'reward_v3.py'
        hasher = hashlib.sha256()
        hasher.update(self.identity.source_sha256.encode('ascii'))
        hasher.update(source.read_bytes())
        self.identity = RewardIdentity(str(source), hasher.hexdigest(),
                                       'diffusion_policy_3d.env.reward_v3/v2_plus_closing')

    def accept_transition(self, total, terms, metrics, history):
        value = float(terms.get('risk_close', float('nan')))
        if not math.isfinite(value) or not -2. <= value <= 0.:
            raise RuntimeError('v3 risk_close bound violation')
        return super().accept_transition(total, terms, metrics, history)
