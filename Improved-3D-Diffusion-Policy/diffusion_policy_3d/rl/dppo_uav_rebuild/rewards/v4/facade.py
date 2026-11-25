"""Versioned v4 facade reusing the unchanged v2 transaction contract."""
import hashlib
import math
from pathlib import Path

from ..v2.facade import RewardV2Facade
from ..v13.identity import RewardIdentity


class RewardV4Facade(RewardV2Facade):
    version = 'v4'

    def __init__(self, space_cost_budget=None):
        super().__init__(space_cost_budget)
        self.term_names = self.term_names + ('goal_potential',)
        source = Path(__file__).resolve().parents[4] / 'env' / 'reward_v4.py'
        hasher = hashlib.sha256()
        hasher.update(self.identity.source_sha256.encode('ascii'))
        hasher.update(source.read_bytes())
        self.identity = RewardIdentity(str(source), hasher.hexdigest(),
            'diffusion_policy_3d.env.reward_v4/v2_plus_goal_potential')

    def accept_transition(self, total, terms, metrics, history):
        value = float(terms.get('goal_potential', float('nan')))
        if not math.isfinite(value) or not -10. <= value <= 9.9 + 1e-9:
            raise RuntimeError('v4 goal_potential bound violation')
        return super().accept_transition(total, terms, metrics, history)
