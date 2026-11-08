"""Versioned adapter for the R1 environment-owned reward transaction."""

import hashlib
from pathlib import Path

from diffusion_policy_3d.env.reward_v2_support.reward import TERM_NAMES
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.interface import RewardResult, RewardState
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v13.identity import RewardIdentity


class RewardV2Facade:
    version = "v2"
    term_names = TERM_NAMES
    offline_density = None

    def __init__(self, space_cost_budget=None):
        if space_cost_budget is not None:
            self.term_names = TERM_NAMES + ("space",)
        root = Path(__file__).resolve().parents[4] / "env" / "reward_v2_support"
        hasher = hashlib.sha256()
        for path in sorted(root.glob("*.py")):
            hasher.update(path.name.encode("utf-8") + b"\0")
            hasher.update(hashlib.sha256(path.read_bytes()).digest())
        self.identity = RewardIdentity(str(root), hasher.hexdigest(),
                                       "diffusion_policy_3d.env.reward_v2_support/source_bundle")

    def initial_state(self, config=None):
        return RewardState({})

    def accept_transition(self, total, terms, metrics, history):
        if set(terms) != set(self.term_names):
            raise RuntimeError("R1 reward term schema mismatch")
        if abs(float(total) - sum(terms.values())) > 1e-8:
            raise RuntimeError("R1 reward sum mismatch")
        return RewardResult(float(total), dict(terms), dict(metrics), RewardState(history))

    def terminal(self, *args, **kwargs):
        raise RuntimeError("R1 terminal requires full executed transition")
