"""Versioned facade; terminal transitions must retain their complete reward."""

import hashlib
from pathlib import Path

from diffusion_policy_3d.env import reward_v17
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.interface import RewardState, RewardResult
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v13.identity import RewardIdentity


class RewardV17Facade:
    version = "v17"
    term_names = reward_v17.TERM_NAMES

    def __init__(self):
        root = Path(reward_v17.__file__).resolve().parent
        paths = [root / "reward_v17.py"] + sorted((root / "reward_v17_support").glob("*.py"))
        h = hashlib.sha256()
        for path in paths:
            h.update(path.relative_to(root).as_posix().encode("utf-8"))
            h.update(b"\0")
            h.update(hashlib.sha256(path.read_bytes()).digest())
        self.identity = RewardIdentity(str(root / "reward_v17.py"), h.hexdigest(),
                                       "diffusion_policy_3d.env.reward_v17/source_bundle")

    def initial_state(self, config=None):
        if config is not None:
            reward_v17.resolve_config(config)
        return RewardState(reward_v17.create_reward_history())

    def accept_transition(self, total, terms, metrics, history):
        if set(terms) != set(self.term_names):
            raise RuntimeError("v17 term schema mismatch")
        if abs(float(total) - sum(terms.values())) > 1e-8:
            raise RuntimeError("v17 reward sum mismatch")
        return RewardResult(float(total), dict(terms), dict(metrics), RewardState(history))

    def terminal(self, *args, **kwargs):
        raise RuntimeError("v17 terminal requires the full executed transition; do not overwrite terms")
