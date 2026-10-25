from collections import deque
from types import ModuleType

import numpy as np

from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.interface import (
    RewardContext,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v13.facade import (
    RewardV13Facade,
)


def _module():
    module = ModuleType("fake_reward_v13")
    module.__file__ = __file__
    module.REWARD_V13_DEFAULTS = {
        "trap_window": 6,
        "loop_window": 8,
        "n_base": 10,
        "flip_episode_floor": -5.0,
        "r_reach": 100.0,
        "r_collision": 50.0,
    }
    module.create_reward_history = lambda **kwargs: {
        "progress": deque(),
        "episode_caps": {"flip": -5.0},
    }
    module.compute_episode_terminal_support_bonus = lambda *args, **kwargs: 4.0

    def compute(*args, history, **kwargs):
        terms = {
            "base": 1.0,
            "pbrs": 2.0,
            "onset": 0.0,
            "future_clearance": 0.0,
            "trap": 0.0,
            "flip": 0.0,
            "support": 0.0,
            "guard": 0.0,
        }
        return 3.0, terms, {"delegated": True}, history

    module.compute_reward_v13 = compute
    return module


def test_facade_delegates_eight_term_reward_and_collision_has_no_reach_bonus():
    facade = RewardV13Facade(_module())
    state = facade.initial_state()
    context = RewardContext(
        state={},
        action_chunk=np.zeros((8, 11), np.float32),
        next_state={},
        goal_world=np.zeros(3),
        obstacles=[],
    )
    dense = facade.compute(context, state)
    collision = facade.terminal("collision", dense.state)

    assert dense.total == 3.0 and dense.metrics["delegated"]
    assert collision.total == -50.0
    assert collision.terms["support"] == 0.0
    assert collision.terms["base"] < 0.0

