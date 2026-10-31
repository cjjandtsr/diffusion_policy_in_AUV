"""Thin delegation to the existing corrected Reward v1.3 NumPy code."""

from __future__ import annotations

import importlib

from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.interface import (
    RewardContext,
    RewardResult,
    RewardState,
)
from .identity import identify_reward_module


TERM_NAMES = (
    "base", "pbrs", "onset", "future_clearance", "trap", "flip", "support", "guard"
)


class RewardV13Facade:
    def __init__(self, module=None):
        self.module = module or importlib.import_module(
            "diffusion_policy_3d.env.reward_v13"
        )
        self.identity = identify_reward_module(self.module)

    def initial_state(self, config=None) -> RewardState:
        config = {**self.module.REWARD_V13_DEFAULTS, **(config or {})}
        return RewardState(
            self.module.create_reward_history(
                trap_window=config["trap_window"],
                loop_window=config["loop_window"],
                n_base=config["n_base"],
                flip_episode_floor=config["flip_episode_floor"],
            )
        )

    def compute(self, context: RewardContext, state: RewardState) -> RewardResult:
        total, terms, metrics, history = self.module.compute_reward_v13(
            context.state,
            context.action_chunk,
            context.next_state,
            pointcloud=context.pointcloud,
            obstacle_json=context.obstacles,
            goal_world=context.goal_world,
            history=state.history,
            offline_density=context.offline_density,
            cfg=context.config,
        )
        if set(terms) != set(TERM_NAMES):
            raise RuntimeError("Reward v1.3 term set changed: {}".format(sorted(terms)))
        return RewardResult(float(total), dict(terms), dict(metrics), RewardState(history))

    def terminal(self, reason: str, state: RewardState, config=None) -> RewardResult:
        config = {**self.module.REWARD_V13_DEFAULTS, **(config or {})}
        terms = {name: 0.0 for name in TERM_NAMES}
        if reason == "reached":
            terms["base"] = float(config["r_reach"])
            terms["support"] = float(
                self.module.compute_episode_terminal_support_bonus(
                    state.history, config, reached=True
                )
            )
        elif reason in (
            "collision",
            "ground_crash",
            "out_of_bounds",
            "altitude_exceed",
            "invalid_state",
            "early_death",
        ):
            terms["base"] = -float(config["r_collision"])
        else:
            raise ValueError("terminal reward requires a canonical terminal reason")
        return RewardResult(
            total=float(sum(terms.values())),
            terms=terms,
            metrics={"terminated": True, "done_reason": reason},
            state=state,
        )
