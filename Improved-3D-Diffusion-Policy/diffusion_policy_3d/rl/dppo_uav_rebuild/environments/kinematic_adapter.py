"""Adapter over the existing KinematicUAVEnv with authoritative terminal facts."""

import numpy as np

class KinematicAdapter:
    def __init__(self, environment, reward, reward_state, reward_config=None):
        self.environment = environment
        self.reward = reward
        self.reward_state = reward_state
        self.reward_config = reward_config or {}

    def reset(self, *args, **kwargs):
        result = self.environment.reset(*args, **kwargs)
        if getattr(self.reward, "version", "v13") in ("v17", "v2"):
            self.reward_state = type(self.reward_state)(self.environment._reward_history)
            return result
        self.reward_state = self.reward.initial_state(self.reward_config)
        self.environment._reward_history = self.reward_state.history
        return result

    def step(self, action_bundle):
        # The legacy environment consumes the unnormalized 8-step policy chunk
        # and advances its first four waypoints.
        action = np.asarray(action_bundle.policy.detach().cpu() if hasattr(action_bundle.policy, "detach") else action_bundle.policy)
        if action.ndim == 3:
            if action.shape[0] != 1:
                raise ValueError("one KinematicAdapter instance accepts batch size 1")
            action = action[0]
        obs, dense_reward, _, legacy_truncated, info = self.environment.step(action)

        if getattr(self.reward, "version", "v13") in ("v17", "v2"):
            facts = info["transition_facts"]
            result = self.reward.accept_transition(dense_reward, info["reward_terms"],
                                                    info["reward_metrics"], self.environment._reward_history)
            self.reward_state = result.state
            return obs, result.total, facts.terminated, facts.truncated, info

        try:
            facts = info["transition_facts"]
        except KeyError as exc:
            raise RuntimeError(
                "environment must publish authoritative transition_facts"
            ) from exc
        if bool(legacy_truncated) != bool(facts.truncated):
            raise RuntimeError("environment truncation disagrees with transition_facts")

        if facts.terminated:
            terminal = self.reward.terminal(
                facts.done_reason, self.reward_state, self.reward_config
            )
            reward_value = terminal.total
            self.reward_state = terminal.state
            info["reward_terms"] = terminal.terms
            info["reward_metrics"] = terminal.metrics
        else:
            reward_value = float(dense_reward)
            history = getattr(self.environment, "_reward_history", None)
            if history is not None:
                self.reward_state = type(self.reward_state)(history)
        info["transition_facts"] = facts
        info["done_reason"] = facts.done_reason
        return obs, reward_value, facts.terminated, facts.truncated, info
