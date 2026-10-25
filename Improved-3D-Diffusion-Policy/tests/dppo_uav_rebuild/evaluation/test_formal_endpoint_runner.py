import numpy as np
import pytest

from scripts.evaluate_dppo_uav_formal_endpoint import (
    EpisodeAccumulator,
    canonical_done_reason,
    canonical_episode_config,
    reset_fixed_episode,
)


class _FixedResetEnv:
    def __init__(self):
        self._done_reason = "collision"
        self.mid_state_init_ratio = 0.3
        self.disturb_enable = True
        self.z_min = 1.0
        self.z_max = 5.0
        self.z_target = 3.0
        self._sample_start_goal = self._random_start_goal

    @staticmethod
    def _random_start_goal():
        return np.ones(3), 1.0, np.ones(3) * 2.0

    def reset(self, seed=None, options=None):
        fixed = (options or {}).get("fixed_start_goal")
        if fixed:
            start, yaw, goal = fixed["start_pos"], fixed["start_yaw"], fixed["goal_pos"]
        else:
            start, yaw, goal = self._sample_start_goal()
        self._drone_pos = np.asarray(start, dtype=np.float32)
        self._drone_yaw = float(yaw)
        self.goal_w = np.asarray(goal, dtype=np.float32)
        return {"seed": seed}, {"world_id": "world"}


def _config():
    return {
        "episode_id": "routeA_rl_v1__anticipatory__rl_test__f__000001__cfg00",
        "world_name": "routeA_rl_v1__anticipatory__rl_test__f__000001",
        "goal_pos": [8.0, 1.0, 2.0],
        "start_pos": [-4.0, 0.0, 1.5],
        "start_yaw": -0.25,
        "z_range": [1.0, 5.0],
        "disturb_enable": False,
        "role": "anticipatory",
    }


def test_fixed_episode_reset_applies_recorded_state_and_clears_stale_reason():
    env = _FixedResetEnv()
    original_sampler = env._sample_start_goal

    obs, info = reset_fixed_episode(env, _config(), seed=123)

    assert obs == {"seed": 123}
    assert info == {"world_id": "world"}
    np.testing.assert_allclose(env._drone_pos, [-4.0, 0.0, 1.5])
    np.testing.assert_allclose(env.goal_w, [8.0, 1.0, 2.0])
    assert env._drone_yaw == pytest.approx(-0.25)
    assert env._done_reason == "none"
    assert env.mid_state_init_ratio == 0.0
    assert env.disturb_enable is False
    assert env._sample_start_goal == original_sampler


def test_config_seed_and_hash_are_stable_and_content_bound():
    first = canonical_episode_config(_config())
    second = canonical_episode_config(dict(reversed(list(_config().items()))))
    changed = dict(_config(), start_yaw=0.5)

    assert first == second
    assert first.config_hash != canonical_episode_config(changed).config_hash
    assert first.seed == int(first.config_hash[:8], 16)


@pytest.mark.parametrize(
    "raw,z,terminated,truncated,expected",
    [
        ("reached", 2.0, True, False, "reached"),
        ("collide(box,clr=-0.1)", 2.0, True, False, "collision"),
        ("penetrate_flyover(box)", 2.0, True, False, "collision"),
        ("collision", 0.1, True, False, "crash_ground"),
        ("out_of_bounds", 2.0, True, False, "out_of_bounds"),
        ("altitude_exceed", 7.0, True, False, "altitude_exceed"),
        ("none", 2.0, False, True, "time_limit"),
    ],
)
def test_terminal_classification_preserves_project_failure_modes(
    raw, z, terminated, truncated, expected
):
    assert canonical_done_reason(raw, z, terminated, truncated) == expected


def test_episode_accumulator_uses_real_reward_v13_fields_and_executed_dz():
    acc = EpisodeAccumulator(initial_position=np.array([0.0, 0.0, 1.0]))
    acc.add_step(
        reward=1.25,
        position=np.array([1.0, -1.0, 1.5]),
        executed_action=np.array([[0.0, 0.0, -0.2], [0.0, 0.0, 0.4]]),
        reward_terms={"future_clearance": 0.75},
        reward_metrics={
            "onset_active": True,
            "d_soft_h": 0.6,
            "trap_recovered": True,
            "loop_detected": True,
            "side_flip_count": 2,
        },
        terminal=False,
    )
    acc.add_step(
        reward=-0.5,
        position=np.array([2.0, 0.5, 1.2]),
        executed_action=np.array([[0.0, 0.0, 0.1]]),
        reward_terms={"future_clearance": -0.25},
        reward_metrics={
            "onset_active": False,
            "d_soft_h": 0.0,
            "trap_recovered": False,
            "loop_detected": False,
            "side_flip_count": 1,
        },
        terminal=True,
    )
    summary = acc.finish(reward_history={"new_support_cells_this_episode": {"a", "b"}})

    assert summary["episode_return"] == pytest.approx(0.75)
    assert summary["onset_first_step"] == 1
    assert summary["future_clearance_reward_sum"] == pytest.approx(0.5)
    assert summary["d_soft_h_min"] == pytest.approx(0.6)
    assert summary["d_soft_h_steps"] == 1
    assert summary["support_cells"] == 2
    assert summary["min_xyz"] == (0.0, -1.0, 1.0)
    assert summary["max_xyz"] == (2.0, 0.5, 1.5)
    assert summary["action_dz_mean"] == pytest.approx(0.1)
    assert summary["action_dz_min"] == pytest.approx(-0.2)
    assert summary["action_dz_max"] == pytest.approx(0.4)
    assert summary["trap_recovery_count"] == 1
    assert summary["loop_count"] == 1
    assert summary["side_flip_count_max"] == 2
