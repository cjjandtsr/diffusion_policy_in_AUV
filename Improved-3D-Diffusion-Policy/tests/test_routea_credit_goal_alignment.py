"""Checks for the Route A goal-alignment and credit-chain analysis."""

import pytest

from scripts.analyze_routea_credit_goal_alignment import (
    recompute_finite_horizon_gae,
    summarize_counterfactual_grid,
    summarize_endpoint_fact_checks,
    summarize_episode_rewards,
    summarize_transition_credit,
)


def _record(*, raw, normalized, progress, outcome="time_limit",
            done_reason="none", guard=False):
    return {
        "advantage_raw": raw,
        "advantage_normalized_outer_update": normalized,
        "net_goal_approach_3d": progress,
        "reward": progress * 2.0,
        "reward_terms": {"progress": progress * 2.0, "future_clearance": 0.0},
        "reward_metrics": {"guard_context": guard},
        "dppo_credit": {"outcome": done_reason},
        "terminal_facts": {"done_reason": done_reason},
        "episode_id": "episode-1",
        "role": "dense_recovery",
    }


def test_transition_summary_exposes_normalization_sign_flip_on_away_timeout():
    records = [
        _record(raw=-2.0, normalized=0.5, progress=-0.2, guard=True),
        _record(raw=1.0, normalized=1.0, progress=0.3),
        _record(raw=-5.0, normalized=-1.0, progress=-0.1,
                done_reason="collision"),
    ]

    result = summarize_transition_credit(records)

    assert result["away_transition_count"] == 2
    assert result["away_positive_normalized_rate"] == 0.5
    assert result["away_raw_nonpositive_to_normalized_positive_rate"] == 0.5
    assert result["guard_away_positive_normalized_count"] == 1


def test_transition_summary_keeps_timeout_terminal_separate_from_collection_boundary():
    timeout = _record(raw=-1.0, normalized=0.25, progress=-0.1,
                      done_reason="time_limit")
    boundary = _record(raw=-1.0, normalized=0.25, progress=-0.1)
    boundary["collection_boundary"] = True

    result = summarize_transition_credit([timeout, boundary])

    assert result["time_limit_terminal_count"] == 1
    assert result["time_limit_terminal_positive_normalized_rate"] == 1.0


def test_endpoint_fact_check_counts_nonreached_entries_without_relabeling_success():
    rows = [
        {"seed": "seed3", "policy": "rl", "done": "time_limit",
         "entered_goal_0p5": "True"},
        {"seed": "seed3", "policy": "rl", "done": "time_limit",
         "entered_goal_0p5": "False"},
        {"seed": "seed3", "policy": "rl", "done": "reached",
         "entered_goal_0p5": "True"},
    ]

    result = summarize_endpoint_fact_checks(rows)

    assert result["seed3"]["rl"]["episodes"] == 3
    assert result["seed3"]["rl"]["nonreached_entered_goal_0p5"] == 1
    assert result["seed3"]["rl"]["reported_reached"] == 1


def test_finite_horizon_counterfactual_cuts_timeout_bootstrap_and_propagates_penalty():
    records = [
        {
            "reward": 1.0, "value": 0.0, "next_value": 2.0,
            "terminated": False, "truncated": False,
            "collection_boundary": False,
            "terminal_facts": {"done_reason": "none"},
        },
        {
            "reward": 0.0, "value": 2.0, "next_value": 3.0,
            "terminated": False, "truncated": True,
            "collection_boundary": False,
            "terminal_facts": {"done_reason": "time_limit"},
        },
    ]

    current = recompute_finite_horizon_gae(records, timeout_penalty=0.0,
                                           cut_timeout_bootstrap=False)
    finite = recompute_finite_horizon_gae(records, timeout_penalty=0.0,
                                          cut_timeout_bootstrap=True)
    penalized = recompute_finite_horizon_gae(records, timeout_penalty=5.0,
                                             cut_timeout_bootstrap=True)

    assert current == pytest.approx([3.892285, 0.97])
    assert finite == pytest.approx([1.099, -2.0])
    assert penalized == pytest.approx([-3.6035, -7.0])


def test_counterfactual_grid_verifies_recorded_gae_and_reports_timeout_shift():
    records = [
        {
            "reward": 1.0, "value": 0.0, "next_value": 2.0,
            "terminated": False, "truncated": False,
            "collection_boundary": False,
            "terminal_facts": {"done_reason": "none"},
            "net_goal_approach_3d": -0.1,
            "advantage_raw": 3.892285,
        },
        {
            "reward": 0.0, "value": 2.0, "next_value": 3.0,
            "terminated": False, "truncated": True,
            "collection_boundary": False,
            "terminal_facts": {"done_reason": "time_limit"},
            "net_goal_approach_3d": -0.1,
            "advantage_raw": 0.97,
        },
    ]

    result = summarize_counterfactual_grid(records, penalties=(0.0, 5.0))

    assert result[0]["current_gae_max_abs_error"] == pytest.approx(0.0)
    assert result[0]["timeout_terminal_raw_advantage_mean"] == pytest.approx(-2.0)
    assert result[1]["timeout_terminal_raw_advantage_mean"] == pytest.approx(-7.0)


def test_episode_reward_summary_keeps_outcomes_and_terms_separate():
    first = _record(raw=0.0, normalized=0.0, progress=1.0)
    first["reward"] = 3.0
    first["reward_terms"] = {
        "terminal": 0.0, "progress": 2.0, "future_clearance": 1.0,
    }
    second = _record(raw=0.0, normalized=0.0, progress=-0.5,
                     done_reason="time_limit")
    second["reward"] = -1.0
    second["reward_terms"] = {
        "terminal": 0.0, "progress": -1.0, "future_clearance": 0.0,
    }
    reached = _record(raw=0.0, normalized=0.0, progress=0.5,
                      done_reason="reached")
    reached["episode_id"] = "episode-2"
    reached["reward"] = 101.0
    reached["reward_terms"] = {
        "terminal": 100.0, "progress": 1.0, "future_clearance": 0.0,
    }

    result = summarize_episode_rewards([first, second, reached])

    assert result["time_limit"]["episode_count"] == 1
    assert result["time_limit"]["episode_return_mean"] == pytest.approx(2.0)
    assert result["time_limit"]["progress_reward_mean"] == pytest.approx(1.0)
    assert result["time_limit"]["future_clearance_reward_mean"] == pytest.approx(1.0)
    assert result["reached"]["episode_count"] == 1
    assert result["reached"]["terminal_reward_mean"] == pytest.approx(100.0)
