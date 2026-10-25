from dataclasses import replace

import numpy as np

from scripts.evaluate_dppo_uav_formal_endpoint import EpisodeAccumulator, canonical_done_reason
from scripts.formal_endpoint_eval_core import PairedEpisodeRecord, PairedEvaluationBank, paired_metrics, v17_behavior_summary


def record(seed, policy):
    return PairedEpisodeRecord(task_id=str(seed), config_id=str(seed), config_hash="a" * 64,
        obstacle_hash="b" * 64, source_manifest_hash="c" * 64, policy_identity_hash="d" * 64,
        world_id="fixture", split="rl_test", role="anticipatory", seed=seed, policy=policy,
        done_reason="time_limit", raw_done_reason="none", episode_return=0., onset_triggered=False,
        onset_first_step=None, future_clearance_reward_sum=0., d_soft_h_min=1., d_soft_h_steps=1,
        support_cells=0, terminal_xyz=(0., 0., 1.), min_xyz=(0., 0., 1.), max_xyz=(0., 0., 1.),
        action_dz_mean=0., action_dz_min=0., action_dz_max=0., trap_recovery_count=0,
        loop_count=0, side_flip_count_max=0, steps=1)


def test_invalid_remains_in_success_denominator_without_fake_action_statistics():
    records = []
    for seed in range(2):
        bc = replace(record(seed, "bc"), done_reason="invalid_state", raw_done_reason="invalid_state",
                     invalid_steps=1, reward_schema_version="v17",
                     action_dz_mean=None, action_dz_min=None, action_dz_max=None,
                     trajectory_statistics_scope="partial_before_invalid")
        rl = replace(record(seed, "rl"), done_reason="reached", raw_done_reason="reached",
                     reward_schema_version="v17")
        records.extend((bc, rl))
    metrics = paired_metrics(PairedEvaluationBank(records))
    assert metrics["success"].mean == 1.
    assert metrics["invalid_state"].mean == -1.
    assert "action_dz_mean" not in metrics


def test_v17_records_intermediate_path_peak_and_onset_modes_separately():
    acc = EpisodeAccumulator([0, 0, 1])
    path = np.array([[0, 0, 1], [.1, 0, 2], [.2, 0, 3], [.3, 0, 2], [.4, 0, 1]])
    acc.add_step(reward=1., position=path[-1], executed_action=np.zeros((4, 11)),
                 reward_terms={"onset": .1}, reward_metrics={"reward_schema_version": "v17",
                 "onset_mode": "contrast", "above_demo_dwell_seconds": .6,
                 "above_demo_unknown_seconds": .6, "above_demo_stall_seconds": 0.,
                 "positive_climb_m": 2., "executed_length_m": 4.1},
                 terminal=False, executed_path=path)
    summary = acc.finish(reward_history={"eligible_cells": []})
    assert summary["max_xyz"][2] == 3.
    assert summary["v17_metrics"]["onset_contrast_steps"] == 1
    assert summary["v17_metrics"]["onset_history_steps"] == 0
    assert summary["v17_metrics"]["above_demo_stall_rate"] is None
    assert summary["trajectory_statistics_scope"] == "complete_executed_polyline"


def test_invalid_observation_does_not_enter_trajectory_summary():
    acc = EpisodeAccumulator([0, 0, 1])
    acc.add_step(reward=-50., position=[np.nan, 0, 1], executed_action=np.full((4, 11), np.nan),
                 reward_terms={}, reward_metrics={"reward_schema_version": "v17", "invalid_state": True},
                 terminal=True)
    summary = acc.finish(reward_history={})
    assert summary["action_dz_mean"] is None and summary["invalid_steps"] == 1
    assert summary["trajectory_statistics_scope"] == "partial_before_invalid"
    assert canonical_done_reason("invalid_state", float("nan"), True, False) == "invalid_state"


def test_above_demo_summary_keeps_all_episode_denominators_and_unknown_time():
    metrics = dict(above_demo_height=True, goal_above_demo_height=False,
                   above_demo_dwell_seconds=2., above_demo_unknown_seconds=1., above_demo_stall_seconds=.5)
    records = [replace(record(0, "bc"), reward_schema_version="v17", v17_metrics=metrics),
               replace(record(0, "rl"), reward_schema_version="v17", v17_metrics=metrics)]
    summary = v17_behavior_summary(PairedEvaluationBank(records))
    assert summary["by_policy"]["bc"]["episodes_above_demo_rate"] == 1.
    assert summary["by_policy"]["bc"]["above_demo_stall_rate"] == .5
    assert summary["by_policy"]["bc"]["above_demo_coverage"] == .5
