"""Diagnostics must keep paired failures, cluster units and RNG unchanged."""

from dataclasses import replace

import numpy as np
import pytest

from scripts.formal_endpoint_eval_core import PairedEpisodeRecord, PairedEvaluationBank
from scripts.diagnose_dppo_uav_endpoint import analyze_bank, select_replay_pairs, world_ci


def record(task, policy, outcome, world="w0", split="rl_test"):
    return PairedEpisodeRecord(
        task_id=task, config_id=task, config_hash="a" * 64, obstacle_hash="b" * 64,
        source_manifest_hash="c" * 64, policy_identity_hash=("d" if policy == "bc" else "e") * 64,
        world_id=world, split=split, role="anticipatory", seed=42, policy=policy,
        done_reason=outcome, raw_done_reason=outcome, episode_return=0.,
        onset_triggered=False, onset_first_step=None, future_clearance_reward_sum=0.,
        d_soft_h_min=1., d_soft_h_steps=1, support_cells=0, terminal_xyz=(0., 0., 1.),
        min_xyz=(0., 0., 1.), max_xyz=(0., 0., 1.), action_dz_mean=0.,
        action_dz_min=0., action_dz_max=0., trap_recovery_count=0, loop_count=0,
        side_flip_count_max=0, steps=1, reward_schema_version="v17",
        v17_metrics={"executed_length_m": 1., "observed_duration_seconds": 1.2,
                     "positive_climb_m": 0., "above_demo_stall_seconds": 0.})


def bank(rows):
    records = []
    for task, bc, rl, world in rows:
        records.extend((record(task, "bc", bc, world), record(task, "rl", rl, world)))
    return PairedEvaluationBank(records)


def test_migration_counts_do_not_replace_gross_losses_with_net_losses():
    data = bank([("a", "reached", "time_limit", "w0"),
                 ("b", "reached", "collision", "w0"),
                 ("c", "collision", "reached", "w1"),
                 ("d", "time_limit", "time_limit", "w1")])
    result = analyze_bank(data)
    assert result["success_lost_pairs"] == 2
    assert result["success_gained_pairs"] == 1
    assert result["safe_arrival_delta"] == -.25
    assert result["transition_counts"]["reached"]["time_limit"] == 1
    assert result["transition_counts"]["collision"]["reached"] == 1


def test_world_ci_uses_world_means_not_replica_weights():
    # One world has three +1 replicas, the other one -1. World mean is 0, not .5.
    result = world_ci({"w0": [1., 1., 1.], "w1": [-1.]})
    assert result["mean"] == 0.
    assert result["worlds"] == 2 and result["pairs"] == 4
    assert result["lower"] == -1. and result["upper"] == 1.
    assert world_ci({"w0": [1.]})["lower"] is None


def test_selection_keeps_all_changed_outcomes_and_deterministic_controls():
    data = bank([("a", "reached", "time_limit", "w0"),
                 ("b", "collision", "reached", "w0"),
                 ("c", "collision", "time_limit", "w0"),
                 ("d", "reached", "reached", "w1"),
                 ("e", "reached", "reached", "w1"),
                 ("f", "time_limit", "time_limit", "w1")])
    first = select_replay_pairs(data)
    second = select_replay_pairs(PairedEvaluationBank(reversed(data.records)))
    assert first == second
    assert [r["task_id"] for r in first] == ["a", "b", "c", "d", "f"]


def test_matched_success_efficiency_excludes_unmatched_survivors():
    data = bank([("a", "reached", "reached", "w0"),
                 ("b", "reached", "time_limit", "w1")])
    records = [replace(r, v17_metrics=dict(r.v17_metrics, executed_length_m=2.))
               if r.policy == "rl" and r.task_id == "a" else r for r in data.records]
    result = analyze_bank(PairedEvaluationBank(records))
    assert result["both_success_efficiency"]["pairs"] == 1
    assert result["both_success_efficiency"]["length_relative_delta"]["mean"] == 1.


def test_invalid_remains_a_failure_in_migration_denominator():
    data = bank([("a", "reached", "time_limit", "w0")])
    invalid = replace(data.records[1], done_reason="invalid_state", raw_done_reason="invalid_state",
                      invalid_steps=1)
    result = analyze_bank(PairedEvaluationBank([data.records[0], invalid]))
    assert result["pairs"] == 1 and result["safe_arrival_delta"] == -1.


def test_analysis_rng_does_not_change_numpy_global_stream():
    data = bank([("a", "reached", "time_limit", "w0"),
                 ("b", "time_limit", "reached", "w1")])
    np.random.seed(17)
    before = np.random.get_state()
    analyze_bank(data)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_pointcloud_observer_preserves_real_synthesis_and_rng(tmp_path):
    import json
    import torch
    from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud, DepthCameraIntrinsics
    from scripts.endpoint_trajectory_trace import PointCloudObserver
    geometry = tmp_path / "world.json"
    geometry.write_text(json.dumps({"obstacles": []}), encoding="utf-8")
    synth = SynthPointCloud(str(geometry), device="cpu", cam_intrinsics=DepthCameraIntrinsics(width=16, height=12))
    kwargs = dict(num_points=32, add_noise=True, rng_seed=97)
    expected = synth.synthesize(np.array([0., 0., 1.]), 0., **kwargs)
    events = []
    traced = PointCloudObserver(synth, events.append)
    state = torch.get_rng_state().clone()
    actual = traced.synthesize(np.array([0., 0., 1.]), 0., **kwargs)
    assert np.array_equal(expected, actual)
    assert torch.equal(state, torch.get_rng_state())
    assert len(events) == 1 and events[0]["hit_count"] > 0
    assert events[0]["fallback"] is False and events[0]["ray_count"] == 192


def test_pointcloud_observer_counts_fallback_from_actual_hits(tmp_path):
    import json
    import sys
    from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud, DepthCameraIntrinsics
    from scripts.endpoint_trajectory_trace import PointCloudObserver
    geometry = tmp_path / "world.json"
    geometry.write_text(json.dumps({"obstacles": []}), encoding="utf-8")
    synth = SynthPointCloud(str(geometry), device="cpu", cam_intrinsics=DepthCameraIntrinsics(width=16, height=12))
    events = []
    with pytest.warns(RuntimeWarning, match="no ray hit"):
        actual = PointCloudObserver(synth, events.append).synthesize(
            np.array([0., 0., 1000.]), 0., num_points=32, rng_seed=97)
    assert events[0]["hit_count"] == 0 and events[0]["fallback"] is True
    assert np.all(actual == synth.cam.far) and sys.getprofile() is None


def test_replay_comparison_rejects_outcome_identity_and_numeric_drift():
    from scripts.endpoint_trajectory_trace import compare_record
    original = record("a", "bc", "reached")
    assert compare_record(original, original) == []
    assert compare_record(original, replace(original, done_reason="time_limit", raw_done_reason="time_limit"))
    assert compare_record(original, replace(original, obstacle_hash="f" * 64))
    assert compare_record(original, replace(original, episode_return=.01))


def test_trace_encoding_preserves_invalid_as_missing_not_zero():
    from scripts.endpoint_trajectory_trace import json_safe
    missing = []
    result = json_safe({"position": np.array([np.nan, 2., 3.])}, missing=missing)
    assert result == {"position": [None, 2., 3.]}
    assert missing == ["$.position[0]"]


def test_trace_adapter_preserves_real_h4_transition_and_observation_history(tmp_path):
    import io
    import json
    import torch
    from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
    from diffusion_policy_3d.env.reward_v17 import resolve_config
    from diffusion_policy_3d.env.reward_v17_support.density import build_density
    from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud, DepthCameraIntrinsics
    from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v17.facade import RewardV17Facade
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import KinematicAdapter
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle
    from scripts.endpoint_trajectory_trace import TraceAdapter
    geometry = tmp_path / "world.json"
    geometry.write_text(json.dumps({"obstacles": []}), encoding="utf-8")
    cfg = resolve_config(dict(gamma_pbrs=.99, d_ref=.7, s_f=.4, w_low=2.5, w_high=1., w_climb=.1))
    density = build_density([dict(episode_id="demo", split="bc_demo", source="frozen_bc",
        start=[0, 0, 1], goal=[10, 0, 1], initial_yaw=0, ground_z=0,
        positions=[[0, 0, 1]])], dataset_sha256="a" * 64)

    def make():
        factory = lambda path: SynthPointCloud(path, device="cpu", cam_intrinsics=DepthCameraIntrinsics(width=16, height=12))
        env = KinematicUAVEnv(world_pool=[("fixture", str(geometry))], synth_pc_factory=factory,
            reward_version="v17", reward_v17_cfg=cfg, offline_density=density,
            disturb_enable=False, mid_state_init_ratio=0., seed=1)
        reward = RewardV17Facade()
        return KinematicAdapter(env, reward, reward.initial_state(cfg), cfg)

    plain, wrapped = make(), make()
    stream = io.StringIO()
    traced = TraceAdapter(wrapped, stream, dict(task_id="fixture", policy="bc"))
    try:
        original, _ = plain.reset(seed=7)
        observed, _ = traced.reset(seed=7)
        np.testing.assert_array_equal(original["point_cloud"], observed["point_cloud"])
        action = ActionBundle.from_full(torch.zeros((1, 16, 11)))
        for _ in range(2):
            expected = plain.step(action)
            actual = traced.step(action)
            np.testing.assert_array_equal(expected[0]["point_cloud"], actual[0]["point_cloud"])
            np.testing.assert_array_equal(expected[0]["agent_pos"], actual[0]["agent_pos"])
            assert expected[1:4] == actual[1:4]
            assert expected[4]["reward_terms"] == actual[4]["reward_terms"]
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert len([e for e in events if e["event"] == "pointcloud"]) == 9
        transitions = [e for e in events if e["event"] == "transition"]
        assert transitions[0]["observation_before"]["actor_latest3_frame_ids"] == [0, 0, 0]
        assert transitions[0]["observation_after"]["actor_latest3_frame_ids"] == [2, 3, 4]
        assert transitions[1]["observation_before"]["actor_latest3_frame_ids"] == [2, 3, 4]
        assert len(transitions[0]["executed_path"]) == 5
        assert len(transitions[0]["reward_terms"]) == 14
        # Render a clearly labelled synthetic smoke case, never a policy result.
        from scripts.render_dppo_endpoint_traces import render_pair
        traced.emit("episode_end", summary=dict(done_reason="synthetic_smoke", steps=2,
                    episode_return=sum(t["reward"] for t in transitions)), reproduction_differences=[])
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        fixture = dict(task_id="SYNTHETIC_CPU_SMOKE", geometry={"obstacles": [
            dict(shape="box", x=2., y=2., size_x=1., size_y=2., size_z=3., yaw=.5),
            dict(shape="cylinder", x=-2., y=2., radius=.3, height=2.)]})
        index = render_pair("SYNTHETIC_CPU_SMOKE", {p: (events, fixture) for p in ("bc", "rl")}, tmp_path)
        assert (tmp_path / index["trajectory_plot"]).stat().st_size > 1000
        assert (tmp_path / index["reward_plot"]).stat().st_size > 1000
    finally:
        plain.environment.close()
        traced.close()


def test_replay_selection_rejects_removed_changed_pairs(tmp_path):
    from scripts.replay_dppo_uav_endpoint_traces import validate_selection
    from scripts.diagnose_dppo_uav_endpoint import sha256
    data = bank([("a", "reached", "time_limit", "w0")])
    hashes = {}
    for name in ("paired_episodes.jsonl", "endpoint_report.json", "evaluation_provenance.json"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        hashes[name] = sha256(path)
    selection = dict(bank_content_hash=data.content_hash, source_sha256=hashes,
                     pairs=select_replay_pairs(data), selection_count=1)
    assert validate_selection(selection, data, tmp_path) == {"a"}
    selection.update(pairs=[], selection_count=0)
    with pytest.raises(ValueError, match="deterministic complete rule"):
        validate_selection(selection, data, tmp_path)


def test_plot_trap_state_does_not_misread_terminal_cleanup_as_recovery():
    from scripts.render_dppo_endpoint_traces import evaluated_trap_state
    step = dict(reward_metrics={"trap_state": "TRAPPED"}, trap_history={"trap_state": "FREE"})
    assert evaluated_trap_state(step) == "TRAPPED"


def test_plot_keeps_collision_height_separate_from_sensor_height():
    from scripts.render_dppo_endpoint_traces import obstacle_wireframes
    geometry = {"obstacles": [dict(shape="cylinder", x=1., y=2., radius=.5, height=2., z=8.)]}
    collision = list(obstacle_wireframes(geometry))
    sensor = list(obstacle_wireframes(geometry, sensor=True))
    assert np.all(collision[0][:, 2] == 7.) and np.all(collision[1][:, 2] == 9.)
    assert np.all(sensor[0][:, 2] == 0.) and np.all(sensor[1][:, 2] == 2.)
