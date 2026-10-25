import pytest

from scripts.formal_endpoint_eval_core import (
    PairedEpisodeRecord,
    PairedEvaluationBank,
    build_endpoint_report,
    judge_endpoint_effect,
    paired_metrics,
    write_endpoint_artifacts,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _record(
    task,
    seed,
    policy,
    *,
    done_reason="time_limit",
    raw_done_reason=None,
    value=0.0,
    config_hash=HASH_A,
    obstacle_hash="f" * 64,
):
    raw_by_reason = {
        "reached": "reached",
        "collision": "collide(box,clr=-0.1)",
        "crash_ground": "crash_ground",
        "out_of_bounds": "out_of_bounds",
        "altitude_exceed": "altitude_exceed",
        "time_limit": "none",
    }
    return PairedEpisodeRecord(
        task_id=task,
        config_id=task + "__config",
        config_hash=config_hash,
        obstacle_hash=obstacle_hash,
        source_manifest_hash=HASH_B,
        policy_identity_hash=HASH_C if policy == "bc" else "d" * 64,
        world_id="routea_rl_v1__anticipatory__rl_test__000001",
        split="rl_test",
        role="anticipatory",
        seed=seed,
        policy=policy,
        done_reason=done_reason,
        raw_done_reason=raw_done_reason or raw_by_reason.get(done_reason, done_reason),
        episode_return=value,
        onset_triggered=True,
        onset_first_step=4,
        future_clearance_reward_sum=value,
        d_soft_h_min=0.4,
        d_soft_h_steps=1,
        support_cells=int(max(value, 0.0)),
        terminal_xyz=(1.0, 2.0, 3.0),
        min_xyz=(0.0, 0.0, 1.0),
        max_xyz=(2.0, 3.0, 4.0),
        action_dz_mean=0.1,
        action_dz_min=-0.2,
        action_dz_max=0.3,
        trap_recovery_count=0,
        loop_count=0,
        side_flip_count_max=0,
        steps=10,
    )


def test_endpoint_judgment_detects_collision_to_early_death_transfer():
    records = []
    for seed in range(8):
        records.extend(
            [
                _record(
                    "task" + str(seed), seed, "bc", done_reason="collision", value=-2.0
                ),
                _record(
                    "task" + str(seed),
                    seed,
                    "rl",
                    done_reason="out_of_bounds",
                    value=-2.0,
                ),
            ]
        )

    metrics = paired_metrics(PairedEvaluationBank(records))
    judgment = judge_endpoint_effect(metrics)

    assert metrics["collision"].mean == -1.0
    assert metrics["early_death"].mean == 1.0
    assert metrics["combined_failure"].mean == 0.0
    assert metrics["success"].mean == 0.0
    assert judgment.status == "failure_mode_transfer"
    assert judgment.start_additional_seed is False
    assert judgment.start_gazebo_px4 is False


def test_endpoint_report_preserves_scope_and_failure_mode_judgment():
    records = []
    for seed in range(4):
        records.extend(
            [
                _record(
                    "task" + str(seed), seed, "bc", done_reason="collision", value=-2.0
                ),
                _record(
                    "task" + str(seed),
                    seed,
                    "rl",
                    done_reason="crash_ground",
                    value=-2.0,
                ),
            ]
        )

    report = build_endpoint_report(PairedEvaluationBank(records))

    assert report["audit_schema_version"] == 1
    assert report["evidence_scope"] == "lightweight_kinematic"
    assert report["gazebo_px4_evidence"] is False
    assert report["pair_count"] == 4
    assert report["metrics"]["early_death"]["mean"] == 1.0
    assert report["policy_aggregate"]["rl"]["rates"]["crash_ground"] == 1.0
    assert report["policy_aggregate"]["rl"]["rates"]["out_of_bounds"] == 0.0
    assert report["policy_aggregate"]["rl"]["rates"]["altitude_exceed"] == 0.0
    assert report["policy_aggregate"]["rl"]["rates"]["early_death"] == 1.0
    assert report["split_aggregate"]["rl_test"]["rl"]["crash_ground_rate"] == 1.0
    assert report["judgment"]["status"] == "failure_mode_transfer"


def test_endpoint_artifacts_are_complete_and_never_overwritten(tmp_path):
    records = []
    for seed in range(2):
        records.extend(
            [
                _record(
                    "task" + str(seed), seed, "bc", done_reason="collision", value=-2.0
                ),
                _record(
                    "task" + str(seed),
                    seed,
                    "rl",
                    done_reason="altitude_exceed",
                    value=-2.0,
                ),
            ]
        )
    bank = PairedEvaluationBank(records)

    paths = write_endpoint_artifacts(tmp_path, bank)

    assert paths["episodes"].read_text(encoding="utf-8").count("\n") == 4
    assert '"failure_mode_transfer"' in paths["report"].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_endpoint_artifacts(tmp_path, bank)


def test_bank_rejects_policies_evaluated_on_different_environment_configs():
    bc = _record("task", 7, "bc", config_hash=HASH_A)
    rl = _record("task", 7, "rl", config_hash="e" * 64)

    with pytest.raises(ValueError, match="paired environment identity mismatch"):
        PairedEvaluationBank([bc, rl])


def test_bank_rejects_policies_evaluated_on_different_obstacle_bytes():
    bc = _record("task", 7, "bc", obstacle_hash="e" * 64)
    rl = _record("task", 7, "rl", obstacle_hash="f" * 64)

    with pytest.raises(ValueError, match="paired environment identity mismatch"):
        PairedEvaluationBank([bc, rl])


def test_record_rejects_unknown_terminal_outcome():
    with pytest.raises(ValueError, match="unknown canonical done_reason"):
        _record("task", 7, "bc", done_reason="early_death")


def test_metrics_use_only_keys_proven_by_reward_v13_schema():
    bc = _record("task0", 1, "bc", done_reason="collision", value=-1.0)
    rl = _record("task0", 1, "rl", done_reason="reached", value=2.0)
    bc2 = _record("task1", 2, "bc", done_reason="collision", value=-1.0)
    rl2 = _record("task1", 2, "rl", done_reason="reached", value=2.0)

    metrics = paired_metrics(PairedEvaluationBank([bc, rl, bc2, rl2]))

    assert "onset_triggered" in metrics
    assert "onset_earliness" in metrics
    assert "future_clearance_reward" in metrics
    assert "d_soft_h_min" in metrics
    assert "trap_recovery_count" in metrics
    assert "loop_count" in metrics
    assert "side_flip_count_max" in metrics
    assert "onset_distance" not in metrics
    assert "dead_end_dwell" not in metrics
