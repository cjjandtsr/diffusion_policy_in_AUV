import json

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle
import scripts.analyze_routea_reward_credit_divergence as d05b
from scripts.analyze_routea_reward_credit_divergence import (
    _build_credit_batches,
    _first_sphere_entry,
    _policy_action_8x11,
    main,
)
from scripts.probe_routea_goal_sensitivity import _frozen_plan_action


def _row(task, decision, reward, *, terminated=False):
    return {
        "task_id": task,
        "decision_index": decision,
        "reward": float(reward),
        "value": 0.0,
        "next_value": 0.0,
        "terminated": bool(terminated),
        "truncated": False,
        "collection_boundary": False,
        "target": True,
    }


def _endpoint_check(policy="bc", done_reason="reached", steps=13):
    return {
        "task_id": "env_001_obs10__cfg01",
        "repeat": 0,
        "policy": policy,
        "planned_rows": steps * 8,
        "executed_rows": steps * 4 + 1,
        "trajectory_file": "trace.csv",
        "trajectory_sha256": "a" * 64,
        "actual": {
            "task_id": "env_001_obs10__cfg01",
            "policy": policy,
            "done_reason": done_reason,
            "steps": steps,
            "observation_validity_summary": {
                "last_policy_input_hash": "b" * 64,
                "last_raw_frame_hash": "c" * 64,
                "last_normalized_history_hash": "d" * 64,
                "status": "valid",
            },
        },
    }


def test_goal_sphere_label_uses_continuous_segment_not_endpoints_only():
    entered, segment, fraction = _first_sphere_entry(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
        ],
        [1.0, 0.0, 0.0],
        radius=0.5,
    )
    assert entered is True
    assert segment == 0
    assert fraction == pytest.approx(0.25)

    missed = _first_sphere_entry(
        [
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.5, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.5, 1.0, 0.0],
        ],
        [1.0, 0.0, 0.0],
        radius=0.5,
    )
    assert missed == (False, None, None)
    with pytest.raises(ValueError, match="shape"):
        _first_sphere_entry(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 0.0, 0.0],
            radius=0.5,
        )


def test_policy_action_serialization_uses_eight_step_slice_only():
    full = torch.arange(16 * 11, dtype=torch.float32).reshape(1, 16, 11)
    action = ActionBundle.from_full(full)
    serialized = _policy_action_8x11(action)
    assert len(serialized) == 8
    assert all(len(step) == 11 for step in serialized)
    assert serialized == action.policy[0].tolist()
    assert serialized != action.full[0, :8].tolist()


def test_frozen_numpy_action_bundle_serializes_policy_and_executed_views():
    """Catch tensor-only serialization on the real frozen replay backend."""

    planned = np.zeros((8, 11), dtype=np.float32)
    rows = np.column_stack(
        (np.ones(8), np.arange(8), np.arange(1, 9) * 0.3, planned)
    )
    action = _frozen_plan_action(
        rows,
        decision_step=1,
        before={
            "position": np.zeros(3, dtype=np.float32),
            "velocity_reference": np.zeros(3, dtype=np.float32),
            "yaw": 0.0,
        },
        exact_bitwise=True,
    )
    assert isinstance(action.policy, np.ndarray)
    assert d05b._policy_action_8x11(action) == action.policy[0].tolist()
    assert d05b._executed_action_4x11(action) == action.rollout_executed[0].tolist()


def test_endpoint_verification_wrapper_is_normalized_before_capture():
    check = _endpoint_check()
    normalized = d05b._normalize_endpoint_check(
        check,
        task_id="env_001_obs10__cfg01",
        repeat=1,
        policy="bc",
        done_reason="reached",
    )
    assert normalized == check["actual"]
    assert normalized["done_reason"] == "reached"
    assert normalized["steps"] == 13
    with pytest.raises(Exception, match="repeat differs"):
        d05b._normalize_endpoint_check(
            check,
            task_id="env_001_obs10__cfg01",
            repeat=2,
            policy="bc",
            done_reason="reached",
        )


def test_capture_summary_contract_rejects_unflattened_verification_wrapper():
    selected = {
        "task_id": "env_001_obs10__cfg01",
        "repeat": 1,
        "bc_done_reason": "reached",
        "seed3_done_reason": "time_limit",
        "expected_bc_summary": _endpoint_check(),
        "expected_seed3_summary": _endpoint_check(
            policy="rl", done_reason="time_limit", steps=60
        ),
    }
    with pytest.raises(Exception, match="normalized endpoint summary"):
        d05b._selected_expected_summary(selected, "u0")


def test_capture_selection_prevalidates_target_and_filler_endpoint_summaries():
    def selected(task_id, repeat):
        bc = _endpoint_check()["actual"]
        rl = _endpoint_check(policy="rl", done_reason="time_limit", steps=60)[
            "actual"
        ]
        bc = dict(bc, task_id=task_id)
        rl = dict(rl, task_id=task_id)
        return {
            "task_id": task_id,
            "repeat": repeat,
            "bc_done_reason": "reached",
            "seed3_done_reason": "time_limit",
            "expected_bc_summary": bc,
            "expected_seed3_summary": rl,
        }

    targets = [selected("target_{:03d}".format(index), 1) for index in range(160)]
    fillers = [selected("filler_{:03d}".format(index), 1) for index in range(480)]
    selection = {
        "target_repeat_count": 160,
        "filler_repeat_count": 480,
        "targets": targets,
        "fillers": fillers,
    }
    assert d05b._validate_capture_selection(selection) == 1280

    selection["fillers"][0]["expected_seed3_summary"] = _endpoint_check(
        policy="rl", done_reason="time_limit", steps=60
    )
    with pytest.raises(Exception, match="normalized endpoint summary"):
        d05b._validate_capture_selection(selection)


def test_credit_batches_keep_target_episode_intact_and_use_formal_s2_gae():
    target = [
        _row("target", 0, 1.0),
        _row("target", 1, 2.0, terminated=True),
    ]
    filler = [
        _row("filler", 0, 0.0),
        _row("filler", 1, 0.0),
        _row("filler", 2, 0.0, terminated=True),
    ]
    population, audits = _build_credit_batches(
        [target],
        filler_entries=[{"task_id": "filler", "repeat": 0}],
        filler_loader=lambda unused: filler,
        lineage="lineage",
        lane="u0",
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert len(population) == 1024
    assert len(audits) == 1
    assert audits[0]["logical_batch_pairs"] == 1280
    assert audits[0]["pair_count"] == 10240
    assert audits[0]["target_collection_boundary_count"] == 0
    assert audits[0]["filler_collection_boundary_count"] == 1
    assert population[0]["raw_advantage"] == pytest.approx(2.881, abs=1e-6)
    assert population[1]["raw_advantage"] == pytest.approx(2.0, abs=1e-6)
    assert len(population[0]["normalized_advantage_s2_by_k"]) == 10


def test_cli_self_check_exercises_source_identity_and_s2_contract(capsys):
    assert main(["--self-check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "SELF_CHECK_PASS"
    assert payload["training_source_set_hash"] == (
        "5aed4fb675591750ac97ad4165ac11b8d3fedf5a15dfa2bf0bc84266c373f637"
    )
    assert payload["training_source_set_unchanged"] is True
    assert payload["s2_pair_count"] == 10240
    assert payload["frozen_numpy_action_serialization"] is True
    assert payload["endpoint_summary_normalization"] is True
