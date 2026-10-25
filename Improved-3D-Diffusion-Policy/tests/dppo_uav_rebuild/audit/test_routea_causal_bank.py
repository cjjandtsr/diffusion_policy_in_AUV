from __future__ import annotations

from dataclasses import replace

import pytest

from scripts.routea_causal_diagnostics.bank import (
    STRATA,
    ParentRepeat,
    assign_stratum,
    build_stratum_manifest,
    freeze_sentinel_subset,
    select_stratified_tasks,
)
from scripts.routea_causal_diagnostics.contracts import DiagnosticFailure


def parent_repeat(
    *,
    task_id="task-0",
    repeat=1,
    reset_seed=10,
    bc=False,
    seed1=False,
    seed2=False,
    seed3=False,
    clear_line=False,
):
    return ParentRepeat(
        task_id=task_id,
        repeat=repeat,
        env_reset_seed=reset_seed,
        world="world-0",
        split="rl_test",
        role="anticipatory",
        start=(0.0, 0.0, 2.0),
        goal=(4.0, 0.0, 2.0),
        bc_reached=bc,
        seed1_reached=seed1,
        seed2_reached=seed2,
        seed3_reached=seed3,
        clear_line=clear_line,
        bc_trace_sha256="a" * 64,
        rl_trace_sha256=("1" * 64, "2" * 64, "3" * 64),
        trace_paths=("bc.csv", "s1.csv", "s2.csv", "s3.csv"),
    )


@pytest.mark.parametrize(
    ("bc", "s1", "s2", "s3", "clear", "expected"),
    [
        (False, True, False, False, False, "S1_SEED1_RESCUE"),
        (True, True, True, False, False, "S2_SEED3_BREAK"),
        (False, False, False, False, False, "S3_STABLE_FAILURE"),
        (True, True, True, True, True, "S4_CLEAR_STABLE_SUCCESS"),
    ],
)
def test_assign_stratum_is_mutually_exclusive(bc, s1, s2, s3, clear, expected):
    row = parent_repeat(
        bc=bc, seed1=s1, seed2=s2, seed3=s3, clear_line=clear
    )
    assert assign_stratum(row) == expected


def test_assign_stratum_returns_none_for_unlisted_mixed_behavior():
    row = parent_repeat(bc=True, seed1=False, seed2=True, seed3=True)
    assert assign_stratum(row) is None


def fixture_rows():
    rows = []
    for task_index in range(10):
        for repeat in (1, 2):
            rows.append(
                parent_repeat(
                    task_id=f"s2-{task_index}",
                    repeat=repeat,
                    reset_seed=1000 + 10 * task_index + repeat,
                    bc=True,
                    seed1=True,
                    seed2=True,
                    seed3=False,
                )
            )
    rows.append(
        parent_repeat(
            task_id="s4-only",
            bc=True,
            seed1=True,
            seed2=True,
            seed3=True,
            clear_line=True,
        )
    )
    return rows


def test_selection_keeps_one_repeat_per_task_is_stable_and_never_backfills():
    selected = select_stratified_tasks(fixture_rows(), max_tasks=8)
    again = select_stratified_tasks(list(reversed(fixture_rows())), max_tasks=8)
    s2 = selected["S2_SEED3_BREAK"]
    assert len(s2) == 8
    assert len({x.task_id for x in s2}) == len(s2)
    assert [x.selection_key for x in s2] == [x.selection_key for x in again["S2_SEED3_BREAK"]]
    assert len(selected["S4_CLEAR_STABLE_SUCCESS"]) == 1
    assert selected.status["S4_CLEAR_STABLE_SUCCESS"] == "UNDERPOWERED_STRATUM"
    assert selected["S1_SEED1_RESCUE"] == ()


def test_selection_rejects_duplicate_parent_identity():
    row = parent_repeat()
    with pytest.raises(DiagnosticFailure, match="PAIRED_IDENTITY_MISMATCH"):
        select_stratified_tasks([row, replace(row, bc_trace_sha256="b" * 64)])


def test_stratum_manifest_has_all_strata_and_no_pooled_permission():
    rows = fixture_rows()
    selected = select_stratified_tasks(rows, max_tasks=8)
    manifest = build_stratum_manifest(rows, selected)
    assert tuple(manifest["strata"]) == STRATA
    assert manifest["allows_parent_weighted_pooling"] is False
    assert manifest["strata"]["S1_SEED1_RESCUE"]["selected_count"] == 0
    assert manifest["strata"]["S1_SEED1_RESCUE"]["inclusion_probability"] is None
    assert manifest["strata"]["S4_CLEAR_STABLE_SUCCESS"]["status"] == "UNDERPOWERED_STRATUM"


def test_freeze_sentinel_subset_uses_preference_fallback_and_two_task_cap():
    rows = []
    for task in ("a", "b", "c"):
        for anchor in ("early", "closest", "clearance"):
            rows.append(
                {
                    "snapshot_id": f"S1-{task}-{anchor}",
                    "stratum": "S1_SEED1_RESCUE",
                    "task_id": task,
                    "repeat": 1,
                    "env_reset_seed": ord(task),
                    "anchor_kind": anchor,
                }
            )
    rows.extend(
        [
            {
                "snapshot_id": "S3-z-early",
                "stratum": "S3_STABLE_FAILURE",
                "task_id": "z",
                "repeat": 2,
                "env_reset_seed": 9,
                "anchor_kind": "early",
            },
            {
                "snapshot_id": "S3-z-clearance",
                "stratum": "S3_STABLE_FAILURE",
                "task_id": "z",
                "repeat": 2,
                "env_reset_seed": 9,
                "anchor_kind": "clearance",
            },
        ]
    )
    frozen = freeze_sentinel_subset(rows)
    s1 = [item for item in frozen if item.stratum == "S1_SEED1_RESCUE"]
    s3 = [item for item in frozen if item.stratum == "S3_STABLE_FAILURE"]
    assert len(s1) == 2
    assert all(item.anchor_kind == "clearance" for item in s1)
    assert len(s3) == 1
    assert s3[0].anchor_kind == "early"  # closest absent; fixed fallback order


def test_freeze_sentinel_allows_one_physical_snapshot_to_carry_multiple_anchors():
    base = {
        "snapshot_id": "S1-task-r1-d5",
        "stratum": "S1_SEED1_RESCUE",
        "task_id": "task",
        "repeat": 1,
        "env_reset_seed": 7,
    }
    frozen = freeze_sentinel_subset(
        [
            {**base, "anchor_kind": "closest"},
            {**base, "anchor_kind": "clearance"},
        ]
    )
    assert len(frozen) == 1
    assert frozen[0].snapshot_id == base["snapshot_id"]
    assert frozen[0].anchor_kind == "clearance"
    assert frozen[0].substitution_reason is None


def test_freeze_sentinel_rejects_unknown_anchor_and_duplicate_snapshot():
    base = {
        "snapshot_id": "same",
        "stratum": "S4_CLEAR_STABLE_SUCCESS",
        "task_id": "task",
        "repeat": 1,
        "env_reset_seed": 1,
        "anchor_kind": "early",
    }
    with pytest.raises(DiagnosticFailure, match="INVALID_SNAPSHOT_MANIFEST"):
        freeze_sentinel_subset([base, dict(base)])
    with pytest.raises(DiagnosticFailure, match="INVALID_SNAPSHOT_MANIFEST"):
        freeze_sentinel_subset([{**base, "snapshot_id": "x", "anchor_kind": "late"}])


def test_freeze_sentinel_rejects_snapshot_id_reused_for_different_physical_identity():
    base = {
        "snapshot_id": "same",
        "stratum": "S1_SEED1_RESCUE",
        "task_id": "task-a",
        "repeat": 1,
        "env_reset_seed": 1,
        "anchor_kind": "closest",
    }
    conflicting = {
        **base,
        "task_id": "task-b",
        "anchor_kind": "clearance",
    }
    with pytest.raises(DiagnosticFailure, match="INVALID_SNAPSHOT_MANIFEST"):
        freeze_sentinel_subset([base, conflicting])
