import math

import pytest
import torch

from diffusion_policy_3d.env.reward_v17 import TERM_NAMES
from scripts.routea_causal_diagnostics.credit_divergence import (
    CREDIT_PROPAGATION_SUSPECTED,
    INCONCLUSIVE,
    REWARD_OBJECTIVE_SUSPECTED,
    SIGNAL_OPPOSES_BAD_ACTION,
    build_d05_verdict,
    classify_endpoint_pair,
    classify_transition_signal,
    reward_group_totals,
    s2_normalize_advantages,
    trace_window_indices,
    validate_reward_trace_row,
)


def _terms(**overrides):
    values = {name: 0.0 for name in TERM_NAMES}
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    "bc,rl,expected",
    [
        ("reached", "time_limit", "break"),
        ("time_limit", "reached", "rescue"),
        ("reached", "reached", "stable_success"),
        ("collision", "time_limit", "stable_failure"),
    ],
)
def test_endpoint_pair_classification_uses_reached_only(bc, rl, expected):
    assert classify_endpoint_pair(bc, rl) == expected


def test_trace_window_clips_registered_bounds_without_inventing_divergence():
    assert trace_window_indices(4, decision_count=12) == tuple(range(1, 10))
    assert trace_window_indices(1, decision_count=5) == (0, 1, 2, 3, 4)
    assert trace_window_indices(None, decision_count=5) == ()
    with pytest.raises(ValueError, match="outside"):
        trace_window_indices(5, decision_count=5)


def test_reward_trace_requires_exact_v17_terms_and_fp64_sum():
    row = {
        "reward": 1.25,
        "reward_terms": _terms(progress=1.0, future_clearance=0.25),
    }
    validated = validate_reward_trace_row(row)
    assert validated["reward_sum_error_abs"] == 0.0

    malformed = dict(row, reward_terms={"progress": 1.25})
    with pytest.raises(ValueError, match="term keys"):
        validate_reward_trace_row(malformed)

    wrong_sum = dict(row, reward=1.2500000002)
    with pytest.raises(ValueError, match="sum error"):
        validate_reward_trace_row(wrong_sum)


def test_reward_groups_match_registered_d2_groups_without_double_counting():
    terms = _terms(
        terminal=3.0,
        progress=-2.0,
        onset=0.5,
        future_clearance=0.25,
        trap=-0.75,
        support_escrow=1.0,
        pbrs_free=0.4,
        guard=-0.2,
    )
    groups = reward_group_totals(terms)
    assert groups == {
        "core": 1.0,
        "proactive": 0.75,
        "recovery": -0.75,
        "support": 1.0,
        "free_pbrs": 0.4,
        "guard": -0.2,
    }
    assert sum(groups.values()) == pytest.approx(sum(terms.values()))


def test_s2_normalization_uses_eight_exact_1280_pair_batches():
    advantages = torch.arange(1024, dtype=torch.float32)
    result = s2_normalize_advantages(
        advantages.tolist(),
        transition_ids=[f"t{i}" for i in range(1024)],
        schedule_seed=2718,
    )
    audit = result["audit"]
    assert audit["transition_count"] == 1024
    assert audit["pair_count"] == 10240
    assert audit["logical_batch_pairs"] == 1280
    assert audit["logical_batch_count"] == 8
    assert audit["duplicate_pair_count"] == 0
    assert audit["missing_pair_count"] == 0
    assert all(len(result["by_transition"][f"t{i}"]) == 10 for i in range(1024))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(2718)
    flat = torch.randperm(10240, generator=generator)
    first_batch = flat[:1280]
    transition_indices = torch.div(first_batch, 10, rounding_mode="floor")
    expanded = advantages[transition_indices]
    expected = (expanded - expanded.mean()) / (expanded.std() + 1e-8)
    first_pair = int(first_batch[0])
    first_transition, first_k = divmod(first_pair, 10)
    observed = next(
        row["normalized_advantage"]
        for row in result["by_transition"][f"t{first_transition}"]
        if row["denoising_index"] == first_k
    )
    assert observed == pytest.approx(float(expected[0]), abs=1e-7)


def test_s2_normalization_rejects_probe_sized_or_duplicate_identity_input():
    with pytest.raises(ValueError, match="exactly 1024"):
        s2_normalize_advantages([0.0] * 128, [f"t{i}" for i in range(128)], 1)
    with pytest.raises(ValueError, match="unique"):
        s2_normalize_advantages([0.0] * 1024, ["same"] * 1024, 1)


@pytest.mark.parametrize(
    "reward,raw,normalized,expected",
    [
        (0.25, -0.1, -0.2, REWARD_OBJECTIVE_SUSPECTED),
        (-0.25, 0.1, 0.2, CREDIT_PROPAGATION_SUSPECTED),
        (-0.25, -0.1, -0.2, SIGNAL_OPPOSES_BAD_ACTION),
        (0.0, 0.0, 0.0, INCONCLUSIVE),
    ],
)
def test_transition_signal_separates_reward_credit_and_opposition(
    reward, raw, normalized, expected
):
    assert classify_transition_signal(reward, raw, normalized) == expected


def test_verdict_uses_unique_task_denominator_and_requires_repeated_support():
    rows = [
        {"task_id": "a", "mechanism": REWARD_OBJECTIVE_SUSPECTED},
        {"task_id": "a", "mechanism": REWARD_OBJECTIVE_SUSPECTED},
        {"task_id": "b", "mechanism": REWARD_OBJECTIVE_SUSPECTED},
        {"task_id": "c", "mechanism": SIGNAL_OPPOSES_BAD_ACTION},
    ]
    verdict = build_d05_verdict(rows)
    assert verdict["status"] == REWARD_OBJECTIVE_SUSPECTED
    assert verdict["determinate_task_count"] == 3
    assert verdict["mechanism_task_counts"] == {
        REWARD_OBJECTIVE_SUSPECTED: 2,
        SIGNAL_OPPOSES_BAD_ACTION: 1,
    }
    assert verdict["evidence_scope"] == "paired_first_divergence_correlation"


def test_verdict_does_not_force_single_mechanism_when_two_routes_repeat():
    verdict = build_d05_verdict(
        [
            {"task_id": "a", "mechanism": REWARD_OBJECTIVE_SUSPECTED},
            {"task_id": "b", "mechanism": REWARD_OBJECTIVE_SUSPECTED},
            {"task_id": "c", "mechanism": CREDIT_PROPAGATION_SUSPECTED},
            {"task_id": "d", "mechanism": CREDIT_PROPAGATION_SUSPECTED},
        ]
    )
    assert verdict["status"] == INCONCLUSIVE
    assert verdict["reason"] == "multiple_mechanisms_have_repeated_task_support"
    assert math.isclose(verdict["coverage"][REWARD_OBJECTIVE_SUSPECTED], 0.5)
