"""T4 metrics (design §4.15): the 12 learning-signal recorders plus
explained variance, as pure functions.

§4.15 per-update records:
  actor_grad_norm, critic_grad_norm, parameter_update_norm,
  per-denoise approx_kl, per-denoise clipfrac,
  ratio mean/std/quantiles, action variance by dimension,
  policy-action diversity, success-conditioned advantage,
  collision-conditioned advantage, role-conditioned advantage,
  executed-vs-unexecuted gradient norm
"""

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.metrics import (
    explained_variance,
    ratio_statistics,
    success_collision_role_conditioned_advantages,
    tensor_mean_std,
)


def test_ratio_statistics_shapes_and_quantiles():
    torch.manual_seed(0)
    ratio = torch.exp(torch.randn(64) * 0.5)
    out = ratio_statistics(ratio)
    assert set(out) == {"mean", "std", "q05", "q50", "q95"}
    qs = torch.tensor([out["q05"], out["q50"], out["q95"]])
    assert torch.all(torch.sort(qs).values == qs)  # monotonic
    assert torch.isfinite(torch.tensor(list(out.values()))).all()


def test_ratio_statistics_matches_numpy():
    torch.manual_seed(1)
    ratio = torch.exp(torch.randn(32) * 0.7)
    out = ratio_statistics(ratio)
    np_ratio = ratio.numpy()
    assert abs(out["mean"] - float(np.mean(np_ratio))) < 1e-6
    assert abs(out["std"] - float(np.std(np_ratio))) < 1e-6
    assert abs(out["q05"] - float(np.quantile(np_ratio, 0.05))) < 1e-4
    assert abs(out["q50"] - float(np.median(np_ratio))) < 1e-4


def test_tensor_mean_std_empty_raises():
    with pytest.raises(ValueError):
        tensor_mean_std(torch.zeros(0))


def test_conditioned_advantages_by_role_and_outcome():
    adv = torch.tensor([1.0, 2.0, -1.0, -2.0, 3.0, 4.0])
    terminated = np.array([False, True, False, True, False, False])
    # R5.8-C (N9): outcome truth — the terminal transition of a
    # successful episode carries success=True; a collision episode
    # carries collision=True. No reward-threshold inference.
    success = np.array([False, True, False, False, False, False])
    collision = np.array([False, False, False, True, False, False])
    roles = ["anticipatory", "anticipatory", "fullspace", "fullspace",
             "dense", "dense"]
    out = success_collision_role_conditioned_advantages(
        advantages=adv, terminated=terminated, success=success,
        collision=collision, roles=roles,
    )
    # success-conditioned: advantage mean where the terminal success is
    # (R6-C §11.4.1: OptionalStatistic two-state contract)
    succ = out["success_conditioned_advantage"]
    assert succ["sample_count"] == 1 and succ["valid"] is True
    assert succ["invalid_reason"] is None
    assert abs(succ["value"] - 2.0) < 1e-6
    # collision-conditioned: the terminal collision
    coll = out["collision_conditioned_advantage"]
    assert coll["sample_count"] == 1 and coll["valid"] is True
    assert abs(coll["value"] - (-2.0)) < 1e-6
    # role-conditioned: per-role means over ALL transitions
    assert abs(out["role_conditioned_advantage"]["anticipatory"] - 1.5) < 1e-6
    assert abs(out["role_conditioned_advantage"]["fullspace"] - (-1.5)) < 1e-6
    assert abs(out["role_conditioned_advantage"]["dense"] - 3.5) < 1e-6


def test_conditioned_advantages_no_success_collision_two_state_null():
    """R6-C (§11.4.1): no terminated success/collision transition ->
    the statistic is the null state (value None/valid False + frozen
    reason) — never NaN and never a silent 0 (0 would look like a real
    signal)."""
    adv = torch.tensor([1.0, 2.0])
    terminated = np.array([False, False])
    success = np.array([False, False])
    collision = np.array([False, False])
    roles = ["anticipatory", "fullspace"]
    out = success_collision_role_conditioned_advantages(
        advantages=adv, terminated=terminated, success=success,
        collision=collision, roles=roles,
    )
    for field, reason in (
        ("success_conditioned_advantage", "no_success_terminal"),
        ("collision_conditioned_advantage", "no_collision_terminal"),
    ):
        stat = out[field]
        assert stat["sample_count"] == 0
        assert stat["value"] is None
        assert stat["valid"] is False
        assert stat["invalid_reason"] == reason


def test_explained_variance_perfect_and_zero():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    ev = explained_variance(y_true, y_true)
    assert ev["valid"] is True and ev["sample_count"] == 4
    assert abs(ev["value"] - 1.0) < 1e-9
    y_pred = np.array([1.5, 2.5, 2.5, 4.5])
    var_y = np.var(y_true)
    expected = 1 - np.var(y_true - y_pred) / var_y
    ev2 = explained_variance(y_true, y_pred)
    assert abs(ev2["value"] - expected) < 1e-9
    assert ev2["target_variance"] == var_y


def test_explained_variance_constant_target_is_null():
    """R6-C (§11.4.1): var_y == 0 -> the statistic is the null state
    (constant_value_target) with target_variance preserved — the
    official NaN (diffusion_ppo agent line 388-390) never round-trips
    through strict JSON."""
    y_true = np.ones(4)
    y_pred = np.zeros(4)
    ev = explained_variance(y_true, y_pred)
    assert ev["target_variance"] == 0.0
    assert ev["sample_count"] == 0
    assert ev["value"] is None
    assert ev["valid"] is False
    assert ev["invalid_reason"] == "constant_value_target"
