"""Calibration selection is deterministic and cannot accept evaluation data."""

import pytest

from diffusion_policy_3d.env.reward_v17_support.calibration import anchor_scale, select_height_weights
from diffusion_policy_3d.env.reward_v17_support.state import create_reward_history, update_trap
from diffusion_policy_3d.env.reward_v17_support.config import resolve_config


def test_small_empirical_anchor_uses_labelled_fixture_not_duplicate_samples():
    empirical = [dict(episode_id="one", world_id="train", role="anticipatory", split="rl_train", d_soft=.5)] * 120
    fixture = [dict(episode_id=str(i), world_id="fixture", role="fixture", split="fixture", d_soft=d)
               for i, d in enumerate((.2, .4, .6, .8, 1.0))]
    result = anchor_scale(empirical, fixture)
    assert result["calibration_source"] == "fixture"
    assert result["empirical_episode_count"] == 1
    assert result["d_ref"] == .6 and result["k_s"] == 1
    assert result["s_f"] == pytest.approx(.2 / .5493061443340548)


def test_evaluation_anchor_is_rejected_instead_of_filtered_silently():
    with pytest.raises(ValueError, match="evaluation"):
        anchor_scale([dict(episode_id="bad", world_id="w4", role="fixture", split="rl_test", d_soft=.5)], [])


def test_height_search_is_27_candidates_and_requires_signal_checks():
    rows = dict(safe=dict(base=10., low=0., high=0., climb=0.),
                fail=dict(base=-50., low=0., high=1., climb=1.))
    result = select_height_weights(rows, [("safe", "fail")], {"onset_contrast": False})
    assert result["status"] == "unresolved" and len(result["candidates"]) == 27
    assert result["selected"] is None


def test_height_search_rejects_nonfinite_features():
    rows = dict(safe=dict(base=float("nan"), low=0., high=1., climb=1.),
                fail=dict(base=-50., low=0., high=1., climb=1.))
    with pytest.raises(ValueError, match="finite"):
        select_height_weights(rows, [("safe", "fail")], {"signal": True})


def test_undiscounted_stress_is_reported_without_replacing_the_discounted_objective():
    rows = dict(safe=dict(base=10., low=0., high=0., climb=0.,
                          undiscounted=dict(base=-60., low=0., high=0., climb=0.)),
                fail=dict(base=-50., low=0., high=1., climb=1.,
                          undiscounted=dict(base=-50., low=0., high=1., climb=1.)))
    result = select_height_weights(rows, [("safe", "fail")], {"signal": True})
    assert "safe>fail" in result["candidates"][0]["failed_undiscounted_orderings"]
    assert result["status"] == "resolved"
    assert result["candidates"][0]["accepted"]
    assert result["selection_objective"] == "discounted_return"
    assert result["undiscounted_role"] == "reported_stress_diagnostic_not_acceptance_gate"


def test_trap_hysteresis_recovery_once_and_timeout_does_not_pay_escape():
    cfg = resolve_config(dict(gamma_pbrs=.99, d_ref=.7, s_f=.4, w_low=2.5, w_high=1., w_climb=1.))
    h = create_reward_history()
    stuck = [dict(progress=0., d_min=.3, d_soft=.3)] * 6
    update_trap(h, stuck, cfg, True, False, False)
    assert h["trap_state"] == "CANDIDATE"
    update_trap(h, stuck, cfg, True, False, False)
    assert h["trap_state"] == "TRAPPED"
    recovery = [dict(progress=.12, d_min=1., d_soft=1.)] * 6
    update_trap(h, recovery, cfg, True, False, False)
    assert h["trap_state"] == "RECOVERING"
    _, m = update_trap(h, recovery, cfg, True, True, False)
    assert h["trap_state"] == "FREE" and m["trap_escape_paid"] == 0
    assert h["caps"]["escape"] == 0
