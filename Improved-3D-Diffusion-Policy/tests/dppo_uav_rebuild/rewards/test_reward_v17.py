"""v1.4 contract regressions. Run in WSL; no policy/GPU training required."""

import math
import json

import numpy as np
import pytest

from diffusion_policy_3d.env.collision_geom import ObstacleSpec
from diffusion_policy_3d.env.reward_v17 import (
    TERM_NAMES, compute_reward_v17, create_reward_history, resolve_config,
)
from diffusion_policy_3d.env.reward_v17_support.geometry import (
    GeometryContext, path_geometry, direct_path, free_potential,
)
from diffusion_policy_3d.env.reward_v17_support.density import (
    build_density, support_cell,
)
from diffusion_policy_3d.env.reward_v17_support.state import guard_cost
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import TransitionFacts


def config(**changes):
    # Explicit fixture values, NOT a formal calibrated configuration.
    return resolve_config({**dict(gamma_pbrs=0.99, d_ref=0.7, s_f=0.4,
                                  w_low=2.5, w_high=1.0, w_climb=0.1), **changes})


def facts(reason="none"):
    return TransitionFacts.resolve(collided=reason == "collision",
                                   early_death=reason == "early_death",
                                   reached=reason == "reached", time_limit=reason == "time_limit",
                                   raw_done_reason=reason)


def density():
    return build_density([dict(episode_id="demo1", split="bc_demo", source="frozen_bc",
                               start=[0, 0, 1], goal=[10, 0, 1], ground_z=0,
                               initial_yaw=0, positions=[[0, 0, 1], [1, 0, 1]])],
                         dataset_sha256="a" * 64)


def evaluate(path, reason="none", obstacles=(), history=None, cfg=None, goal=(10, 0, 1), context=None):
    path = np.asarray(path, dtype=np.float64)
    return compute_reward_v17(
        dict(pos=path[0], yaw=0), path, dict(pos=path[-1], yaw=0), facts(reason),
        obstacles, np.array(goal), np.array([0., 0., 1.]), context or GeometryContext(),
        history or create_reward_history(), density(), cfg or config())


def test_guard_charges_redundancy_not_diagonal_translation_or_onset_budget():
    path = np.array([[0, 0, 1], [.1, .1, 1], [.2, .2, 1], [.3, .3, 1], [.4, .4, 1]])
    assert evaluate(path)[1]["guard"] == 0
    zigzag = np.array([[0, 0, 1], [.2, .2, 1], [.4, -.2, 1], [.6, .2, 1], [.8, 0, 1]])
    a = evaluate(zigzag)
    h = create_reward_history()
    h["caps"]["onset"] = 8.
    b = evaluate(zigzag, history=h)
    assert a[1]["guard"] < 0
    assert a[1]["guard"] == b[1]["guard"]


def test_continuous_geometry_catches_thin_obstacle_and_ground_between_endpoints():
    box = ObstacleSpec("thin", "box", 0, 0, 1, size_x=.02, size_y=.2, size_z=2)
    path = np.array([[-.4, 0, 1], [-.2, 0, 1], [.2, 0, 1], [.4, 0, 1], [.6, 0, 1]])
    g = path_geometry(path, [box], GeometryContext(), config())
    assert g["collision"] and g["d_soft"] == 0
    path[2] = [0, 0, -.1]
    assert path_geometry(path, [], GeometryContext(), config())["ground_collision"]


def test_legal_overflight_has_real_vertical_clearance_and_no_altitude_ceiling():
    obs = ObstacleSpec("wall", "box", 0, 0, .5, size_x=.4, size_y=2, size_z=1)
    path = np.array([[-.4, 0, 1.3], [-.2, 0, 1.3], [0, 0, 1.3], [.2, 0, 1.3], [.4, 0, 1.3]])
    g = path_geometry(path, [obs], GeometryContext(), config())
    assert not g["collision"]
    assert min(g["overflight_clearances"]) == pytest.approx(.2, abs=1e-8)
    path[:, 2] = 100
    total, terms, metrics, _ = evaluate(path, obstacles=[obs])
    assert math.isfinite(total)
    assert metrics["altitude_termination_enabled"] is True
    assert terms["terminal"] == 0


def test_direct_counterfactual_keeps_each_horizontal_length_and_vertical_increment():
    p = np.array([[0, 0, 1], [.1, .2, 1.1], [.1, .2, 1.2], [.2, 0, 1.3], [.4, 0, 1.3]])
    q = direct_path(p, np.array([4., 0., 1.]))
    assert q[1, 0] == pytest.approx(math.sqrt(.05))
    np.testing.assert_allclose(np.diff(q[:, 2]), [.1, .1, .1, 0], atol=1e-12)
    assert q[2, 0] == q[1, 0]
    assert direct_path(p, np.array([0., 0., 3.])) is None


def test_pbrs_terminal_closes_potential_but_time_limit_retains_tail():
    p = np.array([[0, 0, 1], [.1, 0, 1], [.2, 0, 1], [.3, 0, 1], [.4, 0, 1]])
    terminal = evaluate(p, "reached")
    truncated = evaluate(p, "time_limit")
    assert terminal[1]["pbrs_free"] == pytest.approx(-.8)
    assert truncated[1]["pbrs_free"] == pytest.approx(-.008)
    assert terminal[1]["progress"] == pytest.approx(.8)
    assert set(terminal[1]) == set(TERM_NAMES)
    assert sum(terminal[1].values()) == pytest.approx(terminal[0])


def test_support_final_step_is_booked_before_success_payment_but_failure_never_pays():
    p = np.array([[0, 0, 2], [.2, 0, 2], [.4, 0, 2], [.6, 0, 2], [.8, 0, 2]])
    h = create_reward_history()
    h["steps"] = [dict(progress=.1, d_min=2., d_soft=2., risk=0., lateral=0., yaw=0., guard_context=True)] * 5
    reached = evaluate(p, "reached", history=h)
    assert reached[1]["support_escrow"] == .5
    assert evaluate(p, "time_limit", history=h)[1]["support_escrow"] == 0
    assert evaluate(p, "collision", history=h)[1]["support_escrow"] == 0


def test_loop_does_not_double_count_zeros_and_requires_entire_safe_window():
    cfg = config()
    progress = [.1, 0, .1, 0, .1, 0, .1, 0]
    rows = [dict(progress=p, guard_context=True) for p in progress]
    cost, m = guard_cost(0., True, rows, cfg)
    assert cost == 0 and m["loop_sign_changes"] == 0
    rows = [dict(progress=p, guard_context=True) for p in [.1, -.1] * 4]
    assert guard_cost(0., True, rows, cfg)[0] == -1
    rows[0]["guard_context"] = False
    assert guard_cost(0., True, rows, cfg)[0] == 0


def test_coordinate_shift_moves_ground_and_preserves_geometry_and_density_cell():
    p = np.array([[0, 0, 1], [.1, .1, 1], [.2, .1, 1], [.3, .2, 1], [.4, .2, 1]])
    shift = np.array([1000., -1000., 1000.])
    ctx = GeometryContext(ground_z=1000., xy_min=(970., -1030.), xy_max=(1030., -970.))
    a = path_geometry(p, [], GeometryContext(), config())
    b = path_geometry(p + shift, [], ctx, config())
    assert a["ground_clearance"] == b["ground_clearance"] == 1
    assert support_cell(p[-1], p[0], [10, 0, 1], 0, 0) == support_cell(
        p[-1] + shift, p[0] + shift, np.array([10, 0, 1]) + shift, 1000, 0)


def test_invalid_input_never_becomes_ordinary_reward_or_fake_finite_state():
    p = np.array([[0, 0, 1], [0, 0, 1], [np.nan, 0, 1], [0, 0, 1], [0, 0, 1]])
    total, terms, metrics, _ = evaluate(p)
    assert total <= -50 and math.isfinite(total)
    assert metrics["invalid_state"] and not metrics["training_usable"]
    assert terms["onset"] == terms["support_escrow"] == 0


def test_missing_calibration_or_density_fails_explicitly():
    with pytest.raises(ValueError, match="calibrat"):
        resolve_config({"gamma_pbrs": .99})
    with pytest.raises(ValueError, match="density"):
        compute_reward_v17(dict(pos=[0, 0, 1], yaw=0), np.tile([0, 0, 1], (5, 1)),
                           dict(pos=[0, 0, 1], yaw=0), facts(), [], np.array([10, 0, 1]),
                           np.array([0, 0, 1]), GeometryContext(), create_reward_history(), None, config())


def test_ground_soft_barrier_uses_only_legal_part_not_the_depth_of_penetration():
    p = np.array([[0, 0, 1], [.1, 0, .3], [.2, 0, -.5], [.3, 0, .3], [.4, 0, 1]])
    a = evaluate(p, "collision")
    assert a[1]["ground"] == pytest.approx(-20.)
    p[2, 2] = -500
    assert evaluate(p, "collision")[1]["ground"] == pytest.approx(-20.)
    p[:, 2] = -.5
    assert evaluate(p, "collision")[1]["ground"] == 0


def test_potential_clips_after_quantile_not_each_individual_ray(monkeypatch):
    from diffusion_policy_3d.env.reward_v17_support import geometry
    # Three early hits, remaining rays first hit at 10m: p20 interpolates
    # between 1m and 10m, not between 1m and the potential's 2m cap.
    def distances(starts, ends, obstacles):
        assert len(starts) == 1500
        out = np.ones((15, 100))
        out[:3, 9:] = 0
        return out.reshape(-1, 1)
    monkeypatch.setattr(geometry, "segment_clearances", distances)
    assert free_potential([0, 0, 20], [10, 0, 20], [object()], GeometryContext(), config()) == .8


def test_high_altitude_dwell_uses_segments_and_unknown_is_not_stall():
    # Demo maximum is 1m; first segment is half above, next three all above.
    p = [[0, 0, .5], [.1, 0, 1.5], [.2, 0, 1.5], [.3, 0, 1.5], [.4, 0, 1.5]]
    m = evaluate(p)[2]
    assert m["above_demo_dwell_seconds"] == pytest.approx(1.05)
    assert m["above_demo_unknown_seconds"] == pytest.approx(1.05)
    assert m["above_demo_stall_seconds"] == 0
    assert density()["demo_height_max"] == 1.
    json.dumps(m, allow_nan=False)


def test_nonfinite_history_is_invalid_instead_of_silently_zeroing_a_term():
    h = create_reward_history()
    h["caps"]["onset"] = float("nan")
    total, _, metrics, _ = evaluate(np.tile([0, 0, 1], (5, 1)), history=h)
    assert total <= -50 and not metrics["training_usable"]


def test_collision_does_not_erase_soft_cost_on_other_legal_path_portions():
    box = ObstacleSpec("wall", "box", 0, 0, 1, size_x=.2, size_y=1., size_z=2.)
    p = np.array([[-1, 0, 1], [-.5, 0, 1], [0, 0, 1], [.5, 0, 1], [1, 0, 1]])
    assert evaluate(p, "collision", obstacles=[box])[1]["clearance"] < 0.
