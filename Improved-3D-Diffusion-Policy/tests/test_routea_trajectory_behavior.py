"""Geometry and temporal labels must not equate visibility with reachability."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_routea_trajectory_behavior import (Scene, phase_label, runs,
    plan_features, compare_replans)


def cylinder(x, y, z=2., height=4.):
    return dict(name="pillar", shape="cylinder", x=x, y=y, z=z,
                height=height, radius=.5)


def test_visible_but_not_blocking_and_invisible_but_blocking():
    scene = Scene([cylinder(5, 1.5)])
    assert scene.clear([0, 0, 2], [9, 0, 2])
    assert not scene.certified_empty_view([0, 0, 2], 0.)
    blocked = Scene([cylinder(5, 0)])
    assert not blocked.clear([0, 0, 2], [9, 0, 2])
    assert blocked.certified_empty_view([0, 0, 2], np.pi)


def test_true_vertical_geometry_and_sensor_geometry_are_separate():
    scene = Scene([cylinder(5, 0, z=6., height=2.)])
    assert scene.clear([0, 0, 2], [9, 0, 2])
    assert not scene.clear([0, 0, 6], [9, 0, 6])
    assert scene.geometry_mismatches  # Historical renderer grounds cylinders.


def test_clearance_run_is_not_a_single_frame_or_success_claim():
    assert runs([False, True, True, False, True]) == [(1, 3), (4, 5)]
    assert phase_label([False] * 5) == "initially_blocked_never_clear"
    assert phase_label([False, True, False]) == "initially_blocked_brief_clear_only"
    assert phase_label([False, True, True, True, False]) == "initially_blocked_clear_then_reblocked"
    assert phase_label([False, True, True, True]) == "initially_blocked_final_clear_run"
    assert phase_label([True] * 4) == "initially_clear_always_clear"


def test_range_absence_is_conservative():
    scene = Scene([cylinder(25, 0)])
    assert scene.certified_empty_view([0, 0, 2], 0.)
    assert scene.range_empty([0, 0, 2])
    assert not Scene([cylinder(5, 0)]).range_empty([0, 0, 2])


def test_clear_corridor_does_not_imply_original_obstacle_is_behind():
    scene = Scene([cylinder(5, 0)])
    axis, far = scene.initial_blocker_plane([0, 0, 2], [9, 0, 2])
    assert np.allclose(axis, [1, 0, 0])
    assert abs(far-5.6)<1e-6
    assert scene.clear([3, 3, 2], [9, 0, 2])
    assert np.dot([3, 3, 0], axis)<far


def test_delayed_approach_is_not_a_turn_that_was_already_executed():
    # Goal is behind; first four steps go away, the remaining steps return.
    x = np.r_[0., np.arange(1., 5.), np.arange(3., -7., -1.)]
    path = np.column_stack((x, np.zeros(15), np.ones(15)*2))
    f = plan_features(path, [-10, 0, 2], np.zeros(14), [1, 0, 0], 0.)
    assert f['approach_onset'] == 5
    assert f['flight_turn30_onset'] == 5
    assert f['yaw_turn30_onset'] is None
    assert f['h4_progress'] == -4 and f['tail_progress'] == 4
    # A later single good segment is insufficient for sustained approach.
    path[6:, 0] = np.arange(4., 13.)
    assert plan_features(path, [-10, 0, 2], np.zeros(14), [1, 0, 0], 0.)['approach_onset'] is None


def test_world_time_overlap_distinguishes_retention_from_reversal():
    old = np.column_stack((np.arange(9.), np.zeros(9), np.ones(9)*2))
    new = old[4:].copy()
    f = compare_replans(old, new, old[4], [20, 0, 2])
    assert f['overlap_rms_m'] == 0 and f['execution_endpoint_error_m'] == 0
    assert f['old_tail_progress'] == 4 and f['new_prefix_progress'] == 4
    new[:, 0] = [4, 3, 2, 1, 0]
    f = compare_replans(old, new, old[4], [20, 0, 2])
    assert f['new_prefix_progress'] == -4 and f['net_displacement_cosine'] == -1


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print("{} behavioral diagnostic tests passed".format(len(tests)))
