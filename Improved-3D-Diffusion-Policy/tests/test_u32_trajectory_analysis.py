"""Independent checks for the u32 BC/RL trajectory analysis."""

import json
from pathlib import Path

import numpy as np

from scripts.analyze_dppo_u32_trajectory_repeats import (
    aggregate_repeats_by_task,
    aggregate_tasks_by_world,
    obstacle_clearance,
    obstacle_clearances,
    obstacle_metrics,
    parse_trace_csv,
    planning_metrics,
    resolve_cli_path,
    trajectory_metrics,
    world_cluster_interval,
)


def _write_trace(path):
    path.write_text(
        "\n".join([
            "# policy-planned and executed trajectory of one endpoint episode",
            "# task_id=w__cfg00",
            "# world=w",
            "# split=rl_test",
            "# policy=bc",
            "# repeat=1",
            "# env_reset_seed=7",
            "# start=[0.0, 0.0, 1.0]",
            "# start_yaw=0.0",
            "# goal=[2.0, 0.0, 1.0]",
            "# spawn_offset=[0.0, 0.0, 0.0]",
            "# table: planned",
            "decision_step,k,time_s,x,y,z,vx,vy,vz,ax,ay,az,yaw,yaw_rate",
            "1,0,0.3,0.2,0,1,0,0,0,0,0,0,0,0",
            "1,1,0.6,0.4,0,1,0,0,0,0,0,0,0,0",
            "1,2,0.9,0.6,0,1,0,0,0,0,0,0,0,0",
            "1,3,1.2,0.8,0,1,0,0,0,0,0,0,0,0",
            "1,4,1.5,1.2,0,1,0,0,0,0,0,0,0,0",
            "1,5,1.8,1.5,0,1,0,0,0,0,0,0,0,0",
            "1,6,2.1,2.0,0,1,0,0,0,0,0,0,0,0",
            "1,7,2.4,2.3,0,1,0,0,0,0,0,0,0,0",
            "2,0,1.5,1.0,0,1,0,0,0,0,0,0,0,0",
            "2,1,1.8,1.4,0,1,0,0,0,0,0,0,0,0",
            "2,2,2.1,1.8,0,1,0,0,0,0,0,0,0,0",
            "2,3,2.4,2.0,0,1,0,0,0,0,0,0,0,0",
            "2,4,2.7,2.0,0,1,0,0,0,0,0,0,0,0",
            "2,5,3.0,2.0,0,1,0,0,0,0,0,0,0,0",
            "2,6,3.3,2.0,0,1,0,0,0,0,0,0,0,0",
            "2,7,3.6,2.0,0,1,0,0,0,0,0,0,0,0",
            "# table: executed",
            "decision_step,k,time_s,x,y,z",
            "1,0,0.0,0,0,1",
            "1,1,0.3,0.5,0,1",
            "1,2,0.6,1.0,0,1",
            "1,3,0.9,1.5,0,1",
            "1,4,1.2,1.8,0,1",
            "2,1,1.5,1.7,0,1",
            "2,2,1.8,1.8,0,1",
            "2,3,2.1,1.9,0,1",
            "2,4,2.4,2.0,0,1",
        ]),
        encoding="utf-8",
    )


def test_parse_trace_csv_keeps_both_tables_and_typed_header(tmp_path):
    path = tmp_path / "trace.csv"
    _write_trace(path)
    trace = parse_trace_csv(path)
    assert trace.header["task_id"] == "w__cfg00"
    assert trace.header["repeat"] == 1
    assert trace.header["start"] == [0.0, 0.0, 1.0]
    assert trace.planned.shape == (16, 14)
    assert trace.executed.shape == (9, 6)


def test_trajectory_metrics_are_derived_from_raw_polyline(tmp_path):
    path = tmp_path / "trace.csv"
    _write_trace(path)
    trace = parse_trace_csv(path)
    metrics = trajectory_metrics(trace.executed[:, 3:6], np.array(trace.header["goal"]))
    assert metrics["start_distance_m"] == 2.0
    assert metrics["final_distance_m"] == 0.0
    assert metrics["min_distance_m"] == 0.0
    assert metrics["path_length_m"] == 2.2
    assert np.isclose(metrics["backtrack_length_m"], 0.1)
    assert metrics["max_cross_track_m"] == 0.0
    assert metrics["entered_goal_0p5"] is True


def test_planning_metrics_detect_tail_only_goal_and_revision(tmp_path):
    path = tmp_path / "trace.csv"
    _write_trace(path)
    trace = parse_trace_csv(path)
    metrics = planning_metrics(trace.planned, np.array(trace.header["goal"]))
    assert metrics["decisions"] == 2
    assert metrics["tail_only_goal_decisions"] == 1
    # Decision 1 tail and decision 2 executed-horizon plan disagree at all 4 times.
    assert np.isclose(metrics["mean_tail_revision_m"], 0.2)


def test_obstacle_clearance_handles_cylinder_and_rotated_box():
    cylinder = {"name": "c", "shape": "cylinder", "x": 0, "y": 0,
                "z": 1, "radius": 1, "height": 2}
    box = {"name": "b", "shape": "box", "x": 0, "y": 0, "z": 1,
           "yaw": np.pi / 2, "size_x": 2, "size_y": 4, "size_z": 2}
    assert obstacle_clearance(np.array([2.0, 0.0, 1.0]), cylinder) == 1.0
    assert obstacle_clearance(np.array([3.0, 0.0, 1.0]), box) == 1.0
    assert obstacle_clearance(np.array([0.0, 0.0, 3.0]), cylinder) == 1.0
    points = np.array([[2.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    assert np.allclose(obstacle_clearances(points, cylinder), [1.0, 1.0])


def test_obstacle_metrics_use_start_to_goal_line_not_actual_endpoint():
    points = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    goal = np.array([2.0, 0.0, 1.0])
    obstacle = {"name": "c", "shape": "cylinder", "x": 1.0, "y": 0.0,
                "z": 1.0, "radius": 0.2, "height": 2.0}
    metrics = obstacle_metrics(points, goal, [obstacle])
    assert metrics["direct_path_min_clearance_m"] < 0.0
    assert metrics["lateral_onset_path_fraction"] == 1.0


def test_repeat_aggregation_keeps_task_as_unit():
    rows = [
        {"seed": "seed1", "task_id": "a", "world": "w", "role": "r", "repeat": 1,
         "delta_final_distance_m": 1.0, "bc_done": "reached", "rl_done": "time_limit"},
        {"seed": "seed1", "task_id": "a", "world": "w", "role": "r", "repeat": 2,
         "delta_final_distance_m": 3.0, "bc_done": "reached", "rl_done": "reached"},
        {"seed": "seed1", "task_id": "b", "world": "w", "role": "r", "repeat": 1,
         "delta_final_distance_m": -2.0, "bc_done": "collision", "rl_done": "reached"},
    ]
    result = aggregate_repeats_by_task(rows, ["delta_final_distance_m"])
    assert len(result) == 2
    by_task = {row["task_id"]: row for row in result}
    assert by_task["a"]["delta_final_distance_m"] == 2.0
    assert by_task["a"]["repeat_count"] == 2
    assert by_task["a"]["lost_rate"] == 0.5
    assert by_task["b"]["rescue_rate"] == 1.0
    assert by_task["a"]["delta_success_rate"] == -0.5


def test_world_cluster_interval_weights_worlds_not_task_replication():
    rows = [
        {"world": "w0", "effect": 1.0},
        {"world": "w0", "effect": 1.0},
        {"world": "w0", "effect": 1.0},
        {"world": "w1", "effect": -1.0},
    ]
    result = world_cluster_interval(rows, "effect", draws=1000, seed=7)
    assert result["worlds"] == 2
    assert result["tasks"] == 4
    assert result["estimate"] == 0.0
    assert result["lower"] <= 0.0 <= result["upper"]


def test_world_table_averages_four_task_units_without_reexpanding_repeats():
    rows = [
        {"seed": "seed1", "world": "w", "role": "r", "task_id": "a",
         "delta_success_rate": 1.0, "delta_final_distance_m": -2.0},
        {"seed": "seed1", "world": "w", "role": "r", "task_id": "b",
         "delta_success_rate": 0.0, "delta_final_distance_m": 2.0},
    ]
    result = aggregate_tasks_by_world(rows, ["delta_success_rate", "delta_final_distance_m"])
    assert result == [{"seed": "seed1", "world": "w", "role": "r", "task_count": 2,
                       "delta_success_rate": 0.5, "delta_final_distance_m": 0.0}]


def test_geometry_json_fixture_round_trip(tmp_path):
    geometry = tmp_path / "world.obstacles.json"
    geometry.write_text(json.dumps({"obstacles": [{
        "name": "c", "shape": "cylinder", "x": 0, "y": 0,
        "radius": 1, "height": 2,
    }]}), encoding="utf-8")
    loaded = json.loads(geometry.read_text(encoding="utf-8"))["obstacles"][0]
    assert obstacle_clearance(np.array([1.5, 0, 1]), loaded) == 0.5


def test_cli_paths_are_absolutized_before_long_trace_names_are_appended(tmp_path, monkeypatch):
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    resolved = resolve_cli_path(Path(".."))
    assert resolved.is_absolute()
    assert resolved == tmp_path.resolve()
