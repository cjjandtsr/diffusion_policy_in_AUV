#!/usr/bin/env python3
"""Recompute executed-route geometry for the paper's five positive tasks.

This reads archived P2/P3 traces only. Clearance is the minimum horizontal
surface gap while the UAV's vertical interval overlaps an obstacle, matching
the environment's continuous-segment collision proxy (threshold 0.1 m).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_policy_3d.env.collision_geom import (  # noqa: E402
    DRONE_RADIUS_Z, ObstacleSpec, _clip_axis_interval,
    _point_segment_distance_2d, _segment_rect_distance_2d,
)

POLICIES = ('bc', 'rl_2f3f9914db78_u32')
P2_TASKS = (20, 30, 32)
P3_TASKS = (9, 13)


def overlap_clearance(a, b, obstacle):
    """Minimum XY surface gap along one segment during vertical overlap."""
    height = obstacle.height if obstacle.shape == 'cylinder' else obstacle.size_z
    interval = _clip_axis_interval(0.0, 1.0, a[2], b[2] - a[2],
                                   obstacle.z - height / 2 - DRONE_RADIUS_Z,
                                   obstacle.z + height / 2 + DRONE_RADIUS_Z)
    if interval is None:
        return None
    ends = [(a[:2] + t * (b[:2] - a[:2])) - (obstacle.x, obstacle.y)
            for t in interval]
    if obstacle.shape == 'cylinder':
        return _point_segment_distance_2d((0.0, 0.0), ends[0], ends[1]) - obstacle.radius
    c, s = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
    local = [(c * p[0] + s * p[1], -s * p[0] + c * p[1]) for p in ends]
    return _segment_rect_distance_2d(local[0], local[1],
                                     obstacle.size_x / 2, obstacle.size_y / 2)


def above_footprint_crossings(xyz, obstacles):
    """Obstacle footprints intersected while the UAV is fully above them."""
    names = set()
    for a, b in zip(xyz[:-1], xyz[1:]):
        for obstacle in obstacles:
            top = obstacle.z + (obstacle.height if obstacle.shape == 'cylinder'
                                else obstacle.size_z) / 2 + DRONE_RADIUS_Z + 1e-6
            if max(a[2], b[2]) <= top:
                continue
            t0 = 0.0 if a[2] >= top else (top - a[2]) / (b[2] - a[2])
            t1 = 1.0 if b[2] >= top else (top - a[2]) / (b[2] - a[2])
            if t0 > t1:
                t0, t1 = t1, t0
            points = [a + t * (b - a) for t in (t0, t1)]
            if obstacle.shape == 'cylinder':
                gap = _point_segment_distance_2d((obstacle.x, obstacle.y),
                                                 points[0][:2], points[1][:2]) - obstacle.radius
            else:
                c, s = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
                local = [(c * (p[0] - obstacle.x) + s * (p[1] - obstacle.y),
                          -s * (p[0] - obstacle.x) + c * (p[1] - obstacle.y)) for p in points]
                gap = _segment_rect_distance_2d(local[0], local[1],
                                                 obstacle.size_x / 2, obstacle.size_y / 2)
            if gap < 1e-6:
                names.add(obstacle.name)
    return sorted(names)


def trace_metrics(path, expected_policy, expected_task, expected_repeat):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    header, terminal = rows[0], rows[-1]
    steps = [row for row in rows if row['type'] == 'transition']
    assert header['policy'] == expected_policy
    assert header['repeat'] == expected_repeat
    assert header['use_base_policy'] == (expected_policy == 'bc')
    assert header['task']['goal_pos'] and steps
    assert terminal['safe_arrival'] == (terminal['done_reason'] == 'reached')
    assert header['task_id'] == expected_task
    paths = [np.asarray(step['executed_path'], dtype=float) for step in steps]
    assert all(np.allclose(x[-1], y[0], atol=1e-6) for x, y in zip(paths, paths[1:]))
    xyz = np.vstack([paths[0][:1]] + [segment[1:] for segment in paths])
    goal = np.asarray(header['task']['goal_pos'], dtype=float)
    obstacles = [ObstacleSpec.from_dict(o) for o in header['geometry']]
    assert all(o.shape in ('box', 'cylinder') for o in obstacles)
    gaps = [(overlap_clearance(a, b, obstacle), obstacle.name)
            for a, b in zip(xyz[:-1], xyz[1:]) for obstacle in obstacles]
    gaps = [(gap, name) for gap, name in gaps if gap is not None]
    closest = min(gaps, default=(None, None), key=lambda item: item[0])
    start = xyz[0]
    axis = goal[:2] - start[:2]
    axis /= np.linalg.norm(axis)
    relative_xy = xyz[:, :2] - start[:2]
    signed_lateral = axis[0] * relative_xy[:, 1] - axis[1] * relative_xy[:, 0]
    lateral = float(signed_lateral[np.argmax(np.abs(signed_lateral))])
    length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    duration = float(sum(step['executed_time_s'] for step in steps))
    summary = terminal.get('trajectory_metrics', {})
    if 'path_length_m' in summary:
        assert abs(length - summary['path_length_m']) < 1e-5
        assert abs(duration - terminal['physical_time_s']) < 1e-5
    return dict(policy=expected_policy, task=expected_task, repeat=expected_repeat,
                outcome=terminal['done_reason'], safe_arrival=terminal['safe_arrival'],
                world=header['world_id'], trace=str(path), trace_sha256=hashlib.sha256(raw).hexdigest(),
                path_length_m=length, duration_s=duration, z_min_m=float(xyz[:, 2].min()),
                z_max_m=float(xyz[:, 2].max()), z_span_m=float(np.ptp(xyz[:, 2])),
                max_abs_xy_lateral_m=abs(lateral), signed_peak_xy_lateral_m=lateral,
                above_footprint_obstacles=above_footprint_crossings(xyz, obstacles),
                min_xy_gap_during_vertical_overlap_m=closest[0], nearest_obstacle=closest[1],
                min_collision_margin_m=None if closest[0] is None else closest[0] - 0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    p2 = json.loads((workspace / 'jindu/2026-10-15_v3_全策略原轨迹分析数据.json').read_text(encoding='utf-8'))
    p3 = json.loads((workspace / 'wsl_snapshot/paper_p3_limited_v1_run01/results/paired_index.json').read_text(encoding='utf-8'))
    p3_tasks = json.loads((workspace / '论文/研究报告/2026-10-21_论文P3与有限任务评价独立验收数据.json').read_text(encoding='utf-8'))['per_task']
    p2_index = {(r['policy'], r['task'], r['repeat']): r for r in p2['records']}
    p3_ids = {r['task']: r['task_id'] for r in p3_tasks if r['task'] in P3_TASKS}
    p3_index = {(r['task_id'], r['repeat']): r for r in p3['records']}
    result = []
    for task in P2_TASKS:
        for policy in POLICIES:
            for repeat in range(1, 6):
                source = p2_index[(policy, task, repeat)]
                row = trace_metrics(workspace / source['source'], policy, source['task_id'], repeat)
                assert row['safe_arrival'] == source['success']
                assert abs(row['z_span_m'] - source['z_span_m']) < 1e-5
                result.append({**row, 'task': 'T%02d' % task})
    for task in P3_TASKS:
        task_id = p3_ids[task]
        for policy in POLICIES:
            for repeat in range(1, 6):
                entry = p3_index[(task_id, repeat)]['models'][policy]
                base = workspace / 'wsl_snapshot/paper_p3_limited_v1_run01/results/full' / policy
                commit = json.loads((base / entry['commit']).read_text(encoding='utf-8'))
                row = trace_metrics(base / commit['trace'], policy, task_id, repeat)
                assert row['trace_sha256'] == commit['sha256'] == entry['trace_sha256']
                assert row['safe_arrival'] == commit['summary']['safe_arrival']
                result.append({**row, 'task': 'N%02d' % task})
    assert len(result) == 50
    summary = []
    for task in ('T20', 'T30', 'T32', 'N09', 'N13'):
        for policy in POLICIES:
            group = [r for r in result if r['task'] == task and r['policy'] == policy]
            arrivals = [r for r in group if r['safe_arrival']]
            summary.append(dict(task=task, policy=policy, arrivals=sum(r['safe_arrival'] for r in group),
                                z_span_median=float(np.median([r['z_span_m'] for r in group])),
                                abs_xy_lateral_median=float(np.median([r['max_abs_xy_lateral_m'] for r in group])),
                                above_footprint_repeats=sum(bool(r['above_footprint_obstacles']) for r in group),
                                arrival_margin_min_m=min((r['min_collision_margin_m'] for r in arrivals
                                                          if r['min_collision_margin_m'] is not None), default=None),
                                all_repeat_margin_min_m=min(r['min_collision_margin_m'] for r in group
                                                            if r['min_collision_margin_m'] is not None)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(definition='continuous XY gap during vertical overlap minus 0.1 m collision threshold',
                                          records=result, summary=summary), ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', args.output, 'records', len(result))


if __name__ == '__main__':
    main()
