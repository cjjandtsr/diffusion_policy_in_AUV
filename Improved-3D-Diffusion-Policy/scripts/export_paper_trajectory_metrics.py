"""Read completed raw traces; adjudicate paper 1.5m events and continuous margins."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_policy_3d.env.collision_geom import ObstacleSpec
from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
from analyze_paper_success_geometry import overlap_clearance
from routea_paired_support.core import trace_metrics
from diffusion_policy_3d.env.collision_geom import DRONE_RADIUS_Z


def euclidean_segment_clearance(a, b, obstacle):
    """Nonnegative surface distance: vertical cylindrical body (.1m XY/Z).

    Convex scalar minimization on the executed segment, xatol=1e-10 in
    segment fraction, maxiter=200; compare both endpoints explicitly.
    """
    from scipy.optimize import minimize_scalar
    a, b = np.asarray(a, float), np.asarray(b, float)
    height = obstacle.height if obstacle.shape == 'cylinder' else obstacle.size_z
    def distance(t):
        p = a + t * (b-a)
        if obstacle.shape == 'cylinder':
            xy = max(float(np.linalg.norm(p[:2]-[obstacle.x, obstacle.y])) - obstacle.radius, 0.)
        else:
            c, s = np.cos(obstacle.yaw), np.sin(obstacle.yaw)
            dx, dy = p[0]-obstacle.x, p[1]-obstacle.y
            xy = float(np.hypot(max(abs(c*dx+s*dy)-obstacle.size_x/2, 0.),
                                max(abs(-s*dx+c*dy)-obstacle.size_y/2, 0.)))
        horizontal = max(xy-.1, 0.)
        vertical = max(abs(p[2]-obstacle.z)-height/2-DRONE_RADIUS_Z, 0.)
        return float(np.hypot(horizontal, vertical))
    optimum = minimize_scalar(distance, bounds=(0., 1.), method='bounded',
                              options=dict(xatol=1e-10, maxiter=200))
    if not optimum.success:
        raise RuntimeError(optimum.message)
    return min(distance(0.), distance(1.), float(optimum.fun))


def measure(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    header, terminal = rows[0], rows[-1]
    if header['type'] != 'episode_start' or terminal['type'] != 'episode_end':
        raise ValueError('requires completed raw episode')
    transitions = [r for r in rows if r['type'] == 'transition']
    paths = [np.asarray(r['executed_path'], dtype=float) for r in transitions]
    for a, b in zip(paths, paths[1:]):
        if not np.allclose(a[-1], b[0], atol=1e-6, rtol=0):
            raise ValueError('discontinuous executed path')
    xyz = (np.vstack([paths[0][:1]] + [p[1:] for p in paths]) if paths else
           np.asarray([header['task']['start_pos']], dtype=float))
    goal = np.asarray(header['task']['goal_pos'], dtype=float)
    obstacles = [ObstacleSpec.from_dict(o) for o in header['geometry']]
    if any(o.shape not in ('box', 'cylinder') for o in obstacles):
        raise ValueError('continuous overlap metric supports archived boxes and cylinders only')
    event = locate_first_event(np.vstack((xyz, xyz[-1])) if len(xyz) == 1 else xyz, goal, obstacles, .1, header['task']['z_range'][1],
                              goal_radius=1.5,
                              terminate_at_space_limits=header.get('terminate_at_space_limits', True))
    outcome = event.kind if event is not None else terminal['done_reason']
    if event is not None:
        xyz = np.vstack((xyz[:event.segment_index+1], event.position))
    clearance_xyz = xyz if len(xyz) > 1 else np.vstack((xyz, xyz[-1]))
    gaps = [gap - .1 for a, b in zip(clearance_xyz[:-1], clearance_xyz[1:]) for o in obstacles
            if (gap := overlap_clearance(a, b, o)) is not None]
    euclidean = [euclidean_segment_clearance(a, b, o)
                 for a, b in zip(clearance_xyz[:-1], clearance_xyz[1:]) for o in obstacles]
    length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    geometry_metrics = trace_metrics(xyz, goal)
    initial_distance = float(np.linalg.norm(xyz[0]-goal))
    updates = [u for r in transitions for u in r.get('ego', {}).get('updates', [])]
    return dict(policy=header['policy'], task_id=header['task_id'], world_id=header['world_id'],
                repeat=header['repeat'], paper_safe_arrival=outcome == 'reached',
                paper_done_reason=outcome, paper_goal_radius_m=1.5,
                source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
                path_length_m=length, remaining_goal_distance_m=float(np.linalg.norm(xyz[-1]-goal)),
                paper_path_efficiency=initial_distance/(length+float(np.linalg.norm(xyz[-1]-goal))) if length+float(np.linalg.norm(xyz[-1]-goal)) else None,
                paper_path_efficiency_definition='Initial straight distance / (executed length to paper first event + endpoint-to-goal distance), v4.2 section 4.1.4; aggregate only successful flights of successful tasks.',
                initial_goal_distance_m=initial_distance,
                min_goal_distance_m=geometry_metrics['min_goal_distance_m'],
                straight_distance_over_path_length=initial_distance/length if length else None,
                path_ratio_note='Explicit geometric ratio only; not asserted as paper-defined efficiency. Can exceed 1 with radius-based arrival.',
                planner_updates=updates,
                planner_planning_seconds_samples=[u['planning_seconds'] for u in updates if 'planning_seconds' in u],
                planner_service_seconds_samples=[u['service_seconds'] for u in updates if 'service_seconds' in u],
                planner_failed_plans_cumulative=max((u['failed_plans'] for u in updates if 'failed_plans' in u), default=None),
                planner_measurement_boundary='planning_seconds is bridge service callback wall time (not isolated optimizer); service_seconds includes ROS communication/wait. Raw updates retained; timings include original execution after paper cutoff and are not per-replan isolated timings.',
                min_continuous_collision_margin_m=min(gaps) if gaps else None,
                min_continuous_euclidean_clearance_m=min(euclidean) if euclidean else None,
                euclidean_definition='Nonnegative distance between cylinder body radius=.1m halfheight=.1m and solid obstacle; bounded convex minimization xatol=1e-10 segment fraction plus endpoints',
                clearance_definition='XY gap during vertical envelope overlap minus 0.1m; null if no overlap',
                executed_xyz=xyz.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--paired-summary')
    args = parser.parse_args()
    records = [measure(path) for path in args.traces]
    with Path(args.output).open('x', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    if args.paired_summary:
        with Path(args.paired_summary).open('x', encoding='utf-8') as stream:
            json.dump(paired_summary(records), stream, ensure_ascii=False, indent=2, allow_nan=False)


def paired_summary(records):
    """Common majority-success tasks; successful repeats retain their own denominators."""
    from itertools import combinations
    groups = defaultdict(list)
    for row in records:
        groups[(row['policy'], row['task_id'])].append(row)
    successes = defaultdict(set)
    for (policy, task), rows in groups.items():
        if len(rows) != 5 or len({r['repeat'] for r in rows}) != 5:
            raise ValueError('paired task metrics require five distinct repeats')
        if sum(r['paper_safe_arrival'] for r in rows) >= 3:
            successes[policy].add(task)
    def stats(rows):
        values = [r['min_continuous_euclidean_clearance_m'] for r in rows
                  if r['min_continuous_euclidean_clearance_m'] is not None]
        return dict(successful_episodes=len(rows), finite_margin_episodes=len(values),
                    tasks=len({r['task_id'] for r in rows}),
                    median_margin_m=float(np.median(values)) if values else None)
    overall = {p: stats([r for r in records if r['policy'] == p and r['paper_safe_arrival']])
               for p in sorted({r['policy'] for r in records})}
    pairs = []
    for a, b in combinations(sorted(overall), 2):
        common = successes[a] & successes[b]
        paired = []
        for task in sorted(common):
            values = []
            for p in (a, b):
                finite = [r['min_continuous_euclidean_clearance_m'] for r in groups[p, task]
                          if r['paper_safe_arrival'] and r['min_continuous_euclidean_clearance_m'] is not None]
                values.append(float(np.median(finite)) if finite else None)
            paired.append(dict(task_id=task, reference_margin_m=values[0], treatment_margin_m=values[1],
                               difference_m=values[1]-values[0] if None not in values else None))
        differences = [r['difference_m'] for r in paired if r['difference_m'] is not None]
        pairs.append(dict(reference=a, treatment=b, common_success_tasks=len(common),
                          effective_paired_tasks=len(differences), paired_tasks=paired,
                          median_paired_difference_m=float(np.median(differences)) if differences else None))
    efficiency = {}
    for policy in overall:
        selected = [r for r in records if r['policy'] == policy and r['task_id'] in successes[policy]
                    and r['paper_safe_arrival'] and r.get('paper_path_efficiency') is not None]
        efficiency[policy] = dict(successful_tasks=len(successes[policy]), successful_flights=len(selected),
            median=float(np.median([r['paper_path_efficiency'] for r in selected])) if selected else None)
    return dict(overall_successful_flights=overall, comparisons=pairs,
                paper_path_efficiency_successful_tasks=efficiency,
                clearance_metric='min_continuous_euclidean_clearance_m',
                limitation='Nonnegative Euclidean surface distances; collisions are independently adjudicated by the original event rules. '
                           'Empty geometry is missing, never zero or infinity. Per-task values use successful-repeat medians.')


if __name__ == '__main__':
    main()
