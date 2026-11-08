"""First-event evaluation of world-frame odometry, without ROS or policy changes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
from diffusion_policy_3d.env.collision_geom import load_obstacles


def check_sampling(times):
    intervals = np.diff(np.asarray(times, float))
    if len(intervals) == 0 or not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError('invalid odometry timestamps')
    if intervals.max() > 1. / 30 + 1e-6:
        raise ValueError('odometry has intervals longer than 1/30 second; do not interpolate missing data')
    return dict(min_rate_hz=float(1 / intervals.max()), max_gap_s=float(intervals.max()))


def classify_odom(times, positions, goal, obstacles):
    times, positions = np.asarray(times, float), np.asarray(positions, float)
    if (len(times) < 2 or positions.shape != (len(times), 3) or times[0] != 0.
            or not np.isfinite(times).all() or not np.isfinite(positions).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('require finite world xyz and strictly increasing relative seconds starting at 0')
    deadline = 72.
    if times[-1] > deadline:
        index = int(np.searchsorted(times, deadline))
        fraction = (deadline - times[index - 1]) / (times[index] - times[index - 1])
        tip = positions[index - 1] + fraction * (positions[index] - positions[index - 1])
        positions = np.vstack((positions[:index], tip))
        times = np.r_[times[:index], deadline]
    event = locate_first_event(positions, goal, obstacles, .1, 5., terminate_at_space_limits=False)
    if event is None:
        if times[-1] < deadline:
            raise ValueError('incomplete odometry cannot establish a timeout')
        return dict(done_reason='time_limit', event_time_s=deadline, event=None)
    seconds = times[event.segment_index] + event.fraction * (
        times[event.segment_index + 1] - times[event.segment_index])
    detail = asdict(event)
    detail['position'] = event.position.tolist()
    return dict(done_reason=event.kind, event_time_s=float(seconds), event=detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--odom', required=True, help='JSONL: relative time_s, position_world xyz')
    parser.add_argument('--obstacles', required=True)
    parser.add_argument('--goal', required=True, type=float, nargs=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.odom).read_text(encoding='utf-8').splitlines()]
    sampling = check_sampling([r['time_s'] for r in rows])
    result = classify_odom([r['time_s'] for r in rows], [r['position_world'] for r in rows],
                           args.goal, load_obstacles(args.obstacles))
    result['sampling'] = sampling
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(result['done_reason'], result['event_time_s'])


if __name__ == '__main__':
    main()
