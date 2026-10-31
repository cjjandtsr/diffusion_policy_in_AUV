"""Locate the first event on the four physically executed linear segments."""

from dataclasses import dataclass
import math

import numpy as np

from ..collision_geom import segment_obstacle_collision


@dataclass(frozen=True)
class FirstEvent:
    kind: str
    segment_index: int
    fraction: float
    position: np.ndarray
    obstacle_name: str = ""


def _ball_entry(start, end, centre, radius):
    a = np.asarray(start, dtype=np.float64) - centre
    d = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    if float(a @ a) <= radius * radius:
        return 0.0
    aa = float(d @ d)
    if aa == 0.:
        return None
    bb = float(a @ d)
    cc = float(a @ a) - radius * radius
    disc = bb * bb - aa * cc
    if disc < 0.:
        return None
    t = (-bb - math.sqrt(disc)) / aa
    return float(t) if 0. <= t <= 1. else None


def _obstacle_entry(start, end, obstacle, collision_thresh):
    if obstacle.shape == "sphere":
        return _ball_entry(start, end,
                           np.array([obstacle.x, obstacle.y, obstacle.z]),
                           obstacle.radius + collision_thresh)
    if obstacle.shape not in ("cylinder", "box"):
        raise ValueError("unknown obstacle shape: {}".format(obstacle.shape))

    a, b = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)

    def ever_hit(fraction):
        tip = a + fraction * (b - a)
        return segment_obstacle_collision(a, tip, [obstacle], collision_thresh) is not None

    if not ever_hit(1.):
        return None
    if ever_hit(0.):
        return 0.
    lo, hi = 0., 1.
    for _ in range(24):
        middle = (lo + hi) / 2.
        if ever_hit(middle):
            hi = middle
        else:
            lo = middle
    return hi


def _lower_crossing(a, b, threshold):
    if a < threshold:
        return 0.
    if b >= threshold:
        return None
    return float((threshold - a) / (b - a))


def _upper_crossing(a, b, threshold):
    if a > threshold:
        return 0.
    if b <= threshold:
        return None
    return float((threshold - a) / (b - a))


def locate_first_event(path, goal, obstacles, collision_thresh, z_max,
                       goal_radius=.5, tie_tolerance=1e-6,
                       terminate_at_space_limits=True):
    """Return earliest event under the selected space-limit rule; failure wins ties."""
    p = np.asarray(path, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or g.shape != (3,):
        raise ValueError("first-event path must be Nx3 and goal must be xyz")
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("first-event inputs must be finite")
    if not (0. < goal_radius and 0. <= collision_thresh and math.isfinite(z_max)):
        raise ValueError("invalid first-event geometry thresholds")

    best = None
    best_clock = math.inf
    for index, (start, end) in enumerate(zip(p[:-1], p[1:])):
        candidates = []
        goal_t = _ball_entry(start, end, g, goal_radius)
        if goal_t is not None:
            candidates.append((goal_t, "reached", ""))
        for obstacle in obstacles:
            t = _obstacle_entry(start, end, obstacle, collision_thresh)
            if t is not None:
                candidates.append((t, "collision", obstacle.name))
        ground_t = _lower_crossing(start[2], end[2], .2)
        if ground_t is not None:
            candidates.append((ground_t, "ground_crash", ""))
        # Preserve the running training and historical trace contract. The
        # shared evaluator explicitly disables these artificial terminations.
        if terminate_at_space_limits:
            for axis in (0, 1):
                for side in (-30., 30.):
                    t = (_lower_crossing(start[axis], end[axis], side) if side < 0.
                         else _upper_crossing(start[axis], end[axis], side))
                    if t is not None:
                        candidates.append((t, "out_of_bounds", ""))
            altitude_t = _upper_crossing(start[2], end[2], z_max + 1.)
            if altitude_t is not None:
                candidates.append((altitude_t, "altitude_exceed", ""))
        for t, kind, obstacle_name in candidates:
            clock = index + t
            if (clock < best_clock - tie_tolerance or
                (abs(clock - best_clock) <= tie_tolerance and
                 best is not None and best.kind == "reached" and kind != "reached")):
                best_clock = clock
                best = FirstEvent(kind, index, float(t), start + t * (end - start), obstacle_name)
        if best is not None and best.segment_index <= index:
            break
    return best
