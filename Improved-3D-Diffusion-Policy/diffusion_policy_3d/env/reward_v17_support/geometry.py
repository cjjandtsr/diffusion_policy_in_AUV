"""Continuous queries, not a new contact policy or action postprocessor.

Contact reuses collision_geom. Clearance to each convex upright solid is
minimized on the continuous segment using batched golden-section brackets;
this is not waypoint sampling. Error <= segment_length * 0.619**56 (metres).
Footprint intervals are exact rectangle/circle intersections, with box corners
rounded by the existing Euclidean horizontal margin, not AABB inflation.
"""

from dataclasses import dataclass
import math

import numpy as np

from ..collision_geom import DRONE_RADIUS_Z, segment_obstacle_collision


@dataclass(frozen=True)
class GeometryContext:
    ground_z: float = 0.
    xy_min: tuple = (-30., -30.)
    xy_max: tuple = (30., 30.)

    def __post_init__(self):
        if not np.isfinite([self.ground_z, *self.xy_min, *self.xy_max]).all():
            raise ValueError("geometry context must be finite")
        if len(self.xy_min) != 2 or len(self.xy_max) != 2 or np.any(np.array(self.xy_min) >= self.xy_max):
            raise ValueError("invalid XY domain")


def validate_obstacles(obstacles):
    for o in obstacles:
        if o.shape not in ("box", "cylinder"):
            raise ValueError("unsupported obstacle shape: {}".format(o.shape))
        vals = [o.x, o.y, o.z, o.yaw, o.radius, o.height, o.size_x, o.size_y, o.size_z]
        sizes = [o.radius, o.height] if o.shape == "cylinder" else [o.size_x, o.size_y, o.size_z]
        if not np.isfinite(vals).all() or any(s <= 0 for s in sizes):
            raise ValueError("invalid obstacle geometry: {}".format(o.name))


def _clip(interval, value, delta, low, high):
    if interval is None:
        return None
    if delta == 0:
        return interval if low <= value <= high else None
    a, b = sorted(((low - value) / delta, (high - value) / delta))
    lo, hi = max(interval[0], a), min(interval[1], b)
    return (lo, hi) if lo <= hi else None


def _circle_interval(a, d, centre, radius):
    q = a - centre
    scale = max(float(np.max(np.abs(q))), float(np.max(np.abs(d))), radius, 1.)
    q, d, r = q / scale, d / scale, radius / scale
    aa, bb, cc = float(d @ d), float(q @ d), float(q @ q - r * r)
    if aa == 0:
        return (0., 1.) if cc <= 0 else None
    disc = bb * bb - aa * cc
    tolerance = 16 * np.finfo(float).eps * (bb * bb + abs(aa * cc))
    if disc < -tolerance:
        return None
    root = math.sqrt(max(0., disc))
    lo, hi = max(0., (-bb - root) / aa), min(1., (-bb + root) / aa)
    return (lo, hi) if lo <= hi else None


def footprint_interval(start, end, obstacle, margin):
    a = np.asarray(start[:2], dtype=np.float64) - [obstacle.x, obstacle.y]
    d = np.asarray(end[:2], dtype=np.float64) - np.asarray(start[:2], dtype=np.float64)
    if obstacle.shape == "cylinder":
        return _circle_interval(a, d, np.zeros(2), obstacle.radius + margin)
    c, s = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
    rotation = np.array([[c, s], [-s, c]])
    a, d = rotation @ a, rotation @ d
    hx, hy = obstacle.size_x / 2, obstacle.size_y / 2
    intervals = []
    for x, y in ((hx + margin, hy), (hx, hy + margin)):
        interval = _clip((0., 1.), a[0], d[0], -x, x)
        interval = _clip(interval, a[1], d[1], -y, y)
        if interval is not None:
            intervals.append(interval)
    for x in (-hx, hx):
        for y in (-hy, hy):
            interval = _circle_interval(a, d, np.array([x, y]), margin)
            if interval is not None:
                intervals.append(interval)
    return (min(v[0] for v in intervals), max(v[1] for v in intervals)) if intervals else None


def _paired_clearance(starts, ends, obstacles):
    """One segment per obstacle; all computations share vectorized brackets."""
    a, b = np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)
    if len(a) == 0:
        return np.empty(0)
    centres = np.array([[o.x, o.y, o.z] for o in obstacles])
    a, b = a - centres, b - centres
    delta = b - a
    cyl = np.array([o.shape == "cylinder" for o in obstacles])
    radius = np.array([o.radius for o in obstacles])
    hx, hy = np.array([o.size_x / 2 for o in obstacles]), np.array([o.size_y / 2 for o in obstacles])
    hz = np.array([(o.height if o.shape == "cylinder" else o.size_z) / 2 + DRONE_RADIUS_Z for o in obstacles])
    co, si = np.cos([o.yaw for o in obstacles]), np.sin([o.yaw for o in obstacles])

    def distance(t):
        p = a + t[:, None] * delta
        x, y = co * p[:, 0] + si * p[:, 1], -si * p[:, 0] + co * p[:, 1]
        box_xy = np.hypot(np.maximum(np.abs(x) - hx, 0.), np.maximum(np.abs(y) - hy, 0.))
        xy = np.where(cyl, np.maximum(np.hypot(p[:, 0], p[:, 1]) - radius, 0.), box_xy)
        return np.hypot(xy, np.maximum(np.abs(p[:, 2]) - hz, 0.))

    lo, hi = np.zeros(len(a)), np.ones(len(a))
    endpoint = np.minimum(distance(lo), distance(hi))
    if np.array_equal(a, b):
        return endpoint
    ratio = (math.sqrt(5.) - 1.) / 2.
    x, y = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fx, fy = distance(x), distance(y)
    for _ in range(56):
        left = fx <= fy
        hi, lo = np.where(left, y, hi), np.where(left, lo, x)
        x, y = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        fx, fy = distance(x), distance(y)
    result = np.minimum(endpoint, np.minimum(fx, fy))
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite continuous clearance")
    return result


def segment_clearances(starts, ends, obstacles):
    if not obstacles:
        return np.full((len(starts), 0), 100.)
    n = len(obstacles)
    return _paired_clearance(np.repeat(starts, n, axis=0), np.repeat(ends, n, axis=0),
                             list(obstacles) * len(starts)).reshape(len(starts), n)


def risk(distance, cfg):
    x = (cfg["d_risk"] - distance) / cfg["s_d"]
    return 1. / (1. + math.exp(-x)) if x >= 0 else math.exp(x) / (1. + math.exp(x))


def _complement(intervals):
    cursor, result = 0., []
    for low, high in sorted(interval for interval in intervals if interval is not None):
        if low > cursor:
            result.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < 1.:
        result.append((cursor, 1.))
    return result


def path_geometry(path, obstacles, context, cfg):
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("path geometry requires finite Nx3 executed positions")
    validate_obstacles(obstacles)
    starts, ends = p[:-1], p[1:]
    ground = float(np.min(p[:, 2]) - context.ground_z)
    hit = next((hit for a, b in zip(starts, ends)
                for hit in [segment_obstacle_collision(a, b, obstacles, cfg["collision_margin_xy"])]
                if hit is not None), None)
    collision = hit is not None or ground < cfg["ground_contact"]
    out = bool(np.any(p[:, :2] < context.xy_min) or np.any(p[:, :2] > context.xy_max))
    distances = segment_clearances(starts, ends, obstacles)
    per_segment = np.min(distances, axis=1) if obstacles else np.full(len(starts), 100.)
    weights = np.exp(-cfg["time_decay"] * np.arange(len(starts)))
    weights /= weights.sum()
    smallest = float(np.min(per_segment))
    soft = smallest - math.log(float(np.sum(weights * np.exp(-cfg["softmin_beta"] * (per_segment - smallest))))) / cfg["softmin_beta"]
    if collision:
        soft = 0.
    related, encounters, gs, ge, go = set(), [], [], [], []
    for j, (a, b) in enumerate(zip(starts, ends)):
        for index, o in enumerate(obstacles):
            interval = footprint_interval(a, b, o, cfg["collision_margin_xy"])
            top = o.z + (o.height if o.shape == "cylinder" else o.size_z) / 2 + DRONE_RADIUS_Z
            over = None
            if interval is not None:
                related.add(index)
                over = _clip(interval, a[2], b[2] - a[2], top, float("inf"))
                if over is not None:
                    cmin = min(a[2] + t * (b[2] - a[2]) - top for t in over)
                    if cmin > 0:
                        encounters.append(float(cmin))
                    else:
                        over = None  # contact uses hard truth, never a soft flyover
            bottom = o.z - (o.height if o.shape == "cylinder" else o.size_z) / 2 - DRONE_RADIUS_Z
            contact = _clip(interval, a[2], b[2] - a[2], bottom, top)
            # Exclude contact with this obstacle, not the entire failed path.
            # Boundary limits belong to the legal neighboring soft region.
            intervals = _complement((over, contact))
            for low, high in intervals:
                if high > low:
                    gs.append(a + low * (b - a)); ge.append(a + high * (b - a)); go.append(o)
    generic = float(np.min(_paired_clearance(gs, ge, go))) if gs else 100.
    delta = ends - starts
    length = float(np.sum(np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])))
    chord = float(np.linalg.norm(p[-1] - p[0]))
    return dict(collision=bool(collision), obstacle_collision=hit is not None,
                ground_collision=ground < cfg["ground_contact"], out_of_bounds=out,
                raw_done_reason="crash_ground" if ground < cfg["ground_contact"] else
                ("collide({})".format(hit[1]) if hit else ("out_of_bounds" if out else "none")),
                physical_safe=not (collision or out), ground_clearance=ground,
                d_min=0. if collision else smallest, d_soft=soft, risk=risk(soft, cfg),
                generic_clearance=generic, related_count=len(related),
                overflight_clearances=encounters, path_length=length, chord_length=chord,
                excess_length=max(0., length - chord))


def direct_path(path, goal):
    p = np.asarray(path, dtype=np.float64)
    axis = np.asarray(goal, dtype=np.float64)[:2] - p[0, :2]
    norm = np.hypot(*axis)
    if norm == 0:
        return None
    delta = np.diff(p, axis=0)
    horizontal = np.hypot(delta[:, 0], delta[:, 1])[:, None] * (axis / norm)
    return np.vstack((p[0], p[0] + np.cumsum(np.column_stack((horizontal, delta[:, 2])), axis=0)))


def free_potential(position, goal, obstacles, context, cfg):
    p = np.asarray(position, dtype=np.float64)
    axis = np.asarray(goal, dtype=np.float64) - p
    norm = np.linalg.norm(axis)
    if norm == 0:
        return 0.
    axis /= norm
    reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
    side = np.cross(reference, axis); side /= np.linalg.norm(side)
    up = np.cross(axis, side)
    rays = []
    for yaw in np.linspace(-cfg["free_cone_angle"], cfg["free_cone_angle"], 5):
        for pitch in np.linspace(-cfg["free_cone_angle"] / 2, cfg["free_cone_angle"] / 2, 3):
            y, z = math.radians(yaw), math.radians(pitch)
            rays.append(math.cos(z) * (math.cos(y) * axis + math.sin(y) * side) + math.sin(z) * up)
    # Preserve the historical 10m query. Linear quantile interpolation and
    # element-wise clipping do not commute near the potential cap.
    distances = np.arange(cfg["free_ray_step"], cfg["free_ray_range"] + cfg["free_ray_step"] / 2, cfg["free_ray_step"])
    points = p + np.asarray(rays)[:, None, :] * distances[None, :, None]
    flat = points.reshape(-1, 3)
    ds = segment_clearances(flat, flat, obstacles)
    near = np.min(ds, axis=1) < cfg["free_ray_clearance"] if obstacles else np.zeros(len(flat), dtype=bool)
    near |= flat[:, 2] < context.ground_z
    hits = near.reshape(len(rays), -1)
    free = np.where(np.any(hits, axis=1), distances[np.argmax(hits, axis=1)], cfg["free_ray_range"])
    return float(cfg["lambda_f"] * np.clip(np.percentile(free, 20), 0., cfg["C_f"]))
