"""
collision_geom.py — 几何碰撞检测 (纯几何, 无 ROS)

抽自 collision_timeout_monitor.py 的已验证几何函数, RL 环境与实验共享。
- ObstacleSpec: 障碍几何 (cylinder / box), 从 .obstacles.json 加载
- point_to_oriented_rect_distance: 点到带朝向长方体的 XY 距离
- compute_obstacle_clearance: 表面到表面 clearance + 飞越豁免 (world frame)
- segment_obstacle_collision: 同一碰撞代理下的线段连续判撞 (不改轨迹)
- load_obstacles: 从 json 加载障碍列表

垂直分离豁免: 无人机底部严格高于障碍顶, 或顶部严格低于悬空障碍底,
即使 XY 重叠也不算接触。接触/重叠不得豁免; 正净空不足5cm不等于碰撞。
沿用 drone_radius_z=0.1 的直立机体半高代理, 本修正不重新标定机体尺寸。

★坐标系: 全部 world frame。drone 位置与 obstacles.json 必须同 frame。
"""

import math
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


# Shared by point and segment checks; existing upright half-height proxy.
DRONE_RADIUS_Z = 0.1


class GeometryPurpose(str, Enum):
    ADMISSION_ENDPOINT = "admission_endpoint"
    PHYSICAL_COLLISION = "physical_collision"
    SENSOR_RAY = "sensor_ray"
    AUDIT_REFERENCE = "audit_reference"


# ─────────────────────────────────────────────────────────────────
# 障碍几何
# ─────────────────────────────────────────────────────────────────

@dataclass
class ObstacleSpec:
    """Obstacle geometry loaded from .obstacles.json."""
    name: str
    shape: str  # "cylinder" or "box"
    x: float
    y: float
    z: float       # center z (= h/2 for ground-sitting)
    yaw: float = 0.0
    radius: float = 0.0      # cylinder
    height: float = 0.0      # cylinder
    size_x: float = 0.0      # box
    size_y: float = 0.0
    size_z: float = 0.0

    @property
    def bounding_r(self) -> float:
        if self.shape in ("sphere", "cylinder"):
            return self.radius
        return math.hypot(self.size_x, self.size_y) / 2.0

    @classmethod
    def from_dict(cls, d: dict) -> "ObstacleSpec":
        return cls(
            name=d.get("name", "unknown"),
            shape=d.get("shape", "cylinder"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            z=d.get("z", d.get("height", 1.0) / 2 if d.get("shape") == "cylinder"
                     else d.get("size_z", 1.0) / 2),
            yaw=d.get("yaw", 0.0),
            radius=d.get("radius", 0.0),
            height=d.get("height", 0.0),
            size_x=d.get("size_x", 0.0),
            size_y=d.get("size_y", 0.0),
            size_z=d.get("size_z", 0.0),
        )


def _validated_point(point_xyz):
    try:
        point = tuple(float(value) for value in point_xyz)
    except (TypeError, ValueError) as exc:
        raise TypeError("point_xyz must contain three numeric values") from exc
    if len(point) != 3:
        raise ValueError("point_xyz must have shape (3,)")
    if not all(math.isfinite(value) for value in point):
        raise ValueError("point_xyz must be finite")
    return point


def _validate_obstacle_spec(obstacle: ObstacleSpec) -> None:
    if not isinstance(obstacle, ObstacleSpec):
        raise TypeError("obstacle must be an ObstacleSpec")
    common = (obstacle.x, obstacle.y, obstacle.z, obstacle.yaw)
    if not all(math.isfinite(float(value)) for value in common):
        raise ValueError("obstacle center and yaw must be finite")
    if obstacle.shape == "sphere":
        dimensions = (obstacle.radius,)
    elif obstacle.shape == "cylinder":
        dimensions = (obstacle.radius, obstacle.height)
    elif obstacle.shape == "box":
        dimensions = (obstacle.size_x, obstacle.size_y, obstacle.size_z)
    else:
        raise ValueError("unknown obstacle shape: {!r}".format(obstacle.shape))
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in dimensions):
        raise ValueError("obstacle dimensions must be finite and positive")


def signed_clearance(
    point_xyz,
    obstacle: ObstacleSpec,
    *,
    purpose: GeometryPurpose,
    body_radius_m: float = 0.0,
) -> float:
    """Signed Euclidean clearance to one primitive inflated by body radius."""

    if not isinstance(purpose, GeometryPurpose):
        raise TypeError("purpose must be a GeometryPurpose")
    point = _validated_point(point_xyz)
    _validate_obstacle_spec(obstacle)
    try:
        body_radius = float(body_radius_m)
    except (TypeError, ValueError) as exc:
        raise TypeError("body_radius_m must be numeric") from exc
    if not math.isfinite(body_radius) or body_radius < 0.0:
        raise ValueError("body_radius_m must be finite and non-negative")
    dx, dy, dz = point[0] - obstacle.x, point[1] - obstacle.y, point[2] - obstacle.z
    if obstacle.shape == "sphere":
        base = math.sqrt(dx * dx + dy * dy + dz * dz) - obstacle.radius
    elif obstacle.shape == "cylinder":
        radial = math.hypot(dx, dy) - obstacle.radius
        vertical = abs(dz) - obstacle.height / 2.0
        outside = math.hypot(max(radial, 0.0), max(vertical, 0.0))
        base = outside + min(max(radial, vertical), 0.0)
    else:
        cosine, sine = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        offsets = (
            abs(local_x) - obstacle.size_x / 2.0,
            abs(local_y) - obstacle.size_y / 2.0,
            abs(dz) - obstacle.size_z / 2.0,
        )
        outside = math.sqrt(sum(max(value, 0.0) ** 2 for value in offsets))
        base = outside + min(max(offsets), 0.0)
    return float(base - body_radius)


# ─────────────────────────────────────────────────────────────────
# 几何距离
# ─────────────────────────────────────────────────────────────────

def point_to_oriented_rect_distance(
    px: float, py: float,
    cx: float, cy: float,
    sx: float, sy: float,
    yaw: float,
) -> float:
    """Shortest XY distance from point to an oriented rectangle.
    Rectangle: center (cx,cy), full-extents (sx, sy), rotated by yaw."""
    dx = px - cx
    dy = py - cy
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)
    lx = dx * cos_y - dy * sin_y
    ly = dx * sin_y + dy * cos_y
    hx = sx / 2.0
    hy = sy / 2.0
    cx_dist = max(abs(lx) - hx, 0.0)
    cy_dist = max(abs(ly) - hy, 0.0)
    return math.hypot(cx_dist, cy_dist)


def compute_obstacle_clearance(
    drone_x: float, drone_y: float, drone_z: float,
    obstacles: List[ObstacleSpec],
    check_3d: bool = True,
) -> Tuple[float, Optional[str], bool]:
    """Minimum clearance from drone to all obstacles (world frame).

    Returns:
        (min_clearance, closest_obstacle_name, closest_is_flyover)
        min_clearance = surface-to-surface distance (negative if penetrating)
        closest_is_flyover = True: 与最近障碍的垂直区间严格分离 (上绕或下穿)
    """
    min_clear = float("inf")
    closest_name = None
    closest_flyover = False

    for obs in obstacles:
        is_flyover = False
        if obs.shape == "sphere":
            clear_3d = signed_clearance(
                (drone_x, drone_y, drone_z),
                obs,
                purpose=GeometryPurpose.PHYSICAL_COLLISION,
                body_radius_m=0.0,
            )
            if clear_3d < min_clear:
                min_clear, closest_name, closest_flyover = clear_3d, obs.name, False

        elif obs.shape == "cylinder":
            d_xy = math.hypot(drone_x - obs.x, drone_y - obs.y)
            clear_xy = d_xy - obs.radius

            if check_3d:
                obs_top = obs.z + obs.height / 2.0
                obs_bot = obs.z - obs.height / 2.0
                drone_top = drone_z + DRONE_RADIUS_Z
                drone_bot = drone_z - DRONE_RADIUS_Z

                # No safety-margin tolerance may exempt physical overlap/contact.
                if drone_bot > obs_top or drone_top < obs_bot:
                    is_flyover = True
                    dz = min(abs(drone_bot - obs_top), abs(drone_top - obs_bot))
                    clear_3d = math.hypot(max(clear_xy, 0.0), dz)
                elif clear_xy >= 0:
                    clear_3d = clear_xy
                else:
                    clear_3d = clear_xy  # negative → 穿透
            else:
                clear_3d = clear_xy

            if clear_3d < min_clear:
                min_clear, closest_name, closest_flyover = clear_3d, obs.name, is_flyover

        elif obs.shape == "box":
            d_surf = point_to_oriented_rect_distance(
                drone_x, drone_y, obs.x, obs.y, obs.size_x, obs.size_y, obs.yaw)

            if check_3d:
                obs_top = obs.z + obs.size_z / 2.0
                obs_bot = obs.z - obs.size_z / 2.0
                drone_top = drone_z + DRONE_RADIUS_Z
                drone_bot = drone_z - DRONE_RADIUS_Z

                if drone_bot > obs_top or drone_top < obs_bot:
                    is_flyover = True
                    dz = min(abs(drone_bot - obs_top), abs(drone_top - obs_bot))
                    clear_3d = math.hypot(max(d_surf, 0.0), dz)
                else:
                    clear_3d = d_surf  # d_surf>=0 分离; <0 不会发生(rect dist 非负), 穿透靠 3d
            else:
                clear_3d = d_surf

            if clear_3d < min_clear:
                min_clear, closest_name, closest_flyover = clear_3d, obs.name, is_flyover

        else:
            raise ValueError("unknown obstacle shape: {!r}".format(obs.shape))

    return min_clear, closest_name, closest_flyover


def _clip_axis_interval(t0, t1, start, delta, lower, upper):
    """Intersect [t0,t1] with lower <= start + t*delta <= upper."""
    if delta == 0.0:
        return (t0, t1) if lower <= start <= upper else None
    a, b = (lower - start) / delta, (upper - start) / delta
    lo, hi = max(t0, min(a, b)), min(t1, max(a, b))
    return (lo, hi) if lo <= hi else None


def _point_segment_distance_2d(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0.0 else min(1.0, max(
        0.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_sq))
    return math.hypot(a[0] + t * dx - point[0], a[1] + t * dy - point[1])


def _segment_rect_distance_2d(a, b, hx, hy):
    """Euclidean distance to a solid axis-aligned rectangle, not an AABB inflation.

    Intersecting segments have distance zero. Otherwise the minimum between
    the segment and any rectangle edge occurs at an endpoint of either one
    (also sufficient for parallel edges). Thus endpoints + four corners suffice.
    """
    interval = _clip_axis_interval(0.0, 1.0, a[0], b[0] - a[0], -hx, hx)
    if interval is not None:
        interval = _clip_axis_interval(*interval, a[1], b[1] - a[1], -hy, hy)
        if interval is not None:
            return 0.0
    best = min(math.hypot(max(abs(p[0]) - hx, 0.0), max(abs(p[1]) - hy, 0.0))
               for p in (a, b))
    for x in (-hx, hx):
        for y in (-hy, hy):
            best = min(best, _point_segment_distance_2d((x, y), a, b))
    return best


def segment_obstacle_collision(
    start, end, obstacles: List[ObstacleSpec], collision_thresh: float,
) -> Optional[Tuple[float, str]]:
    """Find an obstacle hit anywhere on p(t)=start+t*(end-start), t in [0,1].

    Uses the existing upright half-height and horizontal clearance < threshold
    rule. Strict vertical separation still permits overflight/underpasses;
    this adds neither a safety margin nor a newly calibrated body radius.
    First restrict t to vertical overlap, then minimize XY clearance on THAT
    interval. Independent XY/Z minima would produce false diagonal collisions.
    Box yaw is respected; no sampling interval, dependency or path rewriting.

    Returns (minimum horizontal clearance on the overlap interval, name) for
    the first hit in obstacle-list order, or None. Not a time-of-impact solver.
    Ground is checked by the caller. Inputs are finite world-frame positions.
    """
    a = tuple(float(v) for v in start[:3])
    b = tuple(float(v) for v in end[:3])
    delta = tuple(b[i] - a[i] for i in range(3))
    for obs in obstacles:
        if obs.shape not in ("box", "cylinder"):
            continue  # Same supported shape set as point clearance.
        height = obs.height if obs.shape == "cylinder" else obs.size_z
        interval = _clip_axis_interval(
            0.0, 1.0, a[2], delta[2],
            obs.z - height / 2.0 - DRONE_RADIUS_Z,
            obs.z + height / 2.0 + DRONE_RADIUS_Z,
        )
        if interval is None:
            continue
        xy0, xy1 = [tuple(a[i] + t * delta[i] - (obs.x, obs.y)[i]
                          for i in range(2)) for t in interval]
        if obs.shape == "cylinder":
            clearance = _point_segment_distance_2d((0.0, 0.0), xy0, xy1) - obs.radius
        else:
            c, s = math.cos(obs.yaw), math.sin(obs.yaw)
            local0 = (c * xy0[0] + s * xy0[1], -s * xy0[0] + c * xy0[1])
            local1 = (c * xy1[0] + s * xy1[1], -s * xy1[0] + c * xy1[1])
            clearance = _segment_rect_distance_2d(local0, local1, obs.size_x / 2.0, obs.size_y / 2.0)
        if clearance < collision_thresh:
            return float(clearance), obs.name
    return None


# ─────────────────────────────────────────────────────────────────
# 加载
# ─────────────────────────────────────────────────────────────────

def load_obstacles(json_path: str) -> List[ObstacleSpec]:
    """Load obstacle specs from .obstacles.json."""
    with open(json_path) as f:
        data = json.load(f)
    return [ObstacleSpec.from_dict(d) for d in data.get("obstacles", [])]
