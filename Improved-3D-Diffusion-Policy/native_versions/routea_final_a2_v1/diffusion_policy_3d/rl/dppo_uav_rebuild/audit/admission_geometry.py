"""Read-only occupancy and A* admission auditing.

This module deliberately preserves the production grid contract and only
dispatches obstacle dimensions by shape.  It never replaces an environment's
cached occupancy grid.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from diffusion_policy_3d.env.collision_geom import (
    GeometryPurpose,
    ObstacleSpec,
    signed_clearance,
)


MAX_GRID_CELLS = 50_000_000


@dataclass(frozen=True)
class GridSpec:
    x_min: float
    y_min: float
    z_min: float
    resolution: float
    nx: int
    ny: int
    nz: int
    xy_margin: float = 0.15

    def validate(self) -> "GridSpec":
        numeric = (self.x_min, self.y_min, self.z_min, self.resolution, self.xy_margin)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("grid values must be finite")
        if self.resolution <= 0.0 or self.xy_margin < 0.0:
            raise ValueError("resolution must be positive and margin non-negative")
        if any(isinstance(value, bool) or int(value) != value or int(value) < 1
               for value in (self.nx, self.ny, self.nz)):
            raise ValueError("grid dimensions must be positive integers")
        cells = int(self.nx) * int(self.ny) * int(self.nz)
        if cells > MAX_GRID_CELLS:
            raise ValueError("grid exceeds 50,000,000 cells")
        return self

    @property
    def shape(self):
        return int(self.ny), int(self.nx), int(self.nz)


def grid_spec_from_environment(environment: Any, *, xy_margin: float = 0.15) -> GridSpec:
    """Capture the metadata created by the production occupancy builder."""
    if not hasattr(environment, "_occ_grid_meta"):
        environment._get_occupancy_grid()
    meta = environment._occ_grid_meta
    return GridSpec(
        x_min=float(meta["x_min"]),
        y_min=float(meta["y_min"]),
        z_min=float(meta["z_min"]),
        resolution=float(meta["res"]),
        nx=int(meta["nx"]),
        ny=int(meta["ny"]),
        nz=int(meta["nz"]),
        xy_margin=float(xy_margin),
    ).validate()


def _validate_obstacle(obstacle: ObstacleSpec) -> None:
    common = (obstacle.x, obstacle.y, obstacle.z, obstacle.yaw)
    if not all(math.isfinite(float(value)) for value in common):
        raise ValueError("obstacle position and yaw must be finite")
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


def _z_slice(obstacle: ObstacleSpec, spec: GridSpec):
    if obstacle.shape == "sphere":
        half_height = obstacle.radius + spec.xy_margin
    elif obstacle.shape == "cylinder":
        half_height = obstacle.height / 2.0
    else:
        half_height = obstacle.size_z / 2.0
    bottom = obstacle.z - half_height
    top = obstacle.z + half_height
    iz0 = max(0, int((bottom - spec.z_min) / spec.resolution))
    iz1 = min(spec.nz - 1, int((top - spec.z_min) / spec.resolution))
    if iz1 < 0 or iz0 >= spec.nz or iz1 < iz0:
        return None
    return iz0, iz1


def obstacle_blocked_mask(obstacle: ObstacleSpec, spec: GridSpec) -> np.ndarray:
    """Return one obstacle's blocked cells using production grid conventions."""
    spec.validate()
    _validate_obstacle(obstacle)
    mask = np.zeros(spec.shape, dtype=bool)
    z_bounds = _z_slice(obstacle, spec)
    if z_bounds is None:
        return mask
    iz0, iz1 = z_bounds
    for iy in range(spec.ny):
        wy = spec.y_min + (iy + 0.5) * spec.resolution
        for ix in range(spec.nx):
            wx = spec.x_min + (ix + 0.5) * spec.resolution
            if obstacle.shape != "sphere":
                clearance = signed_clearance(
                    (wx, wy, obstacle.z),
                    obstacle,
                    purpose=GeometryPurpose.AUDIT_REFERENCE,
                    body_radius_m=0.0,
                )
                if clearance < spec.xy_margin:
                    mask[iy, ix, iz0:iz1 + 1] = True
                continue
            for iz in range(iz0, iz1 + 1):
                wz = spec.z_min + (iz + 0.5) * spec.resolution
                clearance = signed_clearance(
                    (wx, wy, wz),
                    obstacle,
                    purpose=GeometryPurpose.AUDIT_REFERENCE,
                    body_radius_m=0.0,
                )
                if clearance < spec.xy_margin:
                    mask[iy, ix, iz] = True
    return mask


def build_shape_reference_grid(
    obstacles: Sequence[ObstacleSpec],
    spec: GridSpec,
) -> np.ndarray:
    spec.validate()
    if len(obstacles) > 10_000:
        raise ValueError("world exceeds 10,000 obstacles")
    grid = np.ones(spec.shape, dtype=bool)
    for obstacle in obstacles:
        grid[obstacle_blocked_mask(obstacle, spec)] = False
    return grid


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def compare_occupancy_grids(
    production: np.ndarray,
    reference: np.ndarray,
    *,
    obstacles: Sequence[ObstacleSpec],
    spec: GridSpec,
) -> Dict[str, Any]:
    spec.validate()
    if production.shape != spec.shape or reference.shape != spec.shape:
        raise ValueError("occupancy grid shape does not match GridSpec")
    if production.dtype != np.bool_ or reference.dtype != np.bool_:
        raise ValueError("occupancy grids must use bool dtype")
    production_hash = _array_sha256(production)
    reference_hash = _array_sha256(reference)
    missed = production & ~reference
    extra = ~production & reference
    by_obstacle: Dict[str, Dict[str, int]] = {}
    by_shape: Dict[str, Dict[str, int]] = {}
    mismatch_names = []
    attribution_count = np.zeros(spec.shape, dtype=np.uint8)
    for obstacle in obstacles:
        if obstacle.name in by_obstacle:
            raise ValueError("duplicate obstacle name: {}".format(obstacle.name))
        obstacle_mask = obstacle_blocked_mask(obstacle, spec)
        attributed = missed & obstacle_mask
        missed_count = int(np.count_nonzero(attributed))
        attribution_count[attributed] = np.minimum(
            attribution_count[attributed] + 1, 2
        )
        by_obstacle[obstacle.name] = {"missed_cells": missed_count}
        shape_record = by_shape.setdefault(
            obstacle.shape, {"obstacle_count": 0, "missed_cells": 0}
        )
        shape_record["obstacle_count"] += 1
        shape_record["missed_cells"] += missed_count
        if missed_count:
            mismatch_names.append(obstacle.name)
    if production_hash != _array_sha256(production) or reference_hash != _array_sha256(reference):
        raise RuntimeError("occupancy comparison mutated an input")
    missed_count = int(np.count_nonzero(missed))
    extra_count = int(np.count_nonzero(extra))
    return {
        "status": "pass" if missed_count == 0 and extra_count == 0 else "mismatch",
        "shape": list(spec.shape),
        "production_sha256": production_hash,
        "reference_sha256": reference_hash,
        "production_blocked_cells": int(np.count_nonzero(~production)),
        "reference_blocked_cells": int(np.count_nonzero(~reference)),
        "missed_blocked_cells": missed_count,
        "extra_blocked_cells": extra_count,
        "mismatch_obstacle_names": sorted(mismatch_names),
        "attribution_semantics": "inclusive_per_obstacle",
        "multi_attributed_missed_cells": int(np.count_nonzero(attribution_count > 1)),
        "by_obstacle": dict(sorted(by_obstacle.items())),
        "by_shape": dict(sorted(by_shape.items())),
    }


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _astar_result(raw: Sequence[Any], grid_meta: Mapping[str, Any]) -> Mapping[str, Any]:
    reachable, path_length, bottleneck, z_min, z_max, path = raw
    if reachable and path:
        z_values = [
            float(grid_meta["z_min"])
            + (int(node[2]) + 0.5) * float(grid_meta["res"])
            for node in path
        ]
        derived_z_min = min(z_values)
        derived_z_max = max(z_values)
        z_bounds_match = (
            math.isclose(float(z_min), derived_z_min, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(float(z_max), derived_z_max, rel_tol=0.0, abs_tol=1e-9)
        )
    else:
        derived_z_min = None
        derived_z_max = None
        z_bounds_match = None
    return {
        "reachable": bool(reachable),
        "path_length_m": float(path_length),
        "bottleneck_clearance_m": float(bottleneck),
        "path_z_min_m": float(z_min),
        "path_z_max_m": float(z_max),
        "derived_path_z_min_m": derived_z_min,
        "derived_path_z_max_m": derived_z_max,
        "reported_path_z_bounds_match_derived": z_bounds_match,
        "bottleneck_clearance_semantics": "production_reported_unverified",
        "path_nodes": len(path),
        "path_sha256": _canonical_hash([list(node) for node in path]),
    }


def audit_astar_pair(
    environment: Any,
    production: np.ndarray,
    reference: np.ndarray,
    start: Sequence[float],
    goal: Sequence[float],
) -> Dict[str, Any]:
    if production.shape != reference.shape or production.dtype != np.bool_ or reference.dtype != np.bool_:
        raise ValueError("A* grids must be same-shape bool arrays")
    production_hash = _array_sha256(production)
    reference_hash = _array_sha256(reference)
    original_meta = copy.deepcopy(environment._occ_grid_meta)
    try:
        environment._occ_grid_meta = copy.deepcopy(original_meta)
        prod = _astar_result(environment._astar_3d(
            production, *[float(v) for v in start], *[float(v) for v in goal]
        ), original_meta)
        environment._occ_grid_meta = copy.deepcopy(original_meta)
        ref = _astar_result(environment._astar_3d(
            reference, *[float(v) for v in start], *[float(v) for v in goal]
        ), original_meta)
    finally:
        environment._occ_grid_meta = original_meta
    if production_hash != _array_sha256(production) or reference_hash != _array_sha256(reference):
        raise RuntimeError("A* shadow mutated an occupancy grid")
    resolution = float(original_meta["res"])
    cells = int(np.prod(production.shape, dtype=np.int64))
    return {
        "production": prod,
        "reference": ref,
        "classification_changed": prod["reachable"] != ref["reachable"],
        "astar_contract_drift": {
            "heuristic_units": "grid_index_while_g_cost_is_meters",
            "horizontal_move_cost_m": resolution,
            "vertical_move_cost_m": math.hypot(resolution, resolution),
            "expected_vertical_move_cost_m": resolution,
            "max_iterations": cells // 5,
        },
    }


__all__ = (
    "GridSpec",
    "MAX_GRID_CELLS",
    "audit_astar_pair",
    "build_shape_reference_grid",
    "compare_occupancy_grids",
    "grid_spec_from_environment",
    "obstacle_blocked_mask",
)
