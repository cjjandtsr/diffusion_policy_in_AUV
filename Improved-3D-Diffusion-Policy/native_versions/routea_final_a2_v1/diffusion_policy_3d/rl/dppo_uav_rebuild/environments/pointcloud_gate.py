"""Observation-quality gate using explicit synth metadata, never point heuristics."""

from dataclasses import dataclass

import numpy as np

from .observation_provenance import PointCloudRejected, require_pointcloud


@dataclass(frozen=True)
class PointCloudGateResult:
    accepted: bool
    no_ray_hit_fraction: float
    reason: str


def validate_pointcloud(point_cloud, *, no_ray_hit_fraction: float, max_fraction: float):
    array = np.asarray(point_cloud)
    if array.shape[-2:] != (4096, 3):
        raise ValueError("point cloud must end with (4096,3)")
    if not np.isfinite(array).all():
        return PointCloudGateResult(False, float(no_ray_hit_fraction), "non_finite")
    fraction = float(no_ray_hit_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("no_ray_hit_fraction must be in [0,1]")
    if fraction > float(max_fraction):
        return PointCloudGateResult(False, fraction, "no_ray_hit_exceeds_limit")
    return PointCloudGateResult(True, fraction, "accepted")


def validate_pointcloud_metadata(point_cloud, metadata, *, max_fraction: float):
    """Strict production gate backed by same-call generator metadata."""

    try:
        require_pointcloud(
            point_cloud, metadata, max_no_ray_hit_fraction=max_fraction
        )
    except PointCloudRejected as exc:
        return PointCloudGateResult(
            False, float(metadata.no_ray_hit_fraction), str(exc)
        )
    return PointCloudGateResult(
        True, float(metadata.no_ray_hit_fraction), "accepted"
    )
