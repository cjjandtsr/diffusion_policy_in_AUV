"""Immutable point-cloud and E5 observation identity contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import math
from pathlib import Path
import re
from typing import Mapping, Tuple

import numpy as np


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PointCloudRejected(RuntimeError):
    """Raised before actor, critic, or buffer use for an invalid raw frame."""


class ObservationIdentityMismatch(RuntimeError):
    """Raised when any recorded observation identity no longer matches bytes."""


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    return np.asarray(value)


def canonical_array_hash(value) -> str:
    """Hash dtype, shape, and C-order bytes; values alone are insufficient."""

    array = np.ascontiguousarray(_numpy(value))
    digest = hashlib.sha256()
    dtype = array.dtype.str.encode("ascii")
    shape = ",".join(str(int(dim)) for dim in array.shape).encode("ascii")
    digest.update(len(dtype).to_bytes(4, "big"))
    digest.update(dtype)
    digest.update(len(shape).to_bytes(4, "big"))
    digest.update(shape)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_mapping_hash(values: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        encoded = str(name).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(canonical_array_hash(values[name])))
    return digest.hexdigest()


@dataclass(frozen=True)
class PointCloudGenerationMeta:
    total_rays: int
    valid_hit_rays: int
    no_ray_hit_fraction: float
    used_far_plane_fallback: bool
    finite: bool
    episode_identity_hash: str
    decision_index: int
    raw_array_hash: str
    coordinate_frame: str = "body"

    def __post_init__(self):
        if (
            isinstance(self.total_rays, bool)
            or int(self.total_rays) != self.total_rays
            or int(self.total_rays) < 1
        ):
            raise ValueError("total_rays must be positive")
        if (
            isinstance(self.valid_hit_rays, bool)
            or int(self.valid_hit_rays) != self.valid_hit_rays
            or not 0 <= int(self.valid_hit_rays) <= int(self.total_rays)
        ):
            raise ValueError("valid_hit_rays must be in [0,total_rays]")
        expected = 1.0 - int(self.valid_hit_rays) / int(self.total_rays)
        if (
            not math.isfinite(float(self.no_ray_hit_fraction))
            or not 0.0 <= float(self.no_ray_hit_fraction) <= 1.0
            or not math.isclose(float(self.no_ray_hit_fraction), expected, abs_tol=1e-12)
        ):
            raise ValueError("no_ray_hit_fraction disagrees with ray counts")
        if not isinstance(self.used_far_plane_fallback, bool):
            raise TypeError("used_far_plane_fallback must be boolean")
        if not isinstance(self.finite, bool):
            raise TypeError("finite must be boolean")
        if not _SHA256.fullmatch(str(self.episode_identity_hash)):
            raise ValueError("episode_identity_hash must be lowercase sha256")
        if not _SHA256.fullmatch(str(self.raw_array_hash)):
            raise ValueError("raw_array_hash must be lowercase sha256")
        if (
            isinstance(self.decision_index, bool)
            or int(self.decision_index) != self.decision_index
            or int(self.decision_index) < 0
        ):
            raise ValueError("decision_index must be non-negative")
        if self.coordinate_frame != "body":
            raise ValueError("point cloud coordinate_frame must be body")


@dataclass(frozen=True)
class ObservationProvenance:
    raw_frame_hash: str
    raw_history_hash: str
    normalizer_identity_hash: str
    normalized_history_hash: str
    policy_input_hash: str
    episode_identity_hash: str
    decision_index: int

    def __post_init__(self):
        for name in (
            "raw_frame_hash", "raw_history_hash", "normalizer_identity_hash",
            "normalized_history_hash", "policy_input_hash",
            "episode_identity_hash",
        ):
            if not _SHA256.fullmatch(str(getattr(self, name))):
                raise ValueError("{} must be lowercase sha256".format(name))
        if (
            isinstance(self.decision_index, bool)
            or int(self.decision_index) != self.decision_index
            or int(self.decision_index) < 0
        ):
            raise ValueError("decision_index must be a non-negative integer")


@dataclass(frozen=True)
class EncodedObservation:
    condition: Mapping[str, object]
    provenance: ObservationProvenance
    normalized_history: Mapping[str, object] = field(repr=False, compare=False)


def normalizer_identity_hash(normalizer) -> str:
    """Hash normalizer class source plus named tensor parameters."""

    digest = hashlib.sha256()
    cls = type(normalizer)
    digest.update((cls.__module__ + "." + cls.__qualname__).encode("utf-8"))
    try:
        source = inspect.getsourcefile(cls)
        if source and Path(source).is_file():
            digest.update(hashlib.sha256(Path(source).read_bytes()).digest())
    except (OSError, TypeError):
        pass
    state = normalizer.state_dict() if hasattr(normalizer, "state_dict") else {}
    if not isinstance(state, Mapping):
        raise TypeError("normalizer state_dict must be a mapping")
    digest.update(bytes.fromhex(canonical_mapping_hash(state)))
    mode = getattr(normalizer, "mode", None)
    if mode is not None:
        digest.update(str(mode).encode("utf-8"))
    return digest.hexdigest()


def validate_metadata_history(
    point_cloud_history,
    metadata_history: Tuple[PointCloudGenerationMeta, ...],
    *,
    episode_identity_hash: str,
    decision_index: int,
    max_no_ray_hit_fraction: float,
    reject_all_no_ray_hit: bool = True,
) -> None:
    points = _numpy(point_cloud_history)
    if points.ndim == 4:
        if points.shape[0] != 1:
            raise ObservationIdentityMismatch("batch_size")
        points = points[0]
    if points.ndim != 4 - 1 or points.shape[0] != len(metadata_history):
        raise ObservationIdentityMismatch("metadata_history_length")
    indices = [int(meta.decision_index) for meta in metadata_history]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ObservationIdentityMismatch("history_order")
    if not indices or indices[-1] != int(decision_index):
        raise ObservationIdentityMismatch("decision_index")
    for frame, meta in zip(points, metadata_history):
        if meta.episode_identity_hash != episode_identity_hash:
            raise ObservationIdentityMismatch("cross_episode_history")
        if canonical_array_hash(frame) != meta.raw_array_hash:
            raise ObservationIdentityMismatch("raw_frame_hash")
        require_pointcloud(
            frame,
            meta,
            max_no_ray_hit_fraction=max_no_ray_hit_fraction,
            reject_all_no_ray_hit=reject_all_no_ray_hit,
        )


def require_pointcloud(
    point_cloud,
    meta: PointCloudGenerationMeta,
    *,
    max_no_ray_hit_fraction: float,
    max_distance_m: float = 50.0,
    reject_all_no_ray_hit: bool = True,
) -> None:
    points = _numpy(point_cloud)
    if points.shape != (4096, 3):
        raise PointCloudRejected("point_shape")
    if not np.isfinite(points).all() or not meta.finite:
        raise PointCloudRejected("non_finite")
    if canonical_array_hash(points) != meta.raw_array_hash:
        raise ObservationIdentityMismatch("raw_frame_hash")
    if not isinstance(reject_all_no_ray_hit, bool):
        raise TypeError("reject_all_no_ray_hit must be boolean")
    if meta.valid_hit_rays == 0:
        if reject_all_no_ray_hit:
            raise PointCloudRejected("all_no_ray")
        if not meta.used_far_plane_fallback:
            raise PointCloudRejected("unverified_all_no_ray")
        if np.any(np.ptp(points, axis=0) <= 1.0):
            raise PointCloudRejected("degenerate_no_return_fallback")
        if np.count_nonzero(points) == 0:
            raise PointCloudRejected("all_zero")
        if float(np.max(np.linalg.norm(points, axis=-1))) > float(max_distance_m):
            raise PointCloudRejected("distance_exceeds_limit")
        return
    if meta.used_far_plane_fallback:
        raise PointCloudRejected("far_plane_fallback")
    if np.count_nonzero(points) == 0:
        raise PointCloudRejected("all_zero")
    if float(np.max(np.linalg.norm(points, axis=-1))) > float(max_distance_m):
        raise PointCloudRejected("distance_exceeds_limit")
    if not 0.0 <= float(max_no_ray_hit_fraction) <= 1.0:
        raise ValueError("max_no_ray_hit_fraction must be in [0,1]")
    if meta.no_ray_hit_fraction > float(max_no_ray_hit_fraction):
        raise PointCloudRejected("partial_no_ray_exceeds_limit")


__all__ = (
    "EncodedObservation",
    "ObservationIdentityMismatch",
    "ObservationProvenance",
    "PointCloudGenerationMeta",
    "PointCloudRejected",
    "canonical_array_hash",
    "canonical_mapping_hash",
    "normalizer_identity_hash",
    "require_pointcloud",
    "validate_metadata_history",
)
