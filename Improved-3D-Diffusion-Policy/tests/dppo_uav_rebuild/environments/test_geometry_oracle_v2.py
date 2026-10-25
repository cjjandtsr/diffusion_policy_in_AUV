"""Primitive signed-clearance contract shared by Route A consumers."""

from __future__ import annotations

import numpy as np
import pytest

from diffusion_policy_3d.env import collision_geom as geom


@pytest.mark.parametrize(
    "obstacle,point,body_radius,expected",
    [
        (
            geom.ObstacleSpec("sphere", "sphere", 0.0, 0.0, 0.0, radius=1.0),
            [2.0, 0.0, 0.0],
            0.5,
            0.5,
        ),
        (
            geom.ObstacleSpec("cylinder", "cylinder", 0.0, 0.0, 1.0, radius=1.0, height=2.0),
            [2.0, 0.0, 1.0],
            0.5,
            0.5,
        ),
        (
            geom.ObstacleSpec("cylinder-top", "cylinder", 0.0, 0.0, 1.0, radius=1.0, height=2.0),
            [0.0, 0.0, 2.5],
            0.5,
            0.0,
        ),
        (
            geom.ObstacleSpec("box", "box", 0.0, 0.0, 1.0, size_x=2.0, size_y=2.0, size_z=2.0),
            [2.0, 0.0, 1.0],
            0.25,
            0.75,
        ),
    ],
)
def test_signed_clearance_has_positive_zero_negative_surface_semantics(
    obstacle, point, body_radius, expected
):
    """Catches center/extent or radius semantics drifting by primitive."""

    actual = geom.signed_clearance(
        np.asarray(point),
        obstacle,
        purpose=geom.GeometryPurpose.AUDIT_REFERENCE,
        body_radius_m=body_radius,
    )
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "obstacle",
    [
        geom.ObstacleSpec("unknown", "mesh", 0.0, 0.0, 0.0),
        geom.ObstacleSpec("bad-radius", "sphere", 0.0, 0.0, 0.0, radius=-1.0),
        geom.ObstacleSpec("bad-height", "cylinder", 0.0, 0.0, 0.0, radius=1.0, height=0.0),
        geom.ObstacleSpec("bad-box", "box", 0.0, 0.0, 0.0, size_x=1.0, size_y=np.nan, size_z=1.0),
    ],
)
def test_signed_clearance_rejects_invalid_primitives(obstacle):
    """Catches fail-open handling of unknown or malformed world geometry."""

    with pytest.raises(ValueError):
        geom.signed_clearance(
            np.zeros(3),
            obstacle,
            purpose=geom.GeometryPurpose.ADMISSION_ENDPOINT,
            body_radius_m=0.0,
        )


def test_signed_clearance_requires_explicit_purpose():
    """Catches silently reusing one margin contract for every consumer."""

    obstacle = geom.ObstacleSpec("s", "sphere", 0.0, 0.0, 0.0, radius=1.0)
    with pytest.raises(TypeError):
        geom.signed_clearance(np.zeros(3), obstacle, body_radius_m=0.0)
