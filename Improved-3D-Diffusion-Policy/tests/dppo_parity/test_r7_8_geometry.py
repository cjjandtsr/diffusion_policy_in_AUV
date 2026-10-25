"""R7-8 Step 1/2 groundwork: policy-INDEPENDENT geometry opportunity
annotation.

Every bank entry carries a frozen CapabilityOpportunity block derived
only from nominal-path geometry / world topology / offline density —
never from any evaluated RL trajectory (F-04, provenance allowlists in
capability_metrics). This file tests the annotation function that turns
a world's obstacles + start/goal (+ optional real offline density) into
that block:

  - anticipatory: nominal path crossing the d_risk (0.9 m) shell;
  - dense: start/goal enclosed by obstacle triangles (trap) or world
    clutter class;
  - support: nominal path sampling inside / at the boundary of / outside
    the offline low-density region.

All rules are deterministic, geometry-only, and fail-closed on malformed
obstacle records.
"""

import pytest

from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    CapabilityOpportunity,
    D_RISK,
    OfflineDensity,
    annotate_geometry_opportunity,
    opportunity_from_dict,
    opportunity_to_dict,
)


def _cylinder(x: float, y: float, radius: float = 0.3,
              z: float = 2.0, height: float = 4.0) -> dict:
    return {"name": "obs", "shape": "cylinder", "x": x, "y": y, "z": z,
            "yaw": 0.0, "radius": radius, "height": height}


def _box(x: float, y: float, size_x: float = 6.0,
         size_y: float = 0.5, yaw: float = 0.0, z: float = 2.0) -> dict:
    return {"name": "wall", "shape": "box", "x": x, "y": y, "z": z,
            "yaw": yaw, "size_x": size_x, "size_y": size_y,
            "size_z": 4.0}


# ═══════════════════════════════════════════════════════════════════
# anticipatory group — nominal path vs the d_risk shell
# ═══════════════════════════════════════════════════════════════════

def test_annotation_crossing_when_cylinder_on_path():
    """A cylinder sitting on the start->goal line puts the nominal path
    inside the d_risk shell -> risk_shell_crossing, provenance
    nominal_path_geometry."""
    op = annotate_geometry_opportunity(
        [_cylinder(0.0, 0.0, radius=0.3)],
        start=[-5.0, 0.0, 2.0], goal=[5.0, 0.0, 2.0],
    )
    assert op.valid is True
    assert op.risk_shell_crossing is True
    assert op.anticipatory_provenance == "nominal_path_geometry"
    assert 0.0 < op.nominal_path_risk <= 1.0


def test_annotation_no_crossing_clear_path():
    """An empty world gives no anticipatory opportunity and no declared
    provenance (group undecleared)."""
    op = annotate_geometry_opportunity([], start=[-5.0, 0.0, 2.0],
                                       goal=[5.0, 0.0, 2.0])
    assert op.valid is True
    assert op.risk_shell_crossing is False
    assert op.nominal_path_risk == 0.0
    assert op.anticipatory_provenance == ""


def test_annotation_no_crossing_when_cylinder_far_off_path():
    """A cylinder 2 m away clears the 0.9 m shell -> no crossing."""
    op = annotate_geometry_opportunity(
        [_cylinder(0.0, 3.0, radius=0.3)],
        start=[-5.0, 0.0, 2.0], goal=[5.0, 0.0, 2.0],
    )
    assert op.risk_shell_crossing is False
    assert op.nominal_path_risk == 0.0


def test_annotation_box_wall_crossing():
    """A box wall across the path (point-to-rect distance) also counts
    as a shell crossing — both obstacle shapes must be handled."""
    op = annotate_geometry_opportunity(
        [_box(0.0, 0.0, size_x=6.0, size_y=0.5)],
        start=[-5.0, 0.0, 2.0], goal=[5.0, 0.0, 2.0],
    )
    assert op.risk_shell_crossing is True
    assert op.nominal_path_risk > 0.0


def test_annotation_clearance_uses_radius():
    """Clearance is surface distance (center distance minus radius): a
    cylinder 1.0 m off the line with r=0.3 gives clearance 0.7 < d_risk
    -> crossing; with r=0.1 clearance 0.9 == d_risk boundary, NOT
    crossing (< is strict)."""
    op = annotate_geometry_opportunity(
        [_cylinder(0.0, 1.0, radius=0.3)],
        start=[-5.0, 0.0, 2.0], goal=[5.0, 0.0, 2.0],
    )
    assert op.risk_shell_crossing is True
    op2 = annotate_geometry_opportunity(
        [_cylinder(0.0, 1.0, radius=0.1)],
        start=[-5.0, 0.0, 2.0], goal=[5.0, 0.0, 2.0],
    )
    assert op2.risk_shell_crossing is False


# ═══════════════════════════════════════════════════════════════════
# dense group — trap / open / cluttered world topology
# ═══════════════════════════════════════════════════════════════════

def test_annotation_trap_when_goal_inside_obstacle_triangle():
    """Three cylinders whose centers enclose the goal form a trap
    opportunity -> topology_class trap, count >= 1, provenance
    world_topology."""
    obs = [
        _cylinder(-4.0, 0.0),   # left
        _cylinder(4.0, 0.0),    # right
        _cylinder(0.0, 5.0),    # far (goal at (0, 2) sits inside)
    ]
    op = annotate_geometry_opportunity(obs, start=[0.0, -5.0, 2.0],
                                       goal=[0.0, 2.0, 2.0])
    assert op.topology_class == "trap"
    assert op.trap_opportunity_count >= 1
    assert op.dense_provenance == "world_topology"


def test_annotation_open_empty_world():
    """No obstacles -> topology_class open, no trap count."""
    op = annotate_geometry_opportunity([], start=[-5.0, 0.0, 2.0],
                                       goal=[5.0, 0.0, 2.0])
    assert op.topology_class == "open"
    assert op.trap_opportunity_count == 0
    assert op.dense_provenance == "world_topology"


def test_annotation_cluttered_many_obstacles_no_trap():
    """>= 5 obstacles without any enclosing triangle -> cluttered. The
    obstacles all sit on ONE side of the start->goal line, so no
    obstacle-center triangle can enclose either waypoint."""
    obs = [_cylinder(float(i * 1.4) - 4.0, 4.0 + float(i % 3) * 0.7)
           for i in range(8)]
    op = annotate_geometry_opportunity(obs, start=[-5.0, 0.0, 2.0],
                                       goal=[5.0, 0.0, 2.0])
    assert op.topology_class == "cluttered"
    assert op.trap_opportunity_count == 0


# ═══════════════════════════════════════════════════════════════════
# support group — nominal path vs the offline density region
# ═══════════════════════════════════════════════════════════════════

def _density(**cells):
    counts = {tuple(map(int, k.split(","))): v for k, v in cells.items()}
    return OfflineDensity(counts=counts)


# The support tests use start=(0,0,2) -> goal=(10,0,2): under the
# frozen reward_v13.compute_support_cell mapping (goal-unit frame,
# round indexing, z bin 0.5 m -> cell z 4) the nominal path covers
# cells x in [-10, 0], y in {0, 1}, z=4.
def _path_cells():
    return {(x, y, 4) for x in range(-10, 1) for y in (0, 1)}


def test_annotation_support_inside_low_density():
    """Part of the path's cells are low-density (uncovered cells count
    as 0 visits < tau_D) -> support_stratum inside, provenance
    offline_density."""
    density = _density(**{
        f"{x},0,4": 5 for x in range(-10, 1)
    } | {
        f"{x},1,4": 5 for x in range(-6, 1)
    })
    op = annotate_geometry_opportunity(
        [], start=[0.0, 0.0, 2.0], goal=[10.0, 0.0, 2.0],
        offline_density=density,
    )
    assert op.support_stratum == "inside"
    assert op.support_provenance == "offline_density"
    assert 0.0 < op.reference_density <= 1.0


def test_annotation_support_outside_when_density_high():
    """Every path-sampling cell AND its 8-neighbourhood is covered with
    counts >= tau_D -> outside (nothing low anywhere near the path)."""
    density = _density(**{
        f"{x},{y},4": 5
        for x in range(-11, 2) for y in range(-1, 3)
    })
    op = annotate_geometry_opportunity(
        [], start=[0.0, 0.0, 2.0], goal=[10.0, 0.0, 2.0],
        offline_density=density,
    )
    assert op.support_stratum == "outside"
    assert op.reference_density == 0.0


def test_annotation_support_boundary_when_neighbourhood_low():
    """The path itself stays high-density, but its neighbour cells are
    uncovered (0 visits = low) -> boundary, not inside."""
    density = _density(**{
        f"{x},{y},4": 5 for x, y, _ in _path_cells()
    })
    op = annotate_geometry_opportunity(
        [], start=[0.0, 0.0, 2.0], goal=[10.0, 0.0, 2.0],
        offline_density=density,
    )
    assert op.support_stratum == "boundary"
    assert op.support_provenance == "offline_density"
    assert op.reference_density == 0.0


def test_annotation_no_density_leaves_support_undecleared():
    """No offline density -> support group stays at the empty defaults
    (the support opportunity cannot be asserted without density)."""
    op = annotate_geometry_opportunity([], start=[-5.0, 0.0, 2.0],
                                       goal=[5.0, 0.0, 2.0])
    assert op.support_stratum == ""
    assert op.support_provenance == ""
    assert op.reference_density == 0.0


# ═══════════════════════════════════════════════════════════════════
# robustness — fail-closed on bad input, never NaN
# ═══════════════════════════════════════════════════════════════════

def test_annotation_malformed_obstacle_rejected():
    """A record without a shape (or without coordinates) hard-fails —
    a malformed world must never silently become 'no opportunity'."""
    with pytest.raises(ValueError):
        annotate_geometry_opportunity(
            [{"name": "ghost"}], start=[-5.0, 0.0, 2.0],
            goal=[5.0, 0.0, 2.0],
        )


def test_annotation_degenerate_path_falls_back():
    """start == goal must not produce NaN (the +x fallback frame) — the
    annotation stays valid and finite."""
    op = annotate_geometry_opportunity([], start=[0.0, 0.0, 2.0],
                                       goal=[0.0, 0.0, 2.0])
    assert op.valid is True
    assert op.nominal_path_risk == 0.0
    assert op.topology_class == "open"


def test_annotation_roundtrip_through_bank_serialization():
    """The annotated block must survive the frozen bank serialization
    (opportunity_to_dict -> opportunity_from_dict) losslessly."""
    op = annotate_geometry_opportunity(
        [_cylinder(0.0, 0.0)], start=[-5.0, 0.0, 2.0],
        goal=[5.0, 0.0, 2.0],
    )
    restored = opportunity_from_dict(opportunity_to_dict(op))
    assert isinstance(restored, CapabilityOpportunity)
    assert restored == op
