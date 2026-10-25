import hashlib
import math
import unittest

import numpy as np

from diffusion_policy_3d.env.collision_geom import ObstacleSpec
from diffusion_policy_3d.rl.dppo_uav_rebuild.audit.admission_geometry import (
    GridSpec,
    audit_astar_pair,
    build_shape_reference_grid,
    compare_occupancy_grids,
)


def _hash_array(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


class _FakeAStarEnvironment:
    def __init__(self):
        self._occ_grid_meta = {
            "sentinel": "unchanged", "res": 0.5, "z_min": 0.0,
        }

    def _astar_3d(self, grid, sx, sy, sz, gx, gy, gz):
        reachable = bool(np.all(grid))
        path = [(0, 0, 0), (1, 0, 0)] if reachable else []
        return reachable, 0.5 if reachable else 0.0, 1.0, sz, gz, path


class AdmissionGeometryTests(unittest.TestCase):
    def test_reference_grid_marks_axis_aligned_box(self):
        spec = GridSpec(-1.0, -1.0, 0.0, 0.5, 5, 5, 4, 0.15)
        box = ObstacleSpec(
            "box", "box", 0.25, 0.25, 0.75,
            size_x=0.5, size_y=0.5, size_z=1.0,
        )
        grid = build_shape_reference_grid([box], spec)
        self.assertEqual(grid.shape, (5, 5, 4))
        self.assertEqual(grid.dtype, np.bool_)
        self.assertGreater(int(np.count_nonzero(~grid)), 0)
        self.assertFalse(bool(grid[2, 2, 1]))

    def test_exact_xy_margin_is_free_but_inside_margin_is_blocked(self):
        spec = GridSpec(0.0, 0.0, 0.0, 0.1, 10, 10, 2, 0.15)
        box = ObstacleSpec(
            "box", "box", 0.5, 0.5, 0.05,
            size_x=0.2, size_y=0.2, size_z=0.1,
        )
        grid = build_shape_reference_grid([box], spec)
        self.assertTrue(bool(grid[5, 7, 0]))
        self.assertFalse(bool(grid[5, 6, 0]))

    def test_rotated_non_square_box_changes_mask(self):
        spec = GridSpec(-2.0, -2.0, 0.0, 0.2, 20, 20, 5, 0.15)
        base = dict(
            name="box", shape="box", x=0.0, y=0.0, z=0.5,
            size_x=2.0, size_y=0.4, size_z=1.0,
        )
        unrotated = build_shape_reference_grid([ObstacleSpec(**base, yaw=0.0)], spec)
        rotated = build_shape_reference_grid(
            [ObstacleSpec(**base, yaw=math.pi / 4.0)], spec
        )
        self.assertFalse(np.array_equal(unrotated, rotated))
        ratio = np.count_nonzero(~rotated) / np.count_nonzero(~unrotated)
        self.assertGreater(ratio, 0.65)
        self.assertLess(ratio, 1.35)

    def test_cylinder_contract_uses_strict_radial_boundary(self):
        spec = GridSpec(0.0, 0.0, 0.0, 0.1, 10, 10, 2, 0.15)
        cylinder = ObstacleSpec(
            "cyl", "cylinder", 0.5, 0.5, 0.05,
            radius=0.1, height=0.1,
        )
        grid = build_shape_reference_grid([cylinder], spec)
        self.assertTrue(bool(grid[5, 7, 0]))
        self.assertFalse(bool(grid[5, 6, 0]))

    def test_compare_reports_missed_cells_and_obstacle_attribution(self):
        spec = GridSpec(-1.0, -1.0, 0.0, 0.5, 5, 5, 4, 0.15)
        box = ObstacleSpec(
            "gate", "box", 0.25, 0.25, 0.75,
            size_x=0.5, size_y=0.5, size_z=1.0,
        )
        reference = build_shape_reference_grid([box], spec)
        production = np.ones_like(reference)
        result = compare_occupancy_grids(
            production, reference, obstacles=[box], spec=spec
        )
        self.assertEqual(result["status"], "mismatch")
        self.assertGreater(result["missed_blocked_cells"], 0)
        self.assertEqual(result["extra_blocked_cells"], 0)
        self.assertEqual(result["mismatch_obstacle_names"], ["gate"])
        self.assertGreater(result["by_obstacle"]["gate"]["missed_cells"], 0)
        self.assertEqual(
            result["by_shape"]["box"]["missed_cells"],
            result["missed_blocked_cells"],
        )

    def test_overlapping_obstacle_attribution_is_explicitly_inclusive(self):
        spec = GridSpec(-1.0, -1.0, 0.0, 0.5, 5, 5, 4, 0.15)
        obstacles = [
            ObstacleSpec(
                name, "box", 0.25, 0.25, 0.75,
                size_x=0.5, size_y=0.5, size_z=1.0,
            )
            for name in ("left", "right")
        ]
        reference = build_shape_reference_grid(obstacles, spec)
        production = np.ones_like(reference)
        result = compare_occupancy_grids(
            production, reference, obstacles=obstacles, spec=spec
        )
        self.assertEqual(result["attribution_semantics"], "inclusive_per_obstacle")
        self.assertEqual(
            result["multi_attributed_missed_cells"],
            result["missed_blocked_cells"],
        )

    def test_compare_does_not_mutate_inputs(self):
        spec = GridSpec(0.0, 0.0, 0.0, 0.5, 2, 2, 2, 0.15)
        production = np.ones((2, 2, 2), dtype=bool)
        reference = production.copy()
        before = (_hash_array(production), _hash_array(reference))
        compare_occupancy_grids(production, reference, obstacles=[], spec=spec)
        self.assertEqual(before, (_hash_array(production), _hash_array(reference)))

    def test_astar_shadow_preserves_inputs_and_reports_contract_drift(self):
        env = _FakeAStarEnvironment()
        production = np.ones((2, 2, 2), dtype=bool)
        reference = production.copy()
        reference[0, 0, 0] = False
        before = (_hash_array(production), _hash_array(reference), dict(env._occ_grid_meta))
        result = audit_astar_pair(
            env, production, reference,
            start=(0.0, 0.0, 0.0), goal=(1.0, 0.0, 0.0),
        )
        self.assertTrue(result["classification_changed"])
        self.assertIn("vertical_move_cost_m", result["astar_contract_drift"])
        self.assertEqual(
            before,
            (_hash_array(production), _hash_array(reference), env._occ_grid_meta),
        )

    def test_astar_shadow_distinguishes_reported_z_fields_from_derived_path_bounds(self):
        env = _FakeAStarEnvironment()
        production = np.ones((2, 2, 2), dtype=bool)
        result = audit_astar_pair(
            env, production, production.copy(),
            start=(0.0, 0.0, 0.0), goal=(1.0, 0.0, 0.0),
        )
        branch = result["production"]
        self.assertIn("derived_path_z_min_m", branch)
        self.assertEqual(branch["derived_path_z_min_m"], 0.25)
        self.assertEqual(branch["derived_path_z_max_m"], 0.25)
        self.assertFalse(branch["reported_path_z_bounds_match_derived"])
        self.assertEqual(
            branch["bottleneck_clearance_semantics"],
            "production_reported_unverified",
        )

    def test_nonfinite_or_oversized_grid_is_rejected_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            GridSpec(0.0, 0.0, 0.0, math.nan, 2, 2, 2, 0.15).validate()
        with self.assertRaisesRegex(ValueError, "50,000,000"):
            GridSpec(0.0, 0.0, 0.0, 1.0, 1000, 1000, 51, 0.15).validate()


if __name__ == "__main__":
    unittest.main()
