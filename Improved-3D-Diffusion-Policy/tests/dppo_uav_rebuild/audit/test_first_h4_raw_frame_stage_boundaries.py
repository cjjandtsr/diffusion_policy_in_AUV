import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from diffusion_policy_3d.model.common import synth_pointcloud
from diffusion_policy_3d.model.common.synth_pointcloud import (
    DepthCameraIntrinsics,
    SynthPointCloud,
)


class FirstH4StageBoundaryTests(unittest.TestCase):
    def _module(self):
        name = "scripts.probe_routea_first_h4_raw_frame_stage_boundaries"
        self.assertIsNotNone(importlib.util.find_spec(name), "stage probe missing")
        return importlib.import_module(name)

    def test_first_mismatch_reports_numeric_difference_with_infinite_misses(self):
        module = self._module()
        first = {
            "ray_dir_world": np.array([[1.0, 2.0]], dtype=np.float32),
            "t_ground": np.array([np.inf, 3.0], dtype=np.float32),
            "pre_noise": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "post_noise": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "raw_frame": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        }
        second = {
            "ray_dir_world": np.array([[1.0, 2.0]], dtype=np.float32),
            "t_ground": np.array([np.inf, 4.0], dtype=np.float32),
            "pre_noise": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "post_noise": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "raw_frame": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        }
        result = module.compare_stage_pair(first, second)
        self.assertEqual(result["first_mismatch_stage"], "t_ground")
        self.assertTrue(result["stages"]["ray_dir_world"]["bitwise_exact"])
        self.assertEqual(result["stages"]["t_ground"]["different_elements"], 1)
        self.assertEqual(result["stages"]["t_ground"]["first_index"], [1])
        self.assertEqual(result["stages"]["t_ground"]["max_abs_finite"], 1.0)

    def test_two_real_synth_calls_are_traced_without_persistent_patch(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as directory:
            geometry = Path(directory) / "test.obstacles.json"
            geometry.write_text(json.dumps({"obstacles": [
                {"shape": "cylinder", "x": 2.0, "y": 0.0,
                 "radius": 0.8, "height": 4.0},
                {"shape": "box", "x": 4.0, "y": 1.0,
                 "size_x": 1.0, "size_y": 1.0, "size_z": 2.0},
            ]}), encoding="utf-8")
            synth = SynthPointCloud(
                str(geometry), device="cpu",
                cam_intrinsics=DepthCameraIntrinsics(width=16, height=12),
            )
            original_cyl = synth_pointcloud.intersect_rays_cylinder
            original_box = synth_pointcloud.intersect_rays_box
            original_ground = synth_pointcloud.intersect_rays_ground
            original_noise = synth._apply_noise
            rng_before = torch.get_rng_state().clone()
            result = module.trace_two_samples(
                synth, np.array([0.0, 0.0, 2.0], dtype=np.float32),
                0.2, 27, "a" * 64,
            )
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            self.assertIs(synth_pointcloud.intersect_rays_cylinder, original_cyl)
            self.assertIs(synth_pointcloud.intersect_rays_box, original_box)
            self.assertIs(synth_pointcloud.intersect_rays_ground, original_ground)
            self.assertEqual(synth._apply_noise, original_noise)
            self.assertEqual(len(result["raw_frame_hashes"]), 2)
            self.assertEqual(result["first_mismatch_stage"], None)
            self.assertEqual(result["stage_order"], [
                "ray_dir_world", "t_cylinder", "t_box", "t_ground",
                "pre_noise", "post_noise", "raw_frame",
            ])
            self.assertTrue(all(stage["bitwise_exact"]
                                for stage in result["stages"].values()))

    def test_expected_terminal_pair_is_a_hard_guard(self):
        module = self._module()
        with self.assertRaisesRegex(ValueError, "expected raw-frame pair"):
            module.require_reproduced_pair(["0" * 64, "1" * 64])
        module.require_reproduced_pair([
            "59d7a01ae51b74848d4df1b63126e47faaa184fbb146da304c0bd85058bae70a",
            "af3fc203486dc1a8b8516c3634719b7d963ec6deb37c7be3477ae4cc05da16a6",
        ])

    def test_direct_script_entrypoint_imports_repo_package(self):
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            [sys.executable, str(root / "scripts" /
                             "probe_routea_first_h4_raw_frame_stage_boundaries.py"), "--help"],
            cwd=str(root), capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
