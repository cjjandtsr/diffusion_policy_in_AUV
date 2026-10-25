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

from diffusion_policy_3d.model.common.synth_pointcloud import (
    DepthCameraIntrinsics,
    SynthPointCloud,
)


class FirstH4RawFrameOrthogonalTests(unittest.TestCase):
    def _module(self):
        name = "scripts.probe_routea_first_h4_raw_frame_orthogonal"
        self.assertIsNotNone(importlib.util.find_spec(name))
        return importlib.import_module(name)

    def test_reports_exact_float32_difference_instead_of_hash_only(self):
        module = self._module()
        first = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        second = first.copy()
        second[0, 1] = np.nextafter(np.float32(2.0), np.float32(3.0))
        result = module.array_difference(first, second)
        self.assertEqual(result["different_elements"], 1)
        self.assertEqual(result["first_index"], [0, 1])
        self.assertEqual(result["max_abs"], 2.384185791015625e-07)

    def test_plain_pair_keeps_global_rng_and_uses_real_synth(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as directory:
            geometry = Path(directory) / "test.obstacles.json"
            geometry.write_text(json.dumps({"obstacles": [
                {"shape": "cylinder", "x": 2.0, "y": 0.0, "radius": 0.8, "height": 4.0}
            ]}), encoding="utf-8")
            synth = SynthPointCloud(
                str(geometry), device="cpu",
                cam_intrinsics=DepthCameraIntrinsics(width=16, height=12),
            )
            before = torch.get_rng_state().clone()
            result = module.sample_pair(
                synth, np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
                0.2, 27, "a" * 64, mode="plain",
            )
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(len(result["raw_frame_hashes"]), 2)
            self.assertTrue(result["within_process_exact"])
            self.assertEqual(result["difference"]["different_elements"], 0)
            self.assertEqual(result["valid_hit_rays"][0], result["valid_hit_rays"][1])
            self.assertEqual(result["total_rays"], [16 * 12, 16 * 12])

    def test_direct_script_entrypoint_imports_repo_package(self):
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            [sys.executable, str(root / "scripts" /
                                 "probe_routea_first_h4_raw_frame_orthogonal.py"), "--help"],
            cwd=str(root), capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
