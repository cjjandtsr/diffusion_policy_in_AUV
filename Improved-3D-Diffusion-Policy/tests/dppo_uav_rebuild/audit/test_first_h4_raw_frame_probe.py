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


class FirstH4RawFrameProbeTests(unittest.TestCase):
    def _probe(self):
        name = "scripts.probe_routea_first_h4_raw_frame"
        self.assertIsNotNone(importlib.util.find_spec(name))
        return importlib.import_module(name)

    def test_reset_identity_reconstructs_recorded_episode_seed(self):
        probe = self._probe()
        identity, seed = probe.derive_episode_seed(
            "env_012_obs22",
            [0.6072418093681335, 10.23682975769043, 3.905195951461792],
            [11.603382110595703, 12.72788143157959, 1.86884343624115],
            3466850540,
        )
        self.assertEqual(
            identity,
            "92f21df5d51e6523a348dbf5ccb90f41332acb9b487c3b050b7bbc3097756374",
        )
        self.assertEqual(seed, 3853349010)

    def test_direct_script_entrypoint_imports_repo_package(self):
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "probe_routea_first_h4_raw_frame.py"),
             "--help"],
            cwd=str(root), capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_raw_frame_replay_uses_episode_local_seed_after_global_rng_change(self):
        probe = self._probe()
        with tempfile.TemporaryDirectory() as directory:
            geometry = Path(directory) / "test.obstacles.json"
            geometry.write_text(json.dumps({
                "obstacles": [
                    {"shape": "cylinder", "x": 2.0, "y": 0.0, "radius": 0.8, "height": 4.0}
                ]
            }), encoding="utf-8")
            synth = SynthPointCloud(
                str(geometry), device="cpu",
                cam_intrinsics=DepthCameraIntrinsics(width=16, height=12),
            )
            position = np.asarray([0.0, 0.0, 2.0], dtype=np.float32)
            identity = "a" * 64
            first = probe.sample_initial_raw_frame(synth, position, 0.2, 27, identity)
            original_rng = torch.get_rng_state()
            try:
                torch.manual_seed(987654321)
                second = probe.sample_initial_raw_frame(synth, position, 0.2, 27, identity)
            finally:
                torch.set_rng_state(original_rng)
            self.assertEqual(first[0], second[0])
            self.assertGreater(first[1], 0)
            self.assertEqual(first[2], 16 * 12)


if __name__ == "__main__":
    unittest.main()
