from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from diffusion_policy_3d.rl.reinflow_fm.build_system import (
    DEFAULT_WSL_REPO_ROOT,
    EXPECTED_REWARD_V13_SHA256,
    PreflightError,
    PreflightInputs,
    resolve_default_wsl_vendor,
    validate_preflight,
)


class BuildSystemPreflightTests(unittest.TestCase):
    def _paths(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        checkpoint = root / "fm.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        vendor = root / "reinflow_official_e722e15"
        vendor.mkdir()
        return checkpoint, vendor

    def _valid(self):
        checkpoint, vendor = self._paths()
        return PreflightInputs(
            checkpoint=checkpoint,
            checkpoint_sha256="a" * 64,
            vendor_root=vendor,
            reward_source_sha256=EXPECTED_REWARD_V13_SHA256,
        )

    def test_default_paths_are_the_authorized_wsl_execution_tree(self):
        self.assertEqual(
            DEFAULT_WSL_REPO_ROOT,
            Path("/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"),
        )
        self.assertEqual(
            resolve_default_wsl_vendor(),
            DEFAULT_WSL_REPO_ROOT / "third_party" / "reinflow_official_e722e15",
        )

    def test_valid_explicit_checkpoint_vendor_and_reward_identity_pass(self):
        validated = validate_preflight(self._valid())
        self.assertTrue(validated.checkpoint.is_file())
        self.assertTrue(validated.vendor_root.is_dir())

    def test_missing_checkpoint_vendor_hash_or_reward_identity_hard_fails(self):
        valid = self._valid()
        cases = (
            ("checkpoint", valid.checkpoint.parent / "missing.ckpt"),
            ("checkpoint_sha256", "bad"),
            ("vendor_root", valid.vendor_root.parent / "missing_vendor"),
            ("reward_source_sha256", "0" * 64),
        )
        for field, value in cases:
            kwargs = dict(valid.__dict__)
            kwargs[field] = value
            with self.subTest(field=field), self.assertRaises(PreflightError):
                validate_preflight(PreflightInputs(**kwargs))


if __name__ == "__main__":
    unittest.main()
