from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from diffusion_policy_3d.rl.reinflow_fm.official_core import vendor_identity
from diffusion_policy_3d.rl.reinflow_fm.official_core.vendor_identity import (
    EXPECTED_REINFLOW_COMMIT,
    verify_vendor_snapshot,
)


def _make_fixture_vendor(root: Path, content: bytes = b"official fixture\n") -> bytes:
    root.mkdir()
    source = root / "model" / "flow" / "ft_ppo" / "ppoflow.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = f"{digest}  model/flow/ft_ppo/ppoflow.py\n".encode()
    (root / "SHA256SUMS").write_bytes(manifest)
    (root / "UPSTREAM_COMMIT").write_text(
        EXPECTED_REINFLOW_COMMIT + "\n", encoding="utf-8"
    )
    return manifest


class VendorIdentityTests(unittest.TestCase):
    def _fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        vendor = Path(temporary.name) / "reinflow_official_e722e15"
        manifest = _make_fixture_vendor(vendor)
        patches = (
            mock.patch.object(
                vendor_identity,
                "EXPECTED_VENDOR_MANIFEST_SHA256",
                hashlib.sha256(manifest).hexdigest(),
            ),
            mock.patch.object(vendor_identity, "EXPECTED_VENDOR_FILE_COUNT", 1),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return vendor

    def test_project_vendor_snapshot_matches_pinned_identity(self):
        repo_root = Path(__file__).resolve().parents[3]
        vendor = repo_root / "third_party" / "reinflow_official_e722e15"

        identity = verify_vendor_snapshot(vendor)

        self.assertEqual(identity.commit, EXPECTED_REINFLOW_COMMIT)
        self.assertGreater(identity.file_count, 0)
        self.assertEqual(len(identity.manifest_sha256), 64)

    def test_fixture_vendor_matches_manifest(self):
        vendor = self._fixture()

        identity = verify_vendor_snapshot(vendor)

        self.assertEqual(identity.commit, EXPECTED_REINFLOW_COMMIT)
        self.assertEqual(identity.file_count, 1)

    def test_vendor_rejects_wrong_commit(self):
        vendor = self._fixture()
        (vendor / "UPSTREAM_COMMIT").write_text("0" * 40 + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "commit mismatch"):
            verify_vendor_snapshot(vendor)

    def test_vendor_rejects_path_escape(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        vendor = Path(temporary.name) / "reinflow_official_e722e15"
        vendor.mkdir()
        manifest = f"{'0' * 64}  ../outside.py\n".encode()
        (vendor / "SHA256SUMS").write_bytes(manifest)
        (vendor / "UPSTREAM_COMMIT").write_text(
            EXPECTED_REINFLOW_COMMIT + "\n", encoding="utf-8"
        )
        with mock.patch.object(
            vendor_identity,
            "EXPECTED_VENDOR_MANIFEST_SHA256",
            hashlib.sha256(manifest).hexdigest(),
        ), mock.patch.object(vendor_identity, "EXPECTED_VENDOR_FILE_COUNT", 1):
            with self.assertRaisesRegex(ValueError, "escapes vendor root"):
                verify_vendor_snapshot(vendor)

    def test_vendor_rejects_modified_content(self):
        vendor = self._fixture()
        source = vendor / "model" / "flow" / "ft_ppo" / "ppoflow.py"
        source.write_bytes(b"modified\n")

        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            verify_vendor_snapshot(vendor)

    def test_vendor_rejects_unmanifested_source_file(self):
        vendor = self._fixture()
        (vendor / "unexpected.py").write_text("pass\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "unmanifested vendor file"):
            verify_vendor_snapshot(vendor)


if __name__ == "__main__":
    unittest.main()
