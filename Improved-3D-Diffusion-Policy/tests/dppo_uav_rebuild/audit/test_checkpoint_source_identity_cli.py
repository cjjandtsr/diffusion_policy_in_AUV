import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[3] / "scripts" / "audit_checkpoint_source_identity.py"


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _make_repo(tmp_path):
    root = tmp_path / "repo"
    relative_paths = [
        "diffusion_policy_3d/rl/dppo_uav_rebuild/core.py",
        "diffusion_policy_3d/rl/dppo_uav_rebuild/config.yaml",
        "third_party/dppo_official_cc7234a/uav_idp3/runtime.py",
        "third_party/dppo_official_cc7234a/uav_idp3/change.patch",
        "diffusion_policy_3d/env/collision_geom.py",
        "diffusion_policy_3d/env/kinematic_uav_env.py",
        "diffusion_policy_3d/env/traj_utils.py",
        "diffusion_policy_3d/model/common/synth_pointcloud.py",
    ]
    entries = []
    for index, relative in enumerate(relative_paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = ("payload-%d\n" % index).encode()
        path.write_bytes(payload)
        entries.append((relative, hashlib.sha256(payload).hexdigest()))
    return root, _canonical_hash(sorted(entries))


def _run(root, header):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(root),
            "--checkpoint-header",
            str(header),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


class CheckpointSourceIdentityCliTest(unittest.TestCase):
    def test_inventory_matches_checkpoint_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            root, digest = _make_repo(tmp_path)
            header = tmp_path / "header.json"
            header.write_text(
                json.dumps({"runtime_identity": {"source_set_hash": digest}}),
                encoding="utf-8",
            )
            result = _run(root, header)
            report = json.loads(result.stdout)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(report["status"], "MATCH")
            self.assertEqual(report["source_file_count"], 8)

    def test_inventory_blocks_drift_and_lists_changed_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            root, digest = _make_repo(tmp_path)
            header = tmp_path / "header.json"
            header.write_text(json.dumps({"source_set_hash": digest}), encoding="utf-8")
            drift = root / "diffusion_policy_3d/rl/dppo_uav_rebuild/new_audit.py"
            drift.write_text("drift\n", encoding="utf-8")

            result = _run(root, header)
            report = json.loads(result.stdout)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(report["status"], "MISMATCH")
            self.assertTrue(
                any(row["path"].endswith("new_audit.py") for row in report["source_files"])
            )

    def test_inventory_rejects_header_without_valid_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            root, _ = _make_repo(tmp_path)
            header = tmp_path / "header.json"
            header.write_text("{}", encoding="utf-8")
            result = _run(root, header)
            report = json.loads(result.stdout)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(report["status"], "ERROR")
            self.assertEqual(report["error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
