import hashlib
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scripts.audit_reward_v17_admission as audit_cli
from diffusion_policy_3d.env.collision_geom import ObstacleSpec
from diffusion_policy_3d.rl.dppo_uav_rebuild.audit.reset_bank import ResetAuditTask
from scripts.audit_reward_v17_admission import (
    _wilson_95,
    build_audit_files,
    determine_exit_code,
    load_json_strict,
    load_production_tasks,
    publish_pack,
    validate_obstacle_path,
)


class _AuditEnvironment:
    def __init__(self):
        self.obstacles = [ObstacleSpec(
            "box", "box", 0.25, 0.25, 0.75,
            size_x=0.5, size_y=0.5, size_z=1.0,
        )]
        self._occ_grid_meta = {
            "x_min": -1.0, "y_min": -1.0, "z_min": 0.0,
            "res": 0.5, "nx": 5, "ny": 5, "nz": 4,
        }
        self._production = np.ones((5, 5, 4), dtype=bool)
        self._drone_pos = np.array([-0.75, -0.75, 0.25], dtype=np.float32)
        self.goal_w = np.array([0.75, 0.75, 0.75], dtype=np.float32)
        self._drone_yaw = 0.0
        self._episode_fallback_type = None
        self._episode_fallback_reasons = []
        self._last_rejection_histogram = {"success": 1}
        self.episode_step = 0

    def reset(self, seed=None):
        self._drone_yaw = float(seed) / 100.0
        return {}, {}

    def _get_occupancy_grid(self):
        return self._production

    def _astar_3d(self, grid, sx, sy, sz, gx, gy, gz):
        reachable = bool(np.all(grid))
        path = [(0, 0, 0), (1, 0, 0)] if reachable else []
        return reachable, 0.5 if reachable else 0.0, 1.0, sz, gz, path


class _PathStretchGateEnvironment(_AuditEnvironment):
    def _astar_3d(self, grid, sx, sy, sz, gx, gy, gz):
        if np.all(grid):
            return True, 2.0, 1.0, sz, gz, [(0, 0, 0), (1, 0, 0)]
        return True, 3.0, 1.0, sz, gz, [(0, 0, 0), (1, 0, 0), (2, 0, 0)]


class AdmissionAuditCliTests(unittest.TestCase):
    def test_zero_count_wilson_interval_stays_inside_probability_bounds(self):
        interval = _wilson_95(0, 1344)
        self.assertGreaterEqual(interval["lower"], 0.0)
        self.assertLessEqual(interval["upper"], 1.0)

    def test_dense_path_stretch_gate_crossing_is_counted_as_classification_change(self):
        with tempfile.TemporaryDirectory() as directory:
            obstacle_path = Path(directory) / "dense.world.obstacles.json"
            obstacle_path.write_text(json.dumps({
                "obstacles": [{
                    "name": "box", "shape": "box",
                    "x": 0.25, "y": 0.25, "z": 0.75,
                    "size_x": 0.5, "size_y": 0.5, "size_z": 1.0,
                    "yaw": 0.0,
                }],
            }), encoding="utf-8")
            task = ResetAuditTask(
                task_id="dense-a",
                world_uid="dense-a",
                role="dense_recovery",
                admission_id="dense-a-admission",
                obstacle_json=str(obstacle_path),
            )
            files, _ = build_audit_files(
                [task], [10],
                lambda current_task, seed: _PathStretchGateEnvironment(),
                provenance={"test": True},
            )
        summary = json.loads(files["summary.json"])
        world = json.loads(files["world_geometry_records.jsonl"].splitlines()[0])
        self.assertIn("astar_path_stretch_classification_change_count", summary)
        self.assertEqual(
            summary["astar_path_stretch_classification_change_count"], 1,
        )
        self.assertIn("path_stretch_classification_changed", world["astar_shadow"][0])
        self.assertTrue(world["astar_shadow"][0]["path_stretch_classification_changed"])
    def test_hash_snapshot_rejects_replaced_config(self):
        checker = getattr(audit_cli, "assert_file_hashes_unchanged", None)
        self.assertIsNotNone(checker, "input hash checker is missing")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "config.json"
            source.write_text('{"version":1}', encoding="utf-8")
            snapshot = {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()}
            checker(snapshot)
            source.write_text('{"version":2}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "input hash changed"):
                checker(snapshot)

    def test_runtime_provenance_uses_nested_hash_bound_contract(self):
        builder = getattr(audit_cli, "build_runtime_provenance", None)
        self.assertIsNotNone(builder, "runtime provenance builder is missing")
        repo_root = Path(__file__).resolve().parents[3]
        sources = {
            "scripts/audit_reward_v17_admission.py":
                repo_root / "scripts/audit_reward_v17_admission.py",
            "diffusion_policy_3d/env/kinematic_uav_env.py":
                repo_root / "diffusion_policy_3d/env/kinematic_uav_env.py",
            "diffusion_policy_3d/env/collision_geom.py":
                repo_root / "diffusion_policy_3d/env/collision_geom.py",
            "diffusion_policy_3d/rl/dppo_uav_rebuild/environments/task_spec.py":
                repo_root / "diffusion_policy_3d/rl/dppo_uav_rebuild/environments/task_spec.py",
            "diffusion_policy_3d/rl/dppo_uav_rebuild/audit/admission_geometry.py":
                repo_root / "diffusion_policy_3d/rl/dppo_uav_rebuild/audit/admission_geometry.py",
            "diffusion_policy_3d/rl/dppo_uav_rebuild/audit/reset_bank.py":
                repo_root / "diffusion_policy_3d/rl/dppo_uav_rebuild/audit/reset_bank.py",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            manifest = root / "manifest.json"
            worlds = root / "worlds_train"
            config.write_text('{"config":true}', encoding="utf-8")
            manifest.write_text('{"manifest":true}', encoding="utf-8")
            worlds.mkdir()
            provenance = builder(
                config=config,
                manifest=manifest,
                worlds=worlds,
                seed_start=100,
                seeds_per_world=64,
                source_files=sources,
                created_utc="2026-09-08T00:00:00+00:00",
            )
        self.assertEqual(
            provenance["schema"], "reward_v17_admission_audit_provenance_v1",
        )
        self.assertEqual(provenance["mode"], "audit_only_no_behavior_change")
        self.assertEqual(provenance["config"], {
            "path": str(config.resolve()), "sha256": hashlib.sha256(b'{"config":true}').hexdigest(),
        })
        self.assertEqual(provenance["admission_manifest"], {
            "path": str(manifest.resolve()),
            "sha256": hashlib.sha256(b'{"manifest":true}').hexdigest(),
        })
        self.assertEqual(provenance["seed_start"], 100)
        self.assertEqual(provenance["seeds_per_world"], 64)
        self.assertEqual(set(provenance["source_files"]), set(sources))
        self.assertTrue(all(
            len(digest) == 64 for digest in provenance["source_files"].values()
        ))

    def test_missing_production_loader_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "admission_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            module_name = "diffusion_policy_3d.rl.dppo_uav.integration"
            with patch.dict(sys.modules, {module_name: None}):
                with self.assertRaisesRegex(
                    RuntimeError, "production admission loader unavailable"
                ):
                    load_production_tasks(manifest, root)

    def test_available_loader_with_empty_manifest_rejects_empty_task_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "admission_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            module_name = "diffusion_policy_3d.rl.dppo_uav.integration"
            fake_module = types.ModuleType(module_name)
            fake_module.manifest_train_entries = lambda _: []
            with patch.dict(sys.modules, {module_name: fake_module}):
                with self.assertRaisesRegex(ValueError, "at least one audit task"):
                    load_production_tasks(manifest, root)

    def test_duplicate_json_key_and_nonfinite_value_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_json_strict(duplicate)
            nonfinite = Path(directory) / "nonfinite.json"
            nonfinite.write_text('{"x":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                load_json_strict(nonfinite)

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside.world.obstacles.json"
            outside.write_text("{}", encoding="utf-8")
            try:
                with self.assertRaisesRegex(ValueError, "escapes worlds-train-dir"):
                    validate_obstacle_path(outside, root)
            finally:
                outside.unlink()

    def test_publish_pack_hashes_files_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pack"
            files = {
                "provenance.json": b"{}\n",
                "world_geometry_records.jsonl": b'{"world":"a"}\n',
                "reset_records.jsonl": b'{"seed":1}\n',
                "summary.json": b'{"status":"blocked"}\n',
            }
            publish_pack(output, files)
            hashes = json.loads((output / "SHA256.json").read_text(encoding="utf-8"))
            for name, payload in files.items():
                self.assertEqual(hashes[name], hashlib.sha256(payload).hexdigest())
            before = (output / "summary.json").read_bytes()
            with self.assertRaises(FileExistsError):
                publish_pack(output, files)
            self.assertEqual(before, (output / "summary.json").read_bytes())

    def test_two_publishers_cannot_both_succeed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pack"
            files = {
                "provenance.json": b"{}\n",
                "world_geometry_records.jsonl": b"",
                "reset_records.jsonl": b"",
                "summary.json": b"{}\n",
            }
            outcomes = []

            def worker():
                try:
                    publish_pack(output, files)
                    outcomes.append("success")
                except FileExistsError:
                    outcomes.append("exists")

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(sorted(outcomes), ["exists", "success"])
            self.assertTrue((output / "SHA256.json").is_file())

    def test_build_rejects_obstacle_replacement_during_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            obstacle_path = Path(directory) / "train-a.world.obstacles.json"
            obstacle_path.write_text('{"obstacles":[]}', encoding="utf-8")
            task = ResetAuditTask(
                task_id="train-a",
                world_uid="train-a",
                role="anticipatory",
                admission_id="train-a-admission",
                obstacle_json=str(obstacle_path),
            )
            changed = {"done": False}

            def make_environment(current_task, seed):
                environment = _AuditEnvironment()
                original_reset = environment.reset

                def reset(seed=None):
                    result = original_reset(seed)
                    if not changed["done"]:
                        obstacle_path.write_text(
                            '{"obstacles":[],"replaced":true}', encoding="utf-8",
                        )
                        changed["done"] = True
                    return result

                environment.reset = reset
                return environment

            with self.assertRaisesRegex(RuntimeError, "input hash changed"):
                build_audit_files(
                    [task], [10], make_environment, provenance={"test": True},
                )

    def test_exit_code_distinguishes_blocked_from_error(self):
        self.assertEqual(determine_exit_code([], 0), 0)
        self.assertEqual(determine_exit_code([{"status": "mismatch"}], 0), 2)
        self.assertEqual(determine_exit_code([], 1), 2)

    def test_build_audit_files_creates_complete_blocked_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            obstacle_path = Path(directory) / "train-a.world.obstacles.json"
            obstacle_path.write_text(json.dumps({
                "obstacles": [{
                    "name": "box", "shape": "box",
                    "x": 0.25, "y": 0.25, "z": 0.75,
                    "size_x": 0.5, "size_y": 0.5, "size_z": 1.0,
                    "yaw": 0.0,
                }],
            }), encoding="utf-8")
            task = ResetAuditTask(
                task_id="train-a",
                world_uid="train-a",
                role="anticipatory",
                admission_id="train-a-admission",
                obstacle_json=str(obstacle_path),
            )
            files, exit_code = build_audit_files(
                [task], list(range(10, 20)),
                lambda current_task, seed: _AuditEnvironment(),
                provenance={
                    "schema": "reward_v17_admission_audit_provenance_v1",
                    "mode": "audit_only_no_behavior_change",
                    "config": {"path": "/config.json", "sha256": "a" * 64},
                    "admission_manifest": {
                        "path": "/manifest.json", "sha256": "b" * 64,
                    },
                    "worlds_train_dir": str(Path(directory)),
                    "source_files": {"scripts/audit.py": "c" * 64},
                    "seed_start": 10,
                    "seeds_per_world": 10,
                    "production_grid_contract": {
                        "resolution": 0.5,
                        "xy_margin": 0.15,
                        "vertical_inflation": 0.0,
                    },
                },
            )
        self.assertEqual(exit_code, 2)
        self.assertEqual(set(files), {
            "provenance.json", "world_geometry_records.jsonl",
            "reset_records.jsonl", "summary.json",
        })
        summary = json.loads(files["summary.json"])
        self.assertEqual(summary["world_record_count"], 1)
        self.assertEqual(summary["reset_count"], 10)
        self.assertEqual(summary["decision"], "blocked")
        self.assertEqual(summary["final_decision"], "blocked")
        self.assertEqual(summary["shape_counts"], {"box": 1, "total": 1})
        self.assertEqual(summary["missed_blocked_cells_total"], 3)
        self.assertEqual(summary["extra_blocked_cells_total"], 0)
        self.assertEqual(summary["astar_shadow_count"], 8)
        self.assertIn("astar_reported_path_z_bounds_mismatch_count", summary)
        self.assertEqual(
            summary["astar_reported_path_z_bounds_mismatch_count"], 8,
        )
        self.assertIn("astar_bottleneck_unverified_reachable_branch_count", summary)
        self.assertEqual(
            summary["astar_bottleneck_unverified_reachable_branch_count"], 8,
        )
        self.assertEqual(summary["fallback"]["overall"]["count"], 0)
        self.assertEqual(summary["rejection_histogram"], {"success": 10})
        self.assertEqual(summary["audit_errors"], [])
        self.assertTrue(summary["final_reasons"])

        provenance = json.loads(files["provenance.json"])
        self.assertEqual(
            provenance["schema"], "reward_v17_admission_audit_provenance_v1",
        )
        self.assertEqual(provenance["mode"], "audit_only_no_behavior_change")
        self.assertRegex(provenance["ordered_task_identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("ordered_task_identities", provenance)

        world = json.loads(files["world_geometry_records.jsonl"].splitlines()[0])
        self.assertEqual(world["schema"], "reward_v17_world_geometry_audit_v1")
        self.assertEqual(world["status"], "mismatch")
        self.assertGreater(world["missed_blocked_cells"], 0)
        self.assertRegex(world["obstacle_json_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(world["obstacle_counts"], {"box": 1, "total": 1})
        self.assertEqual(world["grid"], {
            "shape": [5, 5, 4], "resolution": 0.5, "xy_margin": 0.15,
        })
        self.assertEqual(len(world["astar_shadow"]), 8)
        self.assertEqual([row["seed"] for row in world["astar_shadow"]], list(range(10, 18)))
        for row in world["astar_shadow"]:
            self.assertIn("production", row)
            self.assertIn("reference", row)
            self.assertIn("classification_changed", row)

        resets = [
            json.loads(line) for line in files["reset_records.jsonl"].splitlines()
        ]
        self.assertEqual(len(resets), 10)
        self.assertEqual(resets[0]["schema"], "reward_v17_reset_audit_v1")
        self.assertEqual(resets[0]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
