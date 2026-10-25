import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from diffusion_policy_3d.rl.dppo_uav_rebuild.audit.reset_bank import (
    GeometryAuditPointCloud,
    ResetAuditTask,
    audit_reset_bank,
    canonical_json_bytes,
    normalize_tasks_and_seeds,
)


class _FakeEnvironment:
    def __init__(self):
        self._drone_pos = None
        self._drone_yaw = None
        self.goal_w = None
        self._episode_fallback_type = None
        self._episode_fallback_reasons = []
        self._last_rejection_histogram = {}

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self._drone_pos = rng.normal(size=3).astype(np.float32)
        self._drone_yaw = float(rng.normal())
        self.goal_w = rng.normal(size=3).astype(np.float32)
        self._last_rejection_histogram = {"success": 1}
        return {}, {"world_id": "fake"}


def _task(name="a", world="train_a"):
    return ResetAuditTask(
        task_id=name,
        world_uid=world,
        role="anticipatory",
        admission_id=name + "-admission",
        obstacle_json=name + ".world.obstacles.json",
    )


class ResetBankTests(unittest.TestCase):
    def test_tasks_and_seeds_are_canonicalized(self):
        tasks, seeds = normalize_tasks_and_seeds([_task("b"), _task("a")], [2, 1])
        self.assertEqual([task.task_id for task in tasks], ["a", "b"])
        self.assertEqual(seeds, (1, 2))

    def test_duplicate_or_eval_only_identity_fails_before_reset(self):
        with self.assertRaisesRegex(ValueError, "duplicate task identity"):
            normalize_tasks_and_seeds([_task(), _task()], [1])
        with self.assertRaisesRegex(ValueError, "eval-only"):
            normalize_tasks_and_seeds([_task(world="w4")], [1])
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            normalize_tasks_and_seeds([_task()], [1, 1])

    def test_geometry_pointcloud_is_fixed_float32(self):
        stub = GeometryAuditPointCloud()
        first = stub.synthesize(None, None, num_points=4096, rng_seed=1)
        second = stub.synthesize(None, None, num_points=4096, rng_seed=999)
        self.assertEqual(first.shape, (4096, 3))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)

    def test_bank_is_byte_identical_and_has_exact_denominator(self):
        calls = []

        def make_environment(task, seed):
            calls.append((task.task_id, seed))
            return _FakeEnvironment()

        first = audit_reset_bank([_task("b"), _task("a")], [2, 1], make_environment)
        second = audit_reset_bank(
            [_task("a"), _task("b")], [1, 2],
            lambda task, seed: _FakeEnvironment(),
        )
        self.assertEqual(len(first["records"]), 4)
        self.assertEqual(len(set((r["task_id"], r["seed"]) for r in first["records"])), 4)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(calls, [("a", 1), ("a", 2), ("b", 1), ("b", 2)])
        first_record = first["records"][0]
        self.assertEqual(first_record["schema"], "reward_v17_reset_audit_v1")
        self.assertEqual(first_record["task_index"], 0)
        self.assertTrue(first_record["finite"])
        self.assertEqual(first_record["status"], "pass")
        self.assertEqual(first_record["sampling_outcome"], "sampled")

    def test_real_environment_reset_exhaustion_blocks_collection(self):
        try:
            from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
        except ModuleNotFoundError as exc:
            if exc.name == "gymnasium":
                self.skipTest("gymnasium is unavailable in the Windows audit environment")
            raise
        payload = {
            "spawn_world": [-8.0, 0.0],
            "obstacles": [{
                "name": "cover_all",
                "shape": "box",
                "x": 0.0,
                "y": 0.0,
                "z": 3.0,
                "size_x": 100.0,
                "size_y": 100.0,
                "size_z": 10.0,
                "yaw": 0.0,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            obstacle_path = Path(directory) / "blocked.world.obstacles.json"
            obstacle_path.write_text(json.dumps(payload), encoding="utf-8")
            task = dataclasses.replace(_task(), obstacle_json=str(obstacle_path))

            def make_environment(current_task, seed):
                return KinematicUAVEnv(
                    world_pool=[(current_task.world_uid, current_task.obstacle_json)],
                    synth_pc_factory=lambda _: GeometryAuditPointCloud(),
                    disturb_enable=False,
                    mid_state_init_ratio=0.0,
                    curriculum_stage="C2",
                    episode_role=current_task.role,
                    seed=seed,
                )

            from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.terminal_contract import (
                ResetExhausted,
            )
            with self.assertRaisesRegex(ResetExhausted, "after 50 attempts") as caught:
                audit_reset_bank([task], [17], make_environment)
        self.assertEqual(caught.exception.attempts, 50)
        self.assertEqual(caught.exception.rejection_histogram["clearance_start"], 50)
        self.assertEqual(caught.exception.rejection_histogram["clearance_goal"], 50)

    def test_nonfinite_reset_state_is_rejected(self):
        class BadEnvironment(_FakeEnvironment):
            def reset(self, seed=None):
                super().reset(seed)
                self._drone_pos[0] = np.nan
                return {}, {}

        with self.assertRaisesRegex(ValueError, "non-finite reset state"):
            audit_reset_bank([_task()], [1], lambda task, seed: BadEnvironment())


if __name__ == "__main__":
    unittest.main()
