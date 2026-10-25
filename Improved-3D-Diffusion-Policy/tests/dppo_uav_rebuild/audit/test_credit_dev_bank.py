import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


from scripts.dppo_credit_support.credit_bank import (
    load_dev_bank,
    make_dev_bank,
)


MANIFEST_HASH = "a" * 64


def _tasks():
    return (
        {
            "task_id": "task-b",
            "world_id": "world-b",
            "role": "dense_recovery",
            "task_seed": 102,
            "obstacle_sha256": "b" * 64,
        },
        {
            "task_id": "task-a",
            "world_id": "world-a",
            "role": "anticipatory",
            "task_seed": 101,
            "obstacle_sha256": "c" * 64,
        },
    )


class CreditDevBankTest(unittest.TestCase):
    def test_loader_rejects_boolean_masquerading_as_integer_seed(self):
        tasks = tuple(
            dict(task, task_seed=index + 1) for index, task in enumerate(_tasks())
        )
        payload = make_dev_bank(
            tasks,
            bank_id="credit-dev-v2",
            entry_count=4,
            source_manifest_sha256=MANIFEST_HASH,
            exclude_training_segments=4,
        )
        payload["training_task_seed_domain"][0] = True
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        payload["content_sha256"] = hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed domain"):
                load_dev_bank(
                    path,
                    expected_manifest_sha256=MANIFEST_HASH,
                    expected_tasks=tasks,
                )

    def test_bank_is_stable_when_cross_process_task_seed_mapping_is_permuted(self):
        first_tasks = _tasks()
        second_tasks = tuple(
            dict(task, task_seed=203 - task["task_seed"])
            for task in reversed(first_tasks)
        )

        first = make_dev_bank(
            first_tasks,
            bank_id="credit-dev-v2",
            entry_count=7,
            source_manifest_sha256=MANIFEST_HASH,
            exclude_training_segments=4,
        )
        second = make_dev_bank(
            second_tasks,
            bank_id="credit-dev-v2",
            entry_count=7,
            source_manifest_sha256=MANIFEST_HASH,
            exclude_training_segments=4,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["training_task_seed_domain"], [101, 102])
        self.assertNotIn("task_seed", first["source_tasks"][0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(first), encoding="utf-8")
            loaded = load_dev_bank(
                path,
                expected_manifest_sha256=MANIFEST_HASH,
                expected_tasks=second_tasks,
            )
            self.assertEqual(loaded, first)

    def test_loader_rejects_changed_training_task_seed_domain(self):
        payload = make_dev_bank(
            _tasks(),
            bank_id="credit-dev-v2",
            entry_count=4,
            source_manifest_sha256=MANIFEST_HASH,
            exclude_training_segments=4,
        )
        changed_seed_domain = tuple(
            dict(task, task_seed=task["task_seed"] + 1000) for task in _tasks()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed domain"):
                load_dev_bank(
                    path,
                    expected_manifest_sha256=MANIFEST_HASH,
                    expected_tasks=changed_seed_domain,
                )

    def test_bank_bytes_are_stable_across_python_hash_seeds(self):
        repo_root = Path(__file__).resolve().parents[3]
        program = r'''
import hashlib
import json
from scripts.dppo_credit_support.credit_bank import make_dev_bank

roles = {
    "anticipatory",
    "dense_recovery",
    "support_expansion",
    "yaw_lateral_recovery",
}
tasks = []
for index, role in enumerate(roles):
    tasks.append({
        "task_id": "task-" + role,
        "world_id": "world-" + role,
        "role": role,
        "task_seed": 20260828 + index,
        "obstacle_sha256": hashlib.sha256(role.encode("utf-8")).hexdigest(),
    })
payload = make_dev_bank(
    tasks,
    bank_id="credit-dev-v2",
    entry_count=16,
    source_manifest_sha256="a" * 64,
    exclude_training_segments=4,
)
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
'''
        outputs = []
        for hash_seed in ("1", "2", "3"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            completed = subprocess.run(
                [sys.executable, "-c", program],
                cwd=repo_root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outputs.append(completed.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_bank_rejects_resource_exhaustion_inputs(self):
        with self.assertRaisesRegex(ValueError, "entry_count"):
            make_dev_bank(
                _tasks(),
                bank_id="credit-dev-v1",
                entry_count=100001,
                source_manifest_sha256=MANIFEST_HASH,
                exclude_training_segments=4,
            )
        with self.assertRaisesRegex(ValueError, "exclude_training_segments"):
            make_dev_bank(
                _tasks(),
                bank_id="credit-dev-v1",
                entry_count=4,
                source_manifest_sha256=MANIFEST_HASH,
                exclude_training_segments=20001,
            )

    def test_bank_is_deterministic_task_uniform_and_excludes_training_reset_seeds(self):
        first = make_dev_bank(
            _tasks(),
            bank_id="credit-dev-v1",
            entry_count=7,
            source_manifest_sha256=MANIFEST_HASH,
            exclude_training_segments=4,
        )
        second = make_dev_bank(
            tuple(reversed(_tasks())),
            bank_id="credit-dev-v1",
            entry_count=7,
            source_manifest_sha256=MANIFEST_HASH,
            exclude_training_segments=4,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["entry_count"], 7)
        self.assertEqual(
            [row["task_id"] for row in first["entries"]],
            ["task-a", "task-b", "task-a", "task-b", "task-a", "task-b", "task-a"],
        )
        training_seeds = {
            task["task_seed"] + segment * 1000003
            for task in _tasks()
            for segment in range(5)
        }
        self.assertFalse(
            training_seeds.intersection(row["reset_seed"] for row in first["entries"])
        )
        self.assertEqual(
            len({row["reset_seed"] for row in first["entries"]}), 7
        )
        self.assertEqual(
            len({row["policy_seed"] for row in first["entries"]}), 7
        )

    def test_loader_rejects_tampering_duplicate_keys_and_manifest_mismatch(self):
        payload = make_dev_bank(
            _tasks(),
            bank_id="credit-dev-v1",
            entry_count=4,
            source_manifest_sha256=MANIFEST_HASH,
            exclude_training_segments=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_dev_bank(
                path,
                expected_manifest_sha256=MANIFEST_HASH,
                expected_tasks=_tasks(),
            )
            self.assertEqual(loaded["content_sha256"], payload["content_sha256"])

            tampered = dict(payload)
            tampered["entries"] = list(payload["entries"])
            tampered["entries"][0] = dict(tampered["entries"][0], reset_seed=7)
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content hash"):
                load_dev_bank(
                    path,
                    expected_manifest_sha256=MANIFEST_HASH,
                    expected_tasks=_tasks(),
                )

            duplicate = '{"schema":"one","schema":"two"}'
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_dev_bank(
                    path,
                    expected_manifest_sha256=MANIFEST_HASH,
                    expected_tasks=_tasks(),
                )

            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest"):
                load_dev_bank(
                    path,
                    expected_manifest_sha256="d" * 64,
                    expected_tasks=_tasks(),
                )


if __name__ == "__main__":
    unittest.main()
