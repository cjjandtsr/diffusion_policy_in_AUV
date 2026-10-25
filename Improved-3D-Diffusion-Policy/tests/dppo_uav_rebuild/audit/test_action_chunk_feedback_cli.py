import ast
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_routea_action_chunk_feedback import main


def _write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _manifest(root):
    files = sorted(
        path for path in root.iterdir() if path.is_file() and path.name != "manifest_sha256.txt"
    )
    (root / "manifest_sha256.txt").write_text(
        "".join("{}  {}\n".format(_sha(path), path.name) for path in files),
        encoding="utf-8",
    )


def _action(dx):
    value = np.zeros((8, 11), dtype=np.float32)
    value[:4, 0] = dx
    return value.tolist()


class ActionChunkFeedbackCliTests(unittest.TestCase):
    def test_delivery_sources_parse_with_python38_grammar(self):
        repository = Path(__file__).resolve().parents[3]
        for relative in (
            "scripts/analyze_routea_action_chunk_feedback.py",
            "scripts/routea_causal_diagnostics/action_feedback.py",
        ):
            source = (repository / relative).read_text(encoding="utf-8")
            ast.parse(source, filename=relative, feature_version=(3, 8))

    def test_self_check(self):
        self.assertEqual(main(["--self-check"]), 0)

    def test_cli_publishes_read_only_audit_with_parent_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            final = root / "final"
            b1 = root / "b1"
            output = root / "output"
            capture.mkdir()
            final.mkdir()
            b1.mkdir()
            lineage = "d05b-test-lineage"
            selected = {
                "task_id": "task-a",
                "repeat": 1,
                "k_div": 0,
                "decision_count": 6,
                "outcome_class": "break",
                "role": "anticipatory",
                "episode_config": {
                    "start_pos": [0.0, 0.0, 0.0],
                    "goal_pos": [10.0, 0.0, 0.0],
                },
            }
            selection = {
                "schema": "routea_d05b_selection_v1",
                "run_lineage_id": lineage,
                "target_repeat_count": 1,
                "target_task_count": 1,
                "targets": [selected],
            }
            _write_json(capture / "selection.json", selection)
            null = {
                "task-a": {
                    str(index): {
                        "h4_path": {"threshold": 0.001},
                        "endpoint": {"threshold": 0.001},
                    }
                    for index in range(6)
                }
            }
            _write_json(capture / "bc_repeat_null.json", null)
            transition_path = capture / "paired_trace_transitions.jsonl"
            with transition_path.open("w", encoding="utf-8", newline="\n") as stream:
                for decision in range(6):
                    for lane, dx, y in (
                        ("u0", 0.0, 0.0),
                        ("seed3-u32", 0.1 * (decision + 1), 0.1 * (decision + 1)),
                    ):
                        action = _action(dx)
                        stream.write(
                            json.dumps(
                                {
                                    "schema": "routea_d05b_transition_v1",
                                    "task_id": "task-a",
                                    "repeat": 1,
                                    "lane": lane,
                                    "outcome_class": "break",
                                    "role": "anticipatory",
                                    "decision_index": decision,
                                    "action_seed": 100 + decision,
                                    "policy_input_hash": "same" if decision == 0 else lane,
                                    "position_before_xyz": [float(decision), y, 0.0],
                                    "position_after_xyz": [float(decision + 1), y, 0.0],
                                    "goal_xyz": [10.0, 0.0, 0.0],
                                    "policy_action_8x11": action,
                                    "executed_action_4x11": action[:4],
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
            _manifest(capture)

            _write_json(
                final / "identity.json",
                {
                    "run_lineage_id": lineage,
                    "capture_manifest_sha256": _sha(capture / "manifest_sha256.txt"),
                    "selection_sha256": _sha(capture / "selection.json"),
                },
            )
            _write_json(final / "d05_verdict.json", {"status": "INCONCLUSIVE"})
            (final / "selection.json").write_bytes((capture / "selection.json").read_bytes())
            _manifest(final)

            fields = [
                "checkpoint",
                "update",
                "stratum",
                "task_id",
                "snapshot_id",
                "action_h4_l2",
                "goal_parallel_m",
                "goal_perpendicular_m",
                "vertical_m",
                "yaw_total_variation_rad",
                "path_shape_norm_m",
            ]
            with (b1 / "bc_retention_drift.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for seed in (1, 2, 3):
                    for update in (8, 16, 24, 32):
                        writer.writerow(
                            {
                                "checkpoint": "seed{}-u{}".format(seed, update),
                                "update": update,
                                "stratum": "S1_SEED1_RESCUE",
                                "task_id": "task-a",
                                "snapshot_id": "snapshot-a",
                                "action_h4_l2": seed * update / 100.0,
                                "goal_parallel_m": 0.1,
                                "goal_perpendicular_m": 0.0,
                                "vertical_m": 0.0,
                                "yaw_total_variation_rad": 0.01,
                                "path_shape_norm_m": 0.01,
                            }
                        )
            _manifest(b1)

            self.assertEqual(
                main(
                    [
                        "run-audit",
                        "--d05-capture-dir",
                        str(capture),
                        "--d05-final-dir",
                        str(final),
                        "--b1-dir",
                        str(b1),
                        "--output-dir",
                        str(output),
                    ]
                ),
                0,
            )
            self.assertTrue((output / "manifest_sha256.txt").is_file())
            summary = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["task_count"], 1)
            identity = json.loads((output / "input_identity.json").read_text(encoding="utf-8"))
            self.assertEqual(identity["d05_run_lineage_id"], lineage)
            tripwire = json.loads(
                (output / "tripwire_verification.json").read_text(encoding="utf-8")
            )
            self.assertFalse(tripwire["model_loaded"])
            self.assertFalse(tripwire["training_performed"])
            self.assertEqual(
                main(["validate-output", "--output-dir", str(output)]), 0
            )

            manifest_lines = (capture / "manifest_sha256.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            (capture / "manifest_sha256.txt").write_text(
                "\n".join(
                    line
                    for line in manifest_lines
                    if not line.endswith("  paired_trace_transitions.jsonl")
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "run-audit",
                        "--d05-capture-dir",
                        str(capture),
                        "--d05-final-dir",
                        str(final),
                        "--b1-dir",
                        str(b1),
                        "--output-dir",
                        str(root / "unsealed-output"),
                    ]
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
