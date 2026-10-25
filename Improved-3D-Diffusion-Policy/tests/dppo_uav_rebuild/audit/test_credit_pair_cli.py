import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def _write_probe(root, label, *, bank_hash="a" * 64):
    root.mkdir()
    summary = {
        "schema": "dppo_credit_probe_summary_v2",
        "status": "PROBE_COMPLETE_NO_PARAMETER_UPDATE",
        "policy_label": label,
        "dev_bank_content_sha256": bank_hash,
        "transitions": 2,
        "gradient_minibatches": 1,
        "parameter_delta": {"max_abs": 0.0, "nonzero_elements": 0},
        "training_counters_unchanged": True,
        "parameter_gradients_unchanged": True,
        "training_call_tripwire_triggered": False,
    }
    provenance = {
        "schema": "dppo_credit_probe_provenance_v2",
        "policy_label": label,
        "dev_bank_content_sha256": bank_hash,
        "config_sha256": "b" * 64,
        "base_bc_checkpoint_sha256": "c" * 64,
        "script_sha256": "e" * 64,
        "credit_module_sha256": "f" * 64,
        "credit_bank_module_sha256": "1" * 64,
        "expected_source_set_hash": "2" * 64,
        "actual_source_set_hash": "2" * 64,
        "checkpoint_update": 0 if label == "bc" else 256,
        "training_enabled": False,
        "optimizer_step_called": False,
        "training_call_tripwire_armed": True,
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with (root / "transitions.jsonl").open("w", encoding="utf-8") as stream:
        for decision in range(2):
            stream.write(
                json.dumps(
                    {
                        "schema": "dppo_credit_transition_v2",
                        "policy_label": label,
                        "bank_entry_index": 0,
                        "decision_index": decision,
                    }
                )
                + "\n"
            )
    (root / "gradient_minibatches.jsonl").write_text(
        json.dumps({"schema": "dppo_credit_gradient_v1"}) + "\n",
        encoding="utf-8",
    )


class CreditPairCliTest(unittest.TestCase):
    def _script(self):
        return Path(__file__).resolve().parents[3] / "scripts" / "validate_dppo_credit_pair.py"

    def test_pair_gate_accepts_bound_read_only_outputs_and_rejects_bank_mismatch(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bc = root / "bc"
            rl = root / "rl"
            _write_probe(bc, "bc")
            _write_probe(rl, "rl")
            command = [
                sys.executable,
                str(self._script()),
                "--bc-dir",
                str(bc),
                "--rl-dir",
                str(rl),
                "--expected-transitions",
                "2",
                "--expected-gradient-minibatches",
                "1",
                "--expected-rl-update",
                "256",
                "--expected-source-set-hash",
                "2" * 64,
            ]
            accepted = subprocess.run(
                command,
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(json.loads(accepted.stdout)["status"], "PAIR_GATE_PASS")

            _write_probe(root / "wrong", "rl", bank_hash="d" * 64)
            rejected = subprocess.run(
                command[:5] + [str(root / "wrong")] + command[6:],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("dev bank", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
