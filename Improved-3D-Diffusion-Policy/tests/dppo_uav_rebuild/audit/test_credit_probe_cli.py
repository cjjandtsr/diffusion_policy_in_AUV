import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CreditProbeCliTest(unittest.TestCase):
    EXPECTED_SOURCE_SET_HASH = "e7408a3cf42ec7d19a4b09ee85f36b4a54b4ba416f9b21f7b4ffa8da76592604"

    def _script(self):
        return Path(__file__).resolve().parents[3] / "scripts" / "probe_dppo_credit.py"

    def test_self_check_runs_credit_and_gradient_path_without_parameter_change(self):
        repo_root = Path(__file__).resolve().parents[3]
        script = self._script()
        if not script.is_file():
            self.fail("probe_dppo_credit.py implementation is missing")

        completed = subprocess.run(
            [sys.executable, str(script), "--self-check"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "SELF_CHECK_PASS")
        self.assertEqual(payload["component_count"], 44)
        self.assertEqual(payload["parameter_delta_max_abs"], 0.0)
        self.assertLess(payload["gradient_reconstruction_max_abs"], 1e-7)

    def test_warning_summary_counts_reset_warning_only_on_episode_start(self):
        from scripts.probe_dppo_credit import _warning_summary

        rows = [
            {
                "decision_index": 0,
                "reset_no_ray_warning_count_at_episode_start": 1,
                "step_no_ray_warning_count": 0,
            },
            {
                "decision_index": 1,
                "reset_no_ray_warning_count_at_episode_start": 0,
                "step_no_ray_warning_count": 1,
            },
            {
                "decision_index": 2,
                "reset_no_ray_warning_count_at_episode_start": 0,
                "step_no_ray_warning_count": 1,
            },
        ]

        self.assertEqual(
            _warning_summary(rows),
            {
                "reset_warning_total": 1,
                "step_warning_total": 2,
                "episode_starts_with_reset_warning": 1,
                "transitions_with_step_warning": 2,
            },
        )

    def test_d1b_snapshot_uses_lora_adapter_state_contract(self):
        from dataclasses import dataclass
        from types import SimpleNamespace

        import torch

        from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.lora import (
            LoRABundle,
            LoRAIdentity,
        )
        from scripts.probe_dppo_credit import _d1b_snapshot

        class TinyActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_a = torch.nn.Parameter(torch.tensor([1.0]))
                self.lora_b = torch.nn.Parameter(torch.tensor([2.0]))

        @dataclass
        class Counters:
            attempted_updates: int = 0

        actor = TinyActor()
        bundle = LoRABundle(
            actor,
            SimpleNamespace(enabled=False),
            LoRAIdentity(
                target_names=("tiny",),
                rank=1,
                alpha=1.0,
                trainable_parameters=2,
            ),
        )
        critic = torch.nn.Linear(1, 1)
        actor_optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
        critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)
        system = SimpleNamespace(
            lora_bundle=bundle,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            trainer=SimpleNamespace(
                actor_parameters=tuple(actor.parameters()),
                critic_parameters=tuple(critic.parameters()),
                counters=Counters(),
            ),
            rollout_sampler=object(),
        )
        buffer = SimpleNamespace(state_dict=lambda: {"cursor": 0})

        snapshot = _d1b_snapshot(system, buffer)

        self.assertEqual(set(snapshot["actor"]), {"lora_a", "lora_b"})
        self.assertTrue(torch.equal(snapshot["actor"]["lora_a"], torch.tensor([1.0])))
        self.assertTrue(torch.equal(snapshot["actor"]["lora_b"], torch.tensor([2.0])))

    def test_probe_source_has_no_duplicate_literal_dict_keys(self):
        tree = ast.parse(self._script().read_text(encoding="utf-8"))
        duplicates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            literal_keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            repeated = sorted(
                key for key in set(literal_keys) if literal_keys.count(key) > 1
            )
            if repeated:
                duplicates.append((node.lineno, repeated))
        self.assertEqual(duplicates, [])

    def test_live_probe_rejects_non_divisible_block_before_loading_model(self):
        repo_root = Path(__file__).resolve().parents[3]
        completed = subprocess.run(
            [
                sys.executable,
                str(self._script()),
                "--config",
                "missing.yaml",
                "--policy-label",
                "rl",
                "--dev-bank",
                "missing-bank.json",
                "--rl-checkpoint",
                "missing-checkpoint",
                "--output-dir",
                "unused-output",
                "--expected-source-set-hash",
                self.EXPECTED_SOURCE_SET_HASH,
                "--block-transitions",
                "1000",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("divisible by 128", completed.stderr)

    def test_live_probe_refuses_existing_output_before_loading_model(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as existing:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(self._script()),
                "--config",
                "missing.yaml",
                "--policy-label",
                "rl",
                "--dev-bank",
                "missing-bank.json",
                "--rl-checkpoint",
                    "missing-checkpoint",
                    "--output-dir",
                    existing,
                    "--expected-source-set-hash",
                    self.EXPECTED_SOURCE_SET_HASH,
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("refusing to overwrite", completed.stderr)

    def test_valid_live_arguments_dispatch_into_runtime_loading(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "new-output"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(self._script()),
                    "--config",
                    "missing.yaml",
                    "--policy-label",
                    "rl",
                    "--dev-bank",
                    "missing-bank.json",
                    "--rl-checkpoint",
                    "missing-checkpoint",
                    "--output-dir",
                    str(output),
                    "--expected-source-set-hash",
                    self.EXPECTED_SOURCE_SET_HASH,
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("live probe execution is not yet enabled", completed.stderr)

    def test_rl_requires_checkpoint_and_bc_forbids_it(self):
        repo_root = Path(__file__).resolve().parents[3]
        common = [
            sys.executable,
            str(self._script()),
            "--config",
            "missing.yaml",
            "--dev-bank",
            "missing-bank.json",
            "--output-dir",
            "unused-new-output",
            "--expected-source-set-hash",
            self.EXPECTED_SOURCE_SET_HASH,
        ]
        rl = subprocess.run(
            common + ["--policy-label", "rl"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(rl.returncode, 0)
        self.assertIn("--rl-checkpoint is required", rl.stderr)

        bc = subprocess.run(
            common
            + [
                "--policy-label",
                "bc",
                "--rl-checkpoint",
                "unexpected-checkpoint",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(bc.returncode, 0)
        self.assertIn("forbidden for --policy-label bc", bc.stderr)

    def test_source_set_mismatch_blocks_before_config_or_model_import(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "new-output"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(self._script()),
                    "--config",
                    "missing.yaml",
                    "--policy-label",
                    "bc",
                    "--dev-bank",
                    "missing-bank.json",
                    "--output-dir",
                    str(output),
                    "--expected-source-set-hash",
                    "0" * 64,
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("source-set mismatch before model import", completed.stderr)
        self.assertNotIn("omegaconf", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
