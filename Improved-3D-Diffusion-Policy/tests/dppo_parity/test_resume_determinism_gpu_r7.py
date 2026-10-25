"""F4 (Zread P1, 2026-08-24): same-seed resume determinism on the REAL
GPU stack.

Zread P1: scripts/r7_compare_resume.py ran the R7-6 protocol on CPU with
FakePolicy/MockUAVEnv — a same-seed resume ``pass`` produced by mocks
cannot admit G8.  This test drives the production comparator in
``--real-gpu`` mode:

  control: seed 202, uninterrupted updates 1..5
  resume:  seed 202, updates 1..2 -> v6 save -> NEW PROCESS -> 3..5

on the production components — frozen-loader BC policy (E5 retrain,
sha256 c1fca4ac…), zero-init registered LoRA, E040 critic, the real
UAVVectorEnvAdapter over the admission pool, CUDA, and the production
preflight facts (validate_official_environment) — and asserts the same
structured gate as the CPU protocol: pass=true, metric_diffs=0,
ckpt_diffs=0, first_diff=null, five updates on both sides, all nine
layers compared.

Guide: R7-10 Step 5 contract — code change -> new commit + RTM REWORK +
targeted tests + reviewer re-sign.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "r7_compare_resume.py"

# Production assets (same defaults as tests/dppo_parity/conftest.py; env
# overrides let a mirror point elsewhere).
DEFAULT_BC_CKPT = str(
    REPO_ROOT
    / "data/outputs/2026.08.16/08.53.26_e5_lambda0p5"
    / "checkpoints/latest.ckpt"
)
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS_TRAIN = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
)

EXPECTED_LAYERS = [
    "1_load_complete_state",
    "2_next_task_world_episode",
    "3_rollout_noise_action_chain_logprob",
    "4_reward_outcome_buffer_next_values",
    "5_values_advantages_returns",
    "6_randperm_minibatch",
    "7_actor_critic_grads",
    "8_parameters_optimizer_moments",
    "9_sampler_slots_counters_metrics",
]


@pytest.mark.gpu
def test_same_seed_resume_determinism_gpu_r7(tmp_path):
    """The --real-gpu comparator's summary must pass the exact gate for
    every one of the nine layers across all five updates."""
    if not torch.cuda.is_available():
        pytest.skip("gpu marker: CUDA unavailable")
    for name, default in (
        ("DPPO_BC_CKPT", DEFAULT_BC_CKPT),
        ("DPPO_MANIFEST", DEFAULT_MANIFEST),
        ("DPPO_WORLDS_TRAIN", DEFAULT_WORLDS_TRAIN),
    ):
        path = os.environ.get(name, default)
        if not os.path.exists(path):
            pytest.skip(f"{name} asset not found: {path}")

    workdir = tmp_path / "r7_compare_gpu"
    env = dict(os.environ)
    # 2026-08-24 F4 rework: measured real-GPU cadence is ~20 min/update
    # (first update ~24 min incl. CUDA warm-up); the 10-update protocol
    # (control 5 + resume_a 2 + resume_b 3 + v6 save + fresh process)
    # needs ~3.5 h, so 3600 s timed out after only 2.5 updates.
    proc = subprocess.run(
        [sys.executable, "-u", str(SCRIPT), "--all", "--real-gpu",
         "--workdir", str(workdir)],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=14400,
        env=env,
    )
    assert proc.returncode == 0, (
        "r7_compare_resume.py --real-gpu --all failed with "
        f"exit {proc.returncode}:\n--- stdout ---\n{proc.stdout[-6000:]}\n"
        f"--- stderr ---\n{proc.stderr[-6000:]}"
    )
    summary_path = workdir / "summary.json"
    assert summary_path.is_file(), (
        f"comparator did not write {summary_path} — stdout: "
        f"{proc.stdout[-2000:]}")
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == "r7-compare-v1"
    assert summary["pass"] is True, (
        f"same-seed resume determinism FAILED: {summary['first_diff']}")
    assert summary["metric_diffs"] == 0, (
        f"expected metric_diffs=0, got {summary['metric_diffs']}: "
        f"{summary['first_diff']}")
    assert summary["ckpt_diffs"] == 0, (
        f"expected ckpt_diffs=0, got {summary['ckpt_diffs']}")
    assert summary["first_diff"] is None
    # every update on both sides was compared across all nine layers
    assert summary["control"]["updates"] == 5
    assert summary["resume"]["updates"] == 5
    assert sorted(summary["layers_compared"]) == sorted(EXPECTED_LAYERS), (
        f"layers_compared incomplete: {summary['layers_compared']}")
