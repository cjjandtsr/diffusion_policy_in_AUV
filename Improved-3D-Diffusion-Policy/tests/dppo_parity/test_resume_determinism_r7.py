"""R7-6C (guide R7-6 Step 6-7): fixed same-seed resume determinism.

The production comparator scripts/r7_compare_resume.py runs the fixed
protocol end-to-end:

    control: seed 202, uninterrupted updates 1..5
    resume:  seed 202, updates 1..2 -> v6 save -> NEW PROCESS -> 3..5

every update snapshots the nine comparison layers, and the comparator
stops at the first mismatch (guide Step 6). Step 7 requires EXACT
equality: discrete states/hashes/cursors/counters never use tolerances,
and FP tensors compare by exact hash under the deterministic runtime.

This test drives the comparator in a fresh subprocess (a process that
itself spawns control/resume stages as new processes — the resume side
is a cold start by construction) and asserts the structured summary:

    pass=true, metric_diffs=0, ckpt_diffs=0, first_diff=null
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_same_seed_resume_determinism(tmp_path):
    """The comparator's summary must pass the exact gate for every one
    of the nine layers across all five updates."""
    workdir = tmp_path / "r7_compare"
    proc = subprocess.run(
        [sys.executable, "-u",
         str(REPO_ROOT / "scripts" / "r7_compare_resume.py"),
         "--all", "--workdir", str(workdir)],
        capture_output=True, text=True, cwd=REPO_ROOT,
        timeout=900,
    )
    assert proc.returncode == 0, (
        "r7_compare_resume.py --all failed with "
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
    assert sorted(summary["layers_compared"]) == sorted(
        [
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
    ), f"layers_compared incomplete: {summary['layers_compared']}"
