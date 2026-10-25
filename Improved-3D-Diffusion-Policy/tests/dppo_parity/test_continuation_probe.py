"""R2 P1-3: two-process SEED-RESUME continuation (checkpoint sampler state).

The audit finding: SEED-RESUME only wired the interface — the resume probe
constructed a FRESH sampler every run, so an interrupted run restarted its
episode seeds at base_seed after resume. Production resume must restore
the sampler's attempt/cursor/RNG state from the checkpoint, and the
two-PROCESS continuation must never reuse episode seeds.

This test drives the CPU probe (continuation_probe.py) through two real
process boundaries:
  run1: one_update on a fresh world  -> episode seeds S1, checkpoint C1
  run2: FRESH process, resume_from=C1 -> episode seeds S2
and asserts the strict continuation contract: max(S1) < min(S2). Before
the rework, run2 restarted at base_seed and reused S1 — this test was
RED. After the rework (sampler_state in the checkpoint, restored on
resume), it is GREEN.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parent / "continuation_probe.py"


def _run_probe(
    tmp_path: Path,
    tag: str,
    resume_from: str = "",
    base_seed: int = 1000,
    n_episodes: int = 2,
) -> dict:
    out = tmp_path / f"{tag}.json"
    cmd = [
        sys.executable, str(PROBE),
        "--out", str(out),
        "--base-seed", str(base_seed),
        "--n-episodes", str(n_episodes),
    ]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, cwd=tmp_path
    )
    assert proc.returncode == 0, f"probe {tag} failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(out.read_text())


def test_two_process_resume_sampler_continuation(tmp_path):
    """SEED-RESUME across a process boundary: no seed reuse after resume."""
    run1 = _run_probe(tmp_path, "run1", base_seed=1000, n_episodes=2)
    ckpt = run1["checkpoint_path"]
    assert ckpt and Path(ckpt).is_file(), "run1 wrote no checkpoint"

    run2 = _run_probe(
        tmp_path, "run2", resume_from=ckpt, base_seed=1000, n_episodes=2
    )

    # within-run uniqueness: seeds are base_seed + attempt, strictly unique
    for tag, run in (("run1", run1), ("run2", run2)):
        seeds = run["episode_seeds"]
        assert len(seeds) == len(set(seeds)), f"{tag} reused a seed: {seeds}"
        assert len(seeds) == 2, f"{tag} completed wrong episode count: {seeds}"

    # SEED-RESUME contract: the resumed process CONTINUES the sequence —
    # never restarts at base_seed (run2's first seed == 1000 is the bug).
    assert max(run1["episode_seeds"]) < min(run2["episode_seeds"]), (
        "resumed process reused episode seeds: run1="
        f"{run1['episode_seeds']} run2={run2['episode_seeds']}"
    )

    # the resumed checkpoint accumulates the run's transitions (counters
    # carry through the checkpoint, not per-process reset)
    assert run2["env_steps"] == run1["env_steps"] == 10


def test_resume_checkpoint_carries_sampler_state(tmp_path):
    """The checkpoint written by run1 must persist the sampler's full state.

    Direct assertion at the checkpoint level (not just via the second
    run's behavior): the saved sampler_state must carry the schema and the
    consumed attempt index, so a restored run can continue numbering.
    """
    run1 = _run_probe(tmp_path, "run1")
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder

    data = CheckpointBuilder.load(Path(run1["checkpoint_path"]))
    state = data.sampler_state
    assert state["schema"] == "episode_task_sampler/v1"
    assert state["global_attempt_index"] == 2  # two attempts consumed
    assert state["global_episode_index"] == 2
    assert state["base_seed"] == 1000
    # cursors advanced: at least one role was sampled twice
    assert sum(state["cursors"].values()) == 2
