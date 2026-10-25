"""F3 (Zread P0-2/P1, 2026-08-24): a resume gate result must be bound to
the REAL raw GPU pytest log that produced it.

Zread P1: the resume comparator ran on CPU with FakePolicy/MockUAVEnv
(r7_compare_resume.py) — a hand-written ``pass=true`` JSON cannot be
distinguished from a real run.  Every r7-compare-v1 now must carry:
- ``producer`` from the frozen allowlist (pytest),
- a ``log_path`` that exists and whose sha256 matches ``log_sha256``,
- ``gpu_test_nodeids`` — the GPU tests that actually ran, and every
  nodeid must appear in the bound log text (a CPU run or a run with the
  GPU test skipped would not contain them).

Attack matrix covered here (F3): CPU-mock resume JSON, missing producer,
missing nodeid list, nodeid absent from the real log, log hash mismatch,
log file missing.  The other admission attacks (empty verified_inputs,
forged decision, fake CPU/GPU log path, wrong log hash) are covered in
test_admission_v2_reexecute.py and test_admission_log_binding.py.
"""

import hashlib
import json

import pytest

from test_final_admission_fail_closed import (
    REPO_ROOT,
    _git,
    _input_bundle,
    _evaluate,
)
from diffusion_policy_3d.rl.dppo_uav import admission_parsers as ap

GPU_NODEID = (
    "tests/dppo_parity/test_resume_determinism_gpu_r7.py"
    "::test_same_seed_resume_determinism_gpu_r7"
)


def _write_log(tmp_path, name, content):
    log = tmp_path / name
    log.write_bytes(content)
    return str(log), hashlib.sha256(content).hexdigest()


def _resume_blob(log_path, log_sha, *, producer="pytest",
                 nodeids=(GPU_NODEID,), pass_=True):
    return {
        "schema_version": "r7-compare-v1",
        "pass": pass_,
        "metric_diffs": 0,
        "ckpt_diffs": 0,
        "first_diff": None,
        "control": {"ckpt": "/c.pt", "updates": 5},
        "resume": {"ckpt": "/r.pt", "updates": 5},
        "layers_compared": ["1_load_complete_state",
                            "2_next_task_world_episode",
                            "3_rollout_noise_action_chain_logprob",
                            "4_reward_outcome_buffer_next_values",
                            "5_values_advantages_returns",
                            "6_randperm_minibatch",
                            "7_actor_critic_grads",
                            "8_parameters_optimizer_moments",
                            "9_sampler_slots_counters_metrics"],
        "producer": producer,
        "gpu_test_nodeids": list(nodeids),
        "log_path": log_path,
        "log_sha256": log_sha,
    }


def _bundle_with_resume(tmp_path, *, log_content=None, **resume_overrides):
    """Full valid bundle with a custom resume JSON (rest valid)."""
    paths, _ = _input_bundle(tmp_path)
    content = log_content or (
        b"GPU resume determinism\n" + GPU_NODEID.encode() + b" PASSED\n"
    )
    log_path, log_sha = _write_log(tmp_path, "resume_gpu.log", content)
    blob = _resume_blob(log_path, log_sha)
    blob.update(resume_overrides)
    paths["resume"].write_text(json.dumps(blob))
    return paths


# ────────────────────────────────────────────────────────────────
# RED (Zread P1 repro): CPU-mock / detached resume JSON must be rejected
# ────────────────────────────────────────────────────────────────

def test_cpu_mock_resume_rejected(tmp_path):
    """Hand-written resume JSON with no producer and a fake log (the
    pre-F3 r7_compare_resume.py shape) must fail closed."""
    paths = _bundle_with_resume(
        tmp_path, producer="handwritten",
        log_path="/definitely/not/existing/resume.log",
        log_sha256="cd" * 32,
    )
    d = _evaluate(paths)
    assert d.engine_correct is False
    assert any("producer" in s for s in d.stale_artifacts)


def test_resume_missing_producer_rejected(tmp_path):
    """No producer field at all -> reject (required key)."""
    paths = _bundle_with_resume(tmp_path)
    blob = json.loads(paths["resume"].read_text())
    del blob["producer"]
    paths["resume"].write_text(json.dumps(blob))
    d = _evaluate(paths)
    assert d.engine_correct is False


def test_resume_missing_nodeids_rejected(tmp_path):
    """No gpu_test_nodeids field -> reject (a resume result that cannot
    name the GPU test that produced it is not a GPU resume result)."""
    paths = _bundle_with_resume(tmp_path)
    blob = json.loads(paths["resume"].read_text())
    del blob["gpu_test_nodeids"]
    paths["resume"].write_text(json.dumps(blob))
    d = _evaluate(paths)
    assert d.engine_correct is False


def test_resume_nodeid_absent_from_log_rejected(tmp_path):
    """Log exists + hash matches + producer OK, but the GPU nodeid never
    ran (e.g. CPU run, or GPU test skipped) -> reject."""
    paths = _bundle_with_resume(
        tmp_path,
        log_content=b"all tests collected, no GPU nodeid anywhere\n",
    )
    d = _evaluate(paths)
    assert d.engine_correct is False
    assert any("nodeid" in s for s in d.stale_artifacts)


def test_resume_log_hash_mismatch_rejected(tmp_path):
    """Log exists but declared sha256 is wrong -> reject."""
    log_path, _ = _write_log(tmp_path, "resume_gpu.log",
                             b"real GPU resume log\n")
    paths = _bundle_with_resume(
        tmp_path, log_path=log_path, log_sha256="cd" * 32)
    d = _evaluate(paths)
    assert d.engine_correct is False
    assert any("sha256 mismatch" in s for s in d.stale_artifacts)


def test_resume_log_missing_rejected(tmp_path):
    """log_path points at a missing file -> reject."""
    paths = _bundle_with_resume(
        tmp_path, log_path=str(tmp_path / "nope.log"))
    d = _evaluate(paths)
    assert d.engine_correct is False


def test_valid_gpu_resume_admits(tmp_path):
    """Real log containing the GPU nodeid + allowlisted producer admits."""
    paths = _bundle_with_resume(tmp_path)
    d = _evaluate(paths)
    assert d.engine_correct is True, d.stale_artifacts
