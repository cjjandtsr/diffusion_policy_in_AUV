"""R2 rework #4 lifecycle evidence (审核裁决): the three negative recovery
paths, as an executable self-bound evidence artifact.

The 裁决 reopened VECTOR-ORDER (production train loop + adapter lifecycle
closure), ADMISSION (AdmissionExhausted -> sampler switches task) and
PC-VALID (reset-invalid recovery) and demands ONE evidence artifact that
specifically covers the three negative recovery paths:

  A. reset-invalid -> explicit slot close -> next task reset succeeds
     (P1-1: REJECT_EPISODE must not leak an ACTIVE slot)
  B. AdmissionExhausted -> attempt consumed + admission_blocked counted +
     sampler switches world/task, zero partial mutation (P1-2)
  C. loop-level max_ep_len truncation -> buffer truncated=True + slot
     explicitly closed, no ACTIVE leftover across batches (P1-3)

The executable coverage is the rework #4 lifecycle suite
(tests/dppo_parity/test_lifecycle_rework4.py, mock == real-adapter
semantics per the 裁决) plus the p7 eval recovery pins
(test_p7_eval_persistence.py, the PC-VALID reset-invalid eval path). Since
rework #5 (第五轮裁决, 2026-08-10) the p7 suite also covers the
paired-eval lifecycle closure: path C gains the loop-limit twin
(test_paired_eval_loop_limit_closes_slot_and_records_timeout) and a new
path D covers the dedicated single-slot eval-env boundary
(test_paired_eval_uses_dedicated_single_slot_eval_env). Since rework #6
(第六轮裁决, 2026-08-10) path E covers the eval entry preflight contract:
no_update+eval request hard-fails, missing eval_env fails before ANY
training side effect (env zero reset/step, policy unchanged, optimizer
not stepped, empty output dir, no checkpoint), negative eval_n_episodes
rejected at construction, training env cannot be reused as eval_env.
This runner executes them, records every test outcome, and self-binds
the report to the exact git tree (same discipline as
g3_10k_reset_gate_audit: hard-fail, never write a placeholder).

Usage:
  python scripts/r2_rework4_lifecycle_evidence.py \
      --output program/reports/routeA_rl_env_v1/g3_gate/LIFECYCLE_EVIDENCE_v1.json
"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 裁决 scenario -> test -> recovery path mapping (asserted in the report).
# Each test belongs to exactly one of the three negative recovery paths.
PATH_MAP = {
    "A_reset_invalid_close": [
        "test_loop_reset_reject_closes_slot_next_task_succeeds",
        "test_mock_single_slot_reset_at_rejects_active",
        "test_mock_reject_episode_and_mark_truncated_close_active",
        "test_real_adapter_lifecycle_state_transitions",
        "test_run_paired_eval_gates_invalid_reset_frames",
    ],
    "B_admission_exhausted_switch": [
        "test_loop_admission_exhausted_switches_task",
        "test_multi_slot_zero_partial_mutation_on_admission_exhaust",
        "test_real_adapter_zero_partial_mutation_on_admission_exhaust",
        "test_attempt_limit_exhausted_raises_incomplete_with_admission_blocked",
    ],
    "C_loop_truncation_close": [
        "test_loop_max_ep_len_writes_truncated_and_closes_slot",
        "test_consecutive_batches_loop_truncation_no_active_leftover",
        "test_run_paired_eval_gates_invalid_step_frames",
        # rework #5 (第五轮裁决 P1-1): the paired-eval twin of the C
        # path — a loop-limit exit (env never emits terminated/truncated)
        # must close the slot (mark_truncated) and record timeout=True.
        "test_paired_eval_loop_limit_closes_slot_and_records_timeout",
    ],
    # rework #5 (第五轮裁决 P1-2): paired eval requires a DEDICATED
    # n_envs=1 eval env — the multi-slot training env hard-fails at the
    # interface (n_envs != 1), so the boundary is part of the lifecycle
    # contract.
    "D_paired_eval_dedicated_eval_env": [
        "test_paired_eval_uses_dedicated_single_slot_eval_env",
    ],
    # rework #6 (第六轮裁决 P1/P2): the eval contract is enforced at the
    # run_e2e_batch ENTRY preflight — no_update+eval request hard-fails,
    # missing eval_env fails before any training side effect (env zero
    # reset/step, policy unchanged, optimizer not stepped, empty output
    # dir, no checkpoint), negative eval_n_episodes is rejected at
    # LoopConfig construction, and the training env itself cannot be
    # reused as eval_env. Entry-contract violations must precede any
    # state mutation — a lifecycle contract, not just an interface note.
    "E_eval_entry_preflight": [
        "test_no_update_with_eval_request_hard_fails",
        "test_missing_eval_env_fails_before_rollout_update_and_checkpoint",
        "test_negative_eval_n_episodes_rejected",
        "test_training_env_cannot_be_reused_as_eval_env",
    ],
}

TEST_FILES = [
    "tests/dppo_parity/test_lifecycle_rework4.py",
    "tests/dppo_parity/test_p7_eval_persistence.py",
]


def _tree_hash() -> str:
    """The exact git tree the evidence was produced from (self-binding)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], capture_output=True,
            text=True, cwd=_REPO_ROOT,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    raise RuntimeError(
        "provenance: cannot resolve git tree hash (not a git repo?); "
        "refusing to write a report without self-binding")


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_pytest_output(text: str) -> Dict[str, str]:
    """Collect PASSED/FAILED/ERROR lines into test_name -> outcome.

    pytest -v prints ``node_id PASSED`` (outcome at line END); the
    summary also emits ``PASSED <node_id>`` (outcome at line START) —
    accept both orders."""
    outcomes: Dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        # pytest -v appends a per-test progress suffix like " [  6%]".
        s = re.sub(r"\s+\[\s*\d+%\]\s*$", "", s)
        m = re.match(r"^(PASSED|FAILED|ERROR)\s+(.+)$", s) or \
            re.match(r"^(.+?)\s+(PASSED|FAILED|ERROR)$", s)
        if m:
            name = (m.group(2) if m.group(1) in ("PASSED", "FAILED", "ERROR")
                    else m.group(1)).split("::")[-1]
            outcome = (m.group(1) if m.group(1) in ("PASSED", "FAILED", "ERROR")
                       else m.group(2))
            outcomes[name] = outcome
    return outcomes


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=_REPO_ROOT,
                        help="repo root (evidence binds this tree)")
    args = parser.parse_args(argv)

    # pytest.main writes to the real sys.stdout/sys.stderr — capture them.
    capture = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = capture, capture
        code = pytest.main(
            ["-v", "-p", "no:cacheprovider", "-m", "not gpu", *TEST_FILES]
        )
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    raw = capture.getvalue()

    outcomes = _parse_pytest_output(raw)
    by_path: Dict[str, dict] = {}
    missing: List[str] = []
    for path, tests in PATH_MAP.items():
        results = {t: outcomes.get(t, "NOT-RUN") for t in tests}
        failed = [t for t, r in results.items() if r != "PASSED"]
        if failed:
            missing.append(f"{path}: {failed}")
        by_path[path] = {"tests": results, "failed": failed}

    report = {
        "title": "R2 rework #6 lifecycle evidence (第六轮裁决): negative "
                 "recovery paths A/B/C + paired-eval closure D + eval "
                 "entry preflight E",
        "pass": code == 0 and not missing,
        "exit_code": code,
        "tree_hash": _tree_hash(),
        "command": " ".join(["python", "scripts/r2_rework4_lifecycle_evidence.py",
                             "--output", args.output]),
        "test_files": TEST_FILES,
        "recovery_paths": by_path,
        "missing": missing,
        "log_sha256": _sha256_text(raw),
        "pytest_output": raw,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"lifecycle evidence: {args.output}")
    print(f"  tree {report['tree_hash']}  pass={report['pass']}")
    print(f"  recovery paths: {list(by_path)}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
