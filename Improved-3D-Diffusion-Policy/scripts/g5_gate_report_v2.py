"""g5_gate_report_v2.py — build the structured G5 gate evidence report
(R4.2 corrected semantics).

Mirrors the G3/G6 precedent: a single JSON the design §5.7 / RTM audit
trail can consume without parsing logs.

Inputs:
  --test-log  the pytest -s output of test_g5_real_no_update_gpu.py
              (its evidence prints: SAFE_END traces, the identity
              summary line)
Output: the report JSON at --out (default mirrors G5's location).

The §5.7 gates are reported from the test's own assertion results: the
test passing is the gate; this collector records the evidence values the
test printed plus the corrected-semantics notes:
  - SAFE_END = 0 (Guide R4.2 forbids accepting SAFE_END ~= 1/32; the
    post-R2.5 far-dome + analytic contract must validate the degenerate
    band);
  - identity measured at the production likelihood microbatch = 1
    (behavior batch; TrainerConfig.batch_size=128 is the optimizer
    minibatch, not a likelihood batch — R1 rework #2 / ASSUMPTION-11,
    Guide R1 v1.1 superseding the v1.0 B=128 clause);
  - ratio mean in [0.99999, 1.00001]; zero-advantage pg_loss <= 1e-7.

Run:
  python scripts/g5_gate_report_v2.py --test-log temp/g5_v2.log \
      --out /home/cjj/AUV/program/reports/routeA_rl_env_v1/g5_gate/\
G5_GATE_REPORT_v2.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def _find_line(text: str, prefix: str) -> Optional[str]:
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    return None


def parse_test_log(log_text: str) -> Dict[str, Any]:
    """Extract the structured evidence prints the gate test emits."""
    summary = _find_line(log_text, "G5 identity summary:")
    assert summary, "no G5 identity summary print in test log (test failed?)"
    # G5 identity summary: completed=32 voided=0 safe_end=0
    # invalid_stored=0 max_abs_diff=0.0 ratio_mean=1.0 env_steps=...
    # buffer_n=...
    def field(name: str, cast) -> Any:
        m = re.search(rf"{name}=([^\s]+)", summary)
        assert m, f"summary lacks {name}: {summary!r}"
        return cast(m.group(1))

    traces = [
        line.strip() for line in log_text.splitlines()
        if line.strip().startswith("G5 SAFE_END trace:")
    ]
    return {
        "completed_episodes": field("completed", int),
        "safe_end_episodes": field("safe_end", int),
        "invalid_stored_transitions": field("invalid_stored", int),
        "identity_max_abs_diff": field("max_abs_diff", float),
        "ratio_mean": field("ratio_mean", float),
        "env_steps": field("env_steps", int),
        "buffer_n": field("buffer_n", int),
        "safe_end_traces": traces,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--pytest-exit-code", type=int, default=0,
        help="the pytest exit code of the gate run (capture $? after the "
        "tee) — a FAILED test can still print the summary line before "
        "its later assertion fails, so the report must bind the exit "
        "code (review_R4_part1 P2-1)",
    )
    args = parser.parse_args()
    t = parse_test_log(Path(args.test_log).read_text())
    report: Dict[str, Any] = {
        "gate": "G5",
        "gate_version": "v2",
        "gate_source": (
            "test_g5_real_no_update_gpu.py (Guide R4.2 corrected semantics)"
        ),
        "identity_semantics": {
            "measured_at": "production likelihood microbatch = 1 "
            "(the behavior batch; rollout stores B=1 logprobs once)",
            "optimizer_batch_size": 128,
            "batch_128_is_optimizer_batch_not_likelihood_batch": True,
            "supersedes": (
                "Guide R4.2 'B=128 identity' clause is superseded by the "
                "R1 v1.0->v1.1 Guide amendment + ASSUMPTION-11 (the "
                "measured B=128 kernel dependence was 3.66e-4 > 1e-5 — "
                "informational only, never an acceptance value)"
            ),
        },
        "safe_end_semantics": {
            "required": 0,
            "guide_clause": (
                "Guide R4.2 forbids accepting SAFE_END ~= 1/32; the "
                "post-R2.5 production gate must validate the frozen-sim "
                "degenerate band via the far-dome + analytic contract "
                "(design v1.1 §4.7), not SAFE_END it"
            ),
            "defense_in_depth": (
                "any SAFE_END must still trace 1:1 to a gate-invalid "
                "frame (spy == routing == spy_gate == loop gate)"
            ),
        },
        "gates": {
            "completed_episodes_eq_32": (
                "test-asserted" if t["completed_episodes"] == 32 else "FAIL"
            ),
            "reject_exclude_hard_eq_0": "test-asserted",
            "safe_end_eq_0": (
                "test-asserted" if t["safe_end_episodes"] == 0 else "FAIL"
            ),
            "invalid_stored_transitions_eq_0": (
                "test-asserted"
                if t["invalid_stored_transitions"] == 0 else "FAIL"
            ),
            "executed_action_eq_deployment_slice": "test-asserted",
            "latest_obs_indices_5_6_7": "test-asserted",
            "next_obs_alignment_exact": "test-asserted",
            "chain_logprob_shapes_exact": "test-asserted",
            "bc_hash_unchanged": "test-asserted",
            "identity_max_abs_diff_leq_1e-5": (
                "test-asserted" if t["identity_max_abs_diff"] <= 1e-5
                else "FAIL"
            ),
            "ratio_mean_in_099999_100001": (
                "test-asserted"
                if 0.99999 <= t["ratio_mean"] <= 1.00001 else "FAIL"
            ),
            "zero_advantage_pg_loss_leq_1e-7": "test-asserted",
        },
        "evidence": t,
        "pytest_exit_code": args.pytest_exit_code,
        "status": (
            "PASS"
            if (args.pytest_exit_code == 0
                and t["completed_episodes"] == 32
                and t["safe_end_episodes"] == 0
                and t["invalid_stored_transitions"] == 0
                and t["identity_max_abs_diff"] <= 1e-5
                and 0.99999 <= t["ratio_mean"] <= 1.00001)
            else "FAIL"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
