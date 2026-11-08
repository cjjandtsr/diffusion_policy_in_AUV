#!/usr/bin/env python3
"""R7-7: the ONLY canonical signer of engine_correct /
training_admitted_for_g8 (guide R7-7 Step 1-6 + Step 10).

Every other verifier (r6_admission_report.py) may validate LOCAL
slices; this CLI is the single place that may write an
r7-final-admission-v1 decision JSON.  The training launcher's formal
preflight then reads that JSON back (Step 7).

Usage:
    python scripts/r7_final_admission.py \
        --rtm <rtm.json> \
        --implementation-manifest <manifest.json> \
        --test-impact-matrix <matrix.json> \
        --full-cpu <cpu.json> --full-gpu <gpu.json> \
        --g3 <g3.json> --g5 <g5.json> --g6 <g6.json> \
        --resume <resume.json> \
        --g7 <g7.json> \
        --paired-bank <bank.json> \
        --review <review.json> [--review ...] \
        --expected-commit <hex> --expected-tree <hex> \
        --repo-root <repo> --out <decision.json>

Exit 0 only when engine_correct is true; the decision JSON is still
written on a denial (so the operator can read WHY), but the process
exit code is non-zero (Step 8: denied admission must exit != 0).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
    decision_content_hash,
    evaluate_final_admission,
)


def _pair(argv, name: str, required: bool = True) -> str | None:
    if name in argv:
        idx = argv.index(name)
        if idx + 1 < len(argv):
            return argv[idx + 1]
        raise ValueError(f"{name} requires a value")
    if required:
        raise ValueError(f"missing required {name}")
    return None


def main(argv) -> int:
    try:
        rtm = _pair(argv, "--rtm")
        manifest = _pair(argv, "--implementation-manifest")
        matrix = _pair(argv, "--test-impact-matrix")
        cpu = _pair(argv, "--full-cpu")
        gpu = _pair(argv, "--full-gpu")
        g3 = _pair(argv, "--g3")
        g5 = _pair(argv, "--g5")
        g6 = _pair(argv, "--g6")
        resume = _pair(argv, "--resume")
        bank = _pair(argv, "--paired-bank")
        expected_commit = _pair(argv, "--expected-commit")
        expected_tree = _pair(argv, "--expected-tree")
        repo_root = _pair(argv, "--repo-root")
        out = _pair(argv, "--out")
        g7s = []
        for i, arg in enumerate(argv):
            if arg == "--g7" and i + 1 < len(argv):
                g7s.append(argv[i + 1])
        reviews = []
        for i, arg in enumerate(argv):
            if arg == "--review" and i + 1 < len(argv):
                reviews.append(argv[i + 1])
    except ValueError as e:
        print(f"CLI error: {e}", file=sys.stderr)
        return 2

    decision = evaluate_final_admission(
        rtm_path=rtm,
        implementation_manifest_path=manifest,
        test_impact_matrix_path=matrix,
        full_cpu_result=cpu,
        full_gpu_result=gpu,
        g3_result=g3,
        g5_result=g5,
        g6_result=g6,
        resume_result=resume,
        g7_results=g7s,
        paired_bank_result=bank,
        review_results=reviews,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
        repo_root=repo_root,
    )
    blob = {
        "schema_version": "r7-final-admission-v2",
        "engine_correct": decision.engine_correct,
        "training_admitted_for_g8": decision.training_admitted_for_g8,
        "failed_requirement_ids": list(decision.failed_requirement_ids),
        "unresolved_severities": list(decision.unresolved_severities),
        "missing_artifacts": list(decision.missing_artifacts),
        "stale_artifacts": list(decision.stale_artifacts),
        "review_failures": list(decision.review_failures),
        "verified_inputs": [
            asdict(v) for v in decision.verified_inputs
        ],
        "final_commit": decision.final_commit,
        "final_tree": decision.final_tree,
        # v2 (Zread P0-1): complete input inventory so the training
        # preflight can RE-EXECUTE the aggregator instead of trusting
        # hand-written booleans + a self-hash.
        "inputs": {
            "rtm_path": rtm,
            "implementation_manifest_path": manifest,
            "test_impact_matrix_path": matrix,
            "full_cpu_result": cpu,
            "full_gpu_result": gpu,
            "g3_result": g3,
            "g5_result": g5,
            "g6_result": g6,
            "resume_result": resume,
            "paired_bank_result": bank,
            "g7_results": list(g7s),
            "review_results": list(reviews),
            "expected_commit": expected_commit,
            "expected_tree": expected_tree,
        },
    }
    blob["content_hash"] = decision_content_hash(blob)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    if decision.engine_correct:
        print(
            f"ADMISSION PASS — engine_correct=true, "
            f"training_admitted_for_g8=true @ {decision.final_commit[:12]}"
        )
        return 0
    for label, items in (
        ("failed_requirement_ids", decision.failed_requirement_ids),
        ("unresolved_severities", decision.unresolved_severities),
        ("missing_artifacts", decision.missing_artifacts),
        ("stale_artifacts", decision.stale_artifacts),
        ("review_failures", decision.review_failures),
    ):
        if items:
            print(f"ADMISSION DENIED — {label}: {list(items)}",
                  file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
