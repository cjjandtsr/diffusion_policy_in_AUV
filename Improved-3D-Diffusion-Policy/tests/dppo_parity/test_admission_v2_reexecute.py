"""F1 (Zread P0-1, 2026-08-24): admission v2 — formal preflight re-executes
the aggregator instead of trusting a hand-written boolean JSON.

Zread finding (verified by independent repro): a forged decision JSON with
``verified_inputs=[]`` and a recomputed self-hash is accepted by
``validate_final_admission_for_training`` (FORGED_ACCEPTED=True).  v2 fixes
this by: (a) requiring schema r7-final-admission-v2 with a complete exact
``inputs`` inventory, (b) requiring non-empty ``verified_inputs``, and
(c) re-executing ``evaluate_final_admission()`` on the recorded inputs and
requiring the re-evaluated decision to match the JSON claims exactly.

Guide: R7-10 Step 5 contract — code change -> new commit + RTM REWORK +
targeted tests + reviewer re-sign.
"""

import json

import pytest

from test_final_admission_fail_closed import (
    REPO_ROOT,
    _git,
    _input_bundle,
    _write_json,
)
from diffusion_policy_3d.rl.dppo_uav import admission_aggregate as aa

SCHEMA_V2 = "r7-final-admission-v2"

# The exact input inventory a v2 decision JSON must carry (CLI writer
# contract; every key required, no extras tolerated by the re-execution).
# Keys are the ``evaluate_final_admission`` keyword names so the preflight
# can re-execute with a plain ``**kwargs`` expansion.
REQUIRED_INPUT_KEYS = (
    "rtm_path", "implementation_manifest_path", "test_impact_matrix_path",
    "full_cpu_result", "full_gpu_result",
    "g3_result", "g5_result", "g6_result", "resume_result",
    "paired_bank_result", "g7_results", "review_results",
    "expected_commit", "expected_tree",
)


def _forge_decision(tmp_path, *, verified_inputs=None, inputs=None,
                    schema=SCHEMA_V2):
    """Build a minimal decision JSON the way the CLI serializes it."""
    head = _git(REPO_ROOT, "rev-parse", "HEAD")
    tree = _git(REPO_ROOT, "rev-parse", "HEAD^{tree}")
    blob = {
        "schema_version": schema,
        "engine_correct": True,
        "training_admitted_for_g8": True,
        "failed_requirement_ids": [],
        "unresolved_severities": [],
        "missing_artifacts": [],
        "stale_artifacts": [],
        "review_failures": [],
        "verified_inputs": list(verified_inputs or []),
        "final_commit": head,
        "final_tree": tree,
    }
    if inputs is not None:
        blob["inputs"] = inputs
    blob["content_hash"] = aa.decision_content_hash(blob)
    return _write_json(tmp_path / "decision.json", blob)


def _inputs_from_paths(paths):
    """Record the evaluate input inventory exactly as the CLI will."""
    return {
        "rtm_path": str(paths["rtm"]),
        "implementation_manifest_path": str(paths["manifest"]),
        "test_impact_matrix_path": str(paths["matrix"]),
        "full_cpu_result": str(paths["cpu"]),
        "full_gpu_result": str(paths["gpu"]),
        "g3_result": str(paths["g3"]),
        "g5_result": str(paths["g5"]),
        "g6_result": str(paths["g6"]),
        "resume_result": str(paths["resume"]),
        "paired_bank_result": str(paths["bank"]),
        "g7_results": [str(paths["g7"])],
        "review_results": [str(p) for p in paths["reviews"]],
        "expected_commit": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "expected_tree": _git(REPO_ROOT, "rev-parse", "HEAD^{tree}"),
    }


def _decision_from_bundle(tmp_path):
    """Full valid bundle -> v2 decision JSON (mirrors CLI writer)."""
    paths, _ = _input_bundle(tmp_path)
    mod = aa
    from test_final_admission_fail_closed import _evaluate
    decision = _evaluate(paths)
    assert decision.engine_correct is True, decision.stale_artifacts
    blob = {
        "schema_version": SCHEMA_V2,
        "engine_correct": decision.engine_correct,
        "training_admitted_for_g8": decision.training_admitted_for_g8,
        "failed_requirement_ids": list(decision.failed_requirement_ids),
        "unresolved_severities": list(decision.unresolved_severities),
        "missing_artifacts": list(decision.missing_artifacts),
        "stale_artifacts": list(decision.stale_artifacts),
        "review_failures": list(decision.review_failures),
        "verified_inputs": [
            dict(v.__dict__) for v in decision.verified_inputs
        ],
        "final_commit": decision.final_commit,
        "final_tree": decision.final_tree,
        "inputs": _inputs_from_paths(paths),
    }
    blob["content_hash"] = aa.decision_content_hash(blob)
    return _write_json(tmp_path / "decision.json", blob), paths


# ═══════════════════════════════════════════════════════════════════
# RED #1 (Zread P0-1 repro): forged empty-inventory JSON must be rejected
# ═══════════════════════════════════════════════════════════════════

def _validate(p, **kw):
    """Preflight under test; clean-tree check is orthogonal to the v2
    re-execution logic and covered by its own test below."""
    kw.setdefault("require_clean_tree", False)
    return aa.validate_final_admission_for_training(str(p), str(REPO_ROOT),
                                                    **kw)


def test_forged_empty_verified_inputs_rejected(tmp_path):
    """Hand-written JSON with empty verified_inputs and no inputs inventory
    must fail — this exact JSON was accepted before v2."""
    p = _forge_decision(tmp_path)
    problems = _validate(p)
    assert problems, "forged empty-inventory decision was accepted"


def test_forged_v1_schema_rejected(tmp_path):
    """Pre-v2 schema has no inputs inventory -> fail-closed reject."""
    p = _forge_decision(tmp_path, schema="r7-final-admission-v1")
    problems = _validate(p)
    assert problems


def test_forged_incomplete_inputs_rejected(tmp_path):
    """inputs inventory present but missing keys -> reject."""
    paths, _ = _input_bundle(tmp_path)
    inputs = _inputs_from_paths(paths)
    del inputs["g6_result"]
    p = _forge_decision(tmp_path, inputs=inputs)
    problems = _validate(p)
    assert problems, "incomplete inputs inventory was accepted"


def test_forged_extra_input_key_rejected(tmp_path):
    """Unknown key smuggled into inputs -> reject (exact inventory)."""
    paths, _ = _input_bundle(tmp_path)
    inputs = _inputs_from_paths(paths)
    inputs["evil_result"] = "/tmp/evil.json"
    p = _forge_decision(tmp_path, inputs=inputs)
    problems = _validate(p)
    assert problems, "extra input key was accepted"


# ═══════════════════════════════════════════════════════════════════
# GREEN path: real bundle preflight passes
# ═══════════════════════════════════════════════════════════════════

def test_full_bundle_preflight_passes(tmp_path):
    """A genuine CLI-style v2 decision re-executes to PASS."""
    p, _ = _decision_from_bundle(tmp_path)
    problems = _validate(p)
    assert problems == [], problems


def test_tampered_input_rejected(tmp_path):
    """After signing, tampering any input file must fail the re-execution."""
    p, paths = _decision_from_bundle(tmp_path)
    matrix_path = paths["matrix"]
    matrix = json.loads(matrix_path.read_text())
    matrix["unknown_or_ambiguous_dependencies"] = ["x"]
    matrix_path.write_text(json.dumps(matrix))
    problems = _validate(p)
    assert problems, "tampered matrix input was accepted"


def test_stale_json_rejected(tmp_path):
    """JSON claims PASS but the recorded inputs no longer yield PASS
    (e.g. input deleted after signing) -> must fail closed."""
    p, paths = _decision_from_bundle(tmp_path)
    paths["cpu"].unlink()
    problems = _validate(p)
    assert problems, "stale decision was accepted after input deletion"


def test_verified_inputs_deleted_from_json_rejected(tmp_path):
    """verified_inputs emptied AFTER signing (hash recomputed) -> reject."""
    p, _ = _decision_from_bundle(tmp_path)
    blob = json.loads(p.read_text())
    blob["verified_inputs"] = []
    blob["content_hash"] = aa.decision_content_hash(blob)
    p.write_text(json.dumps(blob))
    problems = _validate(p)
    assert problems, "emptied verified_inputs was accepted"


def test_dirty_tree_rejected(tmp_path):
    """require_clean_tree=True (production default) rejects a dirty tree."""
    marker = REPO_ROOT / ".r7_v2_dirty_tree_probe"
    marker.write_text("probe")
    try:
        dummy = _write_json(tmp_path / "dummy.json", {})
        problems = aa.validate_final_admission_for_training(
            str(dummy), str(REPO_ROOT))
    finally:
        marker.unlink()
    assert any("clean" in str(p) for p in problems)
