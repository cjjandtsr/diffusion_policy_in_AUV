"""R7-7 Step 7: canonical-train preflight consumes the signed final
admission JSON.

The formal launcher must read the decision produced by the ONLY
canonical signer (scripts/r7_final_admission.py) BEFORE any model
construction: engine_correct == true, training_admitted_for_g8 ==
true, recomputed content hash matches, final commit/tree == the
current clean worktree, and every bound artifact still exists and
hash-matches.  A missing --final_admission is a CLI usage error
(exit 2, mirroring the --prereg contract); any validation problem is
a preflight failure (exit 1).  A config file can never hand-write
``engine_pass`` and an env var can never bypass the check.

The preflight function takes ``repo_root`` as a parameter, so the
positive path runs against a real throwaway git repo (init + commit)
instead of depending on the workspace being clean mid-task.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DECISION_SCHEMA = "r7-final-admission-v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        check=True,
    )
    return out.stdout.strip()


def _fresh_repo(tmp_path: Path) -> Path:
    """A real throwaway git repo with one commit and a clean tree."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "r7-test@local")
    _git(repo, "config", "user.name", "r7-test")
    (repo / "tracked.txt").write_text("x")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _decision(tmp_path: Path, repo: Path) -> dict:
    """A valid signed decision: artifacts are real files under tmp_path,
    final identity = the fresh repo's HEAD/tree, content hash recomputed
    independently (not via the production helper)."""
    art = tmp_path / "evidence.json"
    art.write_text(json.dumps({"row": "R7-OFFICIAL-DIRECT"}))
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    decision = {
        "schema_version": DECISION_SCHEMA,
        "engine_correct": True,
        "training_admitted_for_g8": True,
        "failed_requirement_ids": [],
        "unresolved_severities": [],
        "missing_artifacts": [],
        "stale_artifacts": [],
        "review_failures": [],
        "verified_inputs": [{
            "kind": "R7-OFFICIAL-DIRECT",
            "path": str(art),
            "sha256": _sha256_file(art),
            "result_commit": head,
            "result_tree": tree,
            "validated_source_set_hash": "ab" * 32,
        }],
        "final_commit": head,
        "final_tree": tree,
    }
    body = {k: v for k, v in decision.items() if k != "content_hash"}
    decision["content_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return decision


def _write_decision(tmp_path: Path, repo: Path, **overrides) -> Path:
    decision = _decision(tmp_path, repo)
    decision.update(overrides)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision))
    return path


# ═══════════════════════════════════════════════════════════════════
# positive path — a signed decision on a clean final-commit tree admits
# ═══════════════════════════════════════════════════════════════════

def test_valid_signed_decision_passes(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
        validate_final_admission_for_training,
    )

    repo = _fresh_repo(tmp_path)
    path = _write_decision(tmp_path, repo)
    problems = validate_final_admission_for_training(str(path), str(repo))
    assert problems == [], f"expected clean preflight, got {problems}"


# ═══════════════════════════════════════════════════════════════════
# negative path — every violated axis must be named precisely
# ═══════════════════════════════════════════════════════════════════

def test_engine_correct_false_rejected(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
        validate_final_admission_for_training,
    )

    repo = _fresh_repo(tmp_path)
    path = _write_decision(tmp_path, repo, engine_correct=False)
    problems = validate_final_admission_for_training(str(path), str(repo))
    assert any("engine_correct" in p for p in problems)


def test_training_admitted_false_rejected(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
        validate_final_admission_for_training,
    )

    repo = _fresh_repo(tmp_path)
    path = _write_decision(tmp_path, repo, training_admitted_for_g8=False)
    problems = validate_final_admission_for_training(str(path), str(repo))
    assert any("training_admitted_for_g8" in p for p in problems)


def test_content_hash_tamper_rejected(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
        validate_final_admission_for_training,
    )

    repo = _fresh_repo(tmp_path)
    path = _write_decision(tmp_path, repo, content_hash="00" * 32)
    problems = validate_final_admission_for_training(str(path), str(repo))
    assert any("content_hash" in p for p in problems)


def test_final_commit_drift_rejected(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
        validate_final_admission_for_training,
    )

    repo = _fresh_repo(tmp_path)
    path = _write_decision(tmp_path, repo, final_commit="11" * 40)
    problems = validate_final_admission_for_training(str(path), str(repo))
    assert any("final_commit" in p for p in problems)


def test_dirty_tree_rejected(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
        validate_final_admission_for_training,
    )

    repo = _fresh_repo(tmp_path)
    (repo / "uncommitted.txt").write_text("dirty")
    path = _write_decision(tmp_path, repo)
    problems = validate_final_admission_for_training(str(path), str(repo))
    assert any("clean" in p for p in problems)


def test_bound_artifact_missing_rejected(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
        validate_final_admission_for_training,
    )

    repo = _fresh_repo(tmp_path)
    decision = _decision(tmp_path, repo)
    decision["verified_inputs"][0]["path"] = str(
        tmp_path / "gone.json")
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision))
    problems = validate_final_admission_for_training(str(path), str(repo))
    assert any("missing" in p for p in problems)


def test_bound_artifact_hash_drift_rejected(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
        validate_final_admission_for_training,
    )

    repo = _fresh_repo(tmp_path)
    decision = _decision(tmp_path, repo)
    decision["verified_inputs"][0]["sha256"] = "22" * 32
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision))
    problems = validate_final_admission_for_training(str(path), str(repo))
    assert any("drift" in p for p in problems)


def test_unreadable_decision_rejected(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
        validate_final_admission_for_training,
    )

    repo = _fresh_repo(tmp_path)
    problems = validate_final_admission_for_training(
        str(tmp_path / "no.json"), str(repo))
    assert problems, "a missing decision JSON must never preflight-clean"


# ═══════════════════════════════════════════════════════════════════
# launcher entry — --stage formal without --final_admission is exit 2
# (mirrors the --prereg contract; the message must name the flag)
# ═══════════════════════════════════════════════════════════════════

def test_formal_stage_requires_final_admission(tmp_path):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    out = subprocess.run(
        [sys.executable, "scripts/train_dppo_uav_official.py",
         "--stage", "formal", "--output_dir", str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert out.returncode == 2, (
        "formal launch without --final_admission must be a CLI-level "
        f"contract error (exit 2); got {out.returncode}: "
        f"{out.stderr[-400:]}"
    )
    assert "--final_admission" in out.stderr, (
        "the CLI error must name the missing --final_admission flag"
    )


def test_validate_helper_bound_to_decision_schema(tmp_path):
    """The preflight consumer must not accept arbitrary JSON — the schema
    version is part of the contract."""
    from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
        validate_final_admission_for_training,
    )

    repo = _fresh_repo(tmp_path)
    path = tmp_path / "fake.json"
    path.write_text(json.dumps({"engine_correct": True,
                                "training_admitted_for_g8": True}))
    problems = validate_final_admission_for_training(str(path), str(repo))
    assert any("schema_version" in p for p in problems)
