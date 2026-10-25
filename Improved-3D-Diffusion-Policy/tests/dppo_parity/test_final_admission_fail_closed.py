"""RED #7/#8 (R7-ADMISSION): the one fail-closed global admission aggregator.

Guide Task R7-7 (Guide_A_OfficialCore_R7_Remediation_v1_0.md, lines 1263-1426):
``admission_aggregate.evaluate_final_admission`` is the SINGLE admission
decision; DEFAULT-DENY — every input is validated explicitly before any bool
turns true; exceptions / JSON parse errors / missing keys / unknown schemas /
empty lists / NaN / missing paths / hash mismatch / stale identity / source-set
stale / unexplainable commit diffs all return FAIL.

The R7-0 RED tests (4 original cases) used the pre-v1.0 draft interface
``evaluate_admission(dict)``; the FIXED guide v1.0 specifies
``evaluate_final_admission(*, rtm_path, implementation_manifest_path, ...)``.
Following the R7-6 precedent (fixtures adapted to the production wiring), the
4 fail-closed semantics are preserved verbatim in intent, exercised through the
canonical interface. Step 8's anti-cheat matrix is added here: deleting or
tampering EVERY input must yield engine_correct=false, training_admitted_
for_g8=false, precisely-named reason lists, and a non-zero process exit code.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_AGG_MODULE = "diffusion_policy_3d.rl.dppo_uav.admission_aggregate"
_CLI_MODULE = "scripts.r7_final_admission"

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── 14 exact R7 requirement IDs (guide R7-0 Step 7; frozen set) ─────
REQUIRED_IDS = (
    "R7-OFFICIAL-DIRECT",
    "R7-B128",
    "R7-RESUME",
    "R7-ADMISSION",
    "R7-NEXT-VALUE",
    "R7-PC-WINDOW",
    "R7-G3-FRESH",
    "R7-CAP-BANK",
    "R7-CLOSURE",
    "R7-G7",
    "R7-LORA-AUDIT",
    "R7-RUNTIME-HASH",
    "R7-NUMERICS",
    "R7-EVIDENCE",
)

SEVERITY_BY_ID = {
    "R7-OFFICIAL-DIRECT": "P1",
    "R7-B128": "P0",
    "R7-RESUME": "P0",
    "R7-ADMISSION": "P0",
    "R7-NEXT-VALUE": "P1",
    "R7-PC-WINDOW": "P1",
    "R7-G3-FRESH": "P1",
    "R7-CAP-BANK": "P1",
    "R7-CLOSURE": "P1",
    "R7-G7": "P1",
    "R7-LORA-AUDIT": "P2",
    "R7-RUNTIME-HASH": "P2",
    "R7-NUMERICS": "P2",
    "R7-EVIDENCE": "P2",
}

SOURCE_SET_HASH = "ab" * 32  # consistent fake source-set hash for fixtures
REVIEWER = "R7 Independent Reviewer (non-author)"


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        check=True,
    )
    return out.stdout.strip()


def _load_aggregator():
    try:
        import importlib
        return importlib.import_module(_AGG_MODULE)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"admission_aggregate missing — no fail-closed global admission"
            f" aggregator exists: {exc}"
        )


def _load_cli():
    try:
        import importlib
        return importlib.import_module(_CLI_MODULE)
    except ModuleNotFoundError as exc:
        pytest.fail(f"r7_final_admission missing — no canonical signer: {exc}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════════
# canonical-input bundle (all files written under tmp_path)
# ═══════════════════════════════════════════════════════════════════

def _evidence_paths(tmp_path, rows):
    """One evidence file per row, absolute path with inline full sha."""
    out = {}
    for rid in rows:
        p = tmp_path / "evidence" / f"{rid}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"row": rid, "blob": "x" * 8}))
        out[rid] = f"{p} (sha256 {_sha256_file(p)})"
    return out


def _rtm_rows(head, tree, reviewer, evidence_paths, status="PASS",
              review_status="PASS", result_commit=None, reviewer_override=None):
    rows = []
    for rid in REQUIRED_IDS:
        rows.append({
            "id": rid,
            "source": "audit-20260822",
            "finding": f"finding for {rid}",
            "severity": SEVERITY_BY_ID[rid],
            "status": status,
            "code_targets": ["diffusion_policy_3d/rl/dppo_uav/checkpoint.py"],
            "tests": [f"tests/dppo_parity/test_{rid.lower().replace('-', '_')}.py"],
            "evidence_paths": [evidence_paths[rid]],
            "result_commit": result_commit or head,
            "reviewer": reviewer_override or reviewer,
            "review_status": review_status,
        })
    return rows


def _manifest_bindings(head, tree, evidence_paths):
    return {
        rid: {
            "result_commit": head,
            "result_tree": tree,
            "validated_source_set_hash": SOURCE_SET_HASH,
            "artifacts": [{"path": evidence_paths[rid].split(" (")[0],
                           "sha256": _sha256_text(
                               json.dumps({"row": rid, "blob": "x" * 8}))}],
        }
        for rid in REQUIRED_IDS
    }


def _cpu_log_binding(tmp_path, name):
    """log_path/log_sha256 kwargs for a REAL raw suite log file."""
    p = tmp_path / f"{name}_suite.log"
    p.write_bytes(f"{name} full suite log\n".encode())
    return {"log_path": str(p), "log_sha256": _sha256_file(p)}


def _resume_log_binding(tmp_path):
    """log_path/log_sha256 kwargs for a REAL GPU resume pytest log."""
    p = tmp_path / "r7_resume_gpu.log"
    content = ("GPU resume determinism\n"
               + GPU_RESUME_NODEID + " PASSED\n").encode()
    p.write_bytes(content)
    return {"log_path": str(p), "log_sha256": _sha256_file(p)}


def _test_result(commit, tree, *, exit_code=0, skips=None,
                 log_path=None, log_sha256=None, producer="pytest"):
    return {
        "schema_version": "r7-test-result-v1",
        "command": "pytest tests/dppo_parity -m 'not gpu' -ra",
        "started_at": "2026-08-24T00:00:00Z",
        "finished_at": "2026-08-24T01:00:00Z",
        "exit_code": exit_code,
        "collected": 600,
        "passed": 600 if exit_code == 0 else 599,
        "failed": 0 if exit_code == 0 else 1,
        "error": 0,
        "skipped": len(skips or []),
        "xfail": 0,
        "xpass": 0,
        "deselected": 0,
        "commit": commit,
        "tree": tree,
        "validated_source_set_hash": SOURCE_SET_HASH,
        "producer": producer,
        "log_path": log_path or "r7_cpu_suite.log",
        "log_sha256": log_sha256 or ("cd" * 32),
        "skips": skips or [],
    }


def _g3_result(commit, tree, *, invalid_window=0):
    return {
        "schema_version": "r7-g3-result-v1",
        "command": "python scripts/g3_10k_reset_gate_audit.py",
        "n_resets": 10000,
        "invalid_window": invalid_window,
        "no_ray": 0,
        "runtime_hash_mismatch": 0,
        "source_hash_missing": 0,
        "source_hash_mismatch": 0,
        "eval_leakage": 0,
        "unknown_role": 0,
        "unvalidated_fallback": 0,
        "admission_blocked": 0,
        "manifest_hash_mismatch": 0,
        "role_proportion_error": 0.0,
        "reachability_pass_ratio": 1.0,
        "gates": {f"g{i}": "PASS" for i in range(15)},
        "commit": commit,
        "tree": tree,
        "validated_source_set_hash": SOURCE_SET_HASH,
    }


def _g5_result(commit, tree):
    return {
        "schema_version": "r7-g5-result-v1",
        "n_episodes": 32,
        "b1_identity": {"max_abs_diff": 0.0,
                        "ratio_mean": 1.0,
                        "threshold": 1e-5},
        "b128_identity": {"max_abs_diff": 0.0},  # informational (R7-3 ruling)
        "lora_base_before_after": {"base_unchanged": True,
                                   "lora_updated": True,
                                   "lora_checksum": "ef" * 32},
        "numerical_mode": {"deterministic_algorithms": True,
                           "allow_tf32_matmul": False,
                           "allow_tf32_cudnn": False},
        "commit": commit,
        "tree": tree,
        "validated_source_set_hash": SOURCE_SET_HASH,
    }


def _g6_result(commit, tree):
    return {
        "schema_version": "r7-g6-result-v1",
        "actor_grad": {"nonzero": True, "norm": 0.5},
        "critic_grad": {"nonzero": True, "norm": 1.0},
        "lora_base_update": {"base_unchanged": True, "lora_updated": True},
        "unexecuted_grad": {"count": 0},
        "ratio": {"finite": True, "mean": 1.0, "std": 0.1},
        "optimizer_step": {"occurred": True, "step_count_delta": 1},
        "resume": {"worked": True},
        "commit": commit,
        "tree": tree,
        "validated_source_set_hash": SOURCE_SET_HASH,
    }


GPU_RESUME_NODEID = (
    "tests/dppo_parity/test_resume_determinism_gpu_r7.py"
    "::test_same_seed_resume_determinism_gpu_r7"
)


def _resume_result(*, log_path=None, log_sha256=None, producer="pytest"):
    return {
        "schema_version": "r7-compare-v1",
        "pass": True,
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
        "gpu_test_nodeids": [GPU_RESUME_NODEID],
        "log_path": log_path or "r7_resume_gpu.log",
        "log_sha256": log_sha256 or ("cd" * 32),
    }


def _g7_result(commit, tree, *, n_seeds=2, transitions=1000):
    return {
        "schema_version": "r7-g7-result-v1",
        "seeds": [101, 202][:n_seeds],
        "transitions_per_seed": transitions,
        "updates_per_seed": 30,
        "hard_faults": 0,
        "per_seed": {str(s): {"transitions": transitions, "updates": 30,
                              "releases": 1} for s in [101, 202][:n_seeds]},
        "commit": commit,
        "tree": tree,
        "validated_source_set_hash": SOURCE_SET_HASH,
    }


def _bank_result():
    return {
        "schema_version": "r7-paired-bank-v1",
        "bank_size": 100,
        "membership_hash": "ab" * 32,
        "aggregate": {
            "capability_counts": {"anticipatory": 30, "full_space": 25,
                                  "dense_recovery": 22},
            "opportunity": {"anticipatory": 30, "full_space": 25,
                            "dense_recovery": 22},
            "exposure": {"anticipatory": 30, "full_space": 25,
                         "dense_recovery": 22},
            "role_counts": {"anticipatory": 30, "dense_recovery": 25,
                            "support_expansion": 25, "yaw_lateral_recovery": 20},
            "no_opportunity_episodes": 5,
        },
    }


def _review(kind, head, tree, *, reviewer=REVIEWER, status="PASS",
            bound_commit=None, bound_source_set_hash=None):
    return {
        "schema_version": "r7-review-v1",
        "kind": kind,
        "status": status,
        "reviewer": reviewer,
        "bound_commit": bound_commit or head,
        "bound_tree": tree,
        "bound_source_set_hash": bound_source_set_hash or SOURCE_SET_HASH,
        "artifact_hashes": [],
        "reviewed_at": "2026-08-24T00:00:00Z",
    }


REVIEW_KINDS = (
    "R7_IMPLEMENTATION_CODE_REVIEW",
    "R1_R4_REQUIREMENT_REPLAY_AUDIT",
    "ZREAD_OR_EQUIVALENT_FINAL_REVIEW",
    "FINAL_RERUN_REVIEW",
)


def _input_bundle(tmp_path):
    """Write the full valid bundle; return (paths, contents) where contents
    lets tests mutate a specific input before rewriting it."""
    head = _git(REPO_ROOT, "rev-parse", "HEAD")
    tree = _git(REPO_ROOT, "rev-parse", "HEAD^{tree}")

    ev_paths = _evidence_paths(tmp_path, REQUIRED_IDS)
    contents = {
        "rtm": {"schema_version": "r7-rtm-v1",
                "rows": _rtm_rows(head, tree, REVIEWER, ev_paths)},
        "manifest": {"schema_version": "r7-implementation-manifest-v1",
                     "baseline_commit": "25ae25970238b841f5f09451d34d306e725fcac9",
                     "bindings": _manifest_bindings(head, tree, ev_paths)},
        "matrix": {"schema_version": "r7-test-impact-v1",
                   "baseline_commit": "25ae25970238b841f5f09451d34d306e725fcac9",
                   "candidate_code_commit": head,
                   "candidate_code_tree": tree,
                   "validated_source_set_hash": SOURCE_SET_HASH,
                   "source_set": {
                       "member_count": 280,
                       "definition": [
                           "diffusion_policy_3d/**/*.py",
                           "scripts/*.py",
                           "tests/dppo_parity/*.py",
                           "diffusion_policy_3d/config/**/*",
                           "third_party/dppo_official_cc7234a/UPSTREAM_COMMIT",
                           "third_party/dppo_official_cc7234a/SHA256SUMS",
                           "BASELINE_MANIFEST.json",
                       ]},
                   "changed_files": [],
                   "required_checks": ["TARGETED_CPU"],
                   "not_invalidated_checks": [],
                   "unknown_or_ambiguous_dependencies": [],
                   "generated_at": "2026-08-24T00:00:00Z"},
        # F2 (Zread P0-2): suite results must bind to a REAL raw log —
        # write one per suite so every valid bundle admits.
        "cpu": _test_result(head, tree, **_cpu_log_binding(tmp_path, "cpu")),
        "gpu": _test_result(head, tree, **_cpu_log_binding(tmp_path, "gpu")),
        "g3": _g3_result(head, tree),
        "g5": _g5_result(head, tree),
        "g6": _g6_result(head, tree),
        # F3: resume gate must bind to the REAL GPU pytest log that
        # contains the GPU nodeid — write it so every valid bundle admits.
        "resume": _resume_result(**_resume_log_binding(tmp_path)),
        "g7": _g7_result(head, tree),
        "bank": _bank_result(),
        "reviews": {k: _review(k, head, tree) for k in REVIEW_KINDS},
    }
    paths = {}
    for name, content in contents.items():
        if name == "reviews":
            paths[name] = [
                _write_json(tmp_path / f"review_{i}.json", c)
                for i, c in enumerate(content.values())
            ]
            continue
        if name == "rtm":
            paths[name] = _write_json(tmp_path / "rtm.json", content)
            continue
        if name == "manifest":
            paths[name] = _write_json(tmp_path / "manifest.json", content)
            continue
        if name == "matrix":
            paths[name] = _write_json(tmp_path / "matrix.json", content)
            continue
        paths[name] = _write_json(tmp_path / f"{name}.json", content)
    return paths, contents


def _write_json(path: Path, content) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    return path


def _rewrite(path: Path, content) -> None:
    path.write_text(json.dumps(content))


def _evaluate(paths):
    mod = _load_aggregator()
    return mod.evaluate_final_admission(
        rtm_path=paths["rtm"],
        implementation_manifest_path=paths["manifest"],
        test_impact_matrix_path=paths["matrix"],
        full_cpu_result=paths["cpu"],
        full_gpu_result=paths["gpu"],
        g3_result=paths["g3"],
        g5_result=paths["g5"],
        g6_result=paths["g6"],
        resume_result=paths["resume"],
        g7_results=[paths["g7"]],
        paired_bank_result=paths["bank"],
        review_results=paths["reviews"],
        expected_commit=_git(REPO_ROOT, "rev-parse", "HEAD"),
        expected_tree=_git(REPO_ROOT, "rev-parse", "HEAD^{tree}"),
        repo_root=REPO_ROOT,
    )


def _cli_argv(paths, out):
    return [
        "--rtm", str(paths["rtm"]),
        "--implementation-manifest", str(paths["manifest"]),
        "--test-impact-matrix", str(paths["matrix"]),
        "--full-cpu", str(paths["cpu"]),
        "--full-gpu", str(paths["gpu"]),
        "--g3", str(paths["g3"]),
        "--g5", str(paths["g5"]),
        "--g6", str(paths["g6"]),
        "--resume", str(paths["resume"]),
        "--g7", str(paths["g7"]),
        "--paired-bank", str(paths["bank"]),
        "--review", str(paths["reviews"][0]),
        "--review", str(paths["reviews"][1]),
        "--review", str(paths["reviews"][2]),
        "--review", str(paths["reviews"][3]),
        "--expected-commit", _git(REPO_ROOT, "rev-parse", "HEAD"),
        "--expected-tree", _git(REPO_ROOT, "rev-parse", "HEAD^{tree}"),
        "--repo-root", str(REPO_ROOT),
        "--out", str(out),
    ]


# ═══════════════════════════════════════════════════════════════════
# positive path — the full bundle admits
# ═══════════════════════════════════════════════════════════════════

def test_full_valid_bundle_admits(tmp_path):
    paths, _ = _input_bundle(tmp_path)
    mod = _load_aggregator()
    d = mod.evaluate_final_admission(
        rtm_path=paths["rtm"],
        implementation_manifest_path=paths["manifest"],
        test_impact_matrix_path=paths["matrix"],
        full_cpu_result=paths["cpu"],
        full_gpu_result=paths["gpu"],
        g3_result=paths["g3"],
        g5_result=paths["g5"],
        g6_result=paths["g6"],
        resume_result=paths["resume"],
        g7_results=[paths["g7"]],
        paired_bank_result=paths["bank"],
        review_results=paths["reviews"],
        expected_commit=_git(REPO_ROOT, "rev-parse", "HEAD"),
        expected_tree=_git(REPO_ROOT, "rev-parse", "HEAD^{tree}"),
        repo_root=REPO_ROOT,
    )
    assert d.engine_correct is True
    assert d.training_admitted_for_g8 is True
    assert d.failed_requirement_ids == ()
    assert d.missing_artifacts == ()
    assert d.stale_artifacts == ()
    assert d.unresolved_severities == ()
    assert d.review_failures == ()
    assert len(d.verified_inputs) >= 14
    assert d.final_commit == _git(REPO_ROOT, "rev-parse", "HEAD")


# ═══════════════════════════════════════════════════════════════════
# the 4 original R7-0 fail-closed semantics (canonical interface)
# ═══════════════════════════════════════════════════════════════════

def test_resume_pass_false_rejects_admission(tmp_path):
    """RED #7: same-seed resume summary pass=false must fail closed.

    REPORT does not admit — the R7-6 summary's pass flag is read as a raw
    field, never trusted as a status string.
    """
    paths, c = _input_bundle(tmp_path)
    c["resume"]["pass"] = False
    _rewrite(paths["resume"], c["resume"])
    d = _evaluate(paths)
    assert d.engine_correct is False
    assert d.training_admitted_for_g8 is False
    assert any("resume" in str(s).lower() for s in d.stale_artifacts)


def test_rtm_implemented_awaiting_review_rejects_admission(tmp_path):
    """RED #8: an RTM row still IMPLEMENTED_AWAITING_REVIEW must fail
    closed — only status == PASS rows are admissible at signing time."""
    paths, c = _input_bundle(tmp_path)
    c["rtm"]["rows"][0]["status"] = "IMPLEMENTED_AWAITING_REVIEW"
    _rewrite(paths["rtm"], c["rtm"])
    d = _evaluate(paths)
    assert d.engine_correct is False
    assert c["rtm"]["rows"][0]["id"] in d.failed_requirement_ids
    assert c["rtm"]["rows"][0]["severity"] in d.unresolved_severities


def test_default_deny_on_empty_inputs():
    """No inputs -> no admission (default-deny; every missing path is
    reported, nothing raises)."""
    mod = _load_aggregator()
    d = mod.evaluate_final_admission()
    assert d.engine_correct is False
    assert d.training_admitted_for_g8 is False
    assert len(d.missing_artifacts) > 0


def test_rtm_row_schema_enforced(tmp_path):
    """Every RTM row must carry the full R7-0 Step 7 field set."""
    paths, c = _input_bundle(tmp_path)
    c["rtm"]["rows"] = [{"id": "R7-B128", "status": "COMPLETE"}]
    _rewrite(paths["rtm"], c["rtm"])
    d = _evaluate(paths)
    assert d.engine_correct is False
    assert "R7-B128" in d.failed_requirement_ids
    assert any("schema" in str(r).lower() for r in d.failed_requirement_ids
               for r in [c["rtm"]["rows"][0]["id"]] + list(d.stale_artifacts)
               ) or any("schema" in str(r).lower() for r in d.stale_artifacts)


# ═══════════════════════════════════════════════════════════════════
# Step 8 anti-cheat: deleting/tampering EVERY input denies
# ═══════════════════════════════════════════════════════════════════

def _case(name, mutate, expect_list, expect_fragment):
    return pytest.param(mutate, expect_list, expect_fragment, id=name)


ANTI_CHEAT_CASES = [
    # ── RTM 行 ─────────────────────────────────────────────────────
    _case("rtm_row_deleted", lambda c: c["rtm"]["rows"].pop(3),
          "failed_requirement_ids", "R7-ADMISSION"),
    _case("rtm_row_duplicated", lambda c: c["rtm"]["rows"].append(
        dict(c["rtm"]["rows"][0])), "failed_requirement_ids", "duplicate"),
    _case("rtm_row_status_open", lambda c: c["rtm"]["rows"][5].update(
        status="OPEN"), "failed_requirement_ids", "R7-PC-WINDOW"),
    _case("rtm_row_severity_missing", lambda c: c["rtm"]["rows"][0].pop(
        "severity"), "failed_requirement_ids", "R7-OFFICIAL-DIRECT"),
    _case("rtm_evidence_path_relative", lambda c: c["rtm"]["rows"][0].update(
        evidence_paths=["relative/path.json (sha256 " + "ab" * 32 + ")"]),
          "failed_requirement_ids", "absolute"),
    _case("rtm_evidence_missing", lambda c: c["rtm"]["rows"][0].update(
        evidence_paths=[str(Path("/nonexistent/ev.json")) + " (sha256 " +
                        "ab" * 32 + ")"]),
          "failed_requirement_ids", "R7-OFFICIAL-DIRECT"),
    _case("rtm_evidence_hash_drift", lambda c: c["rtm"]["rows"][0].update(
        evidence_paths=[c["rtm"]["rows"][0]["evidence_paths"][0].replace(
            "sha256 ", "sha256 ff")]),
          "failed_requirement_ids", "R7-OFFICIAL-DIRECT"),
    _case("rtm_evidence_no_annotation", lambda c: c["rtm"]["rows"][0].update(
        evidence_paths=[c["rtm"]["rows"][0]["evidence_paths"][0].split(
            " (")[0]]),
          "failed_requirement_ids", "sha256"),
    _case("rtm_reviewer_is_implementer", lambda c: c["rtm"]["rows"][0].update(
        reviewer=_git(REPO_ROOT, "log", "-1", "--format=%an")),
          "failed_requirement_ids", "implementer"),
    _case("rtm_review_status_pending", lambda c: c["rtm"]["rows"][0].update(
        review_status="PENDING"), "failed_requirement_ids", "R7-OFFICIAL-DIRECT"),
    # ── manifest ───────────────────────────────────────────────────
    _case("manifest_binding_missing", lambda c: c["manifest"]["bindings"].pop(
        "R7-RESUME"), "failed_requirement_ids", "R7-RESUME"),
    _case("manifest_commit_mismatch", lambda c: c["manifest"]["bindings"][
        "R7-RESUME"].update(result_commit="11" * 20),
          "failed_requirement_ids", "R7-RESUME"),
    _case("manifest_tree_mismatch", lambda c: c["manifest"]["bindings"][
        "R7-RESUME"].update(result_tree="22" * 20),
          "failed_requirement_ids", "R7-RESUME"),
    _case("manifest_source_set_stale", lambda c: c["manifest"]["bindings"][
        "R7-RESUME"].update(validated_source_set_hash="33" * 32),
          "stale_artifacts", "source-set"),
    _case("manifest_artifact_hash_drift", lambda c: c["manifest"]["bindings"][
        "R7-RESUME"]["artifacts"][0].update(sha256="44" * 32),
          "stale_artifacts", "R7-RESUME"),
    # ── impact matrix ──────────────────────────────────────────────
    _case("matrix_candidate_commit_mismatch", lambda c: c["matrix"].update(
        candidate_code_commit="55" * 20), "stale_artifacts", "matrix"),
    _case("matrix_source_set_hash_drift", lambda c: c["matrix"].update(
        validated_source_set_hash="66" * 32), "stale_artifacts", "source-set"),
    _case("matrix_unknown_dependencies", lambda c: c["matrix"].update(
        unknown_or_ambiguous_dependencies=["unresolved-dep"]),
          "stale_artifacts", "unknown"),
    # ── CPU / GPU ──────────────────────────────────────────────────
    _case("cpu_exit_code_nonzero", lambda c: c["cpu"].update(exit_code=1),
          "stale_artifacts", "exit_code"),
    _case("cpu_failed_nonzero", lambda c: c["cpu"].update(failed=2,
                                                          passed=598),
          "stale_artifacts", "failed"),
    _case("cpu_unknown_skip", lambda c: c["cpu"].update(skips=[
        {"nodeid": "t.py::x", "reason": "mystery skip", "owner": "x",
         "unrelated_proof": "p"}]), "stale_artifacts", "skip"),
    _case("cpu_missing_field", lambda c: c["cpu"].pop("collected"),
          "stale_artifacts", "schema"),
    _case("gpu_missing_file", lambda c: None, "missing_artifacts", "gpu"),
    # ── gates ──────────────────────────────────────────────────────
    _case("g3_invalid_window", lambda c: c["g3"].update(invalid_window=1),
          "stale_artifacts", "invalid_window"),
    _case("g3_short_resets", lambda c: c["g3"].update(n_resets=999),
          "stale_artifacts", "n_resets"),
    _case("g5_b1_identity_drift", lambda c: c["g5"]["b1_identity"].update(
        max_abs_diff=0.02), "stale_artifacts", "identity"),
    _case("g5_base_changed", lambda c: c["g5"]["lora_base_before_after"].update(
        base_unchanged=False), "stale_artifacts", "base"),
    _case("g6_actor_grad_zero", lambda c: c["g6"]["actor_grad"].update(
        nonzero=False), "stale_artifacts", "actor"),
    _case("g6_ratio_nonfinite", lambda c: c["g6"]["ratio"].update(
        finite=False), "stale_artifacts", "ratio"),
    _case("resume_ckpt_diffs", lambda c: c["resume"].update(ckpt_diffs=1),
          "stale_artifacts", "ckpt_diffs"),
    _case("g7_single_seed", lambda c: c["g7"].update(
        seeds=[101], per_seed={"101": {"transitions": 1000, "updates": 30,
                                       "releases": 1}}),
          "stale_artifacts", "seeds"),
    _case("g7_unequal_transitions", lambda c: c["g7"].update(
        transitions_per_seed=999,
        per_seed={"101": {"transitions": 1000, "updates": 30, "releases": 1},
                  "202": {"transitions": 999, "updates": 30, "releases": 1}}),
          "stale_artifacts", "transitions"),
    _case("g7_hard_fault", lambda c: c["g7"].update(hard_faults=1),
          "stale_artifacts", "hard_fault"),
    _case("bank_opportunity_low", lambda c: c["bank"]["aggregate"][
        "opportunity"].update(anticipatory=19), "stale_artifacts",
          "opportunity"),
    _case("bank_exposure_zero", lambda c: c["bank"]["aggregate"][
        "exposure"].update(dense_recovery=0), "stale_artifacts", "exposure"),
    # ── reviews ────────────────────────────────────────────────────
    _case("review_missing_kind", lambda c: c["reviews"].pop(
        "ZREAD_OR_EQUIVALENT_FINAL_REVIEW"), "review_failures", "ZREAD"),
    _case("review_status_fail", lambda c: c["reviews"][
        "R7_IMPLEMENTATION_CODE_REVIEW"].update(status="FAIL"),
          "review_failures", "R7_IMPLEMENTATION_CODE_REVIEW"),
    _case("review_commit_unbound", lambda c: c["reviews"][
        "FINAL_RERUN_REVIEW"].update(bound_commit="77" * 20),
          "review_failures", "FINAL_RERUN_REVIEW"),
    # ── constructed case A: learning-core diff falsely marked DOC_ONLY ──
    _case("learning_core_falsely_doc_only", lambda c: _construct_doc_only(c),
          "stale_artifacts", "DOC_ONLY"),
    # ── constructed case B: unknown dep + stale evidence reuse ─────
    _case("unknown_dep_reuses_old_evidence", lambda c: _construct_unknown(c),
          "stale_artifacts", "unknown"),
]


def _construct_doc_only(c):
    """A changed file between result_commit..expected would need proof the
    diff is DOC_ONLY-only; the matrix claims DOC_ONLY for a file the frozen
    classifier knows is LEARNING_CORE — the aggregator must reject."""
    c["matrix"]["changed_files"] = [{
        "path": "diffusion_policy_3d/rl/dppo_uav/vendor_loader.py",
        "sha256": "88" * 32,
        "changed_symbols": ["x"],
        "impact_classes": ["DOC_ONLY"],
        "classify_reason": "fake DOC_ONLY",
    }]
    c["manifest"]["bindings"]["R7-OFFICIAL-DIRECT"]["result_commit"] = (
        "89" * 20)  # != expected HEAD -> intermediate diff must be proven
    c["rtm"]["rows"][0]["result_commit"] = "89" * 20


def _construct_unknown(c):
    """unknown_or_ambiguous_dependencies non-empty while the manifest reuses
    the old binding — the diff cannot be proven -> reject."""
    c["matrix"]["unknown_or_ambiguous_dependencies"] = ["mystery-dep"]
    c["manifest"]["bindings"]["R7-RESUME"]["result_commit"] = "90" * 20
    c["rtm"]["rows"][2]["result_commit"] = "90" * 20


@pytest.mark.parametrize("mutate,expect_list,expect_fragment",
                         ANTI_CHEAT_CASES)
def test_anti_cheat_denies(tmp_path, mutate, expect_list, expect_fragment):
    paths, c = _input_bundle(tmp_path)
    if mutate is not None:
        mutate(c)
    for name, content in c.items():
        if name == "reviews":
            for i, (k, rev) in enumerate(content.items()):
                _rewrite(paths["reviews"][i], rev)
        elif name in ("rtm", "manifest", "matrix", "cpu", "gpu", "g3", "g5",
                      "g6", "resume", "g7", "bank"):
            _rewrite(paths[name], content)
    if expect_list == "missing_artifacts" and expect_fragment == "gpu":
        paths["gpu"].unlink()  # the delete case — AFTER the rewrite loop
    d = _evaluate(paths)

    assert d.engine_correct is False, f"{d}"
    assert d.training_admitted_for_g8 is False, f"{d}"
    combined = (list(d.failed_requirement_ids)
                + list(d.missing_artifacts)
                + list(d.stale_artifacts)
                + list(d.unresolved_severities)
                + list(d.review_failures))
    assert any(expect_fragment in str(x) for x in combined), (
        f"expected {expect_list} to name {expect_fragment!r}, got {combined}"
    )
    # process exit code != 0 through the canonical signer
    cli = _load_cli()
    out = tmp_path / "decision.json"
    rc = cli.main(["--rtm", str(paths["rtm"]),
                   "--implementation-manifest", str(paths["manifest"]),
                   "--test-impact-matrix", str(paths["matrix"]),
                   "--full-cpu", str(paths["cpu"]),
                   "--full-gpu", str(paths["gpu"]),
                   "--g3", str(paths["g3"]),
                   "--g5", str(paths["g5"]),
                   "--g6", str(paths["g6"]),
                   "--resume", str(paths["resume"]),
                   "--g7", str(paths["g7"]),
                   "--paired-bank", str(paths["bank"]),
                   "--review", str(paths["reviews"][0]),
                   "--review", str(paths["reviews"][1]),
                   "--review", str(paths["reviews"][2]),
                   "--review", str(paths["reviews"][3]),
                   "--expected-commit", _git(REPO_ROOT, "rev-parse", "HEAD"),
                   "--expected-tree", _git(REPO_ROOT, "rev-parse", "HEAD^{tree}"),
                   "--repo-root", str(REPO_ROOT),
                   "--out", str(out)])
    assert rc != 0, "canonical signer must exit non-zero on a denied admission"


def test_allowlisted_skip_is_accepted(tmp_path):
    """A skip on the frozen allowlist (with owner + unrelated proof) is
    permitted; unknown skips are not (covered by anti-cheat above)."""
    paths, c = _input_bundle(tmp_path)
    c["cpu"]["skips"] = [{
        "nodeid": "tests/dppo_parity/test_real_adapter.py::x",
        "reason": "gpu marker: CUDA unavailable",
        "owner": "conftest.py gpu marker hook",
        "unrelated_proof": "asserts no gpu-marked test in CPU suite",
    }]
    _rewrite(paths["cpu"], c["cpu"])
    d = _evaluate(paths)
    assert d.engine_correct is True


def test_cli_writes_decision_json_and_content_hash(tmp_path):
    """The canonical signer writes an r7-final-admission-v2 decision with a
    self-referential content hash and the complete input inventory (the
    training preflight re-executes the aggregator on it)."""
    paths, _ = _input_bundle(tmp_path)
    out = tmp_path / "decision.json"
    cli = _load_cli()
    rc = cli.main(_cli_argv(paths, out))
    assert rc == 0, "valid bundle must sign"
    decision = json.loads(out.read_text())
    assert decision["schema_version"] == "r7-final-admission-v2"
    assert decision["engine_correct"] is True
    assert decision["training_admitted_for_g8"] is True
    inputs = decision["inputs"]
    assert inputs["rtm_path"] == str(paths["rtm"])
    assert inputs["g7_results"] == [str(paths["g7"])]
    assert len(inputs["review_results"]) == 4
    body = {k: v for k, v in decision.items() if k != "content_hash"}
    assert hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest() == decision["content_hash"]
