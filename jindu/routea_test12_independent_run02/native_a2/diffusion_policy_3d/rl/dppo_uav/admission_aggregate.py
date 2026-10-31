"""R7-7: the ONE fail-closed global training admission aggregator.

Default-deny: every boolean starts false; every input (RTM rows,
evidence, commits, reviews, impact matrix, CPU/GPU suite results,
G3/G5/G6/resume/G7/bank gates, hashes) must pass an explicit check
before the decision flips true.  Any exception, JSON parse error,
missing key, unknown schema, empty table, NaN, missing path, hash
drift, stale final identity, stale source-set, or impact-matrix gap
returns FAIL — never an exception-to-warning-to-PASS path.

`evaluate_final_admission()` is the only function that may sign
engine_correct / training_admitted_for_g8 (guide R7-7 Step 1-5);
`scripts/r7_final_admission.py` is the only CLI that writes the
decision JSON.  `validate_final_admission_for_training()` is the
canonical-train preflight consumer (guide R7-7 Step 7).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import admission_parsers as ap
from .admission_rules import (
    GIT_COMMIT_HEX_LEN,
    EvidenceRef,
    classify,
    changed_files_between,
    git_author,
    git_has_commit,
    git_head,
    git_tree_clean,
    git_tree_of,
    is_source_set_path,
    parse_evidence_annotation,
    sha256_file,
)

# 14 exact R7 requirement IDs (guide R7-0 Step 7; frozen set).
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

# 4 review kinds the final admission must collect (guide R7-7 Step 5).
REVIEW_KINDS = (
    "R7_IMPLEMENTATION_CODE_REVIEW",
    "R1_R4_REQUIREMENT_REPLAY_AUDIT",
    "ZREAD_OR_EQUIVALENT_FINAL_REVIEW",
    "FINAL_RERUN_REVIEW",
)

# v2 (2026-08-24, Zread P0-1): the decision JSON must carry the complete
# input inventory and the preflight must re-execute the aggregator instead
# of trusting hand-written booleans + a self-hash.
DECISION_SCHEMA = "r7-final-admission-v2"

# Exact evaluate input inventory a v2 decision must record (CLI writer
# contract; keys are evaluate_final_admission kwargs).
INPUT_KEYS = (
    "rtm_path",
    "implementation_manifest_path",
    "test_impact_matrix_path",
    "full_cpu_result",
    "full_gpu_result",
    "g3_result",
    "g5_result",
    "g6_result",
    "resume_result",
    "paired_bank_result",
    "g7_results",
    "review_results",
    "expected_commit",
    "expected_tree",
)


@dataclass(frozen=True)
class VerifiedArtifact:
    kind: str
    path: str
    sha256: str
    result_commit: str
    result_tree: str
    validated_source_set_hash: str


@dataclass(frozen=True)
class AdmissionDecision:
    engine_correct: bool
    training_admitted_for_g8: bool
    failed_requirement_ids: tuple[str, ...]
    unresolved_severities: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    stale_artifacts: tuple[str, ...]
    review_failures: tuple[str, ...]
    verified_inputs: tuple[VerifiedArtifact, ...]
    final_commit: str
    final_tree: str


def decision_content_hash(decision: dict) -> str:
    """Deterministic content hash: sha256 of canonical JSON excluding the
    hash key itself (same contract as r6 freeze_g8_prereg)."""
    body = {k: v for k, v in decision.items() if k != "content_hash"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _missing(missing_artifacts: list[str], label: str, path) -> bool:
    """Report one missing input; returns True when absent."""
    if path is None:
        missing_artifacts.append(label)
        return True
    if not Path(path).exists():
        missing_artifacts.append(label)
        return True
    return False


def _require_hex(value, name: str) -> None:
    if not isinstance(value, str) or len(value) != GIT_COMMIT_HEX_LEN or not all(
        ch in "0123456789abcdef" for ch in value
    ):
        raise ap.ValidationError(f"{name} must be 40-char lowercase hex")


def _verify_rtm(
    rtm: dict,
    manifest: dict,
    matrix: dict,
    repo: Path,
    expected_commit: str,
    expected_tree: str,
    failed: list[str],
    severities: list[str],
    stale: list[str],
    verified: list[VerifiedArtifact],
) -> None:
    """Step 2: the 14 rows must exist exactly once, be PASS, carry
    absolute full-sha evidence that exists and matches, bind a
    locatable commit whose tree equals the manifest binding, and be
    reviewed by a non-implementer."""
    rows = rtm.get("rows")
    if not isinstance(rows, list):
        raise ap.ValidationError("rtm rows must be a list")
    ids = [r.get("id") for r in rows if isinstance(r, dict)]
    for rid in REQUIRED_IDS:
        if rid not in ids:
            failed.append(f"{rid} missing from RTM")
    if len(ids) != len(set(ids)):
        for rid in REQUIRED_IDS:
            if ids.count(rid) > 1:
                failed.append(f"duplicate RTM row {rid}")
    bindings = manifest.get("bindings", {}) if isinstance(manifest, dict) else {}
    source_hash = matrix.get("validated_source_set_hash")
    for row in rows:
        if not isinstance(row, dict):
            failed.append("RTM row must be an object")
            continue
        rid = row.get("id")
        if not isinstance(rid, str):
            failed.append("RTM row missing id")
            continue
        for key in ("source", "finding", "severity", "status", "code_targets",
                    "tests", "evidence_paths", "result_commit", "reviewer",
                    "review_status"):
            if key not in row:
                failed.append(rid)
                stale.append(f"{rid} RTM schema violation: missing {key}")
                break
        if row.get("status") != "PASS":
            failed.append(rid)
            if isinstance(row.get("severity"), str):
                severities.append(row["severity"])
        if not row.get("code_targets"):
            failed.append(f"{rid} code_targets must be nonempty")
        if not row.get("tests"):
            failed.append(f"{rid} tests must be nonempty")
        evidence_paths = row.get("evidence_paths")
        if not isinstance(evidence_paths, list) or not evidence_paths:
            failed.append(f"{rid} evidence_paths must be nonempty")
        else:
            for entry in evidence_paths:
                try:
                    ref: EvidenceRef = parse_evidence_annotation(entry)
                except ValueError as e:
                    failed.append(f"{rid} {e}")
                    continue
                if not Path(ref.path).is_file():
                    failed.append(f"{rid} evidence missing: {ref.path}")
                    continue
                if sha256_file(Path(ref.path)) != ref.sha256:
                    failed.append(f"{rid} evidence sha256 drift: {ref.path}")
        result_commit = row.get("result_commit")
        binding = bindings.get(rid)
        try:
            _require_hex(result_commit, f"{rid} result_commit")
            if not git_has_commit(repo, result_commit):
                failed.append(f"{rid} result_commit not locatable: "
                              f"{result_commit}")
                continue
            if not isinstance(binding, dict):
                failed.append(f"{rid} manifest binding missing")
                continue
            if binding.get("result_commit") != result_commit:
                failed.append(f"{rid} manifest result_commit mismatch")
                continue
            binding_tree = git_tree_of(repo, result_commit)
            if binding.get("result_tree") != binding_tree:
                failed.append(f"{rid} manifest result_tree mismatch")
                continue
            # impact matrix must prove the intermediate diff is only
            # DOC_ONLY / outside the source set (guide Step 2).
            if result_commit != expected_commit and not _diff_proven_by_matrix(
                repo, result_commit, expected_commit, matrix, stale,
            ):
                failed.append(
                    f"{rid} intermediate diff not proven DOC_ONLY/outside "
                    "source set"
                )
            # manifest binding hash must equal the impact-matrix authority
            binding_hash = binding.get("validated_source_set_hash")
            if binding_hash != source_hash:
                stale.append(f"{rid} source-set hash stale: "
                             f"{binding_hash} != {source_hash}")
            # binding artifacts: exist and hash-match the real files
            artifacts = binding.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                stale.append(f"{rid} manifest artifacts must be nonempty")
            else:
                for art in artifacts:
                    if not isinstance(art, dict):
                        stale.append(f"{rid} manifest artifact must be object")
                        continue
                    art_path = art.get("path")
                    art_hash = art.get("sha256")
                    if not isinstance(art_path, str) or not Path(
                        art_path
                    ).is_file():
                        stale.append(f"{rid} artifact missing: {art_path}")
                        continue
                    if sha256_file(Path(art_path)) != art_hash:
                        stale.append(f"{rid} artifact sha256 drift")
                        continue
                    verified.append(VerifiedArtifact(
                        kind=rid,
                        path=art_path,
                        sha256=art_hash,
                        result_commit=result_commit,
                        result_tree=binding_tree,
                        validated_source_set_hash=source_hash or "",
                    ))
        except ap.ValidationError as e:
            failed.append(f"{rid} {e}")
        if row.get("review_status") != "PASS":
            failed.append(f"{rid} review_status must be PASS")
        reviewer = row.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer:
            failed.append(f"{rid} reviewer must be nonempty")
        else:
            implementer = git_author(repo, result_commit)
            if implementer and reviewer == implementer:
                failed.append(
                    f"{rid} reviewer must differ from implementer "
                    f"({implementer})"
                )
    # impact matrix tree/commit identity
    if matrix.get("candidate_code_commit") != expected_commit:
        stale.append("impact matrix candidate commit != expected commit")
    if matrix.get("candidate_code_tree") != expected_tree:
        stale.append("impact matrix candidate tree != expected tree")


def _diff_proven_by_matrix(
    repo: Path, old_commit: str, new_commit: str, matrix: dict,
    stale: list[str],
) -> bool:
    """Step 2 commit-diff proof: every changed file of
    ``old_commit..new_commit`` must lie OUTSIDE the source set (external
    evidence/docs) or classify DOC_ONLY by the FROZEN table; any
    LEARNING_CORE / TEST_ONLY / UNCLASSIFIED change, or a non-empty
    unknown-dependency list, breaks the proof."""
    changed = changed_files_between(repo, old_commit, new_commit)
    if matrix.get("unknown_or_ambiguous_dependencies"):
        stale.append("impact matrix unknown_or_ambiguous_dependencies "
                     "nonempty — diff cannot be proven")
        return False
    proven = True
    for rel in changed:
        if is_source_set_path(rel):
            classes, _reason = classify(rel)
            if classes != ["DOC_ONLY"]:
                stale.append(
                    f"intermediate diff touches source-set path {rel} with "
                    f"classes {classes} (must be DOC_ONLY or outside)"
                )
                proven = False
    return proven


def _verify_gates(
    paths: dict, repo: Path, expected_commit: str, expected_tree: str,
    source_hash: str, stale: list[str], missing: list[str],
) -> None:
    """Step 3-4: suite results + raw numeric gate fields.  Every
    violation appends to stale/missing."""
    for label, path in (
        ("cpu", paths.get("full_cpu_result")),
        ("gpu", paths.get("full_gpu_result")),
    ):
        if _missing(missing, label, path):
            continue
        try:
            res = ap.parse_test_result(Path(path))
        except ap.ValidationError as e:
            stale.append(f"{label} {e}")
            continue
        if res["commit"] != expected_commit or res["tree"] != expected_tree:
            stale.append(f"{label} commit/tree != expected final identity")
        if res["validated_source_set_hash"] != source_hash:
            stale.append(f"{label} validated_source_set_hash stale")

    gate_labels = ("g3", "g5", "g6", "resume", "bank")
    parsers = {
        "g3": ap.parse_g3_result,
        "g5": ap.parse_g5_result,
        "g6": ap.parse_g6_result,
        "resume": ap.parse_resume_result,
        "g7": ap.parse_g7_result,
        "bank": ap.parse_bank_result,
    }
    parsed: dict[str, dict] = {}
    path_keys = {
        "g3": "g3_result",
        "g5": "g5_result",
        "g6": "g6_result",
        "resume": "resume_result",
        "bank": "paired_bank_result",
    }
    g7_seen = 0
    for label in gate_labels:
        path = paths.get(path_keys[label])
        if _missing(missing, label, path):
            continue
        try:
            parsed[label] = parsers[label](Path(path))
        except ap.ValidationError as e:
            stale.append(f"{label} {e}")
    paths_g7 = paths.get("g7_results") or []
    if not paths_g7:
        missing.append("g7")
    else:
        for p in paths_g7:
            if _missing(missing, "g7", p):
                continue
            try:
                parsed["g7"] = parsers["g7"](Path(p))
                g7_seen += 1
            except ap.ValidationError as e:
                stale.append(f"g7 {e}")
    for label in ("g3", "g5", "g6", "g7"):
        if label in parsed:
            res = parsed[label]
            if res["commit"] != expected_commit or res["tree"] != expected_tree:
                stale.append(f"{label} commit/tree != expected final identity")
            if res["validated_source_set_hash"] != source_hash:
                stale.append(f"{label} validated_source_set_hash stale")
    if g7_seen == 0:
        missing.append("g7")
    # resume carries no commit axis (schema r7-compare-v1, R7-6) —
    # its bindings travel through the manifest result_commit.


def _verify_reviews(review_paths, repo: Path, source_hash: str,
                    failures: list[str]) -> None:
    """Step 5: all 4 review kinds, each PASS, bound to a locatable
    commit and the validated source-set hash."""
    kinds_seen = []
    for path in review_paths or []:
        try:
            review = ap.parse_review(Path(path))
        except ap.ValidationError as e:
            failures.append(f"{e}")
            continue
        kind = review["kind"]
        kinds_seen.append(kind)
        if review["status"] != "PASS":
            failures.append(f"{kind} status must be PASS")
        if not git_has_commit(repo, review["bound_commit"]):
            failures.append(f"{kind} bound_commit not locatable")
        if review["bound_source_set_hash"] != source_hash:
            failures.append(f"{kind} bound source-set hash stale")
    for kind in REVIEW_KINDS:
        if kind not in kinds_seen:
            failures.append(f"missing review kind {kind}")
    if len(kinds_seen) != len(set(kinds_seen)):
        failures.append("duplicate review kind")


def evaluate_final_admission(
    *,
    rtm_path=None,
    implementation_manifest_path=None,
    test_impact_matrix_path=None,
    full_cpu_result=None,
    full_gpu_result=None,
    g3_result=None,
    g5_result=None,
    g6_result=None,
    resume_result=None,
    g7_results=None,
    paired_bank_result=None,
    review_results=None,
    expected_commit=None,
    expected_tree=None,
    repo_root=None,
) -> AdmissionDecision:
    """Default-deny global admission aggregator (guide R7-7 Step 1-5).

    All inputs are optional keyword args; a missing one lands in
    missing_artifacts.  NEVER raises on bad input — it always returns a
    FAIL decision naming every violated input (Step 1: exception →
    FAIL, never warning → PASS).
    """
    failed: list[str] = []
    severities: list[str] = []
    missing: list[str] = []
    stale: list[str] = []
    review_failures: list[str] = []
    verified: list[VerifiedArtifact] = []
    repo = Path(repo_root) if repo_root else Path.cwd()

    try:
        if _missing(missing, "rtm", rtm_path):
            rtm = None
        else:
            rtm = ap._load_json(Path(rtm_path), "rtm")
        if _missing(missing, "manifest", implementation_manifest_path):
            manifest = None
        else:
            manifest = ap._load_json(
                Path(implementation_manifest_path), "manifest")
        if _missing(missing, "matrix", test_impact_matrix_path):
            matrix = None
        else:
            matrix = ap._load_json(Path(test_impact_matrix_path), "matrix")
        if expected_commit is None:
            expected_commit = git_head(repo)
        if expected_tree is None:
            expected_tree = git_tree_of(repo, expected_commit)
        _require_hex(expected_commit, "expected_commit")
        _require_hex(expected_tree, "expected_tree")

        if rtm is not None and manifest is not None and matrix is not None:
            if rtm.get("schema_version") != "r7-rtm-v1":
                stale.append("rtm unknown schema_version")
            if manifest.get("schema_version") != "r7-implementation-manifest-v1":
                stale.append("manifest unknown schema_version")
            if matrix.get("schema_version") != "r7-test-impact-v1":
                stale.append("matrix unknown schema_version")
            source_hash = matrix.get("validated_source_set_hash")
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                stale.append("matrix validated_source_set_hash malformed")
            if rtm.get("schema_version") == "r7-rtm-v1" and manifest.get(
                "schema_version"
            ) == "r7-implementation-manifest-v1" and matrix.get(
                "schema_version"
            ) == "r7-test-impact-v1":
                _verify_rtm(
                    rtm, manifest, matrix, repo,
                    expected_commit, expected_tree,
                    failed, severities, stale, verified,
                )
        if matrix is not None and rtm is not None:
            if matrix.get("unknown_or_ambiguous_dependencies"):
                stale.append("matrix unknown_or_ambiguous_dependencies "
                             "nonempty")
            for entry in matrix.get("changed_files", []) or []:
                if not isinstance(entry, dict) or "path" not in entry:
                    stale.append("matrix changed_files entry malformed")
                    continue
                rel = entry["path"]
                frozen_classes, _reason = classify(rel)
                claim = entry.get("impact_classes") or []
                if set(claim) != set(frozen_classes):
                    stale.append(
                        f"{rel} impact_classes {claim} rejected: frozen "
                        f"classify {frozen_classes} (DOC_ONLY false claim "
                        "cannot hide learning-core changes)"
                    )

        paths = {
            "full_cpu_result": full_cpu_result,
            "full_gpu_result": full_gpu_result,
            "g3_result": g3_result,
            "g5_result": g5_result,
            "g6_result": g6_result,
            "resume_result": resume_result,
            "g7_results": list(g7_results) if g7_results else None,
            "paired_bank_result": paired_bank_result,
        }
        _verify_gates(
            paths, repo, expected_commit, expected_tree,
            matrix.get("validated_source_set_hash") if matrix else "",
            stale, missing,
        )
        _verify_reviews(review_results, repo,
                        matrix.get("validated_source_set_hash") if matrix
                        else "", review_failures)
    except (ap.ValidationError, ValueError) as e:
        stale.append(f"admission evaluation failed closed: {e}")

    engine_ok = not failed and not severities and not missing and not stale \
        and not review_failures
    return AdmissionDecision(
        engine_correct=engine_ok,
        training_admitted_for_g8=engine_ok,
        failed_requirement_ids=tuple(failed),
        unresolved_severities=tuple(severities),
        missing_artifacts=tuple(missing),
        stale_artifacts=tuple(stale),
        review_failures=tuple(review_failures),
        verified_inputs=tuple(verified),
        final_commit=expected_commit,
        final_tree=expected_tree,
    )


def validate_final_admission_for_training(
    path, repo_root, *, require_clean_tree: bool = True,
) -> list[str]:
    """v2 canonical-train preflight (guide R7-7 Step 7; hardened 2026-08-24
    for Zread P0-1).

    v1 only proved that someone could hand-write booleans and recompute a
    self-hash (reproduced FORGED_DECISION_PROBLEMS=[]).  v2 additionally
    requires:
      * schema ``r7-final-admission-v2`` carrying the exact full ``inputs``
        inventory (every evaluate kwarg the CLI writer records);
      * non-empty ``verified_inputs``;
      * a fresh re-execution of ``evaluate_final_admission()`` on the
        recorded inputs — the re-evaluated engine flags, verified_inputs,
        and the absence of missing/stale/review failures must all match
        the JSON claims.  A forged or stale decision fails closed.

    Returns the problems list ([] = admissible).  The launcher exits
    non-zero on any problem; a config file can never hand-write
    ``engine_pass`` and an env var can never bypass this check.
    """
    problems: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            decision = json.load(f)
    except (OSError, ValueError) as e:
        return [f"final admission JSON unreadable: {e}"]
    if not isinstance(decision, dict):
        return ["final admission JSON must be an object"]
    if decision.get("schema_version") != DECISION_SCHEMA:
        problems.append(
            f"final admission unknown schema_version "
            f"{decision.get('schema_version')!r} "
            f"(expected {DECISION_SCHEMA})"
        )
    if decision.get("engine_correct") is not True:
        problems.append("final admission engine_correct must be true")
    if decision.get("training_admitted_for_g8") is not True:
        problems.append(
            "final admission training_admitted_for_g8 must be true"
        )
    if decision_content_hash(decision) != decision.get("content_hash"):
        problems.append("final admission content_hash mismatch")
    repo = Path(repo_root)
    try:
        head = git_head(repo)
        tree = git_tree_of(repo, head)
    except ValueError as e:
        return [f"cannot resolve final identity: {e}"]
    if decision.get("final_commit") != head:
        problems.append(
            f"final admission final_commit {decision.get('final_commit')} "
            f"!= current HEAD {head}"
        )
    if decision.get("final_tree") != tree:
        problems.append(
            f"final admission final_tree {decision.get('final_tree')} "
            f"!= current tree {tree}"
        )
    if require_clean_tree and not git_tree_clean(repo):
        problems.append("working tree must be clean for a formal launch")

    # ── v2: verified_inputs must be non-empty (Zread P0-1) ──────────
    verified_inputs = decision.get("verified_inputs") or []
    if not verified_inputs:
        problems.append(
            "final admission verified_inputs must be non-empty")

    # ── v2: exact full inputs inventory (Zread P0-1) ────────────────
    inputs = decision.get("inputs")
    if not isinstance(inputs, dict):
        problems.append("final admission inputs inventory missing")
        inputs = {}
    else:
        missing_keys = sorted(set(INPUT_KEYS) - set(inputs))
        extra_keys = sorted(set(inputs) - set(INPUT_KEYS))
        if missing_keys:
            problems.append(
                f"final admission inputs missing keys: {missing_keys}")
        if extra_keys:
            problems.append(
                f"final admission inputs extra keys: {extra_keys}")

    # bound-artifact existence/hash checks on the recorded inputs
    for art in verified_inputs:
        if not isinstance(art, dict):
            problems.append("verified_inputs entry must be an object")
            continue
        art_path = art.get("path")
        art_hash = art.get("sha256")
        if not isinstance(art_path, str) or not Path(art_path).is_file():
            problems.append(f"bound artifact missing: {art_path}")
            continue
        if sha256_file(Path(art_path)) != art_hash:
            problems.append(f"bound artifact hash drift: {art_path}")

    # ── v2: re-execute the aggregator on the recorded inputs ────────
    if set(inputs) == set(INPUT_KEYS):
        try:
            reeval = evaluate_final_admission(
                **{k: v for k, v in inputs.items()}, repo_root=repo_root)
        except (TypeError, ValueError) as e:
            problems.append(f"final admission re-execution failed: {e}")
            reeval = None
        if reeval is not None:
            if not reeval.engine_correct:
                problems.append(
                    "final admission re-execution FAILED: missing="
                    f"{sorted(reeval.missing_artifacts)} stale="
                    f"{sorted(reeval.stale_artifacts)} failed="
                    f"{sorted(reeval.failed_requirement_ids)} reviews="
                    f"{sorted(reeval.review_failures)}")
            if (reeval.engine_correct
                    != decision.get("engine_correct")
                    or reeval.training_admitted_for_g8
                    != decision.get("training_admitted_for_g8")):
                problems.append(
                    "final admission re-executed decision disagrees "
                    "with JSON claims")
            reeval_verified = [dict(v.__dict__)
                               for v in reeval.verified_inputs]
            if reeval_verified != verified_inputs:
                problems.append(
                    "verified_inputs drift between JSON and "
                    "re-execution")
    return problems
