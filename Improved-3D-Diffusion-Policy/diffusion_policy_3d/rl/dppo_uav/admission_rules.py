"""R7-7 frozen admission rules: source-set globs, impact classification,
git helpers and the skip allowlist.

The classifier here is a FROZEN COPY of the rules in
``program/reports/.../00_provenance/compute_impact_matrix.py`` (the
matrix generator).  The aggregator must never reclassify a changed file
on its own — it validates that the impact matrix's claims agree with
this frozen table, so a "learning-core diff falsely marked DOC_ONLY"
cannot slip through (R7-7 Step 2 / Step 8 anti-cheat).

Discipline:
- never edit the classification table without re-running the parity
  test that pins it against the generator;
- every git call goes through subprocess against ``repo_root`` and a
  failing git call is reported by the caller as a FAIL (never guessed).
"""

from __future__ import annotations

import fnmatch
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

# (path_glob, kind) — source set membership. Globs are repo-relative.
# Frozen copy of compute_impact_matrix.SOURCE_SET_GLOBS (R7-7 KDD-10).
SOURCE_SET_GLOBS = [
    ("diffusion_policy_3d/**/*.py", "production_python"),
    ("scripts/*.py", "entrypoint_script"),
    ("tests/dppo_parity/*.py", "test"),
    ("diffusion_policy_3d/config/**/*", "config"),
    ("third_party/dppo_official_cc7234a/UPSTREAM_COMMIT", "vendor_identity"),
    ("third_party/dppo_official_cc7234a/SHA256SUMS", "vendor_identity"),
    ("BASELINE_MANIFEST.json", "manifest"),
]

# Impact classification per changed file — frozen copy of
# compute_impact_matrix.CLASSIFY_RULES.
CLASSIFY_RULES = {
    "diffusion_policy_3d/rl/dppo_uav/vendor_loader.py": (
        ["LEARNING_CORE"],
        "official runtime import/verification entry point (R7-1)",
    ),
    "diffusion_policy_3d/rl/dppo_uav/policy_loader.py": (
        ["LEARNING_CORE"],
        "base-actor identity (e5_lambda0p5) + load contract schema v2",
    ),
    "diffusion_policy_3d/rl/dppo_uav/evaluator.py": (
        ["DOC_ONLY"],
        "comment-only change inside validate_eval_artifact; no code behavior",
    ),
}

# Frozen skip allowlist (R7-7 Step 3): prefix of the structured reason.
# Each entry: (reason_prefix, owner, why it is unrelated to the formal
# training path).  Any skip whose reason matches NO prefix is refused.
SKIP_ALLOWLIST = [
    ("gpu marker: CUDA unavailable",
     "conftest.py gpu marker hook",
     "asserts no gpu-marked test in the CPU suite"),
    ("BC checkpoint not found:",
     "conftest.py DEFAULT_BC_CKPT",
     "fixture asset absent on the machine; formal preflight would fail"),
    ("admission manifest not found:",
     "conftest.py production manifest lookup",
     "fixture asset absent; formal run refuses without the manifest"),
    ("worlds dir not found:",
     "conftest.py production worlds lookup",
     "fixture asset absent; formal run refuses without worlds"),
    ("admission manifest has no admitted rl_train entries",
     "conftest.py production manifest lookup",
     "fixture asset lacks entries; formal run refuses"),
    ("obstacles json not found:",
     "test_r3_rework3_positive.py",
     "needs the val-world obstacles asset; unrelated to the RL loop"),
    ("identity check needs a clean tree",
     "test_canonical_entrypoints_r7.py",
     "the identity gate runs on the final commit tree only"),
    ("positive preflight needs a clean tree",
     "test_canonical_entrypoints_r7.py",
     "the preflight gate runs on the final commit tree only"),
    ("production manifest/worlds/BC assets not present",
     "test_canonical_entrypoints_r7.py",
     "fixture assets absent; formal run refuses"),
    ("frozen BC checkpoint not present",
     "test_cfg_contract_r7.py",
     "fixture asset absent; formal run refuses"),
    ("production worlds/manifest absent, or tree dirty "
     "(preflight requires clean; runs on the final commit)",
     "test_world_assets_hash_r7.py",
     "gate runs on the final commit tree only"),
]

SHA256_HEX_LEN = 64
GIT_COMMIT_HEX_LEN = 40


@dataclass(frozen=True)
class EvidenceRef:
    """One RTM evidence entry: absolute path + full sha256 annotation."""

    path: str
    sha256: str


def is_source_set_path(repo_rel_path: str) -> bool:
    """True if the repo-relative path matches any frozen source-set glob."""
    return any(
        fnmatch.fnmatch(repo_rel_path, glob)
        for glob, _kind in SOURCE_SET_GLOBS
    )


def classify(path: str) -> tuple[list[str], str]:
    """Frozen impact classification (mirrors compute_impact_matrix)."""
    if path in CLASSIFY_RULES:
        return CLASSIFY_RULES[path]
    if path.startswith("tests/"):
        return ["TEST_ONLY"], "test file"
    return ["UNCLASSIFIED"], "FAIL-CLOSED: classify manually"


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        ch in "0123456789abcdef" for ch in value
    )


def parse_evidence_annotation(entry: str) -> EvidenceRef:
    """Parse ``"<abs path> (sha256 <64 hex>)"``.

    Raises ValueError with a precise reason when the annotation is
    relative, missing, or the hash is not full 64-char lowercase hex —
    the RTM may only carry full-sha-annotated absolute evidence paths
    (R7-10 rewrites the RTM to that form; until then the aggregator
    fails closed, which is the correct state for rows still awaiting
    review).
    """
    if not isinstance(entry, str):
        raise ValueError(f"evidence entry must be a string, got {entry!r}")
    if " (sha256 " not in entry:
        raise ValueError(
            f"evidence entry must carry a sha256 annotation: {entry!r}"
        )
    path_part, _, hash_part = entry.partition(" (sha256 ")
    hash_part = hash_part.rstrip(")")
    if not path_part or not Path(path_part).is_absolute():
        raise ValueError(f"evidence path must be absolute: {path_part!r}")
    if not _is_hex(hash_part, SHA256_HEX_LEN):
        raise ValueError(
            f"evidence sha256 must be {SHA256_HEX_LEN}-char lowercase hex: "
            f"{hash_part!r}"
        )
    return EvidenceRef(path=path_part, sha256=hash_part)


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a real file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    """Run git against repo_root; return stdout (may be empty)."""
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)!r} failed in {repo}: {out.stderr.strip()}"
        )
    return out.stdout.strip()


def git_head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def git_tree_of(repo: Path, commit: str) -> str:
    return _git(repo, "rev-parse", f"{commit}^{{tree}}")


def git_has_commit(repo: Path, commit: str) -> bool:
    """True when the commit object resolves in this repo."""
    if not _is_hex(commit, GIT_COMMIT_HEX_LEN):
        return False
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet",
         commit + "^{commit}"],
        capture_output=True,
        text=True,
    )
    return out.returncode == 0


def git_author(repo: Path, commit: str) -> str:
    """Committer identity of a resolved commit ('' when unresolvable)."""
    if not git_has_commit(repo, commit):
        return ""
    return _git(repo, "log", "-1", "--format=%an", commit)


def git_tree_clean(repo: Path) -> bool:
    """True when the working tree is clean (porcelain empty)."""
    out = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    return out.returncode == 0 and not out.stdout.strip()


def skip_reason_allowlisted(reason: str) -> str | None:
    """Owner of the allowlisted reason prefix, or None when unknown.

    The decision text keeps the owner so the operator can trace WHY a
    skip was accepted (R7-7 Step 3: every accepted skip must carry a
    reason, an owner and an unrelatedness proof).
    """
    if not isinstance(reason, str):
        return None
    for prefix, owner, _proof in SKIP_ALLOWLIST:
        if reason.startswith(prefix):
            return owner
    return None


def changed_files_between(
    repo: Path, baseline: str, head: str
) -> list[str]:
    """Repo-relative changed paths of ``baseline..head`` (sorted)."""
    out = _git(repo, "diff", "--name-only", baseline + ".." + head)
    return [c for c in out.splitlines() if c.strip()]
