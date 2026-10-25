"""F2 (Zread P0-2, 2026-08-24): gate results must be bound to a REAL raw
log — existence + actual sha256 — and to an allowlisted producer.

Zread repro: ``parse_test_result`` accepted a result whose log_path was
``/definitely/not/existing/r7.log`` (only type/format checks on
log_path/log_sha256).  Every r7-test-result-v1 now must carry a
``producer`` from the allowlist and a log whose file exists and whose
sha256 matches the declared hash.
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

FAKE_LOG = "/definitely/not/existing/r7.log"


def _write_log(tmp_path, name="suite.log", content=b"real suite log\n"):
    log = tmp_path / name
    log.write_bytes(content)
    return str(log), hashlib.sha256(content).hexdigest()


def _bundle_with_cpu(tmp_path, **result_overrides):
    """Full valid bundle with a custom CPU result JSON (rest valid)."""
    paths, _ = _input_bundle(tmp_path)
    log_path, log_sha = _write_log(tmp_path)
    head = _git(REPO_ROOT, "rev-parse", "HEAD")
    tree = _git(REPO_ROOT, "rev-parse", "HEAD^{tree}")
    res = {
        "schema_version": "r7-test-result-v1",
        "command": "pytest tests/dppo_parity -m 'not gpu' -ra",
        "started_at": "2026-08-24T00:00:00Z",
        "finished_at": "2026-08-24T01:00:00Z",
        "exit_code": 0,
        "collected": 600,
        "passed": 600,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "xfail": 0,
        "xpass": 0,
        "deselected": 0,
        "commit": head,
        "tree": tree,
        "validated_source_set_hash": "ab" * 32,
        "producer": "pytest",
        "log_path": log_path,
        "log_sha256": log_sha,
        "skips": [],
    }
    res.update(result_overrides)
    paths["cpu"].write_text(json.dumps(res))
    return paths


def test_fake_log_path_rejected(tmp_path):
    """Zread P0-2 repro: a result bound to a non-existent log must fail."""
    paths = _bundle_with_cpu(tmp_path, log_path=FAKE_LOG)
    d = _evaluate(paths)
    assert d.engine_correct is False
    assert any("log_path does not exist" in s or "log" in s
               for s in d.stale_artifacts)


def test_log_hash_mismatch_rejected(tmp_path):
    """Log exists but the declared sha256 is wrong -> reject."""
    log_path, _ = _write_log(tmp_path)
    paths = _bundle_with_cpu(
        tmp_path, log_path=log_path, log_sha256="cd" * 32)
    d = _evaluate(paths)
    assert d.engine_correct is False
    assert any("sha256 mismatch" in s for s in d.stale_artifacts)


def test_missing_log_file_rejected(tmp_path):
    """log_path points at a missing file -> reject."""
    paths = _bundle_with_cpu(tmp_path, log_path=str(tmp_path / "nope.log"))
    d = _evaluate(paths)
    assert d.engine_correct is False


def test_producer_not_allowlisted_rejected(tmp_path):
    """Hand-written producer -> reject."""
    paths = _bundle_with_cpu(tmp_path, producer="handwritten")
    d = _evaluate(paths)
    assert d.engine_correct is False
    assert any("producer" in s for s in d.stale_artifacts)


def test_missing_producer_field_rejected(tmp_path):
    """No producer field at all -> reject (required key)."""
    log_path, log_sha = _write_log(tmp_path)
    paths = _bundle_with_cpu(tmp_path, log_path=log_path,
                             log_sha256=log_sha)
    blob = json.loads(paths["cpu"].read_text())
    del blob["producer"]
    paths["cpu"].write_text(json.dumps(blob))
    d = _evaluate(paths)
    assert d.engine_correct is False


def test_valid_bound_log_admits(tmp_path):
    """Real log + matching hash + allowlisted producer admits."""
    paths = _bundle_with_cpu(tmp_path)
    d = _evaluate(paths)
    assert d.engine_correct is True, d.stale_artifacts
