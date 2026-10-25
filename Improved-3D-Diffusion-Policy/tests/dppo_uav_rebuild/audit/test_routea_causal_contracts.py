from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from scripts.audit_checkpoint_source_identity import build_inventory
from scripts.routea_causal_diagnostics.contracts import (
    DiagnosticFailure,
    build_diagnostic_source_manifest,
    canonical_array_sha256,
    canonical_tree_sha256,
    capture_rng_state,
    load_numeric_npz,
    load_strict_json,
    restore_rng_state,
    rng_state_sha256,
    write_numeric_npz,
    write_strict_json,
)


REPO_ROOT = Path(__file__).parents[3]
BASELINE_SOURCE_HASH = "5aed4fb675591750ac97ad4165ac11b8d3fedf5a15dfa2bf0bc84266c373f637"


def test_diagnostic_overlay_is_hashed_without_changing_training_source_set():
    training_rows, before = build_inventory(REPO_ROOT)
    manifest = build_diagnostic_source_manifest(REPO_ROOT)
    after = build_inventory(REPO_ROOT)[1]
    assert before == BASELINE_SOURCE_HASH
    assert after == before
    assert manifest["schema"] == "routea_causal_diagnostic_source_manifest_v1"
    assert manifest["files"]
    assert all(row["path"].startswith("scripts/") for row in manifest["files"])
    assert {row["path"] for row in manifest["files"]}.isdisjoint(
        {row["path"] for row in training_rows}
    )


def test_diagnostic_manifest_changes_when_overlay_copy_changes(tmp_path):
    source = tmp_path / "scripts" / "routea_causal_diagnostics" / "core.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original\n")
    before = build_diagnostic_source_manifest(tmp_path)["content_sha256"]
    source.write_bytes(b"modified\n")
    assert build_diagnostic_source_manifest(tmp_path)["content_sha256"] != before


def test_canonical_array_hash_binds_dtype_shape_and_bytes():
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert canonical_array_sha256(base) == canonical_array_sha256(base.copy(order="F"))
    assert canonical_array_sha256(base) != canonical_array_sha256(base.astype(np.float64))
    assert canonical_array_sha256(base) != canonical_array_sha256(base.reshape(3, 2))


def test_tree_hash_is_mapping_order_independent_and_rejects_nonfinite():
    left = {"b": torch.tensor([1.0]), "a": [np.int64(2), None, "x"]}
    right = {"a": [2, None, "x"], "b": np.array([1.0], dtype=np.float32)}
    assert canonical_tree_sha256(left) == canonical_tree_sha256(right)
    with pytest.raises(DiagnosticFailure, match="INVALID_CANONICAL_VALUE"):
        canonical_tree_sha256({"x": float("inf")})


def test_strict_json_roundtrip_and_rejects_duplicates_nonfinite_and_size(tmp_path):
    target = tmp_path / "strict.json"
    write_strict_json(target, {"z": 1, "text": "目标", "none": None})
    assert load_strict_json(target, max_bytes=1024)["text"] == "目标"
    assert target.read_text(encoding="utf-8").endswith("\n")

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(DiagnosticFailure, match="INVALID_JSON_ARTIFACT"):
        load_strict_json(duplicate, max_bytes=1024)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(DiagnosticFailure, match="INVALID_JSON_ARTIFACT"):
        load_strict_json(nonfinite, max_bytes=1024)

    with pytest.raises(DiagnosticFailure, match="ARTIFACT_TOO_LARGE"):
        load_strict_json(target, max_bytes=1)
    with pytest.raises(DiagnosticFailure, match="INVALID_JSON_ARTIFACT"):
        write_strict_json(tmp_path / "bad.json", {"x": np.nan})


def test_numeric_npz_roundtrip_rejects_object_nonfinite_shape_and_size(tmp_path):
    target = tmp_path / "bank.npz"
    arrays = {
        "agent_pos": np.arange(72, dtype=np.float32).reshape(1, 8, 9),
        "point_cloud": np.zeros((1, 8, 4, 3), dtype=np.float32),
    }
    manifest = write_numeric_npz(target, arrays, max_total_bytes=4096)
    loaded = load_numeric_npz(
        target,
        expected_shapes={"agent_pos": (1, 8, 9), "point_cloud": (1, 8, 4, 3)},
        max_total_bytes=4096,
    )
    assert sorted(loaded) == ["agent_pos", "point_cloud"]
    assert manifest["arrays"]["agent_pos"]["sha256"] == canonical_array_sha256(
        loaded["agent_pos"]
    )

    with pytest.raises(DiagnosticFailure, match="INVALID_NUMERIC_ARTIFACT"):
        write_numeric_npz(
            tmp_path / "bad.npz", {"x": np.array([object()])}, max_total_bytes=1024
        )
    with pytest.raises(DiagnosticFailure, match="INVALID_NUMERIC_ARTIFACT"):
        write_numeric_npz(
            tmp_path / "nan.npz", {"x": np.array([np.nan])}, max_total_bytes=1024
        )
    with pytest.raises(DiagnosticFailure, match="ARTIFACT_TOO_LARGE"):
        write_numeric_npz(
            tmp_path / "large.npz", {"x": np.ones(1025, dtype=np.uint8)}, max_total_bytes=1024
        )
    with pytest.raises(DiagnosticFailure, match="NPZ_SHAPE_MISMATCH"):
        load_numeric_npz(
            target, expected_shapes={"agent_pos": (8, 9)}, max_total_bytes=4096
        )


def test_rng_roundtrip_restores_python_numpy_torch_and_hash_is_stable():
    state = capture_rng_state()
    digest = rng_state_sha256(state)
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
    restore_rng_state(state)
    assert rng_state_sha256(capture_rng_state()) == digest


def test_rng_state_is_detached_from_later_generator_mutation():
    state = capture_rng_state()
    before = rng_state_sha256(state)
    random.random()
    np.random.random()
    torch.rand(2)
    assert rng_state_sha256(state) == before
