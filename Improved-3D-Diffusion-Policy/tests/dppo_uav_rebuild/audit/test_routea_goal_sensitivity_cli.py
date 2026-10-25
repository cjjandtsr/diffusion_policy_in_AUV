from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.endpoint_trajectory_trace_v2 import plan_waypoints
from scripts.probe_routea_goal_sensitivity import (
    EXIT_CODES,
    _bank_observation,
    _failure,
    _frozen_plan_action,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import (
    PointCloudGenerationMeta,
    canonical_array_hash,
    validate_metadata_history,
)
from scripts.routea_causal_diagnostics.contracts import (
    DiagnosticFailure,
    write_numeric_npz,
    write_strict_json,
)
from scripts.routea_causal_diagnostics.reporting import (
    MINIMUM_FINAL_OUTPUTS,
    publish_stage_manifest,
    write_csv_rows,
)


REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "scripts" / "probe_routea_goal_sensitivity.py"
SOURCE_MANIFEST = {
    "schema": "routea_causal_diagnostic_source_manifest_v1",
    "content_sha256": "a" * 64,
    "file_count": 1,
    "files": [{"path": "scripts/fake.py", "bytes": 1, "sha256": "b" * 64}],
}


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_self_check_passes_and_preserves_training_source_set():
    result = _run("--self-check")
    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stderr
    assert payload["status"] == "SELF_CHECK_PASS"
    assert payload["goal_roundtrip_max_abs"] <= 1e-4
    assert payload["rng_restored"] is True
    assert payload["numeric_npz_reload"] is True
    assert payload["training_source_set_unchanged"] is True


def test_capture_b0_refuses_existing_output_before_reading_inputs(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    result = _run(
        "capture-b0",
        "--selection-dir",
        tmp_path / "missing-selection",
        "--config",
        tmp_path / "missing.yaml",
        "--output-dir",
        existing,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["status"] == "OUTPUT_DIR_EXISTS"


def test_frozen_plan_action_round_trips_authoritative_world_waypoints():
    before = {
        "position": np.asarray([1.25, -2.5, 3.75], dtype=np.float32),
        "velocity_reference": np.asarray([0.2, -0.1, 0.3], dtype=np.float32),
        "yaw": 2.9,
    }
    raw = np.zeros((8, 11), dtype=np.float32)
    raw[:, :3] = np.asarray(
        [[0.1, -0.2, 0.05], [-0.3, 0.1, -0.02]] * 4, dtype=np.float32
    )
    raw[:, 3:9] = np.arange(48, dtype=np.float32).reshape(8, 6) / 100.0
    raw[:, 9] = np.asarray([0.4, 0.2, -0.3, 0.1, -0.2, 0.3, 0.15, -0.25])
    raw[:, 10] = np.linspace(-0.5, 0.5, 8, dtype=np.float32)
    planned = plan_waypoints(raw, before)
    rows = np.column_stack(
        (
            np.ones(8),
            np.arange(8),
            np.arange(1, 9) * 0.3,
            planned,
        )
    )

    action = _frozen_plan_action(rows, decision_step=1, before=before)

    assert action.full.shape == (1, 16, 11)
    assert np.allclose(plan_waypoints(action.policy, before), planned, rtol=0.0, atol=1e-6)


def test_frozen_plan_action_rejects_incomplete_decision_rows():
    before = {
        "position": np.zeros(3, dtype=np.float32),
        "velocity_reference": np.zeros(3, dtype=np.float32),
        "yaw": 0.0,
    }
    with pytest.raises(ValueError, match="exactly k=0..7"):
        _frozen_plan_action(np.zeros((7, 14)), decision_step=1, before=before)


def test_bank_observation_reconstructs_required_pointcloud_metadata():
    point_cloud = np.ones((1, 8, 4096, 3), dtype=np.float32)
    agent_pos = np.zeros((1, 8, 9), dtype=np.float32)
    episode_hash = "e" * 64
    metadata = []
    for frame in range(8):
        metadata.append(
            {
                "total_rays": 4096,
                "valid_hit_rays": 4096,
                "no_ray_hit_fraction": 0.0,
                "used_far_plane_fallback": False,
                "finite": True,
                "episode_identity_hash": episode_hash,
                "decision_index": frame,
                "raw_array_hash": canonical_array_hash(point_cloud[0, frame]),
                "coordinate_frame": "body",
            }
        )
    snapshot = {
        "pointcloud_meta_history": metadata,
        "episode_identity_hash": episode_hash,
        "context_decision_index": 7,
    }
    observation = _bank_observation(
        {"agent_pos": agent_pos, "point_cloud": point_cloud}, 0, snapshot
    )
    assert all(isinstance(item, PointCloudGenerationMeta) for item in observation["pointcloud_meta_history"])
    validate_metadata_history(
        observation["point_cloud"],
        observation["pointcloud_meta_history"],
        episode_identity_hash=episode_hash,
        decision_index=7,
        max_no_ray_hit_fraction=0.99,
    )


def _make_stages(tmp_path):
    b0, b1, b2 = (tmp_path / name for name in ("b0", "b1", "b2"))
    for path in (b0, b1, b2):
        path.mkdir()

    strata = {
        name: {
            "candidate_task_count": 0,
            "selected_count": 0,
            "status": "UNDERPOWERED_STRATUM",
            "underpowered_reason": "fixture",
        }
        for name in (
            "S1_SEED1_RESCUE",
            "S2_SEED3_BREAK",
            "S3_STABLE_FAILURE",
            "S4_CLEAR_STABLE_SUCCESS",
        )
    }
    write_strict_json(
        b0 / "stratum_manifest.json",
        {"schema": "routea_b0_stratum_manifest_v1", "strata": strata},
    )
    write_numeric_npz(
        b0 / "observation_bank.npz",
        {
            "agent_pos": np.zeros((1, 8, 9), np.float32),
            "point_cloud": np.zeros((1, 8, 4, 3), np.float32),
        },
        max_total_bytes=4096,
    )
    write_strict_json(
        b0 / "observation_bank_manifest.json",
        {"schema": "routea_b0_observation_bank_v1", "snapshots": []},
    )
    write_strict_json(b0 / "tripwire_verification.json", {"status": "READ_ONLY_VERIFIED"})
    publish_stage_manifest(b0, SOURCE_MANIFEST)

    write_strict_json(b1 / "checkpoint_comparability.json", {"status": "COMPARABLE"})
    write_csv_rows(b1 / "bc_retention_drift.csv", [], fieldnames=["checkpoint"])
    write_strict_json(b1 / "b1_summary.json", {"route": "DESCRIPTIVE_DRIFT_ONLY"})
    write_strict_json(b1 / "rng_pairing_verification.json", {"status": "VALID_NOISE_PAIRING"})
    write_strict_json(b1 / "tripwire_verification.json", {"status": "READ_ONLY_VERIFIED"})
    publish_stage_manifest(b1, SOURCE_MANIFEST)

    write_csv_rows(b2 / "b2_sentinel_pairs.csv", [], fieldnames=["policy"])
    write_strict_json(
        b2 / "goal_sensitivity_summary.json",
        {"status": "SENTINEL_NORMAL_B1_NORMAL_TRIGGER_D1F", "strata": {}},
    )
    write_strict_json(b2 / "rng_pairing_verification.json", {"status": "VALID_NOISE_PAIRING"})
    write_strict_json(b2 / "tripwire_verification.json", {"status": "READ_ONLY_VERIFIED"})
    publish_stage_manifest(b2, SOURCE_MANIFEST)
    return b0, b1, b2


def test_finalize_requires_exact_stage_hashes_and_minimum_outputs(tmp_path):
    b0, b1, b2 = _make_stages(tmp_path)
    output = tmp_path / "final"
    result = _run(
        "finalize",
        "--b0-dir",
        b0,
        "--b1-dir",
        b1,
        "--b2-dir",
        b2,
        "--output-dir",
        output,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert MINIMUM_FINAL_OUTPUTS <= {path.name for path in output.iterdir()}
    verdict = (output / "verdict.json").read_text(encoding="utf-8")
    assert "NaN" not in verdict
    assert "Infinity" not in verdict
    assert json.loads(verdict)["evidence_scope"] == "lightweight_kinematic_network_forward"


def test_finalize_rejects_modified_b0_bank_without_partial_output(tmp_path):
    b0, b1, b2 = _make_stages(tmp_path)
    with (b0 / "observation_bank.npz").open("ab") as handle:
        handle.write(b"tamper")
    output = tmp_path / "final"
    result = _run(
        "finalize",
        "--b0-dir",
        b0,
        "--b1-dir",
        b1,
        "--b2-dir",
        b2,
        "--output-dir",
        output,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["status"] == "STAGE_INPUT_HASH_MISMATCH"
    assert not output.exists()


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("SNAPSHOT_REPLAY_MISMATCH", 10),
        ("CHECKPOINT_CONTRACT_MISMATCH", 11),
        ("INVALID_NOISE_PAIRING", 12),
        ("INVALID_OBSERVATION_MUTATION", 12),
        ("INVALID_STATE_MUTATION", 13),
        ("MEASUREMENT_INVALID", 14),
    ],
)
def test_scientific_failure_exit_code_matrix(status, code):
    failure = _failure(status, "fixture")
    assert isinstance(failure, DiagnosticFailure)
    assert failure.exit_code == code
    assert EXIT_CODES[status] == code
