"""G4 (design §5.6): Gazebo—kinematic equivalence adapter non-regression
fixture — schema + integrity test.

The frozen equivalence fixture was never found (ASSUMPTION-4); this gate
uses a REPLACEMENT fixture (``replacement_adapter_non_regression_fixture``)
per design §5.6: it does NOT re-prove Gazebo equivalence ([FACT] the
equivalence holds), it only pins the frozen direct KinematicUAVEnv v2 path
(oracle) against the canonical UAVVectorEnvAdapter boundary (candidate)
over fixed inputs (world, episode config, seed, action sequences).

This test runs on CPU and validates the FIXTURE ARTIFACT only (the replay
itself runs in scripts/run_adapter_regression_gate.py, which also emits
GAZEBO_KINEMATIC_REGRESSION_REPORT.json):

- the fixture exists at the production path;
- required schema keys + the replacement-facility wording;
- tolerance is the design §5.6 recorded threshold (rtol=1e-5, atol=1e-6);
- at least the three deterministic case classes (straight / yaw+lateral /
  z+composite 11d);
- per-case fixed inputs and per-step expected outputs are present and
  shaped correctly;
- the world + obstacles files still hash to the fixture's recorded values
  (a world change invalidates the fixture until rebuilt);
- oracle/candidate source files still hash to the fixture's recorded
  values (a production change invalidates the fixture until rebuilt);
- created_commit resolves in git history (the fixture is anchored).

A FAIL here means the fixture is missing, tampered, or stale — the gate
must be rebuilt/re-run before EQUIV-REG may be claimed PASS.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FIXTURE_DIR = Path(
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/g4_adapter_regression"
)
DEFAULT_FIXTURE = FIXTURE_DIR / "ADAPTER_REGRESSION_FIXTURE.json"
DEFAULT_REPORT = FIXTURE_DIR / "GAZEBO_KINEMATIC_REGRESSION_REPORT.json"

# The production files whose behavior the fixture pins. The fixture's
# recorded sha256s are compared against these LIVE files: any production
# change to the frozen kinematic path or the adapter boundary invalidates
# the fixture until it is rebuilt and the gate re-run.
ORACLE_SOURCES = [
    "diffusion_policy_3d/env/kinematic_uav_env_v2.py",
    "diffusion_policy_3d/env/traj_utils.py",
    "diffusion_policy_3d/env/collision_geom.py",
    "diffusion_policy_3d/model/common/synth_pointcloud.py",
]
CANDIDATE_SOURCES = [
    "diffusion_policy_3d/rl/dppo_uav/environment_adapter.py",
    "diffusion_policy_3d/rl/dppo_uav/integration.py",
    "diffusion_policy_3d/rl/dppo_uav/contracts.py",
    "diffusion_policy_3d/rl/dppo_uav/validated_synth_adapter.py",
    "diffusion_policy_3d/rl/dppo_uav/planner.py",
]

CASE_IDS = ("straight_no_obstacle", "yaw_lateral", "z_composite")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _frame_content_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(frame, dtype=np.float32).tobytes()
    ).hexdigest()


def _assert_frame_sidecars(record: Dict[str, Any], label: str) -> None:
    """Each expected frame .npy must exist and hash to its recorded value."""
    for i, (fp, recorded) in enumerate(
        zip(record["frame_files"], record["pc_frame_sha256"])
    ):
        path = Path(fp)
        assert path.is_file(), f"{label} frame {i} sidecar missing: {path}"
        frame = np.ascontiguousarray(np.load(path), dtype=np.float32)
        assert frame.shape == (4096, 3), frame.shape
        live = _frame_content_sha256(frame)
        assert live == recorded, (
            f"{label} frame {i} sidecar content hash {live[:12]}… != "
            f"recorded {recorded[:12]}… (tampered or overwritten)"
        )


def _load_fixture() -> Dict[str, Any]:
    path = Path(os.environ.get("DPPO_G4_FIXTURE", str(DEFAULT_FIXTURE)))
    if not path.is_file():
        pytest.fail(
            f"G4 fixture missing: {path} — build it with "
            "scripts/build_adapter_regression_fixture.py and re-run the "
            "gate with scripts/run_adapter_regression_gate.py"
        )
    return json.loads(path.read_text())


def test_g4_fixture_schema_and_facility():
    """Fixture top-level schema + replacement-facility wording."""
    fx = _load_fixture()
    assert fx["schema_version"] == "g4-fixture-v1", fx.get("schema_version")
    assert fx["fixture_type"] == "replacement_adapter_non_regression_fixture"
    # The fixture MUST state the historical equivalence as fact and MUST
    # NOT claim it re-proves Gazebo equivalence (design §5.6).
    fact = fx["equivalence_fact"]
    assert "FACT" in fact, "equivalence fact must be marked [FACT]"
    assert "re-prove" in fact or "reprove" in fact, (
        "the fixture must disclaim re-proving Gazebo equivalence"
    )
    tol = fx["tolerance"]
    assert tol == {"rtol": 1e-5, "atol": 1e-6}, tol
    assert fx["oracle"]["name"], "oracle name missing"
    assert fx["candidate"]["name"], "candidate name missing"
    created = fx["created_commit"]
    assert len(created) == 40, f"created_commit not a full sha: {created!r}"
    probe = subprocess.run(
        ["git", "cat-file", "-e", created + "^{commit}"],
        cwd=str(REPO_ROOT), capture_output=True,
    )
    assert probe.returncode == 0, (
        f"created_commit {created} does not resolve in git history"
    )


def test_g4_fixture_source_hashes_match_live_files():
    """Fixture source pins still match the live production files."""
    fx = _load_fixture()
    for side, files in (("oracle", ORACLE_SOURCES),
                        ("candidate", CANDIDATE_SOURCES)):
        recorded = fx[side]["source_files"]
        assert set(recorded.keys()) == set(files), (
            f"{side} source list mismatch: {sorted(recorded)} vs {files}"
        )
        for rel in files:
            live = _sha256(REPO_ROOT / rel)
            assert recorded[rel] == live, (
                f"{rel} changed since the fixture was built"
                f" ({recorded[rel][:12]}… -> {live[:12]}…) — rebuild the"
                " fixture and re-run the gate"
            )


def test_g4_fixture_cases_complete():
    """The three deterministic case classes with well-shaped expected data."""
    fx = _load_fixture()
    cases = fx["cases"]
    assert [c["case_id"] for c in cases] == list(CASE_IDS), (
        f"case ids must be {CASE_IDS}, got {[c['case_id'] for c in cases]}"
    )
    for case in cases:
        # fixed world inputs are present and the on-disk world/obstacles
        # still hash to the fixture's recorded values
        world_path = Path(case["world_file"])
        obstacles_path = Path(case["obstacle_json"])
        assert world_path.is_file(), f"world file gone: {world_path}"
        assert obstacles_path.is_file(), f"obstacles file gone: {obstacles_path}"
        assert _sha256(world_path) == case["world_hash"], (
            f"world file changed since fixture build: {world_path}"
        )
        assert _sha256(obstacles_path) == case["obstacle_hash"], (
            f"obstacles file changed since fixture build: {obstacles_path}"
        )
        # fixed episode inputs
        init = case["initial_state"]
        for key in ("start", "goal", "start_yaw", "episode_config"):
            assert key in init, f"{case['case_id']} initial_state lacks {key}"
        assert np.asarray(init["start"]).shape == (3,)
        assert np.asarray(init["goal"]).shape == (3,)
        assert isinstance(init["start_yaw"], float)
        seed = case["seed"]
        assert isinstance(seed, int), f"seed must be int: {seed!r}"
        actions = np.asarray(case["actions"], dtype=np.float32)
        assert actions.ndim == 3 and actions.shape[1:] == (4, 11), actions.shape
        # expected outputs: one entry per executed step
        expected = case["expected"]
        assert len(expected) == actions.shape[0], (
            f"{case['case_id']}: {len(expected)} expected steps vs "
            f"{actions.shape[0]} action steps"
        )
        for k, exp in enumerate(expected):
            for key in ("position", "yaw", "vel", "agent_pos",
                        "action_command", "pc_frame_sha256", "hit_count",
                        "frame_files", "reward", "terminated", "truncated"):
                assert key in exp, f"step {k} lacks {key}"
            assert np.asarray(exp["position"]).shape == (3,)
            assert np.asarray(exp["vel"]).shape == (3,)
            assert np.asarray(exp["agent_pos"]).shape == (8, 9)
            assert np.array_equal(
                np.asarray(exp["action_command"]), actions[k]
            ), f"step {k} action_command != fixture actions"
            assert len(exp["pc_frame_sha256"]) == 8
            assert len(exp["hit_count"]) == 8
            assert len(exp["frame_files"]) == 8
            _assert_frame_sidecars(exp, f"{case['case_id']} step {k}")
            assert isinstance(exp["terminated"], bool)
            assert isinstance(exp["truncated"], bool)
        _assert_frame_sidecars(
            case["initial_state"]["reset_obs"], f"{case['case_id']} reset"
        )


def test_g4_gate_report_binds_fixture():
    """The gate report (once generated) binds the fixture by sha256."""
    fx_path = Path(os.environ.get("DPPO_G4_FIXTURE", str(DEFAULT_FIXTURE)))
    report_path = Path(
        os.environ.get("DPPO_G4_REPORT", str(DEFAULT_REPORT))
    )
    if not report_path.is_file():
        pytest.fail(
            f"G4 gate report missing: {report_path} — run "
            "scripts/run_adapter_regression_gate.py"
        )
    report = json.loads(report_path.read_text())
    assert report["status"] == "PASS", (
        f"G4 gate status {report['status']} — see {report_path}"
    )
    assert report["fixture_sha256"] == _sha256(fx_path), (
        "fixture file changed after the gate report was generated"
    )
    fx = _load_fixture()
    assert report["fixture"] == os.path.basename(str(fx_path))
    for side in ("oracle", "candidate"):
        assert report[f"{side}_source_hashes"] == fx[side]["source_files"]
    assert report["replacement_label"] == (
        "replacement_adapter_non_regression_fixture"
    )
    assert report["tolerance"] == {"rtol": 1e-5, "atol": 1e-6}
