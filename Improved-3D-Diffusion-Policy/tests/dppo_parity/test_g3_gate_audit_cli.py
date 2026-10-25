"""G3 (design §5.5): 10k gate-audit CLI — RED tests.

The CLI wires the production admission path end-to-end (manifest ->
admission manifest rl_train buckets -> routea train pool -> real adapter ->
10k resets) and writes G3_GATE_REPORT.json. The role-proportion gate is
only asserted with --assert-role-proportion (statistically meaningless
below the 10k scale: ~5.9pp std at n=60); the other seven gates are always
asserted.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_routea_train_pool,
    entry_metadata_hash,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "g3_10k_reset_gate_audit.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("g3_10k_reset_gate_audit", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_world(worlds_dir, name: str, obstacles: list) -> None:
    path = worlds_dir / f"{name}.world.obstacles.json"
    path.write_text(json.dumps({"world_id": name, "obstacles": obstacles}))


def _setup(tmp_path):
    for i, role in enumerate(("anticipatory", "dense_recovery",
                              "support_expansion", "yaw_lateral_recovery")):
        _write_world(tmp_path, f"routeA_rl_v1__{role}__rl_train__f__{i:06d}", [
            {"name": f"c{i}", "shape": "cylinder", "x": 4.0 - i, "y": 0.0,
             "radius": 1.0, "height": 3.0},
        ])
    entries = []
    for i, role in enumerate(("anticipatory", "dense_recovery",
                              "support_expansion", "yaw_lateral_recovery")):
        entries.append({
            "world_uid": f"routeA_rl_v1__{role}__rl_train__f__{i:06d}",
            "split": "rl_train", "role_generated": role,
            "role_validated": role, "role_active": role,
            "admission_status": f"admitted_{role}",
            "bc_fm_sr": 0.5, "bc_fm_cr": 0.1, "ego_sr": 0.8,
        })
    manifest_path = tmp_path / "admission_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "routeA_rl_env_v1", "calibration_date": "2026-07-14",
        "mode": "real_kinematic", "stats": {}, "results": entries,
    }))
    assert len(build_routea_train_pool(str(manifest_path), str(tmp_path))) == 4
    return manifest_path


def test_cli_writes_gate_report_at_small_scale(tmp_path):
    """run() at n=60: report written, seven always-asserted gates PASS,
    role proportion recorded but not asserted."""
    mod = _load_cli()
    manifest = _setup(tmp_path)
    out = tmp_path / "G3_GATE_REPORT.json"
    code = mod.run(
        manifest=str(manifest), worlds_dir=str(tmp_path),
        n_resets=60, seed=0, out=str(out), device="cpu",
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert report["n_resets"] == 60
    # R7-5 (a942f32): obs window 8 -> 3 (contracts.py obs_steps=3) —
    # n_frames = n_resets * obs_steps = 60 * 3.
    assert report["n_frames"] == 60 * 3
    assert sum(report["role_counts"].values()) == 60
    assert report["role_counts"]["yaw_lateral_recovery"] > 0
    for gate in ("invalid_pc", "unvalidated_fallback", "reachability",
                 "clearance", "role_unknown", "eval_leakage", "hash_missing"):
        assert report["gates"][gate] == "PASS", gate
    assert "role_proportion" in report["gates"]
    assert report["role_proportion_asserted"] is False


def test_cli_gate_ok_logic():
    """Exit decision: FAIL of any asserted gate blocks; an unasserted
    role-proportion FAIL does not (10k-scale gate, design §5.5)."""
    mod = _load_cli()
    ok = {"gates": {g: "PASS" for g in mod.GATE_NAMES}}
    assert mod.gates_ok(ok, assert_role_proportion=False)
    assert mod.gates_ok(ok, assert_role_proportion=True)

    bad = {k: dict(v) if k == "gates" else v for k, v in ok.items()}
    bad["gates"]["hash_missing"] = "FAIL"
    assert not mod.gates_ok(bad, assert_role_proportion=False)

    prop_fail = {k: dict(v) if k == "gates" else v for k, v in ok.items()}
    prop_fail["gates"]["role_proportion"] = "FAIL"
    assert mod.gates_ok(prop_fail, assert_role_proportion=False)
    assert not mod.gates_ok(prop_fail, assert_role_proportion=True)


def test_cli_assert_role_proportion_requires_production_scale(tmp_path):
    """--assert-role-proportion below the 10k production scale is a misuse
    (n=60 std ~5.9pp) and rejected, not silently accepted."""
    mod = _load_cli()
    manifest = _setup(tmp_path)
    out = tmp_path / "G3_GATE_REPORT.json"
    with pytest.raises(ValueError, match="production scale"):
        mod.run(manifest=str(manifest), worlds_dir=str(tmp_path),
                n_resets=60, seed=0, out=str(out), device="cpu",
                assert_role_proportion=True)


def test_cli_missing_manifest_fails(tmp_path):
    """A missing admission manifest is a hard error (no silent empty run)."""
    mod = _load_cli()
    with pytest.raises(FileNotFoundError):
        mod.run(manifest=str(tmp_path / "nope.json"),
                worlds_dir=str(tmp_path), n_resets=10, seed=0,
                out=str(tmp_path / "G3_GATE_REPORT.json"), device="cpu")
