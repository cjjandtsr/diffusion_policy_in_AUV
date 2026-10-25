"""G3 (design §5.5): 10k-reset gate audit — RED tests.

The gate runs the REAL admission path (RoleSampler -> TaskSpecRegistry ->
UAVVectorEnvAdapter.reset) over n resets and re-verifies the admission
output independently (clearance + A* reachability of the admitted pair,
PC validity of the returned frame, role bookkeeping, split leakage, hash
completeness). The proportion gate (<=2pp vs targets) is a pure function
tested at the 10k scale separately — 60-sample runs cannot assert it.
"""

import json

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    AdmissionExhausted,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.gate_audit import (
    proportion_gate,
    run_reset_gate_audit,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_train_pool,
    build_task_specs,
    entry_metadata_hash,
)


def _write_world(worlds_dir, name: str, obstacles: list) -> str:
    path = worlds_dir / f"{name}.world.obstacles.json"
    path.write_text(json.dumps({"world_id": name, "obstacles": obstacles}))
    return str(path)


def _sparse_worlds(tmp_path):
    _write_world(tmp_path, "routeA_rl_v1__anticipatory__rl_train__f__000000", [
        {"name": "c1", "shape": "cylinder", "x": 4.0, "y": 0.0,
         "radius": 1.2, "height": 4.0},
    ])
    _write_world(tmp_path, "routeA_rl_v1__dense_recovery__rl_train__b__000001", [
        {"name": "c2", "shape": "cylinder", "x": -5.0, "y": 6.0,
         "radius": 0.9, "height": 3.0},
    ])
    _write_world(tmp_path, "routeA_rl_v1__support_expansion__rl_train__c__000002", [
        {"name": "c3", "shape": "cylinder", "x": 6.0, "y": -4.0,
         "radius": 1.0, "height": 3.5},
    ])
    _write_world(tmp_path, "routeA_rl_v1__yaw_lateral_recovery__rl_train__d__000003", [
        {"name": "c4", "shape": "cylinder", "x": -3.0, "y": -5.0,
         "radius": 0.8, "height": 2.5},
    ])


def _entries() -> list:
    base = {"bc_fm_sr": 0.5, "bc_fm_cr": 0.1, "ego_sr": 0.8,
            "admission_status": "admitted", "role_active": None}
    return [
        dict(base, world_uid="routeA_rl_v1__anticipatory__rl_train__f__000000",
             split="rl_train", role_generated="anticipatory",
             role_validated="anticipatory",
             # production derivation: role = role_active or role_validated
             role="anticipatory",
             start_world=[-8.0, 0.0, 2.0], goal_world=[8.0, 0.0, 2.0],
             start_yaw=0.0),
        dict(base, world_uid="routeA_rl_v1__dense_recovery__rl_train__b__000001",
             split="rl_train", role_generated="dense_recovery",
             role_validated="dense_recovery",
             role="dense_recovery",
             start_world=[-8.0, 0.0, 2.0], goal_world=[8.0, 0.0, 2.0],
             start_yaw=0.0),
        dict(base, world_uid="routeA_rl_v1__support_expansion__rl_train__c__000002",
             split="rl_train", role_generated="support_expansion",
             role_validated="support_expansion",
             role="support_expansion",
             start_world=[-8.0, 0.0, 2.0], goal_world=[8.0, 0.0, 2.0],
             start_yaw=0.0),
        dict(base, world_uid="routeA_rl_v1__yaw_lateral_recovery__rl_train__d__000003",
             split="rl_train", role_generated="yaw_lateral_recovery",
             role_validated="yaw_lateral_recovery",
             role="yaw_lateral_recovery",
             start_world=[-8.0, 0.0, 2.0], goal_world=[8.0, 0.0, 2.0],
             start_yaw=0.0),
    ]


def _setup(tmp_path, n_resets: int = 60, seed: int = 0):
    _sparse_worlds(tmp_path)
    entries = _entries()
    manifest_path = tmp_path / "admission_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "routeA_rl_env_v1", "calibration_date": "2026-07-14",
        "mode": "real_kinematic", "stats": {},
        "results": [{k: v for k, v in e.items()} for e in entries],
    }))
    pool = build_routea_train_pool(str(manifest_path), str(tmp_path))
    assert len(pool) == 4, "all four role worlds must be in the pool"
    specs, _ = build_task_specs([dict(e, manifest_entry_hash=entry_metadata_hash(e))
                                 for e in entries])
    # Composite-key rule-6 records built from the specs (admission_id
    # defaults guaranteed to match the registry).
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs
    }
    contract = RLSystemContract()
    adapter = build_adapter(
        contract, worlds_dir=str(tmp_path), manifest_path=str(manifest_path),
        manifest_records=records, device="cpu", max_episode_steps=30,
    )
    return adapter, specs, contract


def test_proportion_gate_at_10k_scale():
    """Proportion gate: exact targets pass at 10k; a 3pp shift fails."""
    n = 10000
    exact = {r: round(t * n) for r, t in DEFAULT_ROLE_TARGETS.items()}
    exact["anticipatory"] += (n - sum(exact.values()))  # fix rounding
    verdict, deltas = proportion_gate(exact, n, DEFAULT_ROLE_TARGETS)
    assert verdict == "PASS"
    assert all(abs(d) <= 0.02 for d in deltas.values())
    shifted = dict(exact)
    shifted["anticipatory"] = exact["anticipatory"] - 300  # -3pp
    shifted["support_expansion"] = exact["support_expansion"] + 300
    verdict2, deltas2 = proportion_gate(shifted, n, DEFAULT_ROLE_TARGETS)
    assert verdict2 == "FAIL"
    assert abs(deltas2["anticipatory"]) > 0.02


def test_run_reset_gate_audit_real_adapter(tmp_path):
    """60 resets through the REAL adapter: all gate counters stay zero,
    roles bookkept, frame/role/leak/hash audits run."""
    adapter, specs, contract = _setup(tmp_path, n_resets=60, seed=0)
    metrics = run_reset_gate_audit(adapter, specs, n_resets=60, seed=0)
    assert metrics["n_resets"] == 60
    # R7-5: every window frame is independently validated —
    # n_frames = n_resets * obs_steps
    assert metrics["n_frames"] == 60 * contract.obs_steps
    assert metrics["invalid_pc"] == 0
    assert metrics["unvalidated_fallback"] == 0
    assert metrics["reachability_fail"] == 0
    assert metrics["clearance_fail"] == 0
    assert metrics["role_unknown"] == 0
    assert metrics["eval_leakage"] == 0
    assert metrics["hash_missing"] == 0
    assert metrics["admission_blocked"] == 0
    # R7-5 window gates on the honest production path: full window valid,
    # dual hashes present and matching (source vs manifest entry, runtime
    # vs the record's episode hash).
    assert metrics["invalid_window"] == 0
    assert metrics["source_hash_missing"] == 0
    assert metrics["source_hash_mismatch"] == 0
    assert metrics["runtime_hash_mismatch"] == 0
    assert sum(metrics["role_counts"].values()) == 60
    assert metrics["role_counts"]["yaw_lateral_recovery"] > 0  # 0.10 target
    # the pure proportion verdict is computed but NOT asserted at n=60
    assert metrics["gates"]["invalid_pc"] == "PASS"
    assert metrics["gates"]["reachability"] == "PASS"
    assert metrics["gates"]["clearance"] == "PASS"
    assert metrics["gates"]["invalid_window"] == "PASS"
    assert metrics["gates"]["source_hash_missing"] == "PASS"
    assert metrics["gates"]["source_hash_mismatch"] == "PASS"
    assert metrics["gates"]["runtime_hash_mismatch"] == "PASS"


def test_run_reset_gate_audit_role_bookkeeping(tmp_path):
    """Role in reset info must equal the sampled role; unknown roles counted."""
    adapter, specs, contract = _setup(tmp_path, n_resets=60, seed=7)
    metrics = run_reset_gate_audit(adapter, specs, n_resets=60, seed=7)
    assert metrics["role_unknown"] == 0
    assert metrics["role_counts"]["anticipatory"] + \
        metrics["role_counts"]["dense_recovery"] + \
        metrics["role_counts"]["support_expansion"] + \
        metrics["role_counts"]["yaw_lateral_recovery"] == 60


def test_run_reset_gate_audit_missing_role_worlds(tmp_path):
    """R2.1: a role with NO task specs hard-fails at sampler construction —
    the shared EpisodeTaskSampler refuses to fall back to other roles."""
    adapter, specs, contract = _setup(tmp_path, n_resets=60, seed=0)
    # drop support_expansion + yaw_lateral_recovery specs -> only 2 roles available
    restricted = [s for s in specs if s.role in ("anticipatory", "dense_recovery")]
    with pytest.raises(ValueError):
        run_reset_gate_audit(adapter, restricted, n_resets=60, seed=0)


# ═══════════════════════════════════════════════════════════════════
# Real gate coverage. RED contracts — previously failing: fallback was a
# dead counter (never incremented), admission_blocked was counted but
# never gated, hash mismatch was swallowed by AdmissionExhausted, and
# dense path_stretch failures were invisible.
# ═══════════════════════════════════════════════════════════════════

def _all_far_frame(far: float, n_pts: int = 4096) -> np.ndarray:
    """The frozen fallback collapse signature: every row at (far, far, far)."""
    return np.full((n_pts, 3), float(far), dtype=np.float32)


def test_admission_blocked_is_a_real_gate(tmp_path):
    """AdmissionExhausted must FAIL the run: blocked > 0 -> gate FAIL
    (previously blocked was counted but never entered gates)."""
    import numpy as _np

    adapter, specs, _ = _setup(tmp_path, n_resets=60, seed=0)

    def _exhausted(spec, seed):
        raise AdmissionExhausted("forced for RED test")

    adapter._sample_admitted = _exhausted
    metrics = run_reset_gate_audit(adapter, specs, n_resets=60, seed=0)
    assert metrics["admission_blocked"] == 60
    assert metrics["gates"]["admission_blocked"] == "FAIL"


def test_unvalidated_fallback_counter_has_a_real_source(tmp_path):
    """The legacy all-far collapse frame must count as unvalidated fallback
    and fail the gate (previously the counter was initialized to 0 and never
    incremented anywhere — structurally always PASS)."""
    adapter, specs, _ = _setup(tmp_path, n_resets=60, seed=0)
    real_reset = adapter.reset

    def reset_with_fallback(specs_, *, seeds):  # keyword 'seeds' matches gate_audit's call
        obs, info = real_reset(specs_, seeds)
        obs["point_cloud"] = np.stack(
            [_all_far_frame(adapter.cam_far)] * 8
        )[None]  # (1, 8, 4096, 3)
        return obs, info

    adapter.reset = reset_with_fallback
    metrics = run_reset_gate_audit(adapter, specs, n_resets=60, seed=0)
    # R7-5: the collapse signature is counted per CHECKED window frame —
    # 60 resets x obs_steps all-far frames.
    assert metrics["unvalidated_fallback"] == 60 * 3
    assert metrics["gates"]["unvalidated_fallback"] == "FAIL"


def test_manifest_hash_mismatch_is_a_real_gate(tmp_path):
    """A spec whose manifest_entry_hash does not match the recomputed
    source hash must FAIL the run (previously the mismatch surfaced only as
    repeated rule-6 admissions failures -> AdmissionExhausted -> blocked,
    and the CLI could still exit 0)."""
    adapter, specs, _ = _setup(tmp_path, n_resets=60, seed=0)
    expected = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            "tampered-source-hash"
        for s in specs
    }
    metrics = run_reset_gate_audit(
        adapter, specs, n_resets=60, seed=0, expected_hashes=expected
    )
    assert metrics["manifest_hash_mismatch"] == 60
    assert metrics["gates"]["manifest_hash_mismatch"] == "FAIL"


def test_dense_path_stretch_failure_is_a_real_gate(tmp_path):
    """dense_recovery admission attempts that EXHAUST on path_stretch must
    be counted at the adapter and fail the run (previously invisible).
    Per-attempt stretch rejections that re-sample successfully are normal
    admission behavior and are NOT counted — the fake checker below fails
    every attempt, so every dense attempt exhausts on stretch."""
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import AdmissionResult

    adapter, specs, _ = _setup(tmp_path, n_resets=60, seed=0)
    calls = []

    def stretch_failing_checker(uid):
        def check(self, start, goal, role, spec=None):
            calls.append(role)
            return AdmissionResult(
                pass_=False,
                reasons=("path_stretch 2.50 outside (1.1, 1.8)",),
            )

        def raise_if_exhausted(self, attempts):
            # mirror the real AdmissionChecker: rule-7 hard stop
            raise AdmissionExhausted(
                f"admission failed after {attempts} attempts"
            )

        return type(
            "FakeChecker", (), {"check": check, "raise_if_exhausted": raise_if_exhausted}
        )()

    adapter._checker_for = stretch_failing_checker
    metrics = run_reset_gate_audit(adapter, specs, n_resets=60, seed=0)
    assert metrics["admission_blocked"] > 0  # every dense attempt exhausted
    assert metrics["dense_path_stretch_fail"] > 0
    assert metrics["gates"]["dense_path_stretch_fail"] == "FAIL"
    assert any(r == "dense_recovery" for r in calls)


def test_cli_hash_mismatch_exits_nonzero(tmp_path, monkeypatch):
    """End-to-end CLI negative: a spec whose source hash diverges from the
    recomputed manifest entry must FAIL the manifest_hash_mismatch gate and
    exit NONZERO.

    RED contract (before the CLI rework): the CLI copied each spec's OWN
    hash into the expected records, so a tampered spec hash matched itself
    — the mismatch was masked and the CLI exited 0."""
    import importlib.util
    from dataclasses import replace
    from pathlib import Path

    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import EpisodeTaskSpec

    _sparse_worlds(tmp_path)
    entries = _entries()
    manifest_path = tmp_path / "admission_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "routeA_rl_env_v1", "calibration_date": "2026-07-14",
        "mode": "real_kinematic", "stats": {},
        "results": [{k: v for k, v in e.items()} for e in entries],
    }))
    cli_path = Path(__file__).resolve().parents[2] / "scripts/g3_10k_reset_gate_audit.py"
    spec_mod = importlib.util.spec_from_file_location("g3_cli_under_test", str(cli_path))
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    real_build_task_specs = mod.build_task_specs

    def tampered_hash_specs(entries_):
        train, eval_ = real_build_task_specs(entries_)
        return [replace(s, manifest_entry_hash="tampered-source-hash")
                for s in train], eval_

    monkeypatch.setattr(mod, "build_task_specs", tampered_hash_specs)

    out = tmp_path / "report.json"
    code = mod.run(manifest=str(manifest_path), worlds_dir=str(tmp_path),
                   n_resets=60, seed=0, out=str(out))
    report = json.loads(out.read_text())
    assert code != 0, "tampered spec hash must exit nonzero"
    assert report["manifest_hash_mismatch"] > 0
    assert report["gates"]["manifest_hash_mismatch"] == "FAIL"


def test_g3_cli_gates_cover_rework_gates():
    """R2 rework (审核裁决 P1-1): the CLI must gate admission_blocked,
    manifest hash mismatch and dense path_stretch, and run() must pass the
    recomputed source-entry hashes (expected_hashes) into the audit.

    RED contract (before the CLI rework): GATE_NAMES lists only the 8
    original gates — admission_blocked was counted-but-ungated, hash
    mismatch was masked by AdmissionExhausted, and dense stretch was
    invisible — and run() does not wire expected_hashes.
    """
    import ast
    from pathlib import Path

    cli_path = Path(__file__).resolve().parents[2] / "scripts/g3_10k_reset_gate_audit.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"))

    names = []
    run_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "GATE_NAMES":
                    names = [elt.value for elt in node.value.elts]
        elif isinstance(node, ast.FunctionDef) and node.name == "run":
            run_fn = node
    assert {"admission_blocked", "manifest_hash_mismatch",
            "dense_path_stretch_fail"} <= set(names), (
        f"GATE_NAMES must gate the rework counters, got: {names}"
    )

    call_args = []
    for node in ast.walk(run_fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "run_reset_gate_audit":
                call_args = [(a.arg, a.value) for a in node.keywords]
    assert any(arg == "expected_hashes" for arg, _ in call_args), (
        "run() must pass expected_hashes into run_reset_gate_audit"
    )


# ═══════════════════════════════════════════════════════════════════
# The G3 report is not self-bound — it lacks tree_hash / command /
# exit_code / manifest_sha256 / world_inputs_hash, and n_frames=80000
# reads like 80k independently validated frames when it is an 8-row obs
# window per validated reset. RED contracts — a report without
# provenance fails these (KeyError).
# ═══════════════════════════════════════════════════════════════════

def _load_g3_cli():
    import importlib.util
    from pathlib import Path

    cli_path = Path(__file__).resolve().parents[2] / "scripts/g3_10k_reset_gate_audit.py"
    spec = importlib.util.spec_from_file_location("g3_cli_prov", str(cli_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _g3_manifest(tmp_path):
    _sparse_worlds(tmp_path)
    entries = _entries()
    manifest_path = tmp_path / "admission_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "routeA_rl_env_v1", "calibration_date": "2026-07-14",
        "mode": "real_kinematic", "stats": {},
        "results": [{k: v for k, v in e.items()} for e in entries],
    }))
    return str(manifest_path)


def test_g3_cli_report_self_binds_provenance(tmp_path):
    """审核裁决 P2: the report must self-bind its run origin — tree hash,
    command, exit code, manifest sha256, worlds input-set hash — and
    spell out the frame accounting (validated_reset_frames vs
    observation_window_rows)."""
    manifest = _g3_manifest(tmp_path)
    mod = _load_g3_cli()
    out = tmp_path / "report.json"
    code = mod.run(manifest=manifest, worlds_dir=str(tmp_path),
                   n_resets=60, seed=0, out=str(out))
    assert code == 0
    report = json.loads(out.read_text())
    prov = report["provenance"]
    assert len(prov["tree_hash"]) == 40, prov["tree_hash"]
    assert prov["command"].strip() != ""
    assert prov["exit_code"] == code
    assert len(prov["manifest_sha256"]) == 64, prov["manifest_sha256"]
    assert len(prov["world_inputs_hash"]) == 64, prov["world_inputs_hash"]
    # honest frame accounting (R7-5): 60 resets validated; each reset's
    # FULL obs_steps window is validated — every window row is an
    # independently validated frame (3 per reset).
    assert prov["validated_reset_frames"] == 60
    assert prov["observation_window_rows"] == 60 * 3
    # the metric-level keys survive unchanged (compat with existing tests)
    assert report["n_resets"] == 60
    assert report["n_frames"] == 60 * 3


def test_g3_cli_provenance_hard_fails_when_git_unavailable(tmp_path, monkeypatch):
    """审核裁决 P2: provenance failures must hard-fail (raise) and never
    write a report with a placeholder tree hash."""
    manifest = _g3_manifest(tmp_path)
    mod = _load_g3_cli()
    import subprocess as sp

    real_run = sp.run

    class _FakeCompleted:
        returncode = 1
        stdout = ""

    def fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "git":
            return _FakeCompleted()
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(sp, "run", fake_run)
    out = tmp_path / "report.json"
    with pytest.raises(RuntimeError):
        mod.run(manifest=manifest, worlds_dir=str(tmp_path),
                n_resets=60, seed=0, out=str(out))
    assert not out.exists(), "a failed provenance must not write a report"
