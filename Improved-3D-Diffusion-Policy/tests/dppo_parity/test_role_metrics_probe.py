"""R2 rework P1-2: production role-metrics probe — versioned + truthful.

Audit finding (P2-10): "ROLE-TRANSITION 无 production metrics JSON" — the
RTM row cited G3_GATE_REPORT_v2.json, but that report carries reset
counts only, never per-role attempted/completed/voided/transition. These
tests pin the VERSIONED probe (scripts/role_metrics_probe.py) that runs
the REAL production path — EpisodeTaskSampler -> UAVVectorEnvAdapter ->
PointCloudValidityGate -> run_e2e_batch role_stats — and persists the
7-field role stats JSON bound to command/seed/exit-code/tree hash:

- the four role buckets are driven by the real sampler (no mock env);
- accounting invariants: attempted == completed + voided; the sum of
  transition counts equals env_steps (voided episodes contribute no
  transitions — PC-VALID's "invalid buffer transition=0");
- a blocked-world manifest (admission exhausts) must FAIL — a partial
  batch is never reported as pass (negative pin);
- P2-8: n_episodes % n_envs != 0 must hard-fail at startup.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from diffusion_policy_3d.rl.dppo_uav.integration import build_routea_train_pool

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "role_metrics_probe.py"
)


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "role_metrics_probe", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_world(worlds_dir, name: str, obstacles: list) -> None:
    path = worlds_dir / f"{name}.world.obstacles.json"
    path.write_text(json.dumps({"world_id": name, "obstacles": obstacles}))


def _setup(tmp_path, blocked: bool = False):
    for i, role in enumerate(("anticipatory", "dense_recovery",
                              "support_expansion", "yaw_lateral_recovery")):
        if blocked:
            # radius-30 cylinder covers the whole C2 box: admission exhausts
            obstacles = [{"name": "mega", "shape": "cylinder", "x": 0.0,
                          "y": 0.0, "radius": 30.0, "height": 10.0}]
        else:
            obstacles = [{"name": f"c{i}", "shape": "cylinder",
                          "x": 4.0 - i, "y": 0.0, "radius": 1.0,
                          "height": 3.0}]
        _write_world(tmp_path, f"routeA_rl_v1__{role}__rl_train__f__{i:06d}",
                     obstacles)
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


def _run_probe(tmp_path, episodes, n_envs, max_ep_len, seed, manifest,
               tag: str) -> tuple:
    out = tmp_path / f"{tag}.json"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT),
         "--manifest", str(manifest),
         "--worlds-dir", str(tmp_path),
         "--episodes", str(episodes),
         "--n-envs", str(n_envs),
         "--max-ep-len", str(max_ep_len),
         "--seed", str(seed),
         "--output", str(out)],
        capture_output=True, text=True, timeout=900, cwd=str(_SCRIPT.parents[1]),
    )
    assert proc.returncode in (0, 1), f"probe crashed:\n{proc.stderr}"
    return proc.returncode, json.loads(out.read_text())


def test_probe_persists_truthful_role_stats(tmp_path):
    """Real sampler + real adapter: 7-field role stats, accounting
    invariants, all four role buckets attempted, evidence bound."""
    manifest = _setup(tmp_path)
    rc, ev = _run_probe(tmp_path, episodes=16, n_envs=2, max_ep_len=4,
                        seed=0, manifest=manifest, tag="happy")
    assert rc == 0, ev
    assert ev["pass"] is True
    rs = ev["role_stats"]
    for key in ("attempted_episode_counts", "completed_episode_counts",
                "voided_episode_counts", "transition_counts",
                "attempted_episode_proportions",
                "completed_episode_proportions",
                "transition_proportions"):
        assert key in rs, f"missing role_stats key {key}"
    # real sampler drives every role bucket
    attempted = rs["attempted_episode_counts"]
    for role in ("anticipatory", "dense_recovery",
                 "support_expansion", "yaw_lateral_recovery"):
        assert attempted[role] > 0, f"role {role} never attempted"
    # accounting invariants (PC-VALID: voided episodes enter no transitions).
    # Multi-slot: attempted counts RESETS; episodes still in-flight when the
    # collection bound is reached are dropped (neither completed nor
    # voided), so attempted >= completed + voided is the truthful form.
    assert ev["accounting"]["attempted_total"] >= 16
    assert ev["accounting"]["attempted_total"] >= (
        ev["accounting"]["completed_total"] + ev["accounting"]["voided_total"]
    )
    assert ev["accounting"]["in_flight_at_collection_stop"] >= 0
    assert ev["accounting"]["transitions_total"] == ev["accounting"]["env_steps"]
    assert ev["invariants"]["invalid_frames_zero"] is True
    # P2-10 evidence binding
    assert "role_metrics_probe.py" in ev["command"]
    assert len(ev["tree_hash"]) == 40
    assert ev["seed"] == 0


def test_probe_blocked_world_never_passes(tmp_path):
    """Admission exhausts: the probe fails, never a partial-batch pass."""
    manifest = _setup(tmp_path, blocked=True)
    rc, ev = _run_probe(tmp_path, episodes=8, n_envs=2, max_ep_len=4,
                        seed=0, manifest=manifest, tag="blocked")
    assert ev["pass"] is False
    assert rc == 1
    assert ev["exit_code"] == 1
    assert ev.get("error", "") != ""


def test_probe_episode_multiple_hard_fail(tmp_path):
    """P2-8: n_episodes % n_envs != 0 must fail at startup (exit 1)."""
    manifest = _setup(tmp_path)
    out = tmp_path / "bad.json"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT),
         "--manifest", str(manifest),
         "--worlds-dir", str(tmp_path),
         "--episodes", "6", "--n-envs", "4", "--max-ep-len", "4",
         "--seed", "0", "--output", str(out)],
        capture_output=True, text=True, timeout=300, cwd=str(_SCRIPT.parents[1]),
    )
    assert proc.returncode == 1
    assert not out.exists(), "startup validation must fail before writing"


# ═══════════════════════════════════════════════════════════════════
# The probe claimed n_envs=4 in the evidence but build_adapter was never
# passed n_envs (UAVVectorEnvAdapter defaults to 1) and max_episode_steps
# was hard-coded to 30, ignoring max_ep_len. RED contracts — these fail:
#   - evidence has no requested_n_envs / resolved_n_envs fields;
#   - the real adapter runs a single slot whatever the CLI claims;
#   - invariants lack the completeness assertions.
# ═══════════════════════════════════════════════════════════════════

def test_probe_resolves_real_multi_slot_adapter(tmp_path):
    """审核裁决 P1: the evidence must prove the REAL adapter was built
    with the requested slot count — not just a CLI arithmetic claim.

    RED: the old probe never passed n_envs to build_adapter (default 1),
    so resolved_n_envs is missing from the evidence and the batch ran on
    a single-slot adapter even when the CLI said n_envs=2."""
    manifest = _setup(tmp_path)
    rc, ev = _run_probe(tmp_path, episodes=16, n_envs=2, max_ep_len=4,
                        seed=0, manifest=manifest, tag="multislot")
    assert rc == 0, ev
    assert ev["requested_n_envs"] == 2
    assert ev["resolved_n_envs"] == 2  # the real adapter construction arg
    assert ev["resolved_n_envs"] == ev["requested_n_envs"]
    # the two-slot batch really ran to completion: every episode collected
    assert ev["accounting"]["completed_total"] == 16


def test_probe_completeness_invariants(tmp_path):
    """审核裁决 P1-1 (rework #3): the pass condition includes EXACT
    accounting at the precise refill budget — completed == n_episodes,
    attempted == completed + voided, attempted == n_episodes (over-refill
    gone) and in_flight_at_collection_stop == 0 (nothing dropped at the
    collection bound).

    RED history: the first rework version asserted attempted == n_episodes
    (sequential semantics); rework #2 generalized to attempted >= completed
    + voided (tail drop real); rework #3 makes the equality exact again —
    the precise budget keeps completed + active <= n_episodes, so at
    collection stop nothing is left in flight."""
    manifest = _setup(tmp_path)
    rc, ev = _run_probe(tmp_path, episodes=16, n_envs=2, max_ep_len=4,
                        seed=0, manifest=manifest, tag="complete")
    assert rc == 0, ev
    assert ev["accounting"]["completed_total"] == 16
    assert ev["invariants"]["completed_eq_n_episodes"] is True
    assert ev["invariants"]["attempted_eq_completed_plus_voided"] is True
    assert ev["invariants"]["attempted_eq_n_episodes"] is True
    assert ev["invariants"]["in_flight_zero"] is True
    in_flight = ev["accounting"]["in_flight_at_collection_stop"]
    assert in_flight == 0


def test_probe_wires_n_envs_and_ep_len_into_build_adapter():
    """审核裁决 P1: build_adapter must receive n_envs=n_envs and
    max_episode_steps=max_ep_len explicitly (the hard-coded 30 must be
    gone). RED: the old probe passed neither — n_envs defaulted to 1 and
    max_episode_steps was hard-coded to 30."""
    import ast

    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    call_kwargs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "build_adapter":
            call_kwargs = {
                kw.arg: (kw.value.id if isinstance(kw.value, ast.Name)
                         else ast.dump(kw.value))
                for kw in node.keywords
            }
    assert call_kwargs, "build_adapter call not found in the probe"
    assert call_kwargs.get("n_envs") == "n_envs", (
        "build_adapter must receive n_envs=n_envs")
    assert call_kwargs.get("max_episode_steps") == "max_ep_len", (
        "build_adapter must receive max_episode_steps=max_ep_len "
        "(the hard-coded 30 must be gone)")
