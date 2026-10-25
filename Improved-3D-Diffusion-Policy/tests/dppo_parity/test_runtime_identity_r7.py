"""R7-5 runtime identity: window gate negatives + dual-hash independence.

Guide Task R7-5 Step 5 negative tests:

1. [-3] all-far, rest valid -> reject
2. [-2] NaN, rest valid -> reject
3. [-1] ray hit=0 -> reject
4. PC valid but ray report missing -> reject
5. history length/alignment inconsistent -> reject
6. source hash self-matches but runtime content changed -> reject
7. eval-only world entering the train manifest -> reject
8. valid latest-3 -> accept

Plus the buffer.add() hard-fail surface (empty / malformed / mismatched
dual hashes) and the gate-audit runtime-drift counter — the production
rejection surfaces for negative 6. All tests exercise real production
code: the window gate over production-shaped provenance mappings, the
real adapter (runtime hash from the env's ACTUAL post-reset state), the
rollout buffer hard-fail, and the 10k-reset audit counters.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.gate_audit import run_reset_gate_audit
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_train_pool,
    build_task_specs,
    entry_metadata_hash,
    manifest_train_entries,
)
from diffusion_policy_3d.rl.dppo_uav.pointcloud_gate import (
    PointCloudValidityGate,
    build_observation_window_mapping,
    frame_sha256,
    validate_policy_observation_window,
)
from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
    RolloutBuffer,
    RolloutTransition,
)

CONTRACT = RLSystemContract()
OBS_STEPS = CONTRACT.obs_steps
POINT_COUNT = CONTRACT.point_count
CAM_FAR = 10.0
H_A = CONTRACT.action_horizon
D_A = CONTRACT.action_dim
K_F = CONTRACT.ft_denoising_steps
SRC_HASH = "a" * 64
RUNTIME_HASH = "b" * 64


# ── window-gate fixtures ────────────────────────────────────────────


def _valid_frame(seed: int) -> np.ndarray:
    """Content that passes every hard condition: finite, unique, spread,
    hit > 0, not all-equal-far."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.3, 9.5, (POINT_COUNT, 3)).astype(np.float32)


def _window_mapping(
    frames,
    *,
    source_hash: str = SRC_HASH,
    runtime_hash: str = RUNTIME_HASH,
    recorded_valid=None,
    frame_hashes=None,
    hit_counts=None,
    ray_reports=None,
):
    """Production-shaped provenance mapping over ``obs_steps`` frames."""
    mapping = {
        "points": np.asarray(frames, dtype=np.float32),
        "cam_far": CAM_FAR,
        "source_manifest_hash": source_hash,
        "runtime_episode_hash": runtime_hash,
    }
    if recorded_valid is not None:
        mapping["point_cloud_valid"] = list(recorded_valid)
    if frame_hashes is not None:
        mapping["point_cloud_frame_hash"] = list(frame_hashes)
    if hit_counts is not None:
        mapping["hit_count_history"] = list(hit_counts)
    if ray_reports is not None:
        mapping["ray_report_history"] = list(ray_reports)
    return mapping


def _gate() -> PointCloudValidityGate:
    return PointCloudValidityGate(CONTRACT, phase="unit")


def _check(frames, **kwargs):
    return validate_policy_observation_window(
        _window_mapping(frames, **kwargs),
        obs_steps=OBS_STEPS,
        gate=_gate(),
    )


# ── negative 1: all-far frame in the window ─────────────────────────

def test_window_reject_all_far_frame():
    """[-3] all-far, rest valid -> whole window rejected."""
    frames = [_valid_frame(1), _valid_frame(2), _valid_frame(3)]
    frames[0] = np.full((POINT_COUNT, 3), CAM_FAR, dtype=np.float32)
    res = _check(frames, hit_counts=[4096, 4096, 4096])
    assert res.valid is False
    assert res.invalid_frame_indices == (0,)
    assert "far plane" in (res.frame_results[0].invalid_reason or "")


# ── negative 2: NaN frame in the window ─────────────────────────────

def test_window_reject_nan_frame():
    """[-2] NaN, rest valid -> whole window rejected."""
    frames = [_valid_frame(4), _valid_frame(5), _valid_frame(6)]
    frames[1][0, 0] = np.nan
    res = _check(frames, hit_counts=[4096, 4096, 4096])
    assert res.valid is False
    assert res.invalid_frame_indices == (1,)
    assert "finite_ratio" in (res.frame_results[1].invalid_reason or "")


# ── negative 3: no-ray frame in the window ──────────────────────────

def test_window_reject_no_ray_frame():
    """[-1] ray hit=0 without certification -> whole window rejected."""
    frames = [_valid_frame(7), _valid_frame(8), _valid_frame(9)]
    res = _check(frames, hit_counts=[4096, 4096, 0], ray_reports=[None, None, None])
    assert res.valid is False
    assert res.invalid_frame_indices == (2,)
    assert "hit_count <= 0" in (res.frame_results[2].invalid_reason or "")


# ── negative 4: valid flag without a certifying ray report ──────────

def test_window_reject_valid_flag_without_certificate():
    """PC claims valid but the frame needs analytic ray-geometry evidence
    (hit=0) and the report is missing -> rejected."""
    frames = [_valid_frame(10), _valid_frame(11), _valid_frame(12)]
    res = _check(
        frames,
        recorded_valid=[True, True, True],
        hit_counts=[4096, 0, 4096],
        ray_reports=[None, None, None],
    )
    assert res.valid is False
    assert res.invalid_frame_indices == (1,)
    assert "hit_count <= 0" in (res.frame_results[1].invalid_reason or "")


# ── negative 5: history length / alignment inconsistency ────────────

def test_window_reject_misaligned_history():
    """Provenance list length != obs_steps -> structurally invalid, ALL
    window frames invalid (fail-closed)."""
    frames = [_valid_frame(13), _valid_frame(14), _valid_frame(15)]
    res = _check(frames, recorded_valid=[True, True])
    assert res.valid is False
    assert res.invalid_frame_indices == (0, 1, 2)
    assert "length obs_steps" in (res.frame_results[0].invalid_reason or "")
    # hashes still echoed even on the structural failure
    assert res.source_manifest_hash == SRC_HASH
    assert res.runtime_episode_hash == RUNTIME_HASH


def test_window_reject_short_points_window():
    """points array misaligned with obs_steps -> structurally invalid."""
    res = _check([_valid_frame(16), _valid_frame(17)])
    assert res.valid is False
    assert res.invalid_frame_indices == (0, 1, 2)


# ── negative 8: valid latest-3 -> accept ────────────────────────────

def test_window_accepts_valid_latest_3():
    """Full production mapping, honest path -> accepted, hashes echoed,
    every frame hash cross-checked."""
    frames = [_valid_frame(18), _valid_frame(19), _valid_frame(20)]
    res = _check(
        frames,
        recorded_valid=[True, True, True],
        frame_hashes=[frame_sha256(f) for f in frames],
        hit_counts=[4096, 4096, 4096],
        ray_reports=[None, None, None],
    )
    assert res.valid is True
    assert res.invalid_frame_indices == ()
    assert res.source_manifest_hash == SRC_HASH
    assert res.runtime_episode_hash == RUNTIME_HASH
    assert all(fr.valid for fr in res.frame_results)


# ── negative 6: runtime content change detection ────────────────────


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
             role_validated="anticipatory", role="anticipatory",
             start_world=[-8.0, 0.0, 2.0], goal_world=[8.0, 0.0, 2.0],
             start_yaw=0.0),
        dict(base, world_uid="routeA_rl_v1__dense_recovery__rl_train__b__000001",
             split="rl_train", role_generated="dense_recovery",
             role_validated="dense_recovery", role="dense_recovery",
             start_world=[-8.0, 0.0, 2.0], goal_world=[8.0, 0.0, 2.0],
             start_yaw=0.0),
        dict(base, world_uid="routeA_rl_v1__support_expansion__rl_train__c__000002",
             split="rl_train", role_generated="support_expansion",
             role_validated="support_expansion", role="support_expansion",
             start_world=[-8.0, 0.0, 2.0], goal_world=[8.0, 0.0, 2.0],
             start_yaw=0.0),
        dict(base, world_uid="routeA_rl_v1__yaw_lateral_recovery__rl_train__d__000003",
             split="rl_train", role_generated="yaw_lateral_recovery",
             role_validated="yaw_lateral_recovery", role="yaw_lateral_recovery",
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


def test_runtime_hash_agrees_on_honest_path_and_detects_content_change(tmp_path):
    """Runtime hash is derived from the env's ACTUAL post-reset state —
    it agrees with the record on the honest path, and a change in the
    runtime state makes it disagree (the drift the buffer/audit reject)."""
    adapter, specs, _ = _setup(tmp_path)
    spec = specs[0]
    obs, info = adapter.reset([spec], seeds=[7])
    record_hash = info[0]["episode_config_hash"]
    mapping = info[0]["observation_window_mapping"]
    assert len(record_hash) == 64
    assert mapping["runtime_episode_hash"] == record_hash  # honest agreement
    assert mapping["source_manifest_hash"] == spec.manifest_entry_hash

    # Runtime content changed (the admission-time values are NOT what the
    # env ended up in): the recomputed runtime hash must diverge.
    env = adapter._envs[0]
    env._drone_pos = np.array([9.9, 9.9, 9.9], dtype=np.float32)
    drifted = adapter._runtime_episode_hash_of(0, spec, 7)
    assert drifted != record_hash


def test_buffer_rejects_drifted_runtime_hash():
    """The production rejection surface for a runtime content change:
    a transition whose runtime_episode_hash disagrees with its recorded
    episode_config_hash hard-fails in buffer.add()."""
    buf = RolloutBuffer(CONTRACT)
    t = _trans("ep1", 0, seed=1, manifest_hash="c" * 64, episode_hash="e" * 64)
    bad = replace(
        t,
        source_manifest_hash="c" * 64,          # == manifest_entry_hash
        runtime_episode_hash=RUNTIME_HASH,      # != episode_config_hash
    )
    with pytest.raises(ValueError, match="runtime_episode_hash"):
        buf.add(bad)


# ── buffer.add() hard-fail: format and pair completeness ────────────

def _trans(episode_id: str, step: int, seed: int,
           manifest_hash: str = "mh", episode_hash: str = "eh"
           ) -> RolloutTransition:
    rng = np.random.RandomState(seed)
    obs = {
        "point_cloud": rng.uniform(0.3, 9.5, (3, POINT_COUNT, 3)).astype(np.float32),
        "state": rng.uniform(-1, 1, (3, 9)).astype(np.float32),
    }
    return RolloutTransition(
        obs=obs,
        next_obs={
            "point_cloud": rng.uniform(0.3, 9.5, (3, POINT_COUNT, 3)).astype(np.float32),
            "state": rng.uniform(-1, 1, (3, 9)).astype(np.float32),
        },
        chains_policy=rng.uniform(-1, 1, (K_F + 1, H_A, D_A)).astype(np.float32),
        old_logprobs=rng.uniform(-3, 0, (K_F, H_A, D_A)).astype(np.float32),
        reward=float(seed),
        terminated=False,
        truncated=False,
        value_old=float(seed) / 10.0,
        episode_id=episode_id,
        episode_step=step,
        world_uid="w1",
        role="anticipatory",
        manifest_entry_hash=manifest_hash,
        episode_config_hash=episode_hash,
        observation_valid=True,
    )


def test_buffer_rejects_partial_dual_hash_pair():
    """Populated hashes must come together — a partial pair is a
    provenance corruption."""
    buf = RolloutBuffer(CONTRACT)
    t = _trans("ep1", 0, seed=1)
    with pytest.raises(ValueError, match="populated together"):
        buf.add(replace(t, source_manifest_hash=SRC_HASH))
    with pytest.raises(ValueError, match="populated together"):
        buf.add(replace(t, runtime_episode_hash=RUNTIME_HASH))


def test_buffer_rejects_empty_dual_hash():
    """Empty strings are not valid hashes (fail-closed)."""
    buf = RolloutBuffer(CONTRACT)
    t = _trans("ep1", 0, seed=1)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        buf.add(replace(t, source_manifest_hash="", runtime_episode_hash=RUNTIME_HASH))
    with pytest.raises(ValueError, match="64 lowercase hex"):
        buf.add(replace(t, source_manifest_hash=SRC_HASH, runtime_episode_hash=""))


def test_buffer_rejects_malformed_dual_hash():
    """Non-hex 64-char strings are rejected."""
    buf = RolloutBuffer(CONTRACT)
    t = _trans("ep1", 0, seed=1)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        buf.add(replace(t, source_manifest_hash="z" * 64, runtime_episode_hash=RUNTIME_HASH))


def test_buffer_rejects_source_hash_mismatch():
    """source_manifest_hash must equal the transition's recorded
    manifest_entry_hash."""
    buf = RolloutBuffer(CONTRACT)
    t = _trans("ep1", 0, seed=1)
    with pytest.raises(ValueError, match="source_manifest_hash"):
        buf.add(replace(
            t,
            source_manifest_hash=SRC_HASH,  # != manifest_entry_hash "mh"
            runtime_episode_hash=RUNTIME_HASH,
        ))


def test_buffer_accepts_matching_dual_hashes():
    """The honest-path pair (source == manifest_entry_hash,
    runtime == episode_config_hash) is accepted and traced."""
    buf = RolloutBuffer(CONTRACT)
    t = _trans("ep1", 0, seed=1, manifest_hash="c" * 64, episode_hash="e" * 64)
    ok = replace(
        t,
        source_manifest_hash="c" * 64,
        runtime_episode_hash="e" * 64,
    )
    buf.add(ok)
    assert buf.source_manifest_hashes == ["c" * 64]
    assert buf.runtime_episode_hashes == ["e" * 64]
    assert len(buf) == 1


# ── negative 7: eval-only world entering the train manifest ─────────

def test_eval_only_world_excluded_from_train_manifest(tmp_path):
    """An eval-only (rl_test) entry never reaches the train manifest or
    the training pool — the admission path rejects it structurally."""
    _sparse_worlds(tmp_path)
    _write_world(tmp_path, "routeA_rl_v1__anticipatory__rl_test__e__000004", [
        {"name": "c5", "shape": "cylinder", "x": 0.0, "y": 0.0,
         "radius": 1.0, "height": 3.0},
    ])
    entries = _entries() + [dict(
        _entries()[0], world_uid="routeA_rl_v1__anticipatory__rl_test__e__000004",
        split="rl_test", role="anticipatory",
    )]
    manifest_path = tmp_path / "admission_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "routeA_rl_env_v1", "calibration_date": "2026-07-14",
        "mode": "real_kinematic", "stats": {},
        "results": [{k: v for k, v in e.items()} for e in entries],
    }))
    train_entries = manifest_train_entries(str(manifest_path))
    assert all(e["split"] == "rl_train" for e in train_entries)
    assert not any("rl_test" in e["world_uid"] for e in train_entries)
    pool = build_routea_train_pool(str(manifest_path), str(tmp_path))
    assert len(pool) == 4
    assert not any("rl_test" in uid for uid, _ in pool)


def test_audit_fails_on_eval_only_spec_leak(tmp_path):
    """An eval-only spec can never enter training: the shared stateful
    sampler hard-refuses it at construction — the fail-closed entry point
    of the audit/train path (the eval_leakage counter is defense-in-depth
    no public path can reach; the audit itself builds this sampler)."""
    adapter, specs, _ = _setup(tmp_path)
    # R7-8 ruling (2026-08-24): the literal eval-only split is rl_val.
    eval_specs = [replace(s, split="rl_val") for s in specs]
    with pytest.raises(ValueError, match="must never enter training"):
        EpisodeTaskSampler(eval_specs, DEFAULT_ROLE_TARGETS, base_seed=0)


# ── gate-audit runtime-drift counter ────────────────────────────────

def test_audit_fails_on_runtime_hash_drift(tmp_path):
    """A runtime hash that diverges from the record's episode hash is
    counted as runtime_hash_mismatch on EVERY reset and fails the gate."""
    adapter, specs, _ = _setup(tmp_path)
    adapter._runtime_episode_hash_of = lambda env_id, spec, seed: RUNTIME_HASH
    metrics = run_reset_gate_audit(adapter, specs, n_resets=60, seed=0)
    assert metrics["runtime_hash_mismatch"] == 60
    assert metrics["gates"]["runtime_hash_mismatch"] == "FAIL"
    assert metrics["invalid_window"] == 0  # only the hash drifted
