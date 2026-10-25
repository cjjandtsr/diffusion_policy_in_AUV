"""G3 (design §5.5): manifest-driven train pool + deterministic metadata hash.

The production pool semantics are the frozen E040 build_envs (train script
L690-734): admission manifest rl_train buckets -> worlds/train/<uid>.world
.obstacles.json (routeA_rl_v1 naming). The frozen v2 scan_world_pool glob
(env_*) does NOT match the production pool, so integration.py needs a
manifest-driven builder. manifest_entry_hash has no recorded source anywhere
(manifest/sidecar), so the audit derives it deterministically from the
entry's fixed recorded config (ASSUMPTION-6 hash-proxy; R2.3 split: the
runtime SAMPLED start/goal/yaw are covered by episode_config_hash instead).
"""

import json

import pytest

from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_routea_train_pool,
    entry_metadata_hash,
)


def _write_manifest(tmp_path, entries: list) -> str:
    path = tmp_path / "admission_manifest.json"
    path.write_text(json.dumps({
        "schema_version": "routeA_rl_env_v1",
        "calibration_date": "2026-07-14",
        "mode": "real_kinematic",
        "stats": {},
        "results": entries,
    }))
    return str(path)


def _entry(uid: str, role: str = "anticipatory", split: str = "rl_train",
           admitted: bool = True) -> dict:
    return {
        "world_uid": uid, "split": split, "role_generated": role,
        "role_validated": role,
        "role_active": role if admitted else None,
        "admission_status": f"admitted_{role}" if admitted else "rejected",
        "bc_fm_sr": 0.5, "bc_fm_sr_ci_low": 0.3, "bc_fm_sr_ci_high": 0.7,
        "bc_fm_cr": 0.1, "ego_sr": 0.8, "ego_available": True,
        "beyond_boundary": False, "start_world": [-8.0, 0.0, 2.0],
        "goal_world": [8.0, 0.0, 2.0], "start_yaw": 0.0,
    }


def _write_obstacles(tmp_path, uid: str, root: str = "train") -> None:
    """Real obstacles json schema at the E040 path worlds_dir/<uid>.world.obstacles.json."""
    d = tmp_path / root
    d.mkdir(exist_ok=True)
    (d / f"{uid}.world.obstacles.json").write_text(json.dumps(
        {"world_id": uid, "obstacles": []}))


def test_build_routea_train_pool_e040_semantics(tmp_path):
    """build_routea_train_pool: rl_train admitted only, file existence
    checked, duplicates de-duplicated, non-train/rejected excluded."""
    uids = [
        ("routeA_rl_v1__anticipatory__rl_train__f__000000", "anticipatory", "rl_train", True),
        ("routeA_rl_v1__dense_recovery__rl_train__b__000001", "dense_recovery", "rl_train", True),
        ("routeA_rl_v1__anticipatory__rl_val__f__000002", "anticipatory", "rl_val", True),
        ("routeA_rl_v1__anticipatory__rl_train__f__000003", "anticipatory", "rl_train", False),
        ("routeA_rl_v1__anticipatory__rl_train__f__000004", "anticipatory", "rl_train", True),
    ]
    for uid, role, split, admitted in uids:
        _write_obstacles(tmp_path, uid)
    # duplicate uid -> dedup
    entries = [_entry(u, r, s, a) for u, r, s, a in uids]
    entries.append(_entry(uids[0][0]))  # dup of first
    manifest = _write_manifest(tmp_path, entries)

    pool = build_routea_train_pool(manifest, str(tmp_path / "train"))

    uids_in_pool = [u for u, _ in pool]
    assert "routeA_rl_v1__anticipatory__rl_train__f__000000" in uids_in_pool
    assert "routeA_rl_v1__dense_recovery__rl_train__b__000001" in uids_in_pool
    assert uids_in_pool.count("routeA_rl_v1__anticipatory__rl_train__f__000000") == 1
    # rl_val excluded (split), rejected excluded
    assert "routeA_rl_v1__anticipatory__rl_val__f__000002" not in uids_in_pool
    assert "routeA_rl_v1__anticipatory__rl_train__f__000003" not in uids_in_pool
    # json paths point at existing files
    for uid, path in pool:
        assert uid.endswith(".world.obstacles.json") or ".world.obstacles.json" in path
        assert __import__("os").path.isfile(path)


def test_build_routea_train_pool_skips_missing_files(tmp_path):
    """Entries whose obstacles file does not exist are skipped, not failed."""
    _write_obstacles(tmp_path, "routeA_rl_v1__anticipatory__rl_train__f__000000")
    entries = [
        _entry("routeA_rl_v1__anticipatory__rl_train__f__000000"),
        _entry("routeA_rl_v1__anticipatory__rl_train__f__999999"),  # no file
    ]
    manifest = _write_manifest(tmp_path, entries)
    pool = build_routea_train_pool(manifest, str(tmp_path / "train"))
    assert [u for u, _ in pool] == ["routeA_rl_v1__anticipatory__rl_train__f__000000"]


def test_entry_metadata_hash_deterministic_and_sensitive():
    """R2.3: manifest_entry_hash is the MANIFEST SOURCE hash — deterministic
    over the entry's fixed recorded fields; changing ANY fixed field
    (world_uid, role, split, recorded start/goal, yaw) changes it. It no
    longer claims to cover the runtime SAMPLED start/goal (that is
    episode_config_hash, computed at reset)."""
    a = _entry("routeA_rl_v1__anticipatory__rl_train__f__000000")
    b = _entry("routeA_rl_v1__anticipatory__rl_train__f__000000")
    assert entry_metadata_hash(a) == entry_metadata_hash(b)
    for key, value in (
        ("world_uid", "routeA_rl_v1__dense_recovery__rl_train__b__000001"),
        ("role_generated", "dense_recovery"),
        ("split", "rl_val"),
        ("start_world", [-6.0, 2.0, 2.0]),
        ("goal_world", [7.0, 1.0, 2.0]),
        ("start_yaw", 1.1),
    ):
        moved = dict(a, **{key: value})
        assert entry_metadata_hash(a) != entry_metadata_hash(moved), key


def test_entry_metadata_hash_sensitive_to_all_spec_inputs():
    """R2 rework P1-6: the source hash must cover EVERY field the spec
    construction path consumes — including role_active / the resolved
    role / admission_id / generator_version / obstacle_json, whose
    changes left the hash unchanged before (审核裁决 P1-6).

    RED contract (before the rework): only world_uid / role_generated /
    role_validated / split / start_world / goal_world / start_yaw were
    hashed, so a manifest change to any of the five missing fields was
    invisible to rule-6 and to the G3 hash gates.
    """
    a = _entry("routeA_rl_v1__anticipatory__rl_train__f__000000")
    a["role"] = a["role_active"]  # resolved-role field as in manifest_train_entries
    for key, value in (
        ("role_active", "support_expansion"),
        ("role", "support_expansion"),
        ("admission_id", "custom__admission__id"),
        ("generator_version", "routeA_rl_v1.4"),
        ("obstacle_json", "custom.obstacles.json"),
        ("world_uid", "routeA_rl_v1__dense_recovery__rl_train__b__000001"),
        ("role_generated", "dense_recovery"),
        ("role_validated", "dense_recovery"),
        ("split", "rl_val"),
        ("start_world", [-6.0, 2.0, 2.0]),
        ("goal_world", [7.0, 1.0, 2.0]),
        ("start_yaw", 1.1),
    ):
        moved = dict(a, **{key: value})
        assert entry_metadata_hash(a) != entry_metadata_hash(moved), key


def test_entry_metadata_hash_rejects_non_serializable_fields():
    """R2.3: the manifest source hash uses canonical serialization — a
    non-JSON field (e.g. numpy array) hard-fails instead of default=str
    silently swallowing it."""
    import numpy as np

    a = _entry("routeA_rl_v1__anticipatory__rl_train__f__000000")
    bad = dict(a, start_world=np.zeros(3))
    with pytest.raises(TypeError):
        entry_metadata_hash(bad)


# ═══════════════════════════════════════════════════════════════
# The source hash must cover EVERY fixed field of a production admitted
# entry — no whitelist blind spots
# ═══════════════════════════════════════════════════════════════

_PRODUCTION_FIELD_CHANGES = {
    "world_uid": "routeA_rl_v1__dense_recovery__rl_train__b__000001",
    "split": "rl_val",
    "role_generated": "dense_recovery",
    "role_validated": "dense_recovery",
    "role_active": "support_expansion",
    "admission_status": "admitted_dense_recovery",
    "bc_fm_sr": 0.9,
    "bc_fm_sr_ci_low": 0.8,
    "bc_fm_sr_ci_high": 0.95,
    "bc_fm_cr": 0.05,
    "ego_sr": 0.1,
    "ego_available": False,
    "beyond_boundary": True,
    "n_episodes_pass1": 21,
    "n_episodes_final": 51,
    "needs_pass2": False,
    "pass2_trigger_reason": "CI half-width 0.2 > 0.15",
    "stop_reason": "Rejected at pass 2",
    "avg_episode_time_s": 9.5,
    "calibration_time_s": 111.1,
    "start_world": [-6.0, 2.0, 2.0],
    "goal_world": [7.0, 1.0, 2.0],
    "start_yaw": 1.1,
}


def _full_production_entry() -> dict:
    """A production-shaped admitted rl_train entry: the 20 raw manifest
    fields + the resolved ``role`` field manifest_train_entries injects."""
    e = _entry("routeA_rl_v1__anticipatory__rl_train__f__000000")
    e.update({
        "n_episodes_pass1": 20,
        "n_episodes_final": 50,
        "needs_pass2": True,
        "pass2_trigger_reason": "CI half-width 0.1090 > 0.15",
        "stop_reason": "Admitted as anticipatory",
        "avg_episode_time_s": 4.646,
        "calibration_time_s": 232.5,
    })
    e["role"] = e["role_active"]  # resolved role, as manifest_train_entries
    return e


def test_entry_hash_sensitive_to_every_production_field():
    """Rework #3 RED: changing ANY fixed field of a production admitted
    entry changes the hash — the pre-rework whitelist silently ignored 15
    of them (admission_status, bc_fm_*, ego_*, n_episodes_*,
    pass2/stop/calibration fields, ...)."""
    a = _full_production_entry()
    for key, value in _PRODUCTION_FIELD_CHANGES.items():
        moved = dict(a, **{key: value})
        assert entry_metadata_hash(a) != entry_metadata_hash(moved), (
            f"hash blind to production field: {key}"
        )


def test_entry_hash_sensitive_to_new_fields():
    """Rework #3 RED: a NEW fixed field never seen by any whitelist must
    change the hash — a future manifest column must bind immediately."""
    a = _full_production_entry()
    assert entry_metadata_hash(a) != entry_metadata_hash(
        dict(a, future_calibration_budget_s=500.0)
    )


def test_spec_obstacle_path_matches_loaded_path(tmp_path):
    """Rework #3 RED: the task spec's obstacle path must equal the path the
    environment actually loads (<uid>.world.obstacles.json — the frozen
    E040 build_envs / build_routea_train_pool path). The registry default
    f"{uid}.json" recorded a file that is never loaded, so the hash bound
    a phantom input."""
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        TaskSpecRegistry,
    )
    from diffusion_policy_3d.rl.dppo_uav.integration import (
        build_routea_train_pool,
    )

    uid = "routeA_rl_v1__anticipatory__rl_train__f__000000"
    _write_obstacles(tmp_path, uid)
    entry = _full_production_entry()
    specs = TaskSpecRegistry([dict(entry, manifest_entry_hash="h")]).train_specs()
    pool = build_routea_train_pool(
        _write_manifest(tmp_path, [entry]), str(tmp_path / "train")
    )
    loaded = dict(pool)[uid]
    assert specs[0].obstacle_json == f"{uid}.world.obstacles.json"
    assert str(loaded).endswith(specs[0].obstacle_json), (
        f"spec obstacle {specs[0].obstacle_json!r} does not match the loaded"
        f" path {loaded!r}"
    )
