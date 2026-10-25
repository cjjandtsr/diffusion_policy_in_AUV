"""R7-8 Step 3 groundwork: the canonical eval pool must reach the
rl_val ∪ rl_test splits.

User ruling (2026-08-24): "rl_eval" is the normative name for
rl_val ∪ rl_test — the admission manifest carries no literal rl_eval
split, so build_routea_eval_pool / manifest_eval_entries must resolve
the three literal splits, with world files looked up per split
(rl_train -> worlds/train, rl_val -> worlds/val, rl_test ->
worlds/test). Without this, a bank built from eval-only worlds is
rejected by the canonical evaluator's pool check (exit 2).

Also covers the offline-density schema bridge: the EGO-dataset builder
(tools/build_routea_offline_density.py) emits reward_v13 cell_counts;
the capability layer (OfflineDensity) requires the counts schema — the
conversion must be exact and validated (user ruling: real density is
built inside R7-8, never the synthetic placeholder).
"""

import json

import pytest

from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_routea_eval_pool,
    manifest_eval_entries,
)
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import OfflineDensity


# ═══════════════════════════════════════════════════════════════════
# fixtures — a mini admission manifest + per-split world dirs
# ═══════════════════════════════════════════════════════════════════

def _mini_manifest(tmp_path, entries):
    """Mini manifest with one entry per (role, split) pair."""
    results = []
    for i, (role, split) in enumerate(entries):
        results.append({
            "world_uid": f"routeA_rl_v1__{role}__{split}__fam__{i:06d}",
            "split": split,
            "role_generated": role,
            "role_validated": role,
            "role_active": role,
            "admission_status": f"admitted_{role}",
            "bc_fm_sr": 0.5, "bc_fm_cr": 0.7,
            "bc_fm_sr_ci_low": 0.4, "bc_fm_sr_ci_high": 0.6,
            "n_episodes_pass1": 20, "n_episodes_final": 50,
        })
    data = {
        "schema_version": "routeA_rl_env_v1",
        "calibration_date": "r7-8-test",
        "mode": "test",
        "stats": {"total": len(results)},
        "results": results,
    }
    path = tmp_path / "admission_manifest.json"
    path.write_text(json.dumps(data))
    return path


def _world_roots(tmp_path):
    """worlds/{train,val,test} dirs, each seeded with the obstacles file
    of its matching manifest entry (empty obstacle list is fine — the
    pool check is file existence, not content)."""
    entries = [
        ("anticipatory", "rl_train"),
        ("dense_recovery", "rl_val"),
        ("support_expansion", "rl_test"),
    ]
    manifest = _mini_manifest(tmp_path, entries)
    roots = {}
    for split, sub in (("rl_train", "train"),
                       ("rl_val", "val"),
                       ("rl_test", "test")):
        d = tmp_path / f"worlds_{sub}"
        d.mkdir()
        roots[split] = d
    uids = {}
    for i, (role, split) in enumerate(entries):
        uid = f"routeA_rl_v1__{role}__{split}__fam__{i:06d}"
        uids[split] = uid
        (roots[split] / f"{uid}.world.obstacles.json").write_text(
            json.dumps({"world_id": uid, "n_obstacles": 0, "obstacles": []})
        )
    return manifest, roots, uids


# ═══════════════════════════════════════════════════════════════════
# eval pool semantics — the three literal splits, per-split world dirs
# ═══════════════════════════════════════════════════════════════════

def test_eval_pool_reaches_val_and_test(tmp_path):
    """The canonical eval pool covers rl_train ∪ rl_val ∪ rl_test —
    eval-only worlds are reachable by the canonical eval (P1-1)."""
    manifest, roots, uids = _world_roots(tmp_path)
    pool = build_routea_eval_pool(
        str(manifest),
        str(roots["rl_train"]),
        str(roots["rl_val"]),
        str(roots["rl_test"]),
    )
    pool_uids = {uid for uid, _ in pool}
    assert pool_uids == set(uids.values()), (
        "eval pool must reach all three splits; got "
        f"{sorted(pool_uids)}"
    )
    assert uids["rl_val"] in pool_uids
    assert uids["rl_test"] in pool_uids


def test_eval_pool_files_resolved_per_split(tmp_path):
    """Each entry's obstacle JSON resolves inside its OWN split's world
    dir — a rl_test world must never be looked up under worlds/train."""
    manifest, roots, uids = _world_roots(tmp_path)
    pool = dict(build_routea_eval_pool(
        str(manifest),
        str(roots["rl_train"]),
        str(roots["rl_val"]),
        str(roots["rl_test"]),
    ))
    assert pool[uids["rl_train"]].endswith(
        f"worlds_train/{uids['rl_train']}.world.obstacles.json")
    assert pool[uids["rl_val"]].endswith(
        f"worlds_val/{uids['rl_val']}.world.obstacles.json")
    assert pool[uids["rl_test"]].endswith(
        f"worlds_test/{uids['rl_test']}.world.obstacles.json")


def test_eval_pool_backward_compatible_two_args(tmp_path):
    """Callers that pass only the train dir keep working: val/test
    entries without a world dir are skipped (never crash)."""
    manifest, roots, uids = _world_roots(tmp_path)
    pool = build_routea_eval_pool(str(manifest), str(roots["rl_train"]))
    pool_uids = {uid for uid, _ in pool}
    assert uids["rl_train"] in pool_uids
    assert uids["rl_val"] not in pool_uids
    assert uids["rl_test"] not in pool_uids


def test_manifest_eval_entries_include_val_test(tmp_path):
    """manifest_eval_entries resolves the three literal splits."""
    manifest, roots, uids = _world_roots(tmp_path)
    entries = manifest_eval_entries(str(manifest))
    by_uid = {e["world_uid"]: e["split"] for e in entries}
    assert by_uid[uids["rl_val"]] == "rl_val"
    assert by_uid[uids["rl_test"]] == "rl_test"
    assert len(entries) == 3


# ═══════════════════════════════════════════════════════════════════
# offline density schema bridge (reward_v13 cell_counts -> capability)
# ═══════════════════════════════════════════════════════════════════

def _reward13_cell_counts():
    return {
        "cell_counts": {"0,0,0": 20, "1,0,0": 5, "-1,2,3": 2},
        "total": 27,
        "p20_threshold": 2.0,
        "p50_threshold": 5.0,
    }


def test_reward13_density_to_capability_schema():
    """Convert reward_v13 cell_counts -> capability OfflineDensity and
    validate the round trip through OfflineDensity.from_dict."""
    from diffusion_policy_3d.rl.dppo_uav.r7_bank_builder import (
        reward13_density_to_capability,
    )

    density = reward13_density_to_capability(_reward13_cell_counts())
    assert density.counts == {(0, 0, 0): 20, (1, 0, 0): 5, (-1, 2, 3): 2}
    # the converted object must pass the strict capability loader
    reloaded = OfflineDensity.from_dict(density.to_dict())
    assert reloaded.counts == density.counts


def test_reward13_density_empty_rejected():
    """An empty cell_counts must not silently become an empty
    OfflineDensity (which would be a fake 'no low density' fact)."""
    from diffusion_policy_3d.rl.dppo_uav.r7_bank_builder import (
        reward13_density_to_capability,
    )

    with pytest.raises(ValueError):
        reward13_density_to_capability(
            {"cell_counts": {}, "total": 0,
             "p20_threshold": 0.0, "p50_threshold": 0.0}
        )


def test_reward13_density_bad_key_rejected():
    """Non x,y,z cell keys are rejected by the conversion (and would be
    rejected by OfflineDensity anyway — fail fast at the bridge)."""
    from diffusion_policy_3d.rl.dppo_uav.r7_bank_builder import (
        reward13_density_to_capability,
    )

    data = _reward13_cell_counts()
    data["cell_counts"]["not_a_cell"] = 3
    with pytest.raises(ValueError):
        reward13_density_to_capability(data)
