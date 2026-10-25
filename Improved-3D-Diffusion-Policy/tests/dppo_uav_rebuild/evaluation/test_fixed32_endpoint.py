"""Fixed32 selects the actual test panel, never rl_val or training worlds."""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pytest

from scripts.evaluate_dppo_uav_formal_endpoint import (
    _load_fixed_bank,
    _resolve_obstacle,
    _validate_bank_sources,
)
from scripts.formal_endpoint_eval_core import (
    PairedEpisodeRecord,
    PairedEvaluationBank,
    build_endpoint_report,
)


CONFIG_DIR = Path(__file__).resolve().parents[3] / "data/eval_configs"


def _copy_configs(target):
    for split in ("rl_test", "layerA"):
        filename = "episode_configs_{}.json".format(split)
        (target / filename).write_bytes((CONFIG_DIR / filename).read_bytes())


def _manifest_entries(bank):
    return [
        {"world_uid": world, "split": split, "role": fixed.payload["role"]}
        for split, world, fixed in bank if split == "rl_test"
    ]


def test_fixed32_reads_twelve_test_and_twenty_layera_worlds():
    bank = _load_fixed_bank(CONFIG_DIR)
    assert Counter(split for split, _, _ in bank) == {"rl_test": 48, "layerA": 80}
    worlds = Counter((split, world) for split, world, _ in bank)
    assert len(worlds) == 32
    assert set(worlds.values()) == {4}
    assert len({fixed.payload["episode_id"] for _, _, fixed in bank}) == 128
    assert len({fixed.config_hash for _, _, fixed in bank}) == 128


def test_fixed32_rejects_unequal_episodes_even_when_total_stays_128(tmp_path):
    _copy_configs(tmp_path)
    path = tmp_path / "episode_configs_layerA.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    first, second = list(payload)[:2]
    payload[second].append(payload[first].pop())
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="4 episodes per world"):
        _load_fixed_bank(tmp_path)


def test_fixed32_rejects_duplicate_episode_ids(tmp_path):
    _copy_configs(tmp_path)
    path = tmp_path / "episode_configs_layerA.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    world = next(iter(payload))
    payload[world][1]["episode_id"] = payload[world][0]["episode_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate episode_id"):
        _load_fixed_bank(tmp_path)


def test_layera_needs_no_routea_admission_but_cannot_overlap_training():
    bank = _load_fixed_bank(CONFIG_DIR)
    entries = _manifest_entries(bank)
    _validate_bank_sources(bank, entries)
    layera_world = next(world for split, world, _ in bank if split == "layerA")
    entries.append({"world_uid": layera_world, "split": "rl_train"})
    with pytest.raises(ValueError, match="training worlds"):
        _validate_bank_sources(bank, entries)


def test_test_bank_must_match_its_admitted_worlds_not_validation_worlds():
    bank = _load_fixed_bank(CONFIG_DIR)
    entries = _manifest_entries(bank)
    first_world = entries[0]["world_uid"]
    entries = [row for row in entries if row["world_uid"] != first_world]
    entries.append({"world_uid": first_world, "split": "rl_val", "role": "anticipatory"})
    with pytest.raises(ValueError, match="rl_test"):
        _validate_bank_sources(bank, entries)


def test_layera_plain_obstacle_suffix_and_routea_world_suffix(tmp_path):
    (tmp_path / "env_000_obs2.obstacles.json").write_text("[]", encoding="utf-8")
    routea = "routeA_rl_v1__anticipatory__rl_test__fixture__000001"
    (tmp_path / (routea + ".world.obstacles.json")).write_text("[]", encoding="utf-8")
    assert _resolve_obstacle(tmp_path, "layerA", "env_000_obs2").name == "env_000_obs2.obstacles.json"
    assert _resolve_obstacle(tmp_path, "rl_test", routea).name == routea + ".world.obstacles.json"


def test_ambiguous_layera_obstacles_are_not_silently_selected(tmp_path):
    (tmp_path / "env_000_obs2.obstacles.json").write_text("[]", encoding="utf-8")
    (tmp_path / "env_000_obs2.world.obstacles.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_obstacle(tmp_path, "layerA", "env_000_obs2")


def _record(world, split, index, policy, done):
    return PairedEpisodeRecord(
        task_id=world + str(index), config_id=world + str(index),
        config_hash="a" * 64, obstacle_hash="b" * 64,
        source_manifest_hash="c" * 64,
        policy_identity_hash=("d" if policy == "bc" else "e") * 64,
        world_id=world, split=split, role="anticipatory", seed=index,
        policy=policy, done_reason=done,
        raw_done_reason="reached" if done == "reached" else "none",
        episode_return=0.0, onset_triggered=False, onset_first_step=None,
        future_clearance_reward_sum=0.0, d_soft_h_min=1.0, d_soft_h_steps=1,
        support_cells=0, terminal_xyz=(0.0, 0.0, 1.0),
        min_xyz=(0.0, 0.0, 1.0), max_xyz=(0.0, 0.0, 1.0),
        action_dz_mean=0.0, action_dz_min=0.0, action_dz_max=0.0,
        trap_recovery_count=0, loop_count=0, side_flip_count_max=0, steps=2,
    )


def test_layera_generic_role_is_not_pooled_into_routea_capability_claims():
    records = []
    for index in range(2):
        for split, world in (("rl_test", "routea_fixture"), ("layerA", "env_000_obs2")):
            records.append(_record(world, split, index, "bc", "time_limit"))
            records.append(_record(world, split, index, "rl", "reached" if split == "layerA" else "time_limit"))
    report = build_endpoint_report(PairedEvaluationBank(records))
    assert report["world_count"] == 2
    assert report["pair_count"] == 4
    assert report["metrics"]["success"]["mean"] == 0.5
    assert report["metrics"]["anticipatory_success"]["mean"] == 0.0
    assert report["role_aggregate"]["anticipatory"]["rl"]["episodes"] == 2
    assert report["role_aggregate_by_split"]["layerA"]["anticipatory"]["rl"]["episodes"] == 2
    assert report["judgment_is_execution_gate"] is False


def test_layera_records_remain_paired_on_the_same_source_identity():
    bc = _record("env_000_obs2", "layerA", 0, "bc", "time_limit")
    rl = _record("env_000_obs2", "layerA", 0, "rl", "time_limit")
    with pytest.raises(ValueError, match="environment identity mismatch"):
        PairedEvaluationBank([bc, replace(rl, source_manifest_hash="f" * 64)])
