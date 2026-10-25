"""R7-8 Step 7: the frozen r7-paired-bank-v1 artifact — serialization,
integrity hashes and the fail-closed rejection matrix.

Every rejection the guide names must hard-fail:
  - any capability exposure == 0 or < 20      (admission_parsers gate)
  - missing traces                            (validity, covered by the
                                               parse contract / eval)
  - train leakage                             (bank split must be
                                               eval-only + builder check)
  - duplicate episode_config_id               (entry identity)
  - hash tampering                            (membership/content/bank_id)
  - RL-conditioned selection metadata         (opportunity provenance,
                                               capability_metrics)
"""

import pytest

from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    CapabilityOpportunity,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    TRACE_STEP_REQUIRED_KEYS,
    PairedBankEntry,
    _entries_content,
    _entries_from_dicts,
    r7_bank_to_dict,
    r7_bank_validate,
)


def _entry(i: int, *, split: str = "rl_val", role: str = "anticipatory",
           opportunity: object = None) -> PairedBankEntry:
    world_uid = f"routeA_rl_v1__{role}__{split}__fam__{i:06d}"
    return PairedBankEntry(
        world_uid=world_uid,
        role=role,
        split=split,
        admission_id=f"{world_uid}__{role}",
        manifest_entry_hash="ab" * 32,
        obstacle_json=f"{world_uid}.world.obstacles.json",
        generator_version="v1",
        seed=1 + i,
        start=[-8.0, 0.0, 2.0],
        goal=[8.0, 0.0, 2.0],
        start_yaw=0.0,
        episode_config_hash=f"ep_cfg_{i:06d}",
        disturbance={"events": []},
        admission={"start_clearance": 1.0, "goal_clearance": 1.5,
                   "reachability_pass": True, "path_length": 16.0,
                   "path_stretch": 1.0},
        opportunity=opportunity,
    )


def _opportunity(risk: bool = True, traps: int = 0,
                 stratum: str = "", topo: str = "") -> CapabilityOpportunity:
    """topo: "trap" | "cluttered" — dense opportunity (user ruling
    2026-08-24) counts both; an explicit topo overrides the traps
    default so cluttered worlds carry no trap count."""
    dense = topo or ("trap" if traps else "")
    return CapabilityOpportunity(
        valid=True,
        nominal_path_risk=0.5 if risk else 0.0,
        risk_shell_crossing=risk,
        anticipatory_provenance="nominal_path_geometry",
        topology_class=dense,
        trap_opportunity_count=traps,
        dense_provenance="world_topology" if dense else "",
        support_stratum=stratum,
        reference_density=1.0 if stratum else 0.0,
        support_provenance="offline_density" if stratum else "",
    )


def _entries(n: int = 100, **kwargs) -> list:
    return [_entry(i, **kwargs) for i in range(n)]


def _aggregate(entries: list, candidates: dict) -> dict:
    from diffusion_policy_3d.rl.dppo_uav.r7_bank_builder import (
        compute_aggregate,
    )
    return compute_aggregate(entries, candidates)


def _bank_dict(entries: list, **overrides) -> dict:
    bank = r7_bank_to_dict(
        entries,
        source_manifest_hash="cd" * 32,
        bc_checkpoint_hash="ef" * 32,
        builder_source_hash="01" * 32,
        creation_commit="62076cb",
        aggregate=_aggregate(
            entries,
            {"anticipatory": 24, "full_space": 24, "dense_recovery": 54},
        ),
    )
    bank.update(overrides)
    return bank


# ═══════════════════════════════════════════════════════════════════
# positive path — a well-formed 100-entry bank validates clean
# ═══════════════════════════════════════════════════════════════════

def test_bank_entries_carry_required_trace_fields():
    """Guide R7-8 Step 1: every frozen episode record must declare
    required_trace_fields — the trace contract the R7-9 BC replay (v2
    rebuild) must produce. The serialized content carries the field and
    the object roundtrip is symmetric (membership hash binding)."""
    entries = _entries(3)
    for e, d in zip(entries, _entries_content(entries)):
        assert d["required_trace_fields"] == list(TRACE_STEP_REQUIRED_KEYS), e.world_uid
    roundtrip = _entries_from_dicts(_entries_content(entries))
    for e, r in zip(entries, roundtrip):
        assert r.required_trace_fields == list(TRACE_STEP_REQUIRED_KEYS)
        assert _entries_content([e]) == _entries_content([r])
    # the frozen bank JSON carries the field per entry
    bank = _bank_dict(entries)
    for entry in bank["entries"]:
        assert entry["required_trace_fields"] == list(TRACE_STEP_REQUIRED_KEYS)


def test_r7_bank_roundtrip_valid(tmp_path):
    entries = _entries(100, role="dense_recovery")
    bank = _bank_dict(entries)
    r7_bank_validate(bank)  # must not raise
    assert bank["schema_version"] == "r7-paired-bank-v1"
    assert bank["bank_size"] == 100
    assert len(bank["membership_hash"]) == 64
    assert len(bank["content_hash"]) == 64
    assert bank["aggregate"]["exposure"]["anticipatory"] == (
        "NOT_YET_MEASURED")


def test_r7_bank_validate_takes_train_world_uids(tmp_path):
    """The builder passes the train manifest world set — any overlap is
    a hard fail (Step 3)."""
    entries = _entries(100, role="dense_recovery")
    bank = _bank_dict(entries)
    with pytest.raises(ValueError):
        r7_bank_validate(
            bank, train_world_uids=[entries[7].world_uid]
        )


# ═══════════════════════════════════════════════════════════════════
# negative path — every Step-7 rejection axis
# ═══════════════════════════════════════════════════════════════════

def test_membership_hash_tamper_rejected(tmp_path):
    entries = _entries(100, role="dense_recovery")
    bank = _bank_dict(entries)
    bank["entries"][3]["seed"] += 1
    with pytest.raises(ValueError):
        r7_bank_validate(bank)


def test_content_hash_tamper_rejected(tmp_path):
    entries = _entries(100, role="dense_recovery")
    bank = _bank_dict(entries)
    bank["aggregate"]["worlds"]["rl_val"] = 999
    with pytest.raises(ValueError):
        r7_bank_validate(bank)


def test_bank_id_tamper_rejected(tmp_path):
    entries = _entries(100, role="dense_recovery")
    bank = _bank_dict(entries)
    bank["bank_id"] = "00" * 32
    with pytest.raises(ValueError):
        r7_bank_validate(bank)


def test_duplicate_episode_config_hash_rejected(tmp_path):
    entries = _entries(100, role="dense_recovery")
    entries[50] = entries[49]  # same episode_config_hash
    bank = _bank_dict(entries)
    with pytest.raises(ValueError):
        r7_bank_validate(bank)


def test_train_split_entry_rejected(tmp_path):
    """A bank carrying an rl_train split is a leakage symptom — the
    frozen artifact itself must be eval-only."""
    entries = _entries(99, role="dense_recovery")
    entries.append(_entry(999, role="dense_recovery", split="rl_train"))
    bank = _bank_dict(entries)
    with pytest.raises(ValueError):
        r7_bank_validate(bank)


def test_bank_size_mismatch_rejected(tmp_path):
    entries = _entries(100, role="dense_recovery")
    bank = _bank_dict(entries)
    bank["bank_size"] = 99
    with pytest.raises(ValueError):
        r7_bank_validate(bank)


def test_wrong_schema_version_rejected(tmp_path):
    entries = _entries(100, role="dense_recovery")
    bank = _bank_dict(entries)
    bank["schema_version"] = "r7-paired-bank-v0"
    with pytest.raises(ValueError):
        r7_bank_validate(bank)


def test_aggregate_opportunity_mismatch_rejected(tmp_path):
    """The stored aggregate.opportunity must equal a recomputation from
    the entries' frozen opportunity blocks — hand-edited aggregate is a
    tamper symptom."""
    entries = _entries(100, role="dense_recovery")
    bank = _bank_dict(entries)
    bank["aggregate"]["opportunity"]["dense_recovery"] = 100
    with pytest.raises(ValueError):
        r7_bank_validate(bank)


def test_aggregate_opportunity_counts_cluttered_worlds(tmp_path):
    """Dense opportunity (user ruling 2026-08-24) counts cluttered
    topology — 100 cluttered dense entries declare 100 dense
    opportunities, matching what the builder's own aggregate computes.
    A trap-only counter would undercount to 0 and fail the recompute
    equality the validator enforces."""
    entries = _entries(
        100, role="dense_recovery", opportunity=_opportunity(topo="cluttered"))
    bank = _bank_dict(entries)
    assert bank["aggregate"]["opportunity"]["dense_recovery"] == 100
    r7_bank_validate(bank)  # recompute must agree (trap∪cluttered)


# ═══════════════════════════════════════════════════════════════════
# exposure gate — the R7-10 parse_bank_result contract (Step 7 first
# two rejections: exposure == 0, exposure < 20)
# ═══════════════════════════════════════════════════════════════════

def test_parse_exposure_zero_rejected(tmp_path):
    """parse_bank_result rejects exposure == 0 per capability."""
    from diffusion_policy_3d.rl.dppo_uav.admission_parsers import (
        parse_bank_result,
    )

    entries = _entries(100, role="dense_recovery")
    bank = _bank_dict(entries)
    bank["aggregate"]["exposure"] = {
        "anticipatory": 0, "full_space": 0, "dense_recovery": 0,
    }
    path = tmp_path / "bank.json"
    path.write_text(_json_dump(bank))
    with pytest.raises(Exception):
        parse_bank_result(path)


def test_parse_exposure_below_minimum_rejected(tmp_path):
    """parse_bank_result rejects exposure < 20 per capability."""
    from diffusion_policy_3d.rl.dppo_uav.admission_parsers import (
        parse_bank_result,
    )

    entries = _entries(100, role="dense_recovery")
    bank = _bank_dict(entries)
    bank["aggregate"]["exposure"] = {
        "anticipatory": 19, "full_space": 25, "dense_recovery": 30,
    }
    path = tmp_path / "bank.json"
    path.write_text(_json_dump(bank))
    with pytest.raises(Exception):
        parse_bank_result(path)


def _json_dump(bank: dict) -> str:
    import json
    return json.dumps(bank)
