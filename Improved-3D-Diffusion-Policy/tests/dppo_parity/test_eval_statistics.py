"""R6-D (R6-EVAL-1): paired-eval statistics layer.

Guide §11.5.1/§11.5.2/§11.5.5 + §7.1 — the statistics layer is PURE
aggregation, it never rolls out:

- PairUnitKey: the four-field identity (bank_id, episode_config_hash,
  replica_id, policy_checkpoint_hash) derivable without ambiguity from
  artifact provenance + record; missing hash / empty provenance / RNG
  provenance mismatch are hard-fails, never silent defaults;
- BC deduplication: identical BC records across RL artifacts count ONCE
  (the 18/24 counterexample — raw=24, unique/effective=12, SR=9/12);
  different replicas are kept; different RL checkpoints form policy
  clusters, NOT independent policies (3 seeds != 180);
- binary-outcome aggregation: pooled SR with a Wilson CI fixed oracle,
  entry-level cluster SR (unweighted per-episode_config_hash mean), and
  the report ALWAYS carrying raw n / unique n / cluster n;
- paired_policy_delta: per-episode paired deltas over common episode
  keys; provenance mismatch or unpaired episodes hard-fail; identical
  policy + common keys => every delta exactly 0;
- validate_paired_matrix: a 12x5 matrix missing one replica is
  incomplete; duplicate episode keys hard-fail.
"""

import dataclasses
import math

import pytest

from diffusion_policy_3d.rl.dppo_uav.evaluator import EpisodePairedRecord
from diffusion_policy_3d.rl.dppo_uav.paired_rng import (
    RNG_CONTRACT_VERSION,
    policy_noise_seed,
)

from diffusion_policy_3d.rl.dppo_uav.eval_statistics import (
    PairUnitKey,
    aggregate_binary_outcomes,
    deduplicate_policy_records,
    pair_unit_key,
    pair_unit_key_from_artifact,
    paired_eval_report,
    paired_policy_delta,
    validate_paired_matrix,
)

BANK = "bank-abc"
BC_CKPT = "ckpt_bc" + "0" * 20


def _ep_hash(i: int) -> str:
    return f"ep_{i:04d}"


def _outcome(success: bool, ret: float = 10.0, steps: int = 40) -> dict:
    return {
        "success": bool(success),
        "collision": not bool(success),
        "timeout": False,
        "done_reason": "reached" if success else "collision",
        "out_of_bounds": False,
        "altitude_exceed": False,
        # R6-E §11.6.4 (F-10): raw Gymnasium flags + the ONE mutually
        # exclusive terminal category — the fixture maps the two clean
        # cases (success -> reached, collision -> collision).
        "terminated": True,
        "truncated": False,
        "terminal_category": "reached" if success else "collision",
        "return": float(ret),
        "steps": int(steps),
    }


def _record(bank_id: str, ckpt_hash: str, ep_hash: str, replica: int,
            success: bool, *, ret: float = 10.0, rng_seed: int = None,
            rng_version: str = RNG_CONTRACT_VERSION,
            config_world: str = "w1") -> EpisodePairedRecord:
    seed = (rng_seed if rng_seed is not None
            else policy_noise_seed(bank_id, ep_hash, replica))
    outcome = _outcome(success, ret)
    return EpisodePairedRecord(
        episode_config={"world": config_world, "seed": 42,
                        "episode_config_hash": ep_hash},
        seed=42, world=config_world, role="anticipatory",
        start=[0.0, 0.0, 0.0], goal=[1.0, 2.0, 3.0],
        disturbance={"events": []},
        bc=dict(outcome), rl=dict(outcome), trace={},
        policy_noise_seed=seed, replica_id=int(replica),
        rng_contract_version=rng_version,
        # policy identity mirrors the artifact-level checkpoint_hash —
        # the 3-seed cluster test depends on it being on the record.
        policy_checkpoint_hash=ckpt_hash,
    )


def _bc_12(bank_id: str = BANK, ckpt: str = BC_CKPT) -> list:
    """The 12-task BC reference (9 successes — the real 9/12)."""
    return [
        _record(bank_id, ckpt, _ep_hash(i), 0, success=(i < 9), ret=10.0 - i)
        for i in range(12)
    ]


def _bc_60(bank_id: str = BANK, ckpt: str = BC_CKPT) -> list:
    """12 bank entries x 5 policy-noise replicas, all BC checkpoint."""
    return [
        _record(bank_id, ckpt, _ep_hash(i), r, success=(i < 9), ret=10.0 - i)
        for i in range(12) for r in range(5)
    ]


def _rl_60(bank_id: str, ckpt: str, success_at: list) -> list:
    """60 records for one RL policy seed; success_at = entry indices."""
    return [
        _record(bank_id, ckpt, _ep_hash(i), r, success=(i in success_at),
                ret=12.0 - i)
        for i in range(12) for r in range(5)
    ]


class _ArtifactLike:
    """Minimal artifact surface for pair_unit_key_from_artifact."""

    def __init__(self, bank_id: str, checkpoint_hash: str) -> None:
        self.bank_id = bank_id
        self.checkpoint_hash = checkpoint_hash


# ═══════════════════════════════════════════════════════════════════
# PairUnitKey: identity, derivation, hard-fails
# ═══════════════════════════════════════════════════════════════════

def test_pair_unit_key_derived_from_artifact_provenance():
    rec = _bc_12()[0]
    key = pair_unit_key_from_artifact(_ArtifactLike(BANK, BC_CKPT), rec)
    assert key == PairUnitKey(
        bank_id=BANK,
        episode_config_hash=_ep_hash(0),
        replica_id=0,
        policy_checkpoint_hash=BC_CKPT,
    )
    # Same record under a different RL checkpoint is a DIFFERENT unit key
    # (policy seed identity), same record under a different replica too.
    assert pair_unit_key(BANK, "ckpt_101", rec).policy_checkpoint_hash \
        == "ckpt_101"
    assert pair_unit_key(BANK, BC_CKPT, rec) == key


def test_pair_unit_key_missing_episode_config_hash_hard_fails():
    rec = _bc_12()[0]
    rec = dataclasses.replace(
        rec, episode_config={"world": "w1", "seed": 42})
    with pytest.raises(ValueError, match="episode_config_hash"):
        pair_unit_key(BANK, BC_CKPT, rec)


def test_pair_unit_key_empty_provenance_hard_fails():
    rec = _bc_12()[0]
    with pytest.raises(ValueError, match="bank_id"):
        pair_unit_key("", BC_CKPT, rec)
    with pytest.raises(ValueError, match="checkpoint_hash"):
        pair_unit_key(BANK, "", rec)


def test_pair_unit_key_rng_provenance_mismatch_hard_fails():
    """§11.5.5: the record's policy_noise_seed must re-derive from
    (bank_id, episode_config_hash, replica) — a tampered seed breaks the
    paired-RNG contract and is a hard fail, not a silent identity."""
    rec = _bc_12()[0]
    tampered = dataclasses.replace(
        rec, policy_noise_seed=rec.policy_noise_seed + 1)
    with pytest.raises(ValueError, match="RNG provenance"):
        pair_unit_key(BANK, BC_CKPT, tampered)


# ═══════════════════════════════════════════════════════════════════
# Deduplication: the 18/24 counterexample and policy clusters
# ═══════════════════════════════════════════════════════════════════

def test_bc_duplicate_12_counterexample_raw24_unique12_sr9of12():
    """Guide §11.5.2: two seeds' BC sides are the SAME 12 records — the
    effective baseline is 9/12 on the 12 tasks, NOT 18/24 on 24."""
    bc = _bc_12()
    raw = bc + bc  # identical BC records from two RL artifacts
    dedup = deduplicate_policy_records(raw)
    assert dedup.raw_records == 24
    assert dedup.unique_records == 12
    assert dedup.effective_n == 12
    assert dedup.policy_clusters == 1
    agg = aggregate_binary_outcomes(raw)
    assert agg["raw_n"] == 24
    assert agg["unique_n"] == 12
    assert agg["effective_n"] == 12
    assert agg["cluster_n"] == 12
    assert agg["success_count"] == 9
    assert agg["sr"] == 9 / 12


def test_replicas_are_distinct_units_60():
    bc = _bc_60()
    dedup = deduplicate_policy_records(bc)
    assert dedup.raw_records == 60
    assert dedup.unique_records == 60
    assert dedup.effective_n == 60
    assert dedup.policy_clusters == 1
    agg = aggregate_binary_outcomes(bc)
    assert agg["cluster_n"] == 12  # 12 bank entries, not 60
    assert agg["sr"] == 9 / 12


def test_three_policy_seeds_clustered_not_independent():
    """§11.5.1: 3 policy seeds x 60 are 3 clusters, never 180 independent
    policies — per-cluster SR is reported, not one pooled 180 count."""
    seeds = {"101": "ckpt_101", "202": "ckpt_202", "303": "ckpt_303"}
    raw = []
    for ckpt in seeds.values():
        raw.extend(_rl_60(BANK, ckpt, success_at=list(range(9))))
    dedup = deduplicate_policy_records(raw)
    assert dedup.raw_records == 180
    assert dedup.unique_records == 180
    assert dedup.policy_clusters == 3
    agg = aggregate_binary_outcomes(raw)
    assert agg["cluster_n"] == 12
    # Pooled-over-180 would claim n=180; the honest report is per seed.
    for ckpt in seeds.values():
        agg_c = aggregate_binary_outcomes(_rl_60(BANK, ckpt,
                                                 success_at=list(range(9))))
        assert agg_c["unique_n"] == 60
        assert agg_c["sr"] == 9 / 12
        assert agg_c["policy_clusters"] == 1


# ═══════════════════════════════════════════════════════════════════
# Aggregation: Wilson CI oracle, cluster (entry-level) SR
# ═══════════════════════════════════════════════════════════════════

def _wilson_oracle(successes: int, n: int, z: float = 1.96):
    """Independent direct-formula Wilson interval (the fixed oracle)."""
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def test_wilson_ci_fixed_oracle():
    # n=12, SR=9/12: hard-coded values from the closed-form formula
    # (center 0.68937, half 0.22168).
    agg = aggregate_binary_outcomes(_bc_12())
    lo, hi = agg["wilson_ci"]
    assert lo == pytest.approx(0.4677, abs=1e-4)
    assert hi == pytest.approx(0.9111, abs=1e-4)
    # Degenerate n=1, SR=1: upper bound is EXACTLY 1.0.
    one = aggregate_binary_outcomes([_record(BANK, BC_CKPT, _ep_hash(0), 0,
                                             True)])
    assert one["sr"] == 1.0
    assert one["wilson_ci"][1] == pytest.approx(1.0, abs=1e-12)
    assert one["wilson_ci"][0] == pytest.approx(0.206546, abs=1e-5)
    # Production matches the independent oracle across a sweep.
    for n, x in ((5, 3), (12, 9), (60, 45), (1, 1), (1, 0)):
        recs = [_record(BANK, BC_CKPT, _ep_hash(i), 0, i < x)
                for i in range(n)]
        prod = aggregate_binary_outcomes(recs)["wilson_ci"]
        oracle = _wilson_oracle(x, n)
        assert prod[0] == pytest.approx(oracle[0], abs=1e-9)
        assert prod[1] == pytest.approx(oracle[1], abs=1e-9)


def test_aggregate_zero_records_hard_fails():
    with pytest.raises(ValueError, match="no records"):
        aggregate_binary_outcomes([])


def test_entry_level_cluster_sr_unweighted_when_matrix_incomplete():
    """Pooled SR and entry-level cluster SR diverge under an incomplete
    matrix: entry A has 2/2, entry B 1/3 (one replica missing)."""
    recs = (
        [_record(BANK, BC_CKPT, _ep_hash(0), r, True) for r in range(2)]
        + [_record(BANK, BC_CKPT, _ep_hash(1), r, r == 0) for r in range(3)]
    )
    agg = aggregate_binary_outcomes(recs)
    assert agg["raw_n"] == 5
    assert agg["sr"] == 3 / 5  # pooled
    assert agg["cluster_n"] == 2
    assert agg["cluster_sr"] == pytest.approx(2 / 3)  # mean(1.0, 1/3)


# ═══════════════════════════════════════════════════════════════════
# R6-E §11.6.4 (F-10): raw events + terminal-category distribution
# ═══════════════════════════════════════════════════════════════════

def test_raw_events_and_terminal_category_distribution():
    """§11.6.4: the aggregate carries raw-event SR/CR counts (terminated/
    truncated/timeout over the UNIQUE records) and the mutually exclusive
    terminal-category distribution — reached and collision both stay
    visible, never collapsed."""
    agg = aggregate_binary_outcomes(_bc_12())
    assert agg["success_count"] == 9
    assert agg["collision_count"] == 3
    assert agg["cr"] == 3 / 12
    assert agg["raw_events"] == {"terminated": 12, "truncated": 0,
                                 "timeout": 0}
    assert agg["terminal_category_distribution"] == {
        "reached": 9, "collision": 3,
    }


def test_terminal_category_missing_keys_hard_fails():
    """A pre-R6-E artifact (record without the §11.6.4 keys) hard-fails —
    never a fabricated category or a silent 0."""
    old = _outcome(True)
    for k in ("terminated", "truncated", "terminal_category"):
        del old[k]
    rec = dataclasses.replace(
        _record(BANK, BC_CKPT, _ep_hash(0), 0, True),
        bc=dict(old), rl=dict(old),
    )
    with pytest.raises(ValueError, match="11.6.4"):
        aggregate_binary_outcomes([rec])


# ═══════════════════════════════════════════════════════════════════
# paired_policy_delta
# ═══════════════════════════════════════════════════════════════════

def test_paired_delta_identical_policy_sentinel_all_zero():
    """§11.5.5: identical policy over common pair keys => every paired
    delta exactly 0 (binary outcomes — no tolerance needed)."""
    bc = _bc_60()
    pair = paired_policy_delta(bc, bc)
    assert pair.common_n == 60
    assert pair.unpaired_bc == 0 and pair.unpaired_rl == 0
    assert all(d == 0 for d in pair.success_deltas)
    assert all(d == 0 for d in pair.return_deltas)
    assert pair.mean_success_delta == 0.0


def test_paired_delta_hand_computed():
    """RL flips entry 7 (success->fail, -1) and entry 9 (fail->success,
    +1) against the BC reference — per-key deltas hand-checked."""
    bc = _bc_60()
    rl_success = [i < 9 for i in range(12)]
    rl_success[7] = False
    rl_success[9] = True
    rl = _rl_60(BANK, "ckpt_101", success_at=[i for i in range(12)
                                              if rl_success[i]])
    pair = paired_policy_delta(bc, rl)
    assert pair.common_n == 60
    assert pair.mean_success_delta == pytest.approx(0.0)  # +1 and -1 cancel
    assert sum(1 for d in pair.success_deltas if d == 1) == 5  # 5 replicas
    assert sum(1 for d in pair.success_deltas if d == -1) == 5
    assert sum(1 for d in pair.success_deltas if d == 0) == 50
    # Per-key deltas are keyed by (bank_id, episode_config_hash, replica).
    assert pair.per_key[(_ep_hash(7), 0)] == -1
    assert pair.per_key[(_ep_hash(9), 4)] == 1
    assert pair.per_key[(_ep_hash(0), 3)] == 0


def test_paired_delta_provenance_mismatch_hard_fails():
    bc = _bc_12()
    # Same episode key, different RNG contract version.
    rl_bad_rng = [
        _record(BANK, "ckpt_101", _ep_hash(i), 0, True,
                rng_version="paired-rng-v2")
        for i in range(12)
    ]
    with pytest.raises(ValueError, match="RNG provenance"):
        paired_policy_delta(bc, rl_bad_rng)
    # Same episode key + hash, but episode config CONTENT differs.
    rl_bad_cfg = [
        _record(BANK, "ckpt_101", _ep_hash(i), 0, True,
                config_world="w2")
        for i in range(12)
    ]
    with pytest.raises(ValueError, match="episode config"):
        paired_policy_delta(bc, rl_bad_cfg)


def test_paired_delta_unpaired_episode_hard_fails():
    bc = _bc_12()
    rl = _rl_60(BANK, "ckpt_101", success_at=list(range(9)))
    rl.append(_record(BANK, "ckpt_101", _ep_hash(99), 0, True))
    with pytest.raises(ValueError, match="unpaired"):
        paired_policy_delta(bc, rl)


# ═══════════════════════════════════════════════════════════════════
# Matrix completeness and the report aggregator
# ═══════════════════════════════════════════════════════════════════

def test_paired_matrix_missing_replica_incomplete():
    bc = _bc_60()
    dropped = bc.pop(2 * 5 + 2)  # (ep_0002, replica 2)
    assert dropped.replica_id == 2
    with pytest.raises(ValueError, match="incomplete"):
        validate_paired_matrix(bc, expected_entries=12,
                               expected_replicas=5)
    # A complete 12x5 matrix passes.
    validate_paired_matrix(_bc_60(), expected_entries=12,
                           expected_replicas=5)


def test_paired_matrix_duplicate_key_hard_fails():
    bc = _bc_12()
    bc = bc + [_record(BANK, BC_CKPT, _ep_hash(0), 0, True)]
    with pytest.raises(ValueError, match="duplicate"):
        validate_paired_matrix(bc, expected_entries=12,
                               expected_replicas=5)


def test_paired_eval_report_all_three_ns():
    """§11.5.5: the report ALWAYS carries raw n, unique n and cluster n —
    never a single n. BC fed duplicated (the counterexample input: the
    same 60-instance BC saved twice must not double the effective n)."""
    bc_raw = _bc_60() + _bc_60()
    rl_by_seed = {
        "101": _rl_60(BANK, "ckpt_101", success_at=list(range(9))),
        "202": _rl_60(BANK, "ckpt_202", success_at=list(range(9))),
    }
    report = paired_eval_report(bc_raw, rl_by_seed)
    for section in ("bc", "101", "202"):
        for key in ("raw_n", "unique_n", "effective_n", "cluster_n"):
            assert key in report[section], (section, key)
    assert report["bc"]["raw_n"] == 120
    assert report["bc"]["unique_n"] == 60
    assert report["bc"]["effective_n"] == 60
    assert report["bc"]["cluster_n"] == 12
    assert report["bc"]["sr"] == 9 / 12
    assert report["101"]["unique_n"] == 60
    assert report["101"]["cluster_n"] == 12
    assert report["paired"]["101"]["common_n"] == 60


# ═══════════════════════════════════════════════════════════════════
# Gate B R6-EVAL-1 REWORK: 生产接线（bank_id 透传 / 内容一致性）
# ═══════════════════════════════════════════════════════════════════


def test_eval1_report_threads_bank_id_provenance_check():
    """§11.5.5 生产接线（Gate B R6-EVAL-1）: paired_eval_report 必须
    把 bank_id 透传进聚合 —— RNG provenance 校验（tampered
    policy_noise_seed）在 report 路径上必须生效；此前 bank_id 默认
    None 使校验在生产中从未运行。"""
    bc = _bc_12()
    tampered = [
        dataclasses.replace(r, policy_noise_seed=r.policy_noise_seed + 1)
        for r in _rl_60(BANK, "ckpt_101", success_at=list(range(9)))
    ]
    with pytest.raises(ValueError, match="RNG provenance"):
        paired_eval_report(bc, {"101": tampered}, bank_id=BANK)


def test_eval1_dedup_content_agreement_hard_fails_conflict():
    """§11.5.2 内容一致性（P2）: 折叠到同一统计单元（episode,
    replica, ckpt）的记录必须内容一致 —— 同一 episode 的 BC outcome
    在两个 artifact 中互相矛盾 = 证据冲突，hard-fail，绝不静默取其一。"""
    bc = _bc_12()
    conflict = dataclasses.replace(
        bc[0], bc=dict(bc[0].bc, success=not bc[0].bc["success"]))
    with pytest.raises(ValueError, match="content"):
        deduplicate_policy_records(bc + [conflict])


def test_eval1_dedup_content_agreement_identical_ok():
    """§11.5.2 内容一致性（P2）合法路径: 同单元重复记录内容完全一致
    （同一 artifact 的两个 BC 段）→ 正常去重，不误报。"""
    bc = _bc_12()
    dedup = deduplicate_policy_records(bc + list(bc))
    assert dedup.unique_records == 12
