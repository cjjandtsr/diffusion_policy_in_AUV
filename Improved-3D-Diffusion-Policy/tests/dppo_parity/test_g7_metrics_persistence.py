"""G7 gate contract (design §4.15, §5.9): per-update learning-signal
persistence in the canonical training entry.

The canonical entry must RECORD every update's §4.15 metrics (the
trainer computes them; the entry previously discarded them) — the G7
gate items "per-denoise KL/clipfrac 均有结构化输出" and "action
variance 不低于 BC 的 10%" are unverifiable without the persisted
stream, and NaN/Inf must hard-fail the run (MasterPlan §6.1 hard
fault, enforced at persist time — a non-finite metric can never be
silently recorded).

R6-C (§11.4.1/§11.4.3): the NaN-as-no-data sentinel is abolished.
Empty categories are the OptionalStatistic two-state contract
(count==0 -> value None/valid False/frozen reason; count>0 -> finite
value/valid True) and the writer emits strict JSON only
(allow_nan=False + sort_keys + schema_version).
"""

import json

import numpy as np
import pytest

from scripts.train_dppo_uav_official import (
    persist_batch_metrics,
    _json_safe_metrics,
)


def _fake_res(**overrides):
    res = {
        "env_steps": 997,
        "completed_episodes": 32,
        "voided_episodes": 0,
        "routing": {"REJECT_EPISODE": 0, "SAFE_END_EPISODE": 0,
                    "EXCLUDE_TRANSITION": 0, "HARD_FAILURE": 0},
        "metrics": {
            "approx_kl": 0.05,
            "actor_grad_norm": 0.8,
            "per_denoise_approx_kl": {0: 0.02, 1: 0.05, 2: 0.14},
            "per_denoise_clipfrac": {0: 0.9, 1: 0.8},
            "action_variance_by_dim": [0.1, 0.2, 0.3],
            "policy_action_diversity": 0.2,
        },
    }
    res.update(overrides)
    return res


def test_persist_batch_metrics_appends_jsonl(tmp_path):
    """Two batches append two valid JSONL records with the §4.15 keys."""
    path = tmp_path / "metrics.jsonl"
    r1 = persist_batch_metrics(path, 1, _fake_res())
    r2 = persist_batch_metrics(path, 2, _fake_res(env_steps=2001))
    assert r1["batch"] == 1 and r2["batch"] == 2
    lines = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert len(lines) == 2
    for record, batch in zip(lines, (1, 2)):
        assert record["batch"] == batch
        for key in ("env_steps", "completed_episodes", "voided_episodes",
                    "routing", "metrics"):
            assert key in record, record.keys()
    # JSON round-trip stringifies dict keys (the report consumer treats
    # per-denoise step keys opaquely — same convention as the G6 report).
    assert lines[0]["metrics"]["per_denoise_approx_kl"] == {"0": 0.02,
                                                            "1": 0.05,
                                                            "2": 0.14}
    assert lines[1]["env_steps"] == 2001


def test_persist_batch_metrics_rejects_nonfinite_top_level(tmp_path):
    """NaN/Inf in any top-level numeric metric hard-fails (G7 NaN/Inf=0)."""
    path = tmp_path / "metrics.jsonl"
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            persist_batch_metrics(path, 1, _fake_res(metrics={
                "approx_kl": 0.05, "bad": bad}))
    assert not path.exists() or path.read_text() == ""


def test_persist_batch_metrics_rejects_nonfinite_nested(tmp_path):
    """NaN/Inf inside non-sentinel nested dicts hard-fails too
    (per-role conditioned advantage has no NaN sentinel — a role with
    zero transitions would be a buffer bug)."""
    path = tmp_path / "metrics.jsonl"
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            persist_batch_metrics(
                path, 1, _fake_res(metrics={
                    "approx_kl": 0.05,
                    "role_conditioned_advantage": {"anticipatory": bad},
                }),
            )
    assert not path.exists() or path.read_text() == ""


def test_persist_batch_metrics_roundtrips_optional_statistics(tmp_path):
    """R6-C (§11.4.1/§11.4.3): null statistics round-trip as JSON null,
    never as NaN literals — count==0 -> value None/valid False/frozen
    reason; count>0 -> finite value/valid True. A NaN literal anywhere
    in the record text is a strict-JSON violation."""
    path = tmp_path / "metrics.jsonl"
    metrics = {
        "approx_kl": 0.05,
        "actor_grad_norm": 0.8,
        "per_denoise_approx_kl": {0: 0.02, 1: 0.09, 2: 0.14},
        "per_denoise_clipfrac": {0: 0.9, 1: 0.8},
        "success_conditioned_advantage": {
            "sample_count": 0, "value": None, "valid": False,
            "invalid_reason": "no_success_terminal",
        },
        "collision_conditioned_advantage": {
            "sample_count": 3, "value": -2.1, "valid": True,
            "invalid_reason": None,
        },
        "action_variance_by_dim": [0.1, 0.2, 0.3],
    }
    persist_batch_metrics(path, 1, _fake_res(metrics=metrics))
    raw = path.read_text()
    assert "nan" not in raw.lower() and "inf" not in raw.lower()
    m = json.loads(raw.splitlines()[0])["metrics"]
    assert m["success_conditioned_advantage"] == {
        "sample_count": 0, "value": None, "valid": False,
        "invalid_reason": "no_success_terminal",
    }
    assert m["collision_conditioned_advantage"] == {
        "sample_count": 3, "value": -2.1, "valid": True,
        "invalid_reason": None,
    }
    assert m["actor_grad_norm"] == 0.8


def test_json_safe_metrics_accepts_plain_and_nested():
    """The validator passes on finite plain + nested numerics."""
    _json_safe_metrics({"a": 1.0, "b": {"c": 2, "d": -3.5},
                        "e": [0.1, 0.2]})


def test_persist_batch_metrics_no_metrics_is_noop(tmp_path):
    """no_update batches (metrics=None) write nothing and return None
    (review_R4_part1 P1-2: the unconditional call crashed the stage)."""
    path = tmp_path / "metrics.jsonl"
    res = _fake_res()
    res["metrics"] = None
    assert persist_batch_metrics(path, 1, res) is None
    assert not path.exists()


def test_persist_batch_metrics_records_admission_blocked(tmp_path):
    """The admission-blocked counter lands in the JSONL (review_R4_part1
    P2-4: the G7 fault gate must see it)."""
    path = tmp_path / "metrics.jsonl"
    persist_batch_metrics(path, 1, _fake_res(admission_blocked_episodes=2))
    record = json.loads(path.read_text().splitlines()[0])
    assert record["admission_blocked_episodes"] == 2


def test_json_safe_metrics_rejects_inf_even_in_sentinel_fields():
    """Inf is NEVER a sentinel — it hard-fails even in the NaN fields."""
    with pytest.raises(ValueError):
        _json_safe_metrics({"success_conditioned_advantage": float("inf")})
    with pytest.raises(ValueError):
        _json_safe_metrics({"per_denoise_approx_kl": {0: float("inf")}})


def test_json_safe_metrics_rejects_inf_inside_optional_statistic():
    """R6-C (§11.4.1): the two-state contract never carries non-finite
    values — an inf inside an OptionalStatistic dict is a forbidden
    third state and hard-fails at persist time."""
    with pytest.raises(ValueError):
        _json_safe_metrics({"success_conditioned_advantage": {
            "sample_count": 1, "value": float("inf"), "valid": True,
            "invalid_reason": None,
        }})
    with pytest.raises(ValueError):
        _json_safe_metrics({"explained_variance": {
            "target_variance": 1.0, "sample_count": 4, "value": float("nan"),
            "valid": True, "invalid_reason": None,
        }})
