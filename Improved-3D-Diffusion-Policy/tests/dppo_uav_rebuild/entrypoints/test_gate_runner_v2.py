from omegaconf import OmegaConf
import hashlib
import json
import numpy as np
import pytest
import torch

from scripts.run_routea_v17_gates import (
    GATE_NAMES,
    _assert_fallback_compatible,
    _classify_pytest_result,
    load_and_recompute_d1b,
    run_gates,
)
from scripts.dppo_credit_support.credit_probe import assess_sampled_k_gate


def _config():
    return OmegaConf.create({
        "reward_version": "v17", "reward": {"H_e": 4},
        "optimizer": {"actor_lr": 2e-5, "actor_lr_end": 1e-5},
        "run": {"phase": "pilot", "seed": 202609101, "updates": 32,
                "rollout_transitions": 1024, "schedule_label": "S0",
                "checkpoint_updates": [0, 8, 16, 24, 32],
                "expected_optimizer_commits_per_outer_update": 40},
        "dppo": {"denoising_schedule": "sampled_k", "pair_passes": 5, "clip_ploss_coef": 0.1,
                 "logical_batch_pairs": 128, "microbatch_pairs": 128},
        "environment": {
            "max_episode_steps": 60, "disturb_enable": True,
            "disturb_prob": 0.2, "disturb_pos_scale": 0.3,
            "exec_noise_scale": 0.0, "mid_state_init_ratio": 0.3,
            "mid_state_pos_scale": 0.5, "max_reset_attempts": 50,
            "reject_all_no_ray_hit": False,
            "max_no_ray_hit_fraction": 0.9,
        },
    })


def test_missing_d1b_blocks_instead_of_defaulting_to_pass():
    result = run_gates(_config())
    assert result.status == "BLOCKED"
    assert result.selected_schedule is None
    assert tuple(row["name"] for row in result.gates) == GATE_NAMES


def test_gate_runner_never_calls_optimizer_step(monkeypatch):
    monkeypatch.setattr(torch.optim.Optimizer, "step", lambda self: (_ for _ in ()).throw(AssertionError("TRAINING_FORBIDDEN")))
    evidence = {"schema": "routea_v17_d1b_summary_v2", "s0": {"decision": "PASS"},
                "s2": {"decision": "PASS"}, "tripwire": {"all_unchanged": True}}
    preflight = {"schema": "routea_v17_preflight_summary_v2",
                 "gates": {name: "PASS" for name in GATE_NAMES[:10]}}
    result = run_gates(_config(), preflight_summary=preflight, d1b_summary=evidence)
    assert result.status == "PASS"
    assert result.selected_schedule == "S0"


def test_d1b_summary_is_recomputed_from_hash_bound_raw_arrays(tmp_path):
    reference = {"anticipatory": np.array([1.0, 2.0], np.float32)}
    schedules = np.repeat(reference["anticipatory"][None, :], 100, axis=0)
    np.save(tmp_path / "ref.npy", reference["anticipatory"], allow_pickle=False)
    np.save(tmp_path / "schedules.npy", schedules, allow_pickle=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        "schema": "routea_v17_d1b_summary_v2",
        "s0": assess_sampled_k_gate(reference, [{"anticipatory": row} for row in schedules]),
        "s2": {"decision": "PASS"}, "tripwire": {"all_unchanged": True},
        "gradient_array_manifest": {"anticipatory": {
            "full_joint": {"path": "ref.npy", "sha256": sha(tmp_path / "ref.npy")},
            "sampled_k_100": {"path": "schedules.npy", "sha256": sha(tmp_path / "schedules.npy")},
        }},
    }
    body = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    summary["content_sha256"] = hashlib.sha256(body).hexdigest()
    (tmp_path / "d1b_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert load_and_recompute_d1b(tmp_path)["s0"]["decision"] == "PASS"


def test_preflight_classification_rejects_skips_xfails_and_xpasses():
    assert _classify_pytest_result(0, "12 passed") == "PASS"
    for output in ("11 passed, 1 skipped", "1 xfailed", "1 xpassed"):
        assert _classify_pytest_result(0, output) == "BLOCKED"
    assert _classify_pytest_result(0, "12 passed, 1 warning") == "PASS"
    assert _classify_pytest_result(1, "failed") == "BLOCKED"


def test_fallback_may_change_only_registered_schedule_and_output():
    primary = _config()
    fallback = OmegaConf.create(OmegaConf.to_container(primary, resolve=True))
    fallback.run.schedule_label = "S2"
    fallback.dppo.denoising_schedule = "full_joint_accum_batch1280"
    fallback.dppo.logical_batch_pairs = 1280
    primary.output_dir = "s0"
    fallback.output_dir = "s2"
    _assert_fallback_compatible(primary, fallback)
    fallback.reward.H_e = 8
    with pytest.raises(ValueError, match="outside registered schedule"):
        _assert_fallback_compatible(primary, fallback)
