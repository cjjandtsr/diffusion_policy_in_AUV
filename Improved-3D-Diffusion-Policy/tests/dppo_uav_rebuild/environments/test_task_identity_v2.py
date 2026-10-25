"""Stable task identity and seed derivation across ordering/process boundaries."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import (
    EpisodeTaskSpec,
    TRAINABLE_ROLES,
    canonical_task_identity,
    derive_task_seed,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory import (
    canonical_environment_sampling_spec,
)


def _entry(task_id="task_a", world_id="train_001", role="anticipatory"):
    return {
        "task_id": task_id,
        "world_id": world_id,
        "split": "rl_train",
        "role": role,
        "start_goal_spec": {"curriculum": "C2"},
        "admission_schema_version": "v1.3",
    }


def test_task_seed_is_identity_based_not_manifest_index():
    entries = [_entry("a", "train_a"), _entry("b", "train_b", "dense_recovery")]
    forward = {
        entry["task_id"]: derive_task_seed(20260828, canonical_task_identity(entry, "a" * 64))
        for entry in entries
    }
    reverse = {
        entry["task_id"]: derive_task_seed(20260828, canonical_task_identity(entry, "a" * 64))
        for entry in reversed(entries)
    }
    assert forward == reverse
    assert len(set(forward.values())) == 2


def test_identity_rejects_nan_and_untracked_fields_do_not_change_it():
    entry = _entry()
    first = canonical_task_identity(entry, "b" * 64)
    entry["display_name"] = "cosmetic"
    assert canonical_task_identity(entry, "b" * 64) == first
    entry["start_goal_spec"] = {"bad": float("nan")}
    with pytest.raises(ValueError):
        canonical_task_identity(entry, "b" * 64)


def test_roles_and_identity_are_hash_seed_independent(tmp_path):
    code = (
        "from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import *;"
        "e={'task_id':'a','world_id':'train_a','split':'rl_train','role':'anticipatory',"
        "'start_goal_spec':{},'admission_schema_version':'v1.3'};"
        "print(TRAINABLE_ROLES);print(canonical_task_identity(e,'c'*64));"
        "print(derive_task_seed(9,canonical_task_identity(e,'c'*64)))"
    )
    outputs = []
    for seed in ("1", "17", "91", "313", "999"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        outputs.append(subprocess.check_output([sys.executable, "-c", code], env=env, text=True))
    assert len(set(outputs)) == 1


@pytest.mark.parametrize("world_id", ["w4", "w8", "w11", "w12", "w18"])
def test_eval_only_exact_id_is_rejected(world_id):
    with pytest.raises(ValueError, match="eval-only"):
        EpisodeTaskSpec("bad", world_id, "x.json", "anticipatory", 1, True)


def test_w40_is_not_rejected_by_substring_matching():
    task = EpisodeTaskSpec("ok", "w40", "x.json", "anticipatory", 1, True)
    assert task.world_id == "w40"


def _sampling_controls():
    return {
        "max_episode_steps": 60,
        "disturb_enable": True,
        "disturb_prob": 0.2,
        "disturb_pos_scale": 0.3,
        "exec_noise_scale": 0.0,
        "mid_state_init_ratio": 0.3,
        "mid_state_pos_scale": 0.5,
        "max_reset_attempts": 50,
    }


def test_environment_sampling_identity_covers_every_stochastic_control():
    baseline = canonical_environment_sampling_spec(_sampling_controls())
    assert baseline["astar_is_diagnostic_only"] is True
    assert baseline["curriculum_stage"] == "C2"
    for field in _sampling_controls():
        changed_controls = _sampling_controls()
        value = changed_controls[field]
        if isinstance(value, bool):
            changed_controls[field] = not value
        elif field in ("disturb_prob", "mid_state_init_ratio"):
            changed_controls[field] = value / 2
        else:
            changed_controls[field] = value + 1
        assert canonical_environment_sampling_spec(changed_controls) != baseline


def test_environment_sampling_identity_rejects_missing_or_invalid_controls():
    with pytest.raises(ValueError, match="missing"):
        canonical_environment_sampling_spec({"max_episode_steps": 60})
    controls = _sampling_controls()
    controls["mid_state_init_ratio"] = 1.1
    with pytest.raises(ValueError, match="mid_state_init_ratio"):
        canonical_environment_sampling_spec(controls)
