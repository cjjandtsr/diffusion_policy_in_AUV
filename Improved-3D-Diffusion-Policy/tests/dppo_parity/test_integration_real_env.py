"""P8-b (B3, T2-ext): real-env integration entry — RED tests.

The T5 verdict (ASSUMPTION-9) requires real env wiring, not mock-only
integration: the gates G5/G6 and formal training must construct the env from
the frozen world pool (scan_world_pool semantics) and task specs from the
manifest (TaskSpecRegistry), with manifest records reaching the
AdmissionChecker rule 6. integration.py is that single construction point.

The e2e tests run run_e2e_batch over the REAL env (frozen KinematicUAVEnv v2
+ synth point clouds on CPU, FakePolicy) — the mock env is not involved.
"""

import json
import random

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    AdmissionExhausted,
    EpisodeTaskSpec,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_task_specs,
    build_world_pool,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import LoopConfig, run_e2e_batch
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from fake_policy import FakePolicy
from mock_uav_env import ScriptedTaskSampler


def _write_world(worlds_dir, name: str, obstacles: list) -> str:
    """Real obstacles.json schema; the frozen glob needs env_*.obstacles.json."""
    path = worlds_dir / f"{name}.obstacles.json"
    path.write_text(json.dumps({"world_id": name, "obstacles": obstacles}))
    return str(path)


def _sparse_worlds(tmp_path):
    _write_world(tmp_path, "env_test_a", [
        {"name": "c1", "shape": "cylinder", "x": 4.0, "y": 0.0,
         "radius": 1.2, "height": 4.0},
    ])
    _write_world(tmp_path, "env_test_b", [
        {"name": "c2", "shape": "cylinder", "x": -5.0, "y": 6.0,
         "radius": 0.9, "height": 3.0},
    ])


def _blocked_worlds(tmp_path):
    """Radius-30 cylinder covers the whole C2 box -> admission exhausts."""
    _write_world(tmp_path, "env_blocked", [
        {"name": "mega", "shape": "cylinder", "x": 0.0, "y": 0.0,
         "radius": 30.0, "height": 10.0},
    ])


def _entries() -> list:
    """Manifest-style entries: env_test_a trains, env_test_b evaluates."""
    return [
        {"world_uid": "env_test_a", "role": "anticipatory", "split": "rl_train",
         "manifest_entry_hash": "ab" * 32},
        # R7-8 ruling (2026-08-24): the literal eval-only split is rl_val.
        {"world_uid": "env_test_b", "role": "dense_recovery", "split": "rl_val",
         "manifest_entry_hash": "cd" * 32},
    ]


# ═══════════════════════════════════════════════════════════════

def test_build_world_pool_frozen_glob(tmp_path):
    """build_world_pool = the frozen scan_world_pool semantics: only
    env_*.obstacles.json files, name-sorted."""
    _sparse_worlds(tmp_path)
    (tmp_path / "not_env.json").write_text("{}")
    pool = build_world_pool(str(tmp_path))
    assert [n for n, _ in pool] == ["env_test_a", "env_test_b"]


def test_build_adapter_wires_manifest_rule6(tmp_path):
    """manifest_records reach the AdmissionChecker (rule 6): a spec whose
    manifest_entry_hash does not match the composite-key manifest record
    exhausts admission instead of admitting with stale metadata (R2.3)."""
    _sparse_worlds(tmp_path)
    contract = RLSystemContract()
    adapter = build_adapter(
        contract, worlds_dir=str(tmp_path),
        manifest_records={
            manifest_record_key("env_test_a", "anticipatory", "a"):
                "manifest-hash-a",
        },
        device="cpu", max_episode_steps=30,
    )
    good = EpisodeTaskSpec(
        world_uid="env_test_a", obstacle_json="unused", role="anticipatory",
        split="rl_train", admission_id="a", generator_version="routeA_rl_v1.3",
        manifest_entry_hash="manifest-hash-a",
    )
    obs, info = adapter.reset([good], seeds=[1])
    assert obs["point_cloud"].shape == (1, 8, 4096, 3)
    assert info[0]["world_uid"] == "env_test_a"
    adapter.abort_episode([0])  # reset_at refuses ACTIVE slots
    bad = EpisodeTaskSpec(
        world_uid="env_test_a", obstacle_json="unused", role="anticipatory",
        split="rl_train", admission_id="a", generator_version="routeA_rl_v1.3",
        manifest_entry_hash="stale-hash",
    )
    with pytest.raises(AdmissionExhausted):
        adapter.reset([bad], seeds=[2])


def test_build_task_specs_splits_train_eval():
    """TaskSpecRegistry wiring: train/eval specs stay apart, hashes enforced."""
    train, eval_ = build_task_specs(_entries())
    assert [s.world_uid for s in train] == ["env_test_a"]
    assert [s.world_uid for s in eval_] == ["env_test_b"]
    assert all(s.manifest_entry_hash for s in train + eval_)


# ═══════════════════════════════════════════════════════════════
# e2e over the REAL env (CPU, FakePolicy)
# ═══════════════════════════════════════════════════════════════

def _runtime(device: str) -> dict:
    return {
        "timestamp": "2026-08-07T00:00:00Z",
        "device": device,
        "reward_version": "v16",
        "bc_checkpoint_hash": "bc0123",
        "world_assets_hash": "wa456",
        "manifest_hash": "mf456",
        "source_file_hashes": {},
        "project_git_commit": "n/a",
        "project_tree_hash": "n/a",
    }


def _run(tmp_path, mode: str, n_episodes: int = 1, max_ep_len: int = 4):
    # The real-env rollout draws the DDIM initial noise from the GLOBAL
    # torch RNG (FakePolicy, fake_policy.py) — without a pin, the trajectory
    # depends on the RNG state left by whichever tests ran before, and the
    # suite-order variation produced a one-off all-far frame -> SAFE_END
    # (gate invalid_count == 1). Pin the global RNGs here so the e2e
    # verdict is suite-order independent (the env itself is seeded per
    # episode via env.reset(seeds=[attempt])).
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    contract = RLSystemContract()
    _sparse_worlds(tmp_path)
    train_specs, _ = build_task_specs(_entries())
    # Composite-key rule-6 records built from the specs (registry
    # admission_id defaults guaranteed to match).
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in train_specs
    }
    env = build_adapter(
        contract, worlds_dir=str(tmp_path),
        manifest_records=records,
        device="cpu", max_episode_steps=30,
    )
    # This fixture has ONE train spec (env_test_a) — the production
    # EpisodeTaskSampler hard-fails on empty role buckets, so the wiring
    # test drives the loop with a scripted spy sampler over that spec.
    # base_seed=1 -> reset seed 1, byte-identical to seeds=[attempt] with
    # attempt=1 (the trajectory this fixture pins via action_scale=0.0002
    # + torch.manual_seed(7) was probed at seed 1).
    task_sampler = ScriptedTaskSampler(train_specs, base_seed=1)
    # action_scale mirrors the real action normalizer's std: the raw DDIM
    # output (~N(0,1)) maps to physical meter deltas. The fake's CFG branch
    # is live (cond_proj constant in eps); the DDIM predicted-x0
    # reconstruction amplifies that constant by ~1/sqrt(alpha_bar) at the
    # first denoising steps, so the effective first-step displacement is
    # far larger than action_scale suggests. Probed (CFG-mixed sampler,
    # seed pin 7): 0.01/0.005 teleport the drone out of the world -> all-far
    # frame -> SAFE_END (0 transitions); 0.002 valid at seed 7 but invalid
    # at seeds 2/5; 0.0002 valid (transitions=4, invalid_count=0) across
    # seeds 0-5 — the largest scale with margin, keeping the wiring-test
    # trajectory in-world.
    policy = FakePolicy(contract, action_scale=0.0002)
    critic = ValueCritic(contract=contract, encoder=PolicyObsEncoder(policy))
    cfg = LoopConfig(
        mode=mode, n_episodes=n_episodes, max_ep_len=max_ep_len,
        device="cpu", output_dir=str(tmp_path),
    )
    return run_e2e_batch(
        env=env, policy=policy, critic=critic, contract=contract,
        loop_config=cfg, trainer_config=TrainerConfig(device="cpu"),
        task_specs=train_specs,
        actor_optimizer=torch.optim.AdamW(policy.model.parameters(), lr=1e-4),
        critic_optimizer=torch.optim.AdamW(
            critic.value_head_parameters(), lr=1e-4
        ),
        lora_info={"lora_params": []},
        runtime=_runtime("cpu"),
        task_sampler=task_sampler,
    )


def test_real_env_e2e_no_update(tmp_path):
    """run_e2e_batch over the REAL frozen env: rollouts pass the gate, the
    buffer fills, nothing is written (no_update mode)."""
    result = _run(tmp_path, "no_update")
    assert result["completed_episodes"] == 1
    assert result["transitions"] > 0
    assert result["rejected_episodes"] == 0
    assert result["gate_summary"]["invalid_count"] == 0
    assert result["metrics"] is None
    assert not list(tmp_path.glob("*.pt"))


def test_real_env_e2e_one_update(tmp_path):
    """one_update over the REAL env: checkpoint saved with the runtime
    contract hash; role stats wired from the real spec roles."""
    result = _run(tmp_path, "one_update")
    assert result["metrics"] is not None
    assert result["metrics"]["update_batches"] >= 1
    assert result["checkpoint_path"] is not None
    rs = result["role_stats"]
    assert rs["completed_episode_counts"]["anticipatory"] == 1
    assert sum(rs["attempted_episode_counts"].values()) == 1
    assert result["gate_summary"]["invalid_count"] == 0
    data = CheckpointBuilder.load(result["checkpoint_path"])
    assert data.ppo_update_count == 1
    assert data.completed_episode_count == 1
    # The checkpoint must carry BOTH hashes — the manifest SOURCE hash
    # (bound to the spec, "ab"*32) and the sampled runtime episode hash
    # (computed at reset).
    meta = data.exact_resolved_config.get("episode_metadata", {})
    assert meta.get("manifest_entry_hash") == "ab" * 32
    assert meta.get("episode_config_hash")


def test_real_env_e2e_admission_exhausted(tmp_path):
    """A fully blocked world surfaces AdmissionExhausted through the real
    adapter (rule 7) — never a fixed-coordinate fallback in the real run."""
    _blocked_worlds(tmp_path)
    contract = RLSystemContract()
    env = build_adapter(
        contract, worlds_dir=str(tmp_path), device="cpu", max_episode_steps=30,
    )
    spec = EpisodeTaskSpec(
        world_uid="env_blocked", obstacle_json="unused", role="anticipatory",
        split="rl_train", admission_id="b", generator_version="routeA_rl_v1.3",
        manifest_entry_hash="cd" * 32,
    )
    with pytest.raises(AdmissionExhausted):
        env.reset([spec], seeds=[1])
