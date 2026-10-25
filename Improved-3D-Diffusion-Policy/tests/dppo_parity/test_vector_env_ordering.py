"""R2.4: ordered synchronous vector environments on the REAL adapter.

Guide (Guide_A_OfficialCore_R2_Sampler_Env_Contracts_v1_0) Task R2.4:
  - reset(task_specs, seeds) with len == len == n_envs
  - reset_at(env_ids, task_specs, seeds): resets only the named slots,
    results ordered exactly like the env_ids INPUT
  - step(actions) strictly (n_envs, exec_steps, 11); results ALWAYS in
    env_id 0..n_envs-1 order; StepBatch fields are (n_envs,) arrays, info
    entries carry env_id
  - a slot that terminated/truncated must NOT step before reset_at — no
    auto-reset that hides the boundary
  - the train loop maintains an active mask and refills done slots via
    reset_at (covered by test_train_loop_active_mask_two_slots)

Step-1 RED: the silent-drop implementation (single _env, step reads
actions[0] only) returns ONE reward/obs/info — every shape assert here
fails on it.
"""

import random

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    EpisodeTaskSpec,
    UAVVectorEnvAdapter,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import LoopConfig, run_e2e_batch
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from fake_policy import FakePolicy
from mock_uav_env import ScriptedTaskSampler
from test_integration_real_env import _write_world

CONTRACT = RLSystemContract()


def _two_worlds(tmp_path) -> list:
    """env_a: cylinder on the +x path; env_b: cylinder offset at (-5, 6)."""
    _write_world(tmp_path, "env_a", [
        {"name": "c1", "shape": "cylinder", "x": 4.0, "y": 0.0,
         "radius": 1.2, "height": 4.0},
    ])
    _write_world(tmp_path, "env_b", [
        {"name": "c2", "shape": "cylinder", "x": -5.0, "y": 6.0,
         "radius": 0.9, "height": 3.0},
    ])
    return [("env_a", str(tmp_path / "env_a.obstacles.json")),
            ("env_b", str(tmp_path / "env_b.obstacles.json"))]


def _specs() -> list:
    return [
        EpisodeTaskSpec(
            world_uid="env_a", obstacle_json="unused", role="anticipatory",
            split="rl_train", admission_id="a__anticipatory",
            generator_version="routeA_rl_v1.3", manifest_entry_hash="ab" * 32,
        ),
        EpisodeTaskSpec(
            world_uid="env_b", obstacle_json="unused", role="dense_recovery",
            split="rl_train", admission_id="b__dense_recovery",
            generator_version="routeA_rl_v1.3", manifest_entry_hash="cd" * 32,
        ),
    ]


def _adapter2(tmp_path, n_envs=2, max_episode_steps=30, **kwargs):
    return UAVVectorEnvAdapter(
        CONTRACT, world_pool=_two_worlds(tmp_path), device="cpu",
        max_episode_steps=max_episode_steps, n_envs=n_envs, **kwargs,
    )


# ═══════════════════════════════════════════════════════════════
# RED: two slots, two distinct results (never actions[0]-only)
# ═══════════════════════════════════════════════════════════════

def test_two_slots_return_two_distinct_results(tmp_path):
    """Silent-drop RED: reset returns obs/info for BOTH slots, step returns
    (2,) reward/terminated/truncated and 2 info entries; env 1's result is
    not env 0's (different worlds, different actions)."""
    adapter = _adapter2(tmp_path)
    specs = _specs()
    obs, info = adapter.reset(specs, seeds=[41, 42])
    assert obs["point_cloud"].shape == (2, 8, 4096, 3)
    assert obs["agent_pos"].shape == (2, 8, 9)
    assert len(info) == 2
    assert info[0]["env_id"] == 0 and info[1]["env_id"] == 1
    assert info[0]["world_uid"] == "env_a" and info[1]["world_uid"] == "env_b"
    assert info[0]["manifest_entry_hash"] == "ab" * 32
    assert info[1]["manifest_entry_hash"] == "cd" * 32

    actions = np.zeros((2, 4, 11), dtype=np.float32)
    actions[1, :, 0] = 0.05  # slot 1 pushes along x
    batch = adapter.step(actions)
    assert batch.reward.shape == (2,)
    assert batch.terminated.shape == (2,)
    assert batch.truncated.shape == (2,)
    assert len(batch.info) == 2
    assert batch.info[0]["env_id"] == 0 and batch.info[1]["env_id"] == 1
    assert batch.info[0]["world_uid"] == "env_a"
    assert batch.info[1]["world_uid"] == "env_b"
    # env 1's frame differs from env 0's — a [0]-only implementation returns
    # env 0 twice and this fails
    assert not np.array_equal(
        batch.obs["point_cloud"][1], batch.obs["point_cloud"][0]
    )
    assert batch.info[0]["episode_config_hash"] != batch.info[1]["episode_config_hash"]


# ═══════════════════════════════════════════════════════════════
# ordering: reset_at input order, env_id stability, no drift
# ═══════════════════════════════════════════════════════════════

def test_reset_at_returns_in_env_ids_order(tmp_path):
    """reset_at results follow the INPUT env_ids order ([1, 0] -> slot 1
    first); the per-slot hashes bind to each slot's own spec."""
    adapter = _adapter2(tmp_path)
    specs = _specs()
    obs, info = adapter.reset_at([1, 0], [specs[1], specs[0]], seeds=[7, 8])
    assert len(info) == 2
    assert info[0]["env_id"] == 1 and info[0]["world_uid"] == "env_b"
    assert info[0]["manifest_entry_hash"] == "cd" * 32
    assert info[1]["env_id"] == 0 and info[1]["world_uid"] == "env_a"
    assert info[1]["manifest_entry_hash"] == "ab" * 32
    # a full step reports env_id 0..1 with the reset slots' worlds
    batch = adapter.step(np.zeros((2, 4, 11), dtype=np.float32))
    assert batch.info[0]["world_uid"] == "env_a"
    assert batch.info[1]["world_uid"] == "env_b"


def test_partial_reset_at_keeps_slot_identity(tmp_path):
    """reset_at([0]) re-episodes slot 0 ONLY AFTER it is done (rework #3:
    reset_at accepts only done/voided slots — the pre-rework version reset
    an ACTIVE slot and silently discarded its episode); slot 1's episode
    (seed 2) is untouched — env_id does not drift after a partial reset."""
    adapter = _adapter2(tmp_path, max_episode_steps=8)
    specs = _specs()
    adapter.reset(specs, seeds=[1, 2])
    # place slot 0's drone 0.15 m from env_a's cylinder, pushing into it
    env0 = adapter._envs[0]
    env0._drone_pos = np.array([2.65, 0.0, 2.0], dtype=np.float32)
    env0._drone_yaw = 0.0
    env0._drone_vel = np.zeros(3, dtype=np.float32)
    actions = np.zeros((2, 4, 11), dtype=np.float32)
    actions[0, :, 0] = 0.5  # push slot 0 into the cylinder
    batch = adapter.step(actions)
    assert bool(batch.terminated[0]), "slot 0 must be done after collision"
    obs, info = adapter.reset_at([0], [specs[0]], seeds=[99])
    assert info[0]["env_id"] == 0 and info[0]["world_uid"] == "env_a"
    assert adapter.current_episode_config(0)["seed"] == 99
    assert adapter.current_episode_config(1)["seed"] == 2
    assert adapter.current_episode_config(1)["world"] == "env_b"
    batch = adapter.step(np.zeros((2, 4, 11), dtype=np.float32))
    assert batch.info[0]["world_uid"] == "env_a"
    assert batch.info[1]["world_uid"] == "env_b"


# ═══════════════════════════════════════════════════════════════
# partial done: different episode lengths, out-of-order done slots
# ═══════════════════════════════════════════════════════════════

def test_done_slot_blocks_step_until_reset_at(tmp_path):
    """Slot 0 terminates at step 1 (collision), slot 1 keeps flying — an
    out-of-order done with different episode lengths. The done slot MUST
    NOT step again before reset_at (no auto-reset hiding the boundary);
    after reset_at([0]) both slots step, slot 1's episode never restarted."""
    adapter = _adapter2(tmp_path, max_episode_steps=8)
    specs = _specs()
    adapter.reset(specs, seeds=[1, 2])
    # place slot 0's drone 0.15 m from env_a's cylinder, pushing into it
    env0 = adapter._envs[0]
    env0._drone_pos = np.array([2.65, 0.0, 2.0], dtype=np.float32)
    env0._drone_yaw = 0.0
    env0._drone_vel = np.zeros(3, dtype=np.float32)
    actions = np.zeros((2, 4, 11), dtype=np.float32)
    actions[0, :, 0] = 0.5  # push slot 0 into the cylinder
    batch = adapter.step(actions)
    assert bool(batch.terminated[0]) and not bool(batch.terminated[1])
    assert not bool(batch.truncated[0])
    # stepping the done slot before reset_at must raise — no auto-reset
    with pytest.raises(RuntimeError, match="reset_at"):
        adapter.step(np.zeros((2, 4, 11), dtype=np.float32))
    # reset_at([0]) unblocks slot 0; slot 1 continues its ORIGINAL episode
    obs, info = adapter.reset_at([0], [specs[0]], seeds=[21])
    assert info[0]["env_id"] == 0 and info[0]["world_uid"] == "env_a"
    assert adapter.current_episode_config(1)["seed"] == 2
    batch2 = adapter.step(np.zeros((2, 4, 11), dtype=np.float32))
    assert batch2.info[0]["world_uid"] == "env_a"
    assert batch2.info[1]["world_uid"] == "env_b"
    assert not bool(batch2.terminated[0]) or not bool(batch2.terminated[1])


# ═══════════════════════════════════════════════════════════════
# train loop active mask (guide Step 3): reset_at refill, one update
# ═══════════════════════════════════════════════════════════════

def _runtime() -> dict:
    return {
        "timestamp": "2026-08-08T00:00:00Z",
        "device": "cpu",
        "reward_version": "v16",
        "bc_checkpoint_hash": "bc0123",
        "world_assets_hash": "wa456",
        "manifest_hash": "mf456",
        "source_file_hashes": {},
        "project_git_commit": "n/a",
        "project_tree_hash": "n/a",
    }


def test_train_loop_active_mask_two_slots(tmp_path):
    """run_e2e_batch drives BOTH slots through reset_at + an active mask:
    one episode per world completes, both roles contribute transitions, and
    the checkpoint binds slot 0's episode metadata. The pre-R2.4 loop only
    ever reset/stepped slot 0 — completed_episodes stays 1 and this fails."""
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    contract = RLSystemContract()
    pool = _two_worlds(tmp_path)
    specs = _specs()
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs
    }
    env = UAVVectorEnvAdapter(
        contract, world_pool=pool, manifest_records=records, device="cpu",
        max_episode_steps=30, n_envs=2,
    )
    policy = FakePolicy(contract, action_scale=0.0002)
    critic = ValueCritic(contract=contract, encoder=PolicyObsEncoder(policy))
    cfg = LoopConfig(
        mode="one_update", n_episodes=2, max_ep_len=4, device="cpu",
        output_dir=str(tmp_path),
    )
    result = run_e2e_batch(
        env=env, policy=policy, critic=critic, contract=contract,
        loop_config=cfg, trainer_config=TrainerConfig(device="cpu"),
        task_specs=specs,
        actor_optimizer=torch.optim.AdamW(policy.model.parameters(), lr=1e-4),
        critic_optimizer=torch.optim.AdamW(
            critic.value_head_parameters(), lr=1e-4
        ),
        lora_info={"lora_params": []},
        runtime=_runtime(),
        task_sampler=ScriptedTaskSampler(specs, base_seed=1),
    )
    assert result["completed_episodes"] == 2
    assert result["transitions"] == 8  # 2 episodes x 4 steps
    assert result["rejected_episodes"] == 0
    assert result["gate_summary"]["invalid_count"] == 0
    rs = result["role_stats"]
    assert rs["completed_episode_counts"]["anticipatory"] == 1
    assert rs["completed_episode_counts"]["dense_recovery"] == 1
    assert rs["transition_counts"]["anticipatory"] == 4
    assert rs["transition_counts"]["dense_recovery"] == 4
    # checkpoint episode_metadata binds slot 0 (world env_a, seed 1)
    data = CheckpointBuilder.load(result["checkpoint_path"])
    meta = data.exact_resolved_config.get("episode_metadata", {})
    assert meta.get("world") == "env_a"
    assert meta.get("manifest_entry_hash") == "ab" * 32
    assert meta.get("episode_config_hash")


# ═══════════════════════════════════════════════════════════════
# Slot lifecycle — reset_at active hard-fail, mark_voided, step_at
# partial stepping
# ═══════════════════════════════════════════════════════════════

def test_reset_at_active_slot_hard_fails_before_mutation(tmp_path):
    """Rework #3 RED: reset_at on an ACTIVE slot must raise BEFORE any
    mutation — the in-flight episode must never be silently replaced
    (guide R2.4: reset_at only resets done/voided slots). On the current
    code this reset succeeds and the episode is silently discarded."""
    adapter = _adapter2(tmp_path)
    specs = _specs()
    adapter.reset(specs, seeds=[1, 2])
    adapter.step(np.zeros((2, 4, 11), dtype=np.float32))  # both ACTIVE now
    with pytest.raises(RuntimeError, match="ACTIVE"):
        adapter.reset_at([0], [specs[0]], seeds=[99])
    # no mutation happened: slot 0 still runs the ORIGINAL episode
    assert adapter.current_episode_config(0)["seed"] == 1


def test_mark_voided_closes_episode_and_allows_reset(tmp_path):
    """Rework #3 RED: mark_voided(env_ids) is the explicit ACTIVE -> VOIDED
    transition (observation-fault path). A VOIDED slot cannot step (boundary
    is never hidden) but CAN be reset_at — the voided episode is closed.
    mark_voided on a non-ACTIVE slot raises. The current adapter has no
    VOIDED state: mark_voided is missing entirely."""
    adapter = _adapter2(tmp_path)
    specs = _specs()
    adapter.reset(specs, seeds=[1, 2])
    adapter.step(np.zeros((2, 4, 11), dtype=np.float32))
    adapter.mark_voided([0])  # closes slot 0's episode as VOIDED
    # a VOIDED slot must not step
    with pytest.raises(RuntimeError, match="done"):
        adapter.step(np.zeros((2, 4, 11), dtype=np.float32))
    # VOIDED is a legal reset_at target
    obs, info = adapter.reset_at([0], [specs[0]], seeds=[99])
    assert info[0]["env_id"] == 0 and info[0]["world_uid"] == "env_a"
    assert adapter.current_episode_config(0)["seed"] == 99
    # mark_voided on a non-ACTIVE (done) slot raises
    adapter.mark_voided([1])  # closes slot 1 too
    with pytest.raises(RuntimeError, match="ACTIVE"):
        adapter.mark_voided([1])  # now DONE, not ACTIVE


def test_step_at_steps_only_named_slots(tmp_path):
    """Rework #3 RED: step_at(env_ids, actions) steps ONLY the named ACTIVE
    slots — the loop's final partial batch (< n_envs active) never touches
    the other slots; results come back in env_ids INPUT order. The current
    adapter has no step_at (missing AttributeError)."""
    adapter = _adapter2(tmp_path)
    specs = _specs()
    adapter.reset(specs, seeds=[1, 2])
    batch = adapter.step_at([1], np.zeros((1, 4, 11), dtype=np.float32))
    assert batch.reward.shape == (1,)
    assert len(batch.info) == 1
    assert batch.info[0]["env_id"] == 1
    assert batch.info[0]["world_uid"] == "env_b"
    # slot 0 untouched — its episode continues with the ORIGINAL seed
    assert adapter.current_episode_config(0)["seed"] == 1
    assert adapter.current_episode_config(1)["seed"] == 2


def test_step_at_blocks_done_slot(tmp_path):
    """Rework #3 RED: step_at on a done/voided slot raises (same boundary as
    full step — no auto-reset hiding the boundary)."""
    adapter = _adapter2(tmp_path, max_episode_steps=8)
    specs = _specs()
    adapter.reset(specs, seeds=[1, 2])
    # place slot 0's drone 0.15 m from env_a's cylinder, pushing into it —
    # same recipe as test_done_slot_blocks_step_until_reset_at
    env0 = adapter._envs[0]
    env0._drone_pos = np.array([2.65, 0.0, 2.0], dtype=np.float32)
    env0._drone_yaw = 0.0
    env0._drone_vel = np.zeros(3, dtype=np.float32)
    actions = np.zeros((2, 4, 11), dtype=np.float32)
    actions[0, :, 0] = 0.5  # push slot 0 into the cylinder
    batch = adapter.step(actions)
    assert bool(batch.terminated[0]), "slot 0 must be done after collision"
    with pytest.raises(RuntimeError, match="done"):
        adapter.step_at([0], np.zeros((1, 4, 11), dtype=np.float32))


def test_abort_episode_discards_active_slot_without_fault(tmp_path):
    """Rework #3 RED: abort_episode(env_ids) is the explicit ACTIVE -> DONE
    discard for consumers that never step (the gate-audit loop resets the
    SAME slot for every configuration probe — with reset_at refusing ACTIVE
    slots it must close the previous probe episode explicitly). An aborted
    slot is DONE but NOT VOIDED (no fault verdict), is a legal reset_at
    target, and stays unsteppable. The current adapter has no abort:
    missing AttributeError."""
    adapter = _adapter2(tmp_path)
    specs = _specs()
    adapter.reset(specs, seeds=[1, 2])
    adapter.step(np.zeros((2, 4, 11), dtype=np.float32))  # both ACTIVE now
    adapter.abort_episode([0])
    assert not adapter._voided[0], "abort discards WITHOUT a fault verdict"
    # abort on a non-ACTIVE (now DONE) slot raises — same atomicity
    with pytest.raises(RuntimeError, match="ACTIVE"):
        adapter.abort_episode([0])
    # aborted slot is a legal reset_at target; slot 1 untouched
    obs, info = adapter.reset_at([0], [specs[0]], seeds=[99])
    assert info[0]["env_id"] == 0 and info[0]["world_uid"] == "env_a"
    assert adapter.current_episode_config(0)["seed"] == 99
    assert adapter.current_episode_config(1)["seed"] == 2
