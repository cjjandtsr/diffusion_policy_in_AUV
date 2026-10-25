"""Rework #4 (审核裁决 2026-08-09, 第四轮) RED suite: train-loop <-> adapter
slot lifecycle closure.

Three P1 blockers reopened by the ruling:
  P1-1  a reset-frame gate REJECT leaves the real adapter slot ACTIVE (the
        loop only ``continue``s) -> the next refill's reset_at hits the
        ACTIVE preflight RuntimeError (mock_uav_env.py had no ACTIVE guard,
        so tests never saw it);
  P1-2  AdmissionExhausted is never caught in the loop (the exception ends
        the batch instead of switching world/task via the sampler) AND
        reset_at mutates slots one-by-one (a mid-batch admission failure
        leaves a partial reset behind);
  P1-3  loop-level max_ep_len truncation only sets the local done flag —
        the adapter slot stays ACTIVE and the final transition is written
        with the env's raw truncated (False) value.

Every test in this file failed on the pre-fix tree (6b80587 / 9aec72a); the
per-test RED observation is documented in each docstring. The mocks must
mirror the real adapter's exact semantics (ACTIVE guard, two-phase atomic
reset_at, explicit reject_episode/mark_truncated transitions, read-only
slot_state/is_active) — mock/test gaps are how P1-1 hid from the suite.
"""

import json

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    AdmissionExhausted,
    EpisodeTaskSpec,
    UAVVectorEnvAdapter,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    EpisodeCollectionIncomplete,
    LoopConfig,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv, ScriptedTaskSampler
from multi_slot_mock import MultiSlotMockUAVEnv

ROLE_ORDER = (
    "anticipatory",
    "dense_recovery",
    "support_expansion",
    "yaw_lateral_recovery",
)


# ═══════════════════════════════════════════════════════════════
# shared helpers (mirror test_train_loop.py / test_real_adapter.py)
# ═══════════════════════════════════════════════════════════════


def _spec(world_uid: str, role: str) -> EpisodeTaskSpec:
    return EpisodeTaskSpec(
        world_uid=world_uid,
        obstacle_json="{}",
        role=role,
        split="rl_train",
        admission_id=f"{world_uid}__{role}",
        generator_version="mock",
        manifest_entry_hash=f"hash_{world_uid}",
    )


def _task_specs() -> list:
    """One spec per role — the shared EpisodeTaskSampler (R2.1) hard-fails
    on an empty role bucket, so the default set covers all 4 roles."""
    return [_spec(f"mock_{role}", role) for role in ROLE_ORDER]


def _critic(contract, policy):
    from diffusion_policy_3d.rl.dppo_uav.critic import (
        PolicyObsEncoder,
        ValueCritic,
    )

    return ValueCritic(contract=contract, encoder=PolicyObsEncoder(policy))


def _optimizers(policy, critic):
    import torch

    actor_opt = torch.optim.AdamW(policy.model.parameters(), lr=1e-4)
    critic_opt = torch.optim.AdamW(critic.value_head_parameters(), lr=1e-4)
    return actor_opt, critic_opt


def _cfg(tmp_path, mode: str, n_episodes: int = 2,
         max_ep_len: int = 12) -> LoopConfig:
    return LoopConfig(
        mode=mode,
        n_episodes=n_episodes,
        max_ep_len=max_ep_len,
        device="cpu",
        output_dir=str(tmp_path),
        eval_n_episodes=0,
    )


def _run(contract, policy, cfg, task_sampler, env, runtime):
    """Run one e2e batch over the given env (mock or multi-slot mock)."""
    critic = _critic(contract, policy)
    actor_opt, critic_opt = _optimizers(policy, critic)
    return run_e2e_batch(
        env=env,
        policy=policy,
        critic=critic,
        contract=contract,
        loop_config=cfg,
        trainer_config=TrainerConfig(device=cfg.device),
        task_specs=_task_specs(),
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        lora_info={},
        runtime=runtime,
        task_sampler=task_sampler,
    )


def _runtime() -> dict:
    return {
        "timestamp": "2026-08-09T00:00:00Z",
        "device": "cpu",
        "reward_version": "v1.3",
        "bc_checkpoint_hash": "bc0123",
        "manifest_hash": "mf456",
        "source_file_hashes": {},
        "project_git_commit": "n/a",
        "project_tree_hash": "n/a",
    }


# real-adapter world helpers (mirror test_real_adapter.py)

def _write_world(tmp_path, obstacles: list, name: str) -> str:
    path = tmp_path / f"{name}.obstacles.json"
    path.write_text(json.dumps({"world_id": name, "obstacles": obstacles}))
    return str(path)


def _sparse_world(tmp_path) -> str:
    """Two small cylinders — admission has plenty of free pairs."""
    return _write_world(tmp_path, [
        {"name": "c1", "shape": "cylinder", "x": 4.0, "y": 0.0,
         "radius": 1.2, "height": 4.0},
        {"name": "c2", "shape": "cylinder", "x": -6.0, "y": 5.0,
         "radius": 0.9, "height": 3.0},
    ], "test_world")


def _blocked_world(tmp_path) -> str:
    """Radius-30 cylinder covering the whole C2 box: every start/goal pair
    fails clearance + planner -> admission exhausts."""
    return _write_world(tmp_path, [
        {"name": "mega", "shape": "cylinder", "x": 0.0, "y": 0.0,
         "radius": 30.0, "height": 10.0},
    ], "blocked_world")


# ═══════════════════════════════════════════════════════════════
# P1-1: reset-frame REJECT must close the adapter slot explicitly
# ═══════════════════════════════════════════════════════════════

def test_loop_reset_reject_closes_slot_next_task_succeeds(tmp_path):
    """Scenario 1 (裁决 RED #1): a reset-frame REJECT_EPISODE must close the
    slot (explicit ACTIVE -> DONE) so the next sampler task can reset the
    SAME slot.

    RED on 9aec72a: the loop leaves the slot ACTIVE after the REJECT; the
    next refill's reset_at raises the ACTIVE preflight RuntimeError and the
    batch dies."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    sampler = ScriptedTaskSampler(
        [_spec("mock_reject", "dense_recovery"),
         _spec("mock_reach5", "anticipatory")],
        base_seed=10,
    )
    env = MultiSlotMockUAVEnv(
        contract, n_envs=1, episode_scripts={
            "mock_reject": {"reset_invalid": True},
        },
    )
    r1 = _run(
        contract, policy, _cfg(tmp_path, "no_update", n_episodes=1),
        sampler, env, _runtime(),
    )
    assert r1["completed_episodes"] == 1
    assert r1["rejected_episodes"] == 1
    assert sampler.calls == 2  # reject consumed one attempt, then success
    assert env.slot_state(0) == "DONE", (
        "the REJECTed slot must be explicitly closed (not left ACTIVE)"
    )
    # the SAME adapter runs a second batch without an ACTIVE hard-fail
    r2 = _run(
        contract, policy, _cfg(tmp_path, "no_update", n_episodes=1),
        sampler, env, _runtime(),
    )
    assert r2["completed_episodes"] == 1


# ═══════════════════════════════════════════════════════════════
# P1-2: AdmissionExhausted must switch the sampler task, never kill
#        the batch; multi-slot reset_at must be two-phase atomic
# ═══════════════════════════════════════════════════════════════

def test_loop_admission_exhausted_switches_task(tmp_path):
    """Scenario 2 (裁决 RED #2): an AdmissionExhausted reset consumes the
    attempt, is counted, leaves the slot reset-able, and the loop calls the
    sampler again — the next task (a different world) admits and the batch
    completes.

    RED on 9aec72a: the exception is uncaught in the loop and propagates
    out of run_e2e_batch (batch dies instead of switching task)."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    sampler = ScriptedTaskSampler(
        [_spec("world_exhaust", "dense_recovery"),
         _spec("world_ok", "anticipatory")],
        base_seed=20,
    )
    env = MultiSlotMockUAVEnv(
        contract, n_envs=2, episode_scripts={
            "world_exhaust": {"admission_exhausted": True},
        },
    )
    result = _run(
        contract, policy, _cfg(tmp_path, "no_update", n_episodes=1),
        sampler, env, _runtime(),
    )
    assert result["completed_episodes"] == 1
    assert result["admission_blocked_episodes"] == 1, (
        "the exhausted admission attempt must be counted, not swallowed"
    )
    assert sampler.calls == 2  # one blocked attempt + one admitted attempt
    # slot 1 was never touched (single-slot failure stays attributable)
    assert env.slot_state(1) == "INITIAL"


def test_multi_slot_zero_partial_mutation_on_admission_exhaust(tmp_path):
    """Scenario 3 (裁决 RED #3): a multi-slot reset_at where the LAST slot
    exhausts admission must leave ZERO partial resets — the first slot is
    still INITIAL (never mutated). reset_at admits/prepares ALL slots
    before ANY env mutation.

    RED on 9aec72a: slot 0 is mutated (ACTIVE, record set) before slot 1
    raises AdmissionExhausted — the assert below fails on the leftover."""
    contract = RLSystemContract()
    env = MultiSlotMockUAVEnv(
        contract, n_envs=2, episode_scripts={
            "world_exhaust": {"admission_exhausted": True},
        },
    )
    with pytest.raises(AdmissionExhausted):
        env.reset_at(
            [0, 1],
            [_spec("world_ok", "anticipatory"),
             _spec("world_exhaust", "dense_recovery")],
            [7, 8],
        )
    assert env._done[0], "slot 0 must stay INITIAL (zero partial mutation)"
    assert env._records[0] is None, "no record may land on an unapplied slot"
    assert env.slot_state(0) == "INITIAL"
    assert env.slot_state(1) == "INITIAL"
    # both slots remain fully reset-able
    obs, info = env.reset_at(
        [0, 1],
        [_spec("world_ok", "anticipatory"),
         _spec("world_ok2", "dense_recovery")],
        [7, 8],
    )
    assert env.is_active(0) and env.is_active(1)
    assert info[0]["world_uid"] == "world_ok"


def test_real_adapter_zero_partial_mutation_on_admission_exhaust(tmp_path):
    """Scenario 3 on the REAL adapter: two-phase reset_at — slot 1 exhausts
    (blocked world, rule 7) after slot 0 would have admitted; the adapter
    must raise before ANY env mutation.

    RED on 9aec72a: slot 0's env is reset and marked ACTIVE before slot 1's
    admission exhausts (partial reset left behind)."""
    adapter = UAVVectorEnvAdapter(
        RLSystemContract(),
        world_pool=[("test_world", _sparse_world(tmp_path)),
                    ("blocked_world", _blocked_world(tmp_path))],
        device="cpu",
        max_episode_steps=30,
        n_envs=2,
    )
    with pytest.raises(AdmissionExhausted):
        adapter.reset_at(
            [0, 1],
            [_spec("test_world", "anticipatory"),
             _spec("blocked_world", "dense_recovery")],
            [42, 43],
        )
    assert adapter._done[0], (
        "slot 0 must stay INITIAL — zero partial mutation on mid-batch"
        " admission failure"
    )
    assert adapter.slot_state(0) == "INITIAL"
    # both slots remain fully reset-able after the failed batch
    obs, info = adapter.reset_at(
        [0, 1],
        [_spec("test_world", "anticipatory"),
         _spec("test_world", "dense_recovery")],
        [42, 43],
    )
    assert adapter.is_active(0) and adapter.is_active(1)
    assert info[0]["world_uid"] == "test_world"


# ═══════════════════════════════════════════════════════════════
# P1-3: loop-level max_ep_len truncation must write truncated=True
#        AND close the adapter slot
# ═══════════════════════════════════════════════════════════════

def test_loop_max_ep_len_writes_truncated_and_closes_slot(tmp_path):
    """Scenario 4 (裁决 RED #4): when the LOOP truncates at max_ep_len (the
    env never ends), the final transition must be written with
    truncated=True and the adapter slot explicitly closed — the next
    episode's reset must not hit the ACTIVE hard-fail.

    RED on 9aec72a: the transition is written with the env's raw truncated
    (False) and the slot is left ACTIVE."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    sampler = ScriptedTaskSampler(
        [_spec("world_long", "anticipatory")], base_seed=30,
    )
    env = MultiSlotMockUAVEnv(
        contract, n_envs=1,
        reach_step=100, max_steps=100,  # env never ends on its own
    )
    result = _run(
        contract, policy,
        _cfg(tmp_path, "no_update", n_episodes=1, max_ep_len=2),
        sampler, env, _runtime(),
    )
    assert result["transitions"] == 2
    assert result["completed_episodes"] == 1
    assert result["buffer"].truncated[-1] is True, (
        "the loop-truncated transition must read truncated=True"
    )
    assert result["buffer"].terminated[-1] is False
    assert env.slot_state(0) == "DONE", (
        "the loop-truncated slot must be explicitly closed"
    )


def test_consecutive_batches_loop_truncation_no_active_leftover(tmp_path):
    """Scenario 5 (裁决 RED #5): TWO consecutive batches on the SAME adapter
    where batch 1 ends episodes by loop truncation (max_ep_len=2) — batch 2
    must start from a clean slot table. This is the user's repro shape
    (max_ep_len=2, env max_steps=100).

    RED on 9aec72a: batch 1 leaves the slot ACTIVE; batch 2's first
    reset_at raises the ACTIVE preflight RuntimeError."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    sampler = ScriptedTaskSampler(
        [_spec("world_long", "anticipatory"),
         _spec("world_long2", "dense_recovery")], base_seed=40,
    )
    env = MultiSlotMockUAVEnv(
        contract, n_envs=2,
        reach_step=100, max_steps=100,  # env never ends on its own
    )
    r1 = _run(
        contract, policy,
        _cfg(tmp_path, "no_update", n_episodes=2, max_ep_len=2),
        sampler, env, _runtime(),
    )
    assert r1["completed_episodes"] == 2
    # RED observable on 9aec72a: batch 1's loop-truncated slots stay ACTIVE
    # (the loop only set the local done flag) — no ACTIVE slot may survive.
    assert all(env._done[i] for i in range(env.n_envs)), (
        "an episode was left ACTIVE by loop truncation at collection stop"
    )
    assert all(env.slot_state(i) in ("DONE", "INITIAL")
               for i in range(env.n_envs))
    r2 = _run(
        contract, policy,
        _cfg(tmp_path, "no_update", n_episodes=2, max_ep_len=2),
        sampler, env, _runtime(),
    )
    assert r2["completed_episodes"] == 2
    assert all(not env.is_active(i) for i in range(env.n_envs)), (
        "no ACTIVE slot may survive batch 2"
    )


# ═══════════════════════════════════════════════════════════════
# admission-blocked diagnostics on EpisodeCollectionIncomplete
# ═══════════════════════════════════════════════════════════════

def test_attempt_limit_exhausted_raises_incomplete_with_admission_blocked(tmp_path):
    """Scenario 6 (裁决 RED #6): every task exhausts admission — the loop
    must keep catching AdmissionExhausted (counted), and when the attempt
    limit gates further starts it must raise EpisodeCollectionIncomplete
    (with the admission-blocked diagnostic), NEVER the raw
    AdmissionExhausted.

    RED on 9aec72a: the first reset_at's AdmissionExhausted propagates out
    of run_e2e_batch — no EpisodeCollectionIncomplete at all."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    sampler = ScriptedTaskSampler(
        [_spec("world_exhaust", "dense_recovery")], base_seed=50,
    )
    env = MultiSlotMockUAVEnv(
        contract, n_envs=1, episode_scripts={
            "world_exhaust": {"admission_exhausted": True},
        },
    )
    with pytest.raises(EpisodeCollectionIncomplete) as ei:
        _run(
            contract, policy, _cfg(tmp_path, "no_update", n_episodes=2),
            sampler, env, _runtime(),
        )
    limit = max(2 * 2, 4) + 4
    assert ei.value.attempt_limit == limit
    assert ei.value.admission_blocked_episodes == limit, (
        "every attempt must be counted as admission-blocked"
    )
    assert ei.value.completed_episodes == 0
    assert ei.value.attempted_episodes == 0  # nothing was ever admitted


# ═══════════════════════════════════════════════════════════════
# mock <-> real adapter semantic parity (mock_uav_env.py had no
# ACTIVE guard — that is how P1-1 hid from the suite)
# ═══════════════════════════════════════════════════════════════

def test_mock_single_slot_reset_at_rejects_active():
    """Mock parity: reset_at on an ACTIVE slot must raise (the single-slot
    mock had NO ACTIVE guard — silent success hid P1-1 from every loop
    test).

    RED on 9aec72a: the second reset_at silently succeeds."""
    contract = RLSystemContract()
    env = MockUAVEnv(contract)
    env.reset([_spec("world_ok", "anticipatory")], seeds=[1])
    with pytest.raises(RuntimeError):
        env.reset_at([0], [_spec("world_ok", "anticipatory")], [2])
    assert env.is_active(0)


def test_mock_reject_episode_and_mark_truncated_close_active():
    """Mock parity: reject_episode()/mark_truncated() are explicit ACTIVE ->
    DONE transitions (same boundary as the real adapter — raise on
    non-ACTIVE, never silently re-set).

    RED on 9aec72a: the methods do not exist (AttributeError)."""
    contract = RLSystemContract()
    env = MockUAVEnv(contract)
    env.reset([_spec("world_ok", "anticipatory")], seeds=[1])
    env.reject_episode([0])
    assert not env.is_active(0)
    assert env.slot_state(0) == "DONE"
    with pytest.raises(RuntimeError):
        env.reject_episode([0])  # already DONE — not ACTIVE
    env.reset_at([0], [_spec("world_ok", "anticipatory")], [2])
    env.mark_truncated([0])
    assert env.slot_state(0) == "DONE"
    env.reset_at([0], [_spec("world_ok", "anticipatory")], [3])
    env.mark_voided([0])
    assert env.slot_state(0) == "VOIDED"
    env.reset_at([0], [_spec("world_ok", "anticipatory")], [4])
    assert env.slot_state(0) == "ACTIVE"


def test_real_adapter_lifecycle_state_transitions(tmp_path):
    """Real adapter: the full slot_state()/is_active() lifecycle —
    INITIAL -> ACTIVE -> DONE (reject) / VOIDED (fault) / DONE
    (truncate), each explicit transition raising on non-ACTIVE, and
    reset_at never accepting ACTIVE.

    RED on 9aec72a: slot_state()/is_active()/reject_episode()/
    mark_truncated() do not exist (AttributeError)."""
    adapter = UAVVectorEnvAdapter(
        RLSystemContract(),
        world_pool=[("test_world", _sparse_world(tmp_path))],
        device="cpu",
        max_episode_steps=30,
    )
    assert adapter.slot_state(0) == "INITIAL"
    assert not adapter.is_active(0)
    adapter.reset([_spec("test_world", "anticipatory")], seeds=[42])
    assert adapter.slot_state(0) == "ACTIVE"
    assert adapter.is_active(0)
    adapter.reject_episode([0])
    assert adapter.slot_state(0) == "DONE"
    with pytest.raises(RuntimeError):
        adapter.reject_episode([0])  # DONE — not ACTIVE
    adapter.reset([_spec("test_world", "anticipatory")], seeds=[43])
    adapter.mark_truncated([0])
    assert adapter.slot_state(0) == "DONE"
    adapter.reset([_spec("test_world", "anticipatory")], seeds=[44])
    adapter.mark_voided([0])
    assert adapter.slot_state(0) == "VOIDED"
    with pytest.raises(RuntimeError):
        adapter.mark_truncated([0])  # VOIDED — not ACTIVE
    adapter.reset([_spec("test_world", "anticipatory")], seeds=[45])
    assert adapter.slot_state(0) == "ACTIVE"
