"""P7 (ASSUMPTION-9): paired-eval JSON persistence + eval PC validity gate.

Adjudication: "run_e2e_batch() 只返回内存中的 EvalArtifact，没有落盘
paired-eval JSON" and "eval rollout 需要自己的 point-cloud validity gate".

- Paired-eval JSON: one_update + eval writes `paired_eval_{env_steps}.json`
  (full §4.17 artifact dict) next to the checkpoint; the result carries
  `eval_json_path`.
- Eval gate: `run_paired_eval` runs its own PointCloudValidityGate per
  episode; any invalid reset/step frame voids that paired seed (no record)
  and is counted — the count must reach the caller so eval hard-fails
  instead of silently producing a PPO==BC outcome.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import OfflineDensity
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import freeze_paired_bank
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    LoopConfig,
    run_e2e_batch,
    run_paired_eval,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv
from multi_slot_mock import MultiSlotMockUAVEnv


def _contract() -> RLSystemContract:
    return RLSystemContract()


def _task_specs() -> list:
    """One spec per role — the shared EpisodeTaskSampler (R2.1) hard-fails
    on an empty role bucket."""
    return [
        EpisodeTaskSpec(
            world_uid=f"mock_{role}",
            obstacle_json="{}",
            role=role,
            split="rl_train",
            admission_id=f"mock__{role}",
            generator_version="mock",
            manifest_entry_hash=f"hash_{role}",
        )
        for role in ("anticipatory", "dense_recovery",
                     "support_expansion", "yaw_lateral_recovery")
    ]


def _runtime() -> dict:
    return {
        "timestamp": "2026-08-07T00:00:00Z",
        "device": "cpu",
        "reward_version": "v1.3",
        "bc_checkpoint_hash": "bc0123",
        "manifest_hash": "mf456",
        "source_file_hashes": {},
        "project_git_commit": "n/a",
        "project_tree_hash": "n/a",
        "world_assets_hash": "wa456",
    }


def _critic(contract: RLSystemContract, policy: FakePolicy) -> ValueCritic:
    return ValueCritic(contract=contract, encoder=PolicyObsEncoder(policy))


def _bank(contract: RLSystemContract, n: int = 3, env=None):
    """A FROZEN paired bank: freeze n specs on a clean single-slot
    mock, so every entry's episode_config_hash is deterministic — the
    eval-time replay of (spec, seed) reproduces the same hash and the
    freeze check passes. n=3 matches the default eval_n (one entry per
    capability role)."""
    return freeze_paired_bank(
        env or MockUAVEnv(contract),
        _task_specs()[:n],
        tuple(1001 + i for i in range(n)),
    )


def _cfg(tmp_path, eval_n: int = 3, **kwargs) -> LoopConfig:
    # 3 = one record per capability role (anticipatory/dense_recovery/
    # support_expansion in the spec set) — the capability metrics hard-fail
    # on any missing role (D1).
    base = dict(
        mode="one_update",
        n_episodes=2,
        max_ep_len=12,
        device="cpu",
        output_dir=str(tmp_path),
        eval_n_episodes=eval_n,
    )
    base.update(kwargs)
    return LoopConfig(**base)


def _run(tmp_path, env_kwargs: dict = None, **cfg_kwargs):
    contract = _contract()
    policy = FakePolicy(contract)
    cfg = _cfg(tmp_path, **cfg_kwargs)
    env = MockUAVEnv(contract, **(env_kwargs or {}))
    specs = _task_specs()
    # Paired eval uses a DEDICATED n_envs=1 eval env — never the training
    # env (explicit interface boundary) — and replays the FROZEN bank
    # one entry per eval episode, each with its own frozen seed. Capability
    # metrics need the offline density field.
    eval_env_kw = {}
    if cfg.eval_n_episodes > 0:
        eval_env = MockUAVEnv(contract)
        bank = freeze_paired_bank(
            eval_env, specs[:cfg.eval_n_episodes],
            tuple(1001 + i for i in range(cfg.eval_n_episodes)),
        )
        eval_env_kw = {
            "eval_env": eval_env, "paired_bank": bank,
            # bank v3 provenance: the artifact validator refuses an
            # empty bank_id, so the loop must bind one.
            "bank_id": "bank-p7",
            "offline_density": OfflineDensity(
                counts={(x, 0, 2): 5 for x in range(17)}
            ),
        }
    # Non-empty LoRA wiring whenever an eval artifact will be built: the
    # P0-1 gate rejects sha256(b"") (PPO==BC) at artifact validation, so
    # a mock with lora_info={} can no longer produce a valid artifact.
    # Plain-dict axes are the documented form for test doubles (the fake
    # model's layer names don't match the UNet target patterns).
    lora_info = {}
    if cfg.eval_n_episodes > 0:
        lora_info = {
            "lora_params": [torch.nn.Parameter(torch.zeros(2, 2))],
            "lora_rank": 8, "lora_alpha": 8.0, "n_adaptors": 1,
        }
    return run_e2e_batch(
        env=env,
        policy=policy,
        critic=_critic(contract, policy),
        contract=contract,
        loop_config=cfg,
        trainer_config=TrainerConfig(device=cfg.device),
        task_specs=specs,
        actor_optimizer=torch.optim.AdamW(policy.model.parameters(), lr=1e-4),
        critic_optimizer=torch.optim.AdamW(
            _critic(contract, policy).value_head_parameters(), lr=1e-4
        ),
        lora_info=lora_info,
        runtime=_runtime(),
        **eval_env_kw,
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
    )


def _paired(contract: RLSystemContract, cfg: LoopConfig):
    """Direct run_paired_eval call (bypasses the training loop) on a
    frozen bank."""
    policy = FakePolicy(contract)
    return run_paired_eval(
        MockUAVEnv(contract),
        policy,
        policy,
        contract,
        cfg,
        _bank(contract),
    bank_id="test-bank",
    policy_checkpoint_hash="ckpt-eval1",
    )


# ═══════════════════════════════════════════════════════════════════
# Eval entry preflight contract
# ═══════════════════════════════════════════════════════════════════

class _CountingMockUAVEnv(MockUAVEnv):
    """Records reset/step calls so a preflight failure can be proven to
    happen before ANY rollout contact (rework #6 P1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_calls = 0
        self.step_calls = 0

    def reset_at(self, env_ids, task_specs, seeds):
        self.reset_calls += 1
        return super().reset_at(env_ids, task_specs, seeds)

    def step(self, executed_actions):
        self.step_calls += 1
        return super().step(executed_actions)


def _run_manual(tmp_path, env=None, eval_env=None, mode="one_update",
                eval_n=1, offline_density=None):
    """Direct run_e2e_batch with explicit env/eval_env control (rework #6:
    the preflight contract must be testable with eval_env=None / same
    instance, which _run's auto-dedicated-eval_env hides). When an eval
    env is given, a frozen bank is created for it — so an explicit
    offline_density=None reaches the density preflight check."""
    contract = _contract()
    policy = FakePolicy(contract)
    cfg = LoopConfig(
        mode=mode, n_episodes=2, max_ep_len=12, device="cpu",
        output_dir=str(tmp_path), eval_n_episodes=eval_n,
    )
    env = env or _CountingMockUAVEnv(contract)
    kwargs = {"offline_density": offline_density}
    if eval_env is not None:
        kwargs["eval_env"] = eval_env
        kwargs["paired_bank"] = freeze_paired_bank(
            eval_env, _task_specs()[:eval_n],
            tuple(1001 + i for i in range(eval_n)),
        )
        kwargs["bank_id"] = "bank-p7"  # v3 provenance, required by validator
    # Non-empty LoRA wiring whenever an eval artifact will be built
    # (P0-1: empty LoRA == PPO==BC is rejected at artifact validation).
    # Plain-dict axes are the documented form for test doubles (the fake
    # model's layer names don't match the UNet target patterns).
    lora_info = {}
    if eval_env is not None:
        lora_info = {
            "lora_params": [torch.nn.Parameter(torch.zeros(2, 2))],
            "lora_rank": 8, "lora_alpha": 8.0, "n_adaptors": 1,
        }
    return run_e2e_batch(
        env=env,
        policy=policy,
        critic=_critic(contract, policy),
        contract=contract,
        loop_config=cfg,
        trainer_config=TrainerConfig(device=cfg.device),
        task_specs=_task_specs(),
        actor_optimizer=torch.optim.AdamW(policy.model.parameters(), lr=1e-4),
        critic_optimizer=torch.optim.AdamW(
            _critic(contract, policy).value_head_parameters(), lr=1e-4
        ),
        lora_info=lora_info,
        runtime=_runtime(),
        task_sampler=EpisodeTaskSampler(
            _task_specs(), DEFAULT_ROLE_TARGETS, base_seed=0
        ),
        **kwargs,
    )


def test_no_update_with_eval_request_hard_fails(tmp_path):
    """no_update + eval_n_episodes>0 must hard-fail at the entry — the old
    code nested the eval check inside the one_update branch and silently
    returned (no eval, no error) when mode was no_update (rework #6 P1)."""
    with pytest.raises(RuntimeError, match="one_update"):
        _run_manual(tmp_path, mode="no_update", eval_n=1)


def test_missing_eval_env_fails_before_rollout_update_and_checkpoint(tmp_path):
    """eval_n_episodes>0 without eval_env must hard-fail BEFORE any rollout,
    PPO update, or checkpoint write (rework #6 P1: the old check ran after
    CheckpointBuilder.save, leaving a failed-run-with-checkpoint artifact)."""
    env = _CountingMockUAVEnv(_contract())
    policy = FakePolicy(_contract())
    contract = _contract()
    actor_optimizer = torch.optim.AdamW(policy.model.parameters(), lr=1e-4)
    params_before = {
        k: v.clone() for k, v in policy.model.state_dict().items()
    }
    with pytest.raises(RuntimeError, match="eval_env"):
        run_e2e_batch(
            env=env,
            policy=policy,
            critic=_critic(contract, policy),
            contract=contract,
            loop_config=LoopConfig(
                mode="one_update", n_episodes=2, max_ep_len=12,
                device="cpu", output_dir=str(tmp_path), eval_n_episodes=1,
            ),
            trainer_config=TrainerConfig(device="cpu"),
            task_specs=_task_specs(),
            actor_optimizer=actor_optimizer,
            critic_optimizer=torch.optim.AdamW(
                _critic(contract, policy).value_head_parameters(), lr=1e-4
            ),
            lora_info={},
            runtime=_runtime(),
            task_sampler=EpisodeTaskSampler(
                _task_specs(), DEFAULT_ROLE_TARGETS, base_seed=0
            ),
        )
    assert env.reset_calls == 0 and env.step_calls == 0, (
        "preflight failure must precede any env reset/step"
    )
    params_after = policy.model.state_dict()
    assert all(
        torch.equal(params_before[k], params_after[k]) for k in params_before
    ), "policy parameters must be unchanged"
    assert not actor_optimizer.state_dict()["state"], (
        "optimizer must not have stepped"
    )
    assert list(tmp_path.iterdir()) == [], (
        "output dir must be empty — no checkpoint/artifact written"
    )


def test_negative_eval_n_episodes_rejected():
    """eval_n_episodes < 0 must be rejected at LoopConfig construction —
    a negative count silently disabled evaluation in the old code (rework
    #6 P2)."""
    with pytest.raises(AssertionError):
        LoopConfig(
            mode="one_update", n_episodes=2, max_ep_len=12, device="cpu",
            output_dir="/tmp/none", eval_n_episodes=-1,
        )


def test_training_env_cannot_be_reused_as_eval_env(tmp_path):
    """eval_env must be a DEDICATED instance — passing the training env
    itself (even n_envs=1) hard-fails at the entry (rework #6: the
    dedicated-env requirement was only a calling convention)."""
    env = MockUAVEnv(_contract())
    with pytest.raises(RuntimeError, match="dedicated"):
        _run_manual(tmp_path, env=env, eval_env=env)


def test_eval_without_offline_density_hard_fails(tmp_path):
    """eval_n_episodes>0 without the offline density field must hard-fail
    at the entry — capability metrics (support expansion, dense recovery)
    are not computable without it; the eval must never run into a silently
    empty capability block."""
    with pytest.raises(RuntimeError, match="offline density"):
        _run_manual(
            tmp_path, eval_env=MockUAVEnv(_contract()),
            eval_n=1, offline_density=None,
        )


# ═══════════════════════════════════════════════════════════════════
# paired-eval JSON persistence
# ═══════════════════════════════════════════════════════════════════

def test_eval_json_persisted_roundtrip(tmp_path):
    """The paired-eval artifact must land on disk and round-trip exactly."""
    result = _run(tmp_path)
    path = result["eval_json_path"]
    assert path is not None
    assert Path(path).exists()
    # utf-8 explicit: read_text() defaults to the locale encoding (ascii
    # under LANG=C) — same robustness class as the P5 gate fix.
    disk = json.loads(Path(path).read_text(encoding="utf-8"))
    assert disk == result["eval_artifact"].to_dict(), (
        "persisted JSON must equal the in-memory artifact dict"
    )
    assert all("bc" in ep and "rl" in ep for ep in disk["episodes"])
    assert all("trace" in ep and ep["trace"]["bc"] and ep["trace"]["rl"]
               for ep in disk["episodes"]), (
        "the raw per-step trace must ride the artifact"
    )
    # the UNIQUE canonical filename (counters + checkpoint sha12).
    assert Path(path).name.startswith("paired_eval_envstep_"), Path(path).name
    assert Path(path).parent == Path(tmp_path)


def test_eval_json_absent_when_eval_disabled(tmp_path):
    """eval_n_episodes=0 -> no JSON, no eval_json_path."""
    result = _run(tmp_path, eval_n_episodes=0)
    assert result["eval_json_path"] is None
    assert not list(tmp_path.glob("paired_eval_*.json"))


# ═══════════════════════════════════════════════════════════════════
# eval PC validity gate
# ═══════════════════════════════════════════════════════════════════

def test_run_paired_eval_gates_invalid_reset_frames():
    """Invalid reset frame voids the paired seed (no record), counted.

    rework #4 (审核裁决 PC-VALID): the voided episode must leave the slot
    CLOSED (reject_episode, DONE) — the next seed's reset_at would
    otherwise hard-fail on the ACTIVE guard.
    """
    contract = _contract()
    cfg = _cfg(Path("/tmp/none"))
    policy = FakePolicy(contract)
    bank = _bank(contract)
    env = MockUAVEnv(contract, reset_invalid=True)
    records, invalid = run_paired_eval(
        env, policy, policy, contract, cfg, bank,
    bank_id="test-bank",
    policy_checkpoint_hash="ckpt-eval1",
    )
    assert invalid == cfg.eval_n_episodes
    assert records == []
    assert env.slot_state(0) == "DONE", (
        "invalid-reset slot must be closed (reject_episode), not ACTIVE"
    )


def test_run_paired_eval_gates_invalid_step_frames():
    """First invalid frame during the rollout voids the paired seed.

    rework #4 (审核裁决 PC-VALID): the faulted rollout must leave the slot
    CLOSED (mark_voided, VOIDED) — mirroring the train loop's
    SAFE_END_EPISODE closure, not an ACTIVE leak.
    """
    contract = _contract()
    cfg = _cfg(Path("/tmp/none"))
    policy = FakePolicy(contract)
    bank = _bank(contract)
    env = MockUAVEnv(contract, invalid_episode_steps=[1])
    records, invalid = run_paired_eval(
        env, policy, policy, contract, cfg, bank,
    bank_id="test-bank",
    policy_checkpoint_hash="ckpt-eval1",
    )
    assert invalid == cfg.eval_n_episodes
    assert records == []
    assert env.slot_state(0) == "VOIDED", (
        "faulted-step slot must be closed (mark_voided), not ACTIVE"
    )


def test_run_paired_eval_valid_passthrough():
    """Valid frames -> one record per bank entry, zero invalid
    (regression). Each record is bound to BOTH frozen hashes — the
    manifest SOURCE hash and the sampled episode_config_hash — the eval
    replays the bank exactly, one entry per episode."""
    contract = _contract()
    cfg = _cfg(Path("/tmp/none"))
    records, invalid = _paired(contract, cfg)
    assert invalid == 0
    assert len(records) == cfg.eval_n_episodes
    bank = _bank(contract)
    for i, rec in enumerate(records):
        ec = rec.episode_config
        assert ec["manifest_entry_hash"] == bank[i].manifest_entry_hash
        assert ec["episode_config_hash"] == bank[i].episode_config_hash


def test_run_paired_eval_hard_fails_on_frozen_bank_violation():
    """A bank entry whose frozen state no longer matches the runtime
    episode — bank/manifest/world drift — must hard-fail (RuntimeError),
    never produce a silently-different record. Both drift surfaces are
    covered: the episode_config_hash and the world/role identity."""
    contract = _contract()
    cfg = _cfg(Path("/tmp/none"))
    policy = FakePolicy(contract)
    bank = _bank(contract)
    hash_drift = [dataclasses.replace(bank[0], episode_config_hash="0" * 64)]
    with pytest.raises(RuntimeError, match="frozen paired bank violated"):
        run_paired_eval(
            MockUAVEnv(contract), policy, policy, contract, cfg, hash_drift,
        bank_id="test-bank",
        policy_checkpoint_hash="ckpt-eval1",
        )
    id_drift = [dataclasses.replace(bank[0], world_uid="some_other_world")]
    with pytest.raises(RuntimeError, match="frozen paired bank violated"):
        run_paired_eval(
            MockUAVEnv(contract), policy, policy, contract, cfg, id_drift,
        bank_id="test-bank",
        policy_checkpoint_hash="ckpt-eval1",
        )


def test_paired_eval_trace_reads_frozen_geometry():
    """The per-step trace must carry REAL state values from the frozen
    geometry — the mock's s9 mirrors the real env's [pos(3), goal_body(3),
    dist, sin_yaw, cos_yaw] layout (kinematic_uav_env_v2._build_state), so
    goal_dist and yaw are computable from the bank entry, not placeholder
    zeros (the old mock put 'yaw cos' at index 3, which is goal_body[0] in
    the real layout)."""
    contract = _contract()
    cfg = _cfg(Path("/tmp/none"))
    policy = FakePolicy(contract)
    bank = _bank(contract)
    records, invalid = run_paired_eval(
        MockUAVEnv(contract), policy, policy, contract, cfg, bank,
    bank_id="test-bank",
    policy_checkpoint_hash="ckpt-eval1",
    )
    assert invalid == 0
    for rec, entry in zip(records, bank):
        for side in ("bc", "rl"):
            for step in rec.trace[side]:
                pos = np.asarray(step["pos"], dtype=np.float32)
                # mock kinematics: flies along +x at the frozen start_yaw
                exp_pos = np.array(
                    [0.01 * step["step"], 0.0, 0.0], dtype=np.float32
                )
                assert np.allclose(pos, exp_pos), (side, step["step"], pos)
                exp_dist = float(np.linalg.norm(
                    np.asarray(entry.goal, dtype=np.float32) - exp_pos
                ))
                assert abs(step["goal_dist"] - exp_dist) < 1e-5, (
                    side, step["step"], step["goal_dist"], exp_dist
                )
                assert abs(step["yaw"] - entry.start_yaw) < 1e-5
    # two different entries -> two different distance series (the trace
    # reflects the FROZEN entry geometry, not constants)
    d0 = [s["goal_dist"] for s in records[0].trace["bc"]]
    d1 = [s["goal_dist"] for s in records[1].trace["bc"]]
    assert d0 != d1


# ═══════════════════════════════════════════════════════════════════
# Paired-eval lifecycle closure
# ═══════════════════════════════════════════════════════════════════

def test_paired_eval_loop_limit_closes_slot_and_records_rollout_cut():
    """P1-1 (第五轮裁决): the eval loop can exit at max_ep_len WITHOUT the
    env emitting terminated/truncated. rework #5 must close the slot
    explicitly (mark_truncated) — and BOTH the BC and RL rollouts must
    complete (the RL reset used to hard-fail on the ACTIVE slot left by
    the BC rollout).

    R6-E §11.6.4 (F-10): a loop-level cut is NOT an env termination —
    the record must be a rollout_cut (timeout only when
    done_reason == "timeout", and the env never said timeout here). The
    pre-F-10 code collapsed the cut into "timeout", inflating the
    timeout rate.

    Repro: env terminates at max_steps=100, loop max_ep_len=2, point
    clouds always valid — the loop stops at steps=2 with terminated=False
    and truncated=False, so without the fix slot 0 stays ACTIVE after the
    BC episode and the RL episode's reset_at raises.
    """
    contract = _contract()
    cfg = _cfg(Path("/tmp/none"), max_ep_len=2)
    policy = FakePolicy(contract)
    bank = _bank(contract)
    eval_env = MockUAVEnv(contract, max_steps=100)
    records, invalid = run_paired_eval(
        eval_env, policy, policy, contract, cfg, bank,
        bank_id="test-bank",
        policy_checkpoint_hash="ckpt-eval1",
    )
    assert invalid == 0
    assert len(records) == cfg.eval_n_episodes
    for rec in records:
        assert rec.bc["terminal_category"] == "rollout_cut", (
            "loop-limit BC exit is a rollout_cut, never an env category"
        )
        assert rec.rl["terminal_category"] == "rollout_cut", (
            "loop-limit RL exit is a rollout_cut, never an env category"
        )
        assert rec.bc["timeout"] is False, (
            "loop-limit BC exit must NOT be a timeout — the env never "
            "signaled done_reason 'timeout'"
        )
        assert rec.bc["terminated"] is False
        assert rec.bc["truncated"] is False
    assert eval_env.slot_state(0) == "DONE", (
        "loop-limit exit must close the slot (mark_truncated), not ACTIVE"
    )


def test_mock_abort_episode_boundary_mirrors_real_adapter():
    """ the mock's abort_episode mirror (added for freeze_paired_bank)
    raises on a non-ACTIVE slot — same boundary as the real adapter — and
    closes an ACTIVE probe to DONE without a fault verdict."""
    env = MockUAVEnv(_contract())
    with pytest.raises(RuntimeError, match="ACTIVE"):
        env.abort_episode([0])
    env.reset([_task_specs()[0]], [1001])
    env.abort_episode([0])
    assert env.slot_state(0) == "DONE"
    assert not env._voided, "abort is a discard, never a fault verdict"


def test_paired_eval_uses_dedicated_single_slot_eval_env():
    """P1-2 (第五轮裁决): paired eval requires a DEDICATED n_envs=1 eval
    env. The multi-slot training env cannot be implicitly reused for
    single-episode eval — its reset() requires len(task_specs) == n_envs
    and its step() requires (n_envs, exec_steps, 11) actions, so
    n_envs > 1 fails at the interface. rework #5 must hard-fail on
    n_envs != 1 before any rollout and accept a dedicated single-slot env.
    """
    contract = _contract()
    cfg = _cfg(Path("/tmp/none"))
    policy = FakePolicy(contract)
    bank = _bank(contract)
    train_env = MultiSlotMockUAVEnv(contract, n_envs=4)
    with pytest.raises(RuntimeError, match="n_envs"):
        run_paired_eval(
            train_env, policy, policy, contract, cfg, bank,
            bank_id="test-bank",
            policy_checkpoint_hash="ckpt-eval1",
        )
    # dedicated single-slot eval env is accepted and completes
    eval_env = MockUAVEnv(contract)
    records, invalid = run_paired_eval(
        eval_env, policy, policy, contract, cfg, bank,
        bank_id="test-bank",
        policy_checkpoint_hash="ckpt-eval1",
    )
    assert invalid == 0
    assert len(records) == cfg.eval_n_episodes
    assert eval_env.slot_state(0) == "DONE"


# ═══════════════════════════════════════════════════════════════════
# Gate B R6-EVAL-1 REWORK: policy_checkpoint_hash 生产接线
# ═══════════════════════════════════════════════════════════════════


def test_eval1_records_carry_policy_checkpoint_hash():
    """§11.5.1 生产接线（Gate B R6-EVAL-1）: run_paired_eval 必须把
    policy_checkpoint_hash 镜像到每条记录 —— 纯列表统计层的 dedup 靠
    它区分策略身份（18/24 反例）；此前记录默认空串，生产记录喂 dedup
    层必 ValueError。"""
    contract = _contract()
    records, invalid = _paired(contract, _cfg(Path("/tmp/none")))
    assert invalid == 0
    assert records
    for rec in records:
        assert rec.policy_checkpoint_hash == "ckpt-eval1"


def test_eval1_empty_policy_checkpoint_hash_hard_fails():
    """§11.5.1: 空 policy_checkpoint_hash 是 silent provenance break
    （dedup 层会在更下游才硬失败）—— 与 bank_id 同级别的入口硬约束。"""
    contract = _contract()
    cfg = _cfg(Path("/tmp/none"))
    policy = FakePolicy(contract)
    with pytest.raises(ValueError, match="policy_checkpoint_hash"):
        run_paired_eval(
            MockUAVEnv(contract), policy, policy, contract, cfg,
            _bank(contract),
            bank_id="test-bank",
            policy_checkpoint_hash="",
        )
