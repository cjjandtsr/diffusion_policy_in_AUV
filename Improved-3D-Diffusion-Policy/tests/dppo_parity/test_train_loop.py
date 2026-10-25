"""T5 closure: train_loop e2e on CPU with the mock env (no-update/one-update).

Closes the remaining entry-point gaps: rollout sampling -> buffer -> GAE ->
ppo_update -> checkpoint save -> paired eval -> §4.17 artifact with the
runtime contract hash (ASSUMPTION-8 item 6 call site).
"""

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    SCHEMA_VERSION,
    CheckpointBuilder,
    checkpoint_filename,
    contract_hash_of,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    CAPABILITY_METRIC_KEYS,
    GROUP_OF_ROLE,
    OfflineDensity,
)
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    freeze_paired_bank,
    validate_eval_artifact,
)
from diffusion_policy_3d.rl.dppo_uav.reward_norm import (
    REWARD_SCALE_STATE_VERSION,
    RewardScaleConfig,
    validate_reward_scale_state,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    EpisodeCollectionIncomplete,
    LoopConfig,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig
from diffusion_policy_3d.rl.dppo_uav.training_schedules import (
    SCHEDULER_STATE_VERSION,
    LRSchedulerConfig,
    validate_scheduler_state,
)

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv, ScriptedTaskSampler
from multi_slot_mock import MultiSlotMockUAVEnv

ROLE_ORDER = (
    "anticipatory",
    "dense_recovery",
    "support_expansion",
    "yaw_lateral_recovery",
)


class _RecordingEnv:
    """Records every reset(reset_at) call for sampler-wiring asserts."""

    def __init__(self, env: MockUAVEnv) -> None:
        self.env = env
        self.resets = []  # (world_uids, seeds)

    n_envs = 1  # Vector contract — the loop drives reset_at

    def reset(self, task_specs, seeds=None):
        self.resets.append(
            ([s.world_uid for s in task_specs], list(seeds) if seeds is not None else None)
        )
        return self.env.reset(task_specs, seeds)

    def reset_at(self, env_ids, task_specs, seeds):
        self.resets.append(
            ([s.world_uid for s in task_specs], list(seeds))
        )
        return self.env.reset_at(env_ids, task_specs, seeds)

    def step(self, executed_actions):
        return self.env.step(executed_actions)

    def step_at(self, env_ids, executed_actions):
        return self.env.step_at(env_ids, executed_actions)

    # The loop now closes slots explicitly — the recording wrapper
    # forwards the lifecycle calls to the inner env.
    def reject_episode(self, env_ids):
        return self.env.reject_episode(env_ids)

    def mark_truncated(self, env_ids):
        return self.env.mark_truncated(env_ids)

    def mark_voided(self, env_ids):
        return self.env.mark_voided(env_ids)

    @property
    def cam_far(self):
        return self.env.cam_far

    def current_episode_config(self):
        return self.env.current_episode_config()


METRIC_KEYS = {
    "actor_grad_norm",
    "critic_grad_norm",
    "parameter_update_norm",
    "per_denoise_approx_kl",
    "per_denoise_clipfrac",
    "ratio_mean",
    "ratio_std",
    "ratio_q05",
    "ratio_q50",
    "ratio_q95",
    "action_variance_by_dim",
    "action_variance_space",
    "policy_action_diversity",
    "success_conditioned_advantage",
    "collision_conditioned_advantage",
    "role_conditioned_advantage",
    "executed_vs_unexecuted_grad_norm",
    "explained_variance",
    "policy_loss_mean",
    "value_loss_mean",
    "reward_dist",
    "return_dist",
    "advantage_dist",
    "approx_kl",
    "clipfrac",
    "early_stopped",
    "update_batches",
    "role_proportion_report",
    "outcome_counts",   # R6-C §11.4.2 per-batch outcome counters
    "advantages_hash",  # corrected-GAE resume-parity evidence
    "returns_hash",     # corrected-GAE resume-parity evidence
    "next_value_batch_size",  # R7-4: resolved chunk the fill used
    "next_value_provenance",  # R7-4: fill provenance record (dict)
    # R6-F §11.7.2: whether THIS update stepped the actor optimizer (the
    # warmup-window predicate) — the train_loop uses it to decide the
    # actor scheduler step, and the metrics record it per batch
    "actor_stepped",
    # R6-F §11.7.2: per-batch scheduler/scaler observability — LR before/
    # after the update and the resulting step counts (always recorded,
    # none branches keep LR constant while step counts advance)
    "actor_lr_before",
    "actor_lr_after",
    "critic_lr_before",
    "critic_lr_after",
    "actor_scheduler_step_count",
    "critic_scheduler_step_count",
    "reward_scaler_step_count",
    "randperm_hash",  # R7-6 Step 6: minibatch-index determinism evidence
    "grads_hash",     # R7-6 Step 6: actor+critic gradient determinism evidence
}


def _contract() -> RLSystemContract:
    return RLSystemContract()


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
    on an empty role bucket, so the default spec set covers all 4 roles."""
    return [_spec(f"mock_{role}", role) for role in ROLE_ORDER]


def _runtime() -> dict:
    return {
        "timestamp": "2026-08-06T00:00:00Z",
        "device": "cpu",
        "reward_version": "v1.3",
        "bc_checkpoint_hash": "bc0123",
        "manifest_hash": "mf456",
        "world_assets_hash": "world456",
        "source_file_hashes": {},
        "project_git_commit": "n/a",
        "project_tree_hash": "n/a",
    }


def _critic(contract: RLSystemContract, policy: FakePolicy) -> ValueCritic:
    return ValueCritic(contract=contract, encoder=PolicyObsEncoder(policy))


def _optimizers(policy: FakePolicy, critic: ValueCritic):
    actor_opt = torch.optim.AdamW(policy.model.parameters(), lr=1e-4)
    critic_opt = torch.optim.AdamW(
        critic.value_head_parameters(), lr=1e-4
    )  # value head only, never the shared encoder
    return actor_opt, critic_opt


def _run(
    contract: RLSystemContract,
    policy: FakePolicy,
    critic: ValueCritic,
    cfg: LoopConfig,
    task_specs: list,
    runtime: dict,
    env_kwargs: dict = None,
    task_sampler: object = None,
    env: object = None,
    resume_from: str = None,
):
    """Run one e2e batch. Defaults to the REAL shared EpisodeTaskSampler
    (R2.1); tests with a fixed task sequence pass a ScriptedTaskSampler spy
    and tests recording reset wiring pass a _RecordingEnv."""
    env = env or MockUAVEnv(contract, **(env_kwargs or {}))
    actor_opt, critic_opt = _optimizers(policy, critic)
    # Paired eval uses a DEDICATED n_envs=1 eval env — never the training
    # env (explicit interface boundary) — replaying the FROZEN bank
    # one entry per eval episode with its own frozen seed. Capability
    # metrics need the offline density field: eval_n>0 without it
    # hard-fails inside run_e2e_batch.
    eval_env_kw = {}
    if cfg.eval_n_episodes > 0:
        eval_env = MockUAVEnv(contract)
        bank = freeze_paired_bank(
            eval_env, task_specs[:cfg.eval_n_episodes],
            tuple(1001 + i for i in range(cfg.eval_n_episodes)),
        )
        eval_env_kw = {
            "eval_env": eval_env, "paired_bank": bank,
            "bank_id": "bank-train_loop",  # v3 provenance (validator requires)
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
        critic=critic,
        contract=contract,
        loop_config=cfg,
        trainer_config=TrainerConfig(device=cfg.device),
        task_specs=task_specs,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        lora_info=lora_info,
        runtime=runtime,
        resume_from=resume_from,
        **eval_env_kw,
        task_sampler=task_sampler or EpisodeTaskSampler(
            task_specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
    )


def _cfg(tmp_path, mode: str, n_episodes: int = 2, eval_n: int = 0,
         lr_scheduler: LRSchedulerConfig = None,
         reward_scale: RewardScaleConfig = None) -> LoopConfig:
    return LoopConfig(
        mode=mode,
        n_episodes=n_episodes,
        max_ep_len=12,
        device="cpu",
        output_dir=str(tmp_path),
        eval_n_episodes=eval_n,
        lr_scheduler=lr_scheduler or LRSchedulerConfig(method="none"),
        reward_scale=reward_scale or RewardScaleConfig(method="none"),
    )


def test_contract_hash_deterministic_and_differs():
    c1 = RLSystemContract()
    h1 = contract_hash_of(c1)
    assert h1 == contract_hash_of(c1)
    assert len(h1) == 64
    c2 = RLSystemContract(exec_steps=2)  # valid per contract invariants
    assert h1 != contract_hash_of(c2)


def test_no_update_collects_transitions_and_gate_summary(tmp_path):
    contract = _contract()
    policy = FakePolicy(contract)
    result = _run(
        contract, policy, _critic(contract, policy), _cfg(tmp_path, "no_update"),
        _task_specs(), _runtime(),
    )
    assert result["transitions"] == 10  # 2 episodes x 5 steps (reach at 5)
    assert result["metrics"] is None
    assert result["checkpoint_path"] is None
    assert result["eval_artifact"] is None
    assert result["completed_episodes"] == 2
    assert result["env_steps"] == 10
    assert result["rejected_episodes"] == 0
    gate = result["gate_summary"]
    assert gate["phase"] == "formal"
    assert gate["invalid_count"] == 0
    assert gate["passes"] is True
    assert result["lora_checksum"] and len(result["lora_checksum"]) == 64
    # both episodes ran to reach (5 transitions each, steps 0..4)
    steps_by_ep = {}
    for ep, st in zip(
        result["buffer"].episode_ids, result["buffer"].episode_steps
    ):
        steps_by_ep.setdefault(ep, []).append(st)
    assert [v for v in steps_by_ep.values()] == [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    # nothing saved in no-update mode
    assert not list(tmp_path.glob("*.pt"))


def test_non_multiple_budget_completes_exactly_no_hard_fail(tmp_path):
    """P2-8 (审核裁决, rework #3): the n_episodes % n_envs != 0 hard-fail
    is REPLACED by the precise refill budget — a partial final batch
    (n_envs=2, n_episodes=3) completes exactly 3 episodes with attempted
    == completed, voided == 0 and nothing left active at return. The old
    hard-fail (first version) and the old over-refill + silent tail drop
    (rework #2) are both gone."""
    contract = _contract()
    policy = FakePolicy(contract)
    specs = _task_specs()
    sampler = ScriptedTaskSampler(specs, base_seed=100)
    env = MultiSlotMockUAVEnv(
        contract, n_envs=2, episode_scripts=_MULTI_SCRIPTS,
    )
    result = _run(
        contract, policy, _critic(contract, policy),
        _cfg(tmp_path, "no_update", n_episodes=3),
        specs, _runtime(), env=env, task_sampler=sampler,
    )
    assert result["completed_episodes"] == 3
    rs = result["role_stats"]
    attempted = sum(rs["attempted_episode_counts"].values())
    voided = sum(rs["voided_episode_counts"].values())
    assert attempted == 3, f"exact budget: attempted {attempted} != 3"
    assert voided == 0
    assert all(env._done[i] for i in range(env.n_envs)), (
        "an episode was left active at collection stop"
    )


def test_first_fault_safe_ends_episode(tmp_path):
    """Rule 3: first fault of an episode -> SAFE_END, episode VOIDED — the
    faulting transition is dropped but the episode never counts as completed
    (R2.2 truthful accounting)."""
    contract = _contract()
    policy = FakePolicy(contract)
    specs = _task_specs()
    sampler = ScriptedTaskSampler(
        [_spec("mock_void", "support_expansion"),
         _spec("mock_reach5", "anticipatory")],
        base_seed=1000,
    )
    result = _run(
        contract, policy, _critic(contract, policy),
        _cfg(tmp_path, "no_update", n_episodes=1),
        specs, _runtime(),
        env_kwargs={"episode_scripts": {
            "mock_void": {"invalid_episode_steps": [2]},
        }},
        task_sampler=sampler,
    )
    assert result["completed_episodes"] == 1  # the reach-5 episode
    assert result["voided_episodes"] == 1  # the faulting episode
    assert result["transitions"] == 7  # voided: steps 0-1; completed: 0-4
    assert result["routing"]["SAFE_END_EPISODE"] == 1
    assert result["gate_summary"]["invalid_count"] == 1
    steps_by_ep = {}
    for ep, st in zip(
        result["buffer"].episode_ids, result["buffer"].episode_steps
    ):
        steps_by_ep.setdefault(ep, []).append(st)
    assert sorted([v for v in steps_by_ep.values()], key=len) == [
        [0, 1], [0, 1, 2, 3, 4],
    ], "fault at step 2 drops that transition and ends the episode (steps 0-1)"
    assert sampler.calls == 2  # the voided attempt consumed a sampler call
    assert sampler.completed == 1  # only the completed episode is marked
    rs = result["role_stats"]
    assert rs["completed_episode_counts"]["support_expansion"] == 0
    assert rs["voided_episode_counts"]["support_expansion"] == 1


def test_reset_phase_invalid_rejects_episode(tmp_path):
    """Rule 1: invalid frame at reset -> REJECT_EPISODE, episode skipped —
    the sampler advances and a later attempt completes the batch."""
    contract = _contract()
    policy = FakePolicy(contract)
    specs = _task_specs()
    sampler = ScriptedTaskSampler(
        [_spec("mock_reject", "dense_recovery"),
         _spec("mock_reach5", "anticipatory")],
        base_seed=10,
    )
    result = _run(
        contract, policy, _critic(contract, policy),
        _cfg(tmp_path, "no_update", n_episodes=1),
        specs, _runtime(),
        env_kwargs={"episode_scripts": {
            "mock_reject": {"reset_invalid": True},
        }},
        task_sampler=sampler,
    )
    assert result["transitions"] == 5
    assert result["completed_episodes"] == 1
    assert result["rejected_episodes"] == 1
    # R7-5 window gate (a942f32): invalid_count is a FRAME count — the
    # window validator re-validates every frame of the latest obs_steps
    # window through the gate (pointcloud_gate.py:409-419), so ONE
    # rejected reset window contributes obs_steps invalid frames
    # (frame-CONTENT validity is the gate's rolling-statistic unit).
    assert (
        result["gate_summary"]["invalid_count"] == contract.obs_steps
    ), (
        f"one rejected reset window must count exactly obs_steps="
        f"{contract.obs_steps} invalid frames, got "
        f"{result['gate_summary']['invalid_count']}"
    )
    assert result["routing"]["REJECT_EPISODE"] == 1
    assert sampler.calls == 2
    assert sampler.completed == 1


def test_loop_consumes_sampler_per_attempt(tmp_path):
    """R2.2 Step 1: run_e2e_batch drives the shared sampler — one
    next_attempt() per episode attempt; seeds and episode ids come from the
    sample, never from the old task_specs[(attempt-1)%len] / seeds=[attempt]
    pattern (spy seeds 1000/1001 are unproducible by seeds=[attempt])."""
    contract = _contract()
    policy = FakePolicy(contract)
    specs = _task_specs()
    sampler = ScriptedTaskSampler(specs, base_seed=1000)
    env = _RecordingEnv(MockUAVEnv(contract))
    result = _run(
        contract, policy, _critic(contract, policy), _cfg(tmp_path, "no_update"),
        specs, _runtime(), task_sampler=sampler, env=env,
    )
    assert sampler.calls == 2  # one next_attempt per episode attempt
    assert sampler.completed == 2
    assert [s[0] for _, s in env.resets] == [1000, 1001]
    # P2-9: episode ids carry the world:role:episode_seed format (the
    # adapter's EpisodeConfigRecord id) — NOT the global episode index.
    assert sorted(set(result["buffer"].episode_ids)) == [
        "mock_anticipatory:anticipatory:000001000",
        "mock_dense_recovery:dense_recovery:000001001",
    ]
    assert result["completed_episodes"] == 2


def test_two_consecutive_loops_continue_sampler_sequence(tmp_path):
    """R2.2 Step 1: two run_e2e_batch calls sharing ONE sampler — the
    second batch's seeds and episode ids continue (no per-batch restart at
    1, SEED-RESUME in the live loop)."""
    contract = _contract()
    policy = FakePolicy(contract)
    specs = _task_specs()
    sampler = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=5)
    env1 = _RecordingEnv(MockUAVEnv(contract))
    env2 = _RecordingEnv(MockUAVEnv(contract))
    cfg = _cfg(tmp_path, "no_update")
    r1 = _run(contract, policy, _critic(contract, policy), cfg, specs,
              _runtime(), task_sampler=sampler, env=env1)
    r2 = _run(contract, policy, _critic(contract, policy), cfg, specs,
              _runtime(), task_sampler=sampler, env=env2)
    seeds1 = [s[0] for _, s in env1.resets]
    seeds2 = [s[0] for _, s in env2.resets]
    assert seeds1 == [5, 6]
    assert seeds2 == [7, 8]  # strict continuation across batches
    eps1 = set(r1["buffer"].episode_ids)
    eps2 = set(r2["buffer"].episode_ids)
    assert len(eps1) == 2 and len(eps2) == 2
    assert eps1.isdisjoint(eps2)  # episode ids never repeat across batches
    idx1 = [int(ep.split(":")[-1]) for ep in eps1]
    idx2 = [int(ep.split(":")[-1]) for ep in eps2]
    assert min(idx2) == max(idx1) + 1  # global index continues
    known_worlds = {s.world_uid for s in specs}
    known_roles = {s.role for s in specs}
    for ep in eps1 | eps2:
        world, role, index = ep.split(":")
        assert world in known_worlds
        assert role in known_roles
        assert len(index) == 9 and index.isdigit()


def test_role_stats_truthful_accounting(tmp_path):
    """R2.2 Step 2: a 3-transition completed episode, a 7-transition
    completed episode and one observation-fault voided episode — the
    transition proportions differ from the completed proportions and both
    are correct; the voided episode never counts as completed."""
    contract = _contract()
    policy = FakePolicy(contract)
    specs = _task_specs()
    sampler = ScriptedTaskSampler([
        _spec("mock_void", "support_expansion"),
        _spec("mock_reach3", "anticipatory"),
        _spec("mock_reach7", "dense_recovery"),
    ], base_seed=7)
    result = _run(
        contract, policy, _critic(contract, policy),
        _cfg(tmp_path, "no_update", n_episodes=2),
        specs, _runtime(),
        env_kwargs={"episode_scripts": {
            "mock_void": {"invalid_episode_steps": [2]},
            "mock_reach3": {"reach_step": 3},
            "mock_reach7": {"reach_step": 7},
        }},
        task_sampler=sampler,
    )
    assert result["completed_episodes"] == 2
    assert result["voided_episodes"] == 1
    assert result["transitions"] == 12  # void: steps 0-1, reach3: 0-2, reach7: 0-6
    rs = result["role_stats"]
    assert rs["attempted_episode_counts"] == {
        "anticipatory": 1, "dense_recovery": 1,
        "support_expansion": 1, "yaw_lateral_recovery": 0,
    }
    assert rs["completed_episode_counts"] == {
        "anticipatory": 1, "dense_recovery": 1,
        "support_expansion": 0, "yaw_lateral_recovery": 0,
    }
    assert rs["voided_episode_counts"] == {
        "anticipatory": 0, "dense_recovery": 0,
        "support_expansion": 1, "yaw_lateral_recovery": 0,
    }
    assert rs["transition_counts"] == {
        "anticipatory": 3, "dense_recovery": 7,
        "support_expansion": 2, "yaw_lateral_recovery": 0,
    }
    assert rs["attempted_episode_proportions"] == {
        "anticipatory": 1 / 3, "dense_recovery": 1 / 3,
        "support_expansion": 1 / 3, "yaw_lateral_recovery": 0.0,
    }
    assert rs["completed_episode_proportions"]["anticipatory"] == pytest.approx(0.5)
    assert rs["completed_episode_proportions"]["dense_recovery"] == pytest.approx(0.5)
    assert rs["completed_episode_proportions"]["support_expansion"] == 0.0
    assert rs["transition_proportions"]["anticipatory"] == pytest.approx(3 / 12)
    assert rs["transition_proportions"]["dense_recovery"] == pytest.approx(7 / 12)
    assert rs["transition_proportions"]["support_expansion"] == pytest.approx(2 / 12)
    # the two proportions genuinely differ (guide: transition vs completed)
    assert rs["transition_proportions"]["anticipatory"] != pytest.approx(0.5)
    assert sampler.calls == 3
    assert sampler.completed == 2


def test_all_episodes_voided_raises_incomplete(tmp_path):
    """An all-voided batch must raise — never return a partial batch as
    success (guide R2.2 Step 3)."""
    contract = _contract()
    policy = FakePolicy(contract)
    specs = _task_specs()
    sampler = ScriptedTaskSampler(
        [_spec("mock_void", "support_expansion")], base_seed=0)
    with pytest.raises(EpisodeCollectionIncomplete) as ei:
        _run(
            contract, policy, _critic(contract, policy),
            _cfg(tmp_path, "no_update", n_episodes=2),
            specs, _runtime(),
            env_kwargs={"episode_scripts": {
                "mock_void": {"invalid_episode_steps": [1]},
            }},
            task_sampler=sampler,
        )
    assert ei.value.requested_episodes == 2
    assert ei.value.completed_episodes == 0
    assert ei.value.voided_episodes == 8  # every attempt voids
    assert ei.value.attempt_limit == 8  # max(2*2, 4) + 4


def test_all_episodes_rejected_raises_incomplete(tmp_path):
    """All-rejected at reset -> the same structured raise, never a zero-
    transition 'success' batch."""
    contract = _contract()
    policy = FakePolicy(contract)
    specs = _task_specs()
    sampler = ScriptedTaskSampler(
        [_spec("mock_reject", "dense_recovery")], base_seed=0)
    with pytest.raises(EpisodeCollectionIncomplete) as ei:
        _run(
            contract, policy, _critic(contract, policy),
            _cfg(tmp_path, "no_update", n_episodes=2),
            specs, _runtime(),
            env_kwargs={"episode_scripts": {
                "mock_reject": {"reset_invalid": True},
            }},
            task_sampler=sampler,
        )
    assert ei.value.completed_episodes == 0
    assert ei.value.rejected_episodes == 8


def test_one_update_metrics_checkpoint_eval(tmp_path):
    contract = _contract()
    policy = FakePolicy(contract)
    # Three capability roles (anticipatory/dense_recovery/support_expansion
    # in ROLE_ORDER) — the capability metrics hard-fail on any missing role
    # (D1), so the eval must cover all three.
    cfg = _cfg(tmp_path, "one_update", eval_n=3)
    result = _run(contract, policy, _critic(contract, policy), cfg, _task_specs(),
                  _runtime())
    # --- §4.15 metrics ------------------------------------------------
    metrics = result["metrics"]
    assert metrics is not None
    assert set(metrics) == METRIC_KEYS, set(metrics) ^ METRIC_KEYS
    assert metrics["update_batches"] == 1
    assert metrics["early_stopped"] is False
    for key in (
        "ratio_mean",
        "ratio_std",
        "actor_grad_norm",
        "critic_grad_norm",
        "parameter_update_norm",
        "approx_kl",
        "clipfrac",
        "policy_action_diversity",
    ):
        assert np.isfinite(metrics[key]), f"{key} not finite: {metrics[key]}"
    # --- checkpoint (§4.16) -------------------------------------------
    assert result["checkpoint_path"] is not None
    ckpt_files = list(tmp_path.glob("*.pt"))
    assert ckpt_files == [
        tmp_path / checkpoint_filename(result["env_steps"], 1)
    ]
    data = CheckpointBuilder.load(ckpt_files[0])
    assert data.schema_version == SCHEMA_VERSION
    assert data.ppo_update_count == 1
    assert data.optimizer_step_count == metrics["update_batches"]
    assert data.environment_transition_count == result["env_steps"] == 10
    assert data.completed_episode_count == 2
    assert data.contract_hash == contract_hash_of(contract)
    assert data.policy_state and data.value_state
    # R2 P1-3: the checkpoint carries the sampler state (SEED-RESUME) —
    # schema + consumed attempt index, not an empty stub
    assert data.sampler_state["schema"] == "episode_task_sampler/v1"
    assert data.sampler_state["global_attempt_index"] == 2
    # --- §4.17 paired eval artifact ------------------------------------
    art = result["eval_artifact"]
    validate_eval_artifact(
        art,
        expected_exec_steps=contract.exec_steps,
        expected_contract_hash=contract_hash_of(contract),
    )  # runtime contract hash at the call site must validate
    assert art.contract_hash == contract_hash_of(contract)
    assert art.exec_steps == 4
    assert len(art.episodes) == 3
    for ep in art.episodes:
        for tag in ("bc", "rl"):
            for key in ("success", "collision", "timeout", "return", "steps"):
                assert key in ep.__getattribute__(tag), (tag, key)
    # CAPABILITY-AUDIT: the metrics are computed from the paired records —
    # every capability group carries its required key set, and each
    # group's n_episodes aligns with the role's record count.
    assert set(art.capability_metrics) == {
        "anticipatory_avoidance",
        "fullspace_support_expansion",
        "dense_obstacle_ceiling",
    }
    for group, required in CAPABILITY_METRIC_KEYS.items():
        group_metrics = art.capability_metrics[group]
        missing = [key for key in required if key not in group_metrics]
        assert not missing, f"{group} missing keys: {missing}"
        assert group_metrics["n_episodes"] == sum(
            1 for ep in art.episodes if ep.role == GROUP_OF_ROLE[group]
        ), f"{group} role alignment"


def test_checkpoint_exact_config_round_trip(tmp_path):
    """Rework #2 (audit 2026-08-08): the checkpoint's
    exact_resolved_config must persist the runtime contract — policy_contract
    (policy_cfg_lambda=1.5, rollout_policy_batch_size=1,
    likelihood_microbatch_size=1) and the trainer's configured + RESOLVED
    next_value_batch_size — and survive a save -> load round trip without
    drift (resolved == the chunk ppo_update actually used)."""
    contract = _contract()
    policy = FakePolicy(contract)
    cfg = _cfg(tmp_path, "one_update")
    result = _run(contract, policy, _critic(contract, policy), cfg,
                  _task_specs(), _runtime())
    metrics = result["metrics"]
    assert metrics is not None
    ckpt_files = list(tmp_path.glob("*.pt"))
    assert len(ckpt_files) == 1
    data = CheckpointBuilder.load(ckpt_files[0])
    erc = data.exact_resolved_config
    assert erc["policy_contract"] == {
        "policy_cfg_lambda": 1.5,
        "rollout_policy_batch_size": 1,
        "likelihood_microbatch_size": 1,
    }, (
        "checkpoint must pin the rollout/likelihood contract that the"
        " sealed G5 identity was measured under"
    )
    trainer_erc = erc["trainer"]
    assert trainer_erc["next_value_batch_size_configured"] == 64, (
        "configured value must be the TrainerConfig default (64)"
    )
    assert (
        trainer_erc["next_value_batch_size_resolved"]
        == metrics["next_value_batch_size"]
    ), (
        "resolved value must equal the chunk ppo_update actually used"
        f" (metrics {metrics['next_value_batch_size']} vs"
        f" persisted {trainer_erc['next_value_batch_size_resolved']})"
    )


# ═══════════════════════════════════════════════════════════════
# Precise episode budget — no over-refill, nothing active at return, no
# leftovers across consecutive batches
# ═══════════════════════════════════════════════════════════════

_MULTI_SCRIPTS = {
    "mock_anticipatory": {"reach_step": 3},
    "mock_dense_recovery": {"reach_step": 5},
    "mock_support_expansion": {"reach_step": 2},
    "mock_yaw_lateral_recovery": {"reach_step": 4},
}


def test_multi_slot_precise_budget_no_active_at_return(tmp_path):
    """Rework #3 RED: n_envs=4 with heterogeneous episode lengths and a
    NON-multiple n_episodes must complete EXACTLY n_episodes — attempted ==
    completed, voided == 0, and NOTHING left active at return. The old loop
    refilled every inactive slot without subtracting the active count, so
    the last batch overshot (attempted 123 for 120 requested) and silently
    abandoned the tail episodes."""
    contract = _contract()
    policy = FakePolicy(contract)
    specs = _task_specs()
    sampler = ScriptedTaskSampler(specs, base_seed=100)
    env = MultiSlotMockUAVEnv(
        contract, n_envs=4, episode_scripts=_MULTI_SCRIPTS,
    )
    result = _run(
        contract, policy, _critic(contract, policy),
        _cfg(tmp_path, "no_update", n_episodes=6),
        specs, _runtime(), env=env, task_sampler=sampler,
    )
    assert result["completed_episodes"] == 6
    rs = result["role_stats"]
    attempted = sum(rs["attempted_episode_counts"].values())
    voided = sum(rs["voided_episode_counts"].values())
    assert attempted == 6, f"exact budget: attempted {attempted} != 6"
    assert voided == 0
    # no active slot at return: every started episode has a verdict
    assert all(env._done[i] for i in range(env.n_envs)), (
        "an episode was left active at collection stop"
    )


def test_consecutive_batches_same_adapter_no_active_leftover(tmp_path):
    """Rework #3 RED: TWO consecutive run_e2e_batch calls on the SAME
    multi-slot adapter — batch 1 leaves nothing active (precise budget), so
    batch 2 starts from a clean slot table and the sampler continues; the
    old loop left 3 in-flight episodes that the next batch's reset_at
    silently overwrote."""
    contract = _contract()
    policy = FakePolicy(contract)
    specs = _task_specs()
    sampler = ScriptedTaskSampler(specs, base_seed=100)
    env = MultiSlotMockUAVEnv(
        contract, n_envs=2, episode_scripts=_MULTI_SCRIPTS,
    )
    r1 = _run(
        contract, policy, _critic(contract, policy),
        _cfg(tmp_path, "no_update", n_episodes=3),
        specs, _runtime(), env=env, task_sampler=sampler,
    )
    r2 = _run(
        contract, policy, _critic(contract, policy),
        _cfg(tmp_path, "no_update", n_episodes=3),
        specs, _runtime(), env=env, task_sampler=sampler,
    )
    assert r1["completed_episodes"] == 3
    assert r2["completed_episodes"] == 3
    for label, result in (("batch1", r1), ("batch2", r2)):
        rs = result["role_stats"]
        attempted = sum(rs["attempted_episode_counts"].values())
        voided = sum(rs["voided_episode_counts"].values())
        assert attempted == 3, f"{label}: attempted {attempted} != 3"
        assert voided == 0, f"{label}: voided {voided} != 0"
    assert all(env._done[i] for i in range(env.n_envs)), (
        "an episode was left active at collection stop"
    )


# ═══════════════════════════════════════════════════════════════
# R6-F §11.7.2-11.7.4: official scheduler + running reward scaler
# through the REAL e2e path — checkpoint structured state, per-batch
# LR metrics, cross-field step counts, resume continuation
# ═══════════════════════════════════════════════════════════════

_SCHED = LRSchedulerConfig(
    method="official_cosine_warmup",
    first_cycle_steps=10, warmup_steps=3, min_lr=1e-4, max_lr=1e-2,
)
_SCALE = RewardScaleConfig(
    method="official_running_return_std", gamma=0.99,
)


def test_e2e_official_scheduler_scaler_checkpoint_and_metrics(tmp_path):
    """one_update with official scheduler + running scaler: metrics
    record actor/critic LR before/after + step counts; the checkpoint
    carries STRUCTURED lr_scheduler_state/reward_norm_state that pass
    the same validation the resume gate uses; cross-field step counts
    equal the committed update count (warmup window 0 -> actor stepped
    too)."""
    contract = _contract()
    policy = FakePolicy(contract)
    cfg = _cfg(tmp_path, "one_update", lr_scheduler=_SCHED, reward_scale=_SCALE)
    result = _run(contract, policy, _critic(contract, policy), cfg,
                  _task_specs(), _runtime())
    metrics = result["metrics"]
    assert metrics["update_batches"] == 1
    # warmup ramp in progress: LR starts at min_lr and moves INSIDE the
    # update (post-step LR differs), both actor and critic
    assert metrics["actor_lr_before"] == _SCHED.min_lr
    assert metrics["critic_lr_before"] == _SCHED.min_lr
    assert metrics["actor_lr_after"] > metrics["actor_lr_before"]
    assert metrics["critic_lr_after"] > metrics["critic_lr_before"]
    assert metrics["actor_scheduler_step_count"] == 1
    assert metrics["critic_scheduler_step_count"] == 1
    assert metrics["reward_scaler_step_count"] == 1
    # checkpoint carries structured state — never {"enabled": False} or {}
    data = CheckpointBuilder.load(result["checkpoint_path"])
    assert data.ppo_update_count == 1
    sched_state = data.lr_scheduler_state
    assert sched_state["method"] == "official_cosine_warmup"
    assert sched_state["version"] == SCHEDULER_STATE_VERSION
    validate_scheduler_state(
        sched_state, _SCHED, ppo_update_count=1, n_critic_warmup_updates=0
    )
    scale_state = data.reward_norm_state
    assert scale_state["method"] == "official_running_return_std"
    assert scale_state["source"] == "env"
    assert scale_state["params"]["state_version"] == REWARD_SCALE_STATE_VERSION
    validate_reward_scale_state(scale_state, _SCALE, ppo_update_count=1)
    # cross-field: critic == scaler == ppo_update_count; actor == count-0
    assert sched_state["critic"]["step_count"] == 1
    assert sched_state["actor"]["step_count"] == 1
    assert scale_state["params"]["step_count"] == 1
    # the running scaler actually scaled: variance is data-driven now
    assert scale_state["params"]["count"] != 1e-4


def test_e2e_scheduler_scaler_resume_continues_sequence(tmp_path):
    """Two one-update batches with resume_from on the same live env: the
    second batch's scheduler LR continues the exact sequence
    (before(batch2) == after(batch1)), step counts advance to the
    resumed ppo_update_count, and the second checkpoint re-validates
    (checkpoint-to-checkpoint continuation, not a restart)."""
    contract = _contract()
    policy = FakePolicy(contract)
    env = MockUAVEnv(contract)
    cfg = _cfg(tmp_path, "one_update", lr_scheduler=_SCHED, reward_scale=_SCALE)
    r1 = _run(contract, policy, _critic(contract, policy), cfg,
              _task_specs(), _runtime(), env=env)
    m1 = r1["metrics"]
    assert m1["actor_scheduler_step_count"] == 1
    assert m1["critic_scheduler_step_count"] == 1
    r2 = _run(contract, policy, _critic(contract, policy), cfg,
              _task_specs(), _runtime(), env=env,
              resume_from=r1["checkpoint_path"])
    m2 = r2["metrics"]
    # LR sequence continues: batch 2 observes the batch-1 post-step LR
    assert m2["actor_lr_before"] == m1["actor_lr_after"]
    assert m2["critic_lr_before"] == m1["critic_lr_after"]
    assert m2["actor_lr_after"] > m2["actor_lr_before"], (
        "warmup still ramping on the resumed batch"
    )
    assert m2["actor_scheduler_step_count"] == 2
    assert m2["critic_scheduler_step_count"] == 2
    assert m2["reward_scaler_step_count"] == 2
    data = CheckpointBuilder.load(r2["checkpoint_path"])
    assert data.ppo_update_count == 2
    validate_scheduler_state(
        data.lr_scheduler_state, _SCHED,
        ppo_update_count=2, n_critic_warmup_updates=0,
    )
    validate_reward_scale_state(data.reward_norm_state, _SCALE,
                                ppo_update_count=2)
    # cross-field after the resumed update: 2/2/2
    assert data.lr_scheduler_state["critic"]["step_count"] == 2
    assert data.lr_scheduler_state["actor"]["step_count"] == 2
    assert data.reward_norm_state["params"]["step_count"] == 2


def test_e2e_scheduler_resume_warmup_zero(tmp_path):
    """warmup_steps=0（CLI 默认 --scheduler_warmup_steps 0）: cold
    resume 必须成功 —— Gate B R6-TUNE-1 REWORK 反例：旧构造序在
    warmup=0 时构造 LR=max_lr，而 save 侧冻结 min_lr 锚点 →
    resume 侧 optimizer 合同 drift 硬失败（R6.5 阶段 C scheduler
    轴必撞此路径）。"""
    _SCHED0 = LRSchedulerConfig(
        method="official_cosine_warmup",
        first_cycle_steps=10, warmup_steps=0, min_lr=1e-4, max_lr=1e-2,
    )
    contract = _contract()
    policy = FakePolicy(contract)
    env = MockUAVEnv(contract)
    cfg = _cfg(tmp_path, "one_update", lr_scheduler=_SCHED0,
               reward_scale=_SCALE)
    r1 = _run(contract, policy, _critic(contract, policy), cfg,
              _task_specs(), _runtime(), env=env)
    m1 = r1["metrics"]
    assert m1["actor_scheduler_step_count"] == 1
    # cold resume（同一 live env 重启）：不得 optimizer 合同 drift 硬失败
    r2 = _run(contract, policy, _critic(contract, policy), cfg,
              _task_specs(), _runtime(), env=env,
              resume_from=r1["checkpoint_path"])
    m2 = r2["metrics"]
    # LR 序列延续：batch2 观察到 batch1 post-step LR；step count 推进
    assert m2["actor_lr_before"] == m1["actor_lr_after"]
    assert m2["actor_scheduler_step_count"] == 2
    assert m2["critic_scheduler_step_count"] == 2
    data = CheckpointBuilder.load(r2["checkpoint_path"])
    validate_scheduler_state(
        data.lr_scheduler_state, _SCHED0,
        ppo_update_count=2, n_critic_warmup_updates=0,
    )
    assert data.lr_scheduler_state["actor"]["step_count"] == 2
