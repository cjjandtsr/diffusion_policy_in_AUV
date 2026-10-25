"""P5 (ASSUMPTION-9): A2/A4/A5/A6 + reward-norm single source + HISTORICAL guard.

RED tests for the adjudicated contracts:
- A2 (§7.4): use_obstacle_projection must be False everywhere — the
  contract rejects True, train_loop hard-fails on a projection-enabled
  policy, the eval artifact carries the flag and hard-fails when True.
- A4 (§3.4): sampling vs logprob std split into
  min_sampling_denoising_std / min_logprob_denoising_std (official
  diffusion_vpg defaults 0.1/0.1).
- A5: history_indices (observation-selection metadata) enters the eval
  artifact and the train_loop result.
- A6: RoleSampler.proportion_report gets a caller — per-batch role
  proportion statistics in the train_loop result and metrics.
- reward-norm single source: RewardNormState (source=env, method=none),
  checkpointed and preserved on load (附录 H anti-pattern).
- HISTORICAL_INVALID_FOR_FORMAL_TRAINING: formal-mode rejection protection
  for legacy E-series entry points.
"""

import dataclasses
import glob
import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    CheckpointBuilder,
    checkpoint_filename,
    contract_hash_of,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (
    HISTORICAL_FORMAL_ENTRYPOINTS,
    HISTORICAL_INVALID_FOR_FORMAL_TRAINING,
    guard_formal_mode,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
    ROLE_NAMES,
    RoleSampler,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    EvalArtifactError,
    build_eval_artifact,
    validate_eval_artifact,
)
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    CAPABILITY_METRIC_KEYS,
    GROUP_OF_ROLE,
    OPPORTUNITY_INVALID_REASONS,
    OfflineDensity,
)
from diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter import (
    make_get_new_logprobs,
    sample_dppo_rollout,
)
import diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter as adapter_mod
from diffusion_policy_3d.rl.dppo_uav.reward_norm import RewardNormState
from diffusion_policy_3d.rl.dppo_uav.train_loop import LoopConfig, run_e2e_batch
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from fake_policy import FakePolicy, to_tensor_obs
from mock_uav_env import MockUAVEnv


def _contract() -> RLSystemContract:
    return RLSystemContract()


def _obs_batch(contract: RLSystemContract, b: int = 2, seed: int = 0):
    rng = np.random.RandomState(seed)
    obs = {
        "point_cloud": rng.uniform(
            -8.0, 8.0, (b, contract.history_steps, contract.point_count, 3)
        ).astype(np.float32),
        "agent_pos": rng.uniform(-1.0, 1.0, (b, contract.history_steps, 9)).astype(
            np.float32
        ),
    }
    return to_tensor_obs(obs)


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
        for role in ROLE_NAMES
    ]


def _runtime() -> dict:
    return {
        "timestamp": "2026-08-06T00:00:00Z",
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


def _run(
    contract: RLSystemContract,
    policy: FakePolicy,
    cfg: LoopConfig,
    task_specs: list = None,
):
    critic = _critic(contract, policy)
    env = MockUAVEnv(contract, seed=0)
    specs = task_specs or _task_specs()
    return run_e2e_batch(
        env=env,
        policy=policy,
        critic=critic,
        contract=contract,
        loop_config=cfg,
        trainer_config=TrainerConfig(device=cfg.device),
        task_specs=specs,
        actor_optimizer=torch.optim.AdamW(policy.model.parameters(), lr=1e-4),
        critic_optimizer=torch.optim.AdamW(
            critic.value_head_parameters(), lr=1e-4
        ),
        lora_info={},
        runtime=_runtime(),
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
    )


def _cfg(tmp_path, mode: str, **kwargs) -> LoopConfig:
    base = dict(
        mode=mode,
        n_episodes=2,
        max_ep_len=12,
        device="cpu",
        output_dir=str(tmp_path),
    )
    base.update(kwargs)
    return LoopConfig(**base)


# ═══════════════════════════════════════════════════════════════════
# A2 — §7.4 obstacle projection formal contract
# ═══════════════════════════════════════════════════════════════════

def test_contract_rejects_obstacle_projection_true():
    """§7.4: formal training/eval must run use_obstacle_projection == False —
    constructing the contract with True must hard-fail."""
    with pytest.raises(AssertionError):
        RLSystemContract(use_obstacle_projection=True)


def test_contract_serializes_obstacle_projection_false():
    """The False flag must be part of the contract serialization, so the
    checkpoint/eval contract hash carries it."""
    import json
    from dataclasses import asdict

    from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
        contract_to_checkpoint_bytes,
    )

    payload = json.loads(contract_to_checkpoint_bytes(_contract()))
    assert payload["use_obstacle_projection"] is False


def test_train_loop_hard_fails_on_obstacle_projection_policy(tmp_path):
    """§7.4: a policy loaded with use_obstacle_projection=True must never
    reach the training loop."""
    contract = _contract()
    policy = FakePolicy(contract)
    policy.use_obstacle_projection = True
    with pytest.raises(RuntimeError, match="use_obstacle_projection"):
        _run(contract, policy, _cfg(tmp_path, "no_update"))


def _paired() -> object:
    from diffusion_policy_3d.rl.dppo_uav.evaluator import EpisodePairedRecord
    from diffusion_policy_3d.rl.dppo_uav.paired_rng import policy_noise_seed

    ep_hash = "e0" * 32
    return EpisodePairedRecord(
        episode_config={"world": "w1", "seed": 42,
                        "episode_config_hash": ep_hash},
        seed=42,
        world="w1",
        role="anticipatory",
        start=[0.0, 0.0, 0.0],
        goal=[1.0, 2.0, 3.0],
        disturbance={"wind": 0.0},
        # R6-E (§11.6.4): raw terminated/truncated 标志与互斥
        # terminal_category 骑在 record 上；rl 侧 done_reason 必须是
        # "collision" — 严格 mapper 下 collision+done_reason="reached"
        # 是矛盾，validator 会 hard-fail。
        bc={"success": True, "collision": False, "timeout": False, "done_reason": "reached", "out_of_bounds": False, "altitude_exceed": False,
            "terminated": True, "truncated": False, "terminal_category": "reached",
            "return": 95.0, "steps": 40},
        rl={"success": False, "collision": True, "timeout": False, "done_reason": "collision", "out_of_bounds": False, "altitude_exceed": False,
            "terminated": True, "truncated": False, "terminal_category": "collision",
            "return": -20.0, "steps": 12},
        trace=_trace(),
        # R5.8-E (N4): the recorded seed must re-derive from the
        # artifact's own provenance (bank-p5 is the fixture bank id).
        policy_noise_seed=policy_noise_seed("bank-p5", ep_hash, 0),
    )


def _trace() -> dict:
    """A minimal VALID raw per-step trace."""
    step = {
        "step": 0, "elapsed": 0.0,
        "pos": [0.0, 0.0, 1.0], "z": 1.0, "yaw": 0.0,
        "goal_dist": 5.0, "clearance": 1.0, "hit_count": 4096,
        "action_norm": 1.0, "action_max_abs": 0.5,
        "done_reason": "none", "terminated": False, "truncated": False,
    }
    return {"bc": [dict(step)], "rl": [dict(step)]}


def _caps_for(episodes: list) -> dict:
    """Valid capability schema for an arbitrary episode set — groups
    whose role is absent get n_episodes=0 (validator role alignment)."""
    caps = {}
    for group, role in GROUP_OF_ROLE.items():
        n = sum(1 for ep in episodes if ep.role == role)
        caps[group] = {
            key: (n if key == "n_episodes" else 0.0)
            for key in CAPABILITY_METRIC_KEYS[group]
        }
    return caps


def test_eval_artifact_hard_fails_on_obstacle_projection_true():
    """§7.4: an artifact claiming projection-enabled evaluation is invalid."""
    with pytest.raises(EvalArtifactError, match="obstacle projection"):
        build_eval_artifact(  # validates on build — must reject True
            episodes=[_paired()],
            checkpoint_hash="ck",
            config_hash="cf",
            contract_hash="c1",
            manifest_hash="m",
            world_assets_hash="w",
            code_hash="cd",
            reward_version="v1.3",
            exec_steps=4,
            observation_invalid_count=0,
            missing_weight_keys=[],
            unexpected_weight_keys=[],
            lora_checksum="lora",
            timestamp="t",
            device="cpu",
            capability_metrics=_caps_for([_paired()]),
            history_indices=[5, 6, 7],
            use_obstacle_projection=True,
            expected_contract_hash="c1",
            bank_id="bank-p5",
            offline_density=OfflineDensity(
                counts={(x, 0, 2): 5 for x in range(17)}
            ),
        )


# ═══════════════════════════════════════════════════════════════════
# A4 — §3.4 sampling vs logprob std
# ═══════════════════════════════════════════════════════════════════

def test_dual_std_defaults_match_official():
    """Official diffusion_vpg: min_sampling_denoising_std=0.1,
    min_logprob_denoising_std=0.1."""
    sig = inspect.signature(sample_dppo_rollout)
    assert sig.parameters["min_sampling_denoising_std"].default == 0.1
    assert sig.parameters["min_logprob_denoising_std"].default == 0.1


def test_sampling_std_feeds_policy_sampling():
    """The rollout sampling call must use min_sampling_denoising_std
    (independent of the logprob floor — review R5.1 §3.4)."""
    contract = _contract()
    policy = FakePolicy(contract)
    seen = {}
    orig = policy.sample_ddpm_with_logprob

    def spy(**kwargs):
        seen["min_sampling_denoising_std"] = kwargs[
            "min_sampling_denoising_std"
        ]
        seen["min_logprob_denoising_std"] = kwargs[
            "min_logprob_denoising_std"
        ]
        return orig(**kwargs)

    policy.sample_ddpm_with_logprob = spy
    sample_dppo_rollout(
        policy, _obs_batch(contract, b=1), contract,
        min_sampling_denoising_std=0.05,
        min_logprob_denoising_std=0.2,
    )
    assert seen["min_sampling_denoising_std"] == 0.05, (
        "sampling path must use the sampling std, not the logprob std"
    )
    assert seen["min_logprob_denoising_std"] == 0.2, (
        "both independent floors must reach the sampling path"
    )


def test_logprob_std_feeds_logprob_recompute(monkeypatch):
    """The rollout-time logprob recompute must use
    min_logprob_denoising_std (with the sampling floor passed too)."""
    contract = _contract()
    policy = FakePolicy(contract)
    seen = {}

    def recorder(chain, timesteps, model, global_cond, global_cond_uncond,
                 cfg_lambda, scheduler, eta, min_sampling_denoising_std,
                 min_logprob_denoising_std, ft_denoising_steps):
        seen["min_sampling_denoising_std"] = min_sampling_denoising_std
        seen["min_logprob_denoising_std"] = min_logprob_denoising_std
        seen["ft_denoising_steps"] = ft_denoising_steps
        b = chain.shape[0]
        return torch.zeros(b, contract.ft_denoising_steps,
                           contract.action_horizon, contract.action_dim)

    monkeypatch.setattr(adapter_mod, "per_dim_denoising_logprobs", recorder)
    sample_dppo_rollout(
        policy, _obs_batch(contract, b=1), contract,
        min_sampling_denoising_std=0.05,
        min_logprob_denoising_std=0.2,
    )
    assert seen["min_logprob_denoising_std"] == 0.2, (
        "logprob recompute must use the logprob std, not the sampling std"
    )
    assert seen["min_sampling_denoising_std"] == 0.05, (
        "the sampling floor must also reach the recompute (one kernel, "
        "two independent floors)"
    )
    assert seen["ft_denoising_steps"] == contract.ft_denoising_steps


def test_get_new_logprobs_uses_logprob_std(monkeypatch):
    """Official get_logprobs path evaluates likelihood under the LOGPROB
    floor; the sampling floor also reaches the shared kernel (one
    kernel, two independent floors — review R5.1)."""
    contract = _contract()
    policy = FakePolicy(contract)
    seen = {}

    def recorder(x_k, x_next, k_batch, timesteps, model, global_cond,
                 global_cond_uncond, cfg_lambda, scheduler, eta,
                 min_sampling_denoising_std, min_logprob_denoising_std,
                 ft_denoising_steps):
        seen["min_sampling_denoising_std"] = min_sampling_denoising_std
        seen["min_logprob_denoising_std"] = min_logprob_denoising_std
        seen["ft_denoising_steps"] = ft_denoising_steps
        return torch.zeros(x_k.shape[0], contract.action_horizon,
                           contract.action_dim)

    monkeypatch.setattr(adapter_mod, "per_step_denoising_logp", recorder)
    get_new_logprobs = make_get_new_logprobs(
        policy, contract, min_logprob_denoising_std=0.2
    )
    b = 1
    get_new_logprobs(
        _obs_batch(contract, b=b),
        torch.zeros(b, contract.action_horizon, contract.action_dim),
        torch.zeros(b, contract.action_horizon, contract.action_dim),
        torch.zeros(b, dtype=torch.long),
    )
    assert seen["min_logprob_denoising_std"] == 0.2
    assert seen["ft_denoising_steps"] == contract.ft_denoising_steps


def test_dual_std_recorded_in_checkpoint_config(tmp_path):
    """The dual std must land in the checkpoint's exact resolved config."""
    contract = _contract()
    policy = FakePolicy(contract)
    result = _run(
        contract, policy,
        _cfg(tmp_path, "one_update",
             min_sampling_denoising_std=0.05,
             min_logprob_denoising_std=0.2),
    )
    ckpt_files = list(tmp_path.glob("*.pt"))
    assert ckpt_files
    data = CheckpointBuilder.load(ckpt_files[0])
    cfg = data.exact_resolved_config
    assert cfg["min_sampling_denoising_std"] == 0.05
    assert cfg["min_logprob_denoising_std"] == 0.2


# ═══════════════════════════════════════════════════════════════════
# A5 — history_indices into trace/eval artifact
# ═══════════════════════════════════════════════════════════════════

def _not_informative_outcomes():
    """R6-EVAL-2 (§11.5.3): 最小合法五件套 —— artifact 必须携带三组
    五件套；零机会 NOT_INFORMATIVE 形状（gate 绝无捏造 value）。"""
    from diffusion_policy_3d.rl.dppo_uav.evaluator import CAPABILITY_METRICS

    return {
        group: {
            "opportunity_count": 0,
            "exposure_count": 0,
            "event_count": 0,
            "value": None,
            "validity_reason": OPPORTUNITY_INVALID_REASONS["no_opportunity"],
        }
        for group in CAPABILITY_METRICS
    }


def test_eval_artifact_carries_history_indices():
    """§4.17: observation-selection metadata (history_indices) must be part
    of the eval artifact schema."""
    a = build_eval_artifact(
        episodes=[_paired()],
        checkpoint_hash="ck",
        config_hash="cf",
        contract_hash="c1",
        manifest_hash="m",
        world_assets_hash="w",
        code_hash="cd",
        reward_version="v1.3",
        exec_steps=4,
        observation_invalid_count=0,
        missing_weight_keys=[],
        unexpected_weight_keys=[],
        lora_checksum="lora",
        timestamp="t",
        device="cpu",
        capability_metrics=_caps_for([_paired()]),
        capability_outcomes=_not_informative_outcomes(),
        history_indices=[5, 6, 7],
        use_obstacle_projection=False,
        expected_contract_hash="c1",
        bank_id="bank-p5",
        offline_density=OfflineDensity(
            counts={(x, 0, 2): 5 for x in range(17)}
        ),
    )
    assert a.history_indices == [5, 6, 7]
    assert "history_indices" in a.to_dict()
    validate_eval_artifact(a, expected_contract_hash="c1")


def test_train_loop_records_observation_selection(tmp_path):
    """A5: the observation-selection metadata must survive the train_loop
    (T_h=8, T_o=3 -> latest window [5, 6, 7])."""
    contract = _contract()
    policy = FakePolicy(contract)
    result = _run(contract, policy, _cfg(tmp_path, "no_update"))
    sel = result["observation_selection"]
    assert sel["history_indices"] == [5, 6, 7]
    assert sel["history_steps"] == 8
    assert sel["obs_steps"] == 3


# ═══════════════════════════════════════════════════════════════════
# A6 — role proportion statistics (RoleSampler.proportion_report caller)
# ═══════════════════════════════════════════════════════════════════

def test_role_stats_in_train_loop_result(tmp_path):
    """R2.2: the truthful 7-field role stats in the result — in a clean
    batch (no faults) attempted == completed, voided all zero, and each
    completed episode contributes exactly 5 transitions."""
    contract = _contract()
    policy = FakePolicy(contract)
    result = _run(contract, policy, _cfg(tmp_path, "no_update"))
    rs = result["role_stats"]
    for field in (
        "attempted_episode_counts", "completed_episode_counts",
        "voided_episode_counts", "transition_counts",
        "attempted_episode_proportions", "completed_episode_proportions",
        "transition_proportions",
    ):
        assert set(rs[field]) == set(ROLE_NAMES), field
    assert sum(rs["attempted_episode_counts"].values()) == 2
    assert rs["attempted_episode_counts"] == rs["completed_episode_counts"]
    assert all(v == 0 for v in rs["voided_episode_counts"].values())
    assert sum(rs["transition_counts"].values()) == 10  # 2 x 5-step episodes
    for role in ROLE_NAMES:
        assert rs["transition_counts"][role] == \
            5 * rs["completed_episode_counts"][role]
    for prop in (
        "attempted_episode_proportions", "completed_episode_proportions",
        "transition_proportions",
    ):
        assert abs(sum(rs[prop].values()) - 1.0) < 1e-6


def test_role_proportion_report_in_metrics(tmp_path):
    """A6: one_update metrics must carry the role proportion report (the
    RoleSampler.proportion_report caller) — equal to the attempted-episode
    proportions in role_stats (single implementation)."""
    contract = _contract()
    policy = FakePolicy(contract)
    result = _run(contract, policy, _cfg(tmp_path, "one_update"))
    report = result["metrics"]["role_proportion_report"]
    assert set(report) == set(ROLE_NAMES)
    assert abs(sum(report.values()) - 1.0) < 1e-6
    assert report == result["role_stats"]["attempted_episode_proportions"]


def test_proportion_report_contract_matches_role_sampler():
    """The recorded report must come from RoleSampler.proportion_report
    itself (single implementation)."""
    sampler = RoleSampler({r: 0.25 for r in ROLE_NAMES})
    counts = {"anticipatory": 2, "dense_recovery": 0,
              "support_expansion": 0, "yaw_lateral_recovery": 0}
    assert sampler.proportion_report(counts)["anticipatory"] == 1.0


# ═══════════════════════════════════════════════════════════════════
# reward-norm single source (附录 H anti-pattern)
# ═══════════════════════════════════════════════════════════════════

def test_reward_norm_single_source_state():
    """The env's reward module is the ONLY normalization source; the RL core
    performs none (method="none"). Frozen — no post-hoc mutation."""
    state = RewardNormState(version="v1.3")
    assert state.source == "env"
    assert state.method == "none"
    assert state.to_dict() == {
        "version": "v1.3", "source": "env", "method": "none", "params": {},
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.source = "trainer"  # type: ignore[misc]


def test_reward_norm_state_preserved_through_checkpoint_roundtrip(tmp_path):
    """附录 H anti-pattern: a loaded reward-norm state must never be
    overwritten by a live one — the checkpoint is the single carrier."""
    contract = _contract()
    policy = FakePolicy(contract)
    result = _run(contract, policy, _cfg(tmp_path, "one_update"))
    ckpt_files = list(tmp_path.glob("*.pt"))
    data = CheckpointBuilder.load(ckpt_files[0])
    # R6-F (§11.7.3): the carrier is the full single-source state —
    # version/source/method static axes + the scaler's structured params
    # (state_version + step_count for method="none"; step_count == the
    # cross-field ppo_update_count invariant).
    rn = data.reward_norm_state
    assert rn["version"] == "v1.3", (
        "checkpoint must carry the single-source reward-norm version"
    )
    assert rn["source"] == "env", (
        "the env Reward module is the ONLY normalization source"
    )
    assert rn["method"] == "none"
    assert rn["params"]["state_version"] == "v1", (
        f"scaler params must be structured: {rn['params']}"
    )
    assert rn["params"]["step_count"] == data.ppo_update_count, (
        "scaler step_count must equal the checkpoint's ppo_update_count"
    )


# ═══════════════════════════════════════════════════════════════════
# HISTORICAL_INVALID_FOR_FORMAL_TRAINING
# ═══════════════════════════════════════════════════════════════════

def test_historical_guard_rejects_formal_mode():
    """Legacy E-series entry points must reject formal mode."""
    for entry in HISTORICAL_FORMAL_ENTRYPOINTS:
        with pytest.raises(RuntimeError, match=HISTORICAL_INVALID_FOR_FORMAL_TRAINING):
            guard_formal_mode(entry, "formal")
        guard_formal_mode(entry, "smoke")  # non-formal passes


def test_canonical_entry_not_classified_historical():
    """The official core (dppo_uav) is the canonical path — never rejected."""
    guard_formal_mode("diffusion_policy_3d/rl/dppo_uav/train_loop.py", "formal")


def test_legacy_entrypoints_are_classified_historical():
    """Every legacy E-series training script on disk must be classified."""
    legacy = sorted(glob.glob("scripts/train_ddpm_ppo_meanlogp*.py"))
    assert legacy, "no legacy E-series scripts found"
    for path in legacy:
        assert path in HISTORICAL_FORMAL_ENTRYPOINTS, (
            f"{path} missing from HISTORICAL_FORMAL_ENTRYPOINTS"
        )


def test_all_legacy_scripts_wire_the_guard():
    """R3.3: every legacy E-series script carries the guard call — the
    E040 freeze (0-modification) was lifted with hash accounting: pre-hash
    7868d1be8ec9... recorded in ENTRYPOINT_CLOSURE_REPORT.md, the post-hash
    as the guarded snapshot. A legacy script without wiring is a bypass of
    the unique formal entry.
    """
    for path in sorted(glob.glob("scripts/train_ddpm_ppo*.py")):
        # scripts are UTF-8 (em-dash in docstrings); read_text() defaults to
        # the locale encoding, which is ascii under LANG=C — decoded crash in
        # the authority suite run (2026-08-08). Pin utf-8 explicitly.
        text = Path(path).read_text(encoding="utf-8")
        assert "guard_formal_mode" in text, (
            f"{path} missing formal-mode rejection wiring"
        )


def test_legacy_entrypoint_rejects_formal_mode_subprocess():
    """P7: the wiring must fire — `--mode formal` through a legacy E-series
    script exits non-zero with the HISTORICAL rejection; smoke passes."""
    # The guard message carries an em-dash (—): under an ASCII locale
    # (LANG=C) text=True decodes the captured output with ascii and crashed
    # in the authority suite run (2026-08-08, UnicodeDecodeError). Pin the
    # parent decode AND force the child's stdio to UTF-8 so the verdict is
    # locale-independent.
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [sys.executable, "scripts/train_ddpm_ppo_meanlogp_3.py",
         "--dry_run", "--mode", "formal"],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=180,
    )
    assert r.returncode != 0, "formal mode must be rejected at the entry"
    assert HISTORICAL_INVALID_FOR_FORMAL_TRAINING in (r.stdout + r.stderr)
    r2 = subprocess.run(
        [sys.executable, "scripts/train_ddpm_ppo_meanlogp_3.py",
         "--dry_run", "--mode", "smoke"],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=180,
    )
    assert r2.returncode == 0, f"smoke dry-run must pass: {r2.stderr[-2000:]}"
