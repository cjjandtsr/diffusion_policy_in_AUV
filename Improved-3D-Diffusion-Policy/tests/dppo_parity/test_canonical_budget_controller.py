"""R6-A (F-01/F-02): the fixed sampling controller — formal composition
counterexamples (§11.2.1 of
Guide_A_OfficialCore_R6_Training_Admission_v1_0.md).

RED phase (current HEAD 52ce1eb): every test in this file must FAIL before
any production change. The failed output is saved under
program/reports/routeA_rl_env_v1/R6_20260818/02_test_runs/ as the RED
evidence for R6-CTRL-1 / R6-CTRL-2.

Recorded failure modes on current HEAD (2026-08-18):
  red1  target-only budget accepted — the entry runs (preflight exit 1)
        instead of refusing the missing horizon at CLI level (exit 2)
  red2  no release discipline — without a caller-supplied kind NO batch
        ever materializes a release (kinds == [resume]*4), because the
        finality decision is not post-hoc inside the batch driver
  red3  the episode count truncates the transition horizon — n_episodes=1
        stops the batch at 1 transition instead of the full 8 (F-02)
  red4  the explicit collection_mode contract field does not exist — the
        new contract cannot even be expressed (TypeError). The
        cut-continuation semantics themselves are B4-covered; this test
        pins the NEW contract's explicit-mode axis.
  red5  a demanded release with actual env_steps != planned target is
        saved as release instead of hard-failing (F-01 self-proving final
        pattern)

Pure-function oracles (compute_batch_target / is_budget_complete /
LoopConfig validation) live in test_exact_transition_budget.py; this file
holds the formal composition counterexamples through the REAL batch
driver (run_e2e_batch), never just compute_batch_target().
"""

from pathlib import Path

import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    LoopConfig,
    compute_batch_target,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv, ScriptedTaskSampler

torch.manual_seed(0)


# ═══════════════════════════════════════════════════════════════════
# harness (composition-level: real LoopConfig + real batch driver)
# ═══════════════════════════════════════════════════════════════════


def _specs(contract, prefix="mock_budget"):
    """Four-role spec list, world_uid keyed by role (the sampler's task
    source; episode_scripts hook on world_uid)."""
    return [
        EpisodeTaskSpec(
            world_uid=f"{prefix}_{role}",
            obstacle_json="{}",
            role=role,
            split="rl_train",
            admission_id=f"{prefix}__{role}",
            generator_version="mock",
            manifest_entry_hash=f"hash_{prefix}_{role}",
        )
        for role in ("anticipatory", "dense_recovery",
                     "support_expansion", "yaw_lateral_recovery")
    ]


def _runtime(device):
    return {
        "device": device,
        "reward_version": "v16",
        "manifest_hash": "budget-gate",
        "project_git_commit": "?",
        "project_tree_hash": "?",
        "source_file_hashes": {},
        "bc_checkpoint_hash": "?",
        "world_assets_hash": "budget-gate",
        "timestamp": "budget-gate",
    }


def _transition_config(total, horizon, target, n_episodes, output_dir):
    """The script's LoopConfig shape: constant total+horizon (hashed),
    per-batch remaining target (excluded from the hash)."""
    return LoopConfig(
        mode="one_update", n_episodes=n_episodes, max_ep_len=20,
        device="cpu", output_dir=output_dir,
        target_transitions=target,
        total_transition_budget=total,
        rollout_transitions_per_update=horizon,
    )


# ═══════════════════════════════════════════════════════════════════
# RED-1: target-only budget must be rejected at the entry (CLI, exit 2)
# ═══════════════════════════════════════════════════════════════════


def test_red1_target_only_cli_exits_nonzero(tmp_path):
    """§11.2.1 item 1: formal budget mode with ONLY --target_transitions
    (no horizon) must exit non-zero with a CLI-level refusal. On current
    HEAD the entry ACCEPTS the target-only budget and only fails later at
    the environment preflight (exit 1, wrong failure point)."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    out = subprocess.run(
        [sys.executable, "scripts/train_dppo_uav_official.py",
         "--stage", "pilot", "--output_dir", str(tmp_path),
         "--target_transitions", "25"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert out.returncode == 2, (
        "target-only budget must be a CLI-level contract error (exit 2); "
        f"got exit {out.returncode}: {out.stderr[-400:]}"
    )
    assert "--rollout_transitions_per_update" in out.stderr, (
        "the CLI error must name the missing horizon flag"
    )


# ═══════════════════════════════════════════════════════════════════
# RED-2: total=25/horizon=8 through the real batch driver -> [8,8,8,1]
#        with exactly one release on the completing batch
# ═══════════════════════════════════════════════════════════════════


def test_red2_composition_25_8_exactly_one_release(tmp_path):
    """§11.2.1 item 2 + §11.2.5: total=25/horizon=8 via LoopConfig and the
    REAL batch driver must collect env_steps [8,8,8,1], the final
    checkpoint must record environment_transition_count == 25, and — with
    no caller-supplied kind — exactly the completing batch must
    materialize the release (post-hoc finality, not pre-proven)."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    env = MockUAVEnv(contract, reach_step=1000, max_steps=1000)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    specs = _specs(contract)
    trainer_config = TrainerConfig(batch_size=16, device="cpu")
    runtime = _runtime("cpu")
    sampler = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=0)
    lora_info = {"lora_params": [], "n_adaptors": 0}
    opt_a = torch.optim.Adam(list(policy.model.parameters()), lr=1e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)

    from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder

    steps, kinds = [], []
    prev_ckpt = None
    committed = 0
    batch_no = 0
    while committed < 25:
        target = compute_batch_target(25, committed, 8)
        cfg = _transition_config(25, 8, target, 4, str(tmp_path))
        res = run_e2e_batch(
            env, policy, critic, contract, cfg, trainer_config,
            specs, opt_a, opt_c, lora_info, runtime,
            resume_from=prev_ckpt, task_sampler=sampler,
            # R7-6 production cursor law: batch i saves cursor i-1 (records
            # committed before it) — the restore-side drift gate demands
            # cursor == outer_iteration-1 (launcher: batch_idx - 1).
            metrics_writer_cursor=batch_no,
        )
        batch_no += 1
        steps.append(int(res["env_steps"]))
        kinds.append(res["checkpoint_kind"])
        assert res["checkpoint_path"], "every one_update batch saves"
        prev_ckpt = res["checkpoint_path"]
        committed += int(res["env_steps"])

    assert steps == [8, 8, 8, 1], (
        f"composition must produce [8,8,8,1], got {steps}"
    )
    final = CheckpointBuilder.load(Path(prev_ckpt))
    assert final.environment_transition_count == 25, (
        f"final checkpoint counter {final.environment_transition_count} "
        "!= 25 — the cumulative count must equal the collected transitions"
    )
    assert kinds == ["resume", "resume", "resume", "release"], (
        "exactly the completing batch materializes the release — the "
        f"decision is post-hoc (is_budget_complete), got kinds={kinds}"
    )


# ═══════════════════════════════════════════════════════════════════
# RED-3: episode count must not truncate the transition horizon
# ═══════════════════════════════════════════════════════════════════


def test_red3_short_episodes_still_fill_horizon():
    """§11.2.1 item 3 + §11.2.3: 1-transition episodes with n_episodes=1 —
    the single batch must still collect the FULL 8 transitions. On current
    HEAD the episode count (completed == n_episodes) stops the loop at 1
    transition (F-02: n_episodes enters the stop condition)."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    env = MockUAVEnv(contract, reach_step=1, max_steps=1)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    specs = _specs(contract)
    cfg = LoopConfig(
        mode="no_update", n_episodes=1, max_ep_len=20, device="cpu",
        output_dir="/tmp/none",
        target_transitions=8, total_transition_budget=8,
        rollout_transitions_per_update=8,
    )
    trainer_config = TrainerConfig(batch_size=16, device="cpu")
    res = run_e2e_batch(
        env, policy, critic, contract, cfg, trainer_config, specs,
        torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3),
        torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3),
        {"lora_params": [], "n_adaptors": 0}, _runtime("cpu"),
        task_sampler=EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS,
                                        base_seed=0),
    )
    assert res["env_steps"] == 8, (
        f"transition mode must collect the full horizon: env_steps="
        f"{res['env_steps']} != 8 — the episode count (n_episodes=1) must "
        "never truncate the transition budget (F-02)"
    )


# ═══════════════════════════════════════════════════════════════════
# RED-4: long-episode cut + cross-update continuation (new contract)
# ═══════════════════════════════════════════════════════════════════


def test_red4_long_episode_continues_across_updates(tmp_path):
    """§11.2.1 item 4: batch 1 rollout-cuts one long episode at the 8th
    transition (slot stays ACTIVE), batch 2 continues the SAME slot from
    the 9th transition — no reset, no repeated transition 8. The new
    contract is expressed EXPLICITLY (collection_mode="transition"); on
    current HEAD the field does not exist (TypeError) — the continuation
    semantics themselves are B4-covered, this test pins the explicit
    contract axis on top of them."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    env = MockUAVEnv(contract, reach_step=1000, max_steps=1000)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    specs = _specs(contract)
    trainer_config = TrainerConfig(batch_size=16, device="cpu")
    runtime = _runtime("cpu")
    sampler = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=0)
    lora_info = {"lora_params": [], "n_adaptors": 0}
    opt_a = torch.optim.Adam(list(policy.model.parameters()), lr=1e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)

    def _cfg(target):
        return LoopConfig(
            mode="one_update", n_episodes=1, max_ep_len=20, device="cpu",
            output_dir=str(tmp_path),
            collection_mode="transition",
            target_transitions=target, total_transition_budget=16,
            rollout_transitions_per_update=8,
        )

    res1 = run_e2e_batch(
        env, policy, critic, contract, _cfg(8), trainer_config,
        specs, opt_a, opt_c, lora_info, runtime,
        task_sampler=sampler,
    )
    assert res1["env_steps"] == 8
    assert res1["active_slot_states"], "batch 1 must cut mid-episode"
    assert env.is_active(0), "the cut slot must stay ACTIVE"
    b1 = res1["buffer"]
    # The cut transition bootstraps like truncated (rollout_cut) while the
    # env slot is NOT closed.
    assert b1.truncated[-1] is True, "the cut transition must bootstrap"

    res2 = run_e2e_batch(
        env, policy, critic, contract, _cfg(8), trainer_config,
        specs, opt_a, opt_c, lora_info, runtime,
        resume_from=res1["checkpoint_path"], task_sampler=sampler,
    )
    assert res2["env_steps"] == 8, "batch 2 must collect its own horizon"
    b2 = res2["buffer"]
    assert b2.episode_ids[0] == b1.episode_ids[-1], (
        "batch 2 must continue the SAME episode (no reset, no new task)"
    )
    assert b2.episode_steps[0] == b1.episode_steps[-1] + 1, (
        f"continuation must start at the 9th transition, got "
        f"{b2.episode_steps[0]} after {b1.episode_steps[-1]}"
    )
    for key in b2.obs[0]:
        assert (b2.obs[0][key] == b1.next_obs[-1][key]).all(), (
            f"continuation obs diverges at {key} — the restored slot must "
            "reproduce the interrupted world bitwise"
        )


# ═══════════════════════════════════════════════════════════════════
# RED-5: actual env_steps != planned target -> no release, loud failure
# ═══════════════════════════════════════════════════════════════════


def test_red5_mismatch_env_steps_never_materializes_release(tmp_path):
    """§11.2.1 item 5: with actual env_steps != batch target (attempt-limit
    exhaustion leaves the budget incomplete), a caller demanding
    checkpoint_kind="release" must hard-fail and save NOTHING — the
    release decision is post-hoc (is_budget_complete), never a pre-proven
    predicate. On current HEAD the demand is honored: a release checkpoint
    is saved for the incomplete batch (F-01)."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    specs = _specs(contract)
    # A cycle where 3 of 4 tasks exhaust admission: the 8-attempt limit
    # admits only the two "work" tasks -> 2 transitions collected, far
    # short of the 8-transition target (env_steps != planned).
    fail_specs = [EpisodeTaskSpec(
        world_uid=f"r6a_fail_{i}", obstacle_json="{}", role="anticipatory",
        split="rl_train", admission_id=f"r6a_fail__{i}",
        generator_version="mock", manifest_entry_hash=f"hash_fail_{i}",
    ) for i in range(4)]
    work_specs = [EpisodeTaskSpec(
        world_uid=f"r6a_work_{i}", obstacle_json="{}", role="anticipatory",
        split="rl_train", admission_id=f"r6a_work__{i}",
        generator_version="mock", manifest_entry_hash=f"hash_work_{i}",
    ) for i in range(4)]
    cycle = [fail_specs[0], fail_specs[1], fail_specs[2], work_specs[0],
             fail_specs[3], fail_specs[0], fail_specs[1], work_specs[1]]
    scripts = {s.world_uid: {"admission_exhausted": True}
               for s in fail_specs}
    env = MockUAVEnv(contract, reach_step=1, max_steps=1,
                     episode_scripts=scripts)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    cfg = _transition_config(8, 8, 8, 1, str(tmp_path))
    trainer_config = TrainerConfig(batch_size=16, device="cpu")
    sampler = ScriptedTaskSampler(cycle, base_seed=0)
    opt_a = torch.optim.Adam(list(policy.model.parameters()), lr=1e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)

    try:
        run_e2e_batch(
            env, policy, critic, contract, cfg, trainer_config,
            specs, opt_a, opt_c, {"lora_params": [], "n_adaptors": 0},
            _runtime("cpu"), task_sampler=sampler,
            checkpoint_kind="release",
        )
    except RuntimeError as e:
        assert "release" in str(e), e
    else:
        raise AssertionError(
            "a demanded release on a batch whose actual env_steps != "
            "planned target must hard-fail — the release decision is "
            "post-hoc (F-01), never pre-proven"
        )
    assert not list(Path(tmp_path).iterdir()), (
        "the refused release must leave NO checkpoint behind"
    )
