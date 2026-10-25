"""B1-B5 (review R5.3 / §8.3): fixed environment-transition budget with
active-slot continuation.

B1 exact budget: the batch stops the moment len(buffer) >= target and
the stored transition count equals the collected env steps;
B2 real batch cut: a mid-episode slot stays ACTIVE across the boundary
(no fake truncation of the ENV — only the cut transition bootstraps);
B4 resume identity: restoring the active-slot state and continuing
produces the SAME next observation/action/reward/state as an
uninterrupted run (mock env + FakePolicy, CPU);
B5 slot lifecycle: no illegal ACTIVE/DONE/VOIDED transitions.
"""

from pathlib import Path

import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import LoopConfig, run_e2e_batch
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv

torch.manual_seed(0)


def _specs(contract):
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        EpisodeTaskSpec,
    )

    return [
        EpisodeTaskSpec(
            world_uid=f"mock_budget_{role}",
            obstacle_json="{}",
            role=role,
            split="rl_train",
            admission_id=f"mock_budget__{role}",
            generator_version="mock",
            manifest_entry_hash=f"hash_budget_{role}",
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
        "world_assets_hash": "?",
        "timestamp": "budget-gate",
    }


def _batch(contract, target, env=None, resume_slots=None):
    """One budgeted no_update batch. Passing the SAME env continues its
    live slots (uninterrupted); resume_slots restores saved slots onto a
    fresh env (interrupted-resume side)."""
    policy = FakePolicy(contract)
    if env is None:
        # Long episodes (reach far beyond max_ep_len) so the 25-transition
        # budget cuts MID-episode instead of completing 4 short episodes.
        env = MockUAVEnv(contract, reach_step=1000, max_steps=1000)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    specs = _specs(contract)
    lora_info = {"lora_params": [], "n_adaptors": 0}
    loop_config = LoopConfig(
        mode="no_update", n_episodes=4, max_ep_len=20, device="cpu",
        output_dir="/tmp/none",
        target_transitions=target,
        total_transition_budget=target,
        # R6-A (§11.2.2): transition collection requires the fixed
        # per-update horizon — here total == target, so horizon == target
        # preserves the "one batch collects everything" semantics.
        rollout_transitions_per_update=target,
    )
    trainer_config = TrainerConfig(batch_size=16, device="cpu")
    return env, run_e2e_batch(
        env, policy, critic, contract, loop_config, trainer_config, specs,
        torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3),
        torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3),
        lora_info, _runtime("cpu"),
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
        resume_active_slots=resume_slots,
    )


def test_b1_exact_transition_budget():
    """The batch collects EXACTLY the target transitions (no overshoot:
    the budget check runs between steps)."""
    contract = RLSystemContract()
    target = 25
    _, res = _batch(contract, target)
    assert res["env_steps"] == target, (
        f"budget exactness: {res['env_steps']} != {target}"
    )
    assert len(res["buffer"].obs) == target


def test_b2_mid_episode_slot_stays_active():
    """A slot cut mid-episode is carried in active_slot_states (B2) and
    its final stored transition is flagged truncated=True for the GAE
    bootstrap while the ENV slot itself is still ACTIVE."""
    contract = RLSystemContract()
    _, res = _batch(contract, 25)
    active = res["active_slot_states"]
    assert active, "a mid-episode slot must remain active at the cut"
    for env_id, st in active.items():
        assert st["adapter"]["done"] is False, (
            f"slot {env_id} must stay ACTIVE (not closed) at the cut"
        )
    buffer = res["buffer"]
    # Count ONLY the rollout-cut transitions: the last stored transition
    # of each ACTIVE slot's episode (time-limit truncations of COMPLETED
    # episodes are legitimately truncated=True too).
    active_eps = {
        st["adapter"]["record"]["episode_id"] for st in active.values()
    }
    n_cut = sum(
        1 for i, t in enumerate(buffer.truncated)
        if t and buffer.episode_ids[i] in active_eps
    )
    assert n_cut == len(active), (
        f"{n_cut} cut-bootstrapped transitions vs {len(active)} active "
        "slots"
    )


def test_b4_resume_continues_exact():
    """Uninterrupted vs save/restore continuation: the restored world's
    next transition must equal the uninterrupted world's (B4)."""
    contract = RLSystemContract()
    env, res1 = _batch(contract, 25)
    active = res1["active_slot_states"]
    assert active
    snapshots = {int(i): st for i, st in active.items()}
    # Reference: the SAME env continues its live slots (uninterrupted —
    # the loop ADOPTS the active slot in-process).
    _, res2 = _batch(contract, 1, env=env)
    b2 = res2["buffer"]
    assert b2.episode_ids, "reference continuation produced no transition"
    # Restored: a FRESH env loaded from the captured slot states must
    # reproduce the identical next transition.
    _, res3 = _batch(contract, 1, resume_slots=snapshots)
    b3 = res3["buffer"]
    assert len(b3.obs) == len(b2.obs), (
        f"{len(b3.obs)} restored transitions vs {len(b2.obs)} reference"
    )
    for idx in range(len(b2.obs)):
        for key in b2.obs[idx]:
            assert (b2.obs[idx][key] == b3.obs[idx][key]).all(), (
                f"restored continuation diverges at {idx} obs {key}"
            )
        assert b2.episode_ids[idx] == b3.episode_ids[idx]
        assert b2.rewards[idx] == b3.rewards[idx]
        assert b2.episode_steps[idx] == b3.episode_steps[idx]


def test_b5_state_roundtrip_no_illegal_lifecycle():
    """state_dict/load_state_dict round-trips the full per-slot state
    (adapter bookkeeping + env kinematics/history/RNG) without illegal
    lifecycle mutation: the restored snapshot re-serializes identically."""
    contract = RLSystemContract()
    env, res = _batch(contract, 25)
    snapshots = {int(i): st for i, st in res["active_slot_states"].items()}
    assert snapshots
    # Close the live slots so a fresh env can load the snapshots.
    for env_id in snapshots:
        env.abort_episode([env_id])
    env2 = MockUAVEnv(contract, reach_step=1000, max_steps=1000)
    for env_id, st in snapshots.items():
        env2.load_state_dict(env_id, st)
        again = env2.state_dict(env_id)
        assert again["adapter"]["done"] == st["adapter"]["done"]
        assert again["adapter"]["voided"] == st["adapter"]["voided"]
        assert (again["env"]["start"] == st["env"]["start"]).all()
        assert (again["env"]["pc_history"]
                == st["env"]["pc_history"]).all()
        assert again["env"]["episode_step"] == st["env"]["episode_step"]
        # RandomState.get_state() returns a tuple with ndarrays — compare
        # element-wise (== on arrays is ambiguous).
        for a_val, b_val in zip(again["env"]["rng_state"],
                                st["env"]["rng_state"]):
            if hasattr(a_val, "shape"):
                assert (a_val == b_val).all()
            else:
                assert a_val == b_val


def test_p02_checkpoint_roundtrip_with_numpy_slot_states(tmp_path):
    """review partB P0-2: a budget-cut checkpoint carries numpy-heavy
    active_slot_states — CheckpointBuilder.save/load must round-trip
    them (weights_only=True rejects raw numpy)."""
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder

    contract = RLSystemContract()
    env, res = _batch(contract, 25)
    active = res["active_slot_states"]
    assert active, "budget cut must leave active slots"
    data = CheckpointBuilder.load(Path(_run_ckpt_path(tmp_path)))
    assert data.active_slot_states, "slot states lost in the round-trip"
    for env_id, state in active.items():
        loaded = data.active_slot_states[int(env_id)]
        assert (loaded["env"]["start"]
                == state["env"]["start"]).all()
        assert loaded["adapter"]["done"] is False


def test_p01_restore_returns_active_slot_states(tmp_path):
    """review partB P0-1: restore_training_state must RETURN the saved
    active_slot_states (the production resume path consumes them from
    the checkpoint, not from an explicit kwarg)."""
    contract = RLSystemContract()
    env, res = _batch(contract, 25)
    ckpt_path = _run_ckpt_path(tmp_path)
    policy = FakePolicy(contract)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    counters = _restore_counters(ckpt_path, policy, critic, contract)
    assert counters.get("active_slot_states"), (
        "restore_training_state dropped the active slot states (P0-1)"
    )


def _run_ckpt_path(tmp_path):
    """Run a budget-cut one_update batch and return its checkpoint."""
    from diffusion_policy_3d.rl.dppo_uav.train_loop import run_e2e_batch

    contract = RLSystemContract()
    policy = FakePolicy(contract)
    env = MockUAVEnv(contract, reach_step=1000, max_steps=1000)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    specs = _specs(contract)
    loop_config = LoopConfig(
        mode="one_update", n_episodes=4, max_ep_len=20, device="cpu",
        output_dir=str(tmp_path),
        target_transitions=25,
        total_transition_budget=25,
        rollout_transitions_per_update=25,
    )
    trainer_config = TrainerConfig(batch_size=16, device="cpu")
    res = run_e2e_batch(
        env, policy, critic, contract, loop_config, trainer_config, specs,
        torch.optim.Adam(list(policy.model.parameters()), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        {"lora_params": [], "n_adaptors": 0}, _runtime("cpu"),
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
    )
    assert res["active_slot_states"], "budget cut must leave active slots"
    return res["checkpoint_path"]


def _restore_counters(ckpt_path, policy, critic, contract):
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        FORMAL_NUMERICAL_MODE,
        LoopConfig,
        configure_numerical_mode,
        resume_identity_for,
        restore_training_state,
    )

    loop_config = LoopConfig(
        mode="one_update", n_episodes=4, max_ep_len=20, device="cpu",
        output_dir="/tmp/none", target_transitions=25,
        total_transition_budget=25, rollout_transitions_per_update=25,
    )
    trainer_config = TrainerConfig(batch_size=16, device="cpu")
    runtime = _runtime("cpu")
    # Same optimizer shapes as the saving run (param-group size gate).
    opt_a = torch.optim.Adam(list(policy.model.parameters()), lr=1e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    return restore_training_state(
        str(ckpt_path), policy, critic,
        {"lora_params": [], "n_adaptors": 0},
        opt_a,
        opt_c,
        runtime_identity=resume_identity_for(
            contract, loop_config, trainer_config, runtime,
            configure_numerical_mode(FORMAL_NUMERICAL_MODE), policy,
            base_seed=0, n_envs=1,
            lora_info={"lora_params": [], "n_adaptors": 0},
            actor_optimizer=opt_a,
            critic_optimizer=opt_c,
        ),
        task_sampler=None,
        structured_identity=None,
    )


# ═══════════════════════════════════════════════════════════════════
# R5.8-A (2026-08-17): P0-1 budget machinery + P0-2 in-process
# continuation regression tests.
# ═══════════════════════════════════════════════════════════════════


def test_compute_batch_target_8821_sequence():
    """total=25, horizon=8 must produce per-batch targets [8,8,8,1]."""
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        compute_batch_target,
    )

    seq = []
    committed = 0
    while committed < 25:
        t = compute_batch_target(25, committed, 8)
        seq.append(t)
        committed += t
    assert seq == [8, 8, 8, 1], seq
    assert committed == 25


def test_compute_batch_target_rejects_target_only():
    """R6-A (§11.2.2): there is NO target-only compatibility path — a
    total without the per-update horizon is a contract violation, never a
    single-batch target. ``total is None`` (episode mode) dominates and
    stays None regardless of the horizon argument."""
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        compute_batch_target,
    )

    for committed in (0, 10, 25):
        try:
            compute_batch_target(25, committed, None)
        except ValueError as e:
            assert "horizon" in str(e), e
        else:
            raise AssertionError(
                f"compute_batch_target(25, {committed}, None) is the "
                "target-only state — must raise (R6-A §11.2.2)"
            )
    # episode mode dominates: no total -> None, horizon irrelevant
    assert compute_batch_target(None, 10, None) is None
    assert compute_batch_target(None, 10, 8) is None


# ═══════════════════════════════════════════════════════════════════
# R6-A (F-01, §11.2.4): the pure finality oracle + the explicit
# collection-mode contract (pure-function side of the RED file).
# ═══════════════════════════════════════════════════════════════════


def test_is_budget_complete_oracle():
    """§11.2.4: the release materializes ONLY when the ACTUAL env steps
    land exactly on the run total — post-hoc finality, never a pre-proven
    predicate."""
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        TransitionBudgetOvershoot,
        is_budget_complete,
    )

    assert is_budget_complete(25, 17, 8) is True   # 17+8 == 25
    assert is_budget_complete(25, 0, 25) is True   # single completing batch
    assert is_budget_complete(8, 0, 8) is True
    assert is_budget_complete(25, 17, 7) is False  # 24 != 25 — incomplete
    assert is_budget_complete(25, 10, 15) is True
    # committed == total is a stopped run, not a live batch — the loop
    # only asks the oracle while committed < total, so this algebraic
    # identity is unreachable in production
    assert is_budget_complete(25, 25, 0) is True
    for total, committed, actual in ((25, 17, 9), (25, 0, 26)):
        try:
            is_budget_complete(total, committed, actual)
        except TransitionBudgetOvershoot:
            pass
        else:
            raise AssertionError(
                f"is_budget_complete({total}, {committed}, {actual}) "
                "overshoots the run budget — must raise"
            )


def test_loop_config_collection_mode_resolution():
    """§11.2.2: the mode is resolved when None (total set -> "transition",
    else "episode"); explicit modes are validated against their field
    sets; target-only budgets are forbidden — every violation is a loud
    AssertionError naming the missing/misplaced field."""
    from diffusion_policy_3d.rl.dppo_uav.train_loop import LoopConfig

    def _cfg(**kw):
        base = dict(mode="one_update", n_episodes=4, max_ep_len=20,
                    device="cpu", output_dir="/tmp/none")
        base.update(kw)
        return LoopConfig(**base)

    # derived: total set -> transition; total None -> episode
    assert _cfg(total_transition_budget=25,
                rollout_transitions_per_update=8,
                target_transitions=8).collection_mode == "transition"
    assert _cfg().collection_mode == "episode"
    # explicit transition requires total+horizon+target
    assert _cfg(collection_mode="transition",
                total_transition_budget=25,
                rollout_transitions_per_update=8,
                target_transitions=8).collection_mode == "transition"
    # episode mode forbids every budget field
    for bad in (dict(total_transition_budget=25),
                dict(rollout_transitions_per_update=8),
                dict(target_transitions=25),
                dict(total_transition_budget=25,
                     rollout_transitions_per_update=8,
                     target_transitions=8)):
        try:
            _cfg(collection_mode="episode", **bad)
        except AssertionError as e:
            assert "forbids budget fields" in str(e), e
        else:
            raise AssertionError(f"episode mode must refuse {bad}")
    # target-only (total None + target set): resolved to episode, then the
    # misplaced target is refused — the compatibility path does not exist
    try:
        _cfg(target_transitions=25)
    except AssertionError as e:
        assert "forbids budget fields" in str(e), e
    else:
        raise AssertionError("target-only budget must be refused")
    # transition missing the horizon names the flag (F-01 refusal point)
    try:
        _cfg(collection_mode="transition", total_transition_budget=25,
             target_transitions=8)
    except AssertionError as e:
        assert "rollout_transitions_per_update" in str(e), e
    else:
        raise AssertionError("transition without horizon must be refused")
    # transition missing the total
    try:
        _cfg(collection_mode="transition",
             rollout_transitions_per_update=8, target_transitions=8)
    except AssertionError as e:
        assert "total_transition_budget" in str(e), e
    else:
        raise AssertionError("transition without total must be refused")
    # unknown mode value is a contract error
    try:
        _cfg(collection_mode="budget")
    except AssertionError as e:
        assert "'episode' | 'transition'" in str(e), e
    else:
        raise AssertionError("unknown collection_mode must be refused")


def test_compute_batch_target_overshoot_raises():
    """A committed count beyond the total is a budgeting bug — loud."""
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        compute_batch_target,
    )

    try:
        compute_batch_target(25, 26, 8)
    except ValueError as e:
        assert "overshoot" in str(e)
    else:
        raise AssertionError("overshoot must raise")


def test_loop_config_target_requires_total():
    """The per-batch remaining target derives from the run-level total —
    constructing one without the other is a config error (P0-1). R6-A
    (§11.2.2): the full transition contract is total + horizon + target;
    a target-only pair (total without horizon) is also refused."""
    try:
        LoopConfig(
            mode="one_update", n_episodes=4, max_ep_len=20,
            device="cpu", output_dir="/tmp/none",
            target_transitions=25,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("target without total must raise")
    try:
        LoopConfig(
            mode="one_update", n_episodes=4, max_ep_len=20,
            device="cpu", output_dir="/tmp/none",
            target_transitions=25, total_transition_budget=25,
        )
    except AssertionError as e:
        assert "rollout_transitions_per_update" in str(e), e
    else:
        raise AssertionError("total without horizon must raise (R6-A)")
    cfg = LoopConfig(
        mode="one_update", n_episodes=4, max_ep_len=20,
        device="cpu", output_dir="/tmp/none",
        target_transitions=25, total_transition_budget=25,
        rollout_transitions_per_update=25,
    )
    assert cfg.total_transition_budget == 25


def test_config_hash_stable_across_batch_targets():
    """P0-1/A3: consecutive batches vary ONLY the per-batch remaining
    target — the config hash (the resume identity axis) must not change
    across them, or every resumed budgeted run fails the identity gate.
    """
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import config_hash_of
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        FORMAL_NUMERICAL_MODE,
        _config_dict,
        configure_numerical_mode,
    )

    contract = RLSystemContract()
    policy = FakePolicy(contract)
    trainer_config = TrainerConfig(batch_size=16, device="cpu")
    runtime = _runtime("cpu")
    numerical = configure_numerical_mode(FORMAL_NUMERICAL_MODE)
    lora_info = {"lora_params": [], "n_adaptors": 0}

    def _hash(total, target, horizon):
        cfg = LoopConfig(
            mode="one_update", n_episodes=4, max_ep_len=20,
            device="cpu", output_dir="/tmp/none",
            target_transitions=target,
            total_transition_budget=total,
            rollout_transitions_per_update=horizon,
        )
        layer = _config_dict(
            contract, cfg, trainer_config, runtime, numerical, policy,
            lora_info=lora_info,
        )
        return config_hash_of(layer)

    h1 = _hash(25, 8, 8)   # batch 1 of [8,8,8,1]
    h2 = _hash(25, 1, 8)   # batch 4 of [8,8,8,1]
    assert h1 == h2, "per-batch target must not change the config hash"
    h3 = _hash(30, 8, 8)
    assert h1 != h3, "a different TOTAL budget must change the hash"


def test_p02_in_process_budget_continuation_same_env(tmp_path):
    """P0-2 regression: the production chain passes resume_from=prev_ckpt
    on the SAME live env. A mid-episode budget cut leaves the slot
    ACTIVE; the next batch must ADOPT the live slot instead of calling
    load_state_dict on it (the adapter guard raises on ACTIVE slots —
    previously every budgeted multi-batch run crashed here)."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    env = MockUAVEnv(contract, reach_step=1000, max_steps=1000)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    specs = _specs(contract)
    trainer_config = TrainerConfig(batch_size=16, device="cpu")
    lora_info = {"lora_params": [], "n_adaptors": 0}
    runtime = _runtime("cpu")

    def _config(target):
        return LoopConfig(
            mode="one_update", n_episodes=4, max_ep_len=20,
            device="cpu", output_dir=str(tmp_path),
            target_transitions=target, total_transition_budget=25,
            rollout_transitions_per_update=25,
        )

    opt_a = torch.optim.Adam(list(policy.model.parameters()), lr=1e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    res1 = run_e2e_batch(
        env, policy, critic, contract, _config(10), trainer_config,
        specs, opt_a, opt_c, lora_info, runtime,
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
    )
    assert res1["active_slot_states"], "batch 1 must cut mid-episode"
    assert res1["env_steps"] == 10
    assert env.is_active(0), "the live slot must stay ACTIVE after the cut"
    res2 = run_e2e_batch(
        env, policy, critic, contract, _config(15), trainer_config,
        specs, opt_a, opt_c, lora_info, runtime,
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
        resume_from=res1["checkpoint_path"],
    )
    assert res2["env_steps"] == 15, (
        "the resumed batch must collect exactly the remaining budget"
    )
