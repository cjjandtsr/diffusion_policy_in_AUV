"""T5 closure: sample_dppo_rollout / make_get_new_logprobs / to_rollout_transition.

The rollout adapter maps the frozen phase1a policy's full K+1 chain to the
region-sliced DPPO window (K_f+1, H_a, 11) stored in the buffer (design §4.6
option 2, task_plan decision 5). get_new_logprobs recomputes per-dim logps on
the same window -> ratio == 1 before any update.
"""

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.chain_logprobs import per_dim_denoising_logprobs
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import EpisodeTaskSpec
from diffusion_policy_3d.rl.dppo_uav.observation_adapter import (
    select_latest_observations,
)
from diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter import (
    _obs_conditions,
    make_get_new_logprobs,
    sample_dppo_rollout,
    to_rollout_transition,
)
from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import RolloutBuffer

from fake_policy import FakePolicy, to_tensor_obs
from mock_uav_env import MockUAVEnv


def _contract() -> RLSystemContract:
    return RLSystemContract()


def _task_specs() -> list:
    return [
        EpisodeTaskSpec(
            world_uid="mock",
            obstacle_json="{}",
            role="anticipatory",
            split="rl_train",
            admission_id="m1",
            generator_version="mock",
            manifest_entry_hash="abc123",
        )
    ]


def _env_and_selected(contract: RLSystemContract, seed: int = 0):
    env = MockUAVEnv(contract, seed=seed)
    obs_np, info = env.reset(_task_specs(), seeds=[seed])
    obs_t = to_tensor_obs(obs_np)  # (1, T_h, ...)
    selected, _ = select_latest_observations(obs_t, contract.obs_steps)
    return env, obs_np, selected


def test_sample_shapes():
    contract = _contract()
    policy = FakePolicy(contract)
    _, _, selected = _env_and_selected(contract)
    sample = sample_dppo_rollout(policy, selected, contract)
    assert sample.chains_policy.shape == (
        1,
        contract.ft_denoising_steps + 1,
        contract.action_horizon,
        contract.action_dim,
    ), sample.chains_policy.shape
    assert sample.old_logprobs.shape == (
        1,
        contract.ft_denoising_steps,
        contract.action_horizon,
        contract.action_dim,
    ), sample.old_logprobs.shape
    assert sample.executed_action.shape == (1, contract.exec_steps, 11)
    assert sample.obs_condition.shape == (1, contract.obs_steps * 192)
    # R5.8-G (N10): the three explicit index axes
    assert torch.equal(
        sample.optimizer_denoise_indices,
        torch.arange(contract.ft_denoising_steps, dtype=torch.long),
    )
    assert torch.equal(
        sample.scheduler_timesteps,
        policy.noise_scheduler.timesteps[-contract.ft_denoising_steps:]
        .long(),
    )
    assert torch.equal(
        sample.full_chain_indices,
        torch.arange(
            contract.denoising_steps - contract.ft_denoising_steps,
            contract.denoising_steps, dtype=torch.long,
        ),
    )


def test_executed_is_first_exec_steps_of_policy_action():
    contract = _contract()
    policy = FakePolicy(contract)
    _, _, selected = _env_and_selected(contract)
    sample = sample_dppo_rollout(policy, selected, contract)
    a0 = contract.action_start
    assert torch.allclose(
        sample.executed_action,
        sample.full_trajectory[:, a0 : a0 + contract.exec_steps, :],
        atol=1e-6,
    )
    assert torch.allclose(
        sample.executed_action,
        sample.policy_action[:, : contract.exec_steps, :],
        atol=1e-6,
    )


def test_chains_window_matches_full_chain():
    """Region window == full chain[K-K_f:] action-region slice (deterministic)."""
    contract = _contract()
    policy = FakePolicy(contract)
    _, _, selected = _env_and_selected(contract)
    torch.manual_seed(7)
    sample = sample_dppo_rollout(policy, selected, contract)
    torch.manual_seed(7)
    x_t, _, chain = policy.sample_ddpm_with_logprob(obs_dict=selected)
    full = torch.stack(chain)  # (K+1, 1, H_f, 11)
    a0 = contract.action_start
    k_f = contract.ft_denoising_steps
    expected = full[contract.denoising_steps - k_f :, :, a0 : a0 + contract.action_horizon]
    assert torch.allclose(sample.chains_policy, expected.permute(1, 0, 2, 3), atol=1e-6)
    assert torch.allclose(sample.full_trajectory, x_t, atol=1e-6)


def test_logprobs_match_per_dim_anchor():
    """Rollout logprobs == per_dim recompute on the same window at the SAME
    CFG lambda (1.5, the fake's default) with the adapter's uncond — the
    fake-ratio hazard is exactly a mismatch between the rollout λ and the
    recompute λ, so the anchor test pins both at 1.5."""
    contract = _contract()
    policy = FakePolicy(contract)
    _, _, selected = _env_and_selected(contract)
    sample = sample_dppo_rollout(policy, selected, contract)
    _, uncond = _obs_conditions(policy, selected)
    logps = per_dim_denoising_logprobs(
        chain=sample.chains_policy,
        timesteps=policy.noise_scheduler.timesteps,
        model=policy.model,
        global_cond=sample.obs_condition,
        global_cond_uncond=uncond,
        cfg_lambda=1.5,
        scheduler=policy.noise_scheduler,
        eta=1.0,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        ft_denoising_steps=contract.ft_denoising_steps,
    )
    assert torch.allclose(sample.old_logprobs, logps, atol=1e-6)


def test_obs_condition_matches_encoder():
    contract = _contract()
    policy = FakePolicy(contract)
    _, _, selected = _env_and_selected(contract)
    sample = sample_dppo_rollout(policy, selected, contract)
    assert torch.allclose(
        sample.obs_condition, PolicyObsEncoder(policy)(selected), atol=1e-6
    )


def test_to_rollout_transition_passes_buffer_invariants():
    """Two consecutive transitions (rolling-history mock) pass all 6 invariants."""
    contract = _contract()
    policy = FakePolicy(contract)
    critic = ValueCritic(contract=contract, encoder=PolicyObsEncoder(policy))
    env, _, selected = _env_and_selected(contract)
    buffer = RolloutBuffer(contract)

    for step in range(2):
        sample = sample_dppo_rollout(policy, selected, contract)
        with torch.no_grad():
            value_old = float(critic(selected).reshape(-1)[0].item())
        executed_np = (
            policy.normalizer["action"]
            .unnormalize(sample.executed_action.detach())
            .numpy()
        )
        next_np, reward, terminated, truncated, _ = env.step(executed_np)
        next_t = to_tensor_obs(next_np)
        next_selected, _ = select_latest_observations(next_t, contract.obs_steps)
        buffer.add(
            to_rollout_transition(
                sample=sample,
                obs_np={k: v[0].numpy() for k, v in selected.items()},
                next_obs_np={k: v[0].numpy() for k, v in next_selected.items()},
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                value_old=value_old,
                episode_id="ep-0",
                episode_step=step,
                world_uid="mock",
                role="anticipatory",
                manifest_entry_hash="mh-mock",
                episode_config_hash="ch-mock",
                observation_valid=True,
                contract=contract,
            )
        )
        selected = next_selected
    assert len(buffer) == 2
    # the executed action is the physical 11d action in (1, A_e, 11) layout
    assert executed_np.shape == (1, contract.exec_steps, 11)


def test_ratio_one_on_own_chains():
    """make_get_new_logprobs on the rollout's own chains reproduces stored
    old_logprobs (ratio == 1 before any update)."""
    contract = _contract()
    policy = FakePolicy(contract)
    _, _, selected = _env_and_selected(contract)
    sample = sample_dppo_rollout(policy, selected, contract)
    get = make_get_new_logprobs(policy, contract)
    for d in range(contract.ft_denoising_steps):
        logp = get(
            selected,
            sample.chains_policy[:, d],
            sample.chains_policy[:, d + 1],
            torch.full((1,), d, dtype=torch.long),
        )
        assert torch.allclose(logp, sample.old_logprobs[:, d], atol=1e-5), f"step {d}"


def test_denoising_indices_layout():
    """R5.8-G (N10): the ambiguous descending denoising_indices field is
    gone — the three explicit axes replace it (optimizer index ascending,
    scheduler timesteps = the policy scheduler's own last K_f, full-chain
    index ascending)."""
    contract = _contract()
    policy = FakePolicy(contract)
    _, _, selected = _env_and_selected(contract)
    sample = sample_dppo_rollout(policy, selected, contract)
    assert sample.optimizer_denoise_indices.dtype == torch.long
    assert torch.equal(
        sample.optimizer_denoise_indices,
        torch.arange(contract.ft_denoising_steps, dtype=torch.long),
    )
    assert torch.equal(
        sample.full_chain_indices,
        torch.arange(
            contract.denoising_steps - contract.ft_denoising_steps,
            contract.denoising_steps, dtype=torch.long,
        ),
    )
