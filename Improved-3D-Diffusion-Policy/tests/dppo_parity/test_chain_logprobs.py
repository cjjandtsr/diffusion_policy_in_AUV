"""T5 closure: per-dim per-step denoising logprobs (design §4.12, chain window).

[T5 known-gap candidate] dp_sampler.compute_per_step_logp_from_chain sums over
all D = H x Da dims and returns (B, ft); the buffer invariant needs per-dim
(K_f, H_a, D_a) (rollout_buffer.py invariant 5). per_dim_denoising_logprobs
recomputes the SAME mu/sigma math per dimension; parity tests anchor it to the
dp_sampler summed function on the same chains (same timesteps/model/cond).

Window semantics (task_plan decision 5): chain (B, ft+1, R, Da) holds the last
ft+1 denoising steps of the full chain; step i maps to full-step
k = len(timesteps) - ft + i. R5.1 (review F1): every transition —
including the last one INTO valid timestep 0 — is stochastic under the
official kernel (floored std), never a zero-logp terminal step.
"""

import numpy as np
import pytest
import torch

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.policy.dp_sampler import _model_output_to_epsilon
from diffusion_policy_3d.policy.dppo_transition_kernel import (
    compute_per_step_logp_from_chain,
)
from diffusion_policy_3d.rl.dppo_uav.chain_logprobs import (
    per_dim_denoising_logprobs,
    per_step_denoising_logp,
    selected_step_logprobs,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder

from fake_policy import FakePolicy, to_tensor_obs


class _CondModel(torch.nn.Module):
    """Epsilon predictor that depends on global_cond (CFG branch test)."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        gc = global_cond.mean(dim=1, keepdim=True).unsqueeze(-1)
        return self.base(sample, timestep, local_cond, global_cond) + 0.5 * gc


def _obs_batch(contract: RLSystemContract, b: int = 2, seed: int = 0):
    rng = np.random.RandomState(seed)
    obs = {
        "point_cloud": rng.uniform(
            -8.0, 8.0, (b, contract.obs_steps, contract.point_count, 3)
        ).astype(np.float32),
        "agent_pos": rng.uniform(-1.0, 1.0, (b, contract.obs_steps, 9)).astype(
            np.float32
        ),
    }
    return to_tensor_obs(obs)


def _make_chain(contract: RLSystemContract, b: int = 2, seed: int = 0):
    """Real DDIM chain (FakePolicy) + region window (B, K_f+1, H_a, 11)."""
    torch.manual_seed(seed)
    policy = FakePolicy(contract)
    obs_b = _obs_batch(contract, b=b, seed=seed)
    _, _, chain = policy.sample_ddpm_with_logprob(obs_dict=obs_b)
    full = torch.stack(chain)  # (K+1, B, H_f, 11)
    a0 = contract.action_start
    window = full[
        contract.denoising_steps - contract.ft_denoising_steps :, :,
        a0 : a0 + contract.action_horizon,
    ].permute(1, 0, 2, 3)  # (B, K_f+1, H_a, 11)
    enc = PolicyObsEncoder(policy)(obs_b)  # (B, T_o*192)
    return policy, obs_b, full, window, enc


def _per_dim(contract, window, policy, enc, eta=1.0, cfg_lambda=0.0, gc_u=None):
    return per_dim_denoising_logprobs(
        chain=window,
        timesteps=policy.noise_scheduler.timesteps,
        model=policy.model,
        global_cond=enc,
        global_cond_uncond=gc_u,
        cfg_lambda=cfg_lambda,
        scheduler=policy.noise_scheduler,
        eta=eta,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        ft_denoising_steps=contract.ft_denoising_steps,
    )


def test_per_dim_shapes():
    contract = RLSystemContract()
    policy, _, _, window, enc = _make_chain(contract)
    logps = _per_dim(contract, window, policy, enc)
    assert logps.shape == (
        2,
        contract.ft_denoising_steps,
        contract.action_horizon,
        contract.action_dim,
    ), logps.shape


def test_parity_full_chain_against_dp_sampler():
    """per-dim sums == dp_sampler summed anchor, full-chain window (B=2)."""
    contract = RLSystemContract()
    policy, _, full, _, enc = _make_chain(contract, b=2)
    k_f = contract.ft_denoising_steps
    window = full[contract.denoising_steps - k_f :].permute(1, 0, 2, 3)
    logps = _per_dim(contract, window, policy, enc)
    anchor = compute_per_step_logp_from_chain(
        chain=[full[k] for k in range(full.shape[0])],
        timesteps=policy.noise_scheduler.timesteps,
        model=policy.model,
        global_cond=enc,
        global_cond_uncond=None,
        cfg_lambda=0.0,
        scheduler=policy.noise_scheduler,
        eta=1.0,
        sigma_clip_min=0.01,
        sigma_clip_max=10.0,
        ft_denoising_steps=k_f,
    )  # (B, ft)
    assert torch.allclose(logps.sum(dim=(2, 3)), anchor, rtol=1e-5, atol=1e-6), (
        "per-dim logps must sum to the dp_sampler anchor"
    )


def test_parity_region_window():
    """Update-time shape: region-sliced window (B, K_f+1, H_a, 11), B=1."""
    contract = RLSystemContract()
    policy, _, _, window, enc = _make_chain(contract, b=1)
    k_f = contract.ft_denoising_steps
    logps = _per_dim(contract, window, policy, enc)
    anchor = compute_per_step_logp_from_chain(
        chain=[window[:, k] for k in range(window.shape[1])],
        timesteps=policy.noise_scheduler.timesteps[-k_f:],
        model=policy.model,
        global_cond=enc,
        global_cond_uncond=None,
        cfg_lambda=0.0,
        scheduler=policy.noise_scheduler,
        eta=1.0,
        sigma_clip_min=0.01,
        sigma_clip_max=10.0,
    )  # (1, ft)
    assert torch.allclose(logps.sum(dim=(2, 3)), anchor, rtol=1e-5, atol=1e-6)


def test_eta_nonone_hard_fails():
    """The official kernel is eta=1; deterministic-DDIM (eta=0) is not a
    silent-ignore parameter — it hard-fails (review R5.1)."""
    contract = RLSystemContract()
    policy, _, _, window, enc = _make_chain(contract)
    with pytest.raises(ValueError):
        _per_dim(contract, window, policy, enc, eta=0.0)


def test_last_step_into_valid_timestep_0_is_stochastic():
    """The transition INTO valid timestep 0 is a REAL stochastic transition
    under the official kernel (floored std -> non-zero logprob) — the old
    t_prev>0 terminal branch zeroed it (review F1)."""
    contract = RLSystemContract()
    policy, _, _, window, enc = _make_chain(contract)
    logps = _per_dim(contract, window, policy, enc)
    assert torch.isfinite(logps).all()
    assert not torch.allclose(
        logps[:, -1], torch.zeros_like(logps[:, -1]), atol=1e-12
    ), "last window step (into timestep 0) must carry a floored-std logprob"


def test_selected_step_logprobs_gathers_per_element():
    """selected_step_logprobs picks the (B,) denoising index per element."""
    contract = RLSystemContract()
    policy, _, _, window, enc = _make_chain(contract)
    logps = _per_dim(contract, window, policy, enc)
    inds = torch.tensor([0, 3], dtype=torch.long)
    sel = selected_step_logprobs(logps, inds)
    assert sel.shape == (2, contract.action_horizon, contract.action_dim)
    assert torch.allclose(sel[0], logps[0, 0], atol=1e-6)
    assert torch.allclose(sel[1], logps[1, 3], atol=1e-6)


def test_per_step_matches_window_slice():
    """per_step (update-time building block) == window column of per_dim."""
    contract = RLSystemContract()
    policy, _, _, window, enc = _make_chain(contract)
    k_f = contract.ft_denoising_steps
    logps = _per_dim(contract, window, policy, enc)
    for i in range(k_f):
        k_batch = torch.full((2,), contract.denoising_steps - k_f + i, dtype=torch.long)
        one = per_step_denoising_logp(
            x_k=window[:, i],
            x_next=window[:, i + 1],
            k_batch=k_batch,
            timesteps=policy.noise_scheduler.timesteps,
            model=policy.model,
            global_cond=enc,
            global_cond_uncond=None,
            cfg_lambda=0.0,
            scheduler=policy.noise_scheduler,
            eta=1.0,
            min_sampling_denoising_std=0.1,
            min_logprob_denoising_std=0.1,
            ft_denoising_steps=k_f,
        )
        assert torch.allclose(one, logps[:, i], atol=1e-6), f"window step {i}"


def test_model_output_to_epsilon_sample_prediction_batched():
    """prediction_type='sample' with batched t must broadcast (B,1,1).

    T5 GPU catch: the real BC policy's scheduler predicts 'sample'; the
    update batch (B=20) hit the missing broadcast dims in the Tweedie
    conversion (B=1 in no-update mode passed by accident). The scalar-t
    path (E040 sampling) must stay identical.
    """
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="sample",
        clip_sample=False,
    )
    torch.manual_seed(3)
    x_t = torch.randn(3, 8, 11)
    x0 = torch.randn(3, 8, 11)
    t = torch.tensor([5, 10, 20], dtype=torch.long)
    eps = _model_output_to_epsilon(x_t, x0, t, scheduler, "cpu")
    ab = scheduler.alphas_cumprod[t]
    expected = (x_t - torch.sqrt(ab[:, None, None]) * x0) / torch.sqrt(
        1.0 - ab[:, None, None] + 1e-8
    )
    assert torch.allclose(eps, expected, atol=1e-6)
    # scalar-t path unchanged (frozen E040 sampling path)
    eps_s = _model_output_to_epsilon(x_t, x0, 10, scheduler, "cpu")
    ab_s = scheduler.alphas_cumprod[10]
    expected_s = (x_t - torch.sqrt(ab_s) * x0) / torch.sqrt(
        1.0 - ab_s + 1e-8
    )
    assert torch.allclose(eps_s, expected_s, atol=1e-6)


def test_cfg_branch_changes_logprobs():
    """cfg_lambda > 0 with an uncond branch must use the CFG epsilon path."""
    contract = RLSystemContract()
    policy, _, _, window, enc = _make_chain(contract)
    policy.model = _CondModel(policy.model)
    base = _per_dim(contract, window, policy, enc, cfg_lambda=0.0)
    gc_u = torch.zeros_like(enc)
    cfg = _per_dim(contract, window, policy, enc, cfg_lambda=0.7, gc_u=gc_u)
    assert torch.isfinite(cfg).all()
    assert not torch.allclose(cfg, base, atol=1e-6), (
        "CFG-combined epsilon must differ from the cond-only epsilon"
    )
