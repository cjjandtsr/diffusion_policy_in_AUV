import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.symbols import (
    load_official_symbols,
)
from scripts.dppo_credit_support.credit_probe import (
    analytic_reverse_kernel_kl,
    reverse_kernel_kl_record,
    reverse_kernel_parameters,
)


def test_equal_variance_analytic_kl_matches_closed_form():
    current = {
        "mean": torch.tensor([[1.0, -1.0]]),
        "variance": torch.tensor([[2.0, 2.0]]),
    }
    reference = {
        "mean": torch.zeros(1, 2),
        "variance": torch.tensor([[2.0, 2.0]]),
    }
    actual = analytic_reverse_kernel_kl(current, reference, denoising_index=3)
    assert torch.allclose(actual, torch.tensor([[0.25, 0.25]]))


def test_same_kernel_has_zero_kl_and_invalid_variance_is_rejected():
    kernel = {"mean": torch.randn(2, 3), "variance": torch.ones(2, 3)}
    actual = analytic_reverse_kernel_kl(kernel, kernel, denoising_index=0)
    assert torch.equal(actual, torch.zeros_like(actual))
    with pytest.raises(ValueError, match="strictly positive"):
        analytic_reverse_kernel_kl(
            kernel,
            {"mean": kernel["mean"], "variance": torch.zeros(2, 3)},
            denoising_index=0,
        )


def test_reverse_kernel_kl_record_accepts_time_shared_vendor_variance():
    batch_size, horizon, action_dim = 2, 3, 2

    class BroadcastVarianceRuntime:
        ddim_t = torch.arange(10)
        ft_denoising_steps = 10
        ddim_steps = 10
        min_logprob_denoising_std = 0.1

        def p_mean_var(
            self, x, timesteps, *, cond, index, use_base_policy
        ):
            mean = torch.zeros_like(x) if use_base_policy else torch.ones_like(x)
            shared_variance = torch.tensor([2.0, 4.0]).reshape(1, 1, action_dim)
            log_variance = shared_variance.log().expand(batch_size, 1, action_dim)
            return mean, log_variance, torch.ones_like(log_variance)

    batch = SimpleNamespace(
        chains_prev=torch.zeros(batch_size, horizon, action_dim),
        chains_next=torch.zeros(batch_size, horizon, action_dim),
        condition={"state": torch.zeros(batch_size, 1)},
        denoising_inds=torch.tensor([0, 1]),
        oldlogprobs=torch.zeros(batch_size, horizon, action_dim),
    )

    record = reverse_kernel_kl_record(BroadcastVarianceRuntime(), batch)

    assert record["mean"] == pytest.approx(0.1875)
    assert record["max"] == pytest.approx(0.25)
    assert record["sum_per_pair"] == pytest.approx([1.125, 1.125])
    assert record["variance_equal"] is True


def test_reverse_kernel_parameters_canonicalizes_official_ddim_variance_shape():
    vendor_root = os.environ.get("DPPO_VENDOR_ROOT")
    if not vendor_root:
        pytest.skip("DPPO_VENDOR_ROOT is required")
    symbols = load_official_symbols(Path(vendor_root).resolve())
    batch_size, horizon, action_dim = 128, 16, 11

    class Actor:
        def __init__(self, value):
            self.value = float(value)

        def __call__(self, x, timestep, cond):
            return torch.full_like(x, self.value)

    class Eta:
        def __call__(self, cond):
            return torch.ones(batch_size, action_dim)

    class OfficialHarness:
        actor = Actor(0.0)
        actor_ft = Actor(0.1)
        eta = Eta()
        use_ddim = True
        ddim_steps = 10
        ft_denoising_steps = 10
        ddim_t = torch.arange(9, -1, -1)
        ddim_alphas = torch.linspace(0.1, 0.8, 10)
        ddim_alphas_prev = ddim_alphas + 0.1
        ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas).sqrt()
        predict_epsilon = False
        denoised_clip_value = None
        recalculate_epsilon_after_clip = False
        eps_clip_value = None
        native_ddim_pow_compat = False
        native_ddim_cpu_scalar_compat = False
        min_logprob_denoising_std = 0.1

        def p_mean_var(self, *args, **kwargs):
            return symbols.VPGDiffusion.p_mean_var(self, *args, **kwargs)

    batch = SimpleNamespace(
        chains_prev=torch.zeros(batch_size, horizon, action_dim),
        condition={"state": torch.zeros(batch_size, 1)},
        denoising_inds=torch.arange(batch_size) % 10,
    )

    runtime = OfficialHarness()
    current = reverse_kernel_parameters(
        runtime, batch, use_base_policy=False
    )
    reference = reverse_kernel_parameters(
        runtime, batch, use_base_policy=True
    )

    assert current["mean"].shape == (batch_size, horizon, action_dim)
    assert current["variance"].shape == (batch_size, horizon, action_dim)
    torch.testing.assert_close(
        current["variance"][:, 0], current["variance"][:, -1]
    )

    indices = batch.denoising_inds
    timesteps = runtime.ddim_t[indices]
    raw_current_mean, raw_current_log_variance, _ = runtime.p_mean_var(
        batch.chains_prev,
        timesteps,
        cond=batch.condition,
        index=indices,
        use_base_policy=False,
    )
    raw_reference_mean, raw_reference_log_variance, _ = runtime.p_mean_var(
        batch.chains_prev,
        timesteps,
        cond=batch.condition,
        index=indices,
        use_base_policy=True,
    )
    raw_current_variance = raw_current_log_variance.exp().clamp_min(0.1**2)
    raw_reference_variance = raw_reference_log_variance.exp().clamp_min(0.1**2)
    sample = torch.zeros_like(batch.chains_prev)
    raw_logprob = torch.distributions.Normal(
        raw_current_mean, raw_current_variance.sqrt()
    ).log_prob(sample)
    canonical_logprob = torch.distributions.Normal(
        current["mean"], current["variance"].sqrt()
    ).log_prob(sample)
    torch.testing.assert_close(raw_logprob, canonical_logprob, rtol=0.0, atol=0.0)

    raw_kl = 0.5 * (
        torch.log(raw_reference_variance / raw_current_variance)
        + (
            raw_current_variance
            + (raw_current_mean - raw_reference_mean).square()
        )
        / raw_reference_variance
        - 1.0
    )
    canonical_kl = analytic_reverse_kernel_kl(
        current, reference, denoising_index=0
    )
    torch.testing.assert_close(
        raw_kl.expand_as(canonical_kl), canonical_kl, rtol=0.0, atol=0.0
    )


def test_reverse_kernel_parameters_rejects_collapsed_batch_variance():
    batch_size, horizon, action_dim = 2, 3, 2

    class CollapsedBatchRuntime:
        ddim_t = torch.arange(10)
        ft_denoising_steps = 10
        ddim_steps = 10
        min_logprob_denoising_std = 0.1

        def p_mean_var(
            self, x, timesteps, *, cond, index, use_base_policy
        ):
            return (
                torch.zeros_like(x),
                torch.zeros(1, 1, action_dim),
                torch.ones(1, 1, action_dim),
            )

    batch = SimpleNamespace(
        chains_prev=torch.zeros(batch_size, horizon, action_dim),
        condition={"state": torch.zeros(batch_size, 1)},
        denoising_inds=torch.tensor([0, 1]),
    )

    with pytest.raises(ValueError, match="preserve the batch axis"):
        reverse_kernel_parameters(
            CollapsedBatchRuntime(), batch, use_base_policy=False
        )
