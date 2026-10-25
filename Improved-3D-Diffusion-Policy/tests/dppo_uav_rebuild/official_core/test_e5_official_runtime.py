import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.symbols import (
    load_official_symbols,
)


def _load_adapter():
    value = os.environ.get("DPPO_VENDOR_ROOT")
    if not value:
        pytest.skip("DPPO_VENDOR_ROOT is required")
    root = Path(value).resolve()
    symbols = load_official_symbols(root)
    module = importlib.import_module("uav_idp3.e5_runtime")
    return symbols, module


class _SamplePredictor(nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.prediction = prediction
        self.last = None

    def forward(self, *, sample, timestep, local_cond, global_cond):
        self.last = (sample, timestep, local_cond, global_cond)
        return self.prediction.expand_as(sample)


def test_sample_prediction_is_converted_to_epsilon():
    _, adapter = _load_adapter()
    alpha = torch.tensor([0.81, 0.64, 0.49])
    scheduler = SimpleNamespace(
        config=SimpleNamespace(prediction_type="sample"),
        alphas_cumprod=alpha,
    )
    x_t = torch.tensor([[[0.7]], [[-0.2]]])
    x_zero = torch.tensor([[[0.1]], [[0.3]]])
    timestep = torch.tensor([0, 2], dtype=torch.long)

    actual = adapter.model_output_to_epsilon(
        x_t, x_zero, timestep, scheduler
    )
    alpha_t = alpha[timestep].reshape(2, 1, 1)
    expected = (x_t - alpha_t.sqrt() * x_zero) / (1.0 - alpha_t).sqrt()
    torch.testing.assert_close(actual, expected)


def test_e5_backend_uses_only_native_global_condition():
    _, adapter = _load_adapter()
    prediction = torch.tensor(0.25)
    model = _SamplePredictor(prediction)
    scheduler = SimpleNamespace(
        config=SimpleNamespace(prediction_type="epsilon"),
        alphas_cumprod=torch.linspace(0.99, 0.1, 50),
    )
    backend = adapter.E5ActorBackend(
        model=model,
        scheduler=scheduler,
        lora_bundle=None,
        lora_enabled=False,
    )
    sample = torch.zeros(2, 8, 11)
    timestep = torch.tensor([45, 45], dtype=torch.long)
    global_cond = torch.randn(2, 576)

    actual = backend(
        sample,
        timestep,
        cond={"state": global_cond, "global_cond": global_cond},
    )

    torch.testing.assert_close(actual, torch.full_like(sample, 0.25))
    assert model.last[2] is None
    assert model.last[3] is global_cond


def test_e5_backend_preserves_native_sample_prediction_coordinates():
    _, adapter = _load_adapter()
    prediction = torch.tensor(0.25)
    model = _SamplePredictor(prediction)
    scheduler = SimpleNamespace(
        config=SimpleNamespace(prediction_type="sample"),
        alphas_cumprod=torch.linspace(0.99, 0.1, 50),
    )
    backend = adapter.E5ActorBackend(
        model=model,
        scheduler=scheduler,
        lora_bundle=None,
        lora_enabled=False,
    )
    sample = torch.zeros(2, 8, 11)
    actual = backend(
        sample,
        torch.tensor([45, 45], dtype=torch.long),
        cond={"global_cond": torch.randn(2, 576)},
    )

    torch.testing.assert_close(actual, torch.full_like(sample, 0.25))


def test_runtime_inherits_official_sampling_and_ppo_math():
    symbols, adapter = _load_adapter()

    assert issubclass(adapter.E5PPODiffusion, symbols.PPODiffusion)
    assert adapter.E5PPODiffusion.forward is symbols.VPGDiffusion.forward
    assert adapter.E5PPODiffusion.p_mean_var is symbols.VPGDiffusion.p_mean_var
    assert adapter.E5PPODiffusion.get_logprobs is symbols.VPGDiffusion.get_logprobs
    assert (
        adapter.E5PPODiffusion.get_logprobs_subsample
        is symbols.VPGDiffusion.get_logprobs_subsample
    )
    assert adapter.E5PPODiffusion.loss is symbols.PPODiffusion.loss


def test_official_loss_can_credit_e5_executed_slice_2_to_6():
    symbols, _ = _load_adapter()
    batch, horizon, action_dim = 2, 16, 11
    fresh = torch.full(
        (batch, horizon, action_dim), 1e-4, requires_grad=True
    )

    class _Critic:
        def __call__(self, obs):
            return torch.zeros(batch, 1)

    class _Harness:
        device = torch.device("cpu")
        norm_adv = False
        clip_advantage_lower_quantile = 0.0
        clip_advantage_upper_quantile = 1.0
        ft_denoising_steps = 10
        gamma_denoising = 0.8
        clip_ploss_coef_base = 1e-3
        clip_ploss_coef = 0.2
        clip_ploss_coef_rate = 3.0
        clip_vloss_coef = None
        critic = _Critic()

        def get_logprobs_subsample(self, *args, **kwargs):
            return fresh, torch.ones(batch, action_dim)

    outputs = symbols.PPODiffusion.loss(
        _Harness(),
        obs={"state": torch.zeros(batch, 576)},
        chains_prev=torch.zeros(batch, horizon, action_dim),
        chains_next=torch.zeros(batch, horizon, action_dim),
        denoising_inds=torch.tensor([0, 1]),
        returns=torch.zeros(batch),
        oldvalues=torch.zeros(batch),
        advantages=torch.tensor([1.0, 2.0]),
        oldlogprobs=torch.zeros(batch, horizon, action_dim),
        reward_start=2,
        reward_horizon=4,
    )
    outputs[0].backward()

    assert torch.count_nonzero(fresh.grad[:, :2]) == 0
    assert torch.count_nonzero(fresh.grad[:, 2:6]) > 0
    assert torch.count_nonzero(fresh.grad[:, 6:]) == 0


def test_e5_clipping_preserves_native_ddim_epsilon():
    symbols, _ = _load_adapter()
    alpha = torch.tensor(0.5)
    alpha_previous = torch.tensor(0.8)
    x_t = torch.tensor([[[0.3]]])
    predicted_x_zero = torch.tensor(2.0)
    epsilon = (
        x_t - alpha.sqrt() * predicted_x_zero
    ) / (1.0 - alpha).sqrt()

    class _Actor:
        def __call__(self, x, timestep, cond):
            return epsilon.expand_as(x).clone()

    class _Harness:
        actor = _Actor()
        actor_ft = actor
        use_ddim = True
        ddim_steps = 1
        ft_denoising_steps = 1
        predict_epsilon = True
        ddim_alphas = alpha.reshape(1)
        ddim_alphas_prev = alpha_previous.reshape(1)
        ddim_sqrt_one_minus_alphas = (1.0 - alpha).sqrt().reshape(1)
        denoised_clip_value = 1.0
        recalculate_epsilon_after_clip = False
        eps_clip_value = None

    mean, _, _ = symbols.VPGDiffusion.p_mean_var(
        _Harness(),
        x=x_t,
        t=torch.tensor([0]),
        cond={"state": torch.zeros(1, 1)},
        index=torch.tensor([0]),
        deterministic=True,
    )
    expected = alpha_previous.sqrt() + (
        1.0 - alpha_previous
    ).sqrt() * epsilon
    torch.testing.assert_close(mean, expected, rtol=1e-6, atol=1e-6)


def test_official_direct_xzero_branch_uses_derived_epsilon_for_direction():
    symbols, _ = _load_adapter()
    alpha = torch.tensor(0.5)
    alpha_previous = torch.tensor(0.8)
    x_t = torch.tensor([[[0.3]]])
    predicted_x_zero = torch.tensor([[[0.2]]])
    epsilon = (
        x_t - alpha.sqrt() * predicted_x_zero
    ) / (1.0 - alpha).sqrt()

    class _Actor:
        def __call__(self, x, timestep, cond):
            return predicted_x_zero.expand_as(x).clone()

    class _Harness:
        actor = _Actor()
        actor_ft = actor
        use_ddim = True
        ddim_steps = 1
        ft_denoising_steps = 1
        predict_epsilon = False
        ddim_alphas = alpha.reshape(1)
        ddim_alphas_prev = alpha_previous.reshape(1)
        ddim_sqrt_one_minus_alphas = (1.0 - alpha).sqrt().reshape(1)
        denoised_clip_value = 1.0
        recalculate_epsilon_after_clip = False
        eps_clip_value = None

    mean, _, _ = symbols.VPGDiffusion.p_mean_var(
        _Harness(),
        x=x_t,
        t=torch.tensor([0]),
        cond={"state": torch.zeros(1, 1)},
        index=torch.tensor([0]),
        deterministic=True,
    )
    expected = (
        alpha_previous.sqrt() * predicted_x_zero
        + (1.0 - alpha_previous).sqrt() * epsilon
    )
    torch.testing.assert_close(mean, expected, rtol=1e-6, atol=1e-6)
