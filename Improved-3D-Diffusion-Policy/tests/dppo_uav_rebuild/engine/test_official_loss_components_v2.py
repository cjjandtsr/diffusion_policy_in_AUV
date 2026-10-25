"""Unreduced components must reduce exactly like frozen official DPPO."""

from pathlib import Path
from types import SimpleNamespace

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.official_loss_components import (
    AdvantageStats,
    compute_official_components,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.symbols import load_official_symbols


def test_unreduced_components_reduce_to_official_loss():
    root = Path(__file__).resolve().parents[3]
    symbols = load_official_symbols(root / "third_party/dppo_official_cc7234a")
    n = 128
    fresh = torch.linspace(-0.02, 0.03, n * 16 * 11).reshape(n, 16, 11)

    class Critic:
        def __call__(self, obs):
            return obs["state"][:, :1] * 0.2

    class Runtime:
        device = torch.device("cpu")
        norm_adv = True
        clip_advantage_lower_quantile = 0.05
        clip_advantage_upper_quantile = 0.95
        ft_denoising_steps = 10
        gamma_denoising = 0.8
        clip_ploss_coef_base = 1e-3
        clip_ploss_coef = 0.2
        clip_ploss_coef_rate = 3.0
        clip_vloss_coef = 0.2
        critic = Critic()

        def get_logprobs_subsample(self, *args, **kwargs):
            return fresh, torch.full((n, 11), 0.3)

    runtime = Runtime()
    batch = SimpleNamespace(
        condition={"state": torch.linspace(-1, 1, n).reshape(n, 1)},
        chains_prev=torch.zeros(n, 16, 11), chains_next=torch.zeros(n, 16, 11),
        denoising_inds=torch.arange(n).remainder(10),
        returns=torch.linspace(-0.4, 0.6, n), oldvalues=torch.linspace(-0.2, 0.2, n),
        advantages=torch.linspace(-2, 3, n), oldlogprobs=torch.zeros(n, 16, 11),
    )
    stats = AdvantageStats.from_tensor(batch.advantages, 0.05, 0.95)
    parts = compute_official_components(runtime, batch, stats)
    official = symbols.PPODiffusion.loss(
        runtime, batch.condition, batch.chains_prev, batch.chains_next,
        batch.denoising_inds, batch.returns, batch.oldvalues,
        batch.advantages.clone(), batch.oldlogprobs,
        use_bc_loss=False, reward_start=2, reward_horizon=4,
    )
    torch.testing.assert_close(parts.policy.mean(), official[0], atol=1e-8, rtol=0)
    torch.testing.assert_close(parts.value.mean() * 0.5, official[2], atol=1e-8, rtol=0)
    assert float(parts.approx_kl.mean()) == official[4]
    assert float(parts.ratio.mean()) == official[5]
