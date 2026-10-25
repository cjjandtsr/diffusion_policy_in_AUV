"""K5: EVERY fine-tune denoise index (k=0..4) must produce a finite,
non-zero LoRA gradient on controlled non-degenerate fixtures (review
R5.1 / §8.1). The old kernel returned zero logprob for the last two
transitions — this test is the regression gate that would have caught
F1 at the unit level.

Fixtures: a small MLP "epsilon model" with real LoRA adapters, driven
through the official kernel over 3 different observation conditions.
Per fixture: run the full 10-step chain, then per k in the FT window
recompute transition_stats with grad enabled and check
|| d logp_k / d LoRA || > 0 and finite. CPU-only.
"""

import torch

from diffusion_policy_3d.policy.dppo_transition_kernel import (
    DPPOTransitionKernel,
    build_ddim_tables,
)
from diffusion_policy_3d.policy.lora_adapters import insert_lora

B, R, Da = 1, 8, 11
DDIM_STEPS = 10
FT_STEPS = 5
MIN_SAMPLING_STD = 0.1
MIN_LOGPROB_STD = 0.1
NUM_FIXTURES = 3

torch.manual_seed(0)


class _EpsMLP(torch.nn.Module):
    """Tiny epsilon predictor whose Linear layers carry UNet-style names
    so the production insert_lora target filter applies (TARGET_PATTERNS:
    down_modules/mid_modules/up_modules/final_conv)."""

    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.cond_proj = torch.nn.Linear(cond_dim, 64)
        self.mid_modules = torch.nn.ModuleList([
            torch.nn.Linear(R * Da, 64),
            torch.nn.Linear(64, 64),
        ])
        self.final_conv = torch.nn.Linear(64, R * Da)

    def forward(self, sample, timestep=None, local_cond=None,
                global_cond=None):
        h = self.mid_modules[0](sample.reshape(B, -1))
        if global_cond is not None:
            h = h + self.cond_proj(global_cond)
        h = torch.relu(self.mid_modules[1](h))
        return self.final_conv(h).reshape(B, R, Da)


def _make_fixture(seed: int):
    """LoRA'd model + kernel + a sampled chain + condition."""
    torch.manual_seed(seed)
    model = _EpsMLP(cond_dim=32)
    lora_info = insert_lora(model, r=4, alpha=4, verbose=False)
    scheduler = _scheduler()
    kernel = DPPOTransitionKernel(
        build_ddim_tables(scheduler.alphas_cumprod, scheduler.timesteps),
        ft_denoising_steps=FT_STEPS,
    )
    cond = torch.randn(B, 32)
    x = torch.randn(B, R, Da)
    chain = [x.clone()]
    noises = []
    for k in range(DDIM_STEPS):
        eps = model(sample=x, global_cond=cond)
        index_b = torch.full((B,), k, dtype=torch.long)
        stats = kernel.transition_stats(
            x, eps, index_b, MIN_SAMPLING_STD, MIN_LOGPROB_STD
        )
        noise = torch.randn_like(x)
        noises.append(noise)
        x = kernel.sample(stats, noise)
        chain.append(x.clone())
    return model, lora_info, kernel, cond, chain, noises


def _scheduler():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    s = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    s.set_timesteps(DDIM_STEPS)
    return s


def test_all_denoise_indices_have_lora_gradient():
    """K5: per-index LoRA gradient coverage over 3 fixtures."""
    ft_start = DDIM_STEPS - FT_STEPS
    for fixture in range(NUM_FIXTURES):
        model, lora_info, kernel, cond, chain, _ = _make_fixture(
            seed=7 + fixture
        )
        lora_params = list(lora_info["lora_params"])
        for k in range(ft_start, DDIM_STEPS):
            x_k = chain[k].detach().requires_grad_(True)
            x_next = chain[k + 1].detach()
            model.zero_grad()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            eps = model(sample=x_k, global_cond=cond)
            index_b = torch.full((B,), k, dtype=torch.long)
            stats = kernel.transition_stats(
                x_k, eps, index_b, MIN_SAMPLING_STD, MIN_LOGPROB_STD
            )
            logp_k = kernel.log_prob(stats, x_next).mean()
            logp_k.backward()
            grad_norm = 0.0
            for p in lora_params:
                assert p.grad is not None, (
                    f"fixture {fixture} k={k}: a LoRA parameter has no "
                    "gradient — dead transition (F1 symptom)"
                )
                g = p.grad.detach().float()
                assert torch.isfinite(g).all(), (
                    f"fixture {fixture} k={k}: non-finite LoRA grad"
                )
                grad_norm += float((g ** 2).sum().item())
            grad_norm = grad_norm ** 0.5
            assert grad_norm > 0.0, (
                f"fixture {fixture} optimizer-denoise-index "
                f"{k - ft_start}: zero LoRA gradient — the transition "
                "carries no policy gradient (K5 FAIL)"
            )
