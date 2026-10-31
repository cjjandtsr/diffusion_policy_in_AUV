#!/usr/bin/env python3
"""Isolate the first native-vs-official E5 DDIM transition difference."""

import argparse
import importlib
import json
from pathlib import Path

import torch
from torch import nn


def _mount(staging_root: Path) -> None:
    import diffusion_policy_3d
    import diffusion_policy_3d.rl

    package = staging_root.resolve() / "diffusion_policy_3d"
    diffusion_policy_3d.__path__.append(str(package))
    diffusion_policy_3d.rl.__path__.append(str(package / "rl"))


class _UnusedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, obs):
        return torch.zeros(next(iter(obs.values())).shape[0], 1, device=self.anchor.device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("vendor_root", type=Path)
    parser.add_argument("--staging-overlay", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    _mount(args.staging_overlay)
    from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.native_loader import load_native_e5
    from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.symbols import load_official_symbols

    load_official_symbols(args.vendor_root)
    adapter = importlib.import_module("uav_idp3.e5_runtime")
    native = load_native_e5(args.checkpoint, device=args.device)
    policy = native.policy
    device = native.device
    runtime = adapter.E5PPODiffusion(
        model=policy.model,
        scheduler=policy.noise_scheduler,
        critic=_UnusedCritic(),
        device=device,
        num_inference_steps=policy.num_inference_steps,
        ft_denoising_steps=policy.num_inference_steps,
        horizon_steps=policy.horizon,
        action_dim=policy.action_dim,
        obs_dim=policy.n_obs_steps * policy.obs_feature_dim,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    global_cond = torch.randn(1, 576, generator=generator, device=device)
    cond = {"state": global_cond, "global_cond": global_cond}
    policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
    timestep_scalar = policy.noise_scheduler.timesteps[0]
    timestep_batch = timestep_scalar.to(device).reshape(1)
    torch.manual_seed(args.seed)
    x_t = torch.randn(1, policy.horizon, policy.action_dim, device=device)

    with torch.no_grad():
        scalar_output = policy.model(
            sample=x_t,
            timestep=timestep_scalar,
            local_cond=None,
            global_cond=global_cond,
        )
        batch_output = policy.model(
            sample=x_t,
            timestep=timestep_batch,
            local_cond=None,
            global_cond=global_cond,
        )
        actor_output = runtime.actor_ft(x_t, timestep_batch, cond=cond)
        native_next = policy.noise_scheduler.step(
            scalar_output, timestep_scalar, x_t
        ).prev_sample
        official_mean, _, _ = runtime.p_mean_var(
            x=x_t,
            t=timestep_batch,
            cond=cond,
            index=torch.zeros(1, dtype=torch.long, device=device),
            use_base_policy=False,
            deterministic=True,
        )
        scheduler = policy.noise_scheduler
        previous_timestep = (
            int(timestep_scalar)
            - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        )
        alpha = scheduler.alphas_cumprod[timestep_scalar].to(device)
        alpha_previous = (
            scheduler.alphas_cumprod[previous_timestep].to(device)
            if previous_timestep >= 0
            else scheduler.final_alpha_cumprod.to(device)
        )
        beta = 1 - alpha
        predicted_xzero = scalar_output
        epsilon = (
            x_t - alpha ** 0.5 * predicted_xzero
        ) / beta ** 0.5
        if scheduler.config.clip_sample:
            predicted_xzero = predicted_xzero.clamp(
                -scheduler.config.clip_sample_range,
                scheduler.config.clip_sample_range,
            )
        manual_next = (
            alpha_previous ** 0.5 * predicted_xzero
            + (1 - alpha_previous) ** 0.5 * epsilon
        )
        alpha_cpu = scheduler.alphas_cumprod[timestep_scalar]
        alpha_previous_cpu = (
            scheduler.alphas_cumprod[previous_timestep]
            if previous_timestep >= 0
            else scheduler.final_alpha_cumprod
        )
        beta_cpu = 1 - alpha_cpu
        predicted_xzero_cpu_path = scalar_output
        epsilon_cpu_path = (
            x_t - alpha_cpu ** 0.5 * predicted_xzero_cpu_path
        ) / beta_cpu ** 0.5
        if scheduler.config.clip_sample:
            predicted_xzero_cpu_path = predicted_xzero_cpu_path.clamp(
                -scheduler.config.clip_sample_range,
                scheduler.config.clip_sample_range,
            )
        variance = scheduler._get_variance(
            timestep_scalar, previous_timestep
        )
        std_dev = 0.0 * variance ** 0.5
        manual_cpu_path = (
            alpha_previous_cpu ** 0.5 * predicted_xzero_cpu_path
            + (1 - alpha_previous_cpu - std_dev ** 2) ** 0.5
            * epsilon_cpu_path
        )

    print(json.dumps({
        "timestep": int(timestep_scalar),
        "scalar_vs_batch_model_max_abs_diff": float((scalar_output - batch_output).abs().max()),
        "scalar_vs_actor_ft_max_abs_diff": float((scalar_output - actor_output).abs().max()),
        "native_vs_manual_max_abs_diff": float((native_next - manual_next).abs().max()),
        "native_vs_manual_cpu_path_max_abs_diff": float((native_next - manual_cpu_path).abs().max()),
        "official_vs_manual_max_abs_diff": float((official_mean - manual_next).abs().max()),
        "native_vs_official_first_step_max_abs_diff": float((native_next - official_mean).abs().max()),
        "native_vs_official_first_step_mean_abs_diff": float((native_next - official_mean).abs().mean()),
        "native_next_max_abs": float(native_next.abs().max()),
        "official_next_max_abs": float(official_mean.abs().max()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
