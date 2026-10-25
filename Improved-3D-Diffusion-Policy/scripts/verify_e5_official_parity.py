#!/usr/bin/env python3
"""Compare native E5 DDIM sampling with the official DPPO E5 subclass."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
from torch import nn


def _mount_staging_overlay(staging_root: Path) -> None:
    import diffusion_policy_3d
    import diffusion_policy_3d.rl

    policy_overlay = staging_root.resolve() / "diffusion_policy_3d"
    rl_overlay = policy_overlay / "rl"
    diffusion_policy_3d.__path__.append(str(policy_overlay))
    diffusion_policy_3d.rl.__path__.append(str(rl_overlay))


class _UnusedCritic(nn.Module):
    def forward(self, obs):
        return torch.zeros(
            next(iter(obs.values())).shape[0], 1, device=next(self.parameters()).device
        )

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("vendor_root", type=Path)
    parser.add_argument("--staging-overlay", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _mount_staging_overlay(args.staging_overlay)
    from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.native_loader import (
        load_native_e5,
    )
    from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.symbols import (
        load_official_symbols,
    )

    load_official_symbols(args.vendor_root)
    adapter = importlib.import_module("uav_idp3.e5_runtime")
    native = load_native_e5(args.checkpoint, device=args.device)
    policy = native.policy
    device = native.device
    dtype = policy.dtype

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
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    global_cond = torch.randn(
        1,
        policy.n_obs_steps * policy.obs_feature_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    condition_data = torch.zeros(
        1, policy.horizon, policy.action_dim, device=device, dtype=dtype
    )
    condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
    cond = {"state": global_cond, "global_cond": global_cond}

    torch.manual_seed(args.seed)
    native_sample = policy.conditional_sample(
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=global_cond,
    )
    torch.manual_seed(args.seed)
    official_sample = runtime(
        cond=cond,
        deterministic=True,
        return_chain=True,
        use_base_policy=False,
    )
    actual = official_sample.trajectories
    torch.testing.assert_close(actual, native_sample, rtol=args.rtol, atol=args.atol)

    difference = (actual - native_sample).abs()
    start = policy.n_obs_steps - 1
    end = start + policy.n_action_steps
    result = {
        "action_dim": policy.action_dim,
        "action_slice": [start, end],
        "atol": args.atol,
        "chain_shape": list(official_sample.chains.shape),
        "full_shape": list(actual.shape),
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
        "num_inference_steps": policy.num_inference_steps,
        "official_class": type(runtime).__module__ + "." + type(runtime).__name__,
        "rtol": args.rtol,
        "seed": args.seed,
        "status": "PASS",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
