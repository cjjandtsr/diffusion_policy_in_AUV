#!/usr/bin/env python3
"""Real-checkpoint LoRA inventory and zero-delta behavior check."""

import argparse
import json
from pathlib import Path

import torch


def _mount(staging):
    if staging is None:
        return
    import diffusion_policy_3d
    import diffusion_policy_3d.rl

    package = staging.resolve() / "diffusion_policy_3d"
    diffusion_policy_3d.__path__.append(str(package))
    diffusion_policy_3d.rl.__path__.append(str(package / "rl"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--staging-overlay", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    args = parser.parse_args()
    _mount(args.staging_overlay)
    from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.lora import (
        E5_DENOISER_CONV1D_TARGETS,
        E5_DENOISER_LINEAR_TARGETS,
        E5_FROZEN_ENCODER_CONV1D_MODULES,
        E5_FROZEN_ENCODER_LINEAR_MODULES,
        install_lora,
    )
    from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.native_loader import (
        load_native_e5,
    )

    runtime = load_native_e5(args.checkpoint, device=args.device)
    model = runtime.policy.model
    device = runtime.device
    generator = torch.Generator(device=device).manual_seed(20260827)
    sample = torch.randn(1, 16, 11, generator=generator, device=device)
    condition = torch.randn(1, 576, generator=generator, device=device)
    timestep = torch.tensor([45], device=device, dtype=torch.long)
    with torch.no_grad():
        before = model(
            sample=sample,
            timestep=timestep,
            local_cond=None,
            global_cond=condition,
        )
    bundle = install_lora(model, rank=args.rank, alpha=args.alpha)
    with torch.no_grad(), bundle.mode(True):
        after = model(
            sample=sample,
            timestep=timestep,
            local_cond=None,
            global_cond=condition,
        )
    if not torch.equal(before, after):
        raise RuntimeError(
            "zero LoRA changed E5 output; max_abs_diff={}".format(
                float((before - after).abs().max())
            )
        )
    print(
        json.dumps(
            {
                "identity_hash": bundle.identity.content_hash,
                "target_scope": "policy.model denoiser only",
                "linear_targets": E5_DENOISER_LINEAR_TARGETS,
                "conv1d_targets": E5_DENOISER_CONV1D_TARGETS,
                "frozen_encoder_linear_modules": E5_FROZEN_ENCODER_LINEAR_MODULES,
                "frozen_encoder_conv1d_modules": E5_FROZEN_ENCODER_CONV1D_MODULES,
                "p5_full_policy_linear_modules": (
                    E5_DENOISER_LINEAR_TARGETS
                    + E5_FROZEN_ENCODER_LINEAR_MODULES
                ),
                "p5_full_policy_conv1d_modules": (
                    E5_DENOISER_CONV1D_TARGETS
                    + E5_FROZEN_ENCODER_CONV1D_MODULES
                ),
                "target_count": len(bundle.target_names),
                "trainable_parameters": bundle.identity.trainable_parameters,
                "zero_lora_max_abs_diff": 0.0,
                "status": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
