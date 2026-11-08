#!/usr/bin/env python3
"""Build the complete E5 DPPO object graph without rollout or training."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def _mount(staging_root: Path) -> None:
    import diffusion_policy_3d
    import diffusion_policy_3d.rl

    package = staging_root.resolve() / "diffusion_policy_3d"
    diffusion_policy_3d.__path__.append(str(package))
    diffusion_policy_3d.rl.__path__.append(str(package / "rl"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("vendor_root", type=Path)
    parser.add_argument("--staging-overlay", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    _mount(args.staging_overlay)
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system

    config_path = (
        args.staging_overlay
        / "diffusion_policy_3d"
        / "rl"
        / "dppo_uav_rebuild"
        / "config"
        / "train_e5_dppo_v1.yaml"
    )
    config = OmegaConf.load(config_path)
    config.checkpoint = str(args.checkpoint.resolve())
    config.vendor_root = str(args.vendor_root.resolve())
    config.device = args.device
    system = build_system(config, args.staging_overlay.resolve())

    actor_parameters = list(system.lora_bundle.named_adapter_parameters())
    result = {
        "actor_trainable_parameters": sum(value.numel() for _, value in actor_parameters),
        "critic_trainable_parameters": sum(
            value.numel() for value in system.critic.parameters() if value.requires_grad
        ),
        "lora_target_count": len(system.lora_bundle.target_names),
        "official_runtime": type(system.official_runtime).__module__
        + "."
        + type(system.official_runtime).__name__,
        "reward_module": system.reward.identity.implementation,
        "reward_source": system.reward.identity.source_path,
        "status": "PASS",
        "vendor_overlay_files": list(system.vendor_identity.overlay_files),
    }
    if result["lora_target_count"] != 47:
        raise RuntimeError("formal denoiser-only LoRA target count must be 47")
    if not actor_parameters:
        raise RuntimeError("actor optimizer would receive no LoRA parameters")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
