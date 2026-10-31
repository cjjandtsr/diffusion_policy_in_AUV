#!/usr/bin/env python3
"""Load the frozen E5 checkpoint through the canonical native runtime."""

import argparse
import json
from pathlib import Path


DEFAULT_E5_CHECKPOINT_RELPATH = (
    "data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt"
)


def _load_runtime_api(staging_overlay):
    if staging_overlay is not None:
        overlay = staging_overlay.resolve()
        import diffusion_policy_3d
        import diffusion_policy_3d.rl

        policy_overlay = overlay / "diffusion_policy_3d"
        rl_overlay = policy_overlay / "rl"
        if not (rl_overlay / "dppo_uav_rebuild").is_dir():
            raise RuntimeError(
                "staging overlay has no diffusion_policy_3d/rl/dppo_uav_rebuild"
            )
        diffusion_policy_3d.__path__.append(str(policy_overlay))
        diffusion_policy_3d.rl.__path__.append(str(rl_overlay))

    from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.native_loader import (
        load_native_e5,
    )

    return load_native_e5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--staging-overlay",
        type=Path,
        help="Pre-copy verification only: root containing the staged new package",
    )
    parser.add_argument(
        "--checkpoint-relpath",
        default=DEFAULT_E5_CHECKPOINT_RELPATH,
    )
    args = parser.parse_args()

    load_native_e5 = _load_runtime_api(args.staging_overlay)
    runtime = load_native_e5(
        checkpoint_path=args.checkpoint,
        checkpoint_relpath=args.checkpoint_relpath,
        device=args.device,
    )
    total_parameters = sum(parameter.numel() for parameter in runtime.policy.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in runtime.policy.parameters()
        if parameter.requires_grad
    )
    print(
        json.dumps(
            {
                "action_contract_sha256": runtime.identity.action_spec.content_hash,
                "bc_identity_sha256": runtime.identity.content_hash,
                "checkpoint": str(runtime.checkpoint_path),
                "checkpoint_sha256": runtime.identity.checkpoint_sha256,
                "config_sha256": runtime.identity.config_sha256,
                "device": str(runtime.device),
                "normalizer_sha256": runtime.identity.normalizer_sha256,
                "observation_contract_sha256": (
                    runtime.identity.observation_spec.content_hash
                ),
                "policy_target": runtime.identity.policy_target,
                "scheduler_target": runtime.identity.scheduler_target,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "weight_source": runtime.identity.weight_source,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
