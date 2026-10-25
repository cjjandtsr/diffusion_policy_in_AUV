#!/usr/bin/env python
"""Shared native E5 loader (P3/P4/P5).

Constructs the policy exactly as the native workspace does:
    policy = hydra.utils.instantiate(cfg.policy)
then loads state_dicts['model'] (the weight source used by both
scripts/eval_ddpm_bc_baseline.py:147 and rl/dppo_uav/policy_loader.py:273).
"""
import os
import sys

REPO = "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
sys.path.insert(0, REPO)

import torch
import hydra
import omegaconf

CKPT = os.path.join(
    REPO,
    "data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt",
)


def load_ckpt(path=CKPT):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_policy(ckpt, device="cuda:0"):
    """Native construction: hydra instantiate of cfg.policy, like iDP3Workspace."""
    cfg = ckpt["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    policy.to(device)
    policy.eval()
    return policy


def load_weights(policy, ckpt, which="model", strict=False):
    state_dicts = ckpt["state_dicts"]
    sd = state_dicts[which]
    missing, unexpected = policy.load_state_dict(sd, strict=strict)
    return missing, unexpected
