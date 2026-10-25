"""Fail-closed construction of the official ReinFlow × native iDP3 FM core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Optional, Tuple

import torch
from torch import nn

from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.lora import install_lora

from .checkpoint_adapter import EXPECTED_POLICY_TARGET, load_native_fm
from .official_core.symbols import load_official_symbols
from .policy_bridge import IDP3FlowPolicyBridge
from .runtime import PPOFlowBuildConfig, build_idp3_ppo_flow


DEFAULT_WSL_REPO_ROOT = Path(
    "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
)
VENDOR_RELATIVE_PATH = Path("third_party/reinflow_official_e722e15")
EXPECTED_REWARD_V13_SHA256 = (
    "ff22c1385289a736ad9b2fa5ccd1aedc630a083df29e19b276534af02a05429d"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PreflightError(RuntimeError):
    """Raised when construction inputs lack required provenance."""


@dataclass(frozen=True)
class PreflightInputs:
    checkpoint: Path
    checkpoint_sha256: str
    vendor_root: Path
    reward_source_sha256: str


@dataclass(frozen=True)
class LoRAInventory:
    linear_names: Tuple[str, ...]
    conv1d_names: Tuple[str, ...]

    @property
    def linear_count(self) -> int:
        return len(self.linear_names)

    @property
    def conv1d_count(self) -> int:
        return len(self.conv1d_names)

    @property
    def content_hash(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SystemBuildConfig:
    preflight: PreflightInputs
    device: str
    expected_policy_target: str
    lora_rank: int
    lora_alpha: float
    expected_linear_targets: int
    expected_conv1d_targets: int
    actor_lr: float
    critic_lr: float
    weight_decay: float
    critic_hidden_dim: int
    ppo: PPOFlowBuildConfig


@dataclass(frozen=True)
class BuiltReinFlowCore:
    native_runtime: object
    official_symbols: object
    official_runtime: object
    bridge: IDP3FlowPolicyBridge
    lora_bundle: object
    lora_inventory: LoRAInventory
    critic: nn.Module
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    reward: object


def resolve_default_wsl_vendor() -> Path:
    return DEFAULT_WSL_REPO_ROOT / VENDOR_RELATIVE_PATH


def validate_preflight(inputs: PreflightInputs) -> PreflightInputs:
    checkpoint = Path(inputs.checkpoint)
    vendor = Path(inputs.vendor_root)
    if not checkpoint.is_file():
        raise PreflightError(f"checkpoint is missing: {checkpoint}")
    if _SHA256_RE.fullmatch(inputs.checkpoint_sha256 or "") is None:
        raise PreflightError("checkpoint_sha256 must be a lowercase SHA256")
    if not vendor.is_dir():
        raise PreflightError(f"official ReinFlow vendor is missing: {vendor}")
    if inputs.reward_source_sha256 != EXPECTED_REWARD_V13_SHA256:
        raise PreflightError(
            "reward_source_sha256 must equal the admitted Reward v1.3 source hash"
        )
    return PreflightInputs(
        checkpoint=checkpoint.resolve(),
        checkpoint_sha256=inputs.checkpoint_sha256,
        vendor_root=vendor.resolve(),
        reward_source_sha256=inputs.reward_source_sha256,
    )


def inventory_lora_targets(model: nn.Module) -> LoRAInventory:
    linear = []
    conv1d = []
    for name, module in model.named_modules():
        if not name:
            continue
        if type(module) is nn.Linear:
            linear.append(name)
        elif type(module) is nn.Conv1d:
            conv1d.append(name)
    return LoRAInventory(tuple(sorted(linear)), tuple(sorted(conv1d)))


class EncodedConditionValue(nn.Module):
    """Fresh critic over the frozen ordered FM encoder features."""

    def __init__(self, *, history: int, frame_dim: int, hidden_dim: int):
        super().__init__()
        if history <= 0 or frame_dim <= 0 or hidden_dim <= 0:
            raise ValueError("critic dimensions must be positive")
        self.history = int(history)
        self.frame_dim = int(frame_dim)
        self.frame_norm = nn.LayerNorm(frame_dim)
        self.position = nn.Parameter(torch.zeros(history, frame_dim))
        nn.init.normal_(self.position, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(history * frame_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, condition):
        value = condition.get("global_cond", condition.get("state"))
        expected = self.history * self.frame_dim
        if value is None or tuple(value.shape[1:]) != (expected,):
            raise ValueError(f"critic condition must have shape (B,{expected})")
        frames = value.reshape(-1, self.history, self.frame_dim).detach()
        ordered = self.frame_norm(frames) + self.position.unsqueeze(0)
        return self.mlp(ordered.reshape(ordered.shape[0], -1))


def build_core_system(config: SystemBuildConfig) -> BuiltReinFlowCore:
    """Build in provenance order and return no environment/training side effects."""

    preflight = validate_preflight(config.preflight)
    if config.expected_linear_targets < 0 or config.expected_conv1d_targets < 0:
        raise PreflightError(
            "LoRA target counts must come from the real-checkpoint inventory probe"
        )
    symbols = load_official_symbols(preflight.vendor_root)
    native_runtime = load_native_fm(
        preflight.checkpoint,
        expected_sha256=preflight.checkpoint_sha256,
        device=config.device,
        expected_policy_target=config.expected_policy_target,
    )
    policy = native_runtime.policy

    try:
        from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v13.facade import (
            RewardV13Facade,
        )

        reward = RewardV13Facade()
    except Exception as error:
        raise PreflightError("Reward v1.3 implementation could not be loaded") from error
    if reward.identity.source_sha256 != preflight.reward_source_sha256:
        raise PreflightError(
            "Reward v1.3 source hash mismatch: "
            f"expected {preflight.reward_source_sha256}, "
            f"got {reward.identity.source_sha256}"
        )

    model = getattr(policy, "model", None)
    if model is None:
        raise PreflightError("native FM policy has no velocity model")
    inventory = inventory_lora_targets(model)
    expected_counts = (
        int(config.expected_linear_targets),
        int(config.expected_conv1d_targets),
    )
    if (inventory.linear_count, inventory.conv1d_count) != expected_counts:
        raise PreflightError(
            "LoRA inventory mismatch: "
            f"expected {expected_counts}, got "
            f"{(inventory.linear_count, inventory.conv1d_count)}"
        )
    original_lora = install_lora(
        model,
        rank=int(config.lora_rank),
        alpha=float(config.lora_alpha),
        expected_linear=inventory.linear_count,
        expected_conv1d=inventory.conv1d_count,
    )
    frame_dim = int(getattr(policy, "obs_feature_dim", 0))
    history = native_runtime.identity.n_obs_steps
    expected_obs_dim = history * frame_dim
    if config.ppo.cond_steps != history or config.ppo.obs_dim != expected_obs_dim:
        raise PreflightError(
            "PPO condition config does not match checkpoint runtime: "
            f"expected cond_steps={history}, obs_dim={expected_obs_dim}; "
            f"got cond_steps={config.ppo.cond_steps}, obs_dim={config.ppo.obs_dim}"
        )
    bridge = IDP3FlowPolicyBridge(
        policy,
        lora_bundle=original_lora,
        lora_enabled=False,
        cond_enc_dim=frame_dim,
    )
    critic = EncodedConditionValue(
        history=history,
        frame_dim=frame_dim,
        hidden_dim=int(config.critic_hidden_dim),
    ).to(torch.device(config.device))
    official_runtime = build_idp3_ppo_flow(
        symbols, bridge, critic, config.ppo
    )
    fine_tuned_bridge = official_runtime.actor_ft.policy
    fine_tuned_lora = fine_tuned_bridge.__dict__.get("_lora_bundle")
    if fine_tuned_lora is None:
        raise PreflightError("official policy copy lost the LoRA identity bundle")
    adapter_ids = {
        id(parameter)
        for _, parameter in fine_tuned_lora.named_adapter_parameters()
    }
    bridge_adapter_ids = {
        id(parameter)
        for name, parameter in fine_tuned_bridge.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }
    if not adapter_ids or adapter_ids != bridge_adapter_ids:
        raise PreflightError("fine-tuned LoRA bundle does not own official actor parameters")

    actor_parameters = [
        parameter
        for parameter in official_runtime.actor_ft.parameters()
        if parameter.requires_grad
    ]
    if not actor_parameters:
        raise PreflightError("official actor has no trainable parameters")
    actor_optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=float(config.actor_lr),
        weight_decay=float(config.weight_decay),
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(config.critic_lr),
        weight_decay=float(config.weight_decay),
    )
    return BuiltReinFlowCore(
        native_runtime=native_runtime,
        official_symbols=symbols,
        official_runtime=official_runtime,
        bridge=bridge,
        lora_bundle=fine_tuned_lora,
        lora_inventory=inventory,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        reward=reward,
    )


__all__ = [
    "DEFAULT_WSL_REPO_ROOT",
    "EXPECTED_REWARD_V13_SHA256",
    "BuiltReinFlowCore",
    "EncodedConditionValue",
    "LoRAInventory",
    "PreflightError",
    "PreflightInputs",
    "SystemBuildConfig",
    "build_core_system",
    "inventory_lora_targets",
    "resolve_default_wsl_vendor",
    "validate_preflight",
]
