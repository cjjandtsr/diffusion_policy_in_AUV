"""Construct one E5 + official DPPO system from verified components."""

from dataclasses import dataclass
import importlib
from pathlib import Path

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.lora import (
    E5_DENOISER_CONV1D_TARGETS,
    E5_DENOISER_LINEAR_TARGETS,
    install_lora,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.rollout_sampler import (
    DDPMRolloutSampler,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.native_loader import (
    load_native_e5,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.trainer import (
    DDPMTrainer,
    TrainerConfig,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.value import TemporalValueNetwork
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.initialization_parity import (
    construct_r1_critic_with_parity,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_adapter import (
    E5ObservationAdapter,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.symbols import (
    load_official_symbols,
)


@dataclass(frozen=True)
class BuiltSystem:
    bc_runtime: object
    official_runtime: object
    lora_bundle: object
    critic: object
    actor_optimizer: object
    critic_optimizer: object
    trainer: object
    observation_adapter: object
    rollout_sampler: object
    reward: object
    vendor_identity: object
    initialization_parity: object = None


def build_system(config, repo_root: Path):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.select import select_reward
    reward = select_reward(config, repo_root)
    device = torch.device(config.device)
    vendor_root = (repo_root / config.vendor_root).resolve()
    checkpoint = (repo_root / config.checkpoint).resolve()
    symbols = load_official_symbols(vendor_root)
    adapter_module = importlib.import_module("uav_idp3.e5_runtime")
    bc_runtime = load_native_e5(checkpoint, device=str(device))
    policy = bc_runtime.policy
    if policy.dtype != torch.float32:
        raise RuntimeError("formal E5 DDPM RL requires FP32")
    lora_bundle = install_lora(
        policy.model,
        rank=int(config.lora.rank),
        alpha=float(config.lora.alpha),
        expected_linear=E5_DENOISER_LINEAR_TARGETS,
        expected_conv1d=E5_DENOISER_CONV1D_TARGETS,
    )
    initialization_parity = None
    if str(config.reward_version) == "v2":
        critic, initialization_parity = construct_r1_critic_with_parity(
            int(config.critic.hidden_dim), device,
            lora_bundle.named_adapter_parameters(),
            episode_context=config.get("task_protocol") is not None,
        )
    else:
        critic = TemporalValueNetwork(
            history=3, frame_dim=192, hidden_dim=int(config.critic.hidden_dim),
        ).to(device)
    official_runtime = adapter_module.E5PPODiffusion(
        model=policy.model,
        scheduler=policy.noise_scheduler,
        critic=critic,
        device=device,
        lora_bundle=lora_bundle,
        num_inference_steps=10,
        ft_denoising_steps=10,
        horizon_steps=16,
        action_dim=11,
        obs_dim=576,
        min_sampling_denoising_std=float(config.dppo.min_sampling_std),
        min_logprob_denoising_std=float(config.dppo.min_logprob_std),
        gamma_denoising=float(config.dppo.gamma_denoising),
        clip_ploss_coef=float(config.dppo.clip_ploss_coef),
    )
    actor_parameters = [
        parameter for _, parameter in lora_bundle.named_adapter_parameters()
    ]
    actor_optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=float(config.optimizer.actor_lr),
        weight_decay=float(config.optimizer.weight_decay),
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(config.optimizer.critic_lr),
        weight_decay=float(config.optimizer.weight_decay),
    )
    trainer = DDPMTrainer(
        runtime=official_runtime,
        lora_bundle=lora_bundle,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        config=TrainerConfig(
            value_loss_coefficient=float(config.dppo.value_loss_coefficient),
            entropy_loss_coefficient=float(config.dppo.entropy_loss_coefficient),
            reward_start=2,
            reward_horizon=8 if str(config.reward_version) == "v2" else 4,
            max_approx_kl=float(config.gates.max_approx_kl),
            ratio_mean_min=float(config.gates.ratio_mean_min),
            ratio_mean_max=float(config.gates.ratio_mean_max),
            max_grad_norm=float(config.gates.max_grad_norm),
        ),
    )
    return BuiltSystem(
        bc_runtime=bc_runtime,
        official_runtime=official_runtime,
        lora_bundle=lora_bundle,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        trainer=trainer,
        observation_adapter=E5ObservationAdapter(
            policy,
            device,
            max_no_ray_hit_fraction=float(
                getattr(config.environment, "max_no_ray_hit_fraction", 0.99)
            ),
            reject_all_no_ray_hit=getattr(
                config.environment, "reject_all_no_ray_hit", True
            ),
        ),
        rollout_sampler=DDPMRolloutSampler(official_runtime, policy.normalizer),
        reward=reward,
        vendor_identity=symbols.identity,
        initialization_parity=initialization_parity,
    )
