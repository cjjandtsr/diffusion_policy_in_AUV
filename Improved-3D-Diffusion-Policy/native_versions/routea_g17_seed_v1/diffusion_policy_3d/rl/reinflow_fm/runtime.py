"""Construction of an iDP3 bridge through the official PPOFlow constructor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .policy_bridge import IDP3FlowPolicyBridge


class OfficialRuntimeError(RuntimeError):
    """Raised when the official runtime cannot preserve the iDP3 contract."""


@dataclass(frozen=True)
class PPOFlowBuildConfig:
    device: str
    act_min: float
    act_max: float
    obs_dim: int
    cond_steps: int
    noise_scheduler_type: str
    inference_steps: int
    ft_denoising_steps: int
    randn_clip_value: float
    min_sampling_denoising_std: float
    min_logprob_denoising_std: float
    max_logprob_denoising_std: float
    logprob_min: float
    logprob_max: float
    clip_ploss_coef: float
    clip_ploss_coef_base: float
    clip_ploss_coef_rate: float
    clip_vloss_coef: float
    denoised_clip_value: float
    time_dim_explore: int
    learn_explore_time_embedding: bool
    use_time_independent_noise: bool
    noise_hidden_dims: Tuple[int, ...]
    logprob_debug_sample: bool
    logprob_debug_recalculate: bool
    explore_net_activation_type: str

    def __post_init__(self) -> None:
        if self.inference_steps != 4 or self.ft_denoising_steps != 4:
            raise OfficialRuntimeError("iDP3 ReinFlow requires all four FM steps")
        if self.cond_steps <= 0 or self.obs_dim <= 0:
            raise OfficialRuntimeError("condition dimensions must be positive")
        if self.min_logprob_denoising_std > self.max_logprob_denoising_std:
            raise OfficialRuntimeError("minimum logprob std exceeds maximum")
        if not self.use_time_independent_noise and not self.learn_explore_time_embedding:
            raise OfficialRuntimeError(
                "native bridge exposes scalar time only; use learned exploration time embedding"
            )


def _adapter_parameter_names(policy) -> list[str]:
    return [
        name
        for name, parameter in policy.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    ]


def build_idp3_ppo_flow(symbols, bridge, critic, config: PPOFlowBuildConfig):
    """Return an official ``PPOFlow`` subclass with only actor init adapted.

    Sampling, transition likelihood, entropy, PPO ratio/clip/KL, critic loss,
    and optional BC loss remain inherited official methods.
    """

    if not isinstance(bridge, IDP3FlowPolicyBridge):
        raise OfficialRuntimeError("bridge must be IDP3FlowPolicyBridge")
    if bridge.__dict__.get("_lora_bundle") is None:
        raise OfficialRuntimeError("official training runtime requires an installed LoRA bundle")
    official_ppo_flow = symbols.PPOFlow
    official_noisy_flow = symbols.NoisyFlowMLP

    class IDP3PPOFlow(official_ppo_flow):
        def init_actor_ft(self, policy_copy):
            if not isinstance(policy_copy, IDP3FlowPolicyBridge):
                raise OfficialRuntimeError("official PPOFlow copied an unexpected policy")
            policy_copy.set_lora_enabled(True)
            policy_copy.requires_grad_(False)
            names = _adapter_parameter_names(policy_copy)
            if not names:
                raise OfficialRuntimeError("fine-tuned bridge has no LoRA parameters")
            for name, parameter in policy_copy.named_parameters():
                if name.endswith(("lora_a", "lora_b")):
                    parameter.requires_grad_(True)
            self.actor_ft = official_noisy_flow(
                policy=policy_copy,
                denoising_steps=self.inference_steps,
                learn_explore_noise_from=self.inference_steps - self.ft_denoising_steps,
                inital_noise_scheduler_type=self.noise_scheduler_type,
                min_logprob_denoising_std=self.min_logprob_denoising_std,
                max_logprob_denoising_std=self.max_logprob_denoising_std,
                learn_explore_time_embedding=self.learn_explore_time_embedding,
                time_dim_explore=self.time_dim_explore,
                use_time_independent_noise=self.use_time_independent_noise,
                device=self.device,
                noise_hidden_dims=list(self.noise_hidden_dims),
                activation_type=self.explore_net_activation_type,
            )

    try:
        runtime = IDP3PPOFlow(
            device=config.device,
            policy=bridge,
            critic=critic,
            actor_policy_path=None,
            act_dim=bridge.action_dim,
            horizon_steps=bridge.horizon_steps,
            act_min=config.act_min,
            act_max=config.act_max,
            obs_dim=config.obs_dim,
            cond_steps=config.cond_steps,
            noise_scheduler_type=config.noise_scheduler_type,
            inference_steps=config.inference_steps,
            ft_denoising_steps=config.ft_denoising_steps,
            randn_clip_value=config.randn_clip_value,
            min_sampling_denoising_std=config.min_sampling_denoising_std,
            min_logprob_denoising_std=config.min_logprob_denoising_std,
            logprob_min=config.logprob_min,
            logprob_max=config.logprob_max,
            clip_ploss_coef=config.clip_ploss_coef,
            clip_ploss_coef_base=config.clip_ploss_coef_base,
            clip_ploss_coef_rate=config.clip_ploss_coef_rate,
            clip_vloss_coef=config.clip_vloss_coef,
            denoised_clip_value=config.denoised_clip_value,
            max_logprob_denoising_std=config.max_logprob_denoising_std,
            time_dim_explore=config.time_dim_explore,
            learn_explore_time_embedding=config.learn_explore_time_embedding,
            use_time_independent_noise=config.use_time_independent_noise,
            noise_hidden_dims=list(config.noise_hidden_dims),
            logprob_debug_sample=config.logprob_debug_sample,
            logprob_debug_recalculate=config.logprob_debug_recalculate,
            explore_net_activation_type=config.explore_net_activation_type,
        )
    except OfficialRuntimeError:
        raise
    except Exception as error:
        raise OfficialRuntimeError("official PPOFlow construction failed") from error

    if type(runtime).get_actions is not official_ppo_flow.get_actions:
        raise OfficialRuntimeError("get_actions is not inherited from official PPOFlow")
    if type(runtime).get_logprobs is not official_ppo_flow.get_logprobs:
        raise OfficialRuntimeError("get_logprobs is not inherited from official PPOFlow")
    if type(runtime).loss is not official_ppo_flow.loss:
        raise OfficialRuntimeError("loss is not inherited from official PPOFlow")
    return runtime


__all__ = [
    "OfficialRuntimeError",
    "PPOFlowBuildConfig",
    "build_idp3_ppo_flow",
]
