"""Bind the native E5 DDIM actor to the upstream DPPO implementation.

The sampling transition, denoising log-probability, and PPO loss are inherited
unchanged from ``VPGDiffusion`` / ``PPODiffusion``.  This module only supplies
the attributes and actor signature expected by those upstream methods.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
from torch import nn

from model.diffusion.diffusion_ppo import PPODiffusion


class E5RuntimeError(RuntimeError):
    """Raised when E5 and upstream DPPO runtime contracts are incompatible."""


def model_output_to_epsilon(
    x_t: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    """Convert the native E5 scheduler prediction to epsilon coordinates."""
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return model_output
    if prediction_type != "sample":
        raise E5RuntimeError(
            "E5 DPPO supports scheduler prediction_type sample or epsilon; got {}".format(
                prediction_type
            )
        )

    timestep = torch.as_tensor(timestep, device=x_t.device, dtype=torch.long)
    alpha_bar = scheduler.alphas_cumprod.to(x_t.device)[timestep]
    if timestep.ndim > 0:
        alpha_bar = alpha_bar.reshape(
            (-1,) + (1,) * (x_t.ndim - 1)
        )
    denominator = (1.0 - alpha_bar).clamp_min(
        torch.finfo(x_t.dtype).eps
    ).sqrt()
    return (x_t - alpha_bar.sqrt() * model_output) / denominator


class E5ActorBackend(nn.Module):
    """Expose the native conditional U-Net through the DPPO actor signature."""

    def __init__(
        self,
        *,
        model: nn.Module,
        scheduler,
        lora_bundle: Optional[nn.Module],
        lora_enabled: bool,
        register_model: bool = True,
    ) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.lora_enabled = bool(lora_enabled)
        if register_model:
            self.model = model
            self.__dict__["_unregistered_model"] = None
        else:
            # The frozen and fine-tuned backends share one native E5 model.
            # Register it only under actor_ft to avoid duplicate state entries.
            self.__dict__["_unregistered_model"] = model
        if lora_bundle is not None and not hasattr(lora_bundle, "mode"):
            raise E5RuntimeError("lora_bundle must expose mode(enabled)")
        self.__dict__["_lora_bundle"] = lora_bundle

    def _model(self) -> nn.Module:
        registered = self._modules.get("model")
        if registered is not None:
            return registered
        model = self.__dict__.get("_unregistered_model")
        if model is None:
            raise E5RuntimeError("E5 actor model reference is missing")
        return model

    def _gate(self):
        bundle = self.__dict__.get("_lora_bundle")
        if bundle is None:
            return nullcontext()
        return bundle.mode(self.lora_enabled)

    def forward(self, x, timestep, cond):
        if "global_cond_uncond" in cond:
            raise E5RuntimeError("E5 checkpoint has no CFG/unconditional branch")
        global_cond = cond.get("global_cond")
        if global_cond is None:
            raise E5RuntimeError("DPPO condition must contain global_cond")
        global_cond = global_cond.to(device=x.device, dtype=x.dtype)
        with self._gate():
            model_output = self._model()(
                sample=x,
                timestep=timestep,
                local_cond=None,
                global_cond=global_cond,
            )
        if hasattr(model_output, "sample"):
            model_output = model_output.sample
        if tuple(model_output.shape) != tuple(x.shape):
            raise E5RuntimeError(
                "E5 model output shape {} does not match sample {}".format(
                    tuple(model_output.shape), tuple(x.shape)
                )
            )
        # Return the native prediction coordinates unchanged.  The inherited
        # official ``p_mean_var`` already has separate epsilon and direct-x0
        # branches, selected by ``E5PPODiffusion.predict_epsilon``.  Converting
        # x0 -> epsilon -> x0 here is algebraically equivalent but introduces
        # avoidable floating-point drift over the 10-step E5 DDIM chain.
        return model_output


class _OnesEta(nn.Module):
    """Fixed eta=1 required by the stochastic official DDIM likelihood."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)

    def forward(self, cond):
        first = next(iter(cond.values())) if isinstance(cond, dict) else cond
        return torch.ones(
            first.shape[0], self.action_dim, device=first.device, dtype=first.dtype
        )


class E5PPODiffusion(PPODiffusion):
    """Official DPPO math with native E5 model and scheduler state.

    This class deliberately does not override the five upstream algorithm
    methods: ``forward``, ``p_mean_var``, ``get_logprobs``,
    ``get_logprobs_subsample``, and ``loss``.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        scheduler,
        critic: nn.Module,
        device,
        lora_bundle: Optional[nn.Module] = None,
        num_inference_steps: int = 10,
        ft_denoising_steps: int = 10,
        horizon_steps: int = 16,
        action_dim: int = 11,
        obs_dim: int = 576,
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        gamma_denoising: float = 0.8,
        clip_ploss_coef: float = 0.2,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3.0,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0.0,
        clip_advantage_upper_quantile: float = 1.0,
        norm_adv: bool = True,
        denoised_clip_value: Optional[float] = None,
        randn_clip_value: float = 10.0,
    ) -> None:
        # Calling the upstream constructor would create a second model copy and
        # replace the E5 scheduler tables.  Initialize nn.Module, then bind the
        # exact attributes consumed by the inherited upstream methods.
        nn.Module.__init__(self)
        self.device = torch.device(device)
        self.horizon_steps = int(horizon_steps)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        if (self.horizon_steps, self.action_dim, self.obs_dim) != (16, 11, 576):
            raise E5RuntimeError("E5 runtime requires horizon=16, action=11, obs=576")

        scheduler.set_timesteps(
            num_inference_steps=int(num_inference_steps), device=self.device
        )
        ddim_t = scheduler.timesteps.to(self.device, dtype=torch.long)
        if ddim_t.ndim != 1 or len(ddim_t) != int(num_inference_steps):
            raise E5RuntimeError("scheduler produced an invalid DDIM timestep table")
        if len(ddim_t) > 1 and not torch.all(ddim_t[:-1] > ddim_t[1:]):
            raise E5RuntimeError("DDIM timesteps must be strictly descending")
        if not 1 <= int(ft_denoising_steps) <= len(ddim_t):
            raise E5RuntimeError("ft_denoising_steps must be within the DDIM chain")

        betas = scheduler.betas.to(self.device)
        alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
        self.register_buffer("betas", betas)
        self.register_buffer("ddim_t", ddim_t)
        self.register_buffer("ddim_alphas", alphas_cumprod[ddim_t])
        final_alpha = torch.as_tensor(
            scheduler.final_alpha_cumprod,
            device=self.device,
            dtype=alphas_cumprod.dtype,
        ).reshape(1)
        self.register_buffer(
            "ddim_alphas_prev",
            torch.cat((alphas_cumprod[ddim_t[1:]], final_alpha)),
        )
        self.register_buffer(
            "ddim_sqrt_one_minus_alphas",
            (1.0 - self.ddim_alphas).clamp_min(0.0) ** 0.5,
        )
        # Native diffusers keeps scheduler coefficients as CPU 0-d tensors.
        # Preserve those exact operands for deterministic BC/deployment parity;
        # stochastic rollout and PPO recomputation continue to use GPU buffers.
        ddim_t_cpu = ddim_t.detach().cpu()
        native_alphas_cpu = scheduler.alphas_cumprod.detach().cpu()
        native_final_alpha_cpu = torch.as_tensor(
            scheduler.final_alpha_cumprod,
            dtype=native_alphas_cpu.dtype,
        ).reshape(1)
        self.__dict__["native_ddim_alphas_cpu"] = native_alphas_cpu[ddim_t_cpu]
        self.__dict__["native_ddim_alphas_prev_cpu"] = torch.cat(
            (native_alphas_cpu[ddim_t_cpu[1:]], native_final_alpha_cpu)
        )
        self.__dict__["native_ddim_sqrt_one_minus_alphas_cpu"] = (
            1.0 - self.__dict__["native_ddim_alphas_cpu"]
        ) ** 0.5

        self.denoising_steps = int(len(betas))
        self.ddim_steps = int(len(ddim_t))
        self.use_ddim = True
        prediction_type = str(scheduler.config.prediction_type)
        if prediction_type not in ("epsilon", "sample"):
            raise E5RuntimeError(
                "unsupported native scheduler prediction_type: {}".format(
                    prediction_type
                )
            )
        # Official DPPO names the direct x0 path ``predict_epsilon=False``.
        # E5 is prediction_type=sample, so this preserves native coordinates
        # and avoids a numerically lossy x0 -> epsilon -> x0 round trip.
        self.predict_epsilon = prediction_type == "epsilon"
        native_clip_enabled = bool(getattr(scheduler.config, "clip_sample", False))
        native_clip_value = (
            float(getattr(scheduler.config, "clip_sample_range", 1.0))
            if native_clip_enabled
            else None
        )
        if denoised_clip_value is not None and denoised_clip_value != native_clip_value:
            raise E5RuntimeError(
                "denoised_clip_value must match the native E5 scheduler"
            )
        self.denoised_clip_value = native_clip_value
        self.recalculate_epsilon_after_clip = False
        self.native_ddim_pow_compat = True
        self.native_ddim_cpu_scalar_compat = True
        self.eps_clip_value = None
        self.randn_clip_value = float(randn_clip_value)
        self.final_action_clip_value = None

        self.ft_denoising_steps = int(ft_denoising_steps)
        self.ft_denoising_steps_d = 0
        self.ft_denoising_steps_t = 0
        self.ft_denoising_steps_cnt = 0
        self.min_sampling_denoising_std = float(min_sampling_denoising_std)
        self.min_logprob_denoising_std = float(min_logprob_denoising_std)
        self.learn_eta = False
        self.eta = _OnesEta(self.action_dim).to(self.device)

        model = model.to(self.device)
        self.actor = E5ActorBackend(
            model=model,
            scheduler=scheduler,
            lora_bundle=lora_bundle,
            lora_enabled=False,
            register_model=False,
        )
        self.actor_ft = E5ActorBackend(
            model=model,
            scheduler=scheduler,
            lora_bundle=lora_bundle,
            lora_enabled=True,
            register_model=True,
        )
        self.critic = critic.to(self.device)

        self.norm_adv = bool(norm_adv)
        self.clip_ploss_coef = float(clip_ploss_coef)
        self.clip_ploss_coef_base = float(clip_ploss_coef_base)
        self.clip_ploss_coef_rate = float(clip_ploss_coef_rate)
        self.clip_vloss_coef = clip_vloss_coef
        self.gamma_denoising = float(gamma_denoising)
        self.clip_advantage_lower_quantile = float(
            clip_advantage_lower_quantile
        )
        self.clip_advantage_upper_quantile = float(
            clip_advantage_upper_quantile
        )
