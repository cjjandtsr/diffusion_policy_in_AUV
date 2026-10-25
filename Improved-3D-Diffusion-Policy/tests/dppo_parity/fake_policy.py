"""CPU-test fixtures: FakePolicy (real DDIM sampling math, deterministic).

The fake policy mirrors phase1a_policy.py sample_ddpm_with_logprob's structure
(normalize -> encode -> DDIM chain via the legacy
dppo_transition_kernel.ddim_step_with_logprob shim) so that T5 wiring
tests exercise the same math the frozen policy runs at rollout.
The noise predictor is a differentiable deterministic nn.Module (eps = scale*x),
so gradients flow through the update path on CPU without a real UNet.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.policy.dppo_transition_kernel import (
    ddim_step_with_logprob,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract


class _IdentityTransform:
    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


class _ScaledTransform:
    """unnormalize multiplies by action_scale — the stand-in for the real
    action normalizer's std (normalized ~N(0,1) -> physical meter deltas)."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def normalize(self, x):
        return x / self.scale

    def unnormalize(self, x):
        return x * self.scale


class FakeNormalizer:
    """Identity-ish normalizer; dict-style access for ['action'] (real stack
    uses policy.normalizer['action'].unnormalize at the env boundary).

    action_scale mirrors the real action normalizer's std: the raw DDIM
    output is ~N(0,1) in normalized space and unnormalize maps it to
    physical deltas. Default 1.0 = legacy identity behavior (mock envs that
    ignore action magnitudes)."""

    def __init__(self, action_scale: float = 1.0) -> None:
        self._t = _IdentityTransform()
        self._action = _ScaledTransform(action_scale)

    def normalize(self, x):
        if isinstance(x, dict):
            return {k: v for k, v in x.items()}
        return x

    def __getitem__(self, key: str):
        return self._action if key == "action" else self._t


class FakeObsEncoder(nn.Module):
    """(B*F, ...) per-frame tensors -> (B*F, 192); deterministic + differentiable.

    Interface mirrors the real policy obs_encoder (per-frame 192d output).
    """

    def __init__(self, frame_dim: int = 192) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.proj = nn.Linear(12, frame_dim)

    def forward(self, frames: Dict[str, torch.Tensor]) -> torch.Tensor:
        pc = frames["point_cloud"]  # (B*F, N, 3)
        ap = frames["agent_pos"]  # (B*F, 9)
        feats = torch.cat([pc.mean(dim=1), ap], dim=1)  # (B*F, 12)
        return self.proj(feats)


class FakeModel(nn.Module):
    """Deterministic epsilon predictor: eps = scale * x + cond_proj(cond).

    R1 rework #2 (audit 2026-08-08): the model must CONSUME global_cond so
    CFG mixing is numerically live in the fake sampler — the stock model
    ignoring conditions made mixed == pure-cond and CPU contract tests could
    not discriminate the CFG branch from a dead one (the pre-#2
    CondAwareModel in test_cfg_contract existed exactly for this gap)."""

    def __init__(self, contract: RLSystemContract) -> None:
        super().__init__()
        self.contract = contract
        self.scale = nn.Parameter(torch.tensor(0.05))
        self.cond_proj = nn.Linear(contract.obs_steps * 192, 1)

    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        # (B, T, Da) -> (B, T, Da); timestep (B,) long, ignored
        out = sample * self.scale
        if global_cond is not None:
            out = out + self.cond_proj(global_cond).unsqueeze(1)
        return out


class FakePolicy:
    """Phase1aPolicy-shaped object using the REAL dp_sampler step math.

    Attributes consumed by the T5 wiring (mirror the frozen policy):
    horizon, action_dim, n_obs_steps, num_inference_steps, use_pc_color,
    device, dtype, normalizer, obs_encoder, model, noise_scheduler,
    sample_ddpm_with_logprob(obs_dict, eta, sigma_clip_min, sigma_clip_max,
    num_inference_steps, ft_denoising_steps).
    """

    def __init__(
        self,
        contract: RLSystemContract,
        action_scale: float = 1.0,
        cfg_lambda: float = 1.5,
    ) -> None:
        self.contract = contract
        self.horizon = contract.full_horizon
        self.action_dim = contract.action_dim
        self.n_obs_steps = contract.obs_steps
        self.num_inference_steps = contract.denoising_steps
        self.use_pc_color = False
        self.device = "cpu"
        self.dtype = torch.float32
        # CFG contract: the FORMAL default is 1.5 — the adapter resolves
        # cfg_lambda from the policy and hard-fails (ValueError) on
        # anything but 1.5, so CPU contract tests exercise the same
        # CFG-mixed path as production. Pure-cond contrast experiments
        # set cfg_lambda=0.0 explicitly and call the low-level
        # per_dim_denoising_logprobs API directly (never the formal
        # adapters, which are pinned to the policy lambda).
        self.cfg_lambda = cfg_lambda
        self.normalizer = FakeNormalizer(action_scale=action_scale)
        self.obs_encoder = FakeObsEncoder()
        self.model = FakeModel(contract)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=False,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Delegate to the model — mirrors train_loop._policy_state_dict's
        FakePolicy fallback (checkpoint stores the model's state), so the
        save and load paths round-trip the same object."""
        return self.model.state_dict()

    def load_state_dict(self, state: Dict[str, Any], *args,
                        **kwargs) -> Tuple[Any, Any]:
        # R5.5: the restore path passes strict=False for lightweight
        # resume checkpoints — mirror torch's strict-load return.
        return self.model.load_state_dict(state, *args, **kwargs)

    def to(self, device: Any) -> "FakePolicy":
        """Device re-anchor consumed by restore_training_state (G6 run-5
        fix); the fake is CPU-only, but submodules must follow the call."""
        self.model.to(device)
        self.obs_encoder.to(device)
        return self

    def sample_ddpm_with_logprob(
        self,
        obs_dict: Dict[str, torch.Tensor],
        eta: float = 1.0,
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        num_inference_steps: int = None,
        ft_denoising_steps: int = None,
    ):
        """Real DDIM chain via dp_sampler step math; joint logp scalar = 0
        placeholder (per-dim logps are recomputed by the T5 chain_logprobs
        module, which is what the parity tests anchor).

        R1 rework #2 (audit 2026-08-08): CFG mixing is LIVE — the loop
        mirrors phase1a_policy.py:394-443 exactly: encode cond via
        obs_encoder, build uncond with the zeroed quaternion
        (agent_pos[:, :, 3:7] = 0.0) when cfg_lambda > 0, and run the
        double-forward mixing eps = eps_uncond + λ*(eps_cond - eps_uncond).
        A fake sampler ignoring conditions would make mixed == pure-cond and
        CPU contract tests could not discriminate the CFG branch from a dead
        one (the pre-#2 CondAwareModel in test_cfg_contract existed exactly
        for this gap)."""
        device = self.device
        B = next(iter(obs_dict.values())).shape[0]
        T, Da = self.horizon, self.action_dim
        K = num_inference_steps or self.num_inference_steps

        # 1. Encode observation (mirror phase1a_policy.py:388-405)
        nobs = self.normalizer.normalize(obs_dict)
        nobs = {k: v.to(device) for k, v in nobs.items()}
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        this_nobs = {
            k: v[:, : self.n_obs_steps, ...].reshape(-1, *v.shape[2:])
            for k, v in nobs.items()
        }
        global_cond = self.obs_encoder(this_nobs).reshape(B, -1)

        global_cond_uncond = None
        if self.cfg_lambda > 0:
            nobs_u = {k: v.clone() for k, v in nobs.items()}
            nobs_u["agent_pos"][:, :, 3:7] = 0.0
            this_u = {
                k: v[:, : self.n_obs_steps, ...].reshape(-1, *v.shape[2:])
                for k, v in nobs_u.items()
            }
            global_cond_uncond = self.obs_encoder(this_u).reshape(B, -1)

        # 2. Setup scheduler
        self.noise_scheduler.set_timesteps(K)
        timesteps = self.noise_scheduler.timesteps.to(device)

        # 3. Start from noise
        x_t = torch.randn(B, T, Da, device=device, dtype=self.dtype)
        chain = [x_t.clone()]

        # 4. Denoising loop with CFG (mirror phase1a_policy.py:426-443)
        use_cfg = self.cfg_lambda > 0 and global_cond_uncond is not None
        for k in range(K):
            t = int(timesteps[k].item())
            t_prev = int(timesteps[k + 1].item()) if k + 1 < K else 0
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            if use_cfg:
                x_batched = torch.cat([x_t, x_t], dim=0)
                t_batched = torch.cat([t_batch, t_batch], dim=0)
                gc_batched = torch.cat([global_cond, global_cond_uncond], dim=0)
                eps_batched = self.model(
                    sample=x_batched,
                    timestep=t_batched,
                    local_cond=None,
                    global_cond=gc_batched,
                )
                eps_cond, eps_uncond = eps_batched.chunk(2, dim=0)
                eps = eps_uncond + self.cfg_lambda * (eps_cond - eps_uncond)
            else:
                eps = self.model(
                    sample=x_t, timestep=t_batch, local_cond=None,
                    global_cond=global_cond,
                )
            x_next, _ = ddim_step_with_logprob(
                x_t,
                eps,
                t,
                t_prev,
                self.noise_scheduler,
                eta=eta,
            )
            x_t = x_next
            chain.append(x_t.clone())
        return x_t, torch.zeros(B, device=device), chain


def to_tensor_obs(obs_np: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Mock env obs dict (1, T, ...) np -> torch float tensors."""
    return {
        k: torch.from_numpy(v.astype(np.float32)) for k, v in obs_np.items()
    }
