"""
routea_reinflow_fm_policy.py — Route A-RF Actor Wrapper (§4)

Wraps Phase1aPolicyFM with:
  - Adapter/LoRA insertion on FM velocity network
  - ReinFlowNoiseModule integration (stochastic path, log-prob)
  - Teacher anchor (frozen BC/FM reference)
  - RL forward interface (action, log_prob, path, metadata)

Design: RouteA_ReinFlow_FM_RL_Design_v1_2.md §4.1-§4.3, §6.2
"""

import copy
import json
import math
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.policy.phase1a_policy_fm_rl import Phase1aPolicyFM
from diffusion_policy_3d.policy.reinflow_noise_module import (
    ReinFlowNoiseModule,
)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

DEFAULT_LORA_RANK = 4
DEFAULT_LORA_ALPHA = 8

# Target layer patterns for LoRA insertion (first phase: mid + up + final)
# These are substrings matched against named_modules() keys
TARGET_PATTERNS_V1 = [
    "mid_modules",      # bottleneck blocks (1024→1024 × 2)
    "up_modules",       # decoder blocks (2048→512, 512→512, 1024→256, 256→256)
    "final_conv",       # output projection conv
]

# ═══════════════════════════════════════════════════════════════════
# LoRA Adapter Modules
# ═══════════════════════════════════════════════════════════════════


class LoRALinearAdaptor(nn.Module):
    """LoRA low-rank adaptation for nn.Linear modules.

    Wraps a target nn.Linear by adding a low-rank update:
        y = Wx + b + (alpha/r) * x @ A @ B

    A is init with Kaiming normal scaled by 0.01, B is zero-init.
    The original weight/bias are frozen (detached from grad).
    """

    def __init__(self, original: nn.Linear, r: int = 4, alpha: int = 8):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)

        in_features = original.in_features
        out_features = original.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(
            torch.randn(in_features, r) * 0.01
        )
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return LoRA contribution: (alpha/r) * (x @ A @ B)."""
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRAConv1dAdaptor(nn.Module):
    """LoRA low-rank adaptation for nn.Conv1d (kernel=1) modules.

    For Conv1d with kernel_size=1, the operation is a per-position linear
    projection across channels: y[b, c, t] = sum_{k} W[c,k] * x[b,k,t].

    LoRA adapts this as:
        y += (alpha/r) * x @ A @ B  (applied channel-wise, shared across T)
    """

    def __init__(self, original: nn.Conv1d, r: int = 4, alpha: int = 8):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)

        in_channels = original.in_channels
        out_channels = original.out_channels

        self.lora_A = nn.Parameter(
            torch.randn(in_channels, r) * 0.01
        )
        self.lora_B = nn.Parameter(torch.zeros(r, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return LoRA contribution for Conv1d.

        x: (B, C_in, T)
        Returns: (B, C_out, T) — LoRA projection added to each position.
        """
        # (B, T, C_in)
        x_perm = x.permute(0, 2, 1)
        # (B, T, C_out)
        out = (x_perm @ self.lora_A @ self.lora_B) * self.scaling
        return out.permute(0, 2, 1)


# ═══════════════════════════════════════════════════════════════════
# Route A-RF Actor Wrapper
# ═══════════════════════════════════════════════════════════════════


class RouteAReinFlowFMPolicy(nn.Module):
    """Route A-RF actor wrapper: Phase1aPolicyFM + LoRA + ReinFlowNoiseModule.

    Wraps a base Phase1aPolicyFM instance:
      - Base FM weights frozen
      - LoRA adapters inserted on target layers
      - ReinFlowNoiseModule provides per-step stochastic noise
      - Teacher policy (frozen) provides BC anchor
      - Forward returns structured dict for RL training
      - predict_action delegates to base policy for deployment

    Usage:
        base = Phase1aPolicyFM.load_from_checkpoint(...)
        teacher = Phase1aPolicyFM.load_from_checkpoint(...)
        teacher.eval(); teacher.requires_grad_(False)

        noise_module = ReinFlowNoiseModule(num_fm_steps=4)
        policy = RouteAReinFlowFMPolicy(
            base_policy=base,
            teacher_policy=teacher,
            noise_module=noise_module,
            freeze_base=True,
        )

        # RL train
        out = policy(obs_dict)
        action = out['action']        # (B, 8, 11)
        log_prob = out['log_prob']    # (B,)

        # Deploy
        action = policy.predict_action(obs_dict)
    """

    def __init__(
        self,
        base_policy: Phase1aPolicyFM,
        teacher_policy: Phase1aPolicyFM,
        noise_module: ReinFlowNoiseModule,
        adapter_config: Optional[dict] = None,
        freeze_base: bool = True,
        train_anchor_head: bool = False,
        return_path: bool = True,
    ):
        super().__init__()

        self.base_policy = base_policy
        self.teacher_policy = teacher_policy
        self.noise_module = noise_module
        self.return_path = return_path

        # Teacher is always frozen
        self.teacher_policy.eval()
        self.teacher_policy.requires_grad_(False)

        # Freeze base policy weights (obs_encoder, model, normalizer, etc.)
        if freeze_base:
            self.base_policy.requires_grad_(False)

        # Anchor head: optional train
        self._train_anchor_head = train_anchor_head
        if self.base_policy.anchor_head is not None:
            self.base_policy.anchor_head.requires_grad_(train_anchor_head)

        # LoRA adapters
        self._lora_adaptors = nn.ModuleList()
        self._adapter_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._adapter_report: dict = {}

        default_config = {
            "lora_rank": DEFAULT_LORA_RANK,
            "lora_alpha": DEFAULT_LORA_ALPHA,
            "target_patterns": TARGET_PATTERNS_V1,
        }
        if adapter_config is not None:
            default_config.update(adapter_config)
        self.adapter_config = default_config

        self._insert_adapters(self.adapter_config)

        # Copy key attributes from base policy for convenience
        self._copy_base_attrs()

    # ──────────────────────────────────────────────────────────────
    # Attribute forwarding
    # ──────────────────────────────────────────────────────────────

    def _copy_base_attrs(self):
        """Copy frequently-used attributes from base_policy to self for
        clean access in sampling loop."""
        for attr in [
            "num_fm_steps", "sigma_min", "fm_t_scale", "fm_anchor_t_start",
            "cfg_lambda", "cfg_dropout_prob",
            "horizon", "n_action_steps", "n_obs_steps",
            "action_dim", "obs_feature_dim",
            "normalizer", "use_pc_color",
            "solver",
        ]:
            if hasattr(self.base_policy, attr):
                setattr(self, attr, getattr(self.base_policy, attr))
            else:
                # Fallback defaults
                if attr == "num_fm_steps":
                    setattr(self, attr, 4)
                elif attr == "sigma_min":
                    setattr(self, attr, 0.005)
                elif attr == "fm_t_scale":
                    setattr(self, attr, 50)

        self.use_cfg = self.cfg_lambda > 0

    @property
    def device(self):
        return next(self.base_policy.parameters()).device

    @property
    def dtype(self):
        return next(self.base_policy.parameters()).dtype

    # ──────────────────────────────────────────────────────────────
    # LoRA Adapter Insertion
    # ──────────────────────────────────────────────────────────────

    def _is_target_layer(self, module_name: str) -> bool:
        """Check if a module name matches target patterns."""
        patterns = self.adapter_config["target_patterns"]
        return any(p in module_name for p in patterns)

    def _is_linear_like(self, module: nn.Module) -> bool:
        """Check if module is nn.Linear (the primary LoRA target)."""
        return isinstance(module, nn.Linear)

    def _is_conv1d_like(self, module: nn.Module) -> bool:
        """Check if module is nn.Conv1d with kernel=1 (channel projection)."""
        return isinstance(module, nn.Conv1d) and module.kernel_size[0] == 1

    def _make_lora_hook(self, lora_adaptor: nn.Module):
        """Create forward hook that adds LoRA contribution to module output.

        The hook captures `lora_adaptor` in its closure; adaptor params
        are registered in self._lora_adaptors (visible to optimizer).
        """

        def hook(module, input_, output):
            # input_[0] is the tensor passed to the module's forward
            lora_out = lora_adaptor(input_[0])
            return output + lora_out

        return hook

    def _insert_adapters(self, config: dict):
        """Traverse base_policy.model.named_modules() and insert LoRA
        adapters on target layers.

        Saves adapter_insertion_report to self._adapter_report.
        """
        r = config["lora_rank"]
        alpha = config["lora_alpha"]
        target_model = self.base_policy.model
        inserted: List[dict] = []
        total_trainable = 0

        for name, module in target_model.named_modules():
            if not self._is_target_layer(name):
                continue

            adaptor = None
            if self._is_linear_like(module):
                adaptor = LoRALinearAdaptor(module, r=r, alpha=alpha)
            elif self._is_conv1d_like(module):
                adaptor = LoRAConv1dAdaptor(module, r=r, alpha=alpha)

            if adaptor is not None:
                # Register adaptor parameters in wrapper
                self._lora_adaptors.append(adaptor)

                # Register forward hook on target module
                hook = module.register_forward_hook(
                    self._make_lora_hook(adaptor)
                )
                self._adapter_hooks.append(hook)

                n_params = sum(p.numel() for p in adaptor.parameters())
                total_trainable += n_params

                inserted.append({
                    "module_name": name,
                    "module_type": type(module).__name__,
                    "adaptor_type": type(adaptor).__name__,
                    "lora_rank": r,
                    "lora_alpha": alpha,
                    "trainable_params": n_params,
                    "shape": list(module.weight.shape),
                })

        # Collect total base params (model only, not obs_encoder)
        total_base = sum(
            p.numel() for p in self.base_policy.model.parameters()
        )
        # Full policy params (for ratio calc)
        total_policy = sum(
            p.numel() for p in self.base_policy.parameters()
        )

        self._adapter_report = {
            "base_policy_file": "phase1a_policy_fm_rl.py",
            "base_class_name": "Phase1aPolicyFM",
            "checkpoint_id": "v9_2 (n_obs_steps=8, global_cond=1536)",
            "lora_rank": r,
            "lora_alpha": alpha,
            "inserted_layers": inserted,
            "total_trainable_adapter_params": total_trainable,
            "total_base_model_params": total_base,
            "total_policy_params": total_policy,
            "trainable_param_ratio": (
                total_trainable / max(total_base, 1) * 100
            ),
            "trainable_policy_ratio": (
                total_trainable / max(total_policy, 1) * 100
            ),
            "zero_init_verified": False,  # placeholder: measured after construction via measure_zero_init_equivalence()
            "zero_init_teacher_mse": None,  # filled by measure_zero_init_equivalence()
            "n_lora_modules": len(inserted),
        }

    def remove_adapters(self):
        """Remove all LoRA hooks (for checkpointing the clean model)."""
        for hook in self._adapter_hooks:
            hook.remove()
        self._adapter_hooks.clear()
        self._lora_adaptors.clear()

    def get_adapter_report(self) -> dict:
        """Return the adapter insertion report."""
        return dict(self._adapter_report)

    def get_adapter_report_json(self, path: Optional[str] = None) -> str:
        """Return adapter report as formatted JSON. Optionally save to path."""
        report = self.get_adapter_report()
        # Convert numpy values if any
        report_clean = json.loads(json.dumps(report, default=str))
        text = json.dumps(report_clean, indent=2)
        if path is not None:
            with open(path, "w") as f:
                f.write(text)
        return text

    def measure_zero_init_equivalence(
        self,
        atol_b: float = 1e-6,
    ) -> dict:
        """Measure zero-init teacher equivalence.

        With LoRA B=0 at zero init, the LoRA contribution is mathematically zero:
            output_with_adapter = Wx + 0 = Wx = output_without_adapter

        This method verifies:
        1. All LoRA B matrices are truly zero (max abs value < atol_b)
        2. Model forward with adapters (B=0) produces finite output
        3. Perturbing B changes the output (proving adapters are 'live')

        Returns:
            {
                "zero_init_verified": bool,
                "zero_init_max_abs_B": float,
                "n_lora_B_nonzero": int,
                "adapter_perturb_delta": float,
            }
        """
        device = self.device

        # ── 1. Check B matrices are all zeros ──
        max_abs_B = 0.0
        n_nonzero = 0
        for adaptor in self._lora_adaptors:
            B = adaptor.lora_B.data
            max_b = B.abs().max().item()
            max_abs_B = max(max_abs_B, max_b)
            if max_b > atol_b:
                n_nonzero += 1

        # ── 2. Run model forward, verify finite output ──
        model = self.base_policy.model
        B_m = 2
        T = self.horizon
        Da = self.action_dim
        # Auto-detect global_cond_dim from first cond_encoder Linear layer:
        #   cond_encoder in_features = global_cond_dim + diffusion_step_embed_dim
        cond_in = None
        step_enc_in = None
        for name, mod in model.named_modules():
            if 'cond_encoder' in name:
                for sub in mod.modules():
                    if isinstance(sub, torch.nn.Linear):
                        cond_in = sub.in_features; break
            if 'diffusion_step_encoder' in name:
                for sub in mod.modules():
                    if isinstance(sub, torch.nn.Linear):
                        if step_enc_in is None:
                            step_enc_in = sub.in_features  # first Linear: (B,) → embedding
                        else:
                            step_enc_out = sub.out_features
                            break
        if cond_in is not None and step_enc_out is not None:
            cond_dim = cond_in - step_enc_out
        else:
            cond_dim = self.obs_feature_dim  # fallback
        sample = torch.randn(B_m, T, Da, device=device)
        t_input = torch.full((B_m,), 50.0, device=device)
        global_cond = torch.randn(B_m, cond_dim, device=device)

        out_before = model(
            sample=sample, timestep=t_input,
            local_cond=None, global_cond=global_cond,
        )
        assert torch.isfinite(out_before).all(), "Model forward produced NaN/Inf"

        # ── 3. Perturb B, verify output changes ──
        saved_B = []
        for adaptor in self._lora_adaptors:
            saved_B.append(adaptor.lora_B.data.clone())
            adaptor.lora_B.data.add_(torch.randn_like(adaptor.lora_B.data) * 0.01)

        out_after = model(
            sample=sample, timestep=t_input,
            local_cond=None, global_cond=global_cond,
        )
        perturb_delta = (out_before - out_after).abs().max().item()

        # Restore B to zero
        for adaptor, saved in zip(self._lora_adaptors, saved_B):
            adaptor.lora_B.data.copy_(saved)

        # ── Report ──
        zero_init_ok = (max_abs_B < atol_b)
        result = {
            "zero_init_verified": zero_init_ok,
            "zero_init_max_abs_B": max_abs_B,
            "n_lora_B_nonzero": n_nonzero,
            "adapter_perturb_delta": perturb_delta,
        }

        self._adapter_report.update(result)
        return result

    # ──────────────────────────────────────────────────────────────
    # ReinFlow Stochastic FM Sampling
    # ──────────────────────────────────────────────────────────────

    def _reinflow_sample(
        self,
        global_cond: torch.Tensor,
        global_cond_uncond: Optional[torch.Tensor] = None,
        anchor_idx: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_path: bool = True,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """FM Euler sampling loop with ReinFlow noise injection.

        Args:
            global_cond: (B, global_cond_dim) conditional features
            global_cond_uncond: (B, global_cond_dim) unconditional features
                for CFG. If None, CFG is skipped.
            anchor_idx: (B,) anchor indices, or None for vanilla init
            deterministic: If True, skip noise injection (sigma=0)
            return_path: If True, return list of intermediate x_k

        Returns:
            {
                "action": (B, H_action, Da) unnormalized action chunk
                "log_prob": (B,) path log-prob (0 for deterministic)
                "path_states": list of (B, T, Da) normalized tensors (detached, for logging only — Stage-2)
                "noise_sigmas": (num_fm_steps,) sigma w/ grad attached (for noise_loss only)
                "noise_sigmas_log": (num_fm_steps,) sigma, detached, for logging
                "action_norm": (B, H_action, Da) normalized action
            }
        """
        model = self.base_policy.model
        dtype = self.dtype
        device = global_cond.device
        B = global_cond.shape[0]
        T = self.horizon
        Da = self.action_dim
        n_steps = self.num_fm_steps

        # ── Initialize x_t ──
        if anchor_idx is not None and self.base_policy.anchor_bank_full is not None:
            anchor_traj = self.base_policy._get_anchor_trajectory(anchor_idx)
            noise_init = torch.randn(
                anchor_traj.shape, device=device, dtype=dtype
            )
            t_start = self.fm_anchor_t_start
            x_t = (1 - t_start) * noise_init + t_start * anchor_traj
        else:
            noise_init = torch.randn(B, T, Da, device=device, dtype=dtype)
            x_t = noise_init
            t_start = 0.0

        # ── Timesteps ──
        t_end = 1.0 - self.sigma_min
        t_start = min(t_start, t_end - 1e-6)
        timesteps = torch.linspace(
            t_start, t_end, n_steps + 1, device=device
        )
        use_cfg = self.use_cfg and global_cond_uncond is not None

        # ── Noise injection setup ──
        noise_sigmas = self.noise_module()  # (n_steps,)

        # For deterministic: no injected noise
        if deterministic:
            sigmas_eff = torch.zeros_like(noise_sigmas)
        else:
            sigmas_eff = noise_sigmas

        # ── Path tracking ──
        path_states = [x_t.detach().clone()] if return_path else None
        noise_xi_list = []  # collect for log-prob

        # ── Sampling loop ──
        for i in range(n_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            # Timestep encoding
            t_input = (t * self.fm_t_scale).float().expand(B)

            # Model forward (with CFG)
            if use_cfg:
                x_t_batched = torch.cat([x_t, x_t], dim=0)
                t_batched = torch.cat([t_input, t_input], dim=0)
                global_batched = torch.cat(
                    [global_cond, global_cond_uncond], dim=0
                )
                v_batched = model(
                    sample=x_t_batched,
                    timestep=t_batched,
                    local_cond=None,
                    global_cond=global_batched,
                )
                v_cond, v_uncond = v_batched.chunk(2, dim=0)
                v = v_uncond + self.cfg_lambda * (v_cond - v_uncond)
            else:
                v = model(
                    sample=x_t,
                    timestep=t_input,
                    local_cond=None,
                    global_cond=global_cond,
                )

            # Euler step
            x_t = x_t + v * dt

            # ReinFlow noise injection
            if sigmas_eff[i].item() > 0:
                xi = torch.randn_like(x_t)
                x_t = x_t + sigmas_eff[i] * xi
                noise_xi_list.append(xi)
            else:
                noise_xi_list.append(torch.zeros_like(x_t))

            if return_path:
                path_states.append(x_t.detach().clone())

        # ── Compute log-prob of stochastic path ──
        #   x_{k+1} = x_k + dt * v_k + sigma_k * xi_k, xi_k ~ N(0, I)
        #   log p = -0.5 * sum_k (||xi_k||^2 + D*T * log(2*pi*sigma_k^2))
        if deterministic:
            log_prob = torch.zeros(B, device=device)
        else:
            xi_stacked = torch.stack(noise_xi_list, dim=1)  # (B, n_steps, T, Da)
            xi_norm_sq = (xi_stacked ** 2).sum(dim=(2, 3))  # (B, n_steps)
            # Per-step log-det: Da * T * log(2*pi*sigma_k^2)
            log_det = Da * T * torch.log(
                2 * math.pi * sigmas_eff.pow(2) + 1e-10
            )  # (n_steps,)
            log_prob_per_step = -0.5 * (
                xi_norm_sq + log_det.unsqueeze(0)
            )  # (B, n_steps)
            log_prob = log_prob_per_step.sum(dim=1)  # (B,)

        # ── Extract action chunk ──
        naction_pred = x_t[..., :Da]  # (B, T, Da)

        action_pred = self.base_policy.normalizer["action"].unnormalize(
            naction_pred
        )
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action_chunk = action_pred[:, start:end]  # (B, n_action_steps, Da)

        return {
            "action": action_chunk,
            "action_norm": naction_pred[:, start:end],
            "log_prob": log_prob,
            "noise_sigmas": noise_sigmas,                     # preserves grad for noise_loss backprop
            "noise_sigmas_log": noise_sigmas.detach().cpu(),  # logging only
            "path_states": path_states,
        }

    # ──────────────────────────────────────────────────────────────
    # Forward (RL training interface)
    # ──────────────────────────────────────────────────────────────

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        deterministic: bool = False,
        return_log_prob: bool = True,
        return_path: bool = True,
        compute_teacher: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, List, dict]]:
        """RL training forward pass.

        1. Normalize obs, encode global_cond
        2. Run frozen teacher to get BC anchor
        3. Run actor with ReinFlow noise injection
        4. Return structured dict for replay and actor loss

        Args:
            obs_dict: observation dict (unnormalized)
            deterministic: If True, no noise injection (sigma=0)
            return_log_prob: If True, compute path log-prob
            return_path: If True, return intermediate x_k
            seed: Optional integer seed for reproducibility

        Returns:
            {
                "action": (B, H_action, Da),
                "log_prob": (B,) log-prob of stochastic path,
                "path_states": list of x_k tensors (normalized),
                "noise_sigmas": (num_fm_steps,) sigmas,
                "teacher_action": (B, H_action, Da) frozen BC anchor,
                "global_cond": (B, global_cond_dim),
                "metadata": {...}
            }
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = self.device
        dtype = self.dtype
        base = self.base_policy

        # ── Normalize observations ──
        nobs = base.normalizer.normalize(obs_dict)
        nobs = dict_apply(nobs, lambda x: x.to(device, dtype))

        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        B = next(iter(nobs.values())).shape[0]

        # ── Encode global condition ──
        To = base.n_obs_steps
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]),
        )
        global_cond = base.obs_encoder(this_nobs).reshape(B, -1)

        # ── CFG unconditional encoding ──
        global_cond_uncond = None
        if self.use_cfg:
            nobs_u = {k: v.clone() for k, v in nobs.items()}
            # Zero out goal_body and sin/cos_yaw (the conditioning dropout)
            nobs_u["agent_pos"][:, :, 3:7] = 0.0
            this_u = dict_apply(
                nobs_u,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]),
            )
            global_cond_uncond = base.obs_encoder(this_u).reshape(B, -1)

        # ── Anchor selection (frozen, no grad) ──
        anchor_idx = None
        if base.anchor_head is not None:
            with torch.no_grad():
                anchor_idx = base.anchor_head(global_cond).argmax(dim=-1)

        # ── Teacher action (frozen, optional for speed) ──
        if compute_teacher:
            with torch.no_grad():
                teacher_out = self.teacher_policy.predict_action(obs_dict)
                teacher_action = teacher_out["action_pred"]  # (B, 8, 11)
        else:
            teacher_action = None

        # ── Actor ReinFlow sampling ──
        out = self._reinflow_sample(
            global_cond=global_cond,
            global_cond_uncond=global_cond_uncond,
            anchor_idx=anchor_idx,
            deterministic=deterministic,
            return_path=return_path,
        )

        # ── Build return dict ──
        metadata = {
            "deterministic": deterministic,
            "cfg_lambda": self.cfg_lambda,
            "adapter_rank": self.adapter_config["lora_rank"],
            "n_lora_modules": len(self._lora_adaptors),
        }

        result = {
            "action": out["action"],
            "log_prob": out["log_prob"] if return_log_prob
                         else torch.zeros(B, device=device),
            "path_states": out["path_states"],
            "noise_sigmas": out["noise_sigmas"],          # grad-preserving (for noise_loss)
            "noise_sigmas_log": out["noise_sigmas_log"],  # detached cpu copy for logging
            "teacher_action": teacher_action.detach() if teacher_action is not None else None,
            "global_cond": global_cond.detach(),
            "metadata": metadata,
        }

        return result

    # ──────────────────────────────────────────────────────────────
    # predict_action (deployment/eval interface)
    # ──────────────────────────────────────────────────────────────

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Deployment inference.

        Delegates to base_policy.predict_action with adapters active.
        Uses deterministic mean path (no noise injection).

        Args:
            obs_dict: unnormalized observation dict

        Returns:
            {"action_pred": (B, n_action_steps, Da)}
        """
        # Use base policy's predict_action — adapters are on model,
        # so they participate. The CFG + projection still apply here.
        return self.base_policy.predict_action(obs_dict)

    # ──────────────────────────────────────────────────────────────
    # Trainable parameter groups
    # ──────────────────────────────────────────────────────────────

    def get_trainable_param_groups(
        self,
        lr: float = 1e-5,
        anchor_lr_mult: float = 5.0,
        noise_lr_mult: float = 1.0,
    ) -> List[Dict]:
        """Create optimizer param groups for Route A-RF training.

        Returns:
            [
                {"params": LoRA adaptors, "lr": lr},
                {"params": noise module, "lr": lr * noise_lr_mult},
            ]
            + optionally anchor_head group if train_anchor_head=True.
        """
        groups = []

        # LoRA adaptors
        if len(self._lora_adaptors) > 0:
            groups.append({
                "params": self._lora_adaptors.parameters(),
                "lr": lr,
            })

        # Noise module
        groups.append({
            "params": self.noise_module.parameters(),
            "lr": lr * noise_lr_mult,
        })

        # Optional: trainable anchor head
        if self._train_anchor_head and self.base_policy.anchor_head is not None:
            groups.append({
                "params": self.base_policy.anchor_head.parameters(),
                "lr": lr * anchor_lr_mult,
            })

        return groups

    def get_base_param_count(self) -> int:
        """Return total base policy parameter count."""
        return sum(p.numel() for p in self.base_policy.parameters())

    def get_trainable_param_count(self) -> int:
        """Return total trainable parameter count (LoRA + noise + optional)."""
        total = 0
        for adaptor in self._lora_adaptors:
            total += sum(p.numel() for p in adaptor.parameters())
        total += sum(p.numel() for p in self.noise_module.parameters())
        if self._train_anchor_head and self.base_policy.anchor_head is not None:
            total += sum(p.numel() for p in self.base_policy.anchor_head.parameters())
        return total

    def extra_repr(self) -> str:
        report = self._adapter_report
        return (
            f"trainable_params={report.get('total_trainable_adapter_params', '?')}, "
            f"ratio={report.get('trainable_param_ratio', '?'):.3f}%, "
            f"lora_rank={report.get('lora_rank', '?')}, "
            f"n_lora_modules={len(self._lora_adaptors)}"
        )


# ═══════════════════════════════════════════════════════════════════
# Actor Loss Helper
# ═══════════════════════════════════════════════════════════════════


def compute_routea_actor_losses(
    actor_outputs: Dict,
    teacher_action: Optional[torch.Tensor] = None,
    teacher_path_states: Optional[List[torch.Tensor]] = None,
    advantage: Optional[torch.Tensor] = None,
    q_min: Optional[torch.Tensor] = None,
    lambda_bc: float = 1.0,
    lambda_path: float = 0.1,
    lambda_noise: float = 0.01,
    lambda_smooth: float = 0.001,
    lambda_pg: float = 1.0,
    lambda_dqpg: float = 1.0,
    pg_scale_mode: str = "raw_sum",
    pg_logprob_scale_denom: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute Route A-RF actor losses (§4.2).

    L_actor = L_PG + lambda_DQPG * L_DQPG
              + lambda_BC * L_BC + lambda_path * L_path
              + lambda_noise * L_noise + lambda_smooth * L_smooth

    Two PG terms:
      - L_PG (REINFORCE): -log_prob * advantage  — gradient through log_prob (→ noise module)
      - L_DQPG (DDPG-style differentiable Q): -Q(s, π(s)) — gradient through action (→ LoRA)

    Args:
        actor_outputs: output dict from RouteAReinFlowFMPolicy.forward()
        teacher_action: (B, H, Da) frozen teacher action. If None,
                        uses actor_outputs['teacher_action'].
        teacher_path_states: list of teacher path states. If None,
                             path loss is marked unavailable.
        advantage: (B,) advantage estimate for REINFORCE PG term. If None,
                   REINFORCE PG loss is set to 0.
        q_min: (B,) differentiable Q = min(Q1_total, Q2_total) for DQPG.
               If None, DQPG loss is set to 0. Gradient flows from Q through
               action to actor (LoRA) parameters.
        lambda_bc: BC anchor MSE weight (§4.2.2)
        lambda_path: Path regularization weight (§4.2.3)
        lambda_noise: Noise scale regularization weight (§4.2.4)
        lambda_smooth: Smoothness regularization weight (§4.2.5)
        lambda_pg: REINFORCE policy gradient weight
        lambda_dqpg: Differentiable Q gradient weight (§4.2.1b)

    Returns:
        {
            "actor_loss_total": scalar,
            "actor_loss_pg": scalar,
            "actor_loss_dqpg": scalar,
            "actor_loss_bc": scalar,
            "actor_loss_path": scalar,
            "actor_loss_noise": scalar,
            "actor_loss_smooth": scalar,
            "path_loss_available": bool,
            "pg_available": bool,
            "dqpg_available": bool,
        }
    """
    device = actor_outputs["action"].device

    # ── Policy gradient (REINFORCE) — gradient through log_prob → noise module ──
    log_prob = actor_outputs.get("log_prob")
    pg_loss = torch.tensor(0.0, device=device)
    pg_available = False
    _log_prob_for_pg = log_prob  # pre-scale, for diagnostics
    _pg_scale_denom_used = 1.0
    if advantage is not None and log_prob is not None:
        # ── B2a: PG scale normalization ──
        if pg_scale_mode in ("per_action_dim", "per_flow_event"):
            _pg_scale_denom_used = max(float(pg_logprob_scale_denom), 1.0)
            _log_prob_for_pg = log_prob / _pg_scale_denom_used
        elif pg_scale_mode in ("raw_sum", "lambda_only"):
            _pg_scale_denom_used = 1.0
            _log_prob_for_pg = log_prob
        else:
            raise ValueError(f"Unsupported pg_scale_mode: {pg_scale_mode}")
        # L_PG = -E[log_prob * advantage]  (§4.2.1a)
        pg_loss = -(_log_prob_for_pg * advantage.detach()).mean()
        pg_available = True

    # ── Differentiable Q PG (DDPG-style) — gradient through action → LoRA ──
    #   L_DQPG = -E[Q(s, π(s))]
    #   Q gradient backpropagates through critic forward → action → actor (LoRA)
    dqpg_loss = torch.tensor(0.0, device=device)
    dqpg_available = False
    if q_min is not None:
        dqpg_loss = -q_min.mean()
        dqpg_available = True

    # ── BC anchor MSE (§4.2.2) ──
    if teacher_action is None:
        teacher_action = actor_outputs.get("teacher_action")
    if teacher_action is not None:
        bc_loss = F.mse_loss(
            actor_outputs["action"], teacher_action.detach()
        )
    else:
        bc_loss = torch.tensor(0.0, device=device)

    # ── Path regularization (§4.2.3) — Stage-2 (currently DISABLED) ──
    # [FACT] path_states are collected detached (for logging/MSE monitoring only).
    # Option B active: path_loss_available=false until gradient chain is stabilized.
    # In Stage-2, switch to grad-preserving path_states (remove .detach() in _reinflow_sample).
    path_loss = torch.tensor(0.0, device=device)
    path_loss_available = False
    path_mse_stage2 = True  # explicit marker: path loss not yet in training
    path_states = actor_outputs.get("path_states")
    if path_states is not None and teacher_path_states is not None:
        if len(path_states) > 0 and len(teacher_path_states) > 0:
            n_path = min(len(path_states), len(teacher_path_states))
            losses = []
            for k in range(n_path):
                losses.append(
                    F.mse_loss(path_states[k], teacher_path_states[k].detach())
                )
            path_loss = torch.stack(losses).mean()
            # path_loss_available remains False — path_states are detached in collection
            # Re-enable when Stage-2 removes .detach() from _reinflow_sample path tracking
            path_mse_stage2 = False

    # ── Noise scale regularization (§4.2.4) ──
    noise_sigmas = actor_outputs.get("noise_sigmas")
    noise_loss = torch.tensor(0.0, device=device)
    if noise_sigmas is not None:
        # Target: log(sigma) close to -4.6 (~0.01 in normalized space)
        log_sigma = torch.log(noise_sigmas.to(device) + 1e-10)
        log_target = math.log(0.01)
        noise_loss = F.mse_loss(log_sigma, torch.full_like(log_sigma, log_target))

    # ── Smoothness (§4.2.5) ──
    action = actor_outputs["action"]  # (B, H, Da)
    smooth_loss = torch.tensor(0.0, device=device)
    if action.shape[1] > 1:
        # First-order smoothness: sum of squared differences
        diff = action[:, 1:] - action[:, :-1]
        smooth_loss = (diff ** 2).mean()

    # ── Total ──
    total = (
        lambda_pg * pg_loss
        + lambda_dqpg * dqpg_loss
        + lambda_bc * bc_loss
        + lambda_path * path_loss
        + lambda_noise * noise_loss
        + lambda_smooth * smooth_loss
    )

    # ── B2a: Loss-scale diagnostics (no gradient, read-only) ──
    _log_prob_mean = log_prob.detach().mean() if log_prob is not None else torch.tensor(float("nan"), device=device)
    _log_prob_abs_mean = log_prob.detach().abs().mean() if log_prob is not None else torch.tensor(float("nan"), device=device)
    _advantage_mean = advantage.detach().mean() if advantage is not None else torch.tensor(float("nan"), device=device)
    _advantage_abs_mean = advantage.detach().abs().mean() if advantage is not None else torch.tensor(float("nan"), device=device)
    _pg_dqpg_abs_ratio = (
        pg_loss.detach().abs() / (dqpg_loss.detach().abs() + 1e-8)
        if dqpg_available else torch.tensor(float("nan"), device=device)
    )

    return {
        "actor_loss_total": total,
        "actor_loss_pg": pg_loss,
        "actor_loss_dqpg": dqpg_loss,
        "actor_loss_bc": bc_loss,
        "actor_loss_path": path_loss,
        "actor_loss_noise": noise_loss,
        "actor_loss_smooth": smooth_loss,
        "path_loss_available": torch.tensor(path_loss_available, dtype=torch.bool),
        "path_mse_stage2": torch.tensor(path_mse_stage2, dtype=torch.bool),
        "pg_available": torch.tensor(pg_available, dtype=torch.bool),
        "dqpg_available": torch.tensor(dqpg_available, dtype=torch.bool),
        # ── B2a: loss-scale diagnostic fields ──
        "log_prob_mean": _log_prob_mean,
        "log_prob_abs_mean": _log_prob_abs_mean,
        "advantage_mean": _advantage_mean,
        "advantage_abs_mean": _advantage_abs_mean,
        "pg_dqpg_abs_ratio": _pg_dqpg_abs_ratio,
        "pg_scale_mode": torch.tensor(
            ["raw_sum", "lambda_only", "per_action_dim", "per_flow_event"].index(pg_scale_mode),
            device=device, dtype=torch.long,
        ),
        "pg_logprob_scale_denom": torch.tensor(float(_pg_scale_denom_used), device=device),
    }
