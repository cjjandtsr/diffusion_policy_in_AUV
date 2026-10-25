"""Thin Phase1aPolicyFM interface bridge for official ReinFlow classes."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Mapping, Optional

import torch
from torch import nn

from .condition_adapter import FMConditionBatch


class PolicyBridgeError(RuntimeError):
    """Raised when native FM and official ReinFlow interfaces are incompatible."""


def condition_to_official_dict(condition: FMConditionBatch) -> dict[str, torch.Tensor]:
    """Expose encoded conditions through the dict accepted by official ReinFlow."""

    return {
        "state": condition.cond,
        "global_cond": condition.cond,
        "global_cond_uncond": condition.uncond,
        "noise_cond": condition.noise_cond,
    }


class IDP3FlowPolicyBridge(nn.Module):
    """Adapt native CFG velocity to official ReinFlow's unit-time Euler grid.

    Native Phase1aPolicyFM integrates over ``[0, 1-sigma_min]``.  Official
    ReinFlow integrates over ``[0, 1]`` using ``dt=1/K``.  Mapping time by
    ``t_native=(1-sigma_min)*t_official`` and velocity by the same Jacobian
    preserves the native deterministic four-step trajectory exactly while all
    stochastic transitions and log-probabilities remain official code.
    """

    def __init__(
        self,
        native_policy: nn.Module,
        *,
        lora_bundle=None,
        lora_enabled: bool = False,
        cond_enc_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.native_policy = native_policy
        self.horizon_steps = int(getattr(native_policy, "horizon", -1))
        self.action_dim = int(getattr(native_policy, "action_dim", -1))
        if (self.horizon_steps, self.action_dim) != (16, 11):
            raise PolicyBridgeError(
                "native FM bridge requires full trajectory shape (16,11)"
            )
        self.act_dim_total = self.horizon_steps * self.action_dim
        inferred_cond_dim = getattr(native_policy, "obs_feature_dim", None)
        self.cond_enc_dim = int(
            cond_enc_dim if cond_enc_dim is not None else inferred_cond_dim
        )
        if self.cond_enc_dim <= 0:
            raise PolicyBridgeError("cond_enc_dim must be positive")
        self.time_dim = 1
        self.sigma_min = float(getattr(native_policy, "sigma_min", float("nan")))
        self.fm_t_scale = float(
            getattr(native_policy, "fm_t_scale", float("nan"))
        )
        self.cfg_lambda = float(getattr(native_policy, "cfg_lambda", float("nan")))
        if self.sigma_min != 0.005:
            raise PolicyBridgeError("native sigma_min must be 0.005")
        if self.fm_t_scale != 50.0:
            raise PolicyBridgeError("native fm_t_scale must be 50")
        if self.cfg_lambda != 1.5:
            raise PolicyBridgeError("native cfg_lambda must be 1.5")
        if not callable(getattr(native_policy, "_compute_cfg_velocity", None)):
            raise PolicyBridgeError("native policy has no _compute_cfg_velocity")
        if lora_bundle is not None and not hasattr(lora_bundle, "mode"):
            raise PolicyBridgeError("lora_bundle must expose mode(enabled)")
        self.__dict__["_lora_bundle"] = lora_bundle
        self.lora_enabled = bool(lora_enabled)

    def set_lora_enabled(self, enabled: bool) -> None:
        self.lora_enabled = bool(enabled)

    def _lora_context(self):
        bundle = self.__dict__.get("_lora_bundle")
        if bundle is None:
            return nullcontext()
        return bundle.mode(self.lora_enabled)

    def _condition_tensors(
        self, cond: Mapping[str, torch.Tensor], action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = ("global_cond", "global_cond_uncond", "noise_cond")
        missing = [key for key in required if key not in cond]
        if missing:
            raise PolicyBridgeError(f"official condition is missing keys: {missing}")
        global_cond = cond["global_cond"].to(
            device=action.device, dtype=action.dtype
        )
        global_cond_uncond = cond["global_cond_uncond"].to(
            device=action.device, dtype=action.dtype
        )
        noise_cond = cond["noise_cond"].to(
            device=action.device, dtype=action.dtype
        )
        batch = action.shape[0]
        if global_cond.shape[0] != batch or global_cond_uncond.shape != global_cond.shape:
            raise PolicyBridgeError("CFG condition batch/shape mismatch")
        if tuple(noise_cond.shape) != (batch, self.cond_enc_dim):
            raise PolicyBridgeError(
                "noise condition shape mismatch: "
                f"expected {(batch, self.cond_enc_dim)}, got {tuple(noise_cond.shape)}"
            )
        return global_cond, global_cond_uncond, noise_cond

    def forward(
        self,
        action: torch.Tensor,
        time,
        cond: Mapping[str, torch.Tensor],
        output_embedding: bool = False,
        **kwargs,
    ):
        if tuple(action.shape[1:]) != (self.horizon_steps, self.action_dim):
            raise PolicyBridgeError(
                f"action must have shape (B,{self.horizon_steps},{self.action_dim})"
            )
        batch = action.shape[0]
        time_tensor = torch.as_tensor(time, device=action.device, dtype=action.dtype)
        if time_tensor.ndim == 0:
            time_tensor = time_tensor.repeat(batch)
        time_tensor = time_tensor.reshape(-1)
        if time_tensor.shape[0] != batch:
            raise PolicyBridgeError("time must be scalar or have shape (B,)")
        if torch.any(time_tensor < 0) or torch.any(time_tensor >= 1):
            raise PolicyBridgeError("official flow time must be in [0,1)")

        global_cond, global_cond_uncond, noise_cond = self._condition_tensors(
            cond, action
        )
        path_scale = 1.0 - self.sigma_min
        native_time = time_tensor * path_scale
        native_time_input = (native_time * self.fm_t_scale).float()
        with self._lora_context():
            native_velocity = self.native_policy._compute_cfg_velocity(
                action,
                native_time_input,
                global_cond,
                global_cond_uncond,
                None,
            )
        if hasattr(native_velocity, "sample"):
            native_velocity = native_velocity.sample
        if tuple(native_velocity.shape) != tuple(action.shape):
            raise PolicyBridgeError(
                f"native velocity shape {tuple(native_velocity.shape)} "
                f"does not match action {tuple(action.shape)}"
            )
        velocity = native_velocity * path_scale
        if not torch.isfinite(velocity).all():
            raise PolicyBridgeError("native velocity contains NaN or Inf")
        if output_embedding:
            return velocity, native_time[:, None], noise_cond
        return velocity


__all__ = [
    "IDP3FlowPolicyBridge",
    "PolicyBridgeError",
    "condition_to_official_dict",
]
