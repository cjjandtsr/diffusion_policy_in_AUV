"""Temporal critic for the ordered three-frame E5 condition."""

from __future__ import annotations

import torch
from torch import nn


class ValueContractError(RuntimeError):
    pass


class TemporalValueNetwork(nn.Module):
    def __init__(self, history: int = 3, frame_dim: int = 192, hidden_dim: int = 256,
                 history_scalar: bool = False, episode_context: bool = False):
        super().__init__()
        if history != 3 or frame_dim != 192:
            raise ValueContractError("E5 critic requires ordered 3 x 192 features")
        self.history = history
        self.frame_dim = frame_dim
        self.history_scalar = bool(history_scalar)
        self.episode_context = bool(episode_context)
        if self.history_scalar and self.episode_context:
            raise ValueContractError("choose beta or episode context, not both")
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
        if self.history_scalar:
            # Allocate after A0 initialization; zeros consume no RNG state.
            self.history_weight = nn.Parameter(torch.zeros(hidden_dim))
        if self.episode_context:
            self.context_weight = nn.Parameter(torch.zeros(hidden_dim, 6))

    def forward(self, condition):
        global_cond = condition.get("global_cond", condition.get("state"))
        if global_cond is None or tuple(global_cond.shape[1:]) != (
            self.history * self.frame_dim,
        ):
            raise ValueContractError("critic condition must have shape (B,576)")
        frames = global_cond.reshape(-1, self.history, self.frame_dim).detach()
        ordered = self.frame_norm(frames) + self.position.unsqueeze(0)
        flat = ordered.reshape(ordered.shape[0], -1)
        if not self.history_scalar and not self.episode_context:
            return self.mlp(flat)
        key, width = ("episode_context", 6) if self.episode_context else ("beta", 1)
        context = condition.get(key)
        if context is None or tuple(context.shape) != (flat.shape[0], width):
            raise ValueContractError("critic {} must have shape (B,{})".format(key, width))
        if not torch.isfinite(context).all():
            raise ValueContractError("critic context must be finite")
        extra = (context.to(flat) @ self.context_weight.T if self.episode_context
                 else context.to(flat) * self.history_weight)
        hidden = self.mlp[0](flat) + extra
        for layer in self.mlp[1:]:
            hidden = layer(hidden)
        return hidden
