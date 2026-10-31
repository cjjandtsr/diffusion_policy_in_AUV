#!/usr/bin/env python3
"""
chunk_critic.py — Chunk-aware Twin Critic for Route A-RF (§5)

Design: RouteA_ReinFlow_FM_RL_Design_v1_2.md §5

Twin critic with decomposed reward heads, preserving action chunk temporal structure:

    Q_phi(s_t, A_t)  where  s_t = global_cond (1536d),  A_t = (H=8, Da=11)

Heads:
  ┌─────────────────────┬─────────────────────────────────────┐
  │ Head                │ Target (reward decomposition)       │
  ├─────────────────────┼─────────────────────────────────────┤
  │ total               │ sum of all reward terms (default   │
  │                     │ for actor update)                  │
  │ progress            │ base + pbrs                        │
  │ onset               │ onset                              │
  │ future_clearance    │ future_clearance                   │
  │ trap                │ trap + flip                        │
  │ support             │ support                            │
  │ guard               │ guard (penalty diagnosis)          │
  └─────────────────────┴─────────────────────────────────────┘

Usage:
    critic = ChunkTwinCritic(state_dim=1536, da=11, horizon=8)
    target = ChunkTwinCritic(state_dim=1536, da=11, horizon=8)
    target.load_state_dict(critic.state_dict())

    out = critic(state, action)          # {head: (B, 2)}  [Q1, Q2] last dim
    q_min = critic.q_min(state, action)  # (B,)  min(Q1, Q2) for total head
    q1    = critic.q1(state, action)     # {head: (B,)}  first critic only

    target_update(target, critic, tau=0.005)  # Polyak EMA
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

HEAD_NAMES = [
    "total",
    "progress",
    "onset",
    "future_clearance",
    "trap",
    "support",
    "guard",
]

# §5.4 Default beta weights for decomposed heads
DEFAULT_BETAS = {
    "progress": 0.5,
    "onset": 1.0,
    "future_clearance": 1.0,
    "trap": 1.0,
    "support": 1.0,   # set to 0 offline density not loaded
    "guard": 0.5,
}

# §5.5 Default global_cond dimension (v9_2, n_obs_steps=8, multi-stage PointNet)
EXPECTED_STATE_DIM = 1536


# ═══════════════════════════════════════════════════════════════════
# Sub-modules
# ═══════════════════════════════════════════════════════════════════

class TemporalActionEncoder(nn.Module):
    """Action chunk encoder preserving temporal structure.

    Input:  (B, H, Da)  — action chunk, e.g. (8, 11)
    Output: (B, D_out)  — temporal features

    Architecture: Conv1d over time dimension with adaptive pooling.
    Two conv layers with kernel=3 capture local temporal dependencies
    (onset vs future_clearance at different positions in the chunk).
    """

    def __init__(self, da: int = 11, horizon: int = 8,
                 hidden: int = 128, d_out: int = 256):
        super().__init__()
        self.horizon = horizon

        self.conv = nn.Sequential(
            nn.Conv1d(da, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),        # (B, hidden, 1)
        )
        self.fc = nn.Linear(hidden, d_out)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action: (B, H, Da) action chunk
        Returns:
            feat: (B, D_out) temporal features
        """
        # Conv1d expects (B, Da, H)
        x = action.permute(0, 2, 1).contiguous()   # (B, Da, H) -> (B, 11, 8)
        x = self.conv(x)                            # (B, hidden, 1)
        x = x.squeeze(-1)                           # (B, hidden)
        return self.fc(x)                           # (B, D_out)


class StateEncoder(nn.Module):
    """global_cond encoder for critic state.

    Input:  (B, D_state)  — global_cond (1536d)
    Output: (B, D_out)    — state features (256d)
    """

    def __init__(self, d_state: int = 1536, d_out: int = 256):
        super().__init__()
        expected = EXPECTED_STATE_DIM
        if d_state != expected:
            import warnings
            warnings.warn(
                f"[chunk_critic] StateEncoder: d_state={d_state}, "
                f"expected {expected} (from v9_2 n_obs_steps=8). "
                f"Verify checkpoint config."
            )

        self.net = nn.Sequential(
            nn.Linear(d_state, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, D_state) → (B, D_out)"""
        return self.net(state)


class CriticHead(nn.Module):
    """Single Q-value head.

    A lightweight MLP on the critic trunk output.
    """

    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) → (B, 1)"""
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════
# Single Critic (one half of twin)
# ═══════════════════════════════════════════════════════════════════

class SingleChunkCritic(nn.Module):
    """One chunk-aware critic with decomposed reward heads.

    Architecture:
        state(1536d) ─→ StateEncoder ─→ state_feat(256d) ─┐
                                                           ├→ concat(512d)
        action(H,11) ─→ TemporalActionEncoder ─→ act_feat(256d) ─┘
                                                           │
                                                           v
                                                     trunk MLP ─→ 7 heads × 1
    """

    def __init__(self,
                 state_dim: int = 1536,
                 da: int = 11,
                 horizon: int = 8,
                 hidden: int = 256,
                 head_hidden: int = 128,
                 head_names: Optional[List[str]] = None,
                 use_layernorm: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.da = da
        self.horizon = horizon
        self.head_names = head_names or HEAD_NAMES
        self.n_heads = len(self.head_names)
        self.use_layernorm = use_layernorm

        self.state_encoder = StateEncoder(d_state=state_dim, d_out=hidden)
        self.action_encoder = TemporalActionEncoder(
            da=da, horizon=horizon, d_out=hidden)

        # ── B1c: optional LayerNorm for activation scale stability ──
        if use_layernorm:
            self.state_ln = nn.LayerNorm(hidden)
            self.action_ln = nn.LayerNorm(hidden)
            trunk_layers: List[nn.Module] = [
                nn.Linear(hidden * 2, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
            ]
        else:
            self.state_ln = None  # type: ignore[assignment]
            self.action_ln = None  # type: ignore[assignment]
            trunk_layers = [
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
            ]

        # Trunk: fuse state + action features
        self.trunk = nn.Sequential(*trunk_layers)

        # Decomposed reward heads
        self.heads = nn.ModuleDict()
        for name in self.head_names:
            self.heads[name] = CriticHead(d_in=hidden, d_hidden=head_hidden)

        # Parameter count
        n_total = sum(p.numel() for p in self.parameters())
        n_total_m = n_total / 1e6

    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state:  (B, D_state) global_cond features
            action: (B, H, Da)   action chunk

        Returns:
            out: {head_name: (B, 1)} for each head
        """
        s_feat = self.state_encoder(state)       # (B, hidden)
        a_feat = self.action_encoder(action)     # (B, hidden)
        # ── B1c: optional LayerNorm ──
        if self.use_layernorm:
            s_feat = self.state_ln(s_feat)  # type: ignore[misc]
            a_feat = self.action_ln(a_feat)  # type: ignore[misc]
        x = torch.cat([s_feat, a_feat], dim=-1)  # (B, hidden*2)
        x = self.trunk(x)                         # (B, hidden)

        return {name: self.heads[name](x) for name in self.head_names}

    def extra_repr(self) -> str:
        n_total = sum(p.numel() for p in self.parameters())
        return (f"state_dim={self.state_dim}, da={self.da}, "
                f"horizon={self.horizon}, n_heads={self.n_heads}, "
                f"params={n_total/1e6:.2f}M")


# ═══════════════════════════════════════════════════════════════════
# Twin Critic (Q1, Q2)
# ═══════════════════════════════════════════════════════════════════

class ChunkTwinCritic(nn.Module):
    """Twin chunk-aware critic with decomposed reward heads (§5.2).

    Contains two independent SingleChunkCritic instances (Q_phi1, Q_phi2)
    for Clipped Double Q-learning (TD3-style).

    Forward returns dict of {head_name: (B, 2)} where last dim is [Q1, Q2].

    Convenience methods:
        q_min(state, action) -> (B,)  — min(Q1, Q2) for total head
        q1(state, action)    -> dict  — first critic's heads only
        q2(state, action)    -> dict  — second critic's heads only
    """

    def __init__(self,
                 state_dim: int = 1536,
                 da: int = 11,
                 horizon: int = 8,
                 hidden: int = 256,
                 head_hidden: int = 128,
                 head_names: Optional[List[str]] = None,
                 use_layernorm: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.da = da
        self.horizon = horizon
        self.head_names = head_names or HEAD_NAMES
        self.hidden = hidden
        self.use_layernorm = use_layernorm

        # Two independent critics
        self.q1 = SingleChunkCritic(
            state_dim=state_dim, da=da, horizon=horizon,
            hidden=hidden, head_hidden=head_hidden,
            head_names=self.head_names,
            use_layernorm=use_layernorm,
        )
        self.q2 = SingleChunkCritic(
            state_dim=state_dim, da=da, horizon=horizon,
            hidden=hidden, head_hidden=head_hidden,
            head_names=self.head_names,
            use_layernorm=use_layernorm,
        )

        n_total = sum(p.numel() for p in self.parameters())
        n_q1 = sum(p.numel() for p in self.q1.parameters())
        print(f"[ChunkTwinCritic] {n_total/1e6:.2f}M params "
              f"(Q1={n_q1/1e6:.2f}M × 2)"
              f"{' +LayerNorm' if use_layernorm else ''}")

    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state:  (B, D_state) global_cond features
            action: (B, H, Da)   action chunk

        Returns:
            {head_name: (B, 2)} where [:, 0] = Q1, [:, 1] = Q2
        """
        out1 = self.q1(state, action)   # {name: (B, 1)}
        out2 = self.q2(state, action)

        return {
            name: torch.cat([out1[name], out2[name]], dim=-1)  # (B, 2)
            for name in self.head_names
        }

    def q_min(self,
              state: torch.Tensor,
              action: torch.Tensor,
              head: str = "total") -> torch.Tensor:
        """min(Q1, Q2) for Clipped Double Q-learning (§5.3).

        Args:
            state:  (B, D_state)
            action: (B, H, Da)
            head:   Head to take min over (default "total")

        Returns:
            q_min: (B,)  element-wise min of Q1, Q2
        """
        q1_val = self.q1(state, action)[head].squeeze(-1)  # (B,)
        q2_val = self.q2(state, action)[head].squeeze(-1)
        return torch.min(q1_val, q2_val)

    @torch.jit.ignore
    def q1_only(self,
                state: torch.Tensor,
                action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return only Q1 heads (for decomposed target computation).

        Returns: {head_name: (B, 1)}
        """
        return self.q1(state, action)

    @torch.jit.ignore
    def q2_only(self,
                state: torch.Tensor,
                action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return only Q2 heads."""
        return self.q2(state, action)


# ═══════════════════════════════════════════════════════════════════
# Target network helpers
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def target_update(target: ChunkTwinCritic,
                  source: ChunkTwinCritic,
                  tau: float = 0.005):
    """Polyak EMA update for target critic (standard SAC convention).

    target = (1 - tau) * target + tau * source

    Args:
        target: Target critic (updated in-place)
        source: Online critic (current)
        tau:    EMA coefficient (default 0.005). With tau=0.005,
                target slowly tracks online at 0.5% per step.
    """
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


@torch.no_grad()
def hard_update(target: ChunkTwinCritic,
                source: ChunkTwinCritic):
    """B1c: Hard copy online critic → target critic.

    Uses load_state_dict for a complete parameter copy.
    No Polyak blending — target is an exact snapshot of online.

    Args:
        target: Target critic (updated in-place)
        source: Online critic (current)
    """
    target.load_state_dict(source.state_dict())


# ═══════════════════════════════════════════════════════════════════
# Critic loss computation
# ═══════════════════════════════════════════════════════════════════

def compute_critic_loss(
    critic: ChunkTwinCritic,
    target_critic: ChunkTwinCritic,
    state: torch.Tensor,
    action: torch.Tensor,
    reward_total: torch.Tensor,
    reward_heads: Dict[str, torch.Tensor],
    next_state: torch.Tensor,
    next_action: torch.Tensor,
    done: torch.Tensor,
    gamma: float = 0.99,
    betas: Optional[Dict[str, float]] = None,
    reduce: str = "mean",
    # ── R3: Q_progress scale control (§Step 4) ──
    use_huber_progress: bool = False,
    progress_huber_beta: float = 10.0,
    progress_target_clip: Optional[float] = None,
    progress_l2_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute twin critic loss with H-step chunk backup (§5.3-5.4).

    Args:
        critic:           Online twin critic
        target_critic:    Target critic (for stable targets)
        state:            (B, D_state) current global_cond
        action:           (B, H, Da) current action chunk
        reward_total:     (B,) total discounted sum over chunk
        reward_heads:     {head_name: (B,)} per-head discounted sum
        next_state:       (B, D_state) next global_cond
        next_action:      (B, H, Da) next action from current policy
        done:             (B,) terminal flag (1 = done)
        gamma:            Discount factor
        betas:            Per-head loss weights (§5.4 defaults)
        reduce:           "mean" or "sum"
        use_huber_progress:  (R3) Use Huber loss for progress head
        progress_huber_beta: (R3) Huber loss beta for progress head
        progress_target_clip:(R3) Soft clamp on progress target values
        progress_l2_weight:  (R3) L2 penalty on progress head output

    Returns:
        loss:     scalar critic loss
        log_dict: diagnostics dict (includes per-head target/TD for R3)
    """
    betas = betas or DEFAULT_BETAS
    B = state.shape[0]
    gamma_h = gamma ** 8   # H-step power (H=8, matching action chunk)

    # ── Step 1: Target values (§5.3) ──
    with torch.no_grad():
        # Total head: min(Q1, Q2) from target
        q_next_total = target_critic.q_min(next_state, next_action)  # (B,)
        done_float = done.float()  # bool → float for arithmetic
        y_total = reward_total + gamma_h * (1.0 - done_float) * q_next_total

        # Decomposed heads: Q1 from target (singular bar_Q^{(m)})
        # Reward mapping: some critic heads combine multiple reward terms
        #   progress ← base + pbrs
        #   onset    ← onset
        #   future_clearance ← future_clearance
        #   trap     ← trap + flip
        #   support  ← support
        #   guard    ← guard
        HEAD_REWARD_MAP = {
            "progress":          ["base", "pbrs"],
            "onset":             ["onset"],
            "future_clearance":  ["future_clearance"],
            "trap":              ["trap", "flip"],
            "support":           ["support"],
            "guard":             ["guard"],
        }
        q_next_heads = target_critic.q1_only(next_state, next_action)
        y_heads = {}
        y_heads_raw = {}  # R3: store pre-clip targets for logging
        for name in critic.head_names:
            if name == "total":
                continue
            q_next = q_next_heads[name].squeeze(-1)  # (B,)
            # Sum the constituent reward terms
            terms = HEAD_REWARD_MAP.get(name, [name])
            r_head = sum(reward_heads[t] for t in terms)  # (B,)
            y_raw = r_head + gamma_h * (1.0 - done_float) * q_next
            y_heads_raw[name] = y_raw

            # ── R3: Progress head target clip (§Step 4.2) ──
            if name == "progress" and progress_target_clip is not None:
                y_heads[name] = torch.clamp(y_raw, -progress_target_clip, progress_target_clip)
            else:
                y_heads[name] = y_raw

    # ── Step 2: Current Q values ──
    q_current = critic(state, action)  # {name: (B, 2)}

    # ── Step 3: Critic loss (§5.4) ──
    # Total head loss: both critics trained against y_total
    q_total = q_current["total"]  # (B, 2)
    loss_total = F.mse_loss(q_total[:, 0], y_total, reduction=reduce)
    loss_total += F.mse_loss(q_total[:, 1], y_total, reduction=reduce)

    # Decomposed heads loss: weighted by betas
    loss_heads = torch.tensor(0.0, device=state.device)
    progress_l2_penalty = torch.tensor(0.0, device=state.device)
    log_dict = {"critic_loss_total": loss_total.item()}
    for name in critic.head_names:
        if name == "total":
            continue
        beta = betas.get(name, 1.0)
        if beta == 0.0:
            continue
        q_head = q_current[name]  # (B, 2)
        y_h = y_heads[name]       # (B,)

        # ── R3: Progress head Huber loss (§Step 4.1) ──
        if name == "progress" and use_huber_progress:
            loss_h = F.smooth_l1_loss(q_head[:, 0], y_h, beta=progress_huber_beta, reduction=reduce)
            loss_h += F.smooth_l1_loss(q_head[:, 1], y_h, beta=progress_huber_beta, reduction=reduce)
            log_dict["critic_loss_progress_huber"] = loss_h.item()
        else:
            loss_h = F.mse_loss(q_head[:, 0], y_h, reduction=reduce)
            loss_h += F.mse_loss(q_head[:, 1], y_h, reduction=reduce)

        loss_heads = loss_heads + beta * loss_h
        log_dict[f"critic_loss_head_{name}"] = loss_h.item()

        # ── R3: Progress head L2 penalty (§Step 4.3) ──
        if name == "progress" and progress_l2_weight > 0:
            l2_p = (q_head[:, 0].pow(2).mean() + q_head[:, 1].pow(2).mean()) * progress_l2_weight
            progress_l2_penalty = progress_l2_penalty + l2_p

    total_loss = loss_total + loss_heads + progress_l2_penalty

    # Diagnostics
    log_dict["critic_loss_total"] = total_loss.item()
    log_dict["critic_loss_heads"] = loss_heads.item()
    log_dict["critic_loss_mse"] = loss_total.item()
    if progress_l2_weight > 0:
        log_dict["critic_progress_l2_penalty"] = progress_l2_penalty.item()
    with torch.no_grad():
        log_dict["Q_total_mean"] = q_current["total"].mean().item()
        for name in critic.head_names:
            q = q_current[name]
            log_dict[f"Q_{name}_mean"] = q.mean().item()
            log_dict[f"Q_{name}_std"] = q.std().item()
        # ── A2-stab diagnostics (§2.6): reward / target / TD target scale ──
        log_dict["reward_total_mean"] = reward_total.mean().item()
        log_dict["reward_total_std"] = reward_total.std().item()
        log_dict["reward_total_min"] = reward_total.min().item()
        log_dict["reward_total_max"] = reward_total.max().item()
        log_dict["target_q_mean"] = q_next_total.mean().item()
        log_dict["target_q_std"] = q_next_total.std().item()
        log_dict["td_target_mean"] = y_total.mean().item()
        log_dict["td_target_std"] = y_total.std().item()
        log_dict["td_target_min"] = y_total.min().item()
        log_dict["td_target_max"] = y_total.max().item()
        # ── R3: Per-head target/TD diagnostics (§Step 3) ──
        if "progress" in y_heads:
            yp = y_heads["progress"]
            log_dict["target_q_progress_mean"] = yp.mean().item()
            log_dict["target_q_progress_std"] = yp.std().item()
            log_dict["td_target_progress_mean"] = yp.mean().item()  # y_heads already includes reward
            log_dict["td_target_progress_std"] = yp.std().item()
        if "progress" in y_heads_raw:
            ypr = y_heads_raw["progress"]
            log_dict["target_q_progress_raw_mean"] = ypr.mean().item()
            log_dict["target_q_progress_raw_std"] = ypr.std().item()
        # R3: Progress target clip ratio
        if progress_target_clip is not None and "progress" in y_heads_raw:
            ypr = y_heads_raw["progress"]
            ypc = y_heads["progress"]
            clipped_mask = (ypr != ypc)
            log_dict["progress_target_clipped_ratio"] = clipped_mask.float().mean().item()
            log_dict["progress_huber_loss_mean"] = (
                log_dict.get("critic_loss_progress_huber", log_dict.get("critic_loss_head_progress", 0.0))
            )

    return total_loss, log_dict


# ═══════════════════════════════════════════════════════════════════
# Sanity check
# ═══════════════════════════════════════════════════════════════════

def _test_forward():
    """Run forward pass sanity check."""
    from termcolor import cprint

    print("=" * 60)
    print("ChunkTwinCritic — forward sanity test")
    print("=" * 60)

    B = 4
    critic = ChunkTwinCritic(state_dim=1536, da=11, horizon=8)

    state = torch.randn(B, 1536)
    action = torch.randn(B, 8, 11)

    out = critic(state, action)
    total_q = out["total"]  # (B, 2)

    print(f"  Input:  state({state.shape}), action({action.shape})")
    print(f"  Output: {len(out)} heads")
    for name, q in out.items():
        print(f"    {name:20s}: shape={list(q.shape)}, "
              f"Q1_mean={q[:, 0].mean():.3f}, Q1_std={q[:, 0].std():.3f}, "
              f"Q2_mean={q[:, 1].mean():.3f}, Q2_std={q[:, 1].std():.3f}")

    # Test q_min
    qmin = critic.q_min(state, action)
    print(f"  q_min(total): {qmin.shape}, mean={qmin.mean():.3f}")

    # Test q1_only
    q1_out = critic.q1_only(state, action)
    print(f"  q1_only: {len(q1_out)} heads")

    # Test target update
    target = ChunkTwinCritic(state_dim=1536, da=11, horizon=8)
    target_update(target, critic, tau=0.5)
    t_out = target(state, action)
    t_total = t_out["total"]
    diff = (total_q - t_total).abs().mean().item()
    print(f"  target_update(tau=0.5): Q diff={diff:.4f}")

    # Test parameter count
    n_total = sum(p.numel() for p in critic.parameters())
    n_q1 = sum(p.numel() for p in critic.q1.parameters())
    print(f"  Params: total={n_total/1e6:.2f}M, Q1={n_q1/1e6:.2f}M")
    print()

    # Verify no NaN / Inf
    assert torch.isfinite(total_q).all(), "NaN/Inf in critic output"
    print("  ✅ No NaN/Inf in forward output")

    # Verify twin diversity
    q1_vs_q2_diff = (out["total"][:, 0] - out["total"][:, 1]).abs().mean().item()
    assert q1_vs_q2_diff > 0 or B <= 1, "Critics should differ (random init)"
    print(f"  ✅ Twin diversity: Q1-Q2 mean diff={q1_vs_q2_diff:.4f}")

    print(f"\n{'=' * 60}")
    print(f"ALL SANITY CHECKS PASSED")
    print(f"{'=' * 60}")


def _test_critic_loss():
    """Test critic loss computation."""
    from termcolor import cprint

    print("=" * 60)
    print("ChunkTwinCritic — critic loss sanity test")
    print("=" * 60)

    # Use REWARD_TERM_NAMES keys matching the replay buffer collation
    REWARD_TERM_NAMES = [
        "base", "pbrs", "onset", "future_clearance",
        "trap", "flip", "support", "guard",
    ]

    B = 4
    H = 8
    Da = 11

    critic = ChunkTwinCritic(state_dim=1536, da=Da, horizon=H)
    target = ChunkTwinCritic(state_dim=1536, da=Da, horizon=H)
    target.load_state_dict(critic.state_dict())

    state = torch.randn(B, 1536)
    action = torch.randn(B, H, Da)
    next_state = torch.randn(B, 1536)
    next_action = torch.randn(B, H, Da)

    reward_total = torch.randn(B)
    reward_heads = {name: torch.randn(B) for name in REWARD_TERM_NAMES}
    done = torch.zeros(B)

    loss, log = compute_critic_loss(
        critic, target, state, action, reward_total,
        reward_heads, next_state, next_action, done,
    )

    print(f"  loss={loss.item():.4f}")
    for k, v in log.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float)
              else f"    {k}: {v}")

    assert torch.isfinite(loss), "Loss is NaN/Inf"
    print(f"\n  ✅ Critic loss finite, {len(log)} diagnostics logged")

    print(f"\n{'=' * 60}")
    print(f"CRITIC LOSS TEST PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _test_forward()
    print()
    _test_critic_loss()
