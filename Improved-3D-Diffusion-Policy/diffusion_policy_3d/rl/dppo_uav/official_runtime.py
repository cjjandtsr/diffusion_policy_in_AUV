"""official_runtime.py — registered LoRA actor backends + OfficialPhase1aFacade.

Guide R7-2B: the canonical train/rollout/recompute/loss path executes the
FROZEN official DPPO runtime. This module owns the UAV-side adapters:

- ``Phase1aOfficialActorBackend`` — epsilon-predicting actor wrapping the
  Phase1a model: CFG mixing (cond/uncond cat-batched forward), sample→
  epsilon Tweedie conversion, and the registered LoRA bundle's base/FT
  context gate (base backend runs ``mode(False)``, FT ``mode(True)``);
- ``OfficialPhase1aFacade`` — duck-typed stand-in whose methods are PURE
  unbound delegations into the vendor ``VPGDiffusion``/``PPODiffusion``;
  no Gaussian/DDIM/PPO math lives here (guide Step 6);
- ``build_official_phase1a_facade`` — the sanctioned factory: builds the
  two backends over one shared model and the scheduler-derived DDIM
  tables the vendor methods read.

``_model_output_to_epsilon`` lives here (moved from dp_sampler) to break
the import cycle: dp_sampler imports the facade, so the epsilon
conversion cannot stay in dp_sampler. dp_sampler re-exports it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from contextlib import nullcontext
from typing import Dict, Optional

from diffusion_policy_3d.rl.dppo_uav.vendor_loader import (  # noqa: F401
    OfficialDPPOSymbols,
    load_official_dppo_symbols as _load_official_dppo_symbols,
)


class OfficialValueHead(nn.Module):
    """ValueCritic head wrapper for the official loss cond interface.

    The vendor ``PPODiffusion.loss`` calls ``self.critic(obs).view(-1)``
    where obs is the flat cond dict (``global_cond`` (B, To·D)); the
    project ValueCritic takes raw (B, To, D) obs and re-encodes
    internally. The trainer's ``encode_cond`` produces
    cond == PolicyObsEncoder(policy)(obs) (phase1a_actor_adapter.
    _obs_conditions), so this wrapper pools the cond and feeds the
    shared mlp directly — bitwise equal to ValueCritic(raw obs) (same
    pooling, same shared mlp object the value-head optimizer owns; the
    encoder is never re-run).

    The wrapped ValueCritic is held WITHOUT nn.Module registration
    (``object.__setattr__``): its mlp parameters are owned by the
    critic optimizer directly, and the wrapper must not double-register
    them (state_dict duplication / optimizer-id mismatch).
    """

    def __init__(self, critic: "ValueCritic") -> None:
        super().__init__()
        object.__setattr__(self, "_critic", critic)

    def forward(self, obs_cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        global_cond = obs_cond["global_cond"]  # (B, To*D) flat
        b = global_cond.shape[0]
        pooled = global_cond.reshape(
            b, self._critic.contract.obs_steps, self._critic.frame_dim
        ).mean(dim=1)
        return self._critic.mlp(pooled.detach()).reshape(-1)

# ── epsilon conversion (moved from dp_sampler) ──────────────────────

def _model_output_to_epsilon(x_t, model_output, t, scheduler, device):
    """Convert model output to epsilon representation.

    - prediction_type='epsilon': model_output IS epsilon → return as-is
    - prediction_type='sample': model_output IS x_0 → derive epsilon via Tweedie
    """
    pred_type = scheduler.config.prediction_type
    if pred_type == 'epsilon':
        return model_output
    elif pred_type == 'sample':
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        if torch.is_tensor(t) and t.ndim > 0:
            # batched timesteps (B,) -> (B, 1, 1) to broadcast against
            # (B, T, Da) samples (T5: per-step logp recompute path)
            alpha_bar_t = alpha_bar_t.reshape(
                (-1,) + (1,) * (x_t.ndim - 1)
            )
        # ε = (x_t - √(ᾱ_t) · x̂_0) / √(1-ᾱ_t)
        eps = (x_t - torch.sqrt(alpha_bar_t) * model_output) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
        return eps
    else:
        raise ValueError(f"Unknown prediction_type: {pred_type}")


def _require_eta_one(eta: float) -> None:
    """The official-DDIM runtime is eta=1; anything else is a
    silent-ignore hazard — hard-fail instead (guide Step 5)."""
    if float(eta) != 1.0:
        raise ValueError(
            f"official DPPO DDIM runtime is eta=1; got eta={eta} — a "
            "different DDIM eta needs its own pre-registered algorithm "
            "branch, not a silent parameter"
        )


class _OnesEta(nn.Module):
    """Eta module fixed to ones (B, action_dim) on the facade device.

    The vendor computes ``sigma = etas * factor`` with ``factor`` on the
    x device — so eta must emit on the facade device, NOT on the cond
    device (a CPU eta with CUDA x would raise).
    """

    def __init__(self, action_dim: int, device) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.device = device

    def forward(self, cond) -> torch.Tensor:
        # the vendor calls eta(cond) with the (possibly repeated) cond
        # dict — the batch is the first key's leading dim.
        if isinstance(cond, dict):
            first = next(iter(cond.values()))
            batch = first.shape[0]
        else:
            batch = cond.shape[0]
        return torch.ones(batch, self.action_dim, device=self.device)


def make_official_cond(
    global_cond: torch.Tensor,
    global_cond_uncond: Optional[torch.Tensor],
    device,
) -> Dict[str, torch.Tensor]:
    """Build the official cond dict the vendor methods consume.

    All keys are moved to ``device`` — the vendor repeats/indexes cond
    alongside x; a CPU cond with CUDA x would raise.
    """
    cond: Dict[str, torch.Tensor] = {"global_cond": global_cond.to(device)}
    if global_cond_uncond is not None:
        cond["global_cond_uncond"] = global_cond_uncond.to(device)
    return cond


# ── Phase1a official actor backend (guide R7-2B Step 4) ─────────────

class Phase1aOfficialActorBackend(nn.Module):
    """Epsilon-predicting actor backend wrapping a Phase1a model.

    Reads ``global_cond`` / ``global_cond_uncond`` from the cond dict;
    when ``cfg_lambda > 0`` AND an uncond branch is present, runs one
    cat-batched cond/uncond forward and mixes
    ``o = o_u + λ·(o_c − o_u)``. The model output is converted to
    epsilon (``prediction_type='sample'`` checkpoints go through the
    Tweedie conversion; the vendor ``p_mean_var`` consumes epsilon only).
    LoRA base/FT is gated by the registered bundle's context gate:
    the base backend runs ``mode(False)``, the FT backend ``mode(True)``
    (guide Step 4: both backends share ONE model).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        scheduler,
        cfg_lambda: float,
        bundle: Optional[nn.Module],
        enabled: bool,
        device,
    ) -> None:
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.cfg_lambda = cfg_lambda
        self._bundle = bundle
        self._enabled = enabled
        self.device = device

    def _gate(self):
        """Context gate: bundle.mode(enabled); nullcontext without LoRA."""
        if self._bundle is None:
            return nullcontext()
        return self._bundle.mode(self._enabled)

    def forward(
        self,
        x: torch.Tensor,  # (B, H, Da)
        t: torch.Tensor,  # (B,) long — scheduler timestep per element
        cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:  # epsilon (B, H, Da)
        global_cond = cond.get("global_cond")
        if global_cond is None:
            raise RuntimeError(
                "Phase1aOfficialActorBackend cond must carry 'global_cond'"
            )
        global_cond_uncond = cond.get("global_cond_uncond")
        if self.cfg_lambda > 0 and global_cond_uncond is None:
            raise RuntimeError(
                f"CFG lambda={self.cfg_lambda} > 0 but cond has no "
                "'global_cond_uncond' — fail-closed, no silent CFG-off"
            )
        use_cfg = self.cfg_lambda > 0 and global_cond_uncond is not None
        with self._gate():
            if use_cfg:
                gc = torch.cat(
                    [global_cond, global_cond_uncond.to(x.device)], dim=0
                )
                xb = torch.cat([x, x], dim=0)
                tb = torch.cat([t, t], dim=0)
                mo = self.model(
                    sample=xb, timestep=tb, local_cond=None, global_cond=gc
                )
                mo_cond, mo_uncond = mo.chunk(2, dim=0)
                model_output = mo_uncond + self.cfg_lambda * (
                    mo_cond - mo_uncond
                )
            else:
                model_output = self.model(
                    sample=x, timestep=t, local_cond=None,
                    global_cond=global_cond.to(x.device),
                )
        return _model_output_to_epsilon(
            x, model_output, t, self.scheduler, x.device
        )


# ── OfficialPhase1aFacade (guide R7-2B Step 5-6) ────────────────────

class OfficialPhase1aFacade:
    """Duck-typed stand-in exposing the frozen official DPPO runtime.

    Holds exactly the attributes the vendor methods read (DDIM tables,
    actor backends, eta, PPO hyperparameters). Every method is a PURE
    unbound delegation into the vendor classes — the facade contains no
    training math (guide Step 6, verbatim shape).
    """

    def __init__(
        self,
        *,
        symbols: OfficialDPPOSymbols,
        actor: Phase1aOfficialActorBackend,
        actor_ft: Phase1aOfficialActorBackend,
        scheduler,
        device,
        ft_denoising_steps: int,
        eta: nn.Module,
        min_sampling_denoising_std: float,
        min_logprob_denoising_std: float,
        horizon_steps: int,
        action_dim: int,
        critic: Optional[nn.Module] = None,
        gamma_denoising: float = 0.8,
        clip_ploss_coef: float = 0.2,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3.0,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0.0,
        clip_advantage_upper_quantile: float = 1.0,
        norm_adv: bool = True,
        randn_clip_value: float = 10.0,
    ) -> None:
        self._symbols = symbols
        self.device = device
        self.actor = actor
        self.actor_ft = actor_ft
        self.critic = critic
        self.eta = eta
        self.use_ddim = True
        # DDIM tables bound to the facade device — the vendor
        # extract/gather and arange math require index/table/value on one
        # device (guide Step 5; scheduler.timesteps moved to device).
        alphas_cumprod = scheduler.alphas_cumprod.to(device)
        self.ddim_t = scheduler.timesteps.to(device)
        self.ddim_steps = len(self.ddim_t)
        if ft_denoising_steps > self.ddim_steps:
            raise ValueError(
                f"ft_denoising_steps={ft_denoising_steps} exceeds the "
                f"DDIM chain length {self.ddim_steps} — the official "
                "likelihood window cannot cover more transitions than "
                "the chain has (fail closed, no silent mis-timestep)"
            )
        self.ddim_alphas = alphas_cumprod[self.ddim_t]
        if (self.ddim_alphas >= 1.0).any():
            raise ValueError(
                "alpha == 1.0 encountered in the DDIM alpha table — the "
                "vendor sigma factor (1-alpha_prev)/(1-alpha) would be "
                "0/0 NaN; the scheduler must not hit set_alpha_to_one"
            )
        self.ddim_alphas_prev = torch.cat([
            alphas_cumprod[self.ddim_t[1:]],
            torch.ones(1, device=device),
        ])
        self.ddim_sqrt_one_minus_alphas = (1.0 - self.ddim_alphas).sqrt()
        self.predict_epsilon = True
        self.ft_denoising_steps = ft_denoising_steps
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        # clip: None keeps the vendor's raw Normal — the legacy project
        # kernel also defaulted to None (dppo_transition_kernel), and the
        # R7-2 Step-10 parity oracle was constructed with None; UAV 11d
        # actions are unnormalized, so value-clipping would silently bias
        # the likelihood surface. randn_clip_value instead mirrors the
        # vendor forward()'s shell-noise clamp (±10, diffusion.py) at the
        # DDIM sampling step (dp_sampler clamps the draw to it).
        self.denoised_clip_value = None
        self.eps_clip_value = None
        self.randn_clip_value = randn_clip_value
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.gamma_denoising = gamma_denoising
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile
        self.norm_adv = norm_adv
        # R7-3 Step 7: observer cache — the trainer reads the LAST
        # get_logprobs_subsample I/O (the vendor loss call) to derive
        # diagnostics in no_grad; the cache never participates in the
        # loss/backward. Single slot, replaced on every call.
        self.observer_last: Optional[Dict] = None

    # Step 6: pure unbound delegation — no training math below.

    def p_mean_var(self, *args, **kwargs):
        return self._symbols.VPGDiffusion.p_mean_var(self, *args, **kwargs)

    def get_logprobs(self, *args, **kwargs):
        return self._symbols.VPGDiffusion.get_logprobs(self, *args, **kwargs)

    def get_logprobs_subsample(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """Vendor delegation + observer cache (guide R7-3 Step 7).

        Signature mirrors the vendor VPGDiffusion.get_logprobs_subsample
        exactly; the call I/O is cached in ``self.observer_last``
        (single slot, replaced on the next call) so the trainer derives
        observer metrics (per-denoise KL/clipfrac, per-element ratios)
        in ``torch.no_grad()`` from the SAME tensors the vendor loss
        consumed — without duplicating the training formula and without
        providing a second policy loss. The vendor loss calls this with
        ``get_ent=True`` (returns ``(log_prob, eta)``); other callers
        (dp_sampler step-logprob) use the default.
        """
        out = self._symbols.VPGDiffusion.get_logprobs_subsample(
            self,
            obs,
            chains_prev,
            chains_next,
            denoising_inds,
            get_ent=get_ent,
            use_base_policy=use_base_policy,
        )
        if get_ent:
            log_prob, eta = out
        else:
            log_prob, eta = out, None
        self.observer_last = {
            "obs": obs,
            "chains_prev": chains_prev,
            "chains_next": chains_next,
            "denoising_inds": denoising_inds,
            "log_prob": log_prob,
            "eta": eta,
            "get_ent": get_ent,
        }
        return out

    def loss(self, *args, use_bc_loss=False, **kwargs):
        if use_bc_loss:
            raise RuntimeError(
                "R7 formal baseline has BC loss disabled; enable only through "
                "a separately reviewed official-teacher branch"
            )
        return self._symbols.PPODiffusion.loss(
            self, *args, use_bc_loss=False, **kwargs
        )


# ── sanctioned factory ──────────────────────────────────────────────

_SYMBOLS: Optional[OfficialDPPOSymbols] = None


def _get_symbols() -> OfficialDPPOSymbols:
    """Load (and cache) the verified official symbols.

    ``load_official_dppo_symbols`` re-verifies all 401 vendor files on
    every call (~50-100ms); the trainer builds a facade per microbatch
    (128 per update), so the verified symbols are cached once per
    process. Vendor verification still runs on first use.
    """
    global _SYMBOLS
    if _SYMBOLS is None:
        _SYMBOLS = _load_official_dppo_symbols()
    return _SYMBOLS


def build_official_phase1a_facade(
    *,
    model: nn.Module,
    scheduler,
    ft_denoising_steps: int,
    cfg_lambda: float,
    device,
    horizon_steps: int,
    action_dim: int,
    eta: float = 1.0,
    min_sampling_denoising_std: float = 0.1,
    min_logprob_denoising_std: float = 0.1,
    critic: Optional[nn.Module] = None,
    gamma_denoising: float = 0.8,
    clip_ploss_coef: float = 0.2,
    clip_ploss_coef_base: float = 1e-3,
    clip_ploss_coef_rate: float = 3.0,
    clip_vloss_coef: Optional[float] = None,
    clip_advantage_lower_quantile: float = 0.0,
    clip_advantage_upper_quantile: float = 1.0,
    norm_adv: bool = True,
    randn_clip_value: float = 10.0,
) -> OfficialPhase1aFacade:
    """Build the official phase-1a facade over one shared model.

    Two backends wrap the SAME model: the base actor runs the registered
    LoRA bundle with the gate OFF, the FT actor with it ON (guide Step
    4). The LoRA bundle is resolved from the model's registered
    ``dppo_lora_adapters`` submodule; without a bundle (pure base
    policy) both backends run ungated — identical outputs.

    Args:
        model: Phase1a policy model (eps/sample predictor).
        scheduler: HF scheduler with ``timesteps`` / ``alphas_cumprod`` /
            ``config.prediction_type``.
        ft_denoising_steps: fine-tune window size (last K steps).
        cfg_lambda: CFG strength; > 0 requires cond to carry an uncond
            branch (fail-closed otherwise).
        device: facade/table device (== the x device the vendor runs on).
        horizon_steps / action_dim: action chunk / dim of the chain.
        eta: DDIM stochasticity — hard-requires 1.0 (ValueError).
        critic, gamma_denoising, clip_*: PPO hyperparameters read by the
            vendor loss (wired by R7-3).

    Returns:
        OfficialPhase1aFacade instance (duck-typed vendor runtime).
    """
    _require_eta_one(eta)
    bundle = getattr(model, "dppo_lora_adapters", None)
    base_actor = Phase1aOfficialActorBackend(
        model=model, scheduler=scheduler, cfg_lambda=cfg_lambda,
        bundle=bundle, enabled=False, device=device,
    )
    actor_ft = Phase1aOfficialActorBackend(
        model=model, scheduler=scheduler, cfg_lambda=cfg_lambda,
        bundle=bundle, enabled=True, device=device,
    )
    return OfficialPhase1aFacade(
        symbols=_get_symbols(),
        actor=base_actor,
        actor_ft=actor_ft,
        scheduler=scheduler,
        device=device,
        ft_denoising_steps=ft_denoising_steps,
        eta=_OnesEta(action_dim=action_dim, device=device),
        min_sampling_denoising_std=min_sampling_denoising_std,
        min_logprob_denoising_std=min_logprob_denoising_std,
        horizon_steps=horizon_steps,
        action_dim=action_dim,
        critic=critic,
        gamma_denoising=gamma_denoising,
        clip_ploss_coef=clip_ploss_coef,
        clip_ploss_coef_base=clip_ploss_coef_base,
        clip_ploss_coef_rate=clip_ploss_coef_rate,
        clip_vloss_coef=clip_vloss_coef,
        clip_advantage_lower_quantile=clip_advantage_lower_quantile,
        clip_advantage_upper_quantile=clip_advantage_upper_quantile,
        norm_adv=norm_adv,
        randn_clip_value=randn_clip_value,
    )
