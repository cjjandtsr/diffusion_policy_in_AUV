"""R1.2 / R7-3 (guide Task R7-3 + user ruling 2026-08-23): effective-PPO
identity gate at the ACTUAL B=128 minibatch.

History: the old gate reproduced the stored B=1 rollout logprobs at the
B=128 recompute to max_abs_diff <= 1e-5 (per-dim). Measured 2.716e-3
(270x over) with B=1 bitwise exact, CPU at the same magnitude — a
shape-dependent FP32 accumulation-order artifact of the batched kernels,
NOT an algorithmic defect. Ruling 2026-08-23: the raw per-dim
cross-shape diff is downgraded to CROSS_SHAPE_NUMERICAL_DIAGNOSTIC
(recorded, never adjudicating); the adjudicating gate is now the
EFFECTIVE PPO identity — does the stored-B1 old-logprob denominator
change what the vendor ``PPODiffusion.loss`` actually computes and
gradients?

The gate (fixed by the ruling; frozen by the unit tests below):

  - optimizer_step_count == 0
  - actor/base/critic params hash unchanged
  - optimizer state unchanged
  - NaN/Inf == 0 (ratios, logratios, KL, gradients)
  - spurious_clipfrac == 0
  - max(|ratio - 1| / eps_k) <= 0.10
  - ratio_mean in [0.99999, 1.00001]
  - actor-gradient cosine >= 0.99999 (stored-B1 vs same-B128 denominator)
  - actor-gradient relative-norm error <= 1e-3

where ``ratio`` is the VENDOR-EFFECTIVE ratio the loss computes
(third_party/dppo_official_cc7234a/model/diffusion/diffusion_ppo.py
:93-148): clamp per-dim logp to [-5, 2], slice [:,:exec_steps,:], mean
over (exec_steps, action_dim), then exp(new - old) — NOT the per-dim
exp(new-old).mean() of the old gate. ``eps_k`` is the vendor's
per-denoise-index clip schedule (base 1e-3 -> 0.2, rate 3.0,
t = k/(ft-1)).

The diagnostic runs on ONE fixed B=128 minibatch of the real no-update
rollout buffer through the production facade (train_loop.py:2387-2407
mirror): one facade.loss call with the stored B=1 old logprobs (the
production signal), the observer-cached new logprobs (the exact tensor
the loss consumed — official_runtime.py observer_last), then one
facade.loss call with the same-B128 recomputed reference denominator
(the value the fallback materialization would freeze). NO optimizer
step anywhere; parameter / optimizer-state hashes measured around the
whole diagnostic.

Run: pytest -q -m "not gpu" tests/dppo_parity/test_actual_batch_logprob_identity.py
     pytest -q -m gpu tests/dppo_parity/test_actual_batch_logprob_identity.py
Skipped (not failed) when CUDA or the production assets are unavailable.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
# Ruling 2026-08-23: the old 1e-5 raw per-dim identity baseline stays as
# historical evidence (committed RED state) — never clobbered, never
# adjudicating.
BASELINE_JSON = REPO_ROOT / "temp" / "r1_actual_batch_identity_before.json"
REPORTS_R7_DIR = Path(
    os.environ.get(
        "DPPO_REPORTS_R7_DIR",
        "/home/cjj/AUV/program/reports/routeA_rl_env_v1/R7_OfficialCore_Final",
    )
)
EFFECTIVE_IDENTITY_JSON = REPORTS_R7_DIR / "G5_EFFECTIVE_PPO_IDENTITY.json"
# The BLOCKED evidence (localization / memory breakdown / why-no-in-task-fix)
# stays untouched; its raw full-buffer measurement is quoted in the report.
BLOCKED_EVIDENCE_JSON = (
    REPORTS_R7_DIR / "04_r7_3_blocked" / "r7_3_blocked_evidence.json"
)

# ── fixed gate thresholds (user ruling 2026-08-23; unit tests freeze) ──
RATIO_MIN, RATIO_MAX = 0.99999, 1.00001
MAX_RATIO_DEV_OVER_EPS = 0.10
GRAD_COSINE_MIN = 0.99999
GRAD_REL_NORM_MAX = 1e-3
# vendor PPODiffusion.loss hyperparameters (diffusion_ppo.py:43-45 +
# :151-157; facade defaults — production passes no overrides)
CLIP_PLOSS_COEF = 0.2
CLIP_PLOSS_COEF_BASE = 1e-3
CLIP_PLOSS_COEF_RATE = 3.0


def _clean_measurements(**overrides: float) -> dict:
    """All-gates-met measurement dict; tests override one gate at a time."""
    m: Dict[str, object] = {
        "optimizer_step_count": 0,
        "params_hash_unchanged": True,
        "optimizer_state_unchanged": True,
        "nonfinite_count": 0,
        "spurious_clipfrac": 0.0,
        "max_ratio_dev_over_eps": 0.05,
        "ratio_mean": 1.0,
        "actor_grad_cosine": 0.999995,
        "actor_grad_rel_norm_error": 1e-5,
    }
    m.update(overrides)
    return m


def assess_effective_ppo_identity(m: dict) -> dict:
    """Adjudicate the effective-PPO identity over a measurement dict.

    ``m`` keys (all required, all produced by the diagnostic test):
      optimizer_step_count (int) — the diagnostic must never step
      params_hash_unchanged (bool) — actor(base)+base UNet+critic
      optimizer_state_unchanged (bool) — actor/critic optimizer state
      nonfinite_count (int) — NaN/Inf across ratios/logratios/KL/grads
      spurious_clipfrac (float) — production clip fraction caused by the
          cross-shape denominator (the same-shape reference ratio is 1.0,
          so ANY production clipping is spurious by definition)
      max_ratio_dev_over_eps (float) — max |ratio-1|/eps_k over the
          minibatch, per-element vendor eps_k
      ratio_mean (float) — vendor-effective ratio mean
      actor_grad_cosine (float) — stored-B1 vs same-B128 denominator
      actor_grad_rel_norm_error (float) — ||g1-g2|| / ||g1||

    Returns:
      {"failed": bool, "first_failed": str|None,
       "gates": {name: {"value": ..., "threshold": ..., "passed": bool}}}
    """
    gates = {
        "optimizer_step_count": {
            "value": m["optimizer_step_count"],
            "threshold": 0,
            "passed": m["optimizer_step_count"] == 0,
        },
        "params_hash_unchanged": {
            "value": bool(m["params_hash_unchanged"]),
            "threshold": True,
            "passed": bool(m["params_hash_unchanged"]),
        },
        "optimizer_state_unchanged": {
            "value": bool(m["optimizer_state_unchanged"]),
            "threshold": True,
            "passed": bool(m["optimizer_state_unchanged"]),
        },
        "nonfinite_count": {
            "value": m["nonfinite_count"],
            "threshold": 0,
            "passed": m["nonfinite_count"] == 0,
        },
        "spurious_clipfrac": {
            "value": m["spurious_clipfrac"],
            "threshold": 0.0,
            "passed": m["spurious_clipfrac"] == 0.0,
        },
        "max_ratio_dev_over_eps": {
            "value": m["max_ratio_dev_over_eps"],
            "threshold": MAX_RATIO_DEV_OVER_EPS,
            "passed": m["max_ratio_dev_over_eps"] <= MAX_RATIO_DEV_OVER_EPS,
        },
        "ratio_mean": {
            "value": m["ratio_mean"],
            "threshold": [RATIO_MIN, RATIO_MAX],
            "passed": RATIO_MIN <= m["ratio_mean"] <= RATIO_MAX,
        },
        "actor_grad_cosine": {
            "value": m["actor_grad_cosine"],
            "threshold": GRAD_COSINE_MIN,
            "passed": m["actor_grad_cosine"] >= GRAD_COSINE_MIN,
        },
        "actor_grad_rel_norm_error": {
            "value": m["actor_grad_rel_norm_error"],
            "threshold": GRAD_REL_NORM_MAX,
            "passed": m["actor_grad_rel_norm_error"] <= GRAD_REL_NORM_MAX,
        },
    }
    order = [
        "optimizer_step_count", "params_hash_unchanged",
        "optimizer_state_unchanged", "nonfinite_count",
        "spurious_clipfrac", "max_ratio_dev_over_eps", "ratio_mean",
        "actor_grad_cosine", "actor_grad_rel_norm_error",
    ]
    first_failed = next((name for name in order if not gates[name]["passed"]), None)
    return {"failed": first_failed is not None, "first_failed": first_failed,
            "gates": gates}


# ═══════════════════════════════════════════════════════════════════
# Gate unit tests (TDD RED: assess_effective_ppo_identity missing).
# ═══════════════════════════════════════════════════════════════════

def test_gate_fails_on_optimizer_step() -> None:
    m = _clean_measurements(optimizer_step_count=1)
    v = assess_effective_ppo_identity(m)
    assert v["failed"] and v["first_failed"] == "optimizer_step_count"


def test_gate_fails_on_params_hash_change() -> None:
    m = _clean_measurements(params_hash_unchanged=False)
    v = assess_effective_ppo_identity(m)
    assert v["failed"] and v["first_failed"] == "params_hash_unchanged"


def test_gate_fails_on_optimizer_state_change() -> None:
    m = _clean_measurements(optimizer_state_unchanged=False)
    v = assess_effective_ppo_identity(m)
    assert v["failed"] and v["first_failed"] == "optimizer_state_unchanged"


def test_gate_fails_on_nonfinite() -> None:
    m = _clean_measurements(nonfinite_count=2)
    v = assess_effective_ppo_identity(m)
    assert v["failed"] and v["first_failed"] == "nonfinite_count"


def test_gate_fails_on_spurious_clipping() -> None:
    m = _clean_measurements(spurious_clipfrac=1e-4)
    v = assess_effective_ppo_identity(m)
    assert v["failed"] and v["first_failed"] == "spurious_clipfrac"


def test_gate_fails_when_ratio_exceeds_eps_window() -> None:
    m = _clean_measurements(max_ratio_dev_over_eps=MAX_RATIO_DEV_OVER_EPS + 1e-6)
    v = assess_effective_ppo_identity(m)
    assert v["failed"] and v["first_failed"] == "max_ratio_dev_over_eps"


@pytest.mark.parametrize("ratio_mean", [RATIO_MIN - 1e-6, RATIO_MAX + 1e-6])
def test_gate_fails_on_ratio_mean_outside(ratio_mean: float) -> None:
    m = _clean_measurements(ratio_mean=ratio_mean)
    v = assess_effective_ppo_identity(m)
    assert v["failed"] and v["first_failed"] == "ratio_mean"


def test_gate_fails_on_grad_cosine_below() -> None:
    m = _clean_measurements(actor_grad_cosine=GRAD_COSINE_MIN - 1e-6)
    v = assess_effective_ppo_identity(m)
    assert v["failed"] and v["first_failed"] == "actor_grad_cosine"


def test_gate_fails_on_grad_rel_norm_above() -> None:
    m = _clean_measurements(actor_grad_rel_norm_error=GRAD_REL_NORM_MAX * 1.1)
    v = assess_effective_ppo_identity(m)
    assert v["failed"] and v["first_failed"] == "actor_grad_rel_norm_error"


def test_gate_passes_when_all_gates_met() -> None:
    v = assess_effective_ppo_identity(_clean_measurements())
    assert not v["failed"] and v["first_failed"] is None
    assert all(g["passed"] for g in v["gates"].values())


# ═══════════════════════════════════════════════════════════════════
# Helpers shared by the GPU diagnostic.
# ═══════════════════════════════════════════════════════════════════

def _stack_obs(obs_list, device: str):
    """(N, T_o, ...) batched obs tensors — mirrors trainer._stack_obs."""
    return {
        k: torch.from_numpy(np.stack([t[k] for t in obs_list]))
        .float()
        .to(device)
        for k in obs_list[0]
    }


def _state_hash(module: torch.nn.Module) -> str:
    """Deterministic sha256 over a module's state dict (param pin)."""
    h = hashlib.sha256()
    for key in sorted(module.state_dict()):
        h.update(key.encode("utf-8"))
        h.update(
            np.ascontiguousarray(
                module.state_dict()[key].detach().cpu().numpy()
            ).tobytes()
        )
    return h.hexdigest()


def _policy_hash(policy) -> str:
    """Deterministic sha256 over the policy state dict (BC weight check)."""
    return _state_hash(policy.model) if hasattr(policy, "model") else _state_hash(policy)


def _vendor_eps_k(denoising_inds: torch.Tensor, ft: int) -> torch.Tensor:
    """Per-element clip epsilon — vendor diffusion_ppo.py:151-157 replica:
    t = k/(ft-1); eps = base + (clip-base) * (exp(rate*t)-1)/(exp(rate)-1)."""
    k_norm = denoising_inds.float() / (ft - 1)
    return CLIP_PLOSS_COEF_BASE + (CLIP_PLOSS_COEF - CLIP_PLOSS_COEF_BASE) * (
        torch.exp(CLIP_PLOSS_COEF_RATE * k_norm) - 1.0
    ) / (math.exp(CLIP_PLOSS_COEF_RATE) - 1.0)


def _vendor_effective_mean(
    logp: torch.Tensor, exec_steps: int
) -> torch.Tensor:
    """Vendor reduction replica (diffusion_ppo.py:93-102): clamp per-dim
    logp to [-5,2], slice [:,:exec_steps,:], mean over (-1,-2) -> (B,)."""
    return (
        logp.clamp(min=-5.0, max=2.0)[:, :exec_steps, :].mean(dim=(-1, -2))
    )


# ═══════════════════════════════════════════════════════════════════
# The GPU diagnostic (adjudicating test, user ruling 2026-08-23).
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.gpu
def test_effective_ppo_identity_no_optimizer_step(
    real_policy_lora, real_rollout_buffer
):
    """Ruling 2026-08-23 adjudicating gate on ONE fixed B=128 minibatch.

    Production facade.loss with the stored B=1 old logprobs (the
    production signal) vs the same-B128 recomputed reference denominator
    (the fallback materialization's frozen value). NO optimizer step —
    parameter/optimizer-state hashes measured around the whole
    diagnostic. The raw per-dim cross-shape diff is recorded as
    CROSS_SHAPE_NUMERICAL_DIAGNOSTIC only.
    """
    from diffusion_policy_3d.rl.dppo_uav.gae import compute_gae
    from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
        fill_buffer_next_values,
    )
    from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
        OfficialValueHead,
        build_official_phase1a_facade,
        make_official_cond,
    )
    from diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter import (
        _obs_conditions,
    )
    from diffusion_policy_3d.rl.dppo_uav.policy_loader import (
        extract_lora_state,
    )
    from diffusion_policy_3d.rl.dppo_uav.train_loop import lora_checksum_of

    policy, lora_info, device = real_policy_lora
    buffer = real_rollout_buffer
    contract = buffer._contract
    n = len(buffer)
    ft = int(contract.ft_denoising_steps)
    exec_steps = int(contract.exec_steps)
    batch_size = 128
    assert batch_size <= n * ft

    # ── one FIXED B=128 minibatch: official per-epoch randperm, seed 0 ─
    torch.manual_seed(0)
    total = n * ft
    inds_k = torch.randperm(total, device=device)[:batch_size]
    batch_inds_b, denoising_inds_b = torch.unravel_index(inds_k, (n, ft))
    obs_t = _stack_obs(buffer.obs, device)
    chains_t = torch.from_numpy(np.stack(buffer.chains)).float().to(device)
    obs_b = {k: obs_t[k][batch_inds_b] for k in obs_t}
    chains_prev_b = chains_t[batch_inds_b, denoising_inds_b]
    chains_next_b = chains_t[batch_inds_b, denoising_inds_b + 1]
    # stored B=1 rollout old logprobs — written once at rollout time
    old_logprobs_b = (
        torch.from_numpy(np.stack(buffer.logprobs)).float().to(device)
    )[batch_inds_b, denoising_inds_b]
    # real advantages: production GAE over the stored buffer
    # (trainer.py:292-301; the fixture rollout ran without a reward
    # scaler, so reward_scaler=None is the production-identical path).
    # The no_update rollout never post-fills next_values — fill them
    # exactly like the one_update path (train_loop.py:2373-2378) with
    # the rollout-time critic over each transition's OWN next_obs,
    # chunked by the production default, before GAE.
    fill_buffer_next_values(
        buffer, buffer._critic, chunk_size=128
    )
    advantages_np, returns_np = compute_gae(
        np.stack(buffer.rewards),
        np.stack(buffer.values),
        np.stack(buffer.next_values),
        np.stack(buffer.terminated),
        np.stack(buffer.truncated),
        list(buffer.episode_ids),
        contract.gamma_env,
        contract.gae_lambda,
    )
    advantages_b = (
        torch.from_numpy(advantages_np).float().to(device)[batch_inds_b]
    )
    returns_b = (
        torch.from_numpy(returns_np).float().to(device)[batch_inds_b]
    )
    values_b = (
        torch.from_numpy(np.stack(buffer.values)).float().to(device)[
            batch_inds_b
        ]
    )

    # ── production facade (train_loop.py:2387-2407 mirror) ─────────────
    policy.noise_scheduler.set_timesteps(contract.denoising_steps)
    critic = buffer._critic  # the rollout-time critic (production identity)
    facade = build_official_phase1a_facade(
        model=policy.model,
        scheduler=policy.noise_scheduler,
        ft_denoising_steps=ft,
        cfg_lambda=float(policy.cfg_lambda),
        device=device,
        horizon_steps=contract.action_horizon,
        action_dim=contract.action_dim,
        eta=1.0,
        min_sampling_denoising_std=buffer._min_logprob_denoising_std,
        min_logprob_denoising_std=buffer._min_logprob_denoising_std,
        critic=OfficialValueHead(critic),
        gamma_denoising=contract.gamma_denoising,
        clip_vloss_coef=None,
        norm_adv=True,
        randn_clip_value=10.0,
    )
    cond_b = make_official_cond(*_obs_conditions(policy, obs_b), device)

    # ── safety guards AROUND the whole diagnostic ──────────────────────
    lora_ids = {id(p) for p in lora_info["lora_params"]}
    base_params = [
        p for p in policy.model.parameters() if id(p) not in lora_ids
    ]
    assert base_params, "base UNet params must exist"
    model_hash_before = _state_hash(policy.model)
    critic_hash_before = _state_hash(critic)
    lora_checksum_before = lora_checksum_of(extract_lora_state(lora_info))
    # production optimizers exist as objects but are NEVER stepped
    actor_optimizer = torch.optim.AdamW(lora_info["lora_params"], lr=1e-3)
    critic_optimizer = torch.optim.AdamW(
        critic.value_head_parameters(), lr=1e-3
    )
    assert len(actor_optimizer.state_dict()["state"]) == 0
    assert len(critic_optimizer.state_dict()["state"]) == 0
    optimizer_step_count = 0

    # ── production loss call: stored B=1 old-logprob denominator ───────
    pg, ent, v, clipfrac_v, approx_kl_v, ratio_mean_v, _bc, eta_mean = (
        facade.loss(
            cond_b, chains_prev_b, chains_next_b, denoising_inds_b,
            returns_b, values_b, advantages_b, old_logprobs_b,
            use_bc_loss=False, reward_horizon=exec_steps,
        )
    )
    # observer cache: the EXACT new-logprob tensor the loss consumed
    observer = facade.observer_last
    assert observer is not None and observer["get_ent"] is True, (
        "observer cache must be populated by the vendor loss call"
    )
    new_logprobs_b = observer["log_prob"]  # (B, H, D)
    assert new_logprobs_b.shape == old_logprobs_b.shape, (
        f"observer log_prob {tuple(new_logprobs_b.shape)} vs stored "
        f"{tuple(old_logprobs_b.shape)}"
    )
    # production loss composition (trainer.py:417: ent_coef=0, vf_coef=1)
    loss_prod = pg + v

    # ── vendor-effective ratio (stored-B1 denominator) ─────────────────
    new_m = _vendor_effective_mean(new_logprobs_b, exec_steps)
    old_m = _vendor_effective_mean(old_logprobs_b, exec_steps)
    logratio = new_m - old_m
    ratio = logratio.exp()
    eps_k = _vendor_eps_k(denoising_inds_b, ft)
    dev_over_eps = (ratio - 1.0).abs() / eps_k
    clip_state = (ratio - 1.0).abs() > eps_k
    k3_kl = ((ratio - 1.0) - logratio).mean()
    # the replica must be bitwise identical to the vendor's own returns
    # (same tensors, same ops, same shapes)
    assert float(k3_kl) == approx_kl_v, (
        f"replica k3 KL {float(k3_kl)} != vendor approx_kl {approx_kl_v}"
    )
    assert float(clip_state.float().mean()) == clipfrac_v, (
        f"replica clipfrac {float(clip_state.float().mean())} != vendor "
        f"clipfrac {clipfrac_v}"
    )
    assert float(ratio.mean()) == ratio_mean_v, (
        f"replica ratio mean {float(ratio.mean())} != vendor {ratio_mean_v}"
    )

    # raw per-dim cross-shape diagnostic (RECORDED, never adjudicating)
    raw_max_abs_diff = float(
        (new_logprobs_b - old_logprobs_b).abs().max().item()
    )
    raw_ratio_mean_per_dim = float(
        torch.exp(new_logprobs_b - old_logprobs_b).mean().item()
    )

    # ── same-B128 reference denominator (frozen recompute — the value
    #    the fallback materialization would cache) ──────────────────────
    ref_old_b = new_logprobs_b.detach()
    # fresh cond encode: each loss call runs on its own production-grade
    # forward (trainer.py:384 encodes per minibatch); deterministic
    # same-shape mode makes the two conds bitwise identical.
    cond_b2 = make_official_cond(*_obs_conditions(policy, obs_b), device)
    pg2, ent2, v2, _cf2, _kl2, ratio_mean_v2, _bc2, _eta2 = facade.loss(
        cond_b2, chains_prev_b, chains_next_b, denoising_inds_b,
        returns_b, values_b, advantages_b, ref_old_b,
        use_bc_loss=False, reward_horizon=exec_steps,
    )
    loss_ref = pg2 + v2
    assert ratio_mean_v2 == 1.0, (
        f"reference ratio mean {ratio_mean_v2} != 1.0 — the B=128 "
        "recomputed old logprobs must equal the loss's own new logprobs "
        "(same inputs, same kernels, same frozen weights)"
    )

    # ── actor gradients: stored-B1 vs same-B128 denominator ────────────
    actor_params = list(lora_info["lora_params"])
    g_prod = torch.autograd.grad(loss_prod, actor_params)
    g_ref = torch.autograd.grad(loss_ref, actor_params)
    flat_prod = torch.cat([g.reshape(-1) for g in g_prod])
    flat_ref = torch.cat([g.reshape(-1) for g in g_ref])
    assert flat_prod.norm().item() > 0.0, "production actor gradient is zero"
    assert torch.isfinite(flat_prod).all(), "non-finite production gradient"
    assert torch.isfinite(flat_ref).all(), "non-finite reference gradient"
    grad_cosine = float(
        torch.nn.functional.cosine_similarity(flat_prod, flat_ref, dim=0)
    )
    grad_rel_norm = float(
        (flat_prod - flat_ref).norm().item() / flat_prod.norm().item()
    )

    # ── spurious clipping: the same-shape reference ratio is exactly 1.0
    #    (never clipped), so ANY production clipping is by definition
    #    caused by the cross-shape denominator ──────────────────────────
    spurious_clipfrac = float(clip_state.float().mean().item())

    # ── safety guards AFTER: hashes / optimizer state ──────────────────
    assert model_hash_before == _state_hash(policy.model), (
        "policy model params changed across the diagnostic"
    )
    assert critic_hash_before == _state_hash(critic), (
        "critic params changed across the diagnostic"
    )
    assert lora_checksum_before == lora_checksum_of(
        extract_lora_state(lora_info)
    ), "LoRA checksum changed across the diagnostic"
    assert len(actor_optimizer.state_dict()["state"]) == 0, (
        "actor optimizer state non-empty — the diagnostic must not step"
    )
    assert len(critic_optimizer.state_dict()["state"]) == 0, (
        "critic optimizer state non-empty — the diagnostic must not step"
    )
    nonfinite_count = int(
        (~torch.isfinite(ratio)).sum().item()
        + (~torch.isfinite(logratio)).sum().item()
        + int(not torch.isfinite(k3_kl))
        + int(not torch.isfinite(flat_prod).all())
        + int(not torch.isfinite(flat_ref).all())
    )

    # ── assemble + adjudicate (fixed gates) ────────────────────────────
    measurements = {
        "optimizer_step_count": optimizer_step_count,
        "params_hash_unchanged": True,
        "optimizer_state_unchanged": True,
        "nonfinite_count": nonfinite_count,
        "spurious_clipfrac": spurious_clipfrac,
        "max_ratio_dev_over_eps": float(dev_over_eps.max().item()),
        "ratio_mean": float(ratio.mean().item()),
        "actor_grad_cosine": grad_cosine,
        "actor_grad_rel_norm_error": grad_rel_norm,
    }
    verdict = assess_effective_ppo_identity(measurements)
    if verdict["failed"]:
        raise AssertionError(
            "effective-PPO identity gates FAILED (first: "
            f"{verdict['first_failed']}):\n"
            + "\n".join(
                f"  {name}: value={g['value']} threshold={g['threshold']} "
                f"passed={g['passed']}"
                for name, g in verdict["gates"].items()
            )
        )

    # ── report (only reached when every gate passes) ───────────────────
    ratio_quantiles = [
        float(q)
        for q in torch.quantile(
            ratio, torch.tensor([0.50, 0.95, 0.99], device=device)
        ).tolist()
    ]
    per_k = {}
    for k in range(ft):
        mask = denoising_inds_b == k
        count = int(mask.sum().item())
        if count > 0:
            per_k[str(k)] = {
                "count": count,
                "eps_k": float(_vendor_eps_k(
                    torch.tensor([k], device=device), ft
                )[0]),
                "max_abs_ratio_minus_1": float(
                    (ratio[mask] - 1.0).abs().max().item()
                ),
                "max_dev_over_eps": float(dev_over_eps[mask].max().item()),
                "n_clipped": int(clip_state[mask].sum().item()),
            }
    report = {
        "gate": "G5_EFFECTIVE_PPO_IDENTITY",
        "ruling": "user ruling 2026-08-23: effective-PPO identity at the "
                  "actual B=128 minibatch; raw cross-shape per-dim diff "
                  "downgraded to CROSS_SHAPE_NUMERICAL_DIAGNOSTIC",
        "requirement": "LOGP-BATCH-ID-EFFECTIVE-PPO",
        "torch_version": torch.__version__,
        "device": device,
        "batch_size": batch_size,
        "seed": 0,
        "ft_denoising_steps": ft,
        "exec_steps": exec_steps,
        "policy_cfg_lambda": float(policy.cfg_lambda),
        "vendor_loss": "PPODiffusion.loss (diffusion_ppo.py) — clamp[-5,2], "
                       "slice [:,:exec_steps,:], mean(-1,-2), exp(new-old)",
        "clip_schedule": {
            "base": CLIP_PLOSS_COEF_BASE,
            "coef": CLIP_PLOSS_COEF,
            "rate": CLIP_PLOSS_COEF_RATE,
            "per_denoise_index": per_k,
        },
        "cross_shape_numerical_diagnostic": {
            "semantics": "recorded only — never adjudicating",
            "raw_per_dim_max_abs_diff_minibatch": raw_max_abs_diff,
            "raw_per_dim_ratio_mean_minibatch": raw_ratio_mean_per_dim,
            "full_buffer_reference": json.loads(
                BLOCKED_EVIDENCE_JSON.read_text()
            )["gate_measurement"],
        },
        "vendor_effective_ratio": {
            "mean": float(ratio.mean().item()),
            "p50": ratio_quantiles[0],
            "p95": ratio_quantiles[1],
            "p99": ratio_quantiles[2],
            "max": float(ratio.max().item()),
            "max_abs_ratio_minus_1_over_eps": float(
                dev_over_eps.max().item()
            ),
        },
        "k3_approx_kl": float(k3_kl),
        "spurious_clipfrac": spurious_clipfrac,
        "gradient_comparison": {
            "semantics": "production actor gradient (stored-B1 denominator) "
                         "vs same-B128 reference denominator, production "
                         "loss composition pg + vf_coef*v (ent_coef=0)",
            "cosine": grad_cosine,
            "rel_norm_error": grad_rel_norm,
            "prod_norm": float(flat_prod.norm().item()),
            "ref_norm": float(flat_ref.norm().item()),
        },
        "safety": {
            "optimizer_step_count": optimizer_step_count,
            "params_hash_unchanged": True,
            "optimizer_state_unchanged": True,
            "nonfinite_count": nonfinite_count,
        },
        "verdict": {
            "failed": verdict["failed"],
            "first_failed": verdict["first_failed"],
            "gates": {
                name: {"value": g["value"], "threshold": g["threshold"],
                       "passed": g["passed"]}
                for name, g in verdict["gates"].items()
            },
        },
        "policy_hash": _policy_hash(policy),
        "lora_checksum": lora_checksum_before,
        "numerical_mode": {
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
        },
    }
    EFFECTIVE_IDENTITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    EFFECTIVE_IDENTITY_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"G5_EFFECTIVE_PPO_IDENTITY written: {EFFECTIVE_IDENTITY_JSON}")
    print(f"  raw cross-shape per-dim max_abs_diff: {raw_max_abs_diff:.3e}")
    print(f"  vendor-effective ratio mean: {float(ratio.mean()):.9f}")
    print(f"  max(|ratio-1|/eps): {float(dev_over_eps.max()):.3e}")
    print(f"  grad cosine: {grad_cosine:.9f}  rel-norm: {grad_rel_norm:.3e}")


def test_formal_numerical_mode_disables_tf32_and_enables_determinism():
    """R1.2 config gate: the formal mode must resolve to TF32-off +
    deterministic (RED until numerics.py exists — import error)."""
    from diffusion_policy_3d.rl.dppo_uav.numerics import (  # noqa: WPS433
        NumericalMode,
        configure_numerical_mode,
    )

    resolved = configure_numerical_mode(NumericalMode())
    assert resolved["allow_tf32_matmul"] is False
    assert resolved["allow_tf32_cudnn"] is False
    assert resolved["deterministic_algorithms"] is True
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.are_deterministic_algorithms_enabled() is True
