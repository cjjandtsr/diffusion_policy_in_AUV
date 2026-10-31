"""Phase1a actor adapter — maps full-trajectory outputs to policy/exec horizons.

Design: RouteA_RL_Training_System_A_OfficialCore_Design_v1_0 §4.5-4.6.
- a0 = action_start = 2: policy action occupies full[:, a0:a0+H_a].
- executed_action = policy_action[:, :exec_steps] (EQUIV-REG OPEN).
- Region slicing happens ONCE at the adapter boundary (design §4.6 option 2):
  downstream official core sees action tensors indexed from 0.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav.chain_logprobs import (
    per_dim_denoising_logprobs,
    per_step_denoising_logp,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder
from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import RolloutTransition
from diffusion_policy_3d.rl.dppo_uav.vendor_loader import (  # noqa: F401
    load_official_dppo_symbols as _load_official_dppo_symbols,
)


@dataclass
class DPPORolloutSample:
    """One rollout sample in DPPO layout (design §4.5).

    The region-sliced pair is named chains_policy /
    old_logprobs (matching §4.11 RolloutTransition and the five-field §4.5
    taxonomy); full-horizon fields (chains_full / logprobs_full) are not
    produced — logprobs are computed on the region window only (design §4.6
    option 2): window i maps to full-step k = K - K_f + i, i=0 = noisiest
    fine-tune step.
    """

    full_trajectory: torch.Tensor  # (B, H_f, 11)
    policy_action: torch.Tensor  # (B, H_a, 11)
    executed_action: torch.Tensor  # (B, A_e, 11)
    chains_policy: torch.Tensor  # (B, K_f+1, H_a, 11) — region window
    # R5.8-G (N10): three explicit, unambiguous index axes — the old
    # single ambiguous ``denoising_indices`` field is REMOVED from the
    # sample (the standalone helper is deprecated archive-only).
    optimizer_denoise_indices: torch.Tensor  # (K_f,) ascending 0..K_f-1
    scheduler_timesteps: torch.Tensor  # (K_f,) scheduler.timesteps[-K_f:]
    full_chain_indices: torch.Tensor  # (K_f,) ascending K-K_f..K-1
    # Old logprobs are computed at rollout time per transition (B=1, the
    # behavior batch) with the policy's CFG lambda — never recomputed or
    # overwritten afterwards. No deferred B=128 storage pass may redefine
    # buffer old logprobs post-rollout (guide Step 3).
    old_logprobs: Optional[torch.Tensor] = None  # (B, K_f, H_a, 11)
    obs_condition: Optional[torch.Tensor] = None  # (B, T_o*192)


def split_trajectory(
    full_trajectory: torch.Tensor,
    contract: RLSystemContract,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a (B, H_f, 11) full trajectory into policy and executed actions.

    Returns:
        (policy_action (B, H_a, 11), executed_action (B, A_e, 11)).
    """
    a0 = contract.action_start
    policy_action = full_trajectory[:, a0 : a0 + contract.action_horizon, :]
    executed_action = policy_action[:, : contract.exec_steps, :]
    return policy_action, executed_action


def slice_executed_logprobs(
    policy_logprobs: torch.Tensor,
    contract: RLSystemContract,
) -> torch.Tensor:
    """Slice policy-horizon logprobs (B, K_f, H_a, 11) to executed region.

    Design §4.6 option 2: executed_logprobs = policy_logprobs[..., :A_e, :].
    """
    return policy_logprobs[..., : contract.exec_steps, :]


def denoising_indices(
    contract: RLSystemContract, device: str = "cpu"
) -> torch.Tensor:
    """DEPRECATED (R5.8-G, N10): archive-only helper — the ambiguous
    descending field was replaced by the sample's three explicit axes
    (optimizer_denoise_indices / scheduler_timesteps /
    full_chain_indices). No production consumer.

    Legacy DDPM-era metadata (review partA P2-2: this field is
    scheduler-timestep-ordered DESCENDING metadata and is NOT the
    optimizer denoise index used by the trainer — the trainer's axis is
    the chain-window order 0..K_f-1 with gamma^(K_f-1-k) discounting;
    the kernel's optimizer_denoise_index is authoritative). No current
    production consumer; kept for record compatibility only.
    Official DPPO t_single: arange(K_f-1, -1, -1) — index 0 is the noisiest
    fine-tune step (t=K_f-1), index K_f-1 is the cleanest (t=0).

    Matches official diffusion_vpg.py get_logprobs (i=0 -> t=K_f-1 -> smallest
    denoising discount gamma^(K_f-1); i=K_f-1 -> gamma^0=1). Audit-corrected
    direction (2026-07-28 report was inverted).

    Note: official DDIM branch uses ddim_t[-K_f:]; identical to arange(K_f-1,
    -1, -1) only when ddim_steps == denoising_steps. This project runs DDPM
    (K=10, K_f=5), which matches. Pass ``device`` explicitly when the tensor
    will feed GPU loss computation.
    """
    return torch.arange(
        contract.ft_denoising_steps - 1, -1, -1, dtype=torch.long, device=device
    )


def _require_policy_cfg_lambda(policy) -> float:
    """Resolve the policy's CFG lambda with hard fails on absence and on
    any deviation from the formal pin 1.5.

    ``getattr(policy, "cfg_lambda", 0.0)`` would silently evaluate the
    pure-cond distribution for a policy missing the attribute. Formal
    Route A v1 requires the lambda to be EXPLICIT on the policy.

    This is the ONLY resolution point — the formal adapters
    (sample_dppo_rollout / make_get_new_logprobs / _obs_conditions) do not
    accept an external cfg_lambda, because an override lets a policy at
    λ=1.5 store old logprobs computed at another λ while the behavior
    chain was sampled at 1.5 (the fake-ratio defect). Hard fails, in order:
      - missing attribute       -> AttributeError
      - non-finite value        -> ValueError
      - anything != 1.5 exactly -> ValueError (math.isclose with zero
        tolerance; the formal pin is exact — phase1a_policy's default, the
        value the behavior rollout and both likelihood paths run at).
        Pure-cond contrast experiments must call the LOW-LEVEL
        per_dim_denoising_logprobs / per_step_denoising_logp directly.
    """
    lam = getattr(policy, "cfg_lambda", None)
    if lam is None:
        raise AttributeError(
            "formal Route A v1 requires an explicit CFG lambda on the"
            " policy (policy.cfg_lambda); found none."
        )
    lam = float(lam)
    if not math.isfinite(lam):
        raise ValueError(
            "policy.cfg_lambda must be finite, got " + repr(lam)
        )
    if not math.isclose(lam, 1.5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "formal CFG lambda is pinned to 1.5 (the value the behavior"
            f" rollout and both likelihood paths run at); got"
            f" policy.cfg_lambda = {lam}. Pure-cond contrast experiments"
            " must call the low-level per_dim_denoising_logprobs /"
            " per_step_denoising_logp directly."
        )
    return lam


def _obs_conditions(
    policy,
    obs_b: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """(global_cond, global_cond_uncond) for the CFG logp path.

    The likelihood path must use the SAME canonical CFG computation as the
    behavior policy's sampling, so old
    (rollout, B=1) and new (ppo_update microbatch=1) logprobs both
    evaluate the CFG-mixed distribution. ``cond`` is the shared
    PolicyObsEncoder path (normalize -> use_pc_color slice -> per-frame
    reshape -> obs_encoder); ``uncond`` mirrors sample_ddpm_with_logprob's
    CFG construction line-for-line (quaternion zeroed on the NORMALIZED
    obs, [:, :, 3:7]). uncond is None when the CFG lambda is 0 — the
    legacy pure-cond path.

    NO cfg_lambda parameter — the λ resolves exclusively through
    ``_require_policy_cfg_lambda`` (the formal pin 1.5). Any other
    resolution path is the fake-ratio defect.
    """
    cond = PolicyObsEncoder(policy)(obs_b)  # (B, T_o*192)
    lam = _require_policy_cfg_lambda(policy)
    if lam <= 0:
        return cond, None
    nobs = policy.normalizer.normalize(obs_b)
    if not policy.use_pc_color:
        nobs = {**nobs, "point_cloud": nobs["point_cloud"][..., :3]}
    nobs_u = {k: v.clone() for k, v in nobs.items()}
    nobs_u["agent_pos"][:, :, 3:7] = 0.0  # zero the quaternion (sampling path)
    this_u = {
        key: value[:, : policy.n_obs_steps].reshape(-1, *value.shape[2:])
        for key, value in nobs_u.items()
    }
    b = obs_b["agent_pos"].shape[0]
    uncond = policy.obs_encoder(this_u).reshape(b, -1)
    return cond, uncond


def sample_dppo_rollout(
    policy,
    obs_b: Dict[str, torch.Tensor],
    contract: RLSystemContract,
    eta: float = 1.0,
    min_sampling_denoising_std: float = 0.1,
    min_logprob_denoising_std: float = 0.1,
) -> DPPORolloutSample:
    """Roll out the phase1a policy: full K+1 chain -> region-sliced window.

    Runs the frozen policy's sample_ddpm_with_logprob (proven E040 path; the
    internal x[:, :n_obs_steps] slice is an identity when obs_b holds the
    T_o selected frames) and recomputes per-dim per-step logps on the region
    window with the SAME model weights at rollout time (B=1, the behavior
    batch).

    Old logprobs are computed here — per transition, at batch 1, with the
    policy's CFG lambda — and written to the buffer once, never recomputed
    or overwritten afterwards (no deferred B=128 storage pass). The λ is
    the policy's own ``cfg_lambda`` (behavior CFG), so the stored logps
    evaluate the same CFG-mixed distribution the sampling path used.

    §3.4: the sampling noise floor
    (``min_sampling_denoising_std``, official diffusion_vpg default 0.1) and
    the likelihood-evaluation floor (``min_logprob_denoising_std``, 0.1) are
    two explicit parameters — one value must not serve both semantics.

    Hard gates:
      - rollout_policy_batch_size == 1 (formal Route A v1): any B>1
        rollout raises ValueError — the behavior policy never rolls out
        batched;
      - the CFG lambda is resolved ONCE via ``_require_policy_cfg_lambda``
        and feeds BOTH the uncond construction and the likelihood.

    No cfg_lambda parameter — an external override is the fake-ratio defect
    (chain sampled at policy λ, old logprobs computed at the override λ).
    The λ resolves exclusively from the policy and is pinned to 1.5.
    """
    b_rollout = next(iter(obs_b.values())).shape[0]
    if b_rollout != 1:
        raise ValueError(
            "formal Route A v1 requires rollout_policy_batch_size == 1;"
            f" got batch {b_rollout}"
        )
    lam = _require_policy_cfg_lambda(policy)
    x_t, _, chain = policy.sample_ddpm_with_logprob(
        obs_dict=obs_b,
        eta=eta,
        min_sampling_denoising_std=min_sampling_denoising_std,
        min_logprob_denoising_std=min_logprob_denoising_std,
        num_inference_steps=contract.denoising_steps,
        ft_denoising_steps=contract.ft_denoising_steps,
    )
    full_trajectory = x_t  # (B, H_f, 11) — final denoised sample
    policy_action, executed_action = split_trajectory(full_trajectory, contract)
    stacked = torch.stack(chain)  # (K+1, B, H_f, 11)
    a0 = contract.action_start
    window = stacked[
        contract.denoising_steps - contract.ft_denoising_steps :, :,
        a0 : a0 + contract.action_horizon,
    ].permute(1, 0, 2, 3)  # (B, K_f+1, H_a, 11)
    cond, uncond = _obs_conditions(policy, obs_b)
    if lam > 0 and uncond is None:
        raise RuntimeError(
            f"CFG lambda {lam} > 0 but the uncond condition was not"
            " constructed — mixing would be silently dead"
        )
    logprobs = per_dim_denoising_logprobs(
        chain=window,
        timesteps=policy.noise_scheduler.timesteps,
        model=policy.model,
        global_cond=cond,
        global_cond_uncond=uncond,
        cfg_lambda=lam,
        scheduler=policy.noise_scheduler,
        eta=eta,
        min_sampling_denoising_std=min_sampling_denoising_std,
        min_logprob_denoising_std=min_logprob_denoising_std,
        ft_denoising_steps=contract.ft_denoising_steps,
    )  # (B, K_f, H_a, 11)
    k_f = contract.ft_denoising_steps
    k_full = contract.denoising_steps
    offset = k_full - k_f
    return DPPORolloutSample(
        full_trajectory=full_trajectory,
        policy_action=policy_action,
        executed_action=executed_action,
        chains_policy=window,  # region window (H_a)
        # R5.8-G (N10): the three explicit index axes
        optimizer_denoise_indices=torch.arange(
            k_f, dtype=torch.long, device=window.device),
        scheduler_timesteps=policy.noise_scheduler.timesteps[-k_f:]
            .to(window.device),
        full_chain_indices=torch.arange(
            offset, k_full, dtype=torch.long, device=window.device),
        old_logprobs=logprobs,  # region logps (H_a)
        obs_condition=cond,
    )


def make_get_new_logprobs(
    policy,
    contract: RLSystemContract,
    eta: float = 1.0,
    min_logprob_denoising_std: float = 0.1,
):
    """Official get_logprobs_subsample interface for trainer.ppo_update.

    (obs_b, chains_prev_b, chains_next_b, denoising_inds_b) -> (B, H_a, D_a):
    per-dim logp of the selected denoising step (full-step k = K - K_f + d)
    recomputed with the CURRENT model weights. Same math, same canonical CFG
    path (``_obs_conditions``) and same defaults as sample_dppo_rollout, so
    the ratio is 1 on the rollout's own chains — the trainer calls it with
    microbatch=1, the exact behavior batch the rollout used.

    §3.4: likelihood evaluation only — the single std parameter is
    ``min_logprob_denoising_std`` (no sampling std here).

    No cfg_lambda parameter — an external override is the fake-ratio defect.
    The λ resolves exclusively from the policy (pinned to 1.5) via
    ``_require_policy_cfg_lambda``; a policy without cfg_lambda raises
    AttributeError, a non-1.5 policy raises ValueError — at factory time, so
    the trainer's likelihood recompute runs the SAME λ the rollout stored
    its old logprobs at (rollout CFG == update CFG).
    """
    policy.noise_scheduler.set_timesteps(contract.denoising_steps)
    timesteps = policy.noise_scheduler.timesteps
    offset = contract.denoising_steps - contract.ft_denoising_steps
    # resolved ONCE at factory time via the hard-fail helper
    lam = _require_policy_cfg_lambda(policy)

    def get_new_logprobs(
        obs_b, chains_prev_b, chains_next_b, denoising_inds_b
    ) -> torch.Tensor:
        # R5.8-G (B-3): the update recompute must use the SAME schedule
        # the factory resolved — a set_timesteps() with a different K
        # between factory and update would silently mis-map every k.
        cur = policy.noise_scheduler.timesteps
        if not torch.equal(cur, timesteps):
            raise RuntimeError(
                "scheduler timesteps changed since the likelihood "
                "factory was built — the update recompute would use a "
                "different schedule than the rollout"
            )
        k_b = offset + denoising_inds_b  # (B,) full-chain step per element
        cond, uncond = _obs_conditions(policy, obs_b)
        return per_step_denoising_logp(
            x_k=chains_prev_b,
            x_next=chains_next_b,
            k_batch=k_b,
            timesteps=timesteps,
            model=policy.model,
            global_cond=cond,
            global_cond_uncond=uncond,
            cfg_lambda=lam,
            scheduler=policy.noise_scheduler,
            eta=eta,
            min_sampling_denoising_std=min_logprob_denoising_std,
            min_logprob_denoising_std=min_logprob_denoising_std,
            ft_denoising_steps=contract.ft_denoising_steps,
        )

    return get_new_logprobs


def to_rollout_transition(
    sample: DPPORolloutSample,
    obs_np: Dict[str, np.ndarray],
    next_obs_np: Dict[str, np.ndarray],
    reward: float,
    terminated: bool,
    truncated: bool,
    value_old: float,
    episode_id: str,
    episode_step: int,
    world_uid: str,
    role: str,
    manifest_entry_hash: str,
    episode_config_hash: str,
    observation_valid: bool,
    contract: RLSystemContract,
    env_id: int = 0,
    done_reason: str = "none",
    success: bool = False,
    collision: bool = False,
    # R7-5 dual hashes: source_manifest_hash (manifest identity) +
    # runtime_episode_hash (env's ACTUAL post-reset state). None on mock
    # paths; populated transitions are hard-checked in buffer.add().
    source_manifest_hash: Optional[str] = None,
    runtime_episode_hash: Optional[str] = None,
) -> RolloutTransition:
    """Adapters' sample -> §4.11 buffer transition (region-sliced, numpy).

    The sample always carries rollout-time B=1 logprobs (no deferred
    batched pass); the value is written once at add time, never
    overwritten. Both hashes (manifest source + sampled episode) ride the
    transition so the buffer trace is bound to the manifest and the exact
    episode.
    """
    assert sample.chains_policy.shape[1:] == (
        contract.ft_denoising_steps + 1,
        contract.action_horizon,
        contract.action_dim,
    ), sample.chains_policy.shape
    assert sample.old_logprobs is not None, (
        "rollout-time old logprobs required (R1.2 rework: computed at"
        " B=1 in sample_dppo_rollout)"
    )
    logprobs_np = sample.old_logprobs[0].detach().cpu().numpy()
    assert logprobs_np.shape == (
        contract.ft_denoising_steps,
        contract.action_horizon,
        contract.action_dim,
    ), logprobs_np.shape
    return RolloutTransition(
        obs=obs_np,
        next_obs=next_obs_np,
        chains_policy=sample.chains_policy[0].detach().cpu().numpy(),
        old_logprobs=logprobs_np,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        value_old=value_old,
        episode_id=episode_id,
        episode_step=episode_step,
        world_uid=world_uid,
        role=role,
        manifest_entry_hash=manifest_entry_hash,
        episode_config_hash=episode_config_hash,
        observation_valid=observation_valid,
        env_id=env_id,
        done_reason=done_reason,
        success=success,
        collision=collision,
        source_manifest_hash=source_manifest_hash,
        runtime_episode_hash=runtime_episode_hash,
    )
