"""e2e loop: env -> gate -> obs -> actor -> buffer -> GAE/update -> ckpt -> eval.

Closes every dppo_uav entry point in one callable (design §4.5-4.17).
Duck-typed env contract (mock in tests/, real wiring in T7):
  reset(task_specs, seeds) -> (obs dict (1, T_h, ...), info list)
  step(executed_actions (1, A_e, D_a) physical) -> (obs, reward, terminated,
      truncated, info list)
  cam_far property; current_episode_config() -> dict for the §4.17 record.

Gate routing (see task_plan decision, [JUDGMENT] for §6.4):
  - reset frame invalid -> REJECT_EPISODE (rule 1)
  - first invalid frame during rollout -> SAFE_END_EPISODE (rule 3): the
    faulting transition is dropped and the episode ends — RolloutBuffer
    invariant 3 (next_obs[t] == obs[t+1]) forbids mid-episode gaps.
  - EXCLUDE_TRANSITION is unreachable in-loop (any invalid frame is the
    first fault and SAFE_ENDs), kept defensive with RuntimeError.
  - HARD_FAILURE cannot trip in the formal phase (rate=None); RuntimeError.

"""

import base64
import copy
import hashlib
import json
import logging
import os
import pickle
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_policy_3d.env.episode_outcome import terminal_category
from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    SCHEMA_VERSION,
    CheckpointBuilder,
    CheckpointData,
    CheckpointError,
    RuntimeIdentity,
    build_environment_rng_state,
    build_lora_identity,
    checkpoint_filename,
    config_hash_of,
    contract_to_checkpoint_bytes,
    contract_hash_of,
    lora_checksum_of,
    _live_lora_axes,
    validate_checkpoint_lora_usable,
    validate_environment_rng_state,
    validate_lora_identity,
    validate_resume_identity,
    validate_v6_contract,
    validate_vendor_identity,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    AdmissionExhausted,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
    ROLE_NAMES,
    RoleSampler,
)
from diffusion_policy_3d.rl.dppo_uav.metrics import role_stats_report
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    EpisodePairedRecord,
    EvalArtifact,
    PairedBankEntry,
    bank_id_from_entries,
    build_eval_artifact,
    paired_eval_artifact_name,
    write_paired_eval_json,
)
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    OfflineDensity,
    compute_capability_metrics,
    compute_capability_outcomes,
)
from diffusion_policy_3d.rl.dppo_uav.observation_adapter import (
    ObservationSelection,
    select_latest_observations,
)
from diffusion_policy_3d.rl.dppo_uav.paired_rng import (
    RNG_CONTRACT_VERSION,
    paired_rng_context,
    policy_noise_seed,
)
from diffusion_policy_3d.rl.dppo_uav.reward_norm import (
    RewardNormState,
    RewardScaleConfig,
    RunningRewardScaler,
    load_scaler_state,
    validate_reward_scale_state,
)
from diffusion_policy_3d.rl.dppo_uav.training_schedules import (
    LRSchedulerConfig,
    SCHEDULER_STATE_VERSION,
    build_lr_schedulers,
    load_scheduler_state,
    scheduler_state_dict,
    step_scheduler_pair,
    validate_scheduler_state,
)
from diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter import (
    sample_dppo_rollout,
    to_rollout_transition,
    _obs_conditions,
    _require_policy_cfg_lambda,
)
from diffusion_policy_3d.rl.dppo_uav.pointcloud_gate import (
    InvalidFramePolicy,
    PHASE_GATES,
    PointCloudValidityGate,
    build_observation_window_mapping,
    validate_policy_observation_window,
)
from diffusion_policy_3d.rl.dppo_uav.policy_loader import (
    extract_lora_state,
    restore_lora_state,
)
from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
    RolloutBuffer,
    fill_buffer_next_values,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import (
    TrainerConfig,
    ppo_update,
)
from diffusion_policy_3d.rl.dppo_uav.numerics import (
    NumericalMode,
    assert_numerical_mode_configured,
    configure_numerical_mode,
)
from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
    OfficialValueHead,
    build_official_phase1a_facade,
    make_official_cond,
)
from diffusion_policy_3d.rl.dppo_uav.optimizer_contract import (
    ResolvedOptimizerConfig,
    resolved_optimizer_config,
    validate_optimizer_resume_config,
)

logger = logging.getLogger(__name__)

OFFICIAL_DPPO_COMMIT = "cc7234ad7ff39a8f32de3af903606723a16f0648"
# Formal numerical mode — TF32 off + deterministic algorithms, applied at
# the top of run_e2e_batch so every production rollout/update runs under
# the same reproducible mode (identity gate §5.7). Resolved values are
# recorded in the checkpoint exact config.
FORMAL_NUMERICAL_MODE = NumericalMode()


class EpisodeCollectionIncomplete(RuntimeError):
    """Structured failure: attempt_limit exhausted before the batch
    completed n_episodes.

    The loop NEVER returns a partial batch as success — the caller
    (entrypoint) must treat this as a failed batch (no checkpoint, no
    paired eval). Fields carry the truthful counters for diagnosis:
    voided episodes ended on an observation fault and are NOT completed.
    """

    def __init__(
        self,
        *,
        requested_episodes: int,
        attempted_episodes: int,
        completed_episodes: int,
        rejected_episodes: int,
        voided_episodes: int,
        admission_blocked_episodes: int,
        attempt_limit: int,
    ) -> None:
        super().__init__(
            f"episode collection incomplete: completed "
            f"{completed_episodes}/{requested_episodes} after "
            f"{attempted_episodes} attempts (limit {attempt_limit}; "
            f"rejected {rejected_episodes}, voided {voided_episodes},"
            f" admission-blocked {admission_blocked_episodes})"
        )
        self.requested_episodes = requested_episodes
        self.attempted_episodes = attempted_episodes
        self.completed_episodes = completed_episodes
        self.rejected_episodes = rejected_episodes
        self.voided_episodes = voided_episodes
        self.admission_blocked_episodes = admission_blocked_episodes
        self.attempt_limit = attempt_limit


@dataclass(frozen=True)
class LoopConfig:
    """e2e batch loop configuration (device is the single pass-through).

    §3.4: min_sampling_denoising_std (rollout sampling noise floor) and
    min_logprob_denoising_std (likelihood evaluation floor) are explicit —
    recorded in the checkpoint's exact resolved config.
    """

    mode: str  # "no_update" | "one_update"
    n_episodes: int
    max_ep_len: int
    device: str
    output_dir: str
    eval_n_episodes: int = 0
    gate_phase: str = "formal"
    min_sampling_denoising_std: float = 0.1  # official diffusion_vpg default
    min_logprob_denoising_std: float = 0.1  # official diffusion_vpg default
    # R5.3 (review F3): fixed environment-transition budget — when set,
    # the collection loop stops at len(buffer) >= target_transitions and
    # the ACTIVE slots (mid-episode) stay active across update
    # boundaries (rollout_cut: the cut transition bootstraps like
    # truncated, but the slot is NOT closed).
    # P0-1 fix (2026-08-17): target_transitions is the PER-BATCH
    # remaining budget (a run-level accounting knob — NOT hashed);
    # total_transition_budget is the absolute run-level total (constant
    # across batches, hashed); rollout_transitions_per_update is the
    # fixed per-update collection horizon (constant, hashed). A batch
    # collects min(horizon, total - committed) transitions.
    # R6-A (F-01/F-02, §11.2.2): the explicit, hashable collection mode
    # — "episode" (total/horizon/target all None; the ONLY stop
    # condition is completed episodes) or "transition" (total+horizon
    # positive; target derived as min(horizon, total - committed)).
    # Resolved in __post_init__ (None -> derived from the total budget);
    # the two stop conditions are mutually exclusive by construction —
    # there is NO target-only compatibility path.
    target_transitions: Optional[int] = None
    total_transition_budget: Optional[int] = None
    rollout_transitions_per_update: Optional[int] = None
    collection_mode: Optional[str] = None  # resolved to "episode"|"transition"
    # R6-F (F-07 → R6-TUNE-1, §11.7): the scheduler and reward-scaling
    # branches. EVERY config field enters the resolved config hash
    # (_config_dict subkeys — a resume under a different schedule/scaling
    # is a different run, refused at the identity gate); the checkpoint
    # carries only mutable state. Defaults are the none branches (LR
    # constant, raw training reward) — the pre-R6-F behavior, now with
    # structured state.
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    reward_scale: RewardScaleConfig = field(default_factory=RewardScaleConfig)

    def __post_init__(self) -> None:
        assert self.mode in ("no_update", "one_update"), self.mode
        assert self.n_episodes >= 1, self.n_episodes
        assert self.max_ep_len >= 1, self.max_ep_len
        # a negative count must not be silently accepted as "evaluation
        # disabled" — eval_n_episodes < 0 is a config error, not a
        # disabled-eval signal (use 0).
        assert self.eval_n_episodes >= 0, self.eval_n_episodes
        assert self.gate_phase in PHASE_GATES, self.gate_phase
        assert self.min_sampling_denoising_std > 0.0, (
            f"min_sampling_denoising_std={self.min_sampling_denoising_std} "
            "must be > 0"
        )
        assert self.min_logprob_denoising_std > 0.0, (
            f"min_logprob_denoising_std={self.min_logprob_denoising_std} "
            "must be > 0"
        )
        if self.total_transition_budget is not None:
            assert self.total_transition_budget >= 1, (
                f"total_transition_budget={self.total_transition_budget} "
                "must be >= 1"
            )
        if self.rollout_transitions_per_update is not None:
            assert self.rollout_transitions_per_update >= 1, (
                f"rollout_transitions_per_update="
                f"{self.rollout_transitions_per_update} must be >= 1"
            )
        # R6-A: resolve the explicit collection mode BEFORE the budget
        # validations so they can branch on it (frozen dataclass —
        # object.__setattr__ is the sanctioned post-init write).
        if self.collection_mode is None:
            object.__setattr__(
                self, "collection_mode",
                "transition" if self.total_transition_budget is not None
                else "episode",
            )
        if self.collection_mode not in ("episode", "transition"):
            raise AssertionError(
                f"collection_mode={self.collection_mode!r} must be "
                "'episode' | 'transition' (R6-A §11.2.2)"
            )
        if self.collection_mode == "episode":
            if (self.total_transition_budget is not None
                    or self.rollout_transitions_per_update is not None
                    or self.target_transitions is not None):
                raise AssertionError(
                    "episode collection forbids budget fields — total, "
                    "horizon and target must all be None (R6-A §11.2.2)"
                )
        else:
            if self.total_transition_budget is None:
                raise AssertionError(
                    "transition collection requires "
                    "total_transition_budget (R6-A §11.2.2)"
                )
            if self.rollout_transitions_per_update is None:
                raise AssertionError(
                    "transition collection requires "
                    "rollout_transitions_per_update — target-only budgets "
                    "are forbidden (R6-A §11.2.2)"
                )
            if self.target_transitions is None:
                raise AssertionError(
                    "transition collection requires the derived per-batch "
                    "target_transitions = min(horizon, total - committed) "
                    "(R6-A §11.2.2)"
                )


def compute_batch_target(
    total: Optional[int],
    committed: int,
    horizon: Optional[int],
) -> Optional[int]:
    """P0-1: the per-batch collection target = min(remaining, horizon).

    Returns None when no budget is configured (episode-based
    collection). ``committed`` is the cumulative transition count
    restored from the previous checkpoint; the final batch receives the
    exact remaining count (the [8,8,8,1] property). A negative
    remaining is an overshoot — a budgeting bug, never a silent value.

    R6-A (F-01, §11.2.2): there is NO target-only compatibility path —
    a total without the per-update horizon is a contract violation
    (transition collection needs both), never a single-batch target.
    ``total is None`` (episode mode) dominates and stays None regardless
    of the horizon argument.
    """
    if total is None:
        return None
    if horizon is None:
        raise ValueError(
            f"transition budget requires the per-update horizon: "
            f"compute_batch_target({total}, {committed}, None) is the "
            "target-only state — forbidden (R6-A §11.2.2)"
        )
    remaining = total - committed
    if remaining < 0:
        raise ValueError(
            f"transition budget overshoot: committed {committed} "
            f"> total {total}"
        )
    return min(remaining, horizon)


class TransitionBudgetOvershoot(ValueError):
    """committed_before + actual_env_steps > total — the batch executed
    more environment steps than the run-level budget allows. A budgeting
    bug, never a silent value (R6-A §11.2.4)."""


def is_budget_complete(
    total: int,
    committed_before: int,
    actual_env_steps: int,
) -> bool:
    """R6-A §11.2.4: the pure finality oracle — a batch materializes the
    release ONLY when its ACTUAL env steps land exactly on the run
    total. The decision is post-hoc (after the collection), never a
    pre-proven target predicate.
    """
    committed_after = committed_before + actual_env_steps
    if committed_after > total:
        raise TransitionBudgetOvershoot(
            f"transition budget overshoot: committed "
            f"{committed_before}+{actual_env_steps}={committed_after} "
            f"> total {total}"
        )
    return committed_after == total


# ═══════════════════════════════════════════════════════════════════
# small helpers
# ═══════════════════════════════════════════════════════════════════

def _obs_to_tensor(obs_np: Dict[str, np.ndarray],
                   device: str) -> Dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device)
        for k, v in obs_np.items()
    }


def _buffer_obs(obs_t: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """(1, T_o, ...) tensors -> buffer layout (T_o, ...) — batch axis dropped
    at the buffer boundary (np.stack in trainer restores it)."""
    return {k: v.detach().cpu().numpy()[0] for k, v in obs_t.items()}


def _selected(obs_np: Dict[str, np.ndarray], device: str,
              obs_steps: int):
    """Latest-window selection + its metadata.

    Returns (obs_b, ObservationSelection) — the metadata (history_indices)
    is no longer dropped: it enters the result dict and the eval artifact.
    """
    return select_latest_observations(_obs_to_tensor(obs_np, device),
                                      obs_steps)


def _policy_state_dict(policy: Any) -> Dict[str, Any]:
    """policy.state_dict() when available, else the model's (FakePolicy)."""
    target = policy if hasattr(policy, "state_dict") else getattr(
        policy, "model", None
    )
    if target is None or not hasattr(target, "state_dict"):
        return {}
    return {k: v.detach().cpu().clone() for k, v in target.state_dict().items()}


def _optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    sd = optimizer.state_dict()
    state = {}
    for pid, st in sd.get("state", {}).items():
        state[pid] = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in st.items()
        }
    return {"state": state, "param_groups": sd.get("param_groups", [])}


def _rng_states() -> Dict[str, str]:
    states: Dict[str, str] = {
        # pickle+base64 for python/numpy: repr(random.getstate()) is
        # eval-able but repr(np.random.get_state()) embeds array([...])
        # literals (NameError 'array'), so both use pickle for uniformity.
        "python": base64.b64encode(pickle.dumps(random.getstate())).decode(),
        "numpy": base64.b64encode(
            pickle.dumps(np.random.get_state())
        ).decode(),
        "torch": torch.get_rng_state().cpu().numpy().tobytes().hex(),
        # Informational only — PYTHONHASHSEED is fixed at interpreter
        # start and cannot be changed/restored; the key records the hash
        # seed the run executed under (reproducibility audit). The real
        # env-sequence RNG is the EpisodeTaskSampler state (SEED-RESUME).
        "env": os.environ.get("PYTHONHASHSEED", ""),
    }
    if torch.cuda.is_available():
        states["cuda"] = "|".join(
            s.cpu().numpy().tobytes().hex()
            for s in torch.cuda.get_rng_state_all()
        )
    return states


def restore_rng_states(rng_states: Dict[str, str]) -> None:
    """Inverse of _rng_states(): restore python/numpy/torch/cuda RNG.

    The checkpoint snapshot is taken AFTER the PPO update (ppo_update's
    randperm consumes torch/cuda RNG), so a restored world continues
    exactly where the checkpointing world stopped — its next rollout and
    update reproduce the uninterrupted world's.
    pickle is used for python/numpy states (repr is not round-trip-able
    for numpy's array literal); the payload is this project's own
    checkpoint, not external input.
    """
    random.setstate(pickle.loads(base64.b64decode(rng_states["python"])))
    np.random.set_state(pickle.loads(base64.b64decode(rng_states["numpy"])))
    torch.set_rng_state(
        # bytearray keeps the frombuffer view writable (bytes is immutable
        # and set_rng_state rejects non-writable tensors)
        torch.frombuffer(
            bytearray(bytes.fromhex(rng_states["torch"])), dtype=torch.uint8
        )
    )
    if "cuda" in rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            torch.frombuffer(
                bytearray(bytes.fromhex(s)), dtype=torch.uint8
            )
            for s in rng_states["cuda"].split("|")
        )


def _owned_generator_states() -> Dict[str, str]:
    """R7-6 v6: explicitly-owned persistent torch.Generator states
    (name -> state hex) captured at the quiescent save boundary.

    The production path currently owns none — the synth far-dome
    generator is created per-call with manual_seed (stateless,
    validated_synth_adapter.py) and every other RNG flows through the
    globals captured by _rng_states(). A future persistent generator
    must be registered here so the checkpoint can restore it; a
    non-empty recorded set that cannot be restored is a hard error at
    resume (the restore side refuses it).
    """
    return {}


def _actor_optimizer_param_names(
    optimizer: torch.optim.Optimizer, policy: Any
) -> List[str]:
    """R7-6 v6: ordered parameter names of the actor optimizer's first
    param group, resolved through the registered LoRA bundle's
    named_parameters() by id.

    The group's param order IS the bundle registration order, so the
    list is deterministic across processes. The restore side validates
    this list against the live wiring BEFORE any tensor load (a drift
    is a hard error). A policy without a registered bundle (fixture
    FakePolicy world) has no name mapping -> [] — the declared zero-LoRA
    world, covered by the lora_identity rank-0 gate.
    """
    model = getattr(policy, "model", None)
    bundle = getattr(model, "dppo_lora_adapters", None)
    if bundle is None:
        return []
    name_by_id = {id(p): name for name, p in bundle.named_parameters()}
    groups = optimizer.param_groups
    if not groups:
        return []
    names: List[str] = []
    for p in groups[0].get("params", []):
        name = name_by_id.get(id(p))
        if name is None:
            raise CheckpointError(
                "actor optimizer param not found in the registered LoRA "
                "bundle — the optimizer holds a non-LoRA parameter "
                "(R7-3 partition contract violated at save time)"
            )
        names.append(name)
    return names


@dataclass(frozen=True)
class StructuredStateIdentity:
    """The structured state axes a checkpoint saves that the live config
    must reproduce BEFORE any tensor load (C2: saved-and-validated).

    The old checkpoint saved lr_scheduler/sampling_std/reward_norm/
    numerical_mode states and never validated them on restore — the
    "field exists proves content real" anti-pattern. Restore compares
    every axis against the live identity and hard-fails on any mismatch.

    R6-F (§11.7.1): the axes here are the STATIC identity — scheduler
    method/version and reward_norm version/source/method. The per-batch
    MUTABLE state (scheduler step counts/step position, scaler running
    returns) is validated cross-field inside restore
    (validate_scheduler_state/validate_reward_scale_state against the
    checkpoint counters) — it is state to continue, never a config
    identity axis.
    """

    lr_scheduler_state: Dict[str, Any]  # {method, version} — static axes
    sampling_std_adaptive: bool
    min_sampling_denoising_std: float
    min_logprob_denoising_std: float
    # Rework #2 (P1-1) + R6-F: the reward_norm STATIC axes
    # (version/source/method), not the full canonical dict — the params
    # carry per-batch mutable scaler state (step_count/ret/mean/var/
    # count) that differs checkpoint-to-checkpoint by design and is
    # validated cross-field in restore, never as a live-config axis.
    reward_norm_state: Dict[str, Any]
    numerical_mode: Dict[str, Any]


def _structured_identity(
    loop_config: LoopConfig,
    runtime: Dict[str, Any],
    numerical_mode_resolved: Dict[str, Any],
) -> StructuredStateIdentity:
    """The live structured-state identity the restore validates against."""
    return StructuredStateIdentity(
        lr_scheduler_state={
            "method": loop_config.lr_scheduler.method,
            "version": SCHEDULER_STATE_VERSION,
        },
        sampling_std_adaptive=False,  # static sampling std
        min_sampling_denoising_std=loop_config.min_sampling_denoising_std,
        min_logprob_denoising_std=loop_config.min_logprob_denoising_std,
        # static axes only (P1-1 + R6-F) — the mutable scaler params are
        # validated cross-field in restore (§11.7.1)
        reward_norm_state={
            "version": runtime["reward_version"],
            "source": "env",  # the env Reward module is the ONLY source
            "method": loop_config.reward_scale.method,
        },
        numerical_mode=numerical_mode_resolved,
    )


def _validate_structured_state(
    data: CheckpointData, identity: StructuredStateIdentity,
) -> None:
    """C2: every saved structured-state axis must match the live config —
    a mismatch is a hard error raised before any tensor load."""
    saved_rn = data.reward_norm_state or {}
    checks = (
        ("lr_scheduler_state.method",
         data.lr_scheduler_state.get("method"),
         identity.lr_scheduler_state["method"]),
        ("lr_scheduler_state.version",
         data.lr_scheduler_state.get("version"),
         identity.lr_scheduler_state["version"]),
        ("sampling_std_state.adaptive",
         data.sampling_std_state.get("adaptive"),
         identity.sampling_std_adaptive),
        ("sampling_std_state.min_sampling_denoising_std",
         data.sampling_std_state.get("min_sampling_denoising_std"),
         identity.min_sampling_denoising_std),
        ("sampling_std_state.min_logprob_denoising_std",
         data.sampling_std_state.get("min_logprob_denoising_std"),
         identity.min_logprob_denoising_std),
        ("reward_norm_state.version",
         saved_rn.get("version"), identity.reward_norm_state["version"]),
        ("reward_norm_state.source",
         saved_rn.get("source"), identity.reward_norm_state["source"]),
        ("reward_norm_state.method",
         saved_rn.get("method"), identity.reward_norm_state["method"]),
        ("numerical_mode_state",
         data.numerical_mode_state, identity.numerical_mode),
    )
    for name, saved, expected in checks:
        if saved != expected:
            raise CheckpointError(
                f"structured state mismatch: {name} — checkpoint "
                f"{saved!r} != live {expected!r}; refusing to load any "
                "state (a resumed run must reproduce the saved "
                "scheduler/std/reward-norm/numerical-mode config)"
            )


def _validate_sampler_identity(
    data: CheckpointData, task_sampler: EpisodeTaskSampler,
) -> None:
    """P0-2: the sampler identity (base_seed/spec-set) is validated INSIDE
    restore, before ANY state load — the old call-site check ran after
    policy/critic/optimizer/RNG were already restored."""
    saved = data.sampler_state or {}
    checks = (
        ("base_seed", saved.get("base_seed"),
         getattr(task_sampler, "base_seed", None)),
        ("task_spec_set_hash", saved.get("task_spec_set_hash"),
         getattr(task_sampler, "spec_set_hash", None)),
    )
    for name, saved_value, live_value in checks:
        if saved_value != live_value:
            raise CheckpointError(
                f"sampler identity mismatch: {name} — checkpoint "
                f"{saved_value!r} != live {live_value!r}; refusing to "
                "load any state (a resumed run must continue the same "
                "task sequence, never restart or redirect it)"
            )


def _validate_resume_optimizer_contract(
    data: CheckpointData,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
) -> None:
    """R6-E §11.6.2 (F-08) resume: per-field validation of the LIVE
    optimizers against the frozen resolved config persisted in the
    checkpoint's config layer. resume_identity_for already gates the
    config hash — this names the drifted field instead of leaving a bare
    hash mismatch. A checkpoint saved without the optimizer_contract
    layer (pre-R6-E) is refused explicitly."""
    config_layer = (data.exact_resolved_config or {}).get(
        "config_layer", {})
    frozen = (config_layer or {}).get("optimizer_contract") or {}
    for name, optimizer in (("actor", actor_optimizer),
                            ("value", critic_optimizer)):
        frozen_cfg = frozen.get(name)
        if frozen_cfg is None:
            raise CheckpointError(
                f"checkpoint lacks the R6-E optimizer_contract for "
                f"'{name}' — a pre-R6-E checkpoint cannot resume under "
                "the strict optimizer contract"
            )
        validate_optimizer_resume_config(
            name, optimizer, ResolvedOptimizerConfig(**frozen_cfg))


def restore_training_state(
    ckpt_path: str,
    policy: Any,
    critic: torch.nn.Module,
    lora_info: Dict[str, Any],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    runtime_identity: RuntimeIdentity,
    task_sampler: Optional[EpisodeTaskSampler] = None,
    structured_identity: Optional[StructuredStateIdentity] = None,
    # R6-F (§11.7.1): the scheduler/scaler resume wiring — the
    # structured identity gate pins method/version; the per-batch
    # MUTABLE state (scheduler step position/counts, scaler running
    # returns) is cross-field validated against the checkpoint counters
    # and loaded here, before any policy forward (the next update
    # observes the resumed LR and the resumed returns).
    loop_config: Optional[LoopConfig] = None,
    reward_scaler: Optional[RunningRewardScaler] = None,
    actor_sched: Optional[Any] = None,
    critic_sched: Optional[Any] = None,
    n_critic_warmup_updates: int = 0,
    # R7-6 Step 1/5: the running side's vendor binding basis
    # {official_vendor_commit, vendor_manifest_hash} (built by the launcher
    # from the frozen vendor snapshot). None only in fixture/test worlds
    # with no vendor claim; a checkpoint that CLAIMS a vendor binding
    # hard-fails without a basis (fail-closed).
    vendor_identity: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Restore the full training state from a checkpoint (§5.8).

    Order matters (validation first): the FULL runtime identity
    (official commit, project git commit/tree, contract/manifest/BC/
    config hashes, source file hashes), the LoRA configuration identity
    (P0-1: structure + wiring/empty/rank/alpha/n_adaptors gates), the
    sampler identity (P0-2: base_seed/spec-set — validated HERE, before
    any load, not at the call site after the fact), the structured state
    axes (C2: lr_scheduler/sampling_std/reward_norm/numerical_mode are
    SAVED and VALIDATED against the live identity) and the
    environment-RNG identity (C1) are all validated BEFORE any state_dict
    load — a mismatch is a hard CheckpointError, never a
    silently-different continuation. Then the policy base weights, then
    the LoRA adaptors (N2: they are forward-hook closures invisible to
    policy.state_dict(), so the checkpoint's LoRA tensors are the only
    restorable copy of the trained actor), then the critic, then the
    R6-E optimizer contract validation (BEFORE the optimizer load — with
    a wired scheduler torch load_state_dict restores the save-time
    schedule-position lr into param_groups, so the check must compare the
    live construction anchor; no-scheduler path unchanged), then the
    optimizer states (torch load_state_dict casts state tensors to the
    params' device), then the R6-F scheduler/scaler per-batch state (when
    the caller
    passes the live instances + loop_config), then the sampler state is
    loaded INTERNALLY (SEED-RESUME: attempt/cursor/RNG continue — the
    call site must not load it after the fact), then the RNG snapshot is
    restored LAST (R7-6 Step 5: the next operation after restore is the
    exact next sampling operation), then final device/dtype assertions.

    R6-F (§11.7.1/§11.7.2/§11.7.3): when wired, the checkpoint's
    scheduler/scaler state is cross-field validated
    (critic.step_count == scaler.step_count == ppo_update_count; actor
    .step_count == max(0, ppo_update_count - n_critic_warmup_updates))
    and loaded. The scheduler re-applies the LR at the saved step
    position (the resumed update observes the loaded LR — a cold resume
    continues the LR sequence exactly); the scaler carries the running
    per-slot returns into the next scaled batch. The sampling std stays
    static config (adaptive=false — the std keys are part of the hashed
    config layer).

    Returns the saved counters so the caller can continue numbering:
    {ppo_update_count, environment_transition_count,
     completed_episode_count, optimizer_step_count, outer_iteration}
    plus the sampler state (already loaded into task_sampler when one
    was passed).

    Raises:
        CheckpointError: identity mismatch (before any load), or
            unreadable/invalid checkpoint (CheckpointBuilder).
    """
    data = CheckpointBuilder.load(ckpt_path)
    # R7-6 Step 1: schema-first validation order — the v6 quiescent-boundary
    # contract (phase/pending_async_work/source-files double axis) is
    # validated BEFORE any tensor load, immediately after the schema gate
    # in load() and before every identity/LoRA/optimizer-mapping gate.
    validate_v6_contract(data)
    # R7-6 Step 9: metrics writer cursor drift — the cursor records the
    # metrics.jsonl records committed BEFORE this batch (launcher passes
    # batch_idx-1), so a consistent save has cursor == outer_iteration-1.
    # A drifted cursor would make the resumed writer renumber/desync the
    # metrics stream; refused BEFORE any tensor load.
    if data.metrics_writer_cursor != data.outer_iteration - 1:
        raise CheckpointError(
            "checkpoint metrics_writer_cursor "
            f"{data.metrics_writer_cursor} != outer_iteration-1 "
            f"({data.outer_iteration - 1}) — the metrics record cursor "
            "drifted from the batch counter; refusing to load any state "
            "(a resumed writer would renumber/desync metrics.jsonl)"
        )
    # RESUME-HASH-GATE: validation precedes every tensor load — the spy
    # proof in the RED tests requires the state-load spys to stay EMPTY
    # when ANY identity axis diverges.
    validate_resume_identity(data, runtime_identity)
    # R7-6 Step 1: the frozen official vendor binding (R7-1) — commit +
    # 401-file manifest hash must match the running side BEFORE any load.
    validate_vendor_identity(data, vendor_identity)
    validate_lora_identity(data)
    # An empty policy payload is NEVER a valid formal checkpoint — the
    # kind of silent continuation (fresh weights under a saved identity)
    # the identity gate exists to prevent. lora_state may legitimately be
    # {} (no LoRA wired), so only policy_state hard-fails.
    if not data.policy_state:
        raise CheckpointError(
            "checkpoint policy_state must not be empty — refusing formal "
            "resume (an empty policy payload is never a valid checkpoint)"
        )
    validate_checkpoint_lora_usable(data, lora_info)
    # R7-6 Step 5: the v6 LoRA name-order axis — the checkpoint's
    # declared lora_name_order must equal the sorted keys of ITS OWN
    # lora_state (a tampered/desynced mapping is a hard error BEFORE
    # any tensor load; the saved lora_checksum pins the tensor contents,
    # this pins the name axis).
    if list(data.lora_name_order or []) != sorted(data.lora_state or {}):
        raise CheckpointError(
            "checkpoint lora_name_order "
            f"{list(data.lora_name_order or [])!r} does not match its own "
            f"lora_state keys {sorted(data.lora_state or {})!r} — "
            "refusing to load any state (the LoRA name axis must be "
            "internally consistent)"
        )
    if task_sampler is not None:
        _validate_sampler_identity(data, task_sampler)
    if structured_identity is not None:
        _validate_structured_state(data, structured_identity)
    validate_environment_rng_state(data)
    # R7-6 Step 5: the v6 owned-Generator axis — the production path
    # owns none (the synth far-dome generator is per-call manual_seed);
    # a checkpoint claiming a set the live world cannot provide exactly
    # cannot be restored: hard-fail, never a silent partial RNG
    # continuation.
    live_generators = _owned_generator_states()
    if dict(data.owned_generators or {}) != live_generators:
        raise CheckpointError(
            "checkpoint owned_generators "
            f"{dict(data.owned_generators or {})!r} cannot be restored — "
            f"the live world owns {live_generators!r} (R7-6 Step 5: all "
            "RNG states must be exactly restorable)"
        )
    # R5.5 lightweight resume: the base model.* keys are intentionally
    # absent from resume checkpoints — the missing set must be EXACTLY a
    # subset of the frozen base (re-materialized from the BC checkpoint,
    # bound by the identity's bc_checkpoint_hash). Any other missing or
    # unexpected key is a hard error.
    missing, unexpected = policy.load_state_dict(
        data.policy_state, strict=False
    )
    base_keys = {k for k in policy.state_dict() if k.startswith("model.")}
    bad_missing = [k for k in missing if k not in base_keys]
    if bad_missing:
        raise CheckpointError(
            "checkpoint policy_state lacks non-base keys: "
            f"{sorted(bad_missing)[:5]} — a lightweight resume checkpoint "
            "may only omit the frozen base model.* keys"
        )
    if unexpected:
        raise CheckpointError(
            f"checkpoint policy_state has unexpected keys: "
            f"{sorted(unexpected)[:5]}"
        )
    if data.lora_state:
        restore_lora_state(lora_info, data.lora_state)
    # The value checkpoint holds ONLY the value head (mlp.*) —
    # load_value_state_dict rejects any non-mlp payload, so the pollution
    # chain (value_state re-covering the policy normalizer) is structurally
    # impossible.
    critic.load_value_state_dict(data.value_state)
    # R6-E §11.6.2 (F-08) + R6-F (§11.7.2): the frozen resolved optimizer
    # config (type/lr/betas/eps/weight_decay/param_group_count/
    # ordered_parameter_name_hash) is validated per field BEFORE the
    # optimizer state load — a drifted hyperparameter hard-fails with the
    # field named before any tensor load (validation first, like every
    # other gate). With a scheduler wired, torch load_state_dict restores
    # the save-time schedule-position lr into param_groups, so a post-load
    # check could not compare against the hashed construction anchor
    # (min_lr) — pre-load the live lr IS the anchor. The no-scheduler path
    # is unchanged: load never touches lr there, so pre-load and post-load
    # validation agree.
    # R7-6 Step 5: the v6 actor-optimizer name axis — the LIVE wiring
    # (bundle named_parameters by id, the save-side resolver) must equal
    # the checkpoint's declared names BEFORE any optimizer tensor load (a
    # drift means the checkpoint's state would land on different
    # parameters than it was saved from).
    live_param_names = _actor_optimizer_param_names(
        actor_optimizer, policy)
    if list(data.actor_optimizer_param_names or []) != live_param_names:
        raise CheckpointError(
            "checkpoint actor_optimizer_param_names "
            f"{list(data.actor_optimizer_param_names or [])!r} != live "
            f"{live_param_names!r} — refusing to load any optimizer "
            "state (the actor parameter mapping must match the live "
            "wiring)"
        )
    _validate_resume_optimizer_contract(
        data, actor_optimizer, critic_optimizer)
    actor_optimizer.load_state_dict(data.actor_optimizer_state)
    critic_optimizer.load_state_dict(data.value_optimizer_state)
    # R6-F (§11.7.1): the scheduler/scaler per-batch MUTABLE state is
    # cross-field validated against the checkpoint counters (critic
    # step_count == scaler step_count == ppo_update_count; actor ==
    # max(0, ppo_update_count - warmup)) BEFORE any load, then applied
    # AFTER the optimizer load (a loaded scheduler re-applies the LR at
    # the saved step position — the NEXT update observes the resumed LR;
    # a loaded scaler carries the running per-slot returns into the next
    # scaled batch, so a cold resume == uninterrupted oracle). The
    # structured identity gate above already pinned method/version; a
    # pre-R6-F checkpoint ({"enabled": false}-shaped states) is refused
    # there — and by the config-hash gate — before this point.
    if loop_config is not None and reward_scaler is not None:
        validate_reward_scale_state(
            data.reward_norm_state, loop_config.reward_scale,
            ppo_update_count=data.ppo_update_count,
        )
        load_scaler_state(data.reward_norm_state, reward_scaler)
    if loop_config is not None and actor_sched is not None:
        validate_scheduler_state(
            data.lr_scheduler_state, loop_config.lr_scheduler,
            ppo_update_count=data.ppo_update_count,
            n_critic_warmup_updates=n_critic_warmup_updates,
        )
        load_scheduler_state(
            data.lr_scheduler_state, actor_sched, critic_sched)
    # SEED-RESUME production path — the sampler's attempt/cursor/RNG
    # state is restored INSIDE restore (identity already validated above)
    # so the resumed run CONTINUES the episode seed sequence instead of
    # restarting at base_seed. No call-site load after the fact.
    if task_sampler is not None:
        task_sampler.load_state_dict(data.sampler_state)
    # R7-6 Step 5: restore ALL RNG states LAST — after the sampler/env/
    # slots/counters/cursor are in place, so the next operation after
    # restore is the exact next sampling operation with the exact RNG
    # state of the checkpointing world.
    restore_rng_states(data.rng_states)
    # DictOfTensorMixin._load_from_state_dict (vendor, frozen) REPLACES
    # params with the checkpoint tensors instead of copying into the live
    # params, and policy_state is saved as CPU clones — so the
    # normalizer's scale/offset silently move to CPU on the policy load.
    # With the normalizer on CPU, LinearNormalizer.normalize casts its
    # input to scale.device — the next PolicyObsEncoder forward (rollout /
    # critic / get_new_logprobs) feeds CPU observations into the CUDA
    # pointnet. Re-anchor the whole policy to the model's device AFTER
    # every load (the UNet params are untouched by load_state_dict, so
    # their device is the pre-restore anchor; a detector keyed on
    # next(policy.parameters()) does NOT work — the normalizer's params
    # register after obs_encoder/model).
    model = getattr(policy, "model", None)
    anchor = next(model.parameters(), None) if model is not None else None
    if anchor is not None:
        policy.to(anchor.device)
        # Final device/dtype assertions (guide Step 2): the restored world
        # must sit on the policy anchor device in FP32 — a load that moved
        # the normalizer to CPU or changed dtype would silently break the
        # next rollout's numerics (formal training is FP32-only).
        for module, name in ((policy, "policy"), (critic, "critic")):
            parameters = getattr(module, "parameters", None)
            if parameters is None:  # non-nn.Module fixture (FakePolicy)
                continue
            for p in parameters():
                assert p.dtype == torch.float32, (
                    f"{name} param {tuple(p.shape)} not FP32 after restore: "
                    f"{p.dtype}"
                )
                assert p.device == anchor.device, (
                    f"{name} param on {p.device} != policy anchor "
                    f"{anchor.device} after restore"
                )
    return {
        "ppo_update_count": data.ppo_update_count,
        "environment_transition_count": data.environment_transition_count,
        "completed_episode_count": data.completed_episode_count,
        "optimizer_step_count": data.optimizer_step_count,
        # review partB P0-1: the budget-cut ACTIVE slot states must ride
        # the restore return — dropping them silently restarts episodes.
        "active_slot_states": dict(getattr(data, "active_slot_states", {})),
        # cumulative training iteration: fresh=1, each resume +1 — the
        # counter test requires [1, 2, 3] across fresh->resume->resume.
        "outer_iteration": data.outer_iteration,
        # The sampler state is ALWAYS present (schema 1.1 hard-fails at
        # load when missing). It is loaded into task_sampler internally
        # when one was passed; kept in the return for backward-compatible
        # callers that manage their own sampler.
        "sampler_state": data.sampler_state,
        # R7-6 v6 axes — the same-seed comparator (guide Step 6-7)
        # consumes these across the control and resume runs.
        "metrics_writer_cursor": data.metrics_writer_cursor,
        "observation_window_provenance": dict(
            data.observation_window_provenance or {}),
        "next_value_provenance": dict(data.next_value_provenance or {}),
        "lora_name_order": list(data.lora_name_order or []),
        "actor_optimizer_param_names": list(
            data.actor_optimizer_param_names or []),
        "owned_generators": dict(data.owned_generators or {}),
        "voided_episode_count": data.voided_episode_count,
    }


def _canonical_bytes(obj: Any) -> bytes:
    """Deterministic byte encoding for state-hash composition (R7-6).

    Tensors encode shape/dtype + raw bytes (exact under the formal
    deterministic numerical mode); dicts sort keys; no object id() or
    repr ordering may leak in — the hash must be cross-process stable.
    """
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        return (
            repr(tuple(t.shape)).encode("ascii")
            + str(t.dtype).encode("ascii")
            + t.numpy().tobytes()
        )
    if isinstance(obj, dict):
        parts = []
        for k in sorted(obj):
            parts.append(
                str(k).encode("utf-8") + b"\x00"
                + _canonical_bytes(obj[k])
            )
        return b"dict" + repr(len(parts)).encode("ascii") + b"".join(parts)
    if isinstance(obj, (list, tuple)):
        return (
            b"list" + repr(len(obj)).encode("ascii")
            + b"".join(_canonical_bytes(x) for x in obj)
        )
    if isinstance(obj, str):
        return b"str" + obj.encode("utf-8")
    if obj is None:
        return b"none"
    if isinstance(obj, (bool, int, float)):
        return repr(obj).encode("ascii")
    return repr(obj).encode("utf-8")


def _state_hash(
    policy: Any,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    task_sampler: Any,
) -> str:
    """R7-6 Step 6: exact hash of the full restorable state (layer 1).

    The load-complete state of the same-seed comparator: policy weights,
    value head, both optimizers (moments), and the sampler state. Under
    the formal deterministic runtime the hash is exact across processes
    (guide Step 7 — no tolerances).
    """
    h = hashlib.sha256()
    for obj in (
        policy.state_dict(),
        critic.state_dict(),
        actor_optimizer.state_dict(),
        critic_optimizer.state_dict(),
        task_sampler.state_dict() if task_sampler is not None else None,
    ):
        h.update(_canonical_bytes(obj))
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash(config_dict: Dict[str, Any]) -> str:
    """Thin local alias — the single implementation lives in checkpoint.py
    (config_hash_of); the eval side recomputes over the persisted
    config_layer (P1-2)."""
    return config_hash_of(config_dict)


def _config_dict(
    contract: RLSystemContract,
    loop_config: LoopConfig,
    trainer_config: TrainerConfig,
    runtime: Dict[str, Any],
    numerical_mode_resolved: Dict[str, Any],
    policy: Any,
    *,
    base_seed: int = 0,
    n_envs: int = 1,
    lora_info: Optional[Dict[str, Any]] = None,
    actor_optimizer: Optional[torch.optim.Optimizer] = None,
    critic_optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """The reproducible config layer — the config_hash source (schema v2).

    Exactly what a resumed run can recompute BEFORE its batch runs:
    device/reward/loop/gate/std settings, the sealed policy contract
    (G5), the configured trainer fields, and the training_identity axes
    (P0-2, R3 rework): base_seed / n_envs / LoRA rank+alpha / actor and
    value lr / optimizer identity — a resume under different
    seeds/ranks/lrs/optimizers must never be invisible to the identity
    gate. Runtime artifacts (next_value_batch_size_resolved from the
    post-update metrics, the last episode's record) are deliberately
    EXCLUDED — they live only in exact_resolved_config, so save-time and
    resume-time hashes agree.
    """
    axes = _live_lora_axes(lora_info or {})
    # R6-E §11.6.2 (F-08): the RESOLVED optimizer config is frozen at
    # creation (type/lr/betas/eps/weight_decay/param_group_count/
    # ordered_parameter_name_hash) — the identity layer reads the frozen
    # values, never a raw param_groups[0] (a second group would be
    # silently ignored by [0] indexing). require_single_param_group
    # hard-fails here if the invariant is already violated.
    actor_contract = (
        resolved_optimizer_config("actor", actor_optimizer)
        if actor_optimizer is not None else None
    )
    value_contract = (
        resolved_optimizer_config("value", critic_optimizer)
        if critic_optimizer is not None else None
    )
    # R6-F (§11.7.2): when a scheduler owns the LR, the hashed lr axis is
    # the CONSTRUCTION anchor (min_lr) — the scheduler sets lr = min_lr at
    # construction, and pair-steps move the LIVE optimizer lr to the
    # schedule position (so save-time, after the pair-step, differs from
    # resume-time, before any load). min_lr is reproducible on both sides
    # WITHOUT the checkpoint (resume_identity_for runs before any state
    # load), keeping the identity gate side-symmetric; the resumed
    # scheduler re-applies the LR at the saved step position after load.
    scheduler_wired = loop_config.lr_scheduler.method != "none"
    anchor_lr = loop_config.lr_scheduler.min_lr if scheduler_wired else None
    actor_contract = (
        replace(actor_contract, lr=anchor_lr)
        if scheduler_wired and actor_contract is not None else actor_contract
    )
    value_contract = (
        replace(value_contract, lr=anchor_lr)
        if scheduler_wired and value_contract is not None else value_contract
    )
    return {
        "device": runtime["device"],
        "reward_version": runtime["reward_version"],
        # Gate D reopening (2026-08-22): the external world assets are a
        # training-input identity axis — a resume under different world
        # files must fail the config-hash gate (the aggregate hash over
        # every admitted world's scene + obstacle files).
        "world_assets_hash": runtime["world_assets_hash"],
        "loop_mode": loop_config.mode,
        "n_episodes": loop_config.n_episodes,
        "max_ep_len": loop_config.max_ep_len,
        "eval_n_episodes": loop_config.eval_n_episodes,
        "gate_phase": loop_config.gate_phase,
        "min_sampling_denoising_std": loop_config.min_sampling_denoising_std,
        "min_logprob_denoising_std": loop_config.min_logprob_denoising_std,
        # P0-1 fix: the hashed axes are the CONSTANT run-level budget
        # fields — the per-batch remaining target varies by design and
        # must never change the config hash between consecutive
        # checkpoints (a varying hashed field would break every resumed
        # budgeted run at the identity gate).
        # R6-A: the RESOLVED collection mode is a contract axis — a
        # resume under a different collection semantics must fail the
        # identity gate.
        "total_transition_budget": loop_config.total_transition_budget,
        "rollout_transitions_per_update": (
            loop_config.rollout_transitions_per_update
        ),
        "collection_mode": loop_config.collection_mode,
        # R6-F (§11.7.1): scheduler/reward-scaling config — EVERY field
        # enters the hashed layer (a resume under a different schedule or
        # scaling is a different run, refused at the identity gate).
        "lr_scheduler_config": asdict(loop_config.lr_scheduler),
        "reward_scale_config": asdict(loop_config.reward_scale),
        "numerical_mode": numerical_mode_resolved,
        # The runtime contract the sealed G5 identity was measured under
        # must be persistable — a checkpoint whose exact_resolved_config
        # cannot reproduce the rollout/likelihood batch semantics cannot
        # back a reproducibility claim.
        "policy_contract": {
            "policy_cfg_lambda": getattr(policy, "cfg_lambda", 1.5),
            "rollout_policy_batch_size": 1,
            "likelihood_microbatch_size": 1,
        },
        "trainer": {
            "update_epochs": trainer_config.update_epochs,
            "batch_size": trainer_config.batch_size,
            "max_grad_norm": trainer_config.max_grad_norm,
            "target_kl": trainer_config.target_kl,
            "vf_coef": trainer_config.vf_coef,
            "clip_vloss_coef": trainer_config.clip_vloss_coef,
            "n_critic_warmup_updates": (
                trainer_config.n_critic_warmup_updates
            ),
            "critic_grad_norm_limit": (
                trainer_config.critic_grad_norm_limit
            ),
            "next_value_batch_size_configured": (
                trainer_config.next_value_batch_size
            ),
        },
        # The training-input identity axes — see docstring (P0-2).
        # base_seed is None for samplers without one (scripted/test
        # double) — recorded as None, never int()-coerced.
        "training_identity": {
            "base_seed": int(base_seed) if base_seed is not None else None,
            "n_envs": int(n_envs),
            "lora_rank": axes["lora_rank"],
            "lora_alpha": axes["lora_alpha"],
            "lr_actor": (
                actor_contract.lr if actor_contract is not None else None
            ),
            "lr_value": (
                value_contract.lr if value_contract is not None else None
            ),
            "actor_optimizer": (
                actor_contract.type if actor_contract is not None else None
            ),
            "value_optimizer": (
                value_contract.type if value_contract is not None else None
            ),
        },
        # R6-E §11.6.2: the frozen resolved per-optimizer config — the
        # resume side re-validates EVERY field against this layer (a
        # shape-only comparison is explicitly not enough; hyperparameter
        # drift must fail the resume).
        "optimizer_contract": {
            "actor": (
                actor_contract.to_dict()
                if actor_contract is not None else None
            ),
            "value": (
                value_contract.to_dict()
                if value_contract is not None else None
            ),
        },
    }


def resume_identity_for(
    contract: RLSystemContract,
    loop_config: LoopConfig,
    trainer_config: TrainerConfig,
    runtime: Dict[str, Any],
    numerical_mode_resolved: Dict[str, Any],
    policy: Any,
    *,
    base_seed: int = 0,
    n_envs: int = 1,
    lora_info: Optional[Dict[str, Any]] = None,
    actor_optimizer: Optional[torch.optim.Optimizer] = None,
    critic_optimizer: Optional[torch.optim.Optimizer] = None,
) -> RuntimeIdentity:
    """The runtime identity a formal resume validates against.

    Every value is the RUNNING side's: official DPPO commit, project git
    commit + tree hash, contract hash, manifest hash, BC checkpoint
    hash, config hash (over the reproducible layer, INCLUDING the
    training_identity axes — the caller must pass the same base_seed/
    n_envs/lora_info/optimizers the save side used, or the config_hash
    gate refuses the resume) and source file hashes. Single
    implementation — the resume call site and the mechanism tests must
    never reconstruct these hashes by hand.
    """
    return RuntimeIdentity(
        official_dppo_commit=OFFICIAL_DPPO_COMMIT,
        project_git_commit=runtime["project_git_commit"],
        project_tree_hash=runtime["project_tree_hash"],
        contract_hash=contract_hash_of(contract),
        manifest_hash=runtime["manifest_hash"],
        bc_checkpoint_hash=runtime["bc_checkpoint_hash"],
        world_assets_hash=runtime["world_assets_hash"],
        config_hash=_config_hash(_config_dict(
            contract, loop_config, trainer_config, runtime,
            numerical_mode_resolved, policy,
            base_seed=base_seed, n_envs=n_envs, lora_info=lora_info,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )),
        source_file_hashes=runtime["source_file_hashes"],
    )


def _code_hash(source_file_hashes: Dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(source_file_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _bc_snapshot(policy: Any) -> Any:
    """Pre-update BC clone for the paired eval (§4.17).

    deepcopy copies forward-hook function objects by reference, so a plain
    clone would share the live LoRA adaptors and the BC side would see the
    post-update deltas. Clearing the clone's hooks leaves the frozen base
    UNet, which equals the policy at insertion time (lora_B zero-init).
    """
    clone = copy.deepcopy(policy)
    model = getattr(clone, "model", None)
    if model is not None:
        for module in model.modules():
            getattr(module, "_forward_hooks", {}).clear()
    return clone


def _missing_weight_keys(policy: Any) -> List[str]:
    return list(getattr(policy, "_dppo_missing_keys", []))


def _unexpected_weight_keys(policy: Any) -> List[str]:
    return list(getattr(policy, "_dppo_unexpected_keys", []))


# ═══════════════════════════════════════════════════════════════════
# paired evaluation (§4.17)
# ═══════════════════════════════════════════════════════════════════

def _eval_episode(env: Any, policy: Any, contract: RLSystemContract,
                  spec: EpisodeTaskSpec, device: str, max_ep_len: int,
                  seed: int, gate: PointCloudValidityGate,
                  cam_far: float,
                  min_sampling_denoising_std: float,
                  min_logprob_denoising_std: float
                  ) -> Optional[Dict[str, Any]]:
    """Closed-loop BC-style episode with one policy -> outcome record.

    The eval rollout has its OWN point-cloud validity gate — an invalid
    reset frame or first invalid step frame voids the episode (returns
    None); a record from an invalid rollout would pollute the paired eval
    as a PPO==BC outcome (§4.17 [JUDGMENT]).

    EVAL-STD: both resolved denoising floors arrive from the
    LoopConfig — never the policy default. The sampling floor binds here:
    sample_ddpm_with_logprob's sigma_clip_min clamps the DDIM σ_t, which
    IS the sampling noise (x_next = mu_t + σ_t·noise), so the eval rollout
    enforces the same sampling floor as the training rollout. The logprob
    floor is carried for interface uniformity — the eval computes no
    likelihood (it binds in the PPO path's per_dim_denoising_logprobs).
    """
    obs, info = env.reset([spec], seeds=[seed])
    # R7-5: gate the FULL policy window (latest obs_steps frames), not
    # just the newest — the policy consumes all three.
    reset_window = validate_policy_observation_window(
        info[0].get("observation_window_mapping")
        or build_observation_window_mapping(
            obs, contract.obs_steps, cam_far),
        obs_steps=contract.obs_steps,
        gate=gate,
    )
    if not reset_window.valid:
        # Close the slot explicitly — the next seed's reset would
        # hard-fail on the ACTIVE guard otherwise (the real adapter
        # refuses ACTIVE resets, and the single-slot mock mirrors it).
        # Same REJECT_EPISODE closure as the train loop's reset path:
        # DONE, no fault verdict, no record.
        if env.is_active(0):
            env.reject_episode([0])
        return None
    done = False
    steps = 0
    total_r = 0.0
    terminated = False
    last_outcome = {
        "success": False, "collision": False, "done_reason": "none",
    }
    # truncated initialized BEFORE the loop — the loop can exit on
    # max_ep_len with the env never emitting truncated, so the returned
    # timeout must be the OR of the env's truncated and the loop-level
    # timeout (see the loop_timeout closure below).
    truncated = False
    loop_timeout = False
    # Per-step episode trace: the RAW per-step series — never
    # aggregate-only. The s9 state layout [pos(3), goal_body(3), dist,
    # sin_yaw, cos_yaw] is identical on the real env and the mock, so the
    # trace reads from the shared obs interface.
    trace: List[Dict[str, Any]] = []
    t0 = time.monotonic()
    while not done and steps < max_ep_len:
        obs_b, _ = _selected(obs, device, contract.obs_steps)
        with torch.no_grad():
            # EVAL-STD: the resolved sampling floor — the official
            # kernel's min_sampling_denoising_std — never the policy
            # default (R5.1: two independent floors reach the kernel).
            x_t, _, _ = policy.sample_ddpm_with_logprob(
                obs_dict=obs_b,
                min_sampling_denoising_std=min_sampling_denoising_std,
                min_logprob_denoising_std=min_logprob_denoising_std,
                num_inference_steps=contract.denoising_steps,
                ft_denoising_steps=contract.ft_denoising_steps,
            )
        policy_action = x_t[:, contract.action_start:
                            contract.action_start + contract.action_horizon]
        executed = policy_action[:,: contract.exec_steps]
        action_np = executed[0].detach().cpu().numpy()
        action_np = policy.normalizer["action"].unnormalize(action_np)
        # unnormalize returns a tensor on the policy's device (the
        # normalizer's scale/offset live on cuda) — np.asarray on a CUDA
        # tensor raises, so move to host first. Canonical eval on GPU
        # (RULING-7 item ④ gate): TypeError "can't convert cuda:0 device
        # type tensor to numpy" before this fix.
        if torch.is_tensor(action_np):
            action_np = action_np.detach().cpu().numpy()
        # Action summary for the trace — mean norm + max |component| of the
        # executed window; the full (exec_steps, 11) array stays in the
        # normalized action space, the trace keeps a summary.
        act_np = np.asarray(action_np, dtype=np.float32)
        trace.append({
            "step": steps,
            "elapsed": round(time.monotonic() - t0, 3),
            "action_norm": float(np.linalg.norm(act_np)),
            "action_max_abs": float(np.abs(act_np).max()),
        })
        obs, reward, terminated, truncated, info = env.step(action_np[None])
        last_outcome = dict(info[0].get("outcome", {})) or {
            "success": False, "collision": False, "done_reason": "none",
        }
        # The real adapter's StepBatch carries (n_envs,) arrays; the
        # single-slot mock returns scalars — unify before use.
        reward = float(np.asarray(reward).reshape(-1)[0])
        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])
        step_window = validate_policy_observation_window(
            info[0].get("observation_window_mapping")
            or build_observation_window_mapping(
                obs, contract.obs_steps, cam_far),
            obs_steps=contract.obs_steps,
            gate=gate,
        )
        if not step_window.valid:
            # Close the slot explicitly — mirror the train loop's
            # SAFE_END_EPISODE closure (VOIDED). Guarded by is_active(): a
            # frame that terminated AND faulted leaves the slot already
            # DONE — no double-close.
            if env.is_active(0):
                env.mark_voided([0])
            return None  # invalid rollout — no valid outcome record
        total_r += reward
        steps += 1
        done = terminated or truncated
        # Per-step state trace from the POST-step frame — position/yaw/goal
        # distance from the s9 state (identical layout on both envs),
        # clearance = min ray distance of the point cloud, hit count and
        # done_reason from the step info.
        s9 = np.asarray(obs["agent_pos"][0, -1], dtype=np.float32)
        pc = np.asarray(obs["point_cloud"][0, -1], dtype=np.float32)
        # An empty point cloud has no ray distances to measure — record
        # nan ("unmeasurable") instead of crashing the trace.
        clearance = (float(np.linalg.norm(pc, axis=1).min())
                     if len(pc) else float("nan"))
        trace[-1].update({
            "pos": [float(v) for v in s9[0:3]],
            "z": float(s9[2]),
            "yaw": float(np.arctan2(s9[7], s9[8])),
            "goal_dist": float(s9[6]),
            "clearance": clearance,
            "hit_count": int(info[0].get("hit_count", 0)),
            "done_reason": str(info[0].get("done_reason", "none")),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
    # The loop exits on `steps >= max_ep_len` with the env still in flight
    # (terminated/truncated both False) — the slot is still ACTIVE and the
    # next rollout's reset_at would hard-fail on the ACTIVE guard. Close it
    # explicitly (mark_truncated, ACTIVE -> DONE) and record the loop-level
    # timeout; is_active() guards the double-close (a frame that terminated
    # AND truncated already left the slot DONE).
    loop_timeout = (not done) and (steps >= max_ep_len)
    if loop_timeout and env.is_active(0):
        env.mark_truncated([0])
    # R5.8-C (N5): the episode truth is the env's EpisodeOutcome of the
    # final step — never a re-derived `terminated and not _collision_`
    # guess (out_of_bounds/altitude episodes are NOT collisions).
    # R5.9-C4 (chain-E review P1-1): done_reason + early-death flags are
    # recorded so the artifact distinguishes reached / collision /
    # out_of_bounds / altitude_exceed / timeout instead of collapsing
    # them into all-false success/collision.
    # R6-E §11.6.4 (F-10): the ONE mutually exclusive terminal category,
    # derived by the shared mapper from the terminated/truncated raw
    # flags + done_reason + success/collision — an unmappable state
    # hard-fails here (no priority-silent reclassification). A loop-level
    # cut (max_ep_len with the env still in flight) is NOT an env
    # termination: timeout is done_reason=="timeout" only, the cut is
    # other="rollout_cut".
    done_reason = str(last_outcome.get("done_reason", "none"))
    terminal = terminal_category(
        terminated=terminated,
        truncated=truncated,
        success=bool(last_outcome["success"]),
        collision=bool(last_outcome["collision"]),
        done_reason=done_reason,
        out_of_bounds=bool(last_outcome.get("out_of_bounds", False)),
        altitude_exceed=bool(last_outcome.get("altitude_exceed", False)),
        other=("rollout_cut" if loop_timeout else "none"),
    )
    return {
        "success": bool(last_outcome["success"]),
        "collision": bool(last_outcome["collision"]),
        "timeout": (done_reason == "timeout"),
        "done_reason": done_reason,
        "out_of_bounds": bool(last_outcome.get("out_of_bounds", False)),
        "altitude_exceed": bool(last_outcome.get("altitude_exceed", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "terminal_category": terminal,
        "return": float(total_r),
        "steps": int(steps),
        "trace": trace,  # the raw per-step series, never aggregate
    }


def run_paired_eval(eval_env: Any, bc_policy: Any, rl_policy: Any,
                    contract: RLSystemContract, loop_config: LoopConfig,
                    bank_entries: Sequence[PairedBankEntry],
                    bank_id: str = "",
                    replica_id: int = 0,
                    policy_checkpoint_hash: str = "",
                    ) -> Tuple[List[EpisodePairedRecord], int]:
    """BC vs RL on the FROZEN paired bank — one record per bank
    entry, consumed exactly in order with the entry's own seed. No
    runtime task re-selection: the bank IS the episode list.

    R5.8-E (N4): bank_id is REQUIRED non-empty — the per-episode
    policy-noise seed derives from it, so an empty id would silently
    produce a noise stream that no artifact provenance can re-derive.

    Runs its own PointCloudValidityGate per episode; returns (records,
    eval_invalid_count). An entry whose BC or RL run hits an invalid frame
    produces NO record and increments eval_invalid_count — the caller must
    hard-fail rather than evaluate on polluted outcomes.

    The frozen-bank contract: after both rollouts, the
    runtime episode config hash and world/role must EQUAL the bank entry's
    frozen values — (spec, seed) determines the sampled config, so any
    mismatch means the bank and the environment disagree (manifest or
    world drift), a system error that raises RuntimeError instead of
    producing a record.

    EVAL-STD: both resolved denoising floors flow from the
    LoopConfig into every eval episode — never the policy default.

    The first argument is a DEDICATED single-slot eval env — never the
    multi-slot training env. The vector env's reset()/step() demand one
    entry per slot (len(task_specs) == n_envs), while eval rolls one
    episode at a time; and a training env may hold ACTIVE slots, whose
    reset_at would hard-fail. n_envs != 1 hard-fails before any rollout;
    the caller must construct an explicit n_envs=1 eval env (EVAL-STD).
    """
    if not bank_id:
        raise ValueError(
            "run_paired_eval requires a non-empty bank_id — the "
            "per-episode policy-noise seed derives from it and the "
            "artifact provenance must re-derive every recorded seed "
            "(R5.8-E N4); an empty id is a silent provenance break"
        )
    # R6-D §11.5.1 (R6-EVAL-1): every record must carry the policy
    # identity that produced it — the dedup layer keys statistical
    # units on it, so an empty hash would silently collapse distinct
    # policies (or fabricate cross-artifact duplicates) downstream.
    # Same entry-level hard constraint as bank_id.
    if not policy_checkpoint_hash:
        raise ValueError(
            "run_paired_eval requires a non-empty policy_checkpoint_hash "
            "— the dedup/PairUnitKey identity of the evaluated policy; "
            "an empty hash is a silent provenance break (R6-D §11.5.1)"
        )
    if getattr(eval_env, "n_envs", 1) != 1:
        raise RuntimeError(
            "paired eval requires a dedicated n_envs=1 eval env; got"
            f" n_envs={eval_env.n_envs} — the multi-slot training env"
            " cannot be implicitly reused for single-episode eval (its"
            " reset/step demand one entry per slot, and ACTIVE training"
            " slots would hard-fail reset_at)"
        )
    records: List[EpisodePairedRecord] = []
    eval_invalid = 0
    gate = PointCloudValidityGate(contract, phase=loop_config.gate_phase)
    cam_far = float(eval_env.cam_far)
    for entry in bank_entries:
        spec = entry.to_spec()
        seed = entry.seed
        # R5.4: both sides draw the SAME diffusion-noise sequence —
        # derived from (bank_id, entry identity, replica), applied inside
        # an isolated per-side RNG context (globals restored after).
        # R6-G (§11.8.2, 12×5=60/side): the caller passes replica_id
        # (0..N-1); each replica is a DIFFERENT policy-noise stream over
        # the same frozen bank entry — the R6-D 12×5 matrix validator's
        # missing production path (run_paired_eval used to hard-code 0).
        pn_seed = policy_noise_seed(
            bank_id, entry.episode_config_hash, replica_id=replica_id
        )
        with paired_rng_context(pn_seed):
            ep_bc = _eval_episode(
                eval_env, bc_policy, contract, spec,
                loop_config.device, loop_config.max_ep_len, seed, gate,
                cam_far,
                loop_config.min_sampling_denoising_std,
                loop_config.min_logprob_denoising_std,
            )
        with paired_rng_context(pn_seed):
            ep_rl = _eval_episode(
                eval_env, rl_policy, contract, spec,
                loop_config.device, loop_config.max_ep_len, seed, gate,
                cam_far,
                loop_config.min_sampling_denoising_std,
                loop_config.min_logprob_denoising_std,
            )
        if ep_bc is None or ep_rl is None:
            eval_invalid += 1
            continue
        cfg = eval_env.current_episode_config()
        # The frozen-bank contract: the runtime episode must BE the frozen
        # episode. A mismatch is a system error (bank/manifest/world
        # drift), never a silently-different record.
        if cfg["episode_config_hash"] != entry.episode_config_hash:
            raise RuntimeError(
                "frozen paired bank violated: runtime episode config hash "
                f"{cfg['episode_config_hash'][:12]} != bank "
                f"{entry.episode_config_hash[:12]} for {entry.world_uid} "
                f"seed {seed} — the bank and the environment disagree"
            )
        if cfg["world"] != entry.world_uid or cfg["role"] != entry.role:
            raise RuntimeError(
                "frozen paired bank violated: runtime world/role "
                f"({cfg['world']}/{cfg['role']}) != bank "
                f"({entry.world_uid}/{entry.role}) — the bank and the "
                "environment disagree"
            )
        records.append(EpisodePairedRecord(
            episode_config={"loop_mode": loop_config.mode,
                            "task_spec": entry.admission_id,
                            # The paired record is bound to both hashes —
                            # manifest source + frozen episode.
                            "manifest_entry_hash": entry.manifest_entry_hash,
                            "episode_config_hash": entry.episode_config_hash},
            seed=seed,
            world=entry.world_uid,
            role=entry.role,
            start=list(entry.start),
            goal=list(entry.goal),
            disturbance=dict(entry.disturbance),
            bc={k: ep_bc[k] for k in
                ("success", "collision", "timeout", "done_reason",
                 "out_of_bounds", "altitude_exceed", "terminated",
                 "truncated", "terminal_category", "return", "steps")},
            rl={k: ep_rl[k] for k in
                ("success", "collision", "timeout", "done_reason",
                 "out_of_bounds", "altitude_exceed", "terminated",
                 "truncated", "terminal_category", "return", "steps")},
            policy_noise_seed=pn_seed,
            replica_id=replica_id,
            # R6-D §11.5.1 (R6-EVAL-1): the policy identity mirrored onto
            # the record — the artifact-level checkpoint_hash is the
            # provenance, the record carries it so dedup never has to
            # guess.
            policy_checkpoint_hash=policy_checkpoint_hash,
            rng_contract_version=RNG_CONTRACT_VERSION,
            # review partB P2-1: the per-side sampler call count — the
            # number of policy sampling steps each side consumed (the
            # raw trace length).
            sampler_call_counts={
                "bc": len(ep_bc["trace"]), "rl": len(ep_rl["trace"]),
            },
            trace={"bc": ep_bc["trace"], "rl": ep_rl["trace"]},
        ))
    return records, eval_invalid


def build_runtime_eval_artifact(
    records: List[EpisodePairedRecord],
    checkpoint_hash: str,
    config_dict: Dict[str, Any],
    contract: RLSystemContract,
    runtime: Dict[str, Any],
    lora_checksum: str,
    observation_invalid_count: int,
    missing_weight_keys: List[str],
    unexpected_weight_keys: List[str],
    history_indices: List[int],
    *,
    offline_density: Optional[OfflineDensity] = None,
    bank_id: Optional[str] = None,
    load_contract: Optional[Dict[str, Any]] = None,
    opportunity_by_entry: Optional[Dict[str, Any]] = None,
) -> EvalArtifact:
    """§4.17 artifact with the runtime contract hash.

    expected_contract_hash = contract_hash_of(contract) at THIS call site —
    the checkpoint hash and the artifact always agree by construction.
    history_indices (observation-selection metadata) is recorded; §7.4:
    use_obstacle_projection=False is written into the artifact (hard-fails
    on True in validation). Capability metrics are computed from the paired
    records against the frozen offline density field (CAPABILITY-AUDIT):
    an episode set missing any capability role, or a missing density
    field, hard-fails here rather than degrading to zero-improvement.
    config_dict carries the persisted config_layer (exact_resolved_config
    embeds it): the artifact's config_hash is recomputed over THAT layer
    (P1-2) so the train-side artifact and the eval-side artifact record
    the same config identity for the same checkpoint. bank_id (P1-3)
    records which frozen bank the episodes came from.

    load_contract (R6-E §11.6.1, F-06) carries the policy's frozen
    weight-load audit (policy._dppo_load_contract) so an artifact that
    records unexpected keys can prove they are exactly the allowlisted
    anchors of the registered BC checkpoint — validation hard-fails on
    unexpected keys without a valid contract.

    opportunity_by_entry (R6-EVAL-2, §11.5.3) maps episode_config_hash ->
    the entry's frozen CapabilityOpportunity block; the artifact's
    five-piece capability_outcomes are computed from it (the capability
    gate is driven by opportunity evidence, never by the presence of the
    numerics block). A v3 bank without opportunity blocks passes an
    empty map — every group reports NOT_INFORMATIVE (honest no-evidence,
    never a fabricated score). Forgetting the map can only make the gate
    MORE conservative (NOT_INFORMATIVE), never fake a capability value.
    """
    capability_metrics = compute_capability_metrics(records, offline_density)
    capability_outcomes = compute_capability_outcomes(
        records, opportunity_by_entry or {}, offline_density)
    return build_eval_artifact(
        episodes=records,
        checkpoint_hash=checkpoint_hash,
        config_hash=_config_hash(config_dict.get("config_layer", config_dict)),
        contract_hash=contract_hash_of(contract),
        manifest_hash=runtime["manifest_hash"],
        world_assets_hash=runtime.get("world_assets_hash", ""),
        code_hash=_code_hash(runtime["source_file_hashes"]),
        reward_version=runtime["reward_version"],
        exec_steps=contract.exec_steps,
        observation_invalid_count=observation_invalid_count,
        missing_weight_keys=missing_weight_keys,
        unexpected_weight_keys=unexpected_weight_keys,
        lora_checksum=lora_checksum,
        timestamp=runtime["timestamp"],
        device=runtime["device"],
        capability_metrics=capability_metrics,
        capability_outcomes=capability_outcomes,
        history_indices=history_indices,
        use_obstacle_projection=contract.use_obstacle_projection,
        expected_contract_hash=contract_hash_of(contract),
        bank_id=bank_id or "",
        offline_density=offline_density,
        identity_mode=runtime.get("identity_mode", "strict"),
        source_file_delta=runtime.get("source_file_delta"),
        project_git_commit=runtime.get("project_git_commit", ""),
        load_contract=load_contract,
    )


# ═══════════════════════════════════════════════════════════════════
# checkpoint assembly (§4.16)
# ═══════════════════════════════════════════════════════════════════

def _build_checkpoint_data(
    policy: Any,
    critic: torch.nn.Module,
    contract: RLSystemContract,
    loop_config: LoopConfig,
    trainer_config: TrainerConfig,
    metrics: Dict[str, Any],
    env_steps: int,
    completed: int,
    lora_state: Dict[str, Any],
    runtime: Dict[str, Any],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    counters: Optional[Dict[str, int]] = None,
    numerical_mode_resolved: Optional[Dict[str, Any]] = None,
    episode_metadata: Optional[Dict[str, Any]] = None,
    sampler_state: Optional[Dict[str, Any]] = None,
    *,
    base_seed: int = 0,
    n_envs: int = 1,
    lora_info: Optional[Dict[str, Any]] = None,
    active_slot_states: Optional[Dict[int, Dict[str, Any]]] = None,
    checkpoint_kind: Optional[str] = None,
    # R6-F: the post-update scheduler/scaler instances — their per-batch
    # state is frozen into the checkpoint (the exact state the next
    # resume validates cross-field and continues from).
    actor_sched: Optional[Any] = None,
    critic_sched: Optional[Any] = None,
    reward_scaler: Optional[RunningRewardScaler] = None,
    # R7-6 Step 4 (quiescent-boundary snapshot): the metrics writer
    # cursor (committed metrics.jsonl records at save time), the
    # policy-observation window gate state the world ran under, and the
    # next-value provenance of the update just committed.
    metrics_writer_cursor: int = 0,
    observation_window_provenance: Optional[Dict[str, Any]] = None,
    next_value_provenance: Optional[Dict[str, Any]] = None,
) -> CheckpointData:
    """§5.8 gate 9: ``counters`` carries cumulative totals from
    restore_training_state; without it the batch defaults to per-batch
    numbering (ppo_update_count=1, env_steps, completed, update_batches)
    — the pre-resume behavior. Every counter field is the cumulative
    value, never a per-batch delta. ``numerical_mode_resolved`` is the
    resolved TF32/determinism snapshot recorded in the exact config and
    as the checkpoint's numerical_mode_state (schema v2).

    base_seed/n_envs/lora_info (P0-2, schema v3) feed the
    training_identity config layer and the lora_identity field — the
    save side must pass the SAME values the restore side's
    resume_identity_for will use, or the identity gate refuses the
    resume."""
    config_dict = _config_dict(
        contract, loop_config, trainer_config, runtime,
        numerical_mode_resolved, policy,
        base_seed=base_seed, n_envs=n_envs, lora_info=lora_info,
        actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
    )
    # Runtime artifacts excluded from config_hash (not reproducible at
    # resume time — metrics is post-update): the resolved GAE chunk the
    # critic forward actually used, and the last episode's full record
    # (manifest source + sampled episode hashes). They persist only in
    # exact_resolved_config. "config_layer" is the EXACT dict that was
    # hashed (P1-2): the eval side recomputes config_hash over it, so
    # this copy must never diverge from config_dict (no merged runtime
    # artifacts inside).
    exact_resolved_config = {
        **config_dict,
        "config_layer": config_dict,
        "trainer": {
            **config_dict["trainer"],
            "next_value_batch_size_resolved": metrics["next_value_batch_size"],
        },
        "episode_metadata": episode_metadata or {},
    }
    # The sampler's attempt/cursor/RNG state persists with the checkpoint
    # (SEED-RESUME production path). {} resolves the empty case the same
    # way rng_states does — the schema is 1.1, so the field is always
    # present at load (build/load are fields()-driven).
    if (checkpoint_kind or "resume") == "resume":
        # R5.5: the frozen base UNet is NOT duplicated — the checkpoint
        # stores the mutable state (LoRA via lora_state, obs encoder,
        # normalizer, value head, optimizers); the base is re-materialized
        # from the BC checkpoint on restore, bound by bc_checkpoint_hash.
        policy_state = {
            k: v for k, v in _policy_state_dict(policy).items()
            if not k.startswith("model.")
        }
    else:
        policy_state = _policy_state_dict(policy)
    return CheckpointData(
        schema_version=SCHEMA_VERSION,
        official_dppo_commit=OFFICIAL_DPPO_COMMIT,
        project_git_commit=runtime["project_git_commit"],
        project_tree_hash=runtime["project_tree_hash"],
        source_file_hashes=runtime["source_file_hashes"],
        contract_serialized=contract_to_checkpoint_bytes(contract),
        contract_hash=contract_hash_of(contract),
        manifest_hash=runtime["manifest_hash"],
        bc_checkpoint_hash=runtime["bc_checkpoint_hash"],
        config_hash=_config_hash(config_dict),
        policy_state=policy_state,
        checkpoint_kind=checkpoint_kind or "resume",
        # value_state holds exactly the value head (mlp.*) —
        # critic.value_state_dict() never embeds policy keys, so the
        # checkpoint cannot re-pollute the policy on restore.
        value_state={
            k: v.detach().cpu().clone()
            for k, v in critic.value_state_dict().items()
        },
        lora_state=lora_state,
        # The LoRA configuration identity (schema v3, P0-1): axes from
        # the live wiring + checksum over the saved parameter set. The
        # restore side hard-fails when this does not match the live
        # wiring — never a silently different actor.
        lora_identity=build_lora_identity(lora_state, lora_info or {}),
        actor_optimizer_state=_optimizer_state_cpu(actor_optimizer),
        value_optimizer_state=_optimizer_state_cpu(critic_optimizer),
        # R6-F (§11.7.1): the scheduler state is the full structured
        # checkpoint form {method, version, actor, critic} — never an
        # {"enabled": ...} payload; the none branch is structured too
        # (method/version/step_count), so a disabled scheduler is
        # DECLARED, not silent.
        lr_scheduler_state=scheduler_state_dict(actor_sched, critic_sched),
        sampling_std_state={
            "adaptive": False,  # no adaptive sampling std
            "min_sampling_denoising_std": (
                loop_config.min_sampling_denoising_std
            ),
            "min_logprob_denoising_std": (
                loop_config.min_logprob_denoising_std
            ),
        },
        # R6-F (§11.7.3): reward_norm_state carries the scaler's
        # per-batch mutable state (structured for BOTH branches — never
        # an empty dict). env is the only normalization source; a loaded
        # state is never overwritten by a live one (附录 H anti-pattern).
        reward_norm_state=RewardNormState(
            version=runtime["reward_version"],
            method=loop_config.reward_scale.method,
            params=reward_scaler.state_dict(),
        ).to_dict(),
        rng_states=_rng_states(),
        numerical_mode_state=numerical_mode_resolved or {"enabled": False},
        # Environment-sequence RNG identity (schema v4, C1): the episode
        # sequence is driven by the sampler state (SEED-RESUME); a fixed
        # non-empty PYTHONHASHSEED is recorded with restorable=True (the
        # formal entries hard-fail on unset/"random" via
        # require_fixed_python_hashseed preflight), enabling the
        # cross-process env-RNG resume closed in rework #3.
        environment_rng_state=build_environment_rng_state(),
        sampler_state=sampler_state or {},
        # R5.3: mid-episode ACTIVE slot states ride the checkpoint —
        # the next update restores them and continues the SAME episodes.
        active_slot_states=active_slot_states or {},
        # cumulative training iteration: fresh batch = 1, each resumed
        # batch continues from the restored checkpoint's value.
        outer_iteration=counters["outer_iteration"] if counters else 1,
        ppo_update_count=(
            counters["ppo_update_count"] if counters else 1
        ),
        optimizer_step_count=(
            counters["optimizer_step_count"]
            if counters else int(metrics["update_batches"])
        ),
        environment_transition_count=(
            counters["environment_transition_count"]
            if counters else env_steps
        ),
        completed_episode_count=(
            counters["completed_episode_count"]
            if counters else completed
        ),
        wall_clock_timestamp=runtime["timestamp"],
        exact_resolved_config=exact_resolved_config,
        # ── R7-6 schema v6 save-side axes (guide R7-6 Step 1/4) ────────
        # The checkpoint is written ONLY at the quiescent update
        # boundary: every axis below is the post-update committed state.
        # LoRA mapping: the ordered parameter names of the saved state
        # (sorted — the same order lora_checksum_of hashes) and of the
        # actor optimizer's param group (drift from the live wiring is
        # caught at restore BEFORE any tensor load).
        lora_name_order=sorted(lora_state.keys()),
        actor_optimizer_param_names=_actor_optimizer_param_names(
            actor_optimizer, policy),
        # Vendor binding (R7-1) — injected by the launcher's build_runtime
        # from the verified frozen vendor snapshot; "" in fixture worlds
        # that never claim a vendor binding.
        official_vendor_commit=runtime.get("official_vendor_commit", ""),
        vendor_manifest_hash=runtime.get("vendor_manifest_hash", ""),
        # Double axis of the runtime source set (validate_v6_contract
        # demands equality) — mirrors runtime["source_file_hashes"].
        official_runtime_source_files=dict(runtime["source_file_hashes"]),
        # Owned persistent Generators (name -> state). The production
        # path currently owns none (the synth far-dome generator is
        # created per-call with manual_seed — stateless); a non-empty
        # set recorded here must be restorable at resume.
        owned_generators=_owned_generator_states(),
        metrics_writer_cursor=metrics_writer_cursor,
        voided_episode_count=int(metrics.get("voided_episodes", 0)),
        observation_window_provenance=(
            observation_window_provenance or {}),
        next_value_provenance=next_value_provenance or {},
    )


# ═══════════════════════════════════════════════════════════════════
# main entry
# ═══════════════════════════════════════════════════════════════════

def run_e2e_batch(
    env: Any,
    policy: Any,
    critic: torch.nn.Module,
    contract: RLSystemContract,
    loop_config: LoopConfig,
    trainer_config: TrainerConfig,
    task_specs: List[EpisodeTaskSpec],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    lora_info: Dict[str, Any],
    runtime: Dict[str, Any],
    resume_from: Optional[str] = None,
    eval_env: Optional[Any] = None,
    *,
    task_sampler: EpisodeTaskSampler,
    paired_bank: Optional[Sequence[PairedBankEntry]] = None,
    offline_density: Optional[OfflineDensity] = None,
    bank_id: Optional[str] = None,
    resume_active_slots: Optional[Dict[int, Dict[str, Any]]] = None,
    checkpoint_kind: Optional[str] = None,
    # R7-6 Step 4: the metrics writer cursor — how many metrics.jsonl
    # records the caller has committed before this batch (the launcher
    # passes its batch index; the resumed process continues numbering
    # from the checkpoint's recorded cursor). None = unnumbered world.
    metrics_writer_cursor: Optional[int] = None,
) -> Dict[str, Any]:
    """One e2e batch: collect transitions, optionally update, save, evaluate.

    task_sampler is REQUIRED: every episode attempt calls
    task_sampler.next_attempt() and resets the env with
    seeds=[sample.episode_seed] — the episode seed/episode id come from the
    shared persistent sampler (never task_specs[(attempt-1)%len] /
    seeds=[attempt], which restarted at 1 every batch). The sampler is
    stateful across batches: consecutive run_e2e_batch calls continue the
    global attempt/episode counters and the seed sequence (SEED-RESUME).

    [§5.8 gate 9] resume_from: a checkpoint path whose full training state
    (policy + live LoRA + critic + optimizers + RNG snapshot) is restored
    before this batch runs. The checkpoint's cumulative counters then carry
    into this batch's checkpoint: update N of a resumed run is numbered N,
    and a same-seed continuation reproduces the uninterrupted world's
    metrics. Passing a checkpoint produced by THIS world is an idempotent
    no-op — it simply continues the counters.

    Returns: buffer, transitions, env_steps, completed_episodes,
    voided_episodes, rejected_episodes, routing, gate_summary, metrics,
    checkpoint_path, eval_artifact, eval_json_path, lora_state,
    lora_checksum, role_stats. env_steps == len(buffer) (transitions added;
    faulting steps never enter the buffer). Raises
    EpisodeCollectionIncomplete when attempt_limit is exhausted before
    n_episodes completed.
    The paired-eval artifact is persisted as
    `paired_eval_envstep_<9d>_ppoupd_<6d>_<ckpt12>.json`; eval rollouts run their
    own point-cloud validity gate and hard-fail on any voided paired
    seed.

    eval_n_episodes>0 requires `eval_env` — a dedicated n_envs=1 env;
    passing None is a hard-fail, never an implicit reuse of the (possibly
    multi-slot, possibly ACTIVE-slot) training env. It ALSO requires
    `paired_bank`: the eval replays the frozen bank entries exactly
    — no runtime task re-selection — and len(paired_bank) must equal
    eval_n_episodes. Capability metrics need `offline_density`: without
    the frozen offline density field the eval cannot measure support
    expansion or low-density recovery — passing None is a hard-fail, never
    an implicit zero-improvement artifact (CAPABILITY-AUDIT).

    lora_info is the insert_lora return dict; the LoRA snapshot for the
    checkpoint is extracted AFTER the PPO update, so the checkpoint holds
    the trained actor state — never the insertion-time snapshot (adaptors
    are hook closures invisible to policy.state_dict()).

    §7.4: use_obstacle_projection must be False in formal training — a
    policy loaded with projection enabled hard-fails.
    """
    device = loop_config.device
    # The eval contract is an ENTRY preflight — any rollout, resume
    # restore, PPO update, or file write must never run under a
    # misconfigured eval request. All four conditions are validated before
    # any mutation; run_paired_eval keeps its own n_envs==1 check
    # (defense-in-depth).
    if loop_config.eval_n_episodes > 0:
        if loop_config.mode != "one_update":
            raise RuntimeError(
                "paired eval is only supported in one_update mode"
            )
        if eval_env is None:
            raise RuntimeError(
                "eval_n_episodes>0 requires an explicit eval_env"
            )
        if getattr(eval_env, "n_envs", None) != 1:
            raise RuntimeError(
                f"paired eval requires eval_env.n_envs == 1; got"
                f" n_envs={getattr(eval_env, 'n_envs', None)}"
            )
        if eval_env is env:
            raise RuntimeError(
                "eval_env must be a dedicated instance, not the training"
                " env — explicit reuse of the training env is forbidden"
            )
        if paired_bank is None:
            raise RuntimeError(
                "eval_n_episodes>0 requires a frozen paired bank —"
                " the eval replays the bank's entries exactly and never"
                " re-selects tasks at runtime"
            )
        if not bank_id:
            # R5.8-E (N4): a caller that passes the entries WITHOUT the
            # bank file id binds the bank's CONTENT instead — the noise
            # seeds and the artifact provenance stay re-derivable.
            bank_id = bank_id_from_entries(list(paired_bank))
        if len(paired_bank) != loop_config.eval_n_episodes:
            raise RuntimeError(
                f"eval bank mismatch: len(paired_bank)={len(paired_bank)} "
                f"!= eval_n_episodes={loop_config.eval_n_episodes} — the"
                " bank IS the eval episode list"
            )
        if offline_density is None:
            raise RuntimeError(
                "eval_n_episodes>0 requires the offline density field —"
                " capability metrics (support expansion, dense recovery)"
                " cannot be computed without it"
            )
    # R6-A (F-01/F-02, §11.2.2/§11.2.3): the collection-mode contract is
    # enforced at the entry — a transition-mode batch without its derived
    # per-batch target cannot collect by a plan, and multi-env cannot
    # guarantee the exact remaining capacity (G8 hard pin n_envs=1).
    if loop_config.collection_mode == "transition":
        if loop_config.target_transitions is None:
            raise RuntimeError(
                "transition collection requires the derived per-batch "
                "target_transitions (min(horizon, total - committed)) — "
                "refusing to collect without a plan (R6-A §11.2.2)"
            )
        if int(getattr(env, "n_envs", 1)) != 1:
            raise RuntimeError(
                "G8 hard pin (R6-A §11.2.3): transition collection "
                f"requires n_envs == 1; got {int(getattr(env, 'n_envs', 1))}"
                " — multi-env cannot guarantee the exact remaining "
                "transition capacity"
            )
    # Formal numerical mode — single configuration source (guide R7-3
    # Step 1): the production entry applies the mode in preflight before
    # policy load / LoRA / optimizer construction
    # (entrypoint_guard.validate_numerical_mode); run_e2e_batch asserts
    # it is already configured instead of applying a second
    # configuration source. Callers that bypass preflight (tests,
    # diagnostic scripts) must configure before calling. G5/R7-B128
    # identity: max_abs_diff <= 1e-5 at the actual minibatch (B=128)
    # requires TF32 off + deterministic algorithms.
    numerical_mode_resolved = assert_numerical_mode_configured(
        FORMAL_NUMERICAL_MODE
    )
    # R6-F (§11.7.1/§11.7.2/§11.7.3): the scaler/scheduler live ACROSS
    # batches and are checkpointed per batch. Fresh state on a new run;
    # restore loads the saved per-batch state (cross-field validated
    # inside restore). The scheduler construction applies the vendor's
    # lr = min_lr immediately (the torch _LRScheduler constructor's
    # implicit step, replicated for sequence parity — NOT counted as a
    # committed step); on resume, the scheduler load re-applies the LR at
    # the saved step position AFTER the optimizer state load, so the
    # optimizer-contract validation (R6-E) sees the saved position's LR,
    # and the next update observes the resumed LR.
    reward_scaler = RunningRewardScaler(
        loop_config.reward_scale, int(env.n_envs), contract.gamma_env)
    actor_sched, critic_sched = build_lr_schedulers(
        actor_optimizer, critic_optimizer, loop_config.lr_scheduler)
    # §5.8 gate 9: restore the full training state before the batch runs;
    # the restored counters seed the checkpoint numbering below.
    base_counters: Dict[str, int] = {}
    restored_state_hash: Optional[str] = None
    if resume_from is not None:
        # RESUME-HASH-GATE: the runtime identity is validated against the
        # checkpoint BEFORE any state_dict load — a checkpoint saved under
        # a different contract/manifest/config/code is a hard error, never
        # a silently-different continuation. The runtime identity includes
        # the training_identity axes (base_seed/n_envs/lora/optimizers):
        # the save side and this restore side MUST encode the same values
        # or the config_hash gate refuses the resume (P0-2).
        base_counters = restore_training_state(
            resume_from, policy, critic, lora_info,
            actor_optimizer, critic_optimizer,
            runtime_identity=resume_identity_for(
                contract, loop_config, trainer_config, runtime,
                numerical_mode_resolved, policy,
                base_seed=getattr(task_sampler, "base_seed", None),
                n_envs=int(env.n_envs),
                lora_info=lora_info,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
            ),
            task_sampler=task_sampler,
            structured_identity=_structured_identity(
                loop_config, runtime, numerical_mode_resolved,
            ),
            # R6-F: the scheduler/scaler per-batch state + configs — the
            # cross-field validation and load run inside restore, before
            # any policy forward.
            loop_config=loop_config,
            reward_scaler=reward_scaler,
            actor_sched=actor_sched,
            critic_sched=critic_sched,
            n_critic_warmup_updates=trainer_config.n_critic_warmup_updates,
        )
        resume_active_slots = base_counters.get("active_slot_states") or None
        # SEED-RESUME: the sampler's attempt/cursor/RNG state is loaded
        # INSIDE restore (identity validated there, before any tensor
        # load) — no call-site load after the fact.
        # R7-6 Step 6 (layer 1): the load-complete state hash — the
        # resume side of the same-seed comparator; exact under the
        # deterministic runtime, consumed only by
        # scripts/r7_compare_resume.py.
        restored_state_hash = _state_hash(
            policy, critic, actor_optimizer, critic_optimizer, task_sampler)
    policy_projection = bool(
        getattr(policy, "use_obstacle_projection", False)
    )
    if policy_projection != contract.use_obstacle_projection:
        raise RuntimeError(
            "use_obstacle_projection mismatch (§7.4): policy="
            f"{policy_projection} contract="
            f"{contract.use_obstacle_projection} — obstacle projection is "
            "policy-external trajectory rewriting; formal training/eval "
            "requires False"
        )
    trainer_config = replace(trainer_config, device=device)
    gate = PointCloudValidityGate(contract, phase=loop_config.gate_phase)
    cam_far = float(env.cam_far)
    buffer = RolloutBuffer(contract)
    routing = {
        "REJECT_EPISODE": 0,
        "SAFE_END_EPISODE": 0,
        "EXCLUDE_TRANSITION": 0,
        "HARD_FAILURE": 0,
    }
    rejected = 0
    completed = 0
    # AdmissionExhausted resets consumed by the loop — one per blocked
    # task draw; reported on EpisodeCollectionIncomplete and in the result
    # dict (never swallowed silently).
    admission_blocked = 0
    # R6-A (F-02, §11.2.3): the attempt limit gates episode START. In
    # transition mode the episode count no longer bounds the refill (each
    # episode contributes >= 1 transition), so the budget worth of episode
    # starts plus the rejection/blocked slack must fit inside the limit.
    if loop_config.collection_mode == "transition":
        attempt_limit = (
            max(2 * loop_config.n_episodes, 4) + 4
            + int(loop_config.target_transitions)
        )
    else:
        attempt_limit = max(2 * loop_config.n_episodes, 4) + 4
    # Truthful per-role accounting — attempted (reset-accepted), completed
    # (terminal state, no fault), voided (observation fault). A voided
    # episode is NOT completed; a rejected reset is not attempted.
    attempted_counts: Dict[str, int] = {r: 0 for r in ROLE_NAMES}
    completed_counts: Dict[str, int] = {r: 0 for r in ROLE_NAMES}
    voided_counts: Dict[str, int] = {r: 0 for r in ROLE_NAMES}
    # Selection metadata is deterministic per contract (latest T_o of the
    # T_h buffer); the per-reset call refreshes it, this is the
    # all-episodes-rejected fallback so the result key always exists.
    sel_meta = ObservationSelection(
        history_indices=list(
            range(contract.history_steps - contract.obs_steps,
                  contract.history_steps)
        ),
        history_steps=contract.history_steps,
        obs_steps=contract.obs_steps,
    )

    def _live_lora_state() -> Dict[str, Any]:
        """Post-update (live) LoRA snapshot; {} when no adaptors exist."""
        if not lora_info.get("lora_params"):
            return {}
        return extract_lora_state(lora_info)

    # Old logprobs are computed at rollout time per transition
    # (sample_dppo_rollout at B=1, the behavior batch, with the policy's
    # CFG lambda) and written to the buffer once — buffer old logprobs must
    # never be recomputed/redefined after the rollout.
    #
    # The loop drives the synchronous vector env through an ACTIVE MASK —
    # one slot-state per env id (None = inactive). Each iteration refills
    # exactly the budgeted number of inactive slots via reset_at() (one
    # next_attempt() per slot; a rejected reset consumes the attempt and
    # stays inactive), then samples/executes actions for the ACTIVE slots
    # in one batched step_at. A slot that terminated/truncated is resolved
    # and deactivated; the adapter raises if a done slot is ever stepped
    # without a reset_at. n_envs=1 reduces to the sequential semantics.
    #
    # Budget semantics: refill count = remaining target MINUS the slots
    # already active, so completed + active <= n_episodes holds at every
    # iteration — when completed reaches n_episodes, no episode is left in
    # flight (attempted == completed + voided exactly; nothing is silently
    # dropped at collection stop). The attempt limit gates EPISODE START
    # (refill), not stepping — an episode already admitted runs to
    # completion (fault / terminate / max_ep_len) even when its last step
    # lands at the budget edge.
    n_envs = int(getattr(env, "n_envs", 1))
    slots: List[Optional[dict]] = [None] * n_envs
    target = loop_config.target_transitions
    active_slot_states: Dict[int, Dict[str, Any]] = {}
    # R5.3 resume: restore mid-episode ACTIVE slots (exact continuation —
    # the next transition must be bitwise-identical to the uninterrupted
    # run: task/state/history/RNG restored from the checkpoint's per-slot
    # state). Restored slots re-enter the loop ACTIVE with their recorded
    # metadata; obs_b is rebuilt from the restored env history.
    def _adopt_slot(env_id: int, step: int) -> None:
        rec = env.slot_record(env_id)
        spec = env.slot_spec(env_id)
        env_obs = env.slot_obs(env_id)
        # R7-5: the first observation after a restore is gated like any
        # other — the restored window (production provenance when the
        # adapter provides it, content-only fallback otherwise) must pass
        # before the slot resumes. An invalid restored window ends the
        # episode as observation_fault (never collision/reach) and drops
        # the slot; the refill resets it with a fresh episode.
        _mapping = None
        _mapper = getattr(env, "observation_window_mapping", None)
        if callable(_mapper):
            _mapping = _mapper(env_obs, env_id)
        restored_window = validate_policy_observation_window(
            _mapping or build_observation_window_mapping(
                {kk: vv[None] for kk, vv in env_obs.items()},
                contract.obs_steps, cam_far),
            obs_steps=contract.obs_steps,
            gate=gate,
        )
        if not restored_window.valid:
            if env.is_active(env_id):
                env.mark_voided([env_id])
            return  # slot stays None — refill resets it
        # _selected consumes the stacked (n_envs, ...) form (same as
        # the reset path) — add the single-slot batch axis.
        obs_b, _ = _selected(
            {kk: vv[None] for kk, vv in env_obs.items()},
            device, contract.obs_steps,
        )
        slots[env_id] = {
            "spec": spec,
            "ep_id": (
                f"{spec.world_uid}:{spec.role}:"
                f"{rec.episode_seed:09d}"
            ),
            "obs_b": obs_b,
            "step": step,
            "ep_has_fault": False,
            "done": False,
        }

    if resume_active_slots:
        for env_id, st in resume_active_slots.items():
            eid = int(env_id)
            if env.is_active(eid):
                # P0-2 fix (in-process continuation): the live ACTIVE
                # slot IS the state the checkpoint was saved from —
                # adopt it directly. load_state_dict only restores
                # CLOSED slots (cold-process resume); calling it on a
                # live ACTIVE slot raises by adapter guard.
                _adopt_slot(eid, int(st["env"]["episode_step"]))
            else:
                env.load_state_dict(eid, st)
                _adopt_slot(eid, int(st["env"]["episode_step"]))
    if target is not None:
        # In-process continuation: live ACTIVE slots (not covered by an
        # explicit resume) are adopted, never reset_at-replaced.
        for env_id in range(env.n_envs):
            if slots[env_id] is None and env.is_active(env_id):
                _adopt_slot(
                    env_id,
                    int(env.state_dict(env_id)["env"]["episode_step"]),
                )
    attempt = 0
    # The budget check must live INLINE in the while condition — a
    # cached flag computed at the body start goes stale by the time the
    # condition re-evaluates (off-by-one overshoot observed). R6-A
    # (F-02, §11.2.3): the two stop conditions are MUTUALLY EXCLUSIVE —
    # episode mode stops on completed episodes, transition mode stops on
    # the transition budget; the episode count is sampler/admission
    # statistics in transition mode and must never truncate the budget.
    def _collection_complete() -> bool:
        if loop_config.collection_mode == "episode":
            return completed >= loop_config.n_episodes
        return len(buffer) >= int(target)

    while not _collection_complete():
        # ── precise refill budget: the number of NEW episodes this pass =
        #    remaining target MINUS the slots already ACTIVE — never blindly
        #    fill every inactive slot. This keeps completed + active <=
        #    n_episodes at every iteration, so when completed reaches
        #    n_episodes nothing is left in flight and attempted == completed
        #    + voided exactly. Transition mode: capacity = the free env
        #    slots (n_episodes must not enter the refill — §11.2.3).
        active_count = sum(s is not None for s in slots)
        if loop_config.collection_mode == "episode":
            budget = loop_config.n_episodes - completed - active_count
        else:
            budget = n_envs - active_count
        # ── refill loop (attempt limit): start episodes until the
        #    remaining budget is covered OR the attempt limit gates further
        #    starts. A REJECTed reset consumes an attempt and is retried
        #    here (its slot stays inactive), so a rejected reset never
        #    leaves a runnable budget idle; the attempt limit gates episode
        #    START, never stepping — an episode already admitted runs to
        #    completion (fault / terminate / max_ep_len).
        # ── per-slot admission: ONE reset_at call per refilled slot, so an
        #    AdmissionExhausted is attributable to a single slot and that
        #    slot stays fully reset-able (reset_at is two-phase atomic —
        #    nothing mutated). A blocked admission consumes the attempt and
        #    the loop draws the NEXT sampler task — the sampler's own
        #    advance IS the world/task switch (design v1.1:823 rule 7: no
        #    fixed-coordinate fallback). A REJECTed reset frame closes the
        #    slot explicitly (reject_episode) and also retries with the
        #    next task. Per-slot resets process exactly ``refill``
        #    (<= budget) slots per pass.
        refill = [
            i for i, s in enumerate(slots) if s is None
        ][:budget]
        for env_id in refill:
            if attempt >= attempt_limit:
                break  # attempt budget exhausted — stop starting episodes
            admitted = False
            while not admitted and attempt < attempt_limit:
                sm = task_sampler.next_attempt()
                attempt += 1
                spec = sm.task_spec
                try:
                    obs, info = env.reset_at(
                        [env_id], [spec], [sm.episode_seed])
                except AdmissionExhausted:
                    # rule 7: no fixed-coordinate fallback — the slot is
                    # untouched (two-phase atomic), so the next sampler
                    # task reuses it directly.
                    admission_blocked += 1
                    continue
                # The episode id is the SAME unique id the adapter's
                # EpisodeConfigRecord carries (world:role:episode_seed) —
                # global_episode_index counts episode STARTS, not attempts,
                # and diverged from the record id on rejections.
                ep_id = (
                    f"{spec.world_uid}:{spec.role}:"
                    f"{sm.episode_seed:09d}"
                )
                reset_window = validate_policy_observation_window(
                    info[0].get("observation_window_mapping")
                    or build_observation_window_mapping(
                        obs, contract.obs_steps, cam_far),
                    obs_steps=contract.obs_steps,
                    gate=gate,
                )
                if not reset_window.valid:
                    decision = gate.decide(
                        reset_window.frame_results[-1], "reset", ep_id,
                        False)
                    assert decision == InvalidFramePolicy.REJECT_EPISODE
                    routing["REJECT_EPISODE"] += 1
                    rejected += 1
                    # The reset MUTATED the slot (ACTIVE) — close it
                    # explicitly so the next refill's reset_at on this slot
                    # does not hit the ACTIVE preflight hard-fail.
                    env.reject_episode([env_id])
                    continue
                attempted_counts[spec.role] += 1
                obs_b, sel_meta = _selected(
                    obs, device, contract.obs_steps,
                )
                slots[env_id] = {
                    "spec": spec,
                    "ep_id": ep_id,
                    "obs_b": obs_b,
                    "step": 0,
                    "ep_has_fault": False,
                }
                admitted = True
        active_ids = [i for i, s in enumerate(slots) if s is not None]
        if not active_ids:
            break  # attempt budget exhausted mid-refill
        # ── one synchronous step for the ACTIVE slots only (step_at — the
        #    final partial batch < n_envs never touches the other slots;
        #    step_at raises if any named slot is done)
        actions = np.zeros(
            (len(active_ids), contract.exec_steps, contract.action_dim),
            dtype=np.float32,
        )
        values = np.zeros(len(active_ids), dtype=np.float32)
        rollouts: List[Any] = [None] * len(active_ids)
        for k, env_id in enumerate(active_ids):
            s = slots[env_id]
            with torch.no_grad():
                # Rollout-time logprobs at B=1 (the behavior batch) with
                # the policy's CFG lambda — stored once, never recomputed.
                rollouts[k] = sample_dppo_rollout(
                    policy, s["obs_b"], contract,
                    min_sampling_denoising_std=(
                        loop_config.min_sampling_denoising_std),
                    min_logprob_denoising_std=(
                        loop_config.min_logprob_denoising_std),
                )
                values[k] = float(
                    critic(s["obs_b"]).reshape(-1)[0].item()
                )
            executed = policy.normalizer["action"].unnormalize(
                rollouts[k].executed_action[0])
            # unnormalize follows the normalizer's device (CUDA on the
            # fresh-load e2e path) — the actions container is numpy, so the
            # boundary conversion to CPU numpy happens here, before the env
            # boundary (RealUAVEnv/MockUAVEnv.step also accept a tensor per
            # G5 §5.7, but a numpy container cannot hold a CUDA tensor).
            if torch.is_tensor(executed):
                executed = executed.detach().cpu().numpy()
            actions[k] = executed
        obs_new, reward, terminated, truncated, info = env.step_at(
            active_ids, actions)
        for k, env_id in enumerate(active_ids):
            s = slots[env_id]
            spec = s["spec"]
            window = validate_policy_observation_window(
                info[k].get("observation_window_mapping")
                or build_observation_window_mapping(
                    obs_new, contract.obs_steps, cam_far, slot=k),
                obs_steps=contract.obs_steps,
                gate=gate,
            )
            if not window.valid:
                decision = gate.decide(
                    window.frame_results[-1], "rollout", s["ep_id"],
                    episode_first_fault=not s["ep_has_fault"],
                )
                if decision == InvalidFramePolicy.SAFE_END_EPISODE:
                    routing["SAFE_END_EPISODE"] += 1
                    s["ep_has_fault"] = True  # the episode is VOIDED
                    s["done"] = True  # faulting transition dropped (inv. 3)
                    # The explicit ACTIVE -> VOIDED transition — the
                    # adapter must never allow this slot to be stepped as
                    # if nothing happened.
                    env.mark_voided([env_id])
                    continue
                if decision == InvalidFramePolicy.HARD_FAILURE:
                    raise RuntimeError(
                        "hard failure during rollout (formal phase never"
                        " trips the rate gate)"
                    )
                raise RuntimeError(
                    "EXCLUDE_TRANSITION unreachable in-loop: the first fault"
                    " always SAFE_ENDs the episode"
                )
            obs_b_next, _ = _selected(
                {kk: vv[k][None] for kk, vv in obs_new.items()},
                device, contract.obs_steps,
            )
            env_terminated = bool(np.asarray(terminated)[k])
            env_truncated = bool(np.asarray(truncated)[k])
            # s["step"] is still the count of PREVIOUSLY completed steps —
            # this step is the (s["step"]+1)-th, matching the original
            # post-increment check (max_ep_len completed steps per episode).
            loop_trunc = (s["step"] + 1) >= loop_config.max_ep_len
            # When the LOOP truncates at max_ep_len the env never signaled
            # it — the final transition must be recorded with
            # truncated=True (effective_truncated) and the adapter slot
            # closed explicitly, or the next episode's reset_at on this
            # slot hits the ACTIVE hard-fail.
            effective_truncated = env_truncated or (
                loop_trunc and not env_terminated
            )
            # The transition carries its rollout-time B=1 old logprobs
            # (computed inside sample_dppo_rollout) and is added
            # immediately — written once, never overwritten.
            buffer.add(to_rollout_transition(
                rollouts[k],
                obs_np=_buffer_obs(s["obs_b"]),
                next_obs_np=_buffer_obs(obs_b_next),
                reward=float(np.asarray(reward)[k]),
                terminated=env_terminated,
                truncated=effective_truncated,
                value_old=values[k],
                episode_id=s["ep_id"],
                episode_step=s["step"],
                world_uid=spec.world_uid,
                role=spec.role,
                # Both hashes ride the buffer trace — manifest source hash
                # from the spec, sampled-episode hash from reset info.
                manifest_entry_hash=spec.manifest_entry_hash,
                episode_config_hash=info[k]["episode_config_hash"],
                observation_valid=True,
                contract=contract,
                env_id=env_id,
                # R5.8-C: outcome truth from the env's EpisodeOutcome
                # (the terminal transition carries the episode's real
                # success/collision; no reward-threshold re-inference).
                done_reason=str(
                    info[k].get("outcome", {}).get("done_reason", "none")
                ),
                success=bool(
                    info[k].get("outcome", {}).get("success", False)
                ),
                collision=bool(
                    info[k].get("outcome", {}).get("collision", False)
                ),
                # R7-5 dual hashes from the production info (absent on
                # mock paths — buffer.add() then skips the hard-fail);
                # populated transitions are hard-checked in add().
                source_manifest_hash=info[k].get("source_manifest_hash"),
                runtime_episode_hash=info[k].get("runtime_episode_hash"),
            ))
            s["obs_b"] = obs_b_next
            s["step"] += 1
            s["done"] = bool(env_terminated or env_truncated)
            if loop_trunc:
                # loop-level truncation: the env did not end the episode,
                # so the slot is still ACTIVE — close it explicitly.
                if not (env_terminated or env_truncated):
                    env.mark_truncated([env_id])
                s["done"] = True
        # ── resolve finished slots (done or faulted) and deactivate them
        for env_id, s in enumerate(slots):
            if s is None or not s.get("done", False):
                continue
            if s["ep_has_fault"]:
                voided_counts[s["spec"].role] += 1
            else:
                completed += 1
                completed_counts[s["spec"].role] += 1
                task_sampler.mark_completed()
            slots[env_id] = None

    # ── R5.3: transition-budget boundary — when the budget stopped the
    #    loop mid-episode, the ACTIVE slots continue in the next update:
    #    their final transition bootstraps like truncated (rollout_cut)
    #    but the env slot is NOT closed; its full state rides the result
    #    (checkpoint) for exact resume.
    if target is not None:
        for env_id, s in enumerate(slots):
            if s is None or s.get("done", False):
                continue
            # rollout_cut: flip the slot's final stored transition to
            # truncated=True (GAE bootstrap) — env stays ACTIVE.
            for idx in range(len(buffer.episode_ids) - 1, -1, -1):
                if buffer.episode_ids[idx] == s["ep_id"]:
                    buffer.truncated[idx] = True
                    # R6-C (§11.4.2): mark the cut explicitly so the
                    # statistic layer never counts it as a completed
                    # episode (done_reason stays "none" and the flip's
                    # truncated=True would otherwise look like a
                    # loop-level truncation).
                    buffer.rollout_cut[idx] = True
                    break
            active_slot_states[env_id] = env.state_dict(env_id)

    if completed < loop_config.n_episodes and target is None:
        # attempt_limit exhausted with the batch incomplete — never return
        # a partial batch as success.
        raise EpisodeCollectionIncomplete(
            requested_episodes=loop_config.n_episodes,
            attempted_episodes=sum(attempted_counts.values()),
            completed_episodes=completed,
            rejected_episodes=rejected,
            voided_episodes=sum(voided_counts.values()),
            admission_blocked_episodes=admission_blocked,
            attempt_limit=attempt_limit,
        )

    env_steps = len(buffer)
    # R6-A (F-01, §11.2.4): the release decision is POST-HOC — never a
    # pre-proven target predicate. A batch materializes the release only
    # when its ACTUAL env steps land exactly on the run total; the
    # caller's checkpoint_kind is intent, and a release demanded without
    # finality is a hard error (the F-01 self-proving pattern — every
    # batch a release — must die loudly).
    effective_kind = checkpoint_kind or "resume"
    if loop_config.collection_mode == "transition":
        committed_before = base_counters.get(
            "environment_transition_count", 0)
        finality = is_budget_complete(
            loop_config.total_transition_budget, committed_before,
            env_steps,
        )
        actual_matches_plan = env_steps == int(loop_config.target_transitions)
        if checkpoint_kind == "release" and not (
                finality and actual_matches_plan):
            raise RuntimeError(
                "checkpoint_kind='release' requested but this batch does "
                "NOT complete the transition budget — actual env_steps="
                f"{env_steps} != planned {loop_config.target_transitions} "
                f"(committed {committed_before}+{env_steps} of total "
                f"{loop_config.total_transition_budget}); the release "
                "decision is post-hoc (is_budget_complete), never "
                "pre-proven (R6-A/F-01)"
            )
        if finality and actual_matches_plan:
            effective_kind = "release"
    # Truthful per-role statistics — attempted (reset-accepted), completed,
    # voided (observation fault, never completed), and the transitions each
    # role actually contributed to the buffer.
    transition_counts = {
        r: Counter(buffer.roles).get(r, 0) for r in ROLE_NAMES
    }
    role_stats = role_stats_report(
        attempted_counts, completed_counts, voided_counts, transition_counts
    )
    # The metric report is the attempted-episode proportions
    # (RoleSampler.proportion_report — single implementation; the 10k
    # targets are gated in G3).
    role_proportion_report = role_stats["attempted_episode_proportions"]
    gate_summary = gate.summary()

    metrics: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[str] = None
    eval_artifact: Optional[EvalArtifact] = None
    eval_json_path: Optional[str] = None
    lora_state: Dict[str, Any] = {}
    if loop_config.mode == "one_update" and len(buffer) > 0:
        # review partB P1-4: the BC snapshot exists ONLY for the paired
        # eval — a no-eval training batch must not deepcopy the whole
        # policy every update.
        bc_policy = (
            _bc_snapshot(policy) if loop_config.eval_n_episodes > 0
            else None
        )
        # R7-4: GAE consumes a SINGLE value function — the rollout
        # critic. Every transition's next value comes from a chunked
        # critic forward over ITS OWN next_obs (the successor-value
        # shortcut is deleted — it leaked the successor transition's
        # value_old across reset/interleave boundaries), chunked by
        # trainer_config.next_value_batch_size, BEFORE the update
        # touches the critic.
        fill_buffer_next_values(
            buffer,
            critic,
            chunk_size=trainer_config.next_value_batch_size,
        )
        # R7-3 Step 3: the actual-minibatch official update — ONE
        # facade.loss (vendor PPODiffusion.loss) per B=128 minibatch,
        # no B=1 microbatch loop. Build the loss facade and the cond
        # encoder closure once per update; ppo_update drives them.
        # Explicit set_timesteps keeps the facade's DDIM tables aligned
        # with the rollout chains regardless of prior scheduler state
        # (idempotent: recomputes the same timesteps).
        policy.noise_scheduler.set_timesteps(contract.denoising_steps)
        facade = build_official_phase1a_facade(
            model=policy.model,
            scheduler=policy.noise_scheduler,
            ft_denoising_steps=contract.ft_denoising_steps,
            cfg_lambda=_require_policy_cfg_lambda(policy),
            device=device,
            horizon_steps=contract.action_horizon,
            action_dim=contract.action_dim,
            eta=1.0,
            min_sampling_denoising_std=loop_config.min_sampling_denoising_std,
            min_logprob_denoising_std=loop_config.min_logprob_denoising_std,
            # the vendor loss calls self.critic(obs).view(-1) on the
            # flat cond — OfficialValueHead pools it to the shared
            # value-head mlp (bitwise equal to the rollout critic on raw
            # obs; the critic optimizer owns the mlp).
            critic=OfficialValueHead(critic),
            gamma_denoising=contract.gamma_denoising,
            clip_vloss_coef=trainer_config.clip_vloss_coef,
            norm_adv=True,
            randn_clip_value=10.0,
        )

        def encode_cond(obs_b):
            cond, uncond = _obs_conditions(policy, obs_b)
            return make_official_cond(cond, uncond, device)

        # R6-F §11.7.2: per-update LR snapshot — the "before" is the
        # scheduler position at update start (fresh: min_lr from the
        # vendor construction step; resumed: the loaded step position).
        lr_before = {"actor": actor_sched.lr, "critic": critic_sched.lr}
        metrics = ppo_update(
            buffer, contract, facade, encode_cond,
            actor_optimizer, critic_optimizer, trainer_config,
            # R5.8-E (N7): the production variance is PHYSICAL-space
            # (the same space measure_bc_variance measures).
            action_normalizer=policy.normalizer["action"],
            # R5.8-G (C-4): the global update index drives the
            # critic-warmup window (cumulative across resumes).
            update_index=base_counters.get("outer_iteration", 0) + 1,
            # R6-F §11.7.3: the scaler transforms the training reward
            # INSIDE ppo_update (before GAE) — buffer rewards /
            # artifact/episode returns stay raw.
            reward_scaler=reward_scaler,
        )
        # R6-F §11.7.2: exactly one pair-step per COMMITTED update — a
        # failed/unpublished update raised above and committed no
        # scheduler state; the actor advances only when this update
        # actually stepped it (critic-only warmup window).
        step_scheduler_pair(
            actor_sched, critic_sched,
            actor_stepped=bool(metrics["actor_stepped"]),
        )
        # R6-F §11.7.2/§11.7.3: per-batch LR + step-count observability
        # (always recorded — none branches keep LR constant while step
        # counts advance; the metrics carry the scheduler position the
        # checkpoint now freezes).
        metrics.update(
            actor_lr_before=lr_before["actor"],
            actor_lr_after=actor_sched.lr,
            critic_lr_before=lr_before["critic"],
            critic_lr_after=critic_sched.lr,
            actor_scheduler_step_count=actor_sched.step_count,
            critic_scheduler_step_count=critic_sched.step_count,
            reward_scaler_step_count=reward_scaler.step_count,
        )
        # Per-batch role proportion statistics recorded
        metrics["role_proportion_report"] = role_proportion_report
        lora_state = _live_lora_state()  # post-update snapshot
        lora_checksum = lora_checksum_of(lora_state)
        output_dir = Path(loop_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Cumulative counters — this batch's deltas on top of the restored
        # base ({} when never resumed).
        counters = {
            # cumulative training iteration: fresh = 1, each resume +1
            "outer_iteration": base_counters.get("outer_iteration", 0) + 1,
            "ppo_update_count": base_counters.get("ppo_update_count", 0) + 1,
            "optimizer_step_count": (
                base_counters.get("optimizer_step_count", 0)
                + int(metrics["update_batches"])
            ),
            "environment_transition_count": (
                base_counters.get("environment_transition_count", 0)
                + env_steps
            ),
            "completed_episode_count": (
                base_counters.get("completed_episode_count", 0)
                + completed
            ),
        }
        ckpt_path = output_dir / checkpoint_filename(
            counters["environment_transition_count"],
            counters["ppo_update_count"],
        )
        data = _build_checkpoint_data(
            policy, critic, contract, loop_config, trainer_config,
            metrics, env_steps, completed, lora_state, runtime,
            actor_optimizer, critic_optimizer, counters,
            numerical_mode_resolved,
            episode_metadata=env.current_episode_config(),
            # Post-batch snapshot — the checkpoint captures the
            # attempt/cursor state the next (resumed) run continues from.
            sampler_state=task_sampler.state_dict(),
            # R5.3: mid-episode ACTIVE slots at a budget cut.
            active_slot_states=active_slot_states,
            # R5.5: resume (lightweight) vs release (full policy).
            # R6-A: the effective kind is the post-hoc finality verdict.
            checkpoint_kind=effective_kind,
            # The training-input identity axes (P0-2): the same values
            # the next resume's resume_identity_for must reproduce.
            # base_seed comes from the sampler when it exposes one
            # (EpisodeTaskSampler); a sampler without it (scripted/test
            # double) records None — deterministic per sampler type.
            base_seed=getattr(task_sampler, "base_seed", None),
            n_envs=int(env.n_envs),
            lora_info=lora_info,
            # R6-F: the checkpoint freezes the scheduler/scaler per-batch
            # state (post-update — the committed position).
            actor_sched=actor_sched,
            critic_sched=critic_sched,
            reward_scaler=reward_scaler,
            # R7-6 Step 4 (quiescent boundary): the metrics writer cursor
            # (committed records before this batch), the observation
            # window gate state this world ran under, and the next-value
            # provenance of the update just committed — all captured
            # AFTER the update, before the atomic write.
            metrics_writer_cursor=metrics_writer_cursor or 0,
            observation_window_provenance={
                "window_mode": "latest3_gated",
                "phase": gate_summary["phase"],
                "total_frames": gate_summary["total_frames"],
                "invalid_count": gate_summary["invalid_count"],
                "passes": gate_summary["passes"],
            },
            next_value_provenance=dict(
                getattr(buffer, "next_value_provenance", {})),
        )
        CheckpointBuilder.save(data, ckpt_path)
        checkpoint_path = str(ckpt_path)
        if loop_config.eval_n_episodes > 0:
            # The eval contract is enforced at the entry preflight
            # (mode/eval_env/n_envs/dedicated instance/frozen bank) — this
            # branch only RUNS the eval, never validates it. The bank IS
            # the episode list: every entry consumed exactly once.
            records, eval_invalid = run_paired_eval(
                eval_env, bc_policy, policy, contract, loop_config,
                paired_bank,
                # review partB P1-2: the derived policy-noise seed must
                # bind the REAL bank identity, not the empty-string
                # default — otherwise E2 re-derivation/audit fails.
                bank_id=bank_id,
                # R6-D §11.5.1 (R6-EVAL-1): the evaluated policy's
                # checkpoint hash — the SAME hash the artifact embeds
                # (computed once, below, for build_runtime_eval_artifact).
                policy_checkpoint_hash=_file_sha256(ckpt_path),
            )
            # Invalid eval frames hard-fail the run — never persist a
            # paired eval polluted by invalid point clouds.
            if eval_invalid > 0:
                raise RuntimeError(
                    "eval gate: "
                    f"{eval_invalid}/{loop_config.eval_n_episodes} paired"
                    " episodes voided on invalid point-cloud frames"
                )
            if records:
                eval_artifact = build_runtime_eval_artifact(
                    records=records,
                    checkpoint_hash=_file_sha256(ckpt_path),
                    config_dict=data.exact_resolved_config,
                    contract=contract,
                    runtime=runtime,
                    lora_checksum=lora_checksum,
                    observation_invalid_count=int(gate_summary["invalid_count"]),
                    missing_weight_keys=_missing_weight_keys(policy),
                    unexpected_weight_keys=_unexpected_weight_keys(policy),
                    history_indices=list(sel_meta.history_indices),
                    offline_density=offline_density,
                    bank_id=bank_id,
                    # R6-E §11.6.1 (F-06): the policy's frozen weight-load
                    # audit — the RL policy loads the SAME registered BC
                    # checkpoint, so its unexpected keys are exactly the 7
                    # allowlisted anchors; the artifact must carry the
                    # contract that proves it.
                    load_contract=getattr(policy, "_dppo_load_contract", None),
                    # R6-EVAL-2 (§11.5.3): the frozen capability
                    # opportunity blocks ride the bank entries — the
                    # artifact's five-piece outcomes are computed from
                    # them (bank_id -> frozen bank -> re-derivable).
                    opportunity_by_entry={
                        e.episode_config_hash: e.opportunity
                        for e in paired_bank
                        if e.opportunity is not None
                    },
                )
                # Persist the paired-eval JSON next to the checkpoint under
                # the UNIQUE canonical name: env-steps / ppo-update
                # counters are cumulative and the checkpoint sha12 binds
                # the artifact to its checkpoint. Atomic tmp+rename;
                # never overwrites an existing artifact with different
                # content (hard-fail).
                eval_json_path = str(output_dir / paired_eval_artifact_name(
                    counters["environment_transition_count"],
                    counters["ppo_update_count"],
                    _file_sha256(ckpt_path)[:12],
                ))
                write_paired_eval_json(
                    eval_artifact, eval_json_path,
                    expected_contract_hash=contract_hash_of(contract),
                )
    else:
        lora_state = _live_lora_state()
        lora_checksum = lora_checksum_of(lora_state)

    return {
        "buffer": buffer,
        "transitions": env_steps,
        "env_steps": env_steps,
        "completed_episodes": completed,
        "voided_episodes": sum(voided_counts.values()),
        "rejected_episodes": rejected,
        "admission_blocked_episodes": admission_blocked,
        "routing": routing,
        "gate_summary": gate_summary,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
        "checkpoint_kind": effective_kind,  # R5.8-D sidecar truth / R6-A verdict
        "eval_artifact": eval_artifact,
        "eval_json_path": eval_json_path,  # persisted paired-eval JSON
        "lora_state": lora_state,
        "lora_checksum": lora_checksum,
        "observation_selection": asdict(sel_meta),  # history_indices
        "role_stats": role_stats,  # truthful per-role episode/transition stats
        "active_slot_states": active_slot_states,  # R5.3 budget continuation
        # R7-6 Step 6 (layer 1): load-complete state hash on the resume
        # path (None when this batch ran fresh) — the same-seed
        # comparator's resume side.
        "restored_state_hash": restored_state_hash,
    }
