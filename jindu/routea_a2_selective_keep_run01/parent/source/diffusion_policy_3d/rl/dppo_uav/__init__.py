"""A-Official-Core RL system package (design RouteA_RL_Training_System_A_OfficialCore v1.0).

Single source of truth for system contracts; adapters map the frozen E040-era
modules (phase1a policy / kinematic env / reward) into the official DPPO core.
"""

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.observation_adapter import (
    ObservationSelection,
    select_latest_observations,
)
from diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter import (
    DPPORolloutSample,
    denoising_indices,
    make_get_new_logprobs,
    sample_dppo_rollout,
    slice_executed_logprobs,
    split_trajectory,
    to_rollout_transition,
)
from diffusion_policy_3d.rl.dppo_uav.chain_logprobs import (
    per_dim_denoising_logprobs,
    per_step_denoising_logp,
    selected_step_logprobs,
)
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.pointcloud_gate import (
    InvalidFramePolicy,
    PHASE_GATES,
    PointCloudFrame,
    PointCloudValidityGate,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    AdmissionChecker,
    AdmissionExhausted,
    AdmissionResult,
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSpec,
    ROLE_ADMISSION_PARAMS,
    RoleSampler,
    StepBatch,
    TaskSpecRegistry,
    UAVVectorEnvAdapter,
)
from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
    RolloutBuffer,
    RolloutTransition,
)
from diffusion_policy_3d.rl.dppo_uav.gae import compute_gae
from diffusion_policy_3d.rl.dppo_uav.official_loss_adapter import (
    DPPOLossOutput,
    DPPOValueLossOutput,
    dppo_policy_loss,
    dppo_value_loss,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import (
    TrainerConfig,
    ppo_update,
)
from diffusion_policy_3d.rl.dppo_uav.numerics import (
    NumericalMode,
    configure_numerical_mode,
)
from diffusion_policy_3d.rl.dppo_uav.planner import (
    OccGridMeta,
    astar_3d,
    build_occupancy_grid,
    grid_to_world,
    world_to_grid,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_train_pool,
    build_task_specs,
    build_world_pool,
    entry_metadata_hash,
    manifest_train_entries,
)
from diffusion_policy_3d.rl.dppo_uav.gate_audit import (
    proportion_gate,
    run_reset_gate_audit,
)
from diffusion_policy_3d.rl.dppo_uav.metrics import (
    explained_variance,
    ratio_statistics,
    success_collision_role_conditioned_advantages,
    tensor_mean_std,
)
from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    SCHEMA_VERSION,
    CheckpointBuilder,
    CheckpointData,
    CheckpointError,
    checkpoint_filename,
    contract_to_checkpoint_bytes,
    contract_hash_of,
)
from diffusion_policy_3d.rl.dppo_uav.policy_loader import (
    extract_lora_state,
    load_ddpm_policy,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    LoopConfig,
    build_runtime_eval_artifact,
    lora_checksum_of,
    run_e2e_batch,
    run_paired_eval,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    CAPABILITY_METRICS,
    EpisodePairedRecord,
    EvalArtifact,
    EvalArtifactError,
    build_eval_artifact,
    validate_eval_artifact,
)

__all__ = [
    "RLSystemContract",
    "ObservationSelection",
    "select_latest_observations",
    "DPPORolloutSample",
    "make_get_new_logprobs",
    "sample_dppo_rollout",
    "to_rollout_transition",
    "per_dim_denoising_logprobs",
    "per_step_denoising_logp",
    "selected_step_logprobs",
    "PolicyObsEncoder",
    "ValueCritic",
    "split_trajectory",
    "slice_executed_logprobs",
    "denoising_indices",
    "InvalidFramePolicy",
    "PHASE_GATES",
    "PointCloudFrame",
    "PointCloudValidityGate",
    "AdmissionChecker",
    "AdmissionExhausted",
    "AdmissionResult",
    "DEFAULT_ROLE_TARGETS",
    "EpisodeTaskSpec",
    "ROLE_ADMISSION_PARAMS",
    "RoleSampler",
    "StepBatch",
    "TaskSpecRegistry",
    "UAVVectorEnvAdapter",
    "RolloutBuffer",
    "RolloutTransition",
    "compute_gae",
    "DPPOLossOutput",
    "dppo_policy_loss",
    "DPPOValueLossOutput",
    "dppo_value_loss",
    "TrainerConfig",
    "ppo_update",
    "NumericalMode",
    "configure_numerical_mode",
    "OccGridMeta",
    "astar_3d",
    "build_occupancy_grid",
    "grid_to_world",
    "world_to_grid",
    "build_adapter",
    "build_task_specs",
    "build_world_pool",
    "manifest_train_entries",
    "explained_variance",
    "ratio_statistics",
    "success_collision_role_conditioned_advantages",
    "tensor_mean_std",
    "SCHEMA_VERSION",
    "CheckpointBuilder",
    "CheckpointData",
    "CheckpointError",
    "checkpoint_filename",
    "contract_to_checkpoint_bytes",
    "contract_hash_of",
    "extract_lora_state",
    "load_ddpm_policy",
    "LoopConfig",
    "build_runtime_eval_artifact",
    "lora_checksum_of",
    "run_e2e_batch",
    "run_paired_eval",
    "CAPABILITY_METRICS",
    "EpisodePairedRecord",
    "EvalArtifact",
    "EvalArtifactError",
    "build_eval_artifact",
    "validate_eval_artifact",
]
