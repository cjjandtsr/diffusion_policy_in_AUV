"""scripts.capacity_budget_p0 — P0 capacity budget audit."""

from .aggregator import (
    GATE_PASS_KEY,
    SUB_REPORT_CONTRACT,
    SUB_REPORT_MAP,
    SCHEMA_VERSION,
    run_capacity_budget_p0_audit,
)

from .artifact_identity import (
    ALLOWLISTED_RANK_DIFF_KEYS,
    sha256_file,
    run_artifact_identity_audit,
)

from .lora_scaling import (
    evaluate_scaling_gate,
    run_lora_scaling_audit,
)

from .resume_state import (
    FullStateCheckpoint,
    RoundTripHarness,
    SurrogateActor,
    FULL_STATE_FIELDS,
    _capture_rng,
    _restore_rng,
    _compare_rng_states,
    run_resume_state_audit,
)

from .eval_bank import (
    validate_bank_quota,
    build_pairing_key,
    run_eval_bank_audit,
    FORMAL_ROLES,
    PER_ROLE_QUOTA,
    MIN_WORLDS_PER_ROLE,
    MIN_EPS_PER_WORLD,
    MAX_SINGLE_WORLD_SHARE,
)

from .reward_wiring import (
    evaluate_reward_wiring_gate,
    evaluate_role_distribution_gate,
    run_reward_wiring_audit,
    run_role_distribution_audit,
)

from .multicheckpoint_schema import (
    validate_fourpoint_schema,
    aggregate_formal_overall,
    check_pairing_consistency,
    run_multicheckpoint_schema_audit,
)

__all__ = [
    "GATE_PASS_KEY",
    "SUB_REPORT_CONTRACT",
    "SUB_REPORT_MAP",
    "SCHEMA_VERSION",
    "run_capacity_budget_p0_audit",
    "ALLOWLISTED_RANK_DIFF_KEYS",
    "sha256_file",
    "run_artifact_identity_audit",
    "evaluate_scaling_gate",
    "run_lora_scaling_audit",
    "FullStateCheckpoint",
    "RoundTripHarness",
    "SurrogateActor",
    "FULL_STATE_FIELDS",
    "run_resume_state_audit",
    "validate_bank_quota",
    "build_pairing_key",
    "run_eval_bank_audit",
    "FORMAL_ROLES",
    "PER_ROLE_QUOTA",
    "MIN_WORLDS_PER_ROLE",
    "MIN_EPS_PER_WORLD",
    "MAX_SINGLE_WORLD_SHARE",
    "evaluate_reward_wiring_gate",
    "evaluate_role_distribution_gate",
    "run_reward_wiring_audit",
    "run_role_distribution_audit",
    "validate_fourpoint_schema",
    "aggregate_formal_overall",
    "check_pairing_consistency",
    "run_multicheckpoint_schema_audit",
]
