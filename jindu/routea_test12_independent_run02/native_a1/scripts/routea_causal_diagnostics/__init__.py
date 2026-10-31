"""Read-only Route A causal diagnostic support.

This package is deliberately outside the production checkpoint source closure.
Its own source manifest is mandatory for every diagnostic output directory.
"""

from .contracts import (
    DIAGNOSTIC_SCHEMA_VERSION,
    DiagnosticFailure,
    build_diagnostic_source_manifest,
    canonical_array_sha256,
    canonical_tree_sha256,
    capture_rng_state,
    load_numeric_npz,
    load_strict_json,
    restore_rng_state,
    rng_state_sha256,
    write_numeric_npz,
    write_strict_json,
)
from .goals import GOAL_VARIANTS, make_goal_variant, reconstruct_goal_history
from .runtime import (
    CheckpointContract,
    ForwardRecord,
    compare_checkpoint_contracts,
    paired_forward,
    verify_forward_pairing,
)

__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DiagnosticFailure",
    "build_diagnostic_source_manifest",
    "canonical_array_sha256",
    "canonical_tree_sha256",
    "capture_rng_state",
    "load_numeric_npz",
    "load_strict_json",
    "restore_rng_state",
    "rng_state_sha256",
    "write_numeric_npz",
    "write_strict_json",
    "GOAL_VARIANTS",
    "make_goal_variant",
    "reconstruct_goal_history",
    "CheckpointContract",
    "ForwardRecord",
    "compare_checkpoint_contracts",
    "paired_forward",
    "verify_forward_pairing",
]
