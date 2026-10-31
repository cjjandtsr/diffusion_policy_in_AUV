"""Formal numerical mode: deterministic, TF32-free computation.

The rollout stores old logprobs; ppo_update recomputes the likelihood
in one actual-minibatch call (B=128, the same batch size the rollout
behavior used per step — guide R7-3: single facade.loss forward per
minibatch, no B=1 microbatch loop). The stored logprobs reproduce
exactly under the formal numerical mode because the recompute batch
equals the behavior batch (G5/R7-B128 identity: max_abs_diff <= 1e-5 at
the actual minibatch). The measured batch-size kernel dependence
(B=128 vs B=1 on identical inputs: mean 4.9e-4, max 0.129 pre-fix —
test_actual_batch_logprob_identity.py) came from TF32 cuDNN kernels and
non-deterministic algorithm selection; the formal mode (TF32 off +
deterministic algorithms) is what makes the actual-batch identity hold.

Configuration is single-sourced: the production entry
(entrypoint_guard.validate_numerical_mode) applies the mode in
preflight, before policy load / LoRA insertion / optimizer
construction; run_e2e_batch asserts that it is already configured
(assert_numerical_mode_configured) instead of applying a second
configuration source. Resolved values are recorded in the checkpoint
exact config and gate reports so the evidence is reproducible (G5 §5.7:
max_abs_diff <= 1e-5 at the actual minibatch).
"""

import os
from dataclasses import dataclass
from typing import Dict

import torch

# PyTorch requires CUBLAS_WORKSPACE_CONFIG to be present before the first
# cuBLAS handle creation for deterministic algorithms to work on CUDA >=
# 10.2 (RuntimeError otherwise: "Deterministic behavior was enabled ...").
# setdefault at import time (earliest load point, train_loop imports this
# module) and again in configure_numerical_mode for direct callers; cuBLAS
# reads it lazily at handle creation, so setting it before the first CUDA
# matmul is sufficient.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class NumericalMode:
    """Formal training/eval numerical settings.

    The formal defaults are TF32 off + deterministic algorithms on; a
    different mode is only ever used for diagnostics (never to relax a
    gate).
    """

    allow_tf32_matmul: bool = False
    allow_tf32_cudnn: bool = False
    deterministic_algorithms: bool = True


def configure_numerical_mode(mode: NumericalMode) -> Dict[str, bool]:
    """Apply the numerical mode and return the resolved values.

    Torch backends are process-global, so the formal entry calls this
    once before loading the policy, building optimizers or producing
    rollouts (idempotent). The returned dict goes into the checkpoint
    exact config and the gate report.

    R7-6 Step 3: the deterministic runtime is the full four-axis set —
    CUBLAS_WORKSPACE_CONFIG=:4096:8 (cublas determinism, read at handle
    creation), use_deterministic_algorithms (any unsupported
    nondeterministic op hard-fails with RuntimeError — the gate is never
    switched off), cudnn.benchmark=False + cudnn.deterministic=True
    (deterministic kernel selection). TF32 stays pinned to the R1
    values. The resolved dict records the CUBLAS_WORKSPACE_CONFIG value
    in effect so the checkpoint evidence pins the launcher environment
    that actually ran.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = mode.allow_tf32_matmul
    torch.backends.cudnn.allow_tf32 = mode.allow_tf32_cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode.deterministic_algorithms)
    return {
        "allow_tf32_matmul": bool(mode.allow_tf32_matmul),
        "allow_tf32_cudnn": bool(mode.allow_tf32_cudnn),
        "deterministic_algorithms": bool(mode.deterministic_algorithms),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ""),
    }


def assert_numerical_mode_configured(
    mode: NumericalMode = NumericalMode(),
) -> Dict[str, bool]:
    """Assert the process-global numerical mode matches the expected mode.

    Single configuration source (guide R7-3 Step 1): the production
    entry applies the formal mode in preflight, before policy load /
    LoRA insertion / optimizer construction
    (entrypoint_guard.validate_numerical_mode); run_e2e_batch and test
    fixtures must NOT apply a second configuration source. Callers that
    bypass preflight (tests and diagnostic scripts invoking
    run_e2e_batch directly) must call configure_numerical_mode
    themselves beforehand — the assert then fails closed with a
    RuntimeError naming the mismatch.

    Returns the resolved dict (same shape as configure_numerical_mode)
    so callers can record it in the checkpoint exact config and gate
    reports. ``mode`` defaults to the formal mode (NumericalMode() =
    TF32 off + deterministic); the frozen dataclass default is safe as
    an immutable argument.
    """
    got = {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ""),
    }
    expected = {
        "allow_tf32_matmul": bool(mode.allow_tf32_matmul),
        "allow_tf32_cudnn": bool(mode.allow_tf32_cudnn),
        "deterministic_algorithms": bool(mode.deterministic_algorithms),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
    }
    if got != expected:
        raise RuntimeError(
            "numerical mode not configured: expected "
            f"{expected!r}, got {got!r} — the formal entry configures "
            "the numerical mode exactly once in preflight "
            "(entrypoint_guard.validate_numerical_mode); callers of "
            "run_e2e_batch must configure before calling (single "
            "configuration source, guide R7-3 Step 1)"
        )
    return got
