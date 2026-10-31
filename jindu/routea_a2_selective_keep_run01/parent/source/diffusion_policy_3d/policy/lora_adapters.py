"""
lora_adapters.py — LoRA low-rank adaptation for ConditionalUNet1D.

Inserts LoRA adapters into target UNet layers via forward hooks.
Only LoRA parameters are trainable; original UNet weights are frozen.

Design: audit of routea_reinflow_fm_policy.py with fixes applied.
R7-2A: adapters live in a RegisteredLoRABundle registered as the
``dppo_lora_adapters`` submodule of the model; a per-bundle
contextvars gate toggles base (mode False) vs FT (mode True) outputs.
"""

import contextvars
import torch
import torch.nn as nn
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

# ── Fixed constants ──
DEFAULT_LORA_RANK = 4
DEFAULT_LORA_ALPHA = 4  # alpha/r = 1 (was 8/4=2)

# Target layer patterns: all UNet blocks + residual convs + final output
TARGET_PATTERNS = [
    "down_modules",    # encoder blocks
    "mid_modules",     # bottleneck blocks
    "up_modules",      # decoder blocks
    "final_conv",      # output projection
]

# FIX P0-2: Remove cond_encoder exclusion — DPPO official uses full fine-tuning.
# KL early-stop + per-step epsilon schedule prevent BC drift (safer than freezing FiLM).
EXCLUDE_PATTERNS = [
]

# Registered-bundle attribute name on the host model (guide R7-2A Step 1).
BUNDLE_ATTR = "dppo_lora_adapters"


def _is_target(name: str) -> bool:
    """Check if a module name should receive a LoRA adapter."""
    if any(p in name for p in EXCLUDE_PATTERNS):
        return False
    return any(p in name for p in TARGET_PATTERNS)


# ═══════════════════════════════════════════════════════════════════
# LoRA Adapter Modules
# ═══════════════════════════════════════════════════════════════════

class LoRALinearAdaptor(nn.Module):
    """LoRA for nn.Linear: y = Wx + b + (alpha/r) * x @ A @ B.

    A: Kaiming uniform init. B: zero init.
    Original weight/bias are frozen.
    """

    def __init__(self, original: nn.Linear, r: int = 4, alpha: int = 4):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.empty(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRAConv1dAdaptor(nn.Module):
    """LoRA for nn.Conv1d(kernel_size=1): per-position linear projection.

    Conv1d(k=1) is equivalent to Linear applied channel-wise across T.
    LoRA: y += (alpha/r) * x @ A @ B, applied per position.
    """

    def __init__(self, original: nn.Conv1d, r: int = 4, alpha: int = 4):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)

        in_channels = original.in_channels
        out_channels = original.out_channels

        self.lora_A = nn.Parameter(torch.empty(in_channels, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_channels))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T) → LoRA: (B, C_out, T)."""
        x_perm = x.permute(0, 2, 1)                    # (B, T, C_in)
        out = (x_perm @ self.lora_A @ self.lora_B) * self.scaling  # (B, T, C_out)
        return out.permute(0, 2, 1)                    # (B, C_out, T)


# ═══════════════════════════════════════════════════════════════════
# Registered LoRA Bundle (guide R7-2A Step 1-2)
# ═══════════════════════════════════════════════════════════════════

class RegisteredLoRABundle(nn.Module):
    """Registered adapter bundle with a per-bundle base/FT context gate.

    Registered on the host model as ``dppo_lora_adapters`` so every LoRA
    parameter is visible to ``named_parameters()`` / ``state_dict()``.
    Each bundle owns an independent ``contextvars.ContextVar`` gate
    (default ``True`` — the historical always-active hook behavior for
    non-facade callers); ``mode(False)`` makes every hook return the base
    output unchanged. The token is always reset in ``finally`` so an
    exception inside the context cannot leak the gate.
    """

    def __init__(self, adapters: List[nn.Module]) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(adapters)
        self._gate: contextvars.ContextVar[bool] = contextvars.ContextVar(
            f"lora_gate_{id(self)}", default=True
        )

    @contextmanager
    def mode(self, enabled: bool) -> Iterator[None]:
        """Context-manage the gate; always restores the previous value.

        Args:
            enabled: ``False`` = base pass (hooks return base output),
                ``True`` = FT pass (LoRA contribution added).
        """
        token = self._gate.set(enabled)
        try:
            yield
        finally:
            self._gate.reset(token)

    def is_enabled(self) -> bool:
        """Current gate value — ``True`` means LoRA contributions apply."""
        return self._gate.get()

    def trainable_parameters(self) -> Tuple[nn.Parameter, ...]:
        """All LoRA parameters across the bundle, deduplicated."""
        params: List[nn.Parameter] = []
        for adaptor in self.adapters:
            for p in adaptor.parameters():
                params.append(p)
        return tuple(params)

    def named_lora_state(self) -> OrderedDict[str, torch.Tensor]:
        """LoRA parameters keyed ``<adapter_index>.<param_name>``.

        Single source of truth for optimizer/checkpoint bookkeeping of
        the bundle's own weights.
        """
        state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for i, adaptor in enumerate(self.adapters):
            for name, p in adaptor.named_parameters():
                state[f"{i}.{name}"] = p
        return state


# ═══════════════════════════════════════════════════════════════════
# LoRA Insertion
# ═══════════════════════════════════════════════════════════════════

def _make_lora_hook(adaptor: nn.Module, bundle: RegisteredLoRABundle):
    """Create forward hook that adds LoRA contribution to module output.

    The hook is gated by the bundle's context gate: ``mode(False)``
    (base pass) returns the module output unchanged.
    """
    def hook(module, inputs, output):
        if not bundle.is_enabled():
            return output
        return output + adaptor(inputs[0])
    return hook


def insert_lora(
    model: nn.Module,
    r: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    verbose: bool = True,
) -> Dict:
    """Insert LoRA adapters into a ConditionalUNet1D model.

    Traverses model.named_modules(), inserts LoRA on target Linear/Conv1d
    layers, registers the adapter bundle as the ``dppo_lora_adapters``
    submodule (guide R7-2A Step 1). Returns a dict with lora_params (for
    optimizer) and insertion report.

    Args:
        model: ConditionalUNet1D instance (e.g., policy.model)
        r: LoRA rank
        alpha: LoRA alpha (alpha/r = scaling factor)
        verbose: print insertion summary

    Returns:
        {
            "lora_params": list of nn.Parameter (for optimizer),
            "lora_modules": nn.ModuleList (all adaptor modules),
            "hooks": list of hook handles,
            "report": dict with insertion metadata,
        }

    Raises:
        RuntimeError: the model already carries a ``dppo_lora_adapters``
            bundle — duplicate insertion is forbidden (fail-closed, no
            partial mutation).
    """
    if hasattr(model, BUNDLE_ATTR):
        raise RuntimeError(
            "LoRA already inserted; duplicate insertion forbidden"
        )

    targets: List[Tuple[nn.Module, nn.Module]] = []  # (target, adaptor)
    inserted: List[dict] = []

    for name, module in model.named_modules():
        if not _is_target(name):
            continue

        adaptor = None
        if isinstance(module, nn.Linear):
            adaptor = LoRALinearAdaptor(module, r=r, alpha=alpha)
        # FIX P0-2: Adapt Conv1d with stride=1 (k=5 Conv1dBlocks included).
        # Skip stride>1 (Downsample1d) to avoid spatial dimension mismatch.
        elif isinstance(module, nn.Conv1d) and module.stride[0] == 1:
            adaptor = LoRAConv1dAdaptor(module, r=r, alpha=alpha)

        if adaptor is not None:
            targets.append((module, adaptor))
            inserted.append({
                "name": name,
                "type": type(module).__name__,
                "adaptor": type(adaptor).__name__,
                "weight_shape": list(module.weight.shape),
                "lora_params": sum(p.numel() for p in adaptor.parameters()),
            })

    # Register the bundle as a model submodule BEFORE wiring hooks, so
    # the hooks gate on the single registered bundle instance.
    bundle = RegisteredLoRABundle([a for _, a in targets])

    # Move LoRA modules to model device and rebuild param list
    device = next(model.parameters()).device
    bundle.to(device)

    hooks: List = []
    for (module, adaptor) in targets:
        hook = module.register_forward_hook(_make_lora_hook(adaptor, bundle))
        hooks.append(hook)

    model.add_module(BUNDLE_ATTR, bundle)

    # lora_modules/lora_params MUST come from the bundle — no second
    # object list (guide R7-2A Step 1 single source of truth).
    lora_modules = bundle.adapters
    lora_params = list(bundle.trainable_parameters())

    # R5.2 (review F2 / §4.2): freeze the base model at the AUTOGRAD
    # level — the official DPPO freezes the base actor explicitly
    # (diffusion_vpg.py:82-85). Merely excluding base weights from the
    # optimizer left them requires_grad=True, so every actor backward
    # still computed and accumulated base gradients (wasted memory and
    # a long-training NaN/Inf hazard). LoRA adapters stay trainable.
    lora_ids = {id(p) for p in lora_params}
    n_base_frozen = 0
    for name, param in model.named_parameters():
        if id(param) in lora_ids:
            param.requires_grad_(True)
        else:
            if param.requires_grad:
                n_base_frozen += 1
            param.requires_grad_(False)

    total_lora = sum(p.numel() for p in lora_params)
    total_base = sum(p.numel() for p in model.parameters())
    report = {
        "lora_rank": r,
        "lora_alpha": alpha,
        "scaling": alpha / max(r, 1),
        "n_adaptors": len(inserted),
        "total_lora_params": total_lora,
        "total_base_params": total_base,
        "base_params_frozen": n_base_frozen,
        "trainable_params": len(lora_params),
        "ratio": f"{100 * total_lora / max(total_base, 1):.2f}%",
        "inserted": inserted,
    }

    if verbose:
        print(f"[LORA] rank={r}, alpha={alpha}, scaling={alpha/max(r,1):.1f}")
        print(f"[LORA] Inserted {len(inserted)} adaptors, {total_lora:,} params "
              f"({100*total_lora/max(total_base,1):.2f}% of {total_base:,} base)")
        for item in inserted:
            print(f"  {item['name']:55s} {item['type']:10s} "
                  f"{str(item['weight_shape']):20s}")

    return {
        "lora_params": lora_params,
        "lora_modules": lora_modules,
        "hooks": hooks,
        "report": report,
    }
