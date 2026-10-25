"""RED #6 (R7-LORA-AUDIT): LoRA registered as a model submodule.

Current state (measured 2026-08-22): ``insert_lora`` (lora_adapters.py)
builds the adapters in a LOCAL nn.ModuleList (line 130) and attaches them
via forward-hook closures — they are invisible to
``policy.model.named_parameters()`` / ``state_dict()``. The actor
optimizer is built from the insertion report's ``lora_params`` list, so
the audit cannot see the trainable actor parameters from the model (audit
P2-1: optimizer/parameter visibility gap).

R7-LORA-AUDIT contract (guide R7-2A): the adapter bundle is registered as
a submodule of the policy model (``model.add_module("dppo_lora_adapters",
bundle)``); every LoRA parameter appears in
``model.named_parameters()`` / ``model.state_dict()``; the actor
optimizer's parameter set is exactly the bundle's trainable parameters.
Guide Step 9 adds the base/FT context-gate behavior contract
(``RegisteredLoRABundle.mode``, per-bundle contextvars gate, finally
reset, duplicate-insertion hard-fail).

These tests MUST FAIL now and go green in R7-2A.
"""

import pytest
import torch

from diffusion_policy_3d.policy.lora_adapters import insert_lora


class MiniUNet(torch.nn.Module):
    """UNet-like model carrying the R7 LoRA target patterns.

    Named modules match lora_adapters.TARGET_PATTERNS
    (down_modules/mid_modules/up_modules/final_conv) so insert_lora
    attaches adapters; tiny shapes keep this CPU-runnable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.down_modules = torch.nn.ModuleList(
            [torch.nn.Linear(64, 64), torch.nn.Linear(64, 64)]
        )
        self.mid_modules = torch.nn.ModuleList([torch.nn.Linear(64, 64)])
        self.up_modules = torch.nn.ModuleList([torch.nn.Linear(64, 64)])
        self.final_conv = torch.nn.Conv1d(64, 64, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,64) -> (B,64); Conv1d(k=1) acts per-position on (B,64,1)."""
        for m in self.down_modules:
            x = m(x)
        x = self.mid_modules[0](x)
        for m in self.up_modules:
            x = m(x)
        return self.final_conv(x.unsqueeze(-1)).squeeze(-1)


def _make_model_with_lora():
    model = MiniUNet()
    lora_info = insert_lora(model, r=4, alpha=4, verbose=False)
    return model, lora_info


def test_lora_params_registered_in_model_named_parameters() -> None:
    """Every LoRA parameter must be a parameter of the model.

    The actor optimizer is built from model parameters; an external
    module list would leave trainable weights outside the model's
    parameter view.
    """
    model, lora_info = _make_model_with_lora()
    model_param_ids = {id(p) for p in model.parameters()}
    lora_ids = {id(p) for p in lora_info["lora_params"]}
    assert lora_ids.issubset(model_param_ids), (
        f"{len(lora_ids - model_param_ids)} LoRA parameters are NOT"
        " registered on the model — the actor weights are invisible to"
        " named_parameters()/state_dict()"
    )


def test_lora_params_in_model_state_dict() -> None:
    """LoRA weights must be part of the model's state_dict.

    Checkpoint round-trip (schema v6, R7-6) saves model.state_dict();
    unregistered adapters would never be persisted.
    """
    model, _ = _make_model_with_lora()
    sd = model.state_dict()
    lora_keys = [k for k in sd if "lora_A" in k or "lora_B" in k]
    assert lora_keys, (
        "no LoRA keys in model.state_dict() — adapters are external and"
        " would be lost on checkpoint save"
    )
    # Sanity: the bundle namespace is the sanctioned container name.
    bundle_keys = [k for k in sd if k.startswith("dppo_lora_adapters.")]
    assert bundle_keys, (
        "LoRA keys found but not under the 'dppo_lora_adapters' registered"
        " submodule namespace"
    )


def test_optimizer_param_identity_matches_lora_params() -> None:
    """R7-3: the actor optimizer's parameter set is exactly the LoRA set.

    Frozen so an optimizer accidentally built from model.parameters()
    (which would include the frozen base) fails closed.
    """
    model, lora_info = _make_model_with_lora()
    optimizer = torch.optim.AdamW(lora_info["lora_params"], lr=1e-5)
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    lora_ids = {id(p) for p in lora_info["lora_params"]}
    assert opt_ids == lora_ids, (
        "optimizer parameter set != LoRA parameter set — actor optimizer"
        " must own exactly the trainable adapters"
    )
    # Base parameters stay frozen at autograd level (R7-2A base view).
    base_grad_ok = any(
        not p.requires_grad and "lora" not in n
        for n, p in model.named_parameters()
    )
    assert base_grad_ok, "no frozen base parameter found — base must be frozen"


# ── Guide R7-2A Step 1-2 / Step 9 behavior contract ──────────────────

def test_lora_modules_are_bundle_adapters() -> None:
    """insert_lora report must reference the registered bundle, not a
    second private ModuleList (guide Step 1: single source of truth)."""
    model, lora_info = _make_model_with_lora()
    bundle = model.dppo_lora_adapters
    assert lora_info["lora_modules"] is bundle.adapters, (
        "lora_modules must be the registered bundle's adapters"
    )
    assert {id(p) for p in lora_info["lora_params"]} == {
        id(p) for p in bundle.trainable_parameters()
    }, "lora_params must come from the bundle, not a second object list"


def test_duplicate_insertion_raises() -> None:
    """insert_lora on an already-bundled model must hard-fail (Step 1)."""
    model, _ = _make_model_with_lora()
    with pytest.raises(RuntimeError, match="duplicate insertion"):
        insert_lora(model, r=4, alpha=4, verbose=False)


def test_lora_b_zero_init_base_equals_ft() -> None:
    """LoRA B=0 init ⇒ FT output identical to base output (≤1e-7)."""
    model, _ = _make_model_with_lora()
    bundle = model.dppo_lora_adapters
    x = torch.randn(2, 64)
    with bundle.mode(False):
        y_base = model(x)
    with bundle.mode(True):
        y_ft = model(x)
    assert torch.max(torch.abs(y_base - y_ft)).item() <= 1e-7, (
        "base/FT outputs must match under B=0 initialization"
    )


def test_nonzero_b_changes_ft_only() -> None:
    """A nonzero LoRA B must change FT output only; base stays exact."""
    model, _ = _make_model_with_lora()
    bundle = model.dppo_lora_adapters
    x = torch.randn(2, 64)
    with bundle.mode(False):
        y_base_before = model(x)
    bundle.adapters[0].lora_B.data.fill_(0.1)
    with bundle.mode(False):
        y_base_after = model(x)
    with bundle.mode(True):
        y_ft = model(x)
    assert torch.equal(y_base_before, y_base_after), (
        "base output must be unchanged when the gate is off"
    )
    assert not torch.allclose(y_base_after, y_ft, atol=1e-6), (
        "FT output must change when a LoRA B is nonzero"
    )


def test_gate_restores_after_exception() -> None:
    """Context exit via exception must still reset the gate (finally)."""
    model, _ = _make_model_with_lora()
    bundle = model.dppo_lora_adapters
    assert bundle.is_enabled()
    with pytest.raises(RuntimeError):
        with bundle.mode(False):
            assert not bundle.is_enabled()
            raise RuntimeError("boom")
    assert bundle.is_enabled(), "gate must be restored after exception exit"


def test_two_bundles_gates_independent() -> None:
    """Each bundle owns its gate; toggling one leaves the other on."""
    model1, _ = _make_model_with_lora()
    model2, _ = _make_model_with_lora()
    b1 = model1.dppo_lora_adapters
    b2 = model2.dppo_lora_adapters
    with b1.mode(False):
        assert not b1.is_enabled()
        assert b2.is_enabled(), "bundle-2 gate must not be affected by bundle-1"
    assert b1.is_enabled()
    assert b2.is_enabled()


def test_state_dict_roundtrip_preserves_ft_output() -> None:
    """state_dict must carry LoRA weights; a reloaded model reproduces
    the FT output (Step 9, checkpoint schema v6 round-trip)."""
    model, _ = _make_model_with_lora()
    bundle = model.dppo_lora_adapters
    bundle.adapters[0].lora_B.data.fill_(0.05)
    x = torch.randn(2, 64)
    with bundle.mode(True):
        y_ft = model(x)
    sd = model.state_dict()
    assert any(k.startswith("dppo_lora_adapters.") for k in sd)
    model2 = MiniUNet()
    insert_lora(model2, r=4, alpha=4, verbose=False)
    model2.load_state_dict(sd)
    with model2.dppo_lora_adapters.mode(True):
        y_ft2 = model2(x)
    assert torch.allclose(y_ft, y_ft2, atol=1e-7), (
        "FT output must survive a state_dict round-trip"
    )


def test_bundle_trainable_base_frozen_autograd() -> None:
    """Step 3: bundle params trainable; base params autograd-frozen."""
    model, _ = _make_model_with_lora()
    bundle = model.dppo_lora_adapters
    trainable = bundle.trainable_parameters()
    assert trainable, "bundle must expose its trainable parameters"
    assert all(p.requires_grad for p in trainable), (
        "all LoRA params must be requires_grad=True"
    )
    frozen = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("dppo_lora_adapters.")
    ]
    assert frozen, "expected frozen base parameters"
    assert all(not p.requires_grad for p in frozen), (
        "all base model params must be requires_grad=False"
    )
