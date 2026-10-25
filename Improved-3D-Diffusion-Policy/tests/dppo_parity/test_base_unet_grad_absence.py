"""O1-O4: base UNet must be frozen at the autograd level, not merely
excluded from the optimizer (review R5.2 / §4.2, §8.2).

The official DPPO freezes the base actor explicitly
(diffusion_vpg.py:82-85); the project previously relied only on the
optimizer's LoRA-only parameter set, so base gradients were still
computed and accumulated on every backward — wasted memory/compute and
a hidden long-training hazard.

Gates:
  O1 base .grad is None after an actor backward;
  O2 LoRA grads finite and non-zero;
  O3 optimizer parameter IDs exactly equal the LoRA parameter IDs;
  O4 base weights bitwise unchanged + LoRA changed after one step.
"""

import hashlib

import torch

from diffusion_policy_3d.policy.lora_adapters import insert_lora

B, R, Da = 1, 8, 11


class _UNetLike(torch.nn.Module):
    """UNet-style named module so the production insert_lora target
    filter applies (TARGET_PATTERNS: down/mid/up_modules, final_conv)."""

    def __init__(self) -> None:
        super().__init__()
        self.down_modules = torch.nn.ModuleList([
            torch.nn.Linear(R * Da, 64),
        ])
        self.mid_modules = torch.nn.ModuleList([
            torch.nn.Linear(64, 64),
        ])
        self.up_modules = torch.nn.ModuleList([
            torch.nn.Linear(64, 64),
        ])
        self.final_conv = torch.nn.Linear(64, R * Da)

    def forward(self, sample, timestep=None, **kwargs):
        h = torch.relu(self.down_modules[0](sample.reshape(B, -1)))
        h = torch.relu(self.mid_modules[0](h))
        h = torch.relu(self.up_modules[0](h))
        return self.final_conv(h).reshape(B, R, Da)


def _param_hash(params) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_base_frozen_and_lora_trainable():
    """O1/O2 prerequisites: after insert_lora, every non-LoRA parameter
    is requires_grad=False and every LoRA parameter is True."""
    model = _UNetLike()
    info = insert_lora(model, r=4, alpha=4, verbose=False)
    lora_ids = {id(p) for p in info["lora_params"]}
    assert lora_ids, "insert_lora produced no adapters"
    for name, p in model.named_parameters():
        if id(p) in lora_ids:
            assert p.requires_grad, f"LoRA param {name} frozen"
        else:
            assert not p.requires_grad, (
                f"base param {name} not frozen (O1 precondition)"
            )


def test_base_grad_absent_after_backward():
    """O1 + O2: one actor backward leaves every base .grad None and
    every LoRA grad finite/non-zero."""
    torch.manual_seed(0)
    model = _UNetLike()
    info = insert_lora(model, r=4, alpha=4, verbose=False)
    lora_ids = {id(p) for p in info["lora_params"]}
    x = torch.randn(B, R, Da)
    eps = model(x)
    target = torch.randn_like(eps)
    loss = ((eps - target) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        if id(p) in lora_ids:
            assert p.grad is not None, f"LoRA {name}: no grad (O2)"
            assert torch.isfinite(p.grad).all(), f"LoRA {name}: non-finite"
        else:
            assert p.grad is None, (
                f"base param {name}: .grad is {p.grad} — must be None (O1)"
            )
    lora_norm = sum(
        float((p.grad.detach() ** 2).sum()) for p in info["lora_params"]
    ) ** 0.5
    assert lora_norm > 0.0, "aggregate LoRA grad norm must be > 0 (O2)"


def test_optimizer_param_set_equals_lora():
    """O3: the actor optimizer built from lora_info must cover EXACTLY
    the LoRA parameter set (no base param, no missing adapter)."""
    model = _UNetLike()
    info = insert_lora(model, r=4, alpha=4, verbose=False)
    optimizer = torch.optim.Adam(info["lora_params"], lr=1e-3)
    opt_ids = {id(p) for group in optimizer.param_groups
               for p in group["params"]}
    lora_ids = {id(p) for p in info["lora_params"]}
    assert opt_ids == lora_ids, (
        f"optimizer params ({len(opt_ids)}) != LoRA params ({len(lora_ids)})"
    )


def test_one_step_moves_only_lora():
    """O4: after one optimizer step the base weights are bitwise
    unchanged and the LoRA checksum changes."""
    torch.manual_seed(1)
    model = _UNetLike()
    info = insert_lora(model, r=4, alpha=4, verbose=False)
    lora_ids = {id(p) for p in info["lora_params"]}
    base_params = [p for n, p in model.named_parameters()
                   if id(p) not in lora_ids]
    base_before = _param_hash(base_params)
    lora_before = _param_hash(info["lora_params"])
    optimizer = torch.optim.Adam(info["lora_params"], lr=1e-3)
    x = torch.randn(B, R, Da)
    target = torch.randn_like(model(x))
    optimizer.zero_grad()
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    assert _param_hash(base_params) == base_before, (
        "base weights moved (O4)"
    )
    assert _param_hash(info["lora_params"]) != lora_before, (
        "LoRA did not move (O4)"
    )
