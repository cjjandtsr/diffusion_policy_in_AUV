"""RED #1 (R7-OFFICIAL-DIRECT): production must execute the frozen vendor.

Current state (measured 2026-08-22):
- production imports a SECOND training formula: trainer.py:68
  ``official_loss_adapter``, chain_logprobs.py:19 and dp_sampler.py:23
  ``dppo_transition_kernel``;
- no production module imports the frozen official runtime
  (``diffusion_policy_3d.rl.dppo_uav.vendor_loader`` does not exist).

These tests freeze the R7 target contract and MUST FAIL now; R7-1 makes
them green by adding the offline vendor loader, R7-2/R7-3 route the
production loss/logprob/transition paths through the vendor facade and
delete the second-formula production references.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The canonical production call stack of the R7 architecture.
PRODUCTION_FILES = [
    "diffusion_policy_3d/rl/dppo_uav/trainer.py",
    "diffusion_policy_3d/rl/dppo_uav/chain_logprobs.py",
    "diffusion_policy_3d/rl/dppo_uav/phase1a_actor_adapter.py",
    "diffusion_policy_3d/policy/dp_sampler.py",
    "diffusion_policy_3d/rl/dppo_uav/official_runtime.py",
]

# The local re-implementations of official PPO/DDIM math — forbidden in
# the production import graph (guide file map: "删除生产引用但保留历史/测试").
SECOND_FORMULA_MODULES = ("official_loss_adapter", "dppo_transition_kernel")


def _imported_module_names(path: Path) -> set:
    """Top-level imported module names declared in the file (AST, no exec)."""
    tree = ast.parse(path.read_text())
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_vendor_loader_importable() -> None:
    """The offline vendor loader must exist and import offline."""
    import importlib

    try:
        importlib.import_module("diffusion_policy_3d.rl.dppo_uav.vendor_loader")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "vendor_loader missing — production cannot reach the frozen"
            f" official runtime: {exc}"
        )


def test_production_declares_vendor_usage() -> None:
    """Production source must declare the vendor loader in its import graph.

    R7-OFFICIAL-DIRECT: the canonical production stack actually enters
    the frozen vendor; ``vendor_loader`` is the single sanctioned entry.
    """
    for rel in PRODUCTION_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            pytest.fail(f"production file missing: {rel}")
        imported = _imported_module_names(path)
        vendor_visible = any(
            name == "vendor_loader" or name.endswith(".vendor_loader")
            for name in imported
        )
        assert vendor_visible, (
            f"{rel} does not import the frozen vendor loader — production"
            " never reaches the official runtime"
        )


def test_no_second_training_formula_in_production() -> None:
    """No local re-implementation may be reachable from production.

    Guide file map: official_loss_adapter and dppo_transition_kernel are
    removed from the production import graph (test/parity helpers only).
    """
    for rel in PRODUCTION_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            pytest.fail(f"production file missing: {rel}")
        imported = _imported_module_names(path)
        for forbidden in SECOND_FORMULA_MODULES:
            assert forbidden not in imported, (
                f"{rel} imports second training formula {forbidden!r}"
            )
