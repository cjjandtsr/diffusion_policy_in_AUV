"""Verified import boundary for the frozen official DPPO package."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import sys
from typing import Any, Iterable

from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.vendor_identity import (
    VendorIdentity,
    verify_vendor_snapshot,
)


class OfficialImportError(RuntimeError):
    """Raised when ``model`` would resolve outside the verified vendor."""


@dataclass(frozen=True)
class OfficialSymbols:
    identity: VendorIdentity
    DiffusionModel: Any
    VPGDiffusion: Any
    PPODiffusion: Any


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _module_locations(module: Any) -> Iterable[Path]:
    module_file = getattr(module, "__file__", None)
    if module_file:
        yield Path(module_file)
    module_path = getattr(module, "__path__", None)
    if module_path:
        for value in module_path:
            yield Path(value)


def _assert_no_conflicting_model_modules(vendor_root: Path) -> None:
    conflicts = []
    for name, module in tuple(sys.modules.items()):
        if name != "model" and not name.startswith("model."):
            continue
        if module is None:
            continue
        locations = tuple(_module_locations(module))
        if not locations or any(
            not _is_within(location, vendor_root) for location in locations
        ):
            conflicts.append(
                "{}={}".format(name, ",".join(str(item) for item in locations) or "unknown")
            )
    if conflicts:
        raise OfficialImportError(
            "conflicting preloaded module(s): {}".format("; ".join(conflicts))
        )


def _assert_symbol_path(symbol: Any, vendor_root: Path) -> None:
    try:
        source = Path(inspect.getfile(symbol)).resolve()
    except (TypeError, OSError) as exc:
        raise OfficialImportError(
            "cannot resolve official symbol source for {}".format(symbol)
        ) from exc
    if not _is_within(source, vendor_root):
        raise OfficialImportError(
            "official symbol resolved outside vendor: {}".format(source)
        )


def load_official_symbols(vendor_root: Path) -> OfficialSymbols:
    """Verify the source tree, isolate top-level ``model``, then import symbols."""
    root = Path(vendor_root).resolve()
    _assert_no_conflicting_model_modules(root)
    identity = verify_vendor_snapshot(root)

    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    try:
        diffusion_module = importlib.import_module("model.diffusion.diffusion")
        vpg_module = importlib.import_module("model.diffusion.diffusion_vpg")
        ppo_module = importlib.import_module("model.diffusion.diffusion_ppo")
        diffusion_model = diffusion_module.DiffusionModel
        vpg_diffusion = vpg_module.VPGDiffusion
        ppo_diffusion = ppo_module.PPODiffusion
    except Exception as exc:
        raise OfficialImportError("verified official DPPO import failed") from exc

    _assert_no_conflicting_model_modules(root)
    for symbol in (diffusion_model, vpg_diffusion, ppo_diffusion):
        _assert_symbol_path(symbol, root)
    if not issubclass(vpg_diffusion, diffusion_model):
        raise OfficialImportError("VPGDiffusion inheritance mismatch")
    if not issubclass(ppo_diffusion, vpg_diffusion):
        raise OfficialImportError("PPODiffusion inheritance mismatch")

    return OfficialSymbols(
        identity=identity,
        DiffusionModel=diffusion_model,
        VPGDiffusion=vpg_diffusion,
        PPODiffusion=ppo_diffusion,
    )
