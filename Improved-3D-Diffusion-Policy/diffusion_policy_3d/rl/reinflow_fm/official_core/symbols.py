"""Fail-closed imports from the cryptographically verified ReinFlow vendor."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import sys
from typing import Any, Iterable

from .vendor_identity import VendorIdentity, verify_vendor_snapshot


_VENDOR_TOP_LEVEL_NAMESPACES = ("model", "agent", "util", "env")


class OfficialImportError(RuntimeError):
    """Raised when an official ReinFlow symbol cannot be imported safely."""


@dataclass(frozen=True)
class OfficialReinFlowSymbols:
    """Direct references to classes implemented by the official vendor."""

    identity: VendorIdentity
    PPOFlow: Any
    NoisyFlowMLP: Any
    TrainPPOFlowAgent: Any


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _module_locations(module: Any) -> Iterable[Path]:
    module_file = getattr(module, "__file__", None)
    if module_file:
        yield Path(module_file)
    module_path = getattr(module, "__path__", None)
    if module_path:
        for value in module_path:
            yield Path(value)


def _belongs_to_vendor_namespace(name: str) -> bool:
    return any(
        name == namespace or name.startswith(namespace + ".")
        for namespace in _VENDOR_TOP_LEVEL_NAMESPACES
    )


def _assert_no_conflicting_modules(vendor_root: Path) -> None:
    conflicts: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if not _belongs_to_vendor_namespace(name) or module is None:
            continue
        locations = tuple(_module_locations(module))
        if not locations or any(
            not _is_within(location, vendor_root) for location in locations
        ):
            location_text = ",".join(str(item) for item in locations) or "unknown"
            conflicts.append(f"{name}={location_text}")
    if conflicts:
        raise OfficialImportError(
            "conflicting preloaded module(s): " + "; ".join(conflicts)
        )


def _assert_symbol_path(symbol: Any, vendor_root: Path) -> None:
    try:
        source = Path(inspect.getfile(symbol)).resolve()
    except (TypeError, OSError) as error:
        raise OfficialImportError(
            f"cannot resolve official ReinFlow symbol source for {symbol!r}"
        ) from error
    if not _is_within(source, vendor_root):
        raise OfficialImportError(
            f"official ReinFlow symbol resolved outside vendor: {source}"
        )


def load_official_symbols(vendor_root: Path | str) -> OfficialReinFlowSymbols:
    """Verify the snapshot, isolate its top-level packages, and import classes.

    The returned classes are the original official objects.  This module does not
    copy, subclass, or reimplement ReinFlow's PPO or stochastic-flow equations.
    """

    root = Path(vendor_root).resolve()
    _assert_no_conflicting_modules(root)
    identity = verify_vendor_snapshot(root)

    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    try:
        ppo_module = importlib.import_module("model.flow.ft_ppo.ppoflow")
        flow_module = importlib.import_module("model.flow.mlp_flow")
        agent_module = importlib.import_module(
            "agent.finetune.reinflow.train_ppo_flow_agent"
        )
        ppo_flow = ppo_module.PPOFlow
        noisy_flow_mlp = flow_module.NoisyFlowMLP
        train_ppo_flow_agent = agent_module.TrainPPOFlowAgent
    except Exception as error:
        raise OfficialImportError(
            "verified official ReinFlow import failed; install the dependencies "
            "declared by the pinned vendor before building the training runtime"
        ) from error

    _assert_no_conflicting_modules(root)
    for symbol in (ppo_flow, noisy_flow_mlp, train_ppo_flow_agent):
        _assert_symbol_path(symbol, root)

    return OfficialReinFlowSymbols(
        identity=identity,
        PPOFlow=ppo_flow,
        NoisyFlowMLP=noisy_flow_mlp,
        TrainPPOFlowAgent=train_ppo_flow_agent,
    )


__all__ = [
    "OfficialImportError",
    "OfficialReinFlowSymbols",
    "load_official_symbols",
]
