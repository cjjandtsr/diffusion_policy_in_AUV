import inspect
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.symbols import (
    OfficialImportError,
    load_official_symbols,
)


def _real_vendor_root():
    value = os.environ.get("DPPO_VENDOR_ROOT")
    if not value:
        pytest.skip("DPPO_VENDOR_ROOT is required for the real-vendor import test")
    return Path(value)


def test_official_symbols_resolve_inside_the_verified_vendor():
    root = _real_vendor_root().resolve()
    symbols = load_official_symbols(root)

    assert symbols.identity.root == root
    for symbol in (
        symbols.DiffusionModel,
        symbols.VPGDiffusion,
        symbols.PPODiffusion,
    ):
        Path(inspect.getfile(symbol)).resolve().relative_to(root)


def test_official_import_rejects_a_preloaded_conflicting_top_level_model(
    tmp_path, monkeypatch
):
    fake = ModuleType("model")
    fake.__file__ = str(tmp_path / "different_install" / "model" / "__init__.py")
    monkeypatch.setitem(sys.modules, "model", fake)

    with pytest.raises(OfficialImportError, match="conflicting preloaded module"):
        load_official_symbols(tmp_path / "vendor")
