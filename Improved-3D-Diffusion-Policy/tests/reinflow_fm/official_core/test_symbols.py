from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

from diffusion_policy_3d.rl.reinflow_fm.official_core.vendor_identity import (
    VendorIdentity,
)
from diffusion_policy_3d.rl.reinflow_fm.official_core.symbols import (
    OfficialImportError,
    load_official_symbols,
)


_OFFICIAL_NAMESPACES = ("model", "agent", "util", "env")


def _is_official_namespace(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in _OFFICIAL_NAMESPACES)


class OfficialSymbolTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if _is_official_namespace(name)
        }
        for name in tuple(sys.modules):
            if _is_official_namespace(name):
                sys.modules.pop(name, None)
        self._saved_path = list(sys.path)

    def tearDown(self):
        for name in tuple(sys.modules):
            if _is_official_namespace(name):
                sys.modules.pop(name, None)
        sys.modules.update(self._saved_modules)
        sys.path[:] = self._saved_path

    def _fixture_vendor(self) -> tuple[Path, tempfile.TemporaryDirectory]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "reinflow_official_e722e15"
        files = {
            "model/__init__.py": "",
            "model/flow/__init__.py": "",
            "model/flow/mlp_flow.py": "class NoisyFlowMLP:\n    pass\n",
            "model/flow/ft_ppo/__init__.py": "",
            "model/flow/ft_ppo/ppoflow.py": "class PPOFlow:\n    pass\n",
            "agent/__init__.py": "",
            "agent/finetune/__init__.py": "",
            "agent/finetune/reinflow/__init__.py": "",
            "agent/finetune/reinflow/train_ppo_flow_agent.py": (
                "class TrainPPOFlowAgent:\n    pass\n"
            ),
        }
        for relative, content in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return root, temporary

    def test_symbols_are_the_objects_defined_inside_verified_vendor(self):
        root, temporary = self._fixture_vendor()
        self.addCleanup(temporary.cleanup)
        identity = VendorIdentity(root=root, commit="fixture", manifest_sha256="0" * 64, file_count=8)

        with mock.patch(
            "diffusion_policy_3d.rl.reinflow_fm.official_core.symbols.verify_vendor_snapshot",
            return_value=identity,
        ):
            symbols = load_official_symbols(root)

        self.assertIs(symbols.identity, identity)
        for symbol in (
            symbols.PPOFlow,
            symbols.NoisyFlowMLP,
            symbols.TrainPPOFlowAgent,
        ):
            inspect_path = Path(inspect.getfile(symbol)).resolve()
            inspect_path.relative_to(root.resolve())

    def test_preloaded_external_model_module_is_rejected_before_verification(self):
        root, temporary = self._fixture_vendor()
        self.addCleanup(temporary.cleanup)
        fake = ModuleType("model")
        fake.__file__ = str(root.parent / "different_install" / "model" / "__init__.py")
        sys.modules["model"] = fake

        with self.assertRaisesRegex(OfficialImportError, "conflicting preloaded module"):
            load_official_symbols(root)

    @unittest.skipUnless(
        os.environ.get("REINFLOW_VENDOR_ROOT"),
        "REINFLOW_VENDOR_ROOT is required for the dependency-complete integration test",
    )
    def test_real_vendor_symbols_resolve_inside_pinned_tree(self):
        root = Path(os.environ["REINFLOW_VENDOR_ROOT"]).resolve()
        symbols = load_official_symbols(root)

        for symbol in (
            symbols.PPOFlow,
            symbols.NoisyFlowMLP,
            symbols.TrainPPOFlowAgent,
        ):
            Path(inspect.getfile(symbol)).resolve().relative_to(root)


if __name__ == "__main__":
    unittest.main()
