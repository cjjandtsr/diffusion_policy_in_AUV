"""R7-1: offline vendor loader provenance tests.

Positive: the REAL frozen vendor verifies 401/401 with the pinned commit,
and ``load_official_dppo_symbols`` works with every network/shell channel
monkeypatched to explode (call count 0).

Negative (tmp_path mini-vendor, real vendor NEVER modified): single-byte
tamper, missing file, ``../`` path escape and wrong commit each hard-fail
via VendorVerificationError.
"""

import hashlib
import importlib
import inspect
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

import diffusion_policy_3d.rl.dppo_uav.vendor_loader as vl

FAKE_COMMIT = "cc7234ad7ff39a8f32de3af903606723a16f0648"


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════

def _make_mini_vendor(
    tmp_path: Path,
    files: dict = None,
    commit: str = FAKE_COMMIT,
) -> Path:
    """Build a minimal vendor-like snapshot in tmp_path."""
    files = files if files is not None else {"a.py": b"x" * 100}
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    sums = [
        f"{hashlib.sha256(content).hexdigest()}  {name}"
        for name, content in files.items()
    ]
    (tmp_path / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (tmp_path / "UPSTREAM_COMMIT").write_text(commit + "\n")
    return tmp_path


@pytest.fixture
def mini_vendor(tmp_path, monkeypatch):
    """Point vendor_loader at a valid mini vendor snapshot."""
    root = _make_mini_vendor(tmp_path)
    monkeypatch.setattr(vl, "VENDOR_ROOT", root)
    monkeypatch.setattr(vl, "EXPECTED_COMMIT", FAKE_COMMIT)
    return root


# ═══════════════════════════════════════════════════════════════════
# positive: real vendor
# ═══════════════════════════════════════════════════════════════════

def test_real_vendor_verifies_401() -> None:
    report = vl.verify_vendor_snapshot()
    assert report["checked_files"] == 401, (
        f"checked {report['checked_files']} != 401"
    )
    assert report["missing"] == []
    assert report["mismatched"] == []
    assert report["unexpected_path_escape"] == []
    assert report["upstream_commit"] == vl.EXPECTED_COMMIT


def test_load_symbols_offline_never_touches_network(monkeypatch) -> None:
    """Anti-network: every socket/urlopen/subprocess call must hard-fail;
    the loader must succeed with zero such calls."""
    calls = {"socket": 0, "urlopen": 0, "run": 0}

    def boom(name):
        def _boom(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} called — network/shell access")

        return _boom

    monkeypatch.setattr(socket, "socket", boom("socket"))
    monkeypatch.setattr(urllib.request, "urlopen", boom("urlopen"))
    monkeypatch.setattr(subprocess, "run", boom("run"))

    symbols = vl.load_official_dppo_symbols()
    assert calls == {"socket": 0, "urlopen": 0, "run": 0}
    assert symbols.upstream_commit == vl.EXPECTED_COMMIT


def test_symbols_resolve_inside_vendor() -> None:
    symbols = vl.load_official_dppo_symbols()
    vendor_root = symbols.vendor_root.resolve()
    for cls in (symbols.PPODiffusion, symbols.VPGDiffusion):
        src = Path(inspect.getfile(cls)).resolve()
        assert src.is_relative_to(vendor_root), (
            f"{cls.__name__} resolves outside the vendor root: {src}"
        )


def test_load_idempotent_and_path_not_duplicated() -> None:
    vl.load_official_dppo_symbols()
    vl.load_official_dppo_symbols()
    assert sys.path.count(str(vl.VENDOR_ROOT)) == 1, (
        "repeated loads must not pile up vendor_root on sys.path"
    )


def test_load_restores_single_front_entry_after_preinjection() -> None:
    """Legacy code / pytest's prepend-mode basedir juggling may have
    already placed the vendor root somewhere OTHER than sys.path[0]; a
    load must still leave EXACTLY one occurrence at the front — never a
    duplicate (R7-1 regression: the old guard only checked sys.path[0],
    so a pre-injected entry further down made the next load re-insert
    and pile up two copies)."""
    vendor = str(vl.VENDOR_ROOT)
    saved = list(sys.path)
    try:
        sys.path[:] = [
            p for p in sys.path if p is not None and str(p) != vendor
        ]
        sys.path.append(vendor)  # pre-injected at a NON-front position
        vl.load_official_dppo_symbols()
        assert sys.path[0] == vendor, (
            "load must put the vendor root at sys.path[0]"
        )
        assert sys.path.count(vendor) == 1, (
            "pre-injected occurrence must be merged, not duplicated"
        )
    finally:
        sys.path[:] = saved


# ═══════════════════════════════════════════════════════════════════
# negative: tamper / path-escape / missing / commit (mini vendor)
# ═══════════════════════════════════════════════════════════════════

def test_single_byte_tamper_hard_fails(mini_vendor) -> None:
    (mini_vendor / "a.py").write_bytes(b"y" * 100)  # 1 byte different
    with pytest.raises(vl.VendorVerificationError, match="mismatched"):
        vl.verify_vendor_snapshot()


def test_missing_file_hard_fails(mini_vendor) -> None:
    (mini_vendor / "a.py").unlink()
    with pytest.raises(vl.VendorVerificationError, match="missing"):
        vl.verify_vendor_snapshot()


def test_path_escape_hard_fails(mini_vendor) -> None:
    # Rewrite SHA256SUMS with a ../ entry; file may not exist — the
    # escape check must fire before any existence/hash check.
    (mini_vendor / "SHA256SUMS").write_text(
        hashlib.sha256(b"x" * 10).hexdigest() + "  ../outside.py\n"
    )
    with pytest.raises(vl.VendorVerificationError, match="path_escape"):
        vl.verify_vendor_snapshot()


def test_wrong_commit_hard_fails(mini_vendor) -> None:
    (mini_vendor / "UPSTREAM_COMMIT").write_text("deadbeef" * 5 + "\n")
    with pytest.raises(vl.VendorVerificationError, match="UPSTREAM_COMMIT"):
        vl.verify_vendor_snapshot()


def test_malformed_sha256sums_line_hard_fails(mini_vendor) -> None:
    (mini_vendor / "SHA256SUMS").write_text("not-a-hash  a.py\n")
    with pytest.raises(vl.VendorVerificationError):
        vl.verify_vendor_snapshot()


def test_nonvendor_model_module_refused(monkeypatch) -> None:
    """A pre-imported non-vendor 'model' must hard-fail the load."""
    fake_model = importlib.import_module("json")  # any real module
    monkeypatch.setitem(sys.modules, "model", fake_model)
    with pytest.raises(vl.VendorVerificationError, match="(?i)non-vendor"):
        vl.load_official_dppo_symbols()
