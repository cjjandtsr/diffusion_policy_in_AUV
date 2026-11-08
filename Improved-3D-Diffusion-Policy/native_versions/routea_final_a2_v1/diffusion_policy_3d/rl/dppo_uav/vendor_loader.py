"""Offline loader for the frozen official DPPO runtime (guide R7-1).

Single sanctioned entry point into ``third_party/dppo_official_cc7234a``:

- resolves the vendor root from THIS file's location — no environment
  overrides, no URL, no network fallback;
- verifies the full 401-file SHA256SUMS allowlist and the pinned
  UPSTREAM_COMMIT inside Python BEFORE any import (missing /
  mismatched / path-escape all hard-fail);
- only then injects the vendor root at ``sys.path[0]`` and imports
  ``model.diffusion.diffusion_ppo.PPODiffusion`` and
  ``model.diffusion.diffusion_vpg.VPGDiffusion``;
- hard-fails when a non-vendor module named ``model`` is already
  imported (import-cache pollution guard).

No DPPO/DDIM math lives here — this module locates, verifies and exposes
the frozen symbols only (guide file map single responsibility).
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = REPO_ROOT / "third_party" / "dppo_official_cc7234a"
EXPECTED_COMMIT = "cc7234ad7ff39a8f32de3af903606723a16f0648"

SHA256SUMS_NAME = "SHA256SUMS"
UPSTREAM_COMMIT_NAME = "UPSTREAM_COMMIT"


class VendorVerificationError(RuntimeError):
    """Vendor snapshot verification failed; importing it is forbidden."""


@dataclass(frozen=True)
class OfficialDPPOSymbols:
    """Verified handles into the frozen official runtime."""

    PPODiffusion: type
    VPGDiffusion: type
    vendor_root: Path
    upstream_commit: str


def _parse_sha256sums(text: str) -> List[Tuple[str, str]]:
    """Parse ``SHA256SUMS`` lines: ``<64hex>  <relative path>``.

    Raises VendorVerificationError on any malformed line — a malformed
    allowlist must never be accepted silently.
    """
    entries: List[Tuple[str, str]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise VendorVerificationError(
                f"SHA256SUMS:{lineno} expected '<hex>  <path>', got: {line!r}"
            )
        digest, rel = parts
        if len(digest) != 64 or digest.lower() != digest or not all(
            c in "0123456789abcdef" for c in digest
        ):
            raise VendorVerificationError(
                f"SHA256SUMS:{lineno} digest not 64-lowercase-hex: {digest!r}"
            )
        entries.append((digest, rel))
    return entries


def verify_vendor_snapshot() -> Dict[str, object]:
    """Verify the vendor snapshot; returns the verification report.

    Report keys: ``checked_files``, ``missing``, ``mismatched``,
    ``unexpected_path_escape``, ``upstream_commit``, ``vendor_root``.
    Raises VendorVerificationError (not merely reports) on malformed
    SHA256SUMS / missing vendor root / wrong UPSTREAM_COMMIT.
    """
    if not VENDOR_ROOT.is_dir():
        raise VendorVerificationError(f"vendor root missing: {VENDOR_ROOT}")
    sums_path = VENDOR_ROOT / SHA256SUMS_NAME
    commit_path = VENDOR_ROOT / UPSTREAM_COMMIT_NAME
    if not sums_path.is_file() or not commit_path.is_file():
        raise VendorVerificationError(
            f"vendor snapshot incomplete: {sums_path} / {commit_path}"
        )

    entries = _parse_sha256sums(sums_path.read_text())
    missing: List[str] = []
    mismatched: List[str] = []
    unexpected_path_escape: List[str] = []
    vendor_root_resolved = VENDOR_ROOT.resolve()

    for digest, rel in entries:
        # SHA256SUMS uses "./"-prefixed paths; resolve from the vendor root.
        candidate = (VENDOR_ROOT / rel).resolve()
        if not candidate.is_relative_to(vendor_root_resolved):
            unexpected_path_escape.append(rel)
            continue
        if not candidate.is_file():
            missing.append(rel)
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != digest:
            mismatched.append(f"{rel} (expected {digest}, got {actual})")

    commit = commit_path.read_text().strip()
    if commit != EXPECTED_COMMIT:
        raise VendorVerificationError(
            f"UPSTREAM_COMMIT {commit!r} != expected {EXPECTED_COMMIT!r}"
        )
    if missing or mismatched or unexpected_path_escape:
        raise VendorVerificationError(
            "vendor snapshot verification FAILED:"
            f" missing={missing} mismatched={mismatched}"
            f" path_escape={unexpected_path_escape}"
        )
    return {
        "checked_files": len(entries),
        "missing": missing,
        "mismatched": mismatched,
        "unexpected_path_escape": unexpected_path_escape,
        "upstream_commit": commit,
        "vendor_root": str(VENDOR_ROOT),
    }


def _ensure_vendor_on_path() -> None:
    """Ensure VENDOR_ROOT sits exactly once, at sys.path[0].

    Idempotent even when the same root was already injected elsewhere on
    sys.path (legacy test modules, pytest's prepend-mode basedir
    juggling): every existing occurrence is removed first, then the root
    is inserted once at the front — repeated loads never pile up
    duplicates. A guard that only checked sys.path[0] left a
    pre-injected copy further down in place, so the next load created a
    second entry (R7-1 regression, fixed 2026-08-23).
    """
    vendor = str(VENDOR_ROOT)
    sys.path[:] = [p for p in sys.path if p is not None and str(p) != vendor]
    sys.path.insert(0, vendor)


def _guard_model_import_cache() -> None:
    """Refuse a pre-imported non-vendor ``model`` package.

    The vendor ships a top-level ``model`` package; a ``model`` already in
    sys.modules from somewhere else would silently shadow it. Hard-fail
    instead of reusing the polluted cache.
    """
    existing = sys.modules.get("model")
    if existing is None:
        return
    try:
        src = inspect.getfile(existing)
    except TypeError:
        src = None
    if src is None or not Path(src).resolve().is_relative_to(
        VENDOR_ROOT.resolve()
    ):
        raise VendorVerificationError(
            "module 'model' already imported from a NON-vendor location:"
            f" {src} — refusing to reuse the polluted import cache"
        )


def load_official_dppo_symbols() -> OfficialDPPOSymbols:
    """Verify the frozen vendor snapshot and import its core symbols.

    Callable repeatedly (idempotent); each call re-verifies hashes so a
    vendor change between calls cannot go unnoticed.
    """
    report = verify_vendor_snapshot()
    _guard_model_import_cache()
    _ensure_vendor_on_path()

    from model.diffusion.diffusion_ppo import PPODiffusion  # noqa: E402
    from model.diffusion.diffusion_vpg import VPGDiffusion  # noqa: E402

    vendor_root_resolved = VENDOR_ROOT.resolve()
    for symbol in (PPODiffusion, VPGDiffusion):
        src = inspect.getfile(symbol)
        if src is None or not Path(src).resolve().is_relative_to(
            vendor_root_resolved
        ):
            raise VendorVerificationError(
                f"{symbol.__name__} resolved to {src} — outside the frozen"
                " vendor root; import path pollution"
            )

    return OfficialDPPOSymbols(
        PPODiffusion=PPODiffusion,
        VPGDiffusion=VPGDiffusion,
        vendor_root=VENDOR_ROOT,
        upstream_commit=str(report["upstream_commit"]),
    )
