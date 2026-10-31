"""Strict identity verification for the frozen official DPPO source tree."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PureWindowsPath
import re
from typing import Dict, Optional, Tuple


EXPECTED_DPPO_COMMIT = "cc7234ad7ff39a8f32de3af903606723a16f0648"
EXPECTED_VENDOR_MANIFEST_SHA256 = (
    "5594f7d9a68d27d5b8e1bbeb4317faf8334cbe0d6202e3c8a51b103db16fee91"
)
EXPECTED_VENDOR_FILE_COUNT = 401
EXPECTED_IDP3_OVERLAY_SHA256 = {
    "model/diffusion/diffusion_ppo.py": (
        "cd770a61980a2bbd415725fb67eaddf17ae911905b5bf5530ad7684f26bff9db"
    ),
    "model/diffusion/diffusion_vpg.py": (
        "7ab594d7c3f22a2127219b1b62c665449427d552bedb92f3b3b4378e0dc8d70c"
    )
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VendorIdentity:
    root: Path
    commit: str
    manifest_sha256: str
    file_count: int
    overlay_files: Tuple[str, ...] = ()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_vendor_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "third_party" / "dppo_official_cc7234a"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "cannot locate third_party/dppo_official_cc7234a from installed package"
    )


def _parse_manifest(manifest: Path, root: Path) -> Dict[Path, str]:
    entries: Dict[Path, str] = {}
    resolved_root = root.resolve()
    text = manifest.read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError("blank SHA256SUMS entry at line {}".format(line_number))
        parts = raw_line.split(maxsplit=1)
        if len(parts) != 2 or _SHA256_RE.fullmatch(parts[0].lower()) is None:
            raise ValueError("malformed SHA256SUMS entry at line {}".format(line_number))
        digest, raw_path = parts[0].lower(), parts[1].strip()
        if raw_path.startswith("*"):
            raw_path = raw_path[1:]
        if not raw_path:
            raise ValueError("empty SHA256SUMS path at line {}".format(line_number))
        relative = Path(raw_path)
        if relative.is_absolute() or PureWindowsPath(raw_path).is_absolute():
            raise ValueError("manifest path escapes vendor root: {}".format(raw_path))
        candidate = (resolved_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                "manifest path escapes vendor root: {}".format(raw_path)
            ) from exc
        if candidate in entries:
            raise ValueError("duplicate SHA256SUMS path: {}".format(raw_path))
        entries[candidate] = digest
    if not entries:
        raise ValueError("SHA256SUMS contains no files")
    return entries


def verify_vendor_snapshot(root: Optional[Path] = None) -> VendorIdentity:
    """Verify every file and the pinned manifest identity before official import."""
    vendor_root = Path(root) if root is not None else _discover_vendor_root()
    vendor_root = vendor_root.resolve()
    if not vendor_root.is_dir():
        raise FileNotFoundError("vendor root does not exist: {}".format(vendor_root))

    manifest = vendor_root / "SHA256SUMS"
    commit_file = vendor_root / "UPSTREAM_COMMIT"
    if not manifest.is_file():
        raise FileNotFoundError("vendor SHA256SUMS is missing")
    if not commit_file.is_file():
        raise FileNotFoundError("vendor UPSTREAM_COMMIT is missing")

    commit = commit_file.read_text(encoding="utf-8").strip()
    if commit != EXPECTED_DPPO_COMMIT:
        raise ValueError(
            "vendor commit mismatch: expected {}, got {}".format(
                EXPECTED_DPPO_COMMIT, commit
            )
        )

    entries = _parse_manifest(manifest, vendor_root)
    overlay_files = []
    for path, expected_digest in entries.items():
        if not path.is_file():
            raise FileNotFoundError("manifest file is missing: {}".format(path))
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            relative = path.relative_to(vendor_root).as_posix()
            overlay_digest = EXPECTED_IDP3_OVERLAY_SHA256.get(relative)
            if actual_digest != overlay_digest:
                raise ValueError(
                    "vendor digest mismatch for {}: expected upstream {} or "
                    "audited overlay {}, got {}".format(
                        relative, expected_digest, overlay_digest, actual_digest
                    )
                )
            overlay_files.append(relative)

    manifest_sha256 = _sha256_file(manifest)
    if manifest_sha256 != EXPECTED_VENDOR_MANIFEST_SHA256:
        raise ValueError(
            "vendor manifest identity mismatch: expected {}, got {}".format(
                EXPECTED_VENDOR_MANIFEST_SHA256, manifest_sha256
            )
        )
    if len(entries) != EXPECTED_VENDOR_FILE_COUNT:
        raise ValueError(
            "vendor file count mismatch: expected {}, got {}".format(
                EXPECTED_VENDOR_FILE_COUNT, len(entries)
            )
        )

    return VendorIdentity(
        root=vendor_root,
        commit=commit,
        manifest_sha256=manifest_sha256,
        file_count=len(entries),
        overlay_files=tuple(sorted(overlay_files)),
    )
