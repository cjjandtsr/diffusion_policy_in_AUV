"""Cryptographic identity checks for the pinned official ReinFlow snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import string
from typing import Iterable


EXPECTED_REINFLOW_COMMIT = "e722e151bed767f3ffef47527cf697f2358af55d"
EXPECTED_VENDOR_MANIFEST_SHA256 = (
    "1ae2d625888ce9bdd4ed79911f42eb4a6dc9c5197c3ed1a163d1d74d1ddcda09"
)
EXPECTED_VENDOR_FILE_COUNT = 577
VENDOR_DIRECTORY_NAME = "reinflow_official_e722e15"

_METADATA_FILES = frozenset({"SHA256SUMS", "UPSTREAM_COMMIT"})
_CACHE_DIRECTORIES = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
)
_CACHE_SUFFIXES = frozenset({".pyc", ".pyo"})
_HEX_DIGITS = frozenset(string.hexdigits)


@dataclass(frozen=True)
class VendorIdentity:
    """Verified immutable identity of a vendored ReinFlow snapshot."""

    root: Path
    commit: str
    manifest_sha256: str
    file_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_vendor_root() -> Path:
    """Locate the sibling ``third_party`` vendor from this installed source tree."""

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "third_party" / VENDOR_DIRECTORY_NAME
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not locate third_party/{VENDOR_DIRECTORY_NAME} from {__file__}"
    )


def _parse_manifest(manifest_path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        try:
            expected_digest, relative_text = raw_line.split("  ", 1)
        except ValueError as error:
            raise ValueError(
                f"invalid SHA256SUMS line {line_number}: expected two-space separator"
            ) from error
        digest = expected_digest.lower()
        if len(digest) != 64 or any(char not in _HEX_DIGITS for char in digest):
            raise ValueError(f"invalid SHA-256 digest on line {line_number}")

        posix_path = PurePosixPath(relative_text)
        windows_path = PureWindowsPath(relative_text)
        if (
            not relative_text
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in posix_path.parts
            or ".." in windows_path.parts
        ):
            raise ValueError(
                f"manifest path escapes vendor root on line {line_number}: "
                f"{relative_text!r}"
            )
        normalized = posix_path.as_posix()
        if normalized in seen:
            raise ValueError(f"duplicate manifest path: {normalized}")
        seen.add(normalized)
        entries.append((digest, normalized))
    return entries


def _is_ignored_runtime_file(relative_path: Path) -> bool:
    return (
        relative_path.as_posix() in _METADATA_FILES
        or any(part in _CACHE_DIRECTORIES for part in relative_path.parts)
        or relative_path.suffix.lower() in _CACHE_SUFFIXES
    )


def _iter_vendor_files(root: Path) -> Iterable[str]:
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root)
        if not _is_ignored_runtime_file(relative):
            yield relative.as_posix()


def verify_vendor_snapshot(vendor_root: Path | str | None = None) -> VendorIdentity:
    """Verify commit, manifest, file hashes, and absence of extra vendor files.

    The check is intentionally fail-closed.  Project code must never silently import
    a locally modified or partially copied ReinFlow implementation.
    """

    root = Path(vendor_root) if vendor_root is not None else _discover_vendor_root()
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"ReinFlow vendor root is not a directory: {root}")

    commit_path = root / "UPSTREAM_COMMIT"
    manifest_path = root / "SHA256SUMS"
    if not commit_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"ReinFlow vendor metadata missing under {root}: "
            "UPSTREAM_COMMIT and SHA256SUMS are required"
        )

    commit = commit_path.read_text(encoding="utf-8").strip()
    if commit != EXPECTED_REINFLOW_COMMIT:
        raise ValueError(
            "ReinFlow vendor commit mismatch: "
            f"expected {EXPECTED_REINFLOW_COMMIT}, found {commit}"
        )

    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != EXPECTED_VENDOR_MANIFEST_SHA256:
        raise ValueError(
            "ReinFlow SHA256SUMS digest mismatch: "
            f"expected {EXPECTED_VENDOR_MANIFEST_SHA256}, found {manifest_sha256}"
        )

    entries = _parse_manifest(manifest_path)
    if len(entries) != EXPECTED_VENDOR_FILE_COUNT:
        raise ValueError(
            "ReinFlow vendor file count mismatch: "
            f"expected {EXPECTED_VENDOR_FILE_COUNT}, found {len(entries)}"
        )

    manifest_files = {relative for _, relative in entries}
    for expected_digest, relative in entries:
        candidate = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"manifest path escapes vendor root: {relative!r}") from error
        if not candidate.is_file():
            raise FileNotFoundError(f"manifested vendor file missing: {relative}")
        actual_digest = _sha256(candidate)
        if actual_digest != expected_digest:
            raise ValueError(
                f"ReinFlow vendor digest mismatch for {relative}: "
                f"expected {expected_digest}, found {actual_digest}"
            )

    actual_files = set(_iter_vendor_files(root))
    unmanifested = sorted(actual_files - manifest_files)
    if unmanifested:
        preview = ", ".join(unmanifested[:5])
        raise ValueError(f"unmanifested vendor file(s): {preview}")
    missing = sorted(manifest_files - actual_files)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"manifested vendor file(s) missing: {preview}")

    return VendorIdentity(
        root=root,
        commit=commit,
        manifest_sha256=manifest_sha256,
        file_count=len(entries),
    )


__all__ = [
    "EXPECTED_REINFLOW_COMMIT",
    "EXPECTED_VENDOR_FILE_COUNT",
    "EXPECTED_VENDOR_MANIFEST_SHA256",
    "VendorIdentity",
    "verify_vendor_snapshot",
]
