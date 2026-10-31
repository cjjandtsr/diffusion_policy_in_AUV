"""Bounded path I/O, single-writer run locks, and atomic directory publish."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePath
import stat


class SecurePathError(RuntimeError):
    pass


def _relative_parts(relative_path):
    text = os.fspath(relative_path)
    path = Path(text)
    if not text or text == "." or path.is_absolute():
        raise SecurePathError("path must be a non-empty relative file")
    parts = path.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise SecurePathError("path traversal is forbidden")
    return parts


def _reject_reparse(stat_result):
    if stat.S_ISLNK(stat_result.st_mode):
        raise SecurePathError("symlink path component is forbidden")
    attributes = getattr(stat_result, "st_file_attributes", 0)
    if attributes & 0x400:
        raise SecurePathError("reparse path component is forbidden")


def secure_open_under(root, relative_path, *, max_bytes):
    """Open a bounded regular file beneath a trusted root without symlinks."""

    if isinstance(max_bytes, bool) or int(max_bytes) < 1:
        raise ValueError("max_bytes must be positive")
    raw_root = Path(root)
    root_info = raw_root.lstat()
    _reject_reparse(root_info)
    root = raw_root.resolve(strict=True)
    parts = _relative_parts(relative_path)
    if os.name == "posix" and hasattr(os, "O_NOFOLLOW"):
        directory_fds = []
        try:
            current_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            directory_fds.append(current_fd)
            for part in parts[:-1]:
                current_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current_fd,
                )
                directory_fds.append(current_fd)
            descriptor = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd
            )
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                os.close(descriptor)
                raise SecurePathError("target must be a regular file")
            if info.st_size <= 0 or info.st_size > int(max_bytes):
                os.close(descriptor)
                raise SecurePathError("file size is outside the accepted range")
            return os.fdopen(descriptor, "rb", closefd=True)
        except (FileNotFoundError, NotADirectoryError, OSError) as exc:
            if isinstance(exc, SecurePathError):
                raise
            raise SecurePathError("openat path validation failed") from exc
        finally:
            for descriptor in reversed(directory_fds):
                os.close(descriptor)
    current = root
    for part in parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise SecurePathError("path does not exist") from exc
        _reject_reparse(info)
    try:
        current.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise SecurePathError("resolved path escapes trusted root") from exc
    info = current.stat()
    if not stat.S_ISREG(info.st_mode):
        raise SecurePathError("target must be a regular file")
    if info.st_size <= 0 or info.st_size > int(max_bytes):
        raise SecurePathError("file size is outside the accepted range")
    stream = current.open("rb")
    opened = os.fstat(stream.fileno())
    if (opened.st_dev, opened.st_ino, opened.st_size) != (
        info.st_dev, info.st_ino, info.st_size
    ):
        stream.close()
        raise SecurePathError("file identity changed during open")
    return stream


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _validate_json_limits(value, *, depth=0, counts=None):
    if depth > 32:
        raise ValueError("JSON nesting exceeds 32")
    if counts is None:
        counts = [0]
    counts[0] += 1
    if counts[0] > 100000:
        raise ValueError("JSON entry count exceeds 100000")
    if isinstance(value, str) and len(value.encode("utf-8")) > 4096:
        raise ValueError("JSON string exceeds 4096 bytes")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_limits(key, depth=depth + 1, counts=counts)
            _validate_json_limits(item, depth=depth + 1, counts=counts)
    elif isinstance(value, list):
        for item in value:
            _validate_json_limits(item, depth=depth + 1, counts=counts)


def load_bounded_json(root, relative_path, *, max_bytes):
    with secure_open_under(root, relative_path, max_bytes=max_bytes) as stream:
        raw = stream.read()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant: {}".format(value))
            ),
        )
    except UnicodeDecodeError as exc:
        raise ValueError("JSON must be UTF-8") from exc
    _validate_json_limits(value)
    return value


class RunLock:
    def __init__(self, path, owner_token):
        self.path = Path(path)
        self.owner_token = str(owner_token)
        self._released = False

    @classmethod
    def acquire(cls, output_dir, owner_token):
        if not str(owner_token) or len(str(owner_token)) > 4096:
            raise ValueError("owner_token is invalid")
        raw_output = Path(output_dir)
        if raw_output.exists():
            _reject_reparse(raw_output.lstat())
        output = raw_output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        path = output / "run.lock"
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            payload = (json.dumps({"owner_token": str(owner_token)}, sort_keys=True) + "\n").encode("utf-8")
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return cls(path, owner_token)

    def release(self):
        if self._released:
            return
        payload = load_bounded_json(self.path.parent, self.path.name, max_bytes=8192)
        if payload != {"owner_token": self.owner_token}:
            raise PermissionError("run lock owner token mismatch")
        self.path.unlink()
        self._released = True


def atomic_publish_directory(temporary, destination, manifest):
    raw_temporary = Path(temporary)
    raw_destination = Path(destination)
    _reject_reparse(raw_temporary.lstat())
    if raw_destination.exists():
        _reject_reparse(raw_destination.lstat())
    temporary = raw_temporary.resolve()
    destination = raw_destination.resolve()
    if temporary.parent != destination.parent:
        raise ValueError("temporary and destination must share a parent")
    if destination.exists():
        raise FileExistsError("destination already exists")
    committed = temporary / "COMMITTED.json"
    if committed.exists():
        raise FileExistsError("commit manifest already exists")
    with committed.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for child in temporary.iterdir():
        if child.is_file():
            _reject_reparse(child.lstat())
            with child.open("r+b") as stream:
                os.fsync(stream.fileno())
    os.replace(temporary, destination)
    try:
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError:
        if os.name != "nt":
            raise
    return destination


__all__ = (
    "RunLock",
    "SecurePathError",
    "atomic_publish_directory",
    "load_bounded_json",
    "secure_open_under",
)
