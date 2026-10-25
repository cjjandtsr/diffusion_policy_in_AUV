"""Checkpoint v2 rejects path and JSON attacks before deserialization."""

import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.io_safety import (
    SecurePathError,
    load_bounded_json,
    secure_open_under,
)


@pytest.mark.parametrize("name", ["../escape", "/abs/path", "", "."])
def test_secure_open_rejects_escape(tmp_path, name):
    with pytest.raises(SecurePathError):
        secure_open_under(tmp_path, name, max_bytes=1024)


def test_secure_open_rejects_oversize_and_duplicate_json(tmp_path):
    (tmp_path / "large.bin").write_bytes(b"x" * 1025)
    with pytest.raises(SecurePathError, match="size"):
        secure_open_under(tmp_path, "large.bin", max_bytes=1024)
    (tmp_path / "duplicate.json").write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_bounded_json(tmp_path, "duplicate.json", max_bytes=1024)
