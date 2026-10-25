"""Exactly one writer owns a run output directory."""

import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.io_safety import RunLock


def test_only_one_owner_acquires_and_release_is_idempotent(tmp_path):
    first = RunLock.acquire(tmp_path, "owner-a")
    with pytest.raises(FileExistsError):
        RunLock.acquire(tmp_path, "owner-b")
    first.release()
    first.release()
    second = RunLock.acquire(tmp_path, "owner-b")
    second.release()
