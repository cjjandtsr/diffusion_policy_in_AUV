import hashlib
from pathlib import Path

import pytest

from scripts.prepare_reward_v17 import evidence


def _payload(relative_path, digest):
    return {
        "source_evidence": {
            "path_base": "repo_root",
            "coordinate_reference": {"frame": "fixture_world_z_up"},
            "files": [{"path": relative_path, "sha256": digest}],
        }
    }


def test_repo_relative_evidence_is_host_path_independent(tmp_path):
    first = tmp_path / "checkout_a"
    second = tmp_path / "checkout_b"
    relative = "scripts/generate_reward_v17_fixtures.py"
    payload = b"frozen generator\n"
    digest = hashlib.sha256(payload).hexdigest()
    for root in (first, second):
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)

    left = evidence(_payload(relative, digest), first / "data/reward_v17", repo_root=first)
    right = evidence(_payload(relative, digest), second / "data/reward_v17", repo_root=second)

    assert left == right
    assert left["path_base"] == "repo_root"
    assert left["files"] == [{"path": relative, "sha256": digest}]


def test_repo_relative_evidence_rejects_path_escape(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="escapes repo_root"):
        evidence(
            _payload("../outside.py", digest),
            tmp_path / "repo/data/reward_v17",
            repo_root=tmp_path / "repo",
        )


def test_repo_relative_evidence_rejects_symlink(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    link = root / "scripts/link.py"
    link.parent.mkdir(parents=True)
    link.write_bytes(b"real\n")
    digest = hashlib.sha256(link.read_bytes()).hexdigest()
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: True if self == link else original(self),
    )
    with pytest.raises(ValueError, match="regular non-symlink"):
        evidence(
            _payload("scripts/link.py", digest),
            root / "data/reward_v17",
            repo_root=root,
        )
