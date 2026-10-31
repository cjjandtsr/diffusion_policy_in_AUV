"""Read-only cold-resume probe for a real saved DPPO checkpoint."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from pathlib import Path

from omegaconf import OmegaConf

from .schema import CheckpointHeader


CHECKPOINT_FILES = frozenset(("header.json", "state.pt", "SHA256.json", "COMMITTED.json"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_hashes(root):
    root = Path(root)
    files = {path.name for path in root.iterdir() if path.is_file()}
    if files != CHECKPOINT_FILES:
        raise ValueError(
            "checkpoint file set mismatch: expected {}, got {}".format(
                sorted(CHECKPOINT_FILES), sorted(files)
            )
        )
    return {name: _sha256(root / name) for name in sorted(files)}


def _counter_payload(counters):
    if is_dataclass(counters):
        return asdict(counters)
    return dict(vars(counters))


def verify_real_resume(config_path, checkpoint_path, repo_root, *, run_fn=None):
    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    repo_root = Path(repo_root).resolve()
    header = CheckpointHeader.from_json(
        (checkpoint_path / "header.json").read_text(encoding="utf-8")
    )
    config = OmegaConf.load(config_path)
    configured_updates = int(config.run.updates)
    if configured_updates != header.update_index:
        raise ValueError(
            "resume probe config would continue training: updates={} checkpoint={}".format(
                configured_updates, header.update_index
            )
        )
    before = _checkpoint_hashes(checkpoint_path)
    if run_fn is None:
        from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import run

        run_fn = run
    counters = run_fn(config, repo_root, resume_from=checkpoint_path)
    after = _checkpoint_hashes(checkpoint_path)
    if before != after:
        raise RuntimeError("resume probe modified checkpoint artifacts")
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_unchanged": True,
        "update_index": header.update_index,
        "runtime_identity_hash": header.runtime_identity_hash,
        "lora_identity_hash": header.lora_identity_hash,
        "file_sha256": after,
        "counters": _counter_payload(counters),
    }
