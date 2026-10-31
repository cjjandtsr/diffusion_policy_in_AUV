#!/usr/bin/env python3
"""Build a frozen held-out-reset bank for paired DDPO credit probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dppo_credit_support.credit_bank import make_dev_bank


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(repo_root: Path, value) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def build(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory import build_episode_tasks

    config_path = args.config.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite dev bank: {}".format(output_path))
    config = OmegaConf.load(config_path)
    manifest_path = _resolve(REPO_ROOT, config.admission_manifest)
    worlds_root = _resolve(REPO_ROOT, config.worlds_train)
    task_specs = build_episode_tasks(
        manifest_path,
        worlds_root,
        int(config.run.seed),
        OmegaConf.to_container(config.environment, resolve=True),
    )
    tasks = [
        {"task_id": task.task_id, "world_id": task.world_id, "role": task.role,
         "task_seed": task.seed, "task_identity_hash": task.identity_hash,
         "task_seed_preimage": task.seed_preimage, "task_seed_digest": task.seed_digest,
         "obstacle_sha256": _sha256(Path(task.obstacle_json))}
        for task in task_specs
    ]

    payload = make_dev_bank(
        tasks,
        bank_id=args.bank_id,
        entry_count=args.entries,
        source_manifest_sha256=_sha256(manifest_path),
        exclude_training_segments=args.exclude_training_segments,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with output_path.open("xb") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    result = {
        "status": "DEV_BANK_BUILT",
        "output": str(output_path),
        "file_sha256": _sha256(output_path),
        "content_sha256": payload["content_sha256"],
        "bank_id": payload["bank_id"],
        "entry_count": payload["entry_count"],
        "source_task_count": payload["source_task_count"],
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "training_reset_seed_overlap_count": payload[
            "training_reset_seed_overlap_count"
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bank-id", required=True)
    # 8,192 guarantees enough entries for an 8,192-transition probe even if
    # every episode terminates after its first macro transition.
    parser.add_argument("--entries", type=int, default=8192)
    parser.add_argument("--exclude-training-segments", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.entries <= 0:
        parser.error("--entries must be positive")
    if args.exclude_training_segments < 0:
        parser.error("--exclude-training-segments must be non-negative")
    if args.output.exists():
        parser.error("refusing to overwrite existing --output")
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
