#!/usr/bin/env python3
"""Cold-load a real DPPO checkpoint without performing another update."""

import argparse
import json
from pathlib import Path

from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.resume_probe import (
    verify_real_resume,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    result = verify_real_resume(
        args.config,
        args.checkpoint,
        repo_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
