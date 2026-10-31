#!/usr/bin/env python3
"""Verify the frozen official DPPO source snapshot without network access."""

import argparse
import json
from pathlib import Path

from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.vendor_identity import (
    verify_vendor_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "vendor_root",
        nargs="?",
        type=Path,
        help="DPPO vendor root; omit after the staged package is copied into REPO",
    )
    args = parser.parse_args()
    identity = verify_vendor_snapshot(args.vendor_root)
    print(
        json.dumps(
            {
                "commit": identity.commit,
                "file_count": identity.file_count,
                "manifest_sha256": identity.manifest_sha256,
                "overlay_files": list(identity.overlay_files),
                "root": str(identity.root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
