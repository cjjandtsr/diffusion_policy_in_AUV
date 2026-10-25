#!/usr/bin/env python3
"""Write the deterministic Reward v1.7 geometric calibration manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_policy_3d.env.reward_v17_support.fixture_manifest import make_fixture_manifest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("refusing to overwrite existing artifact: {}".format(output))
    source = Path(__file__).resolve()
    manifest = make_fixture_manifest(str(source), sha256(source))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="written_fixture_manifest_not_calibrated", output=str(output),
                          sha256=sha256(output), episodes=len(manifest["episodes"]),
                          comparisons=len(manifest["comparisons"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
