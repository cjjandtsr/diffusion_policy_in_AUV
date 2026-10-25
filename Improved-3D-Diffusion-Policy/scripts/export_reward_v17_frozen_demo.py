#!/usr/bin/env python3
"""Export frozen v6.5 state to Reward v1.7 density-source JSON; never mutates Zarr."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_policy_3d.env.reward_v17_support.frozen_demo import array_sha256, export_frozen_demo


def file_sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-file", action="append", required=True,
                        help="recording/conversion source file; repeat for the complete evidence chain")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("refusing to overwrite existing artifact: {}".format(output))
    dataset = Path(args.dataset).resolve()
    root = zarr.open(str(dataset), mode="r")
    state = root["data/state"][:]
    ends = root["meta/episode_ends"][:].astype(np.int64)
    files = []
    for value in args.source_file:
        path = Path(value).resolve()
        if not path.is_file():
            parser.error("source evidence file missing: {}".format(path))
        files.append(dict(path=str(path), sha256=file_sha256(path)))
    result = export_frozen_demo(
        state, ends, ground_z=0., state_array_sha256=array_sha256(state),
        episode_ends_array_sha256=array_sha256(ends), source_files=files)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="written_frozen_export_not_density", output=str(output),
                          sha256=file_sha256(output), **result["summary"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
