#!/usr/bin/env python3
"""Run the formal endpoint evaluation and file the artifacts under the shared root.

`evaluate_dppo_uav_formal_endpoint.py` is pinned by every provenance file it
writes -- the S2 pilot pack pins its current bytes -- so editing it would make
`dump_dppo_uav_endpoint_traces.py` refuse to touch that already-collected
evidence. This wrapper adds the shared output root without touching it: it
fills in `--output-dir` and calls the pinned script's `main()` unchanged.

With no `--output-dir` the run lands at

    <root>/formal_eval/<YYYYMMDD_HHMMSS>_<checkpoint run name>/

so an artifact identifies itself by when it ran and which checkpoint it scored.
Every other argument is passed through exactly as given; `--output-dir` still
wins when supplied, which keeps a rerun-beside-an-old-one possible.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.endpoint_output_root import (  # noqa: E402
    DEFAULT_ROOT, outside_default_root, resolve_output_dir,
)
from scripts.evaluate_dppo_uav_formal_endpoint import main as evaluate_main  # noqa: E402


def main(argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--output-dir", type=Path)
    probe.add_argument("--rl-checkpoint", type=Path)
    known, rest = probe.parse_known_args(argv)
    if known.output_dir is not None:
        return evaluate_main(argv)
    label = known.rl_checkpoint.resolve().parent.name if known.rl_checkpoint else "run"
    output = resolve_output_dir(None, "formal_eval", label)
    if outside_default_root(output):
        print("warning: output is outside the shared endpoint root {}: {}".format(
            DEFAULT_ROOT, output), flush=True)
    print("endpoint output: {}".format(output), flush=True)
    # The probe consumed --rl-checkpoint; hand it back so the pinned parser sees it.
    rest = list(rest)
    if known.rl_checkpoint is not None:
        rest += ["--rl-checkpoint", str(known.rl_checkpoint)]
    return evaluate_main(rest + ["--output-dir", str(output)])


if __name__ == "__main__":
    raise SystemExit(main())
