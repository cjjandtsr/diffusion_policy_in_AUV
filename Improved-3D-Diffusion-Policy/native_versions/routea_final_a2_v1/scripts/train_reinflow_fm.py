#!/usr/bin/env python3
"""Formal-training admission guard for the ReinFlow × iDP3 FM route.

This Windows staging revision intentionally has no R6 environment loop.  It
accepts a formal request only after an immutable R0-R5 admission artifact and a
later configuration revision explicitly enable formal training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FormalAdmissionError(RuntimeError):
    pass


def _mapping(path: Path, *, yaml_file: bool) -> Mapping:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) if yaml_file else json.load(stream)
    if not isinstance(value, Mapping):
        raise FormalAdmissionError(f"artifact must be a mapping: {path}")
    return value


def verify_r0_r5(admission: Mapping) -> None:
    for index in range(6):
        gate = f"R{index}"
        value = admission.get(gate)
        if not isinstance(value, Mapping) or value.get("status") != "PASS":
            raise FormalAdmissionError(f"{gate} is not PASS")
    r4 = admission["R4"]
    if r4.get("rollback_state_hash_equal") is not True:
        raise FormalAdmissionError("R4 lacks exact rollback hash evidence")
    r5 = admission["R5"]
    if r5.get("cold_resume_next_update_equal") is not True:
        raise FormalAdmissionError("R5 lacks cold-resume next-update evidence")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "reinflow_fm" / "smoke.yaml",
    )
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args(argv)

    config = _mapping(args.config.resolve(), yaml_file=True)
    admission = _mapping(args.admission.resolve(), yaml_file=False)
    verify_r0_r5(admission)
    formal_enabled = bool(config.get("gates", {}).get("formal_training_enabled"))
    if args.formal:
        if not formal_enabled:
            raise FormalAdmissionError(
                "formal training is disabled in this R0-R3 Windows staging revision"
            )
        raise FormalAdmissionError(
            "formal execution requires the later R6 environment-runner revision; "
            "this guard will not substitute an unverified training loop"
        )
    print(
        json.dumps(
            {
                "status": "R0_R5_ADMISSION_VALID",
                "formal_training_enabled": formal_enabled,
                "next_gate": "R6 environment-runner implementation and 2-update smoke",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
