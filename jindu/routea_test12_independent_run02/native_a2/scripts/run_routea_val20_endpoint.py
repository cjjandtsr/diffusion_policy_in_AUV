"""Run the identity-pinned formal endpoint on the RL validation split only."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import evaluate_dppo_uav_formal_endpoint as endpoint
from scripts.formal_endpoint_eval_core import PairedEpisodeRecord


ARRIVAL_MIN_TOLERANCE_M = 1e-6
_ARRIVAL_ERROR = "reached must mean safe three-dimensional arrival within 0.5 m"


class Val20EpisodeRecord(PairedEpisodeRecord):
    """Accept only float32 rounding beyond an authoritative safe goal event."""

    boundary_events = []

    def __post_init__(self):
        try:
            super().__post_init__()
        except ValueError as exc:
            if str(exc) != _ARRIVAL_ERROR:
                raise
            if (self.done_reason != "reached" or self.raw_done_reason != "reached"
                    or self.goal_xyz is None or self.terminal_safety_flags):
                raise
            distance = math.sqrt(sum(
                (float(a) - float(b)) ** 2
                for a, b in zip(self.terminal_xyz, self.goal_xyz)
            ))
            coordinate_ulp = np.abs(np.spacing(np.abs(np.asarray(
                self.terminal_xyz, dtype=np.float32
            )))).astype(np.float64)
            rounding_allowance = max(
                ARRIVAL_MIN_TOLERANCE_M,
                0.5 * float(np.linalg.norm(coordinate_ulp)) + 1e-9,
            )
            if not 0.5 < distance <= 0.5 + rounding_allowance:
                raise
            self.boundary_events.append({
                "task_id": self.task_id,
                "config_hash": self.config_hash,
                "policy": self.policy,
                "world_id": self.world_id,
                "done_reason": self.done_reason,
                "raw_done_reason": self.raw_done_reason,
                "terminal_safety_flags": self.terminal_safety_flags,
                "distance_m": distance,
                "excess_m": distance - 0.5,
                "rounding_allowance_m": rounding_allowance,
                "terminal_xyz": self.terminal_xyz,
                "goal_xyz": self.goal_xyz,
            })

    @property
    def success(self):
        # The environment has already committed its first safe goal event.
        # The base class' exact coordinate test can disagree after float32 commit.
        return self.done_reason == "reached" and not self.terminal_safety_flags


def install_val_split():
    endpoint.BENCHMARK_SPLITS["routea-val"] = ("rl_val",)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_boundary_audit(output_dir):
    path = output_dir / "arrival_boundary_audit.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite arrival boundary audit")
    payload = {
        "schema_version": 1,
        "evaluation_wrapper_sha256": _sha256(__file__),
        "identity_pinned_endpoint_sha256": _sha256(endpoint.__file__),
        "evaluation_provenance_sha256": _sha256(output_dir / "evaluation_provenance.json"),
        "radius_m": 0.5,
        "minimum_rounding_tolerance_m": ARRIVAL_MIN_TOLERANCE_M,
        "rounding_allowance_rule": "max(1e-6 m, half Euclidean float32 coordinate ULP + 1e-9 m)",
        "accepted_boundary_count": len(Val20EpisodeRecord.boundary_events),
        "accepted_boundary_events": Val20EpisodeRecord.boundary_events,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rl-checkpoint", required=True, type=Path)
    parser.add_argument("--eval-config-dir", required=True, type=Path)
    parser.add_argument("--benchmark", choices=("routea-val",), required=True)
    parser.add_argument("--worlds-val", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-update", required=True, type=int)
    args = parser.parse_args(argv)
    args.worlds_test = None
    args.worlds_layera = None
    install_val_split()
    endpoint.PairedEpisodeRecord = Val20EpisodeRecord
    Val20EpisodeRecord.boundary_events.clear()
    result = endpoint.run(args)
    if result == 0:
        _write_boundary_audit(args.output_dir.resolve())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
