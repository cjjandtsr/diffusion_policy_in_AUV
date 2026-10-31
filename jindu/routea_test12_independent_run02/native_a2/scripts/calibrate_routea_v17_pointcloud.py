#!/usr/bin/env python3
"""Freeze a distribution-free partial-no-ray threshold from BC-only records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


_MAX_FILE_BYTES = 512 * 1024 * 1024
_MAX_LINE_CHARS = 1024 * 1024
_MAX_ROWS = 1_000_000


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON constant: {}".format(value))


def load_transition_rows(path):
    """Stream bounded JSONL without duplicate-key or non-finite ambiguity."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("BC transition evidence must be a regular non-symlink file")
    if source.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError("BC transition evidence exceeds 512 MiB")
    rows = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        for line_number in range(1, _MAX_ROWS + 2):
            line = stream.readline(_MAX_LINE_CHARS + 1)
            if not line:
                break
            if len(line) > _MAX_LINE_CHARS:
                raise ValueError("JSONL line exceeds 1 MiB at line {}".format(line_number))
            if not line.strip():
                continue
            if len(rows) >= _MAX_ROWS:
                raise ValueError("BC transition evidence exceeds row limit")
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            if not isinstance(row, dict):
                raise ValueError("JSONL row must be an object at line {}".format(line_number))
            rows.append(row)
    return rows


def calibrate(rows, *, confidence=0.95):
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and in (0,1)")
    values = []
    ray_counts = set()
    verified_no_return_frame_count = 0
    for row in rows:
        if row.get("policy_label") != "bc":
            raise ValueError("point-cloud calibration accepts BC records only")
        value = float(row["pointcloud_no_ray_hit_fraction"])
        hits = int(row["pointcloud_valid_hit_rays"])
        total = int(row["pointcloud_total_rays"])
        fallback = row.get("pointcloud_used_far_plane_fallback", False)
        if not isinstance(fallback, bool):
            raise ValueError("pointcloud fallback flag must be boolean")
        if (
            total < 1 or not 0 <= hits <= total or not 0.0 <= value <= 1.0
            or not math.isclose(value, 1.0 - hits / total, abs_tol=1e-12)
            or (hits == 0) != fallback
        ):
            raise ValueError("calibration bank contains an invalid point-cloud frame")
        if hits == 0:
            verified_no_return_frame_count += 1
        values.append(value)
        ray_counts.add(total)
    if len(values) < 1024:
        raise ValueError("point-cloud calibration requires at least 1024 BC frames")
    if len(ray_counts) != 1:
        raise ValueError("calibration bank mixes different ray-grid sizes")
    total_rays = ray_counts.pop()
    observed_max = float(max(values))
    # A sample maximum is not a safe family-wise hard gate for a much longer
    # training run: with n calibration frames and m future frames, an IID
    # exchangeability calculation gives P(no future value exceeds the observed
    # max) = n/(n+m).  Keep the BC quantiles as drift evidence, while the hard
    # threshold encodes only the physical validity requirement of >=1 ray hit.
    threshold = float(1.0 - 1.0 / total_rays)
    # For an IID validation sample, its maximum is a distribution-free
    # one-sided tolerance bound with content alpha**(1/n) at confidence 1-alpha.
    alpha = 1.0 - confidence
    content_lower_bound = float(alpha ** (1.0 / len(values)))
    return {
        "schema": "routea_v17_pointcloud_calibration_v2",
        "method": "physical_at_least_one_hit_gate_with_bc_reference_quantiles",
        "bc_frame_count": len(values),
        "total_rays_per_frame": total_rays,
        "confidence": confidence,
        "content_lower_bound": content_lower_bound,
        "max_no_ray_hit_fraction": threshold,
        "max_observed_no_ray_hit_fraction": observed_max,
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "verified_no_return_frame_count": verified_no_return_frame_count,
        "all_no_ray_policy": "accept_verified_max_range_censoring",
        "all_no_ray_always_rejected": False,
        "partial_no_ray_action": "monitor_only",
        "sample_max_not_used_as_hard_gate": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-transitions", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = load_transition_rows(args.bc_transitions)
    result = calibrate(rows)
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    names = [template.name for template in args.template_config]
    if len(names) != len(set(names)):
        raise ValueError("template config basenames must be unique")
    for template in args.template_config:
        if template.is_symlink() or not template.is_file():
            raise ValueError("template config must be a regular non-symlink file")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "pointcloud_calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for template in args.template_config:
        config = OmegaConf.load(template)
        config.environment.max_no_ray_hit_fraction = result["max_no_ray_hit_fraction"]
        config.environment.reject_all_no_ray_hit = False
        config.environment.pointcloud_calibration_sha256 = result["content_sha256"]
        destination = args.output_dir / template.name
        OmegaConf.save(config, destination)
    print(json.dumps({"status": "CALIBRATION_FROZEN", "threshold": result["max_no_ray_hit_fraction"], "configs": len(args.template_config)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
