#!/usr/bin/env python3
"""Build the immutable fresh-seed formal plan from three completed pilots."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from diffusion_policy_3d.rl.dppo_uav_rebuild.evaluation.statistics import formal_seed_count


PILOT_SEEDS = (202609101, 202609102, 202609103)
FORMAL_SEED_START = 202609201


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON constant: {}".format(value))


def build_formal_plan(pilot_rows, *, available_seed_budget=None):
    rows = tuple(pilot_rows)
    if tuple(sorted(int(row["training_seed"]) for row in rows)) != PILOT_SEEDS:
        raise ValueError("pilot rows must contain exactly the three registered pilot seeds")
    deltas = [float(row["safe_arrival_delta"]) for row in sorted(rows, key=lambda row: int(row["training_seed"]))]
    power = formal_seed_count(deltas)
    seeds = tuple(range(FORMAL_SEED_START, FORMAL_SEED_START + power.n_formal))
    underpowered = available_seed_budget is not None and int(available_seed_budget) < power.n_formal
    payload = {
        "schema": "routea_v17_formal_power_plan_v2",
        "status": "UNDERPOWERED_FORMAL" if underpowered else "PLAN_READY",
        "pilot_seeds": list(PILOT_SEEDS),
        "pilot_safe_arrival_deltas": deltas,
        "power": asdict(power),
        "fresh_formal_seeds": list(seeds),
        "available_seed_budget": available_seed_budget,
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--available-seed-budget", type=int)
    args = parser.parse_args(argv)
    if args.pilot.is_symlink() or not args.pilot.is_file():
        raise ValueError("pilot result must be a regular non-symlink file")
    if args.pilot.stat().st_size > 1024 * 1024:
        raise ValueError("pilot result input exceeds 1 MiB")
    if args.available_seed_budget is not None and args.available_seed_budget < 0:
        raise ValueError("available seed budget must be non-negative")
    rows = json.loads(
        args.pilot.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(rows, list):
        raise ValueError("pilot result root must be a list")
    plan = build_formal_plan(rows, available_seed_budget=args.available_seed_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": plan["status"], "n_formal": plan["power"]["n_formal"]}, sort_keys=True))
    return 2 if plan["status"] == "UNDERPOWERED_FORMAL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
