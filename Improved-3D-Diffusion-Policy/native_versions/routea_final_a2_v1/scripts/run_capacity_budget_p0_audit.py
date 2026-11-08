#!/usr/bin/env python3
"""run_capacity_budget_p0_audit.py — CLI entry for P0 capacity budget audit.

Loads seven sub-report JSONs from the output directory, runs the aggregator,
and prints/saves the total-gate decision.

Usage:
    python scripts/run_capacity_budget_p0_audit.py \\
        --output_dir <path> [--run_id <str>]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.capacity_budget_p0 import run_capacity_budget_p0_audit


def main():
    ap = argparse.ArgumentParser(
        description="P0 capacity budget audit — aggregate seven sub-reports "
                    "into a total-gate decision."
    )
    ap.add_argument(
        "--output_dir", required=True,
        help="Path to the P0 output directory containing sub-report JSONs."
    )
    ap.add_argument(
        "--run_id", default=None,
        help="Run identifier (default: derived from output_dir basename)."
    )
    args = ap.parse_args()

    config = {"output_dir": args.output_dir}
    if args.run_id:
        config["run_id"] = args.run_id

    result = run_capacity_budget_p0_audit(config)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    gate_path = Path(args.output_dir) / "p0_gate_decision.json"
    print(f"\nSaved: {gate_path}")

    has_refusal = "refused_pass_reason" in result
    if has_refusal:
        print(f"REFUSED: {result['refused_pass_reason']}")
        sys.exit(2)

    verdict = "PASS" if result["overall_pass"] else "FAIL"
    print(f"VERDICT: {verdict}")
    sys.exit(0 if result["overall_pass"] else 1)


if __name__ == "__main__":
    main()
