#!/usr/bin/env python3
"""Q2 export: flat CSV of every sampled draw (objective data, no statistics).

Columns: mode, policy, state_id, group, draw, seed, initial_noise_hash,
h4_pos_x/y/z, h4_yaw, h8_pos_x/y/z, h8_yaw.
"""
import csv
import json
import sys
from pathlib import Path

OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
               "/home/cjj/AUV/program/results/routea_paired_evaluation/"
               "routea_q2_samestate_dispersion_20261025_run01")
EXPORT = Path(sys.argv[2] if len(sys.argv) > 2 else
              "/home/cjj/AUV/plan/q2q3q4_20261025/q2_samples_flat.csv")
MODES = ("deterministic", "random")
POLICIES = ("bc", "rl_391a6656a0f7_u128", "rl_391a6656a0f7_u256")
FIELDS = ["mode", "policy", "state_id", "group", "draw", "seed",
          "initial_noise_hash",
          "h4_pos_x", "h4_pos_y", "h4_pos_z", "h4_yaw",
          "h8_pos_x", "h8_pos_y", "h8_pos_z", "h8_yaw"]


def main():
    rows = []
    for mode in MODES:
        for policy in POLICIES:
            pdir = OUT_DIR / mode / policy
            if not pdir.is_dir():
                continue
            for state_file in sorted(pdir.glob("*.json")):
                if state_file.name == "bc_disabled_branch_check.json":
                    continue
                record = json.loads(state_file.read_text())
                for sample in record["samples"]:
                    rows.append({
                        "mode": mode,
                        "policy": policy,
                        "state_id": record["state_id"],
                        "group": record.get("group"),
                        "draw": sample["draw"],
                        "seed": sample["seed"],
                        "initial_noise_hash": sample["initial_noise_hash"],
                        "h4_pos_x": sample["h4_pos"][0],
                        "h4_pos_y": sample["h4_pos"][1],
                        "h4_pos_z": sample["h4_pos"][2],
                        "h4_yaw": sample["h4_yaw"],
                        "h8_pos_x": sample["h8_pos"][0],
                        "h8_pos_y": sample["h8_pos"][1],
                        "h8_pos_z": sample["h8_pos"][2],
                        "h8_yaw": sample["h8_yaw"],
                    })
    with EXPORT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("rows:", len(rows))
    print("saved:", EXPORT)


if __name__ == "__main__":
    main()
