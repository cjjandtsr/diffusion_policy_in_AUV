#!/usr/bin/env python3
"""One offline verification command for the DDPM RL rebuild."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _run(command, cwd):
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "verification command failed ({}): {}".format(
                completed.returncode, " ".join(str(item) for item in command)
            )
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--staging-overlay", type=Path)
    parser.add_argument("--rl-config", type=Path)
    parser.add_argument("--rl-checkpoint", type=Path)
    args = parser.parse_args(argv)
    if (args.rl_config is None) != (args.rl_checkpoint is None):
        parser.error("--rl-config and --rl-checkpoint must be supplied together")
    repo = Path(__file__).resolve().parents[1]
    vendor = repo / "third_party" / "dppo_official_cc7234a"
    python = sys.executable

    _run([python, "-m", "scripts.verify_dppo_vendor_snapshot", str(vendor)], repo)
    _run(
        [
            python,
            "-m",
            "pytest",
            "tests/dppo_uav_rebuild",
            "-q",
            "--ignore",
            "tests/dppo_uav_rebuild/official_core/test_e5_official_runtime.py",
            "-k",
            "not test_official_symbols_resolve_inside_the_verified_vendor",
        ],
        repo,
    )
    _run(
        [
            python,
            "scripts/verify_e5_identity_config.py",
            "diffusion_policy_3d/rl/dppo_uav_rebuild/config/e5_bc_identity.yaml",
        ],
        repo,
    )
    real_status = "NOT_RUN_CHECKPOINT_REQUIRED"
    if args.checkpoint is not None:
        common = []
        if args.staging_overlay is not None:
            common = ["--staging-overlay", str(args.staging_overlay)]
        _run(
            [
                python,
                "scripts/verify_e5_lora_inventory.py",
                str(args.checkpoint),
                "--device",
                args.device,
            ]
            + common,
            repo,
        )
        _run(
            [
                python,
                "scripts/verify_e5_system_build.py",
                str(args.checkpoint),
                str(vendor),
                "--device",
                args.device,
                "--staging-overlay",
                str(args.staging_overlay or repo),
            ],
            repo,
        )
        _run(
            [
                python,
                "scripts/verify_e5_official_parity.py",
                str(args.checkpoint),
                str(vendor),
                "--device",
                args.device,
                "--staging-overlay",
                str(args.staging_overlay or repo),
            ],
            repo,
        )
        real_status = "PASS"
    resume_status = "NOT_RUN_RL_CHECKPOINT_REQUIRED"
    if args.rl_checkpoint is not None:
        _run(
            [
                python,
                "scripts/verify_dppo_checkpoint_resume.py",
                "--config",
                str(args.rl_config),
                "--checkpoint",
                str(args.rl_checkpoint),
            ],
            repo,
        )
        resume_status = "PASS"
    print(
        json.dumps(
            {
                "local_contracts": "PASS",
                "network_used": False,
                "real_e5_parity": real_status,
                "real_rl_checkpoint_resume": resume_status,
                "real_gpu_engine": "NOT_RUN_TRAINING_REQUIRED",
                "gazebo_capability": "NOT_RUN_GAZEBO_REQUIRED",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
