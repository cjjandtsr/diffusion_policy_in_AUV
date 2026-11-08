#!/usr/bin/env python3
"""Validate the declarative E5 identity file with the existing OmegaConf."""

import argparse

from omegaconf import OmegaConf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    assert cfg.role == "expected_identity_only"
    assert cfg.checkpoint.weight_source == "model"
    assert cfg.observation.policy_history == 3
    assert cfg.observation.policy_condition_dim == 576
    assert cfg.action.deployment_exec_horizon == 2
    assert cfg.action.rollout_exec_horizon == 4
    assert cfg.diffusion.prediction_type == "sample"
    assert cfg.diffusion.scheduler_step_accepts_arbitrary_kwargs is False
    print("e5_identity_config=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
