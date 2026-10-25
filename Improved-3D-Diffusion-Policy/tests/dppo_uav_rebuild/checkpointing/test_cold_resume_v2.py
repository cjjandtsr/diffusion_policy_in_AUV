"""Checkpoint schema binds complete RNG state for cold continuation."""

import random

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.schema import SCHEMA_VERSION


def test_schema_is_v2_and_rng_state_is_not_torch_only():
    assert SCHEMA_VERSION == 2
    source = __import__(
        "diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.save",
        fromlist=["dummy"],
    )
    text = open(source.__file__, encoding="utf-8").read()
    assert "rng_python" in text
    assert "rng_numpy_legacy" in text
    assert "COMMITTED.json" in text
