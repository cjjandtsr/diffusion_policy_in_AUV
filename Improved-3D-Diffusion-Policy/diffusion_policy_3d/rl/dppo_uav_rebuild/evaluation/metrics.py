"""Paired deltas and deterministic bootstrap confidence intervals."""

from dataclasses import dataclass
import math

import numpy as np

from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import TRAINABLE_ROLES


@dataclass(frozen=True)
class DeltaCI:
    mean: float
    lower: float
    upper: float
    pairs: int


def bootstrap_delta(values, *, seed=20260827, samples=10000, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired bootstrap requires at least two scalar deltas")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return DeltaCI(
        mean=float(values.mean()),
        lower=float(np.quantile(means, alpha / 2)),
        upper=float(np.quantile(means, 1 - alpha / 2)),
        pairs=len(values),
    )


def paired_metrics(bank):
    pairs = bank.pairs
    deltas = {
        "success": [float(rl.success) - float(bc.success) for bc, rl in pairs],
        "collision": [float(rl.collision) - float(bc.collision) for bc, rl in pairs],
        "return": [rl.episode_return - bc.episode_return for bc, rl in pairs],
        "onset_distance": [
            rl.avoidance_onset_distance - bc.avoidance_onset_distance for bc, rl in pairs
        ],
        "support_cells": [rl.support_cells - bc.support_cells for bc, rl in pairs],
    }
    result = {name: bootstrap_delta(value) for name, value in deltas.items()}
    for role in TRAINABLE_ROLES:
        selected = [(bc, rl) for bc, rl in pairs if bc.role == role]
        if len(selected) >= 2:
            result[role + "_success"] = bootstrap_delta(
                [float(rl.success) - float(bc.success) for bc, rl in selected]
            )
    endpoint_pairs = [
        (bc, rl) for bc, rl in pairs
        if math.isfinite(bc.final_distance_3d) and math.isfinite(rl.final_distance_3d)
    ]
    if len(endpoint_pairs) >= 2:
        for name in ("final_distance_3d", "final_error_xy", "final_error_z"):
            result[name] = bootstrap_delta([
                float(getattr(rl, name)) - float(getattr(bc, name))
                for bc, rl in endpoint_pairs
            ])
    return result
