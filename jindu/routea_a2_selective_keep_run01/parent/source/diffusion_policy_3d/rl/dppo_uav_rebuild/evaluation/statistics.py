"""Pre-registered seed-level inference and formal-run power planning."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import chi2, norm, t

from .metrics import DeltaCI


@dataclass(frozen=True)
class PowerPlan:
    pilot_seed_count: int
    pilot_sd: float
    sigma_floor: float
    sigma_upper_multiplier: float
    sigma_used: float
    chi_square_probability: float
    effect: float
    nominal_power: float
    n_formal: int


def seed_level_t_interval(seed_deltas: Sequence[float], alpha: float = 0.05) -> DeltaCI:
    values = np.asarray(seed_deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("seed-level t interval requires at least two finite deltas")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(len(values)))
    radius = float(t.ppf(1.0 - alpha / 2.0, df=len(values) - 1) * sem)
    return DeltaCI(mean=mean, lower=mean - radius, upper=mean + radius, pairs=len(values))


def holm_adjust(p_values: Mapping[str, float], alpha: float = 0.05):
    ordered = sorted((float(value), str(name)) for name, value in p_values.items())
    decisions = {name: False for name in p_values}
    for index, (value, name) in enumerate(ordered):
        threshold = alpha / (len(ordered) - index)
        if value > threshold:
            break
        decisions[name] = True
    return decisions


def studentized_hierarchical_bootstrap(records, *, samples=10000, seed=20260910):
    """Sensitivity CI: resample seed, then task/world, while retaining BC/RL pairs."""

    if samples < 10000:
        raise ValueError("formal hierarchical bootstrap requires at least 10000 samples")
    grouped = {}
    for record in records:
        grouped.setdefault(int(record["training_seed"]), {}).setdefault(
            (record["task_id"], record["world_id"]), {}
        ).setdefault(record["pair_id"], {})[record["policy"]] = float(record["success"])
    seeds = sorted(grouped)
    if len(seeds) < 2:
        raise ValueError("hierarchical bootstrap requires at least two training seeds")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        seed_means = []
        for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
            strata = grouped[int(sampled_seed)]
            stratum_keys = tuple(strata)
            deltas = []
            for key in rng.choice(len(stratum_keys), size=len(stratum_keys), replace=True):
                pairs = strata[stratum_keys[int(key)]]
                pair_ids = tuple(pairs)
                for pair_index in rng.choice(len(pair_ids), size=len(pair_ids), replace=True):
                    pair = pairs[pair_ids[int(pair_index)]]
                    if set(pair) != {"bc", "rl"}:
                        raise ValueError("bootstrap input contains an incomplete BC/RL pair")
                    deltas.append(pair["rl"] - pair["bc"])
            seed_means.append(float(np.mean(deltas)))
        estimates.append(float(np.mean(seed_means)))
    point = []
    for seed_value in seeds:
        pairs = grouped[seed_value]
        delta = [
            policies["rl"] - policies["bc"]
            for stratum in pairs.values()
            for policies in stratum.values()
        ]
        point.append(float(np.mean(delta)))
    return DeltaCI(
        mean=float(np.mean(point)),
        lower=float(np.quantile(estimates, 0.025)),
        upper=float(np.quantile(estimates, 0.975)),
        pairs=len(seeds),
    )


def formal_seed_count(
    pilot_seed_deltas,
    *,
    effect=0.05,
    sigma_upper_confidence=0.80,
    sigma_floor=0.05,
    nominal_power=0.80,
):
    values = np.asarray(pilot_seed_deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("power planning requires at least three finite pilot seed deltas")
    if sigma_upper_confidence != 0.80:
        raise ValueError("the registered plan fixes the variance upper confidence at 0.80")
    probability = 1.0 - sigma_upper_confidence
    pilot_sd = float(values.std(ddof=1))
    multiplier = math.sqrt((len(values) - 1) / chi2.ppf(probability, len(values) - 1))
    sigma_used = max(sigma_floor, pilot_sd * multiplier)
    z_sum = norm.ppf(0.975) + norm.ppf(nominal_power)
    n_formal = max(5, int(math.ceil((z_sum * sigma_used / effect) ** 2)))
    return PowerPlan(
        pilot_seed_count=len(values), pilot_sd=pilot_sd, sigma_floor=sigma_floor,
        sigma_upper_multiplier=float(multiplier), sigma_used=float(sigma_used),
        chi_square_probability=probability, effect=float(effect),
        nominal_power=float(nominal_power), n_formal=n_formal,
    )
