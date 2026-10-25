import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.evaluation.statistics import (
    holm_adjust, seed_level_t_interval,
)


def test_seed_interval_counts_seeds_not_episodes():
    result = seed_level_t_interval([0.0, 0.05, 0.1, 0.05, 0.0])
    assert result.pairs == 5
    assert result.mean == pytest.approx(0.04)


def test_holm_stops_after_first_non_rejection():
    result = holm_adjust({"a": 0.001, "b": 0.03, "c": 0.04})
    assert result == {"a": True, "b": False, "c": False}

