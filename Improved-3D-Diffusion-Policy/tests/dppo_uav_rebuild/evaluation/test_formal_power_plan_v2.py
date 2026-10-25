import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.evaluation.statistics import formal_seed_count


def test_m3_uses_low_tail_chi_square_quantile():
    plan = formal_seed_count([0.0, 0.05, 0.10])
    assert plan.chi_square_probability == pytest.approx(0.20)
    assert plan.sigma_upper_multiplier == pytest.approx(2.117, rel=1e-3)
    assert plan.n_formal >= 5


def test_zero_sd_uses_floor():
    assert formal_seed_count([0.02, 0.02, 0.02]).sigma_used == pytest.approx(0.05)
