"""RLSystemContract construction and hard-assertion tests (design §4.3)."""

import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract


def test_contract_defaults():
    """Default contract encodes design §2 symbol table."""
    c = RLSystemContract()
    assert c.history_steps == 8
    assert c.obs_steps == 3
    assert c.point_count == 4096
    assert c.point_dim == 3
    assert c.state_dim == 9
    assert c.full_horizon == 16
    assert c.action_start == 2
    assert c.action_horizon == 8
    assert c.exec_steps == 4  # de-facto value, EQUIV-REG OPEN
    assert c.action_dim == 11
    assert c.denoising_steps == 10
    assert c.ft_denoising_steps == 5
    assert c.gamma_env == 0.99
    assert c.gae_lambda == 0.95
    assert c.gamma_denoising == 0.8


def test_action_start_is_obs_steps_minus_one():
    """action_start == obs_steps - 1 is a hard design invariant (§2.3/§4.3)."""
    with pytest.raises(AssertionError):
        RLSystemContract(obs_steps=3, action_start=3)


def test_action_region_within_full_horizon():
    """action_start + action_horizon <= full_horizon (design §4.3)."""
    with pytest.raises(AssertionError):
        RLSystemContract(action_horizon=15)


def test_exec_steps_bounds():
    """1 <= exec_steps <= action_horizon (design §4.3)."""
    with pytest.raises(AssertionError):
        RLSystemContract(exec_steps=0)
    with pytest.raises(AssertionError):
        RLSystemContract(exec_steps=9)


def test_action_dim_must_be_11():
    """11d action is a verified global constraint (CLAUDE.md §2)."""
    with pytest.raises(AssertionError):
        RLSystemContract(action_dim=10)


def test_obs_steps_leq_history_steps():
    with pytest.raises(AssertionError):
        RLSystemContract(obs_steps=9)


def test_contract_is_frozen():
    c = RLSystemContract()
    with pytest.raises(Exception):
        c.action_horizon = 4  # type: ignore[misc]


def test_ft_denoising_steps_bounds():
    """1 <= K_f <= K is a natural invariant (review-fix)."""
    with pytest.raises(AssertionError):
        RLSystemContract(ft_denoising_steps=0)
    with pytest.raises(AssertionError):
        RLSystemContract(ft_denoising_steps=11)
