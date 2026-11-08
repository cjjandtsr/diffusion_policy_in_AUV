"""System contract — single source of truth for A-Official-Core dimensions.

Design: RouteA_RL_Training_System_A_OfficialCore_Design_v1_0 §4.3.
Symbols follow design §2: T_h=8, T_o=3, N=4096, S=9, H_f=16, a0=2,
H_a=8, A_e=exec_steps, D_a=11, K=10, K_f=5, gamma_env=0.99, lambda=0.95,
gamma_denoising=0.8.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RLSystemContract:
    """Frozen dimensional contract; every downstream module reads from here.

    Construct-time hard assertions enforce the design invariants (§4.3).
    """

    history_steps: int = 8
    obs_steps: int = 3
    point_count: int = 4096
    point_dim: int = 3
    state_dim: int = 9
    full_horizon: int = 16
    action_start: int = 2  # obs_steps - 1
    action_horizon: int = 8
    exec_steps: int = 4
    action_dim: int = 11
    denoising_steps: int = 10
    ft_denoising_steps: int = 5
    gamma_env: float = 0.99
    gae_lambda: float = 0.95
    gamma_denoising: float = 0.8
    # §7.4: obstacle projection is policy-external (KD-Tree
    # trajectory rewriting); formal training/eval must run with False and the
    # flag is part of the contract (checkpoint hash + eval artifact).
    use_obstacle_projection: bool = False

    def __post_init__(self) -> None:
        assert self.history_steps >= self.obs_steps >= 1, (
            f"history_steps={self.history_steps} >= obs_steps={self.obs_steps} >= 1"
        )
        assert self.action_start == self.obs_steps - 1, (
            f"action_start={self.action_start} must equal obs_steps-1"
            f"={self.obs_steps - 1}"
        )
        assert (
            self.action_start + self.action_horizon <= self.full_horizon
        ), f"action region {self.action_start}:{self.action_start + self.action_horizon} exceeds full_horizon={self.full_horizon}"
        assert 1 <= self.exec_steps <= self.action_horizon, (
            f"exec_steps={self.exec_steps} must satisfy 1 <= exec_steps"
            f" <= action_horizon={self.action_horizon}"
        )
        assert self.action_dim == 11, (
            f"action_dim={self.action_dim} must be 11 (verified global constraint)"
        )
        assert 1 <= self.ft_denoising_steps <= self.denoising_steps, (
            f"ft_denoising_steps={self.ft_denoising_steps} must satisfy"
            f" 1 <= ft_denoising_steps <= denoising_steps={self.denoising_steps}"
        )
        assert not self.use_obstacle_projection, (
            "use_obstacle_projection must be False (§7.4): KD-Tree obstacle"
            " projection is policy-external trajectory rewriting — formal"
            " training/eval hard-fails on True"
        )
