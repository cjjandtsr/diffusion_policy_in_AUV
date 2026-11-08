"""
dsrl_wrapper.py — DSRL-SAC 噪声空间 env wrapper (Route B, v2 low-rank + validity)

把真实 action space (8,11) 换成 latent space (latent_rank,) 或 噪声空间 (176,),
observation 用冻结 FM encoder 编码为 global_cond + validity feature.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple


class NoiseActionWrapper(gym.Wrapper):
    """DSRL-SAC wrapper (单 env 版, v2)

    Low-rank mode: SAC actor -> u(latent_rank) -> z(176) = sigma * P @ u -> FM -> real action
    Full mode:     SAC actor -> z(176) -> FM -> real action

    Observation: global_cond(1536) + validity(4) = 1540d
    """

    def __init__(
        self,
        env: gym.Env,
        base_policy,                # FMBasePolicy: (obs_dict, z) -> action(8,11)
        action_magnitude: float = 1.5,
        action_horizon: int = 16,
        action_dim: int = 11,
        obs_mean: np.ndarray = None,
        obs_std: np.ndarray = None,
        # ── low-rank latent steering ──
        use_lowrank_latent: bool = True,
        latent_rank: int = 16,
        latent_sigma: float = 0.5,
        projection_seed: int = 20260627,
        # ── validity observation ──
        append_validity_features: bool = True,
    ):
        super().__init__(env)
        self.base_policy = base_policy
        self.action_magnitude = action_magnitude
        self.action_horizon = action_horizon       # 16
        self.action_dim = action_dim               # 11
        self.noise_dim = action_horizon * action_dim  # 176
        self.obs_dim = base_policy.obs_dim         # 1536 (global_cond)

        # ── global_cond 归一化 ──
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32) if obs_mean is not None else None
        self.obs_std = np.asarray(obs_std, dtype=np.float32) if obs_std is not None else None
        if self.obs_mean is not None:
            assert self.obs_mean.shape == (self.obs_dim,), \
                f"obs_mean shape {self.obs_mean.shape} != ({self.obs_dim},)"
        if self.obs_std is not None:
            assert self.obs_std.shape == (self.obs_dim,), \
                f"obs_std shape {self.obs_std.shape} != ({self.obs_dim},)"

        # ── low-rank projection ──
        self.use_lowrank_latent = use_lowrank_latent
        self.latent_rank = latent_rank
        self.latent_sigma = latent_sigma
        self.projection_seed = projection_seed
        self.P = None  # shape (176, latent_rank)
        if use_lowrank_latent:
            assert latent_rank <= self.noise_dim
            rng = np.random.RandomState(projection_seed)
            P_raw = rng.randn(self.noise_dim, latent_rank).astype(np.float32)
            self.P, _ = np.linalg.qr(P_raw)  # orthogonal columns
            assert self.P.shape == (self.noise_dim, latent_rank)

        # ── validity ──
        self.append_validity_features = append_validity_features
        self.validity_dim = 4 if append_validity_features else 0
        self.total_obs_dim = self.obs_dim + self.validity_dim  # 1536+4=1540 or 1536

        # ── action space ──
        act_dim = latent_rank if use_lowrank_latent else self.noise_dim
        self.action_space = spaces.Box(
            low=-action_magnitude * (latent_sigma if use_lowrank_latent else 1.0),
            high=action_magnitude * (latent_sigma if use_lowrank_latent else 1.0),
            shape=(act_dim,), dtype=np.float32)

        # ── observation space ──
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.total_obs_dim,), dtype=np.float32)

        # 记录真实 action space (供诊断)
        self.real_action_space = env.action_space

    # ═══════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════

    def _normalize_obs(self, encoded: np.ndarray) -> np.ndarray:
        if self.obs_mean is not None and self.obs_std is not None:
            return (encoded - self.obs_mean) / (self.obs_std + 1e-8)
        return encoded

    def _u_to_z(self, u: np.ndarray) -> np.ndarray:
        """low-rank u -> full 176d z"""
        if self.use_lowrank_latent:
            u_flat = u.reshape(-1)
            z = self.latent_sigma * (self.P @ u_flat)
        else:
            z = u.reshape(-1)
        return np.clip(z, -self.action_magnitude, self.action_magnitude).astype(np.float32)

    def _compute_validity(self, raw_obs: dict, info: dict) -> np.ndarray:
        """从 raw obs + info 计算 4d validity feature"""
        pc = raw_obs['point_cloud']  # (n_obs_steps, 4096, 3) or (4096, 3)
        if pc.ndim == 3:
            pc = pc[-1]  # latest frame
        pc_flat = pc.reshape(-1)
        pc_norm = float(np.linalg.norm(pc_flat) / np.sqrt(max(pc_flat.size, 1)))
        pc_nonzero = float(np.mean(np.abs(pc_flat) > 1e-6))
        sensor_valid = 1.0 if (pc_nonzero > 0.01 and pc_norm > 1e-4) else 0.0
        z_world = float(info.get('z_world', 0.0))
        out_of_support = 1.0 if z_world > 4.15 else 0.0
        return np.array([sensor_valid, pc_nonzero, pc_norm, out_of_support], dtype=np.float32)

    def _build_obs(self, raw_obs: dict, info: dict) -> np.ndarray:
        """raw obs -> global_cond + validity -> normalized observation"""
        encoded = self.base_policy.encode_obs(raw_obs)  # (1536,)
        normed = self._normalize_obs(encoded)
        if self.append_validity_features:
            validity = self._compute_validity(raw_obs, info)
            return np.concatenate([normed, validity]).astype(np.float32)
        return normed.astype(np.float32)

    # ═══════════════════════════════════════════════════════════
    # Gym API
    # ═══════════════════════════════════════════════════════════

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """SAC u -> z -> FM -> real action -> env -> obs

        Args:
            action: (latent_rank,) or (176,) depending on use_lowrank_latent
        """
        # 1) u -> z
        z = self._u_to_z(action)  # (176,)
        z_batched = z.reshape(1, self.noise_dim)

        # 2) Get raw obs -> z -> FM -> real action
        raw_obs = self.unwrapped._get_obs_dict()
        real_action = self.base_policy(raw_obs, z_batched)  # (1, 8, 11)

        # 3) Env step
        raw_obs_next, reward, terminated, truncated, info = self.env.step(real_action[0])
        info['z_latent'] = action if self.use_lowrank_latent else None
        info['z_full'] = z

        # 4) Build observation
        obs_rl = self._build_obs(raw_obs_next, info)
        return obs_rl, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        raw_obs, info = self.env.reset(seed=seed, options=options)
        info['z_world'] = float(self.unwrapped._drone_pos[2])
        obs_rl = self._build_obs(raw_obs, info)
        return obs_rl, info

    # ── metadata for model saving ──
    def get_config(self) -> dict:
        return dict(
            use_lowrank_latent=self.use_lowrank_latent,
            latent_rank=self.latent_rank,
            latent_sigma=self.latent_sigma,
            projection_seed=self.projection_seed,
            noise_dim=self.noise_dim,
            obs_dim=self.obs_dim,
            total_obs_dim=self.total_obs_dim,
            append_validity_features=self.append_validity_features,
        )
