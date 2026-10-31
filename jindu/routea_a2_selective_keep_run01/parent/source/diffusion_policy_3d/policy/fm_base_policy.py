"""
fm_base_policy.py — FM 黑盒适配层 (Route B DSRL-SAC)

把冻结 FM 包成 DSRL DiffusionPolicyEnvWrapper 期望的 base_policy 接口:
    base_policy(obs, noise_actions) -> diffused_actions

职责:
  1. obs → global_cond (冻结 encoder)
  2. z → init_noise 注入 conditional_sample_fm
  3. unnormalize + 取 action chunk
"""

import torch
import numpy as np
from typing import Dict


class FMBasePolicy:
    """FM 作为 DSRL base_policy 的黑盒适配器"""

    def __init__(self, fm_policy, device: str = 'cuda:0'):
        self.fm = fm_policy
        self.device = torch.device(device)
        self.fm.to(self.device)
        self.fm.eval()

        # 冻结全部 FM 参数
        for p in self.fm.parameters():
            p.requires_grad_(False)

        # ── RL 投影配置 (Route B 实现指导 §0.5) ──
        self.fm.use_yaw_projection = False    # 删 dyaw 回拉：RL 自己学恢复
        self.fm.use_z_clamp = True            # 留 safety envelope [1.0, 5.0]
        self.fm.use_obstacle_projection = False  # 关障碍投影

        self.horizon = self.fm.horizon              # 16
        self.action_dim = self.fm.action_dim          # 11
        self.n_action_steps = self.fm.n_action_steps  # 8
        self.n_obs_steps = self.fm.n_obs_steps        # 8
        self.obs_dim = self.fm.obs_encoder.output_shape() * self.n_obs_steps  # 1536

    @torch.no_grad()
    def encode_obs(self, obs: Dict) -> np.ndarray:
        """编码 raw obs → global_cond (供 SAC 作为 observation)

        Args:
            obs: {'agent_pos': (n_obs_steps, 9), 'point_cloud': (n_obs_steps, 4096, 3)}

        Returns:
            global_cond: (1536,) float32 numpy array
        """
        # 加 batch 维
        obs_batched = {
            k: np.expand_dims(v, 0) if isinstance(v, np.ndarray)
               else v.unsqueeze(0)
            for k, v in obs.items()
        }
        nobs = self.fm.normalizer.normalize(obs_batched)
        nobs = {k: torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray)
                else v.to(self.device)
                for k, v in nobs.items()}

        this_nobs = {}
        for k, v in nobs.items():
            this_nobs[k] = v[:, :self.n_obs_steps, ...].reshape(-1, *v.shape[2:])

        global_cond = self.fm.obs_encoder(this_nobs).reshape(1, -1)  # (1, 1536)
        return global_cond.cpu().numpy().squeeze(0)  # (1536,)

    @torch.no_grad()
    def __call__(self, obs: Dict, noise_actions: torch.Tensor) -> np.ndarray:
        """
        DSRL base_policy 接口适配

        Args:
            obs: gym 环境传来的观测 dict
                 {'agent_pos': (n_envs, n_obs_steps, 9),
                  'point_cloud': (n_envs, n_obs_steps, 4096, 3)}
            noise_actions: SAC actor 输出的 z, shape (n_envs, action_horizon * action_dim)
                           = (n_envs, 176)  [16×11]

        Returns:
            action_pred: (n_envs, n_action_steps, action_dim) = (n_envs, 8, 11)
                         unnormalized world-frame action chunk
        """
        B = noise_actions.shape[0]
        Da = self.action_dim
        T = self.horizon

        # 1) obs → global_cond (冻结 encoder)
        # 加 batch 维 (wrapper 传入单样本 dict)
        obs_batched = {
            k: np.expand_dims(v, 0) if isinstance(v, np.ndarray)
               else v.unsqueeze(0)
            for k, v in obs.items()
        }
        nobs = self.fm.normalizer.normalize(obs_batched)
        nobs = {k: torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray)
                else v.to(self.device)
                for k, v in nobs.items()}

        # encoder 需要 (B * n_obs_steps, ...) 的输入
        this_nobs = {}
        for k, v in nobs.items():
            this_nobs[k] = v[:, :self.n_obs_steps, ...].reshape(-1, *v.shape[2:])

        global_cond = self.fm.obs_encoder(this_nobs).reshape(B, -1)  # (B, 1536)

        # 2) CFG unconditional (保持部署 λ=1.5)
        global_cond_uncond = None
        if self.fm.cfg_lambda > 0:
            nobs_u = {k: v.clone() for k, v in nobs.items()}
            nobs_u['agent_pos'][:, :, 3:7] = 0.0  # 抹 goal_body + sin/cos yaw
            this_u = {}
            for k, v in nobs_u.items():
                this_u[k] = v[:, :self.n_obs_steps, ...].reshape(-1, *v.shape[2:])
            global_cond_uncond = self.fm.obs_encoder(this_u).reshape(B, -1)

        # 3) z → condition_data 形状 + init_noise 注入
        z = torch.as_tensor(noise_actions, device=self.device).view(B, T, Da)  # (B, 16, 11)

        cond_data = torch.zeros(B, T, Da, device=self.device, dtype=self.fm.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # 4) 调 conditional_sample_fm (z 作为 init_noise)
        nsample = self.fm.conditional_sample_fm(
            cond_data, cond_mask,
            local_cond=None,
            global_cond=global_cond,
            global_cond_uncond=global_cond_uncond,
            anchor_idx=None,
            init_noise=z,               # ★Route B: z 替换随机噪声
        )

        # 5) unnormalize + 取 n_action_steps chunk
        naction_pred = nsample[..., :Da]
        action_pred = self.fm.normalizer['action'].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]  # (B, 8, 11)

        return action.cpu().numpy()

    def to(self, device):
        self.device = torch.device(device)
        self.fm.to(self.device)
        return self
