"""
phase1a_policy_prevcond.py — Phase 1a + Previous Trajectory Explicit Conditioning
================================================================================
继承 Phase1aPolicy，增加前次轨迹显式条件化机制。

核心改动:
  1. global_cond_dim: 576 → 640 (追加 64 维 prev_traj_feat)
  2. 新增 PrevTrajEncoder (Conv1D, ~5K 参数)
  3. 新增 null_prev_embed (可学习, 64 维)
  4. predict_action / compute_loss 注入 prev_traj 条件

设计依据: 2026-05-25-prev-traj-cond-design.md v1.3
"""

import time
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from termcolor import cprint

from diffusion_policy_3d.policy.phase1a_policy import Phase1aPolicy
from diffusion_policy_3d.model.common.prev_traj_encoder import PrevTrajEncoder
from diffusion_policy_3d.common.pytorch_util import dict_apply


class Phase1aPrevCondPolicy(Phase1aPolicy):
    """
    Phase 1a + 前次轨迹显式条件化。

    新增参数:
      prev_traj_output_dim: PrevTrajEncoder 输出维度 (默认 64)
      prev_dropout_prob:     训练时 prev_traj 被置零的概率 (默认 0.1)
    """

    def __init__(self,
                 # 基类参数 (透传)
                 shape_meta, noise_scheduler, horizon, n_action_steps,
                 n_obs_steps, num_inference_steps=None, obs_as_global_cond=True,
                 diffusion_step_embed_dim=256, down_dims=(256, 512, 1024),
                 kernel_size=5, n_groups=8, condition_type="film",
                 use_down_condition=True, use_mid_condition=True,
                 use_up_condition=True, use_pc_color=False,
                 pointnet_type="pointnet", pointcloud_encoder_cfg=None,
                 point_downsample=False,
                 # Phase 1a 参数 (透传)
                 anchor_bank_path=None,
                 anchor_loss_weight=0.5,
                 cfg_dropout_prob=0.15,
                 cfg_lambda=1.5,
                 truncated_steps=50,
                 truncation_ratio=1.0,
                 use_projection=True,
                 z_min=1.0, z_max=5.0,
                 yaw_max_per_step=0.524,
                 yaw_max_accumulated=3.14159,
                 use_obstacle_projection=False,
                 obstacle_safe_margin=0.5,
                 obstacle_k_neighbors=20,
                 training_phase='joint',
                 # === prev_traj 条件化参数 ===
                 prev_traj_output_dim=64,
                 prev_dropout_prob=0.1,
                 **kwargs):
        # 先调基类 __init__ — 它会创建 UNet 和 obs_encoder
        # 但此时 global_cond_dim 还是 576，需要在创建 UNet 后替换 → 见下方 Post-init
        super().__init__(
            shape_meta=shape_meta, noise_scheduler=noise_scheduler,
            horizon=horizon, n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps, num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims, kernel_size=kernel_size, n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            use_pc_color=use_pc_color, pointnet_type=pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            point_downsample=point_downsample,
            anchor_bank_path=anchor_bank_path,
            anchor_loss_weight=anchor_loss_weight,
            cfg_dropout_prob=cfg_dropout_prob,
            cfg_lambda=cfg_lambda,
            truncated_steps=truncated_steps,
            truncation_ratio=truncation_ratio,
            use_projection=use_projection,
            z_min=z_min, z_max=z_max,
            yaw_max_per_step=yaw_max_per_step,
            yaw_max_accumulated=yaw_max_accumulated,
            use_obstacle_projection=use_obstacle_projection,
            obstacle_safe_margin=obstacle_safe_margin,
            obstacle_k_neighbors=obstacle_k_neighbors,
            training_phase=training_phase,
            **kwargs)

        self.prev_traj_output_dim = prev_traj_output_dim
        self.prev_dropout_prob = prev_dropout_prob

        # ── Post-init: 替换 UNet 的 cond_encoder 层以适配新的 global_cond_dim ──
        # 基类创建 UNet 时 global_cond_dim = 576 (obs_feat * n_obs_steps)
        # 现在需要改为 640 (576 + prev_traj_output_dim)
        self._rebuild_unet_cond_encoders(prev_traj_output_dim)

        # ── PrevTrajEncoder ──
        self.prev_traj_encoder = PrevTrajEncoder(
            horizon=horizon, action_dim=self.action_dim,
            output_dim=prev_traj_output_dim)

        # ── Null prev embedding (首个规划周期 / prev_dropout 时使用) ──
        self.null_prev_embed = nn.Parameter(
            torch.randn(prev_traj_output_dim) * 0.02)

        # ── 推理时缓存: 上一周期的输出轨迹 (unnormalize 后) ──
        self.prev_traj_cache = None

        cprint(f'[PrevCond] global_cond_dim: {self._global_cond_dim_full} '
               f'(obs={self.obs_feature_dim * n_obs_steps} + '
               f'prev={prev_traj_output_dim})', 'cyan')
        cprint(f'[PrevCond] prev_dropout_prob: {prev_dropout_prob}', 'cyan')

    # ═══════════════════════════════════════════════════════════════
    # Post-init: 重建 UNet cond_encoder 层
    # ═══════════════════════════════════════════════════════════════

    def _rebuild_unet_cond_encoders(self, prev_dim):
        """
        重建 ConditionalUnet1D 中所有 ConditionalResidualBlock1D 的 cond_encoder，
        使其接受新的 global_cond_dim = (obs_feat * n_obs_steps + prev_dim)。

        cond_encoder 的类型取决于 condition_type:
          - 'film' / 'add':  nn.Sequential(Mish, Linear(cond_dim, out*N), Rearrange)
          - 'cross_attention_*': CrossAttention(in_channels, cond_dim, out*N)
        """
        new_global_cond_dim = self.obs_feature_dim * self.n_obs_steps + prev_dim
        # cond_dim = dsed + global_cond_dim (见 conditional_unet1d.py L180-L182)
        new_cond_dim = self.model.diffusion_step_embed_dim + new_global_cond_dim

        model = self.model
        replaced_count = 0

        # 遍历 UNet 的下采样、中间、上采样模块
        for module_list in [model.down_modules, model.mid_modules, model.up_modules]:
            if module_list is None:
                continue
            for resblock in module_list:
                if not hasattr(resblock, 'cond_encoder'):
                    # ConditionalResidualBlock1D 在双层 ResBlock 中
                    continue
                old_encoder = resblock.cond_encoder
                if old_encoder is None:
                    continue

                out_channels = resblock.out_channels
                # 确定 cond_encoder 输出维度
                if self.condition_type == 'film':
                    # MLP: Mish → Linear(cond_dim, 2*out_channels) → Rearrange
                    old_linear = old_encoder[-2]  # 倒数第二层是 Linear
                    out_features = old_linear.out_features
                    new_encoder = nn.Sequential(
                        nn.Mish(),
                        nn.Linear(new_cond_dim, out_features),
                        old_encoder[-1],  # 复用 Rearrange('batch t -> batch t 1')
                    )
                elif self.condition_type == 'add':
                    old_linear = old_encoder[-2]
                    out_features = old_linear.out_features
                    new_encoder = nn.Sequential(
                        nn.Mish(),
                        nn.Linear(new_cond_dim, out_features),
                        old_encoder[-1],
                    )
                else:
                    raise NotImplementedError(
                        f"cond_encoder rebuild not implemented "
                        f"for condition_type='{self.condition_type}'")

                resblock.cond_encoder = new_encoder
                replaced_count += 1

        self._global_cond_dim_full = new_global_cond_dim
        cprint(f'[PrevCond] Rebuilt {replaced_count} cond_encoder(s) '
               f'with cond_dim={new_cond_dim}', 'yellow')

    # ═══════════════════════════════════════════════════════════════
    # encode_prev_traj: 编码前次轨迹
    # ═══════════════════════════════════════════════════════════════

    def _encode_prev_traj(self, prev_traj, B, device):
        """
        prev_traj: (B, H, 11) unnormalize 后的物理量，或 None (首个周期)
        返回: (B, prev_traj_output_dim) 编码特征
        """
        if prev_traj is None:
            return self.null_prev_embed.unsqueeze(0).expand(B, -1).to(device)
        # normalize → encode
        prev_norm = self.normalizer['action'].normalize(prev_traj)
        return self.prev_traj_encoder(prev_norm)

    # ═══════════════════════════════════════════════════════════════
    # predict_action (覆盖)
    # ═══════════════════════════════════════════════════════════════

    def predict_action(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        device = self.device
        nobs = dict_apply(nobs, lambda x: x.to(device))

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        dtype = self.dtype

        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(
                -1, *x.shape[2:]))

        # ── 计时：obs_encoder ──
        if self._deploy_timing and device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        obs_feat = self.obs_encoder(this_nobs).reshape(B, -1)  # (B, 576)

        # ── 编码前次轨迹 ──
        prev_feat = self._encode_prev_traj(self.prev_traj_cache, B, device)

        # 拼接条件: obs_feat + prev_feat
        global_cond = torch.cat([obs_feat, prev_feat], dim=-1)  # (B, 640)

        if self._deploy_timing and device.type == 'cuda':
            torch.cuda.synchronize()
        t_enc = time.time() - t0

        anchor_idx = None
        # ── Anchor 保持禁用 (prev_cond 替代 anchor 的模态一致性功能) ──

        global_cond_uncond = None
        if self.cfg_lambda > 0:
            nobs_u = {k: v.clone() for k, v in nobs.items()}
            nobs_u['agent_pos'][:, :, 3:7] = 0.0
            this_u = dict_apply(
                nobs_u, lambda x: x[:, :self.n_obs_steps, ...].reshape(
                    -1, *x.shape[2:]))
            obs_feat_u = self.obs_encoder(this_u).reshape(B, -1)
            # uncond 也拼接 prev_feat (不 dropout prev — 简化 CFG)
            global_cond_uncond = torch.cat([obs_feat_u, prev_feat], dim=-1)

        cond_data = torch.zeros(B, T, Da, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # ── 计时：去噪循环 ──
        if self._deploy_timing and device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        nsample = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=None, global_cond=global_cond,
            global_cond_uncond=global_cond_uncond,
            anchor_idx=anchor_idx, **self.kwargs)

        if self._deploy_timing and device.type == 'cuda':
            torch.cuda.synchronize()
        t_den = time.time() - t0

        if self._deploy_timing:
            print(f"[Timing] obs_encoder: {t_enc*1000:.0f}ms, "
                  f"denoise: {t_den*1000:.0f}ms")

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        # 投影后处理 (保留 Yaw/Z 约束)
        current_pos = obs_dict['agent_pos'][:, -1, :3]
        if torch.is_tensor(current_pos):
            current_pos = current_pos.to(self.device)
        current_z = current_pos[:, 2]
        sin_yaw = obs_dict['agent_pos'][:, -1, 7]
        cos_yaw = obs_dict['agent_pos'][:, -1, 8]
        if torch.is_tensor(sin_yaw):
            sin_yaw = sin_yaw.to(self.device)
            cos_yaw = cos_yaw.to(self.device)
        current_yaw = torch.atan2(sin_yaw, cos_yaw)
        pc_for_proj = obs_dict.get('point_cloud', None)
        if pc_for_proj is not None:
            pc_for_proj = pc_for_proj[:, -1]
            if torch.is_tensor(pc_for_proj):
                pc_for_proj = pc_for_proj.to(self.device)

        action = self._project_trajectory(
            action, current_z=current_z,
            point_cloud=pc_for_proj, current_pos=current_pos,
            current_yaw=current_yaw)

        # ── 缓存当前轨迹供下周期使用 (存 unnormalize 物理量) ──
        self.prev_traj_cache = action_pred.detach().clone()  # (B, H, 11)

        return {'action_pred': action}

    # ═══════════════════════════════════════════════════════════════
    # compute_loss (覆盖)
    # ═══════════════════════════════════════════════════════════════

    def compute_loss(self, batch):
        obs_dict = {k: v.clone() for k, v in batch['obs'].items()}

        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        dtype = self.dtype

        # CFG dropout
        if self.training and self.cfg_dropout_prob > 0:
            drop = torch.rand(B, device=device) < self.cfg_dropout_prob
            if drop.any():
                obs_dict['agent_pos'][drop, :, 3:7] = 0.0

        nobs = self.normalizer.normalize(obs_dict)
        nobs = dict_apply(nobs, lambda x: x.to(device))

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        nactions = self.normalizer['action'].normalize(batch['action'])
        nactions = nactions.to(device)

        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # ── 观测编码 ──
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        obs_feat = self.obs_encoder(this_nobs).reshape(B, -1)

        # ── 前次轨迹编码 ──
        if 'prev_action' in batch:
            prev_traj = batch['prev_action'].to(device)
            # prev_action 在数据集里是零向量 (null) 或 真实轨迹 (unnormalize 物理量)
            # 零向量通过 prev_dropout 或 episode 边界产生
            is_null = (prev_traj.abs().sum(dim=(1, 2)) < 1e-6)  # (B,)
            prev_norm = self.normalizer['action'].normalize(prev_traj)
            prev_feat = self.prev_traj_encoder(prev_norm)
        else:
            prev_feat = self.null_prev_embed.unsqueeze(0).expand(B, -1)

        # 拼接条件
        global_cond = torch.cat([obs_feat, prev_feat], dim=-1)  # (B, 640)

        trajectory = nactions
        self._ensure_scheduler_device(device)

        noise = torch.randn(trajectory.shape, device=device, dtype=dtype)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device).long()

        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        cond_data = torch.zeros(B, T, Da, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        noisy[cond_mask] = cond_data[cond_mask]

        pred = self.model(sample=noisy, timestep=timesteps,
                          local_cond=None, global_cond=global_cond)

        target = noise if self.noise_scheduler.config.prediction_type == 'epsilon' \
                 else trajectory
        loss_mask = ~cond_mask
        diff_loss = F.mse_loss(pred, target, reduction='none')
        diff_loss = (diff_loss * loss_mask.float()).mean()

        # Anchor loss: prev_cond 方案不使用 anchor
        anchor_loss = torch.tensor(0.0, device=device)
        anchor_acc = 0.0

        total = diff_loss + self.anchor_loss_weight * anchor_loss
        return total, {
            'bc_loss': diff_loss.item(),
            'anchor_loss': anchor_loss.item(),
            'anchor_acc': anchor_acc,
            'total_loss': total.item(),
        }

    # ═══════════════════════════════════════════════════════════════
    # 训练阶段控制 (覆盖)
    # ═══════════════════════════════════════════════════════════════

    def set_training_phase(self, phase):
        """
        训练阶段:
          - 'cond_only': 仅训练 cond_encoders + PrevTrajEncoder + null_prev_embed
          - 'joint':     全参数训练
        """
        self.training_phase = phase
        if phase == 'cond_only':
            # 冻结 UNet Conv1D 主体和 obs_encoder
            for p in self.model.parameters():
                p.requires_grad_(False)
            for p in self.obs_encoder.parameters():
                p.requires_grad_(False)
            # 解冻 cond_encoders
            for module_list in [self.model.down_modules,
                               self.model.mid_modules,
                               self.model.up_modules]:
                if module_list is None:
                    continue
                for resblock in module_list:
                    if hasattr(resblock, 'cond_encoder') and resblock.cond_encoder is not None:
                        for p in resblock.cond_encoder.parameters():
                            p.requires_grad_(True)
            # 训练 prev_traj_encoder 和 null_prev_embed
            for p in self.prev_traj_encoder.parameters():
                p.requires_grad_(True)
            self.null_prev_embed.requires_grad_(True)
        elif phase == 'joint':
            for p in self.parameters():
                p.requires_grad_(True)
        cprint(f'[PrevCond] phase={phase}', 'cyan')

    def get_optimizer_groups(self, lr, anchor_lr_mult=5.0):
        """
        参数分组:
          - UNet body: lr (冻结在 cond_only 阶段, joint 阶段低 lr)
          - obs_encoder: lr (冻结)
          - cond_encoders: lr
          - prev_traj_encoder + null_prev_embed: lr
        """
        groups = [
            {'params': list(self.model.parameters()), 'lr': lr},
            {'params': list(self.obs_encoder.parameters()), 'lr': lr},
            {'params': list(self.prev_traj_encoder.parameters()), 'lr': lr},
            {'params': [self.null_prev_embed], 'lr': lr},
        ]
        return groups
