#!/usr/bin/env python3
"""
critic_network.py — Phase 2 Step 2: TrajectoryCollisionCritic
================================================================================
Critic 网络：评估 (obs, trajectory) → collision_prob ∈ [0, 1]

设计原则:
  - 复用 Phase 1a 的 iDP3Encoder (point cloud encoder)
  - 不修改 Phase1aPolicy
  - 输入是 obs_dict + trajectory，输出标量

放置位置：
  diffusion_policy_3d/policy/critic_network.py
"""

import torch
import torch.nn as nn
from termcolor import cprint


class TrajectoryCollisionCritic(nn.Module):
    """
    评估给定 (obs, trajectory) 是否会导致碰撞。

    输入:
      obs_dict = {
        'point_cloud': [B, n_obs_steps, num_points, 3],
        'agent_pos':   [B, n_obs_steps, 9]
      }
      trajectory: [B, n_action_steps, action_dim]  (11d)

    输出:
      collision_prob: [B, 1] ∈ [0, 1]
    """

    def __init__(self,
                 point_encoder,             # 复用 Phase 1a 的 iDP3Encoder
                 pc_feat_dim=192,
                 state_dim=9,
                 n_obs_steps=3,
                 n_action_steps=8,
                 action_dim=11,
                 hidden_dim=128,
                 freeze_pc_encoder=True):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim

        # 点云 encoder（复用 Phase 1a）
        self.point_encoder = point_encoder
        if freeze_pc_encoder:
            for p in self.point_encoder.parameters():
                p.requires_grad_(False)
            cprint('[Critic] point_encoder 已冻结', 'cyan')

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        # Trajectory encoder
        self.traj_encoder = nn.Sequential(
            nn.Linear(n_action_steps * action_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # Fusion
        fusion_input_dim = pc_feat_dim + 32 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 参数统计
        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters()
                          if p.requires_grad)
        cprint(f'[Critic] 参数: total={n_total/1e6:.2f}M, '
               f'trainable={n_trainable/1e6:.2f}M', 'cyan')

    def forward(self, obs_dict, trajectory):
        """
        Args:
          obs_dict:
            point_cloud: [B, T_obs, N, 3]
            agent_pos:   [B, T_obs, 9]
          trajectory:    [B, T_act, 11]

        Returns:
          collision_prob: [B, 1] ∈ [0, 1]
        """
        B = trajectory.shape[0]

        # 取 obs 最后一帧（当前观测）
        pc_now = obs_dict['point_cloud'][:, -1]  # [B, N, 3]
        state_now = obs_dict['agent_pos'][:, -1]  # [B, 9]

        # iDP3Encoder 期望输入 dict 形式
        # 这里直接调用 point_encoder 部分（取决于具体实现）
        # 假设 point_encoder 是 iDP3Encoder 的 PointNet 部分
        pc_feat = self._encode_point_cloud(pc_now)  # [B, pc_feat_dim]

        state_feat = self.state_encoder(state_now)  # [B, 32]
        traj_flat = trajectory.reshape(B, -1)        # [B, T_act*11]
        traj_feat = self.traj_encoder(traj_flat)     # [B, 64]

        fused = torch.cat([pc_feat, state_feat, traj_feat], dim=-1)
        logit = self.fusion(fused)                   # [B, 1]
        prob = torch.sigmoid(logit)
        return prob

    def _encode_point_cloud(self, pc):
        """
        调用 iDP3Encoder 的点云编码部分。
        
        iDP3Encoder 实际上是 obs_encoder，它接收 obs_dict 并输出
        融合特征。这里我们只想要点云特征。
        
        注：实际实现需要根据 iDP3Encoder 的接口调整。如果 iDP3Encoder
            不能单独输出点云特征，需要构造 obs dict 调用并取相应分支。
        """
        # 方案 A: 如果 point_encoder 是独立的 PointNet 模块
        if hasattr(self.point_encoder, 'extractor'):
            # 通常 iDP3Encoder 内部有 self.extractor 是 PointNet
            return self.point_encoder.extractor(pc)
        # 方案 B: 调用 iDP3Encoder 但只传点云
        else:
            # 构造一个最小的 obs dict
            B = pc.shape[0]
            dummy_state = torch.zeros(B, self.point_encoder.state_dim
                                       if hasattr(self.point_encoder, 'state_dim') else 9,
                                       device=pc.device, dtype=pc.dtype)
            obs = {'point_cloud': pc, 'agent_pos': dummy_state}
            return self.point_encoder(obs)


# ─────────────────────────────────────────────────────────────────
# 简化版 Critic（不依赖 iDP3Encoder，用独立 PointNet）
# ─────────────────────────────────────────────────────────────────

class SimplePointNet(nn.Module):
    """轻量级 PointNet，用于不复用 iDP3Encoder 的场景"""
    def __init__(self, in_dim=3, out_dim=192):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.GELU(),
        )

    def forward(self, pc):
        """pc: [B, N, 3] → [B, out_dim]"""
        x = self.mlp1(pc)         # [B, N, 256]
        x = x.max(dim=1)[0]       # max pool → [B, 256]
        return self.mlp2(x)       # [B, out_dim]


class SimpleCritic(nn.Module):
    """
    独立版 Critic，不依赖 Phase 1a 的 iDP3Encoder。

    Zread 审查修正 (R1):
      - 输入用全部 n_obs_steps 帧（而非仅最新一帧），与 Phase 1a 一致
      - 点云：每帧独立编码后 max-pool 跨时间，捕获运动信息
      - 状态：每帧独立编码后 concat（保留时间结构）
    """
    def __init__(self,
                 num_points=2048,
                 state_dim=9,
                 n_obs_steps=3,
                 n_action_steps=8,
                 action_dim=11,
                 pc_feat_dim=192,
                 hidden_dim=128):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.point_encoder = SimplePointNet(in_dim=3, out_dim=pc_feat_dim)

        # 状态编码：每帧 9 维 → 32 维
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        self.traj_encoder = nn.Sequential(
            nn.Linear(n_action_steps * action_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # Fusion: pc_feat (across time, max-pool) + state_feat (concat across time) + traj_feat
        # pc: [B, n_obs, pc_feat_dim] → max-pool → [B, pc_feat_dim]
        # state: [B, n_obs, 32] → flatten → [B, n_obs * 32]
        # traj: [B, 64]
        fusion_input = pc_feat_dim + n_obs_steps * 32 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        n_total = sum(p.numel() for p in self.parameters())
        cprint(f'[SimpleCritic] 参数: {n_total/1e6:.2f}M, '
               f'n_obs={n_obs_steps}', 'cyan')

    def forward(self, obs_dict, trajectory):
        B = trajectory.shape[0]
        T_obs = self.n_obs_steps

        pc = obs_dict['point_cloud']        # [B, n_obs, N, 3]
        state = obs_dict['agent_pos']       # [B, n_obs, 9]

        # 1. 编码每帧点云：[B*n_obs, N, 3] → [B*n_obs, pc_feat_dim] → [B, n_obs, pc_feat_dim]
        pc_flat = pc.reshape(B * T_obs, pc.shape[2], pc.shape[3])
        pc_feat_per_frame = self.point_encoder(pc_flat)  # [B*n_obs, pc_feat_dim]
        pc_feat_per_frame = pc_feat_per_frame.reshape(B, T_obs, -1)
        # Max pool over time → [B, pc_feat_dim]
        pc_feat = pc_feat_per_frame.max(dim=1)[0]

        # 2. 编码每帧状态：保留时间结构 → [B, n_obs * 32]
        state_flat = state.reshape(B * T_obs, -1)
        state_feat_per_frame = self.state_encoder(state_flat)  # [B*n_obs, 32]
        state_feat = state_feat_per_frame.reshape(B, T_obs * 32)

        # 3. 编码 trajectory
        traj_flat = trajectory.reshape(B, -1)
        traj_feat = self.traj_encoder(traj_flat)  # [B, 64]

        # 4. Fusion
        fused = torch.cat([pc_feat, state_feat, traj_feat], dim=-1)
        logit = self.fusion(fused)
        return torch.sigmoid(logit)


if __name__ == '__main__':
    # Sanity check
    print("Testing SimpleCritic forward pass (n_obs=3)...")
    critic = SimpleCritic(n_obs_steps=3)
    B = 4
    obs = {
        'point_cloud': torch.randn(B, 3, 2048, 3),
        'agent_pos': torch.randn(B, 3, 9),
    }
    traj = torch.randn(B, 8, 11)
    out = critic(obs, traj)
    print(f"  Input traj shape: {traj.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    assert out.shape == (B, 1), "Output shape mismatch"
    assert (out >= 0).all() and (out <= 1).all(), "Output not in [0, 1]"
    print("  ✓ SimpleCritic 测试通过")