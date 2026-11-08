"""
prev_traj_encoder.py — Previous Trajectory Encoder

将前次规划周期输出的动作轨迹 a_{k-1} ∈ R^(H × D_a) 编码为紧凑特征向量 ∈ R^(D_p)，
作为扩散模型的条件注入，实现跨规划周期模态一致性。

设计依据: 2026-05-25-prev-traj-cond-design.md §4.2
"""

import torch
import torch.nn as nn


class PrevTrajEncoder(nn.Module):
    """
    前次轨迹编码器。

    输入: (B, horizon, action_dim) — 上一规划周期的完整输出轨迹
    输出: (B, output_dim)          — 紧凑特征向量

    结构: Conv1D × 2 + AdaptiveAvgPool1d + LayerNorm，保留轨迹的时序邻近结构。
    参数: ~5K（output_dim=64 时），推断时间 < 1ms。
    """

    def __init__(self, horizon=16, action_dim=11, output_dim=64):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv1d(action_dim, 32, kernel_size=3, padding=1),  # (B, 32, H)
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),          # (B, 64, H)
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                               # (B, 64, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        # x: (B, horizon, action_dim)
        x = x.transpose(1, 2)   # (B, action_dim, horizon)
        x = self.conv(x)        # (B, 64, 1)
        x = x.squeeze(-1)       # (B, 64)
        x = self.fc(x)          # (B, output_dim)
        return x
