"""
anchor_head.py — Phase 1a AnchorHead 模块
================================================================================
MLP 分类器，从 obs 编码选择最佳 anchor。

放置路径:
  ~/AUV/Improved-3D-Diffusion-Policy/diffusion_policy_3d/policy/anchor_head.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorHead(nn.Module):
    """
    从 obs encoding 预测 anchor 选择概率。

    输入: global_cond (B, obs_feat_dim)  — obs_encoder 输出
    输出: logits (B, K+1)               — anchor 选择概率
    """

    def __init__(self, input_dim, n_anchors, hidden_dim=256):
        super().__init__()
        self.n_anchors = n_anchors
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_anchors),
        )

    def forward(self, global_cond):
        """
        global_cond: (B, obs_feat_dim)
        Returns: logits (B, K+1)
        """
        return self.net(global_cond)

    def predict(self, global_cond):
        """返回选中的 anchor index"""
        logits = self.forward(global_cond)
        return logits.argmax(dim=-1)  # (B,)