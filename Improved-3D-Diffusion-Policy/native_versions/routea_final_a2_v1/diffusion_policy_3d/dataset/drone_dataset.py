"""
drone_dataset.py — v8 UAV 数据集（Phase 1a 版）
================================================================================
Phase 0.5 诊断后: 移除 yaw_loss_boost（已证伪），恢复标准 gaussian normalizer。
Δyaw 的多模态问题由 Phase 1a 的 Anchor + CFG 机制解决，不在 normalizer 层面处理。

放置路径:
  ~/AUV/Improved-3D-Diffusion-Policy/diffusion_policy_3d/dataset/drone_dataset.py
"""

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from termcolor import cprint

try:
    import diffusion_policy_3d.model.vision_3d.point_process as point_process
    _has_point_process = True
except ImportError:
    _has_point_process = False


def _uniform_sampling_fallback(point_cloud, num_points):
    orig_shape = point_cloud.shape
    N, C = orig_shape[-2], orig_shape[-1]
    if point_cloud.ndim == 2:
        idx = np.random.choice(N, num_points, replace=(N < num_points))
        return point_cloud[idx]
    flat = point_cloud.reshape(-1, N, C)
    T = flat.shape[0]
    result = np.zeros((T, num_points, C), dtype=point_cloud.dtype)
    for t in range(T):
        idx = np.random.choice(N, num_points, replace=(N < num_points))
        result[t] = flat[t, idx]
    return result.reshape(orig_shape[:-2] + (num_points, C))


def uniform_sampling_numpy(point_cloud, num_points):
    if _has_point_process:
        return point_process.uniform_sampling_numpy(point_cloud, num_points)
    return _uniform_sampling_fallback(point_cloud, num_points)


class DroneDataset(BaseDataset):

    def __init__(self, zarr_path, horizon=16, pad_before=0, pad_after=0,
                 seed=42, val_ratio=0.1, max_train_episodes=None,
                 task_name='drone_trajectory', num_points=4096,
                 augment_pc=True, pc_dropout_range=(0.7, 1.0),
                 pc_jitter_std=0.02, pc_rotation_range=30.0,
                 pc_depth_noise_coeff=0.001, pc_depth_noise_max=0.03,
                 end_weight=3.0, end_dist_threshold=2.0,
                 # prev_traj 条件化参数
                 n_action_steps=8,
                 prev_dropout_prob=0.1,
                 include_prev_action=True):
        """
        end_weight: 末段窗口（dist < end_dist_threshold）的采样权重倍数。
                    解决 SequenceSampler 窗口约束导致末段样本稀缺（仅 0.14%）的问题。
        end_dist_threshold: 判定末段的距离阈值（米），对应 state[6]（dist to goal）。
        n_action_steps: 部署时的动作执行步数，用于构造 (a_{k-1}, a_k) 重规划对的
                        时间偏移。prev_action 从 buffer 绝对位置
                        buffer_start_idx - n_action_steps 开始取 horizon 帧。
        prev_dropout_prob: 训练时随机将 prev_action 替换为零向量的概率。
        include_prev_action: 是否在 __getitem__ 返回中包含 prev_action。
        """
        super().__init__()
        self.task_name = task_name
        self.num_points = num_points
        self.augment_pc = augment_pc
        self.pc_dropout_range = pc_dropout_range
        self.pc_jitter_std = pc_jitter_std
        self.pc_rotation_range_rad = np.radians(pc_rotation_range)
        self.pc_depth_noise_coeff = pc_depth_noise_coeff
        self.pc_depth_noise_max = pc_depth_noise_max
        self._is_train = True
        self.end_weight = end_weight
        self.end_dist_threshold = end_dist_threshold
        self.n_action_steps = n_action_steps
        self.prev_dropout_prob = prev_dropout_prob
        self.include_prev_action = include_prev_action
        self.action_dim = 11

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before, pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # 预计算 episode 边界（用于 prev_action 越界检测）
        self.episode_ends = self.replay_buffer.episode_ends
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        # 计算末段加权采样权重
        # 原因: SequenceSampler 窗口约束 (t + n_action_steps <= ep_len) 导致
        #        d_goal < 1m 的有效窗口仅 74 个 (0.14%)，模型在 goal 附近是外推行为
        # 依据: 对 merged_body_12.zarr 的统计: 988 eps, 60151 frames,
        #        d_goal<1m 有 3743 帧但仅 74 个有效窗口起点
        self.sample_weights = self._compute_sample_weights()

        n_end = int((self.sample_weights > 1.0).sum())
        cprint(f'[DroneDataset] {train_mask.sum()} train / '
               f'{val_mask.sum()} val, {len(self.sampler)} windows, '
               f'{n_end} end-weighted (x{end_weight}, d<{end_dist_threshold}m)', 'green')

    def _compute_sample_weights(self):
        """
        计算每个采样窗口的权重。
        窗口起始帧的 dist_to_goal (state[6]) < end_dist_threshold 时，
        权重为 end_weight，否则为 1.0。

        原因: SequenceSampler 窗口约束使末段样本极度稀缺
        依据: 数据统计 d_goal<1m 有效窗口仅 0.14%
        """
        n = len(self.sampler)
        if n == 0 or self.end_weight <= 1.0:
            return np.ones(n, dtype=np.float64)

        # 验证 sampler.indices 格式
        # 原因: indices 格式异常会导致训练崩溃
        # 依据: zread R2 审查 Opt 2
        assert hasattr(self.sampler, 'indices'), \
            "SequenceSampler 没有 indices 属性"
        assert self.sampler.indices.ndim == 2, \
            f"indices shape 异常: {self.sampler.indices.shape}"
        assert self.sampler.indices.shape[1] >= 1, \
            f"indices 列数不足（至少需要 1 列 buffer_start_idx）: {self.sampler.indices.shape}"

        # 取每个窗口起始帧的 state[6] (dist to goal)
        states = self.replay_buffer['state']
        weights = np.ones(n, dtype=np.float64)
        for i in range(n):
            buf_start = self.sampler.indices[i, 0]  # buffer_start_idx
            dist = states[buf_start, 6]  # state[6] = dist to goal
            if dist < self.end_dist_threshold:
                weights[i] = self.end_weight
        return weights

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before, pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        val_set._is_train = False
        return val_set

    def get_normalizer(self, mode='gaussian', **kwargs):
        """标准 gaussian normalizer，不加任何维度加权"""
        data = {'action': self.replay_buffer['action']}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def _augment(self, pc, state, action):
        T, N, C = pc.shape
        pc, st, act = pc.copy(), state.copy(), action.copy()

        # Dropout
        keep = np.random.uniform(*self.pc_dropout_range)
        n_keep = max(1, int(N * keep))
        for t in range(T):
            idx = np.random.choice(N, n_keep, replace=False)
            mask = np.zeros(N, dtype=bool); mask[idx] = True
            pc[t, ~mask, :] = 0.0

        # Jitter
        if self.pc_jitter_std > 0:
            pc[:, :, :3] += np.random.randn(T, N, 3).astype(np.float32) * self.pc_jitter_std

        # Depth noise
        if self.pc_depth_noise_coeff > 0:
            d = np.linalg.norm(pc[:, :, :3], axis=-1, keepdims=True)
            s = np.minimum(self.pc_depth_noise_coeff * d, self.pc_depth_noise_max)
            pc[:, :, :3] += np.random.randn(T, N, 3).astype(np.float32) * s

        # Rotate world
        if self.pc_rotation_range_rad > 0:
            a = np.random.uniform(-self.pc_rotation_range_rad,
                                   self.pc_rotation_range_rad)
            ca, sa = np.float32(np.cos(a)), np.float32(np.sin(a))
            R = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=np.float32)
            st[:, 0:3] = np.einsum('ij,tj->ti', R, st[:, 0:3])
            yaw = np.arctan2(st[:, 7], st[:, 8]) + a
            st[:, 7] = np.sin(yaw).astype(np.float32)
            st[:, 8] = np.cos(yaw).astype(np.float32)
            for s in (slice(0,3), slice(3,6), slice(6,9)):
                act[:, s] = np.einsum('ij,tj->ti', R, act[:, s])

        return pc, st, act

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)
        action = sample['action'].astype(np.float32)
        point_cloud = uniform_sampling_numpy(point_cloud, self.num_points)
        if self._is_train and self.augment_pc:
            point_cloud, agent_pos, action = self._augment(
                point_cloud, agent_pos, action)
        return {
            'obs': {'agent_pos': agent_pos, 'point_cloud': point_cloud},
            'action': action,
        }

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        if self.include_prev_action:
            # 从 sampler.indices 获取 buffer 绝对位置
            buffer_start_idx = int(self.sampler.indices[idx, 0])

            # 查找当前 idx 属于哪个 episode
            ep_idx = np.searchsorted(self.episode_ends, buffer_start_idx, side='right')
            episode_start = self.episode_starts[ep_idx]

            # 前次规划的 buffer 绝对起始
            prev_buf_start = buffer_start_idx - self.n_action_steps

            if prev_buf_start < episode_start:
                prev_action = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
            else:
                prev_action = self.replay_buffer['action'][
                    prev_buf_start : prev_buf_start + self.horizon
                ].astype(np.float32)

            # 训练时随机 dropout
            if self._is_train and np.random.rand() < self.prev_dropout_prob:
                prev_action = np.zeros_like(prev_action)

            data['prev_action'] = prev_action

        return dict_apply(data,
            lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)