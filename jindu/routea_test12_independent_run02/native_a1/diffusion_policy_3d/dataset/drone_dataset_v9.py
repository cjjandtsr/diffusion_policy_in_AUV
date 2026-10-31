"""
drone_dataset_v9.py — Phase α v9 UAV 数据集（α2 样本重加权 + α3 stride 支持）
================================================================================
基于 drone_dataset.py（Phase 1a 版），Phase α 扩展：

α2: 在 __getitem__ 中计算每样本的 min_dist（body-frame 点云中最近障碍距离），
    用于训练期 SAFE-GIL 式样本重加权。

α3 (可选): 支持历史帧采样 stride 扩展时序窗口（默认 stride=1，不改帧数）。

放置路径:
  ~/AUV/.../diffusion_policy_3d/dataset/drone_dataset_v9.py
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


class DroneDatasetV9(BaseDataset):
    """Phase α 版 DroneDataset: α2 min_dist + α3 stride 支持"""

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
                 include_prev_action=True,
                 # ── Phase α3: 历史帧采样 stride ──
                 obs_stride=1):
        """
        Args:
            obs_stride: 历史帧采样步长。1=原始行为（连续帧），>1=跳帧扩窗。
                       不改 n_obs_steps 帧数，只改帧间隔。
                       如 stride=3: 当前帧 + 3帧前 + 6帧前（间隔~0.9s per step）。
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
        self.obs_stride = obs_stride

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed)

        # α3: SequenceSampler 的 sequence_length 需要考虑 stride
        # 当 stride > 1 时，需要更长的窗口来覆盖相同的 n_obs_steps
        effective_horizon = horizon + (pad_before + 1) * (obs_stride - 1)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=effective_horizon,
            pad_before=pad_before * obs_stride,  # α3: pad 也要乘 stride
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.effective_horizon = effective_horizon

        # 预计算 episode 边界
        self.episode_ends = self.replay_buffer.episode_ends
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        # 末段加权采样权重
        self.sample_weights = self._compute_sample_weights()

        n_end = int((self.sample_weights > 1.0).sum())
        stride_info = f', obs_stride={obs_stride}' if obs_stride > 1 else ''
        cprint(f'[DroneDatasetV9] {train_mask.sum()} train / '
               f'{val_mask.sum()} val, {len(self.sampler)} windows, '
               f'{n_end} end-weighted (x{end_weight}, d<{end_dist_threshold}m)'
               f'{stride_info}', 'green')

    def _compute_sample_weights(self):
        n = len(self.sampler)
        if n == 0 or self.end_weight <= 1.0:
            return np.ones(n, dtype=np.float64)

        assert hasattr(self.sampler, 'indices'), \
            "SequenceSampler 没有 indices 属性"
        assert self.sampler.indices.ndim == 2, \
            f"indices shape 异常: {self.sampler.indices.shape}"
        assert self.sampler.indices.shape[1] >= 1, \
            f"indices 列数不足: {self.sampler.indices.shape}"

        states = self.replay_buffer['state']
        weights = np.ones(n, dtype=np.float64)
        for i in range(n):
            buf_start = self.sampler.indices[i, 0]
            dist = states[buf_start, 6]
            if dist < self.end_dist_threshold:
                weights[i] = self.end_weight
        return weights

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.effective_horizon,
            pad_before=self.pad_before * self.obs_stride,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        val_set._is_train = False
        return val_set

    def get_normalizer(self, mode='gaussian', **kwargs):
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

    def _compute_min_dist(self, point_cloud):
        """
        α2: 计算 body-frame 点云中最近障碍距离。

        在 body frame 中，无人机位于原点 (0,0,0)。
        min_dist = min(||p_i||) 对所有有效点云点。

        Args:
            point_cloud: (T, N, 3) — body-frame 点云（已降采样、已增广）

        Returns:
            min_dist: float — 最近障碍距离 (m)
        """
        # 取最后一帧点云（当前观测）
        pc = point_cloud[-1, :, :3]  # (N, 3)
        # 排除零点（被 dropout 置零的点）
        valid = (pc != 0).any(dim=-1) if isinstance(pc, torch.Tensor) else \
                (np.abs(pc).sum(axis=-1) > 1e-8)
        if valid.sum() == 0:
            return 5.0  # 全 dropout，返回大值（无信息）
        if isinstance(pc, torch.Tensor):
            dists = torch.norm(pc[valid], dim=-1)
            return dists.min().item()
        else:
            dists = np.linalg.norm(pc[valid], axis=-1)
            return float(dists.min())

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)
        action = sample['action'].astype(np.float32)

        # α3: stride 采样历史观测帧
        if self.obs_stride > 1:
            # 保留完整序列用于 action 和 point_cloud，只对 state 做 stride 采样
            T_total = agent_pos.shape[0]
            # obs 部分: 取前 pad_before*stride+1 帧中的每 stride 帧
            # 保持原始数据不变，在 obs 提取时做 stride
            # 实际处理: agent_pos 按 stride 采样子集
            obs_len = self.pad_before + 1  # n_obs_steps
            obs_indices = np.arange(obs_len) * self.obs_stride
            # 确保索引不越界
            obs_indices = obs_indices[obs_indices < T_total]
            agent_pos_obs = agent_pos[obs_indices]
            # 重建 agent_pos: 观测帧用 stride 采样，其余保持不变
            # 但 point_cloud 和 action 还按原始长度
        else:
            agent_pos_obs = agent_pos

        point_cloud = uniform_sampling_numpy(point_cloud, self.num_points)

        if self._is_train and self.augment_pc:
            point_cloud, agent_pos, action = self._augment(
                point_cloud, agent_pos, action)

        # α2: 计算 min_dist（在增广后，因为增广改变了点云）
        # 转 tensor 用于计算
        pc_tensor = torch.from_numpy(point_cloud) if isinstance(point_cloud, np.ndarray) else point_cloud
        min_dist = self._compute_min_dist(pc_tensor)

        return {
            'obs': {'agent_pos': agent_pos, 'point_cloud': point_cloud},
            'action': action,
            'min_dist': np.float32(min_dist),  # α2: 用于样本重加权
        }

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        if self.include_prev_action:
            buffer_start_idx = int(self.sampler.indices[idx, 0])
            ep_idx = np.searchsorted(self.episode_ends, buffer_start_idx, side='right')
            episode_start = self.episode_starts[ep_idx]
            prev_buf_start = buffer_start_idx - self.n_action_steps

            if prev_buf_start < episode_start:
                prev_action = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
            else:
                prev_action = self.replay_buffer['action'][
                    prev_buf_start : prev_buf_start + self.horizon
                ].astype(np.float32)

            if self._is_train and np.random.rand() < self.prev_dropout_prob:
                prev_action = np.zeros_like(prev_action)

            data['prev_action'] = prev_action

        return dict_apply(data,
            lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
