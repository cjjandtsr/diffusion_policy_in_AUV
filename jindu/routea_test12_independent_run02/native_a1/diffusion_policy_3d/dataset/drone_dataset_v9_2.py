"""
drone_dataset_v9_2.py — Phase α-aug v9.2 UAV 数据集（A1 世界平移 + A2 z 扰动回正）
====================================================================================
基于 drone_dataset_v9.py，Phase α-aug 扩展：

A1: 世界平移增广（z + x,y）— 对称变换，教模型忽略绝对坐标
A2: z 扰动 + 合成回正 — 制造 goal_bz≠0 的监督配对，教模型 z 修正能力

Guide: Phase α-aug — z 能力增广重训（goal_bz 带宽 + 分布外鲁棒）
上级文档: iDP3_UAV_v9_Master_Design.md v9.5

放置路径:
  ~/AUV/.../diffusion_policy_3d/dataset/drone_dataset_v9_2.py
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


class DroneDatasetV92(BaseDataset):
    """Phase α-aug 版 DroneDataset: A1 世界平移 + A2 z 扰动回正"""

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
                 obs_stride=1,
                 # ── Phase α-aug A1: 世界平移 ──
                 aug_a1_enable=True,
                 aug_a1_dz_range=(-1.5, 3.0),
                 aug_a1_dxy_range=(-20.0, 20.0),
                 # ── Phase α-aug A2: z 扰动 + 合成回正 ──
                 aug_a2_enable=True,
                 aug_a2_prob=0.5,
                 aug_a2_dz_range=(-0.5, 0.5),
                 aug_a2_tau=5.0,
                 aug_a2_k=1.0,
                 aug_a2_traj_dt=0.3,
                 aug_a2_modify_action=False,   # aux loss 版: 不改 action
                 # ── A2 验证模式 (P0 检查用) ──
                 aug_a2_verify_mode=False,
                 ):
        """
        Args:
            aug_a1_enable: 开启 A1 世界平移增广
            aug_a1_dz_range: z 平移范围 (m)，覆盖高空到 ~5m
            aug_a1_dxy_range: x,y 平移范围 (m)
            aug_a2_enable: 开启 A2 z 扰动 + 合成回正增广
            aug_a2_prob: A2 应用概率 (0-1)，推荐 0.5
            aug_a2_dz_range: z 扰动幅度范围 (m)，保守 ±0.5
            aug_a2_tau: 回正指数衰减步数 (步)，ablation {3,5,8}
            aug_a2_k: 回正比例 (0-1)，1.0=全量回正
            aug_a2_traj_dt: 轨迹时间步长 (s)，用于 dvel/acc 差分
            aug_a2_verify_mode: True 时打印 P0 验证信息（地面点平移正确性）
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

        # A1 params
        self.aug_a1_enable = aug_a1_enable
        self.aug_a1_dz_range = aug_a1_dz_range
        self.aug_a1_dxy_range = aug_a1_dxy_range

        # A2 params
        self.aug_a2_enable = aug_a2_enable
        self.aug_a2_prob = aug_a2_prob
        self.aug_a2_dz_range = aug_a2_dz_range
        self.aug_a2_tau = aug_a2_tau
        self.aug_a2_k = aug_a2_k
        self.aug_a2_traj_dt = aug_a2_traj_dt
        self.aug_a2_modify_action = aug_a2_modify_action
        self.aug_a2_verify_mode = aug_a2_verify_mode

        # A2 验证计数器
        self._a2_verify_count = 0
        self._a2_verify_max = 5  # 只打印前 N 次

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed)

        # α3: SequenceSampler 的 sequence_length 需要考虑 stride
        effective_horizon = horizon + (pad_before + 1) * (obs_stride - 1)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=effective_horizon,
            pad_before=pad_before * obs_stride,
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
        a1_info = f', A1(world_trans)' if aug_a1_enable else ''
        a2_mode = 'aux_loss' if not aug_a2_modify_action else '3ch_correct'
        a2_info = f', A2(z_perturb p={aug_a2_prob}, dz=±{aug_a2_dz_range[1]}m, mode={a2_mode})' if aug_a2_enable else ''
        cprint(f'[DroneDatasetV92] {train_mask.sum()} train / '
               f'{val_mask.sum()} val, {len(self.sampler)} windows, '
               f'{n_end} end-weighted (x{end_weight}, d<{end_dist_threshold}m)'
               f'{stride_info}{a1_info}{a2_info}', 'green')

        # P0: A2 启用时验证配置合理性
        if aug_a2_enable:
            self._verify_a2_config()

    def _verify_a2_config(self):
        """P0 配置验证"""
        if not self.aug_a2_modify_action:
            cprint(f'[A2-Mode] aux_loss 版: 只改观测(pos_world_z/goal_bz/pc_z)，'
                   f'不改 GT action，由 L_z=MSE(Σdz_pred, -Δz) 监督回正', 'cyan')
            return
        # 激进版: 检查回正速率
        dz_max = max(abs(self.aug_a2_dz_range[0]), abs(self.aug_a2_dz_range[1]))
        H = self.horizon
        tau = self.aug_a2_tau
        w = np.exp(-np.arange(H) / tau)
        w = w / w.sum()
        max_corr_dpos = dz_max * self.aug_a2_k * w[0]

        cprint(f'[A2-Verify] max|dz|={dz_max:.3f}m, tau={tau}, H={H}, '
               f'max|corr_dpos[0]|={max_corr_dpos:.4f}m/step', 'cyan')
        cprint(f'[A2-Verify] 参考: dpos_z p95≈0.079m (zarr 统计)', 'cyan')
        if max_corr_dpos > 0.10:
            cprint(f'[A2-WARN] max|corr_dpos|={max_corr_dpos:.4f} > 0.10, '
                   f'建议增大 tau 或减小 k 避免激进 z 机动', 'yellow')

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

    # ═══════════════════════════════════════════════════════════════
    # A1: 世界平移增广
    # ═══════════════════════════════════════════════════════════════

    def _augment_a1_world_translation(self, pc, st, act):
        """
        A1: z + x,y 世界平移 — 对称变换，教模型"绝对坐标不重要"。

        修改:
          - state pos_world[:, 0:3] += [dxy_x, dxy_y, dz]
          - goal_body: 不变 (相对量)
          - 点云: 不变 (body frame)
          - action: 不变 (world-frame 位移量，平移不变)
          - dist: 不变 (||goal_body||)

        [FACT] A1 是纯对称变换，零风险。
        """
        dz = np.random.uniform(*self.aug_a1_dz_range)
        dxy = np.random.uniform(*self.aug_a1_dxy_range, size=2)

        st = st.copy()
        st[:, 0] += dxy[0]  # pos_world x
        st[:, 1] += dxy[1]  # pos_world y
        st[:, 2] += dz       # pos_world z

        # goal_body (st[:, 3:6]), dist (st[:, 6]), sin/cos_yaw (st[:, 7:9])
        # 全部不变 — 相对量在平移下不变
        # pc 不变 — body frame
        # action 不变 — world-frame delta 在平移下不变

        return pc, st, act  # pc 和 act 未被修改

    # ═══════════════════════════════════════════════════════════════
    # A2: z 扰动 + 合成回正
    # ═══════════════════════════════════════════════════════════════

    def _augment_a2_z_perturb_recovery(self, pc, st, act):
        """
        A2: z 扰动 + 合成回正 — 制造 goal_bz≠0 的监督配对。

        物理含义: 只有无人机偏移了 Δz，场景与目标不动。
        专家若在此偏移位置，正确行为 = 原计划 + 回正。

        必须同步修改的四个量:
          1. state 绝对 z: pos_world[:, 2] += dz
          2. 目标相对位置: goal_body[:, 2] -= dz
          3. 点云 z 同步平移: point_cloud[..., 2] -= dz
          4. action 回正 (取决于 modify_action 模式):
             - modify_action=True: 三通道 dpos_z + dvel_z + acc_z 指数衰减回正
             - modify_action=False (aux loss 版): 不改 action，由 policy loss 监督

        Returns:
            (pc, st, act, dz) — dz 用于 aux loss 计算
        """
        T, N, C = pc.shape
        H = act.shape[0]

        dz = np.random.uniform(*self.aug_a2_dz_range)

        # ── 1. state 绝对 z ──
        st = st.copy()
        st[:, 2] += dz  # pos_world z

        # ── 2. 目标相对位置 z ──
        st[:, 5] -= dz  # goal_body z (goal_body = [gx, gy, gz])

        # ── 3. 更新 dist ──
        st[:, 6] = np.sqrt(st[:, 3]**2 + st[:, 4]**2 + st[:, 5]**2)

        # ── 4. 点云 z 同步平移 ──
        pc = pc.copy()
        pc[..., 2] -= dz

        # P0 验证: 打印前 N 次增广的点云 z 范围
        if self.aug_a2_verify_mode and self._a2_verify_count < self._a2_verify_max:
            self._a2_verify_count += 1
            pc_z = pc[..., 2]
            cprint(f'[A2-P0-Verify #{self._a2_verify_count}] '
                   f'dz={dz:+.4f}m | '
                   f'pc_z range: [{pc_z.min():.3f}, {pc_z.max():.3f}] | '
                   f'expected ground ~{-(1.5+dz):.2f} (原 1.5m 高度下) | '
                   f'pos_world_z: {st[0, 2]:.3f} | '
                   f'goal_bz: {st[0, 5]:.3f} | '
                   f'modify_action={self.aug_a2_modify_action}',
                   'cyan')

        # ── 5. action 回正 ──
        if self.aug_a2_modify_action:
            # 三通道一致回正 (原始激进版，需 SEMANTICS_CONFIRMED 才能用)
            act = act.copy()
            dt = self.aug_a2_traj_dt

            w = np.exp(-np.arange(H) / self.aug_a2_tau)
            w = w / w.sum()

            corr_dpos = -dz * self.aug_a2_k * w  # (H,)
            act[:, 2] += corr_dpos.astype(np.float32)  # dpos_z

            corr_dvel = np.diff(corr_dpos, prepend=0.0) / dt  # (H,)
            act[:, 5] += corr_dvel.astype(np.float32)  # dvel_z

            corr_dacc = np.diff(corr_dvel, prepend=0.0) / dt  # (H,)
            act[:, 8] += corr_dacc.astype(np.float32)  # acc_z
        # else: aux loss 版 — 不改 action，返回 dz 由 policy loss 监督

        return pc, st, act, dz

    # ═══════════════════════════════════════════════════════════════
    # 主增广方法 (修改自 drone_dataset_v9)
    # ═══════════════════════════════════════════════════════════════

    def _augment(self, pc, state, action):
        """
        增广流水线:
          0. 原始 yaw 旋转 + PC dropout/jitter/noise (继承 v9)
          1. A1 世界平移 (p=1.0)
          2. A2 z 扰动 + 回正 (p=0.5)

        注意: A1 在 yaw 旋转后执行。因为 A1 平移 state pos_world
        (已在 yaw 旋转中同步旋转)，平移不破坏旋转一致性。
        """
        T, N, C = pc.shape
        pc, st, act = pc.copy(), state.copy(), action.copy()

        # ── 原始增广: Dropout ──
        keep = np.random.uniform(*self.pc_dropout_range)
        n_keep = max(1, int(N * keep))
        for t in range(T):
            idx = np.random.choice(N, n_keep, replace=False)
            mask = np.zeros(N, dtype=bool); mask[idx] = True
            pc[t, ~mask, :] = 0.0

        # ── 原始增广: Jitter ──
        if self.pc_jitter_std > 0:
            pc[:, :, :3] += np.random.randn(T, N, 3).astype(np.float32) * self.pc_jitter_std

        # ── 原始增广: Depth noise ──
        if self.pc_depth_noise_coeff > 0:
            d = np.linalg.norm(pc[:, :, :3], axis=-1, keepdims=True)
            s = np.minimum(self.pc_depth_noise_coeff * d, self.pc_depth_noise_max)
            pc[:, :, :3] += np.random.randn(T, N, 3).astype(np.float32) * s

        # ── 原始增广: Yaw rotation ──
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

        # ═════════════════════════════════════════════════
        # Phase α-aug: A1 世界平移 (p=1.0)
        # ═════════════════════════════════════════════════
        if self.aug_a1_enable:
            pc, st, act = self._augment_a1_world_translation(pc, st, act)

        # ═════════════════════════════════════════════════
        # Phase α-aug: A2 z 扰动 + 合成回正 (p=0.5)
        # ═════════════════════════════════════════════════
        aug_dz = 0.0
        if self.aug_a2_enable and np.random.rand() < self.aug_a2_prob:
            pc, st, act, dz = self._augment_a2_z_perturb_recovery(pc, st, act)
            aug_dz = float(dz)

        return pc, st, act, aug_dz

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
            T_total = agent_pos.shape[0]
            obs_len = self.pad_before + 1  # n_obs_steps
            obs_indices = np.arange(obs_len) * self.obs_stride
            obs_indices = obs_indices[obs_indices < T_total]
            agent_pos_obs = agent_pos[obs_indices]
        else:
            agent_pos_obs = agent_pos

        point_cloud = uniform_sampling_numpy(point_cloud, self.num_points)

        if self._is_train and self.augment_pc:
            point_cloud, agent_pos, action, aug_dz = self._augment(
                point_cloud, agent_pos, action)
        else:
            aug_dz = 0.0

        # α2: 计算 min_dist（在增广后，因为增广改变了点云）
        pc_tensor = torch.from_numpy(point_cloud) if isinstance(point_cloud, np.ndarray) else point_cloud
        min_dist = self._compute_min_dist(pc_tensor)

        return {
            'obs': {'agent_pos': agent_pos, 'point_cloud': point_cloud},
            'action': action,
            'min_dist': np.float32(min_dist),  # α2: 用于样本重加权
            'aug_dz': np.float32(aug_dz),       # A2 aux loss: z 扰动值
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
