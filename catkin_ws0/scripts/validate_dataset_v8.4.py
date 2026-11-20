#!/usr/bin/env python3
"""
validate_dataset_v8.py — v8 数据集验证工具
================================================================================
检查经过 bag2zarr_body_v5 → merge_zarr_v8 → convert_v5_to_v6lite 流水线
转换后的数据集是否满足 v8.4 设计要求。

自动检测数据格式（15d / 9d），运行全套验证并输出报告。

用法：
  # 检查合并后的 15d 数据集
  python3 validate_dataset_v8.py ~/AUV/avoidance_dataset_v8/merged_body_v8_15d.zarr

  # 检查转换后的 9d 训练数据集
  python3 validate_dataset_v8.py ~/AUV/avoidance_dataset_v8/merged_body_v8.zarr

  # 检查单个 episode zarr（bag2zarr 直接输出）
  python3 validate_dataset_v8.py ~/AUV/avoidance_dataset_v8/avoidance_dataset_1.zarr

  # 指定训练参数验证兼容性
  python3 validate_dataset_v8.py ~/AUV/.../merged.zarr \\
    --n_obs_steps 3 --n_action_steps 8 --horizon 16

检查项：
  [结构] zarr 目录结构、key 完整性
  [维度] state/action/point_cloud shape 匹配
  [完整性] episode_ends 单调递增、总帧数一致、无空 episode
  [数值] 无 NaN/Inf、各维度范围合理
  [state] 位置/速度/yaw/dist 合理性
  [action] Δpos/Δyaw 范围、异常大值检测（EP33 类型）
  [点云] 非全零、范围合理、覆盖率
  [训练] episode 最短长度 vs 训练参数兼容性
  [normalizer] Gaussian normalizer 兼容性（极端 outlier 检测）
  [copycat] vel→action R² 检测（仅 15d，检查捷径风险）
"""

import numpy as np
import zarr
import os
import sys
import argparse
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

# 15d state layout
STATE_15D_NAMES = [
    'pos_x', 'pos_y', 'pos_z',        # [0:3]
    'vel_x', 'vel_y', 'vel_z',        # [3:6]
    'acc_x', 'acc_y', 'acc_z',        # [6:9]
    'goal_bx', 'goal_by', 'goal_bz',  # [9:12]
    'dist',                            # [12]
    'sin_yaw',                         # [13]
    'cos_yaw',                         # [14]
]

# 9d state layout (convert 后)
STATE_9D_NAMES = [
    'pos_x', 'pos_y', 'pos_z',        # [0:2]
    'goal_bx', 'goal_by', 'goal_bz',  # [3:5]
    'dist',                            # [6]
    'sin_yaw',                         # [7]
    'cos_yaw',                         # [8]
]

# 11d action layout
ACTION_NAMES = [
    'dpos_x', 'dpos_y', 'dpos_z',     # [0:3]
    'dvel_x', 'dvel_y', 'dvel_z',     # [3:6]
    'acc_x', 'acc_y', 'acc_z',        # [6:9]
    'dyaw',                            # [9]
    'yaw_rate',                        # [10]
]


class ValidationResult:
    """单条验证结果"""
    def __init__(self, name, passed, message, severity='INFO'):
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity  # INFO, WARN, FAIL

    def __str__(self):
        icon = '✓' if self.passed else ('⚠' if self.severity == 'WARN' else '✗')
        return f"  {icon} [{self.severity}] {self.name}: {self.message}"


class DatasetValidator:
    """v8 数据集验证器"""

    def __init__(self, zarr_path, n_obs_steps=3, n_action_steps=8, horizon=16):
        self.zarr_path = zarr_path
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.results = []
        self.root = None
        self.state = None
        self.action = None
        self.episode_ends = None
        self.ep_starts = None
        self.state_dim = None
        self.is_15d = None

    def add(self, name, passed, message, severity='INFO'):
        r = ValidationResult(name, passed, message, severity)
        self.results.append(r)
        return r

    def fail(self, name, message):
        return self.add(name, False, message, 'FAIL')

    def warn(self, name, message):
        return self.add(name, False, message, 'WARN')

    def ok(self, name, message):
        return self.add(name, True, message, 'INFO')

    # ── 1. 结构检查 ──

    def check_structure(self):
        """检查 zarr 目录结构"""
        if not os.path.exists(self.zarr_path):
            self.fail('存在性', f'路径不存在: {self.zarr_path}')
            return False

        try:
            self.root = zarr.open(str(self.zarr_path), mode='r')
        except Exception as e:
            self.fail('zarr打开', f'无法打开: {e}')
            return False

        # 必需的 key
        required_keys = {
            'data/state': 'state 数组',
            'data/action': 'action 数组',
            'data/point_cloud': 'point_cloud 数组',
            'meta/episode_ends': 'episode_ends 数组',
        }
        all_present = True
        for key, desc in required_keys.items():
            parts = key.split('/')
            node = self.root
            found = True
            for p in parts:
                if p in node:
                    node = node[p]
                else:
                    found = False
                    break
            if found:
                self.ok(f'key:{key}', f'{desc} 存在')
            else:
                self.fail(f'key:{key}', f'{desc} 缺失')
                all_present = False

        return all_present

    # ── 2. 维度检查 ──

    def check_dimensions(self):
        """检查各数组的 shape"""
        state_shape = self.root['data']['state'].shape
        action_shape = self.root['data']['action'].shape
        pc_shape = self.root['data']['point_cloud'].shape
        ends_shape = self.root['meta']['episode_ends'].shape

        N = state_shape[0]
        self.state_dim = state_shape[1]
        self.is_15d = (self.state_dim == 15)

        # state 维度
        if self.state_dim == 15:
            self.ok('state维度', f'shape={state_shape}，15d 格式（合并后未降维）')
        elif self.state_dim == 9:
            self.ok('state维度', f'shape={state_shape}，9d 格式（已降维，训练用）')
        else:
            self.fail('state维度', f'shape={state_shape}，期望 15d 或 9d，实际 {self.state_dim}d')

        # action 维度
        action_ndim = len(action_shape)
        if action_ndim == 2:
            if action_shape[1] != 11:
                self.fail('action维度', f'shape={action_shape}，期望 action_dim=11')
            elif action_shape[0] != N:
                self.fail('action维度', f'帧数不匹配: state={N}, action={action_shape[0]}')
            else:
                self.ok('action维度', f'shape={action_shape}，11d per-frame')
        elif action_ndim == 3:
            if action_shape[2] != 11:
                self.fail('action维度', f'shape={action_shape}，期望 action_dim=11')
            elif action_shape[0] != N:
                self.fail('action维度', f'帧数不匹配: state={N}, action={action_shape[0]}')
            else:
                self.ok('action维度', f'shape={action_shape}，chunk-wise horizon={action_shape[1]}')
        else:
            self.fail('action维度', f'未知 action ndim={action_ndim}')

        # point_cloud 维度
        if len(pc_shape) != 3:
            self.fail('点云维度', f'期望 3D，实际 {len(pc_shape)}D shape={pc_shape}')
        else:
            if pc_shape[0] != N:
                self.fail('点云维度', f'帧数不匹配: state={N}, pc={pc_shape[0]}')
            if pc_shape[1] != 4096:
                self.warn('点云点数', f'期望 4096 点，实际 {pc_shape[1]}')
            if pc_shape[2] != 6:
                self.warn('点云通道', f'期望 6 通道 (xyz+color)，实际 {pc_shape[2]}')
            self.ok('点云维度', f'shape={pc_shape}')

        return True

    # ── 3. Episode 完整性 ──

    def check_episode_integrity(self):
        """检查 episode_ends 一致性"""
        self.episode_ends = np.array(self.root['meta']['episode_ends'][:])
        N = self.root['data']['state'].shape[0]
        n_eps = len(self.episode_ends)

        if n_eps == 0:
            self.fail('episode数量', '没有 episode')
            return False

        self.ok('episode数量', f'{n_eps} episodes')

        # 单调递增
        if np.all(np.diff(self.episode_ends) > 0):
            self.ok('episode单调性', 'episode_ends 严格单调递增')
        else:
            bad = np.where(np.diff(self.episode_ends) <= 0)[0]
            self.fail('episode单调性',
                      f'episode_ends 不单调，违反位置: {bad[:5].tolist()}...')

        # 最后一个 end = N
        last_end = int(self.episode_ends[-1])
        if last_end == N:
            self.ok('episode总帧数', f'episode_ends[-1]={last_end} == N={N}')
        else:
            self.fail('episode总帧数', f'episode_ends[-1]={last_end} != N={N}')

        # episode 长度
        self.ep_starts = np.concatenate([[0], self.episode_ends[:-1]]).astype(int)
        lengths = self.episode_ends.astype(int) - self.ep_starts

        self.ok('episode长度',
                f'min={lengths.min()}, max={lengths.max()}, '
                f'mean={lengths.mean():.1f}, median={np.median(lengths):.1f}')

        # 最短长度 vs 训练参数
        min_required = self.n_obs_steps + self.n_action_steps - 1
        short_eps = (lengths < min_required).sum()
        if short_eps > 0:
            self.fail('episode最短长度',
                      f'{short_eps} episodes 长度 < {min_required} '
                      f'(n_obs={self.n_obs_steps} + n_act={self.n_action_steps} - 1)，'
                      f'训练时会 index-out-of-bounds')
        else:
            self.ok('episode最短长度',
                    f'所有 episode ≥ {min_required} 步 (min={lengths.min()})，'
                    f'与训练参数兼容')

        # 空 episode
        empty = (lengths == 0).sum()
        if empty > 0:
            self.fail('空episode', f'{empty} 个空 episode')

        # n_action_steps 约束
        if self.n_action_steps <= self.horizon - self.n_obs_steps + 1:
            self.ok('action_steps约束',
                    f'n_action_steps={self.n_action_steps} <= '
                    f'horizon-n_obs+1={self.horizon - self.n_obs_steps + 1}')
        else:
            self.fail('action_steps约束',
                      f'n_action_steps={self.n_action_steps} > '
                      f'horizon-n_obs+1={self.horizon - self.n_obs_steps + 1}，'
                      f'会导致运行时崩溃！')

        return True

    # ── 4. 数值检查 ──

    def check_numerical(self):
        """检查 NaN / Inf"""
        self.state = self.root['data']['state'][:]
        action_arr = self.root['data']['action']
        if action_arr.ndim == 2:
            self.action = action_arr[:]
        else:
            self.action = action_arr[:, 0, :]   # 取第一步

        # NaN
        state_nan = np.isnan(self.state).sum()
        action_nan = np.isnan(self.action).sum()
        if state_nan > 0:
            self.fail('state NaN', f'{state_nan} 个 NaN 值')
        else:
            self.ok('state NaN', '无 NaN')

        if action_nan > 0:
            self.fail('action NaN', f'{action_nan} 个 NaN 值')
        else:
            self.ok('action NaN', '无 NaN')

        # Inf
        state_inf = np.isinf(self.state).sum()
        action_inf = np.isinf(self.action).sum()
        if state_inf > 0:
            self.fail('state Inf', f'{state_inf} 个 Inf 值')
        else:
            self.ok('state Inf', '无 Inf')

        if action_inf > 0:
            self.fail('action Inf', f'{action_inf} 个 Inf 值')
        else:
            self.ok('action Inf', '无 Inf')

    # ── 5. State 范围检查 ──

    def check_state_ranges(self):
        """检查各 state 维度的值域"""
        names = STATE_15D_NAMES if self.is_15d else STATE_9D_NAMES

        print(f"\n  ── State 各维度统计 ({self.state_dim}d) ──")
        print(f"  {'维度':<12} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}")
        print(f"  {'─'*54}")
        for i, name in enumerate(names):
            col = self.state[:, i]
            print(f"  {name:<12} {col.min():>10.4f} {col.max():>10.4f} "
                  f"{col.mean():>10.4f} {col.std():>10.4f}")

        # dist_to_goal
        dist_idx = 12 if self.is_15d else 6
        dist = self.state[:, dist_idx]
        if dist.min() < 0:
            self.fail('dist范围', f'dist_to_goal 有负值: min={dist.min():.4f}')
        else:
            self.ok('dist范围', f'dist ≥ 0, range=[{dist.min():.2f}, {dist.max():.2f}]m')

        if dist.max() > 50:
            self.warn('dist范围', f'dist 最大值 {dist.max():.2f}m > 50m，可能有异常数据')

        # sin/cos yaw
        sin_idx = 13 if self.is_15d else 7
        cos_idx = 14 if self.is_15d else 8
        sin_yaw = self.state[:, sin_idx]
        cos_yaw = self.state[:, cos_idx]

        sin_ok = np.all(np.abs(sin_yaw) <= 1.01)
        cos_ok = np.all(np.abs(cos_yaw) <= 1.01)
        if sin_ok and cos_ok:
            self.ok('sin/cos范围', f'sin∈[{sin_yaw.min():.4f},{sin_yaw.max():.4f}], '
                    f'cos∈[{cos_yaw.min():.4f},{cos_yaw.max():.4f}]')
        else:
            self.fail('sin/cos范围', f'sin/cos 超出 [-1,1]: '
                      f'sin∈[{sin_yaw.min():.4f},{sin_yaw.max():.4f}], '
                      f'cos∈[{cos_yaw.min():.4f},{cos_yaw.max():.4f}]')

        # sin²+cos² ≈ 1
        norm = sin_yaw**2 + cos_yaw**2
        norm_err = np.abs(norm - 1.0)
        if norm_err.max() < 0.05:
            self.ok('sin²+cos²', f'max error={norm_err.max():.6f} < 0.05')
        else:
            bad_frames = (norm_err > 0.05).sum()
            self.warn('sin²+cos²',
                      f'{bad_frames} 帧 sin²+cos² 偏离 1.0 > 0.05, '
                      f'max_err={norm_err.max():.4f}')

        # 位置范围（local frame, spawn at -12,0,0）
        pos = self.state[:, 0:3]
        pos_range = np.abs(pos).max(axis=0)
        if pos_range.max() > 100:
            self.warn('位置范围', f'pos 最大绝对值 {pos_range.max():.1f}m > 100m，可能有飞出边界的数据')
        else:
            self.ok('位置范围', f'pos 范围合理，max|pos|={pos_range}')

        # 15d 特有检查
        if self.is_15d:
            vel = self.state[:, 3:6]
            vel_mag = np.linalg.norm(vel, axis=1)
            if vel_mag.max() > 10:
                self.warn('速度范围', f'vel 最大值 {vel_mag.max():.2f}m/s > 10m/s')
            else:
                self.ok('速度范围', f'vel max={vel_mag.max():.2f}m/s')

    # ── 6. Action 范围检查 ──

    def check_action_ranges(self):
        """检查 action 各维度的值域 + 异常检测"""
        print(f"\n  ── Action 各维度统计 (11d) ──")
        print(f"  {'维度':<12} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}")
        print(f"  {'─'*54}")
        for i, name in enumerate(ACTION_NAMES):
            col = self.action[:, i]
            print(f"  {name:<12} {col.min():>10.4f} {col.max():>10.4f} "
                  f"{col.mean():>10.4f} {col.std():>10.4f}")

        # dpos 范围
        dpos = self.action[:, 0:3]
        dpos_mag = np.linalg.norm(dpos, axis=1)
        if dpos_mag.max() > 5.0:
            self.warn('dpos范围', f'|Δpos| max={dpos_mag.max():.4f}m > 5.0m/step，'
                      f'检查是否有异常 episode')
        else:
            self.ok('dpos范围', f'|Δpos| max={dpos_mag.max():.4f}m')

        # dyaw 范围
        dyaw = self.action[:, 9]
        if np.abs(dyaw).max() > np.pi:
            bad = (np.abs(dyaw) > np.pi).sum()
            self.warn('dyaw范围', f'{bad} 帧 |Δyaw| > π，可能有角度未 normalize')
        else:
            self.ok('dyaw范围', f'|Δyaw| max={np.abs(dyaw).max():.4f} rad '
                    f'({np.degrees(np.abs(dyaw).max()):.1f}°)')

        # ── EP33 类型异常检测：per-episode |Δpos| 均值 ──
        print(f"\n  ── 异常 episode 检测 (EP33类型) ──")
        ep_dpos_means = []
        for i in range(len(self.episode_ends)):
            s = int(self.ep_starts[i])
            e = int(self.episode_ends[i])
            ep_dpos = np.linalg.norm(self.action[s:e, 0:3], axis=1)
            ep_dpos_means.append(ep_dpos.mean())

        ep_dpos_means = np.array(ep_dpos_means)
        global_mean = ep_dpos_means.mean()
        global_std = ep_dpos_means.std()

        # 超过 3σ 的 episode
        threshold_3sigma = global_mean + 3 * global_std
        outlier_eps = np.where(ep_dpos_means > threshold_3sigma)[0]

        if len(outlier_eps) > 0:
            self.warn('异常episode',
                      f'{len(outlier_eps)} 个 episode |Δpos| 均值 > 3σ '
                      f'(thresh={threshold_3sigma:.4f}m)')
            for idx in outlier_eps[:10]:
                s = int(self.ep_starts[idx])
                e = int(self.episode_ends[idx])
                print(f"    EP{idx}: |Δpos| mean={ep_dpos_means[idx]:.4f}m, "
                      f"len={e-s}, range=[{s},{e})")
        else:
            self.ok('异常episode', f'无 3σ 异常 episode '
                    f'(mean={global_mean:.4f}, std={global_std:.4f})')

        # 极端 outlier（>= 2m/step，类似 EP33 的 11.7m）
        extreme = np.where(ep_dpos_means > 2.0)[0]
        if len(extreme) > 0:
            self.fail('极端异常',
                      f'{len(extreme)} 个 episode |Δpos| 均值 > 2.0m/step！'
                      f'会严重损坏 gaussian normalizer')
            for idx in extreme[:5]:
                print(f"    EP{idx}: mean={ep_dpos_means[idx]:.4f}m ← 必须删除")
        else:
            self.ok('极端异常', '无 |Δpos| > 2.0m/step 的 episode')

    # ── 7. 点云检查 ──

    def check_point_cloud(self, sample_n=50):
        """抽样检查点云质量"""
        pc_arr = self.root['data']['point_cloud']
        N = pc_arr.shape[0]
        sample_idx = np.random.choice(N, min(sample_n, N), replace=False)

        n_all_zero = 0
        n_sparse = 0
        xyz_ranges = []

        for idx in sample_idx:
            pc = np.array(pc_arr[idx])  # (4096, 6)
            xyz = pc[:, :3]

            # 全零检测
            if np.all(np.abs(xyz) < 1e-6):
                n_all_zero += 1
                continue

            # 非零点数
            nonzero = np.any(np.abs(xyz) > 1e-6, axis=1).sum()
            if nonzero < 100:
                n_sparse += 1

            xyz_ranges.append(np.abs(xyz).max())

        if n_all_zero > 0:
            pct = n_all_zero / len(sample_idx) * 100
            if pct > 20:
                self.warn('点云全零', f'{n_all_zero}/{len(sample_idx)} ({pct:.0f}%) '
                          f'帧点云全零（可能是空旷环境的正常现象）')
            else:
                self.ok('点云全零', f'{n_all_zero}/{len(sample_idx)} 帧全零')
        else:
            self.ok('点云全零', f'抽样 {len(sample_idx)} 帧，无全零')

        if n_sparse > 0:
            self.warn('点云稀疏', f'{n_sparse}/{len(sample_idx)} 帧非零点 < 100')
        else:
            self.ok('点云密度', f'抽样帧点云密度正常')

        if xyz_ranges:
            max_range = max(xyz_ranges)
            if max_range > 50:
                self.warn('点云范围', f'xyz 最大值 {max_range:.1f}m > 50m')
            else:
                self.ok('点云范围', f'xyz max={max_range:.2f}m')

    # ── 8. Normalizer 兼容性 ──

    def check_normalizer_compat(self):
        """检查 gaussian normalizer 是否会被极端值破坏"""
        print(f"\n  ── Gaussian Normalizer 兼容性 ──")

        # state 各维度的 std
        for i in range(self.state_dim):
            col = self.state[:, i]
            std = col.std()
            if std < 1e-6:
                name = (STATE_15D_NAMES[i] if self.is_15d else STATE_9D_NAMES[i])
                self.warn(f'state[{i}] std', f'{name} std={std:.2e} ≈ 0，'
                          f'gaussian normalizer 会除零')

        # action 各维度
        for i in range(11):
            col = self.action[:, i]
            std = col.std()
            if std < 1e-6:
                self.warn(f'action[{i}] std', f'{ACTION_NAMES[i]} std={std:.2e} ≈ 0')

        # 检查 state/action 各维度的 kurtosis（尖峰=有极端 outlier）
        from scipy import stats as sp_stats
        state_kurtosis = []
        for i in range(self.state_dim):
            k = sp_stats.kurtosis(self.state[:, i])
            state_kurtosis.append(k)
            if k > 20:
                name = (STATE_15D_NAMES[i] if self.is_15d else STATE_9D_NAMES[i])
                self.warn(f'state kurtosis',
                          f'{name} kurtosis={k:.1f} > 20，'
                          f'有极端 outlier 可能影响 gaussian normalizer')

        action_kurtosis = []
        for i in range(11):
            k = sp_stats.kurtosis(self.action[:, i])
            action_kurtosis.append(k)
            if k > 20:
                self.warn(f'action kurtosis',
                          f'{ACTION_NAMES[i]} kurtosis={k:.1f} > 20')

        if max(state_kurtosis) <= 20 and max(action_kurtosis) <= 20:
            self.ok('kurtosis', f'所有维度 kurtosis ≤ 20，normalizer 安全')

    # ── 9. Copycat 检测（仅 15d）──

    def check_copycat(self):
        """检测 vel→action 捷径风险（仅 15d state）"""
        if not self.is_15d:
            self.ok('copycat', '9d state 已移除 vel，无 vel→action 捷径风险')
            return

        print(f"\n  ── Copycat 检测 (vel→action R²) ──")
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            self.warn('copycat', 'sklearn 未安装，跳过 copycat 检测')
            return

        vel = self.state[:, 3:6]   # (N, 3)
        action_dpos = self.action[:, 0:3]  # (N, 3)

        r2_scores = []
        for i in range(3):
            r2 = LinearRegression().fit(vel, action_dpos[:, i]).score(vel, action_dpos[:, i])
            r2_scores.append(r2)
            dim_name = ['Δpx', 'Δpy', 'Δpz'][i]
            print(f"    R²(vel→{dim_name}) = {r2:.4f}")

        max_r2 = max(r2_scores)
        if max_r2 > 0.8:
            self.warn('copycat',
                      f'vel→action 最大 R²={max_r2:.4f} > 0.8，'
                      f'存在 copycat 捷径风险！'
                      f'训练时应使用 9d state（删除 vel）')
        elif max_r2 > 0.6:
            self.warn('copycat',
                      f'vel→action 最大 R²={max_r2:.4f} > 0.6，'
                      f'中等捷径风险，建议使用 9d state')
        else:
            self.ok('copycat', f'vel→action 最大 R²={max_r2:.4f} ≤ 0.6')

    # ── 10. Episode 质量统计 ──

    def check_episode_quality(self):
        """per-episode 质量统计"""
        print(f"\n  ── Episode 质量统计 ──")

        dist_idx = 12 if self.is_15d else 6
        n_eps = len(self.episode_ends)

        min_dists = []
        final_dists = []
        ep_lengths = []

        for i in range(n_eps):
            s = int(self.ep_starts[i])
            e = int(self.episode_ends[i])
            dist = self.state[s:e, dist_idx]
            min_dists.append(dist.min())
            final_dists.append(dist[-1])
            ep_lengths.append(e - s)

        min_dists = np.array(min_dists)
        final_dists = np.array(final_dists)
        ep_lengths = np.array(ep_lengths)

        n_success = (min_dists < 0.5).sum()
        n_partial = ((min_dists >= 0.5) & (min_dists < 1.5)).sum()
        n_incomplete = (min_dists >= 1.5).sum()

        print(f"  到达统计 (min_dist):")
        print(f"    成功 (<0.5m):  {n_success}/{n_eps} ({n_success/n_eps*100:.1f}%)")
        print(f"    部分 (<1.5m):  {n_partial}/{n_eps} ({n_partial/n_eps*100:.1f}%)")
        print(f"    未完成 (≥1.5m): {n_incomplete}/{n_eps} ({n_incomplete/n_eps*100:.1f}%)")
        print(f"    min_dist: min={min_dists.min():.3f} max={min_dists.max():.3f} "
              f"mean={min_dists.mean():.3f}")
        print(f"    final_dist: mean={final_dists.mean():.3f} "
              f"median={np.median(final_dists):.3f}")

        if n_success / n_eps < 0.5:
            self.warn('到达率', f'仅 {n_success/n_eps*100:.1f}% episode 到达目标 (dist<0.5m)，'
                      f'可能需要更多数据或降低难度')
        else:
            self.ok('到达率', f'{n_success/n_eps*100:.1f}% episode 到达目标')

        # episode 数量
        if n_eps >= 900:
            self.ok('episode数量', f'{n_eps} ≥ 900 (v8 目标)')
        elif n_eps >= 500:
            self.warn('episode数量', f'{n_eps} < 900 (v8 目标)，但可能足够初步训练')
        else:
            self.warn('episode数量', f'{n_eps} < 500，数据量可能不足')

        # 总帧数
        total = self.state.shape[0]
        if total >= 25000:
            self.ok('总帧数', f'{total} ≥ 25000')
        else:
            self.warn('总帧数', f'{total} < 25000，数据量偏少')

    # ── 11. 跨 episode 污染检测 ──

    def check_cross_episode(self):
        """检查 episode 边界是否有数据穿越"""
        dist_idx = 12 if self.is_15d else 6

        n_suspicious = 0
        for i in range(len(self.episode_ends) - 1):
            e = int(self.episode_ends[i])
            # episode i 的最后一帧 和 episode i+1 的第一帧
            if e < self.state.shape[0]:
                # 位置跳变应该很大（不同起点）
                pos_diff = np.linalg.norm(
                    self.state[e, 0:3] - self.state[e - 1, 0:3])
                if pos_diff < 0.01:
                    n_suspicious += 1

        if n_suspicious > len(self.episode_ends) * 0.1:
            self.warn('跨episode',
                      f'{n_suspicious} 个 episode 边界位置差 < 0.01m，'
                      f'可能有数据污染或重复')
        else:
            self.ok('跨episode', f'episode 边界位置跳变正常')

    # ── 运行全部检查 ──

    def run(self):
        """运行全部验证"""
        print("═" * 75)
        print(f"  validate_dataset_v8 — 数据集验证")
        print(f"  路径: {self.zarr_path}")
        print(f"  训练参数: n_obs={self.n_obs_steps}, n_act={self.n_action_steps}, "
              f"horizon={self.horizon}")
        print("═" * 75)

        # 1. 结构
        print("\n[1/9] 结构检查...")
        if not self.check_structure():
            self._print_summary()
            return

        # 2. 维度
        print("\n[2/9] 维度检查...")
        self.check_dimensions()

        # 3. Episode 完整性
        print("\n[3/9] Episode 完整性...")
        if not self.check_episode_integrity():
            self._print_summary()
            return

        # 4. 数值
        print("\n[4/9] 数值检查...")
        self.check_numerical()

        # 5. State 范围
        print("\n[5/9] State 范围...")
        self.check_state_ranges()

        # 6. Action 范围
        print("\n[6/9] Action 范围...")
        self.check_action_ranges()

        # 7. 点云
        print("\n[7/9] 点云抽样检查...")
        self.check_point_cloud()

        # 8. Normalizer
        print("\n[8/9] Normalizer 兼容性...")
        try:
            self.check_normalizer_compat()
        except ImportError:
            self.warn('normalizer', 'scipy 未安装，跳过 kurtosis 检查')

        # 9. Copycat + Episode 质量 + 跨 episode
        print("\n[9/9] 综合检查...")
        self.check_copycat()
        self.check_episode_quality()
        self.check_cross_episode()

        self._print_summary()

    def _print_summary(self):
        """打印汇总报告"""
        n_pass = sum(1 for r in self.results if r.passed)
        n_warn = sum(1 for r in self.results if not r.passed and r.severity == 'WARN')
        n_fail = sum(1 for r in self.results if not r.passed and r.severity == 'FAIL')

        print(f"\n{'═' * 75}")
        print(f"  验证汇总")
        print(f"{'═' * 75}")

        if n_fail > 0:
            print(f"\n  ✗ FAIL ({n_fail} 项):")
            for r in self.results:
                if not r.passed and r.severity == 'FAIL':
                    print(f"    {r}")

        if n_warn > 0:
            print(f"\n  ⚠ WARN ({n_warn} 项):")
            for r in self.results:
                if not r.passed and r.severity == 'WARN':
                    print(f"    {r}")

        print(f"\n  统计: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
        print(f"        共 {len(self.results)} 项检查")

        if n_fail == 0 and n_warn == 0:
            print(f"\n  ★ 数据集完全通过验证，可以直接用于训练")
        elif n_fail == 0:
            print(f"\n  ◆ 数据集有 {n_warn} 个警告，建议检查后再训练")
        else:
            print(f"\n  ✗ 数据集有 {n_fail} 个错误，必须修复后才能训练")

        print(f"{'═' * 75}")


def main():
    parser = argparse.ArgumentParser(
        description='validate_dataset_v8: 验证 v8 数据集是否满足训练要求')
    parser.add_argument('zarr_path', type=str,
                        help='zarr 数据集路径')
    parser.add_argument('--n_obs_steps', type=int, default=3,
                        help='训练 n_obs_steps (default 3)')
    parser.add_argument('--n_action_steps', type=int, default=8,
                        help='训练 n_action_steps (default 8)')
    parser.add_argument('--horizon', type=int, default=16,
                        help='训练 horizon (default 16)')

    args = parser.parse_args()

    validator = DatasetValidator(
        args.zarr_path,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        horizon=args.horizon,
    )
    validator.run()


if __name__ == '__main__':
    main()