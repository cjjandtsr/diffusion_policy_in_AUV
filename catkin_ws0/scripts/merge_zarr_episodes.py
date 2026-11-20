#!/usr/bin/env python3
"""
merge_zarr_v8.py — v8 合并工具（修正版卡住截断）
================================================================================
修正原 merge_zarr_v8.4.py 中的三个 bug：
  Bug 1: stall_length = consecutive_stall_windows * stall_window 严重高估实际卡住步数
  Bug 2: first_stall_window_start 指向末尾而非卡住起点
  Bug 3: Bug1 导致 transient stall 机制失效

修正后算法：
  1. 先检查末尾是否卡住（最后 stall_window 步无进展）
  2. 如果末尾正常 → 保留全部（中途停顿自动被保留）
  3. 如果末尾卡住 → 从末尾往前扫描，找到卡住起始点 → 截断

这自然处理了"中途停顿后恢复"的场景：只要末尾在推进，整条轨迹保留。

用法：
  python3 merge_zarr_v8.py \\
    --input_dir ~/AUV/avoidance_dataset_v8 \\
    --output ~/AUV/avoidance_dataset_v8/merged_body_v8_15d.zarr \\
    --enable_stall_truncation

  然后用 convert_v5_to_v6lite.py 降维到 9d。
"""

import numpy as np
import zarr
import os
import re
import argparse
import time
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 卡住检测（修正版）
# ─────────────────────────────────────────────────────────────────────────────

def _dynamic_speed_threshold(dist_to_goal, base_speed):
    """接近目标时允许更低的速度阈值"""
    if dist_to_goal < 2.0:
        return base_speed * 0.6   # 0.03 if base=0.05
    elif dist_to_goal < 5.0:
        return base_speed * 0.8   # 0.04 if base=0.05
    else:
        return base_speed


def _is_window_progressing(dist, dpos_mag, window_start, window_end,
                           min_progress, base_speed):
    """判断一个窗口 [window_start, window_end) 是否在推进"""
    dist_progress = dist[window_start] - dist[window_end - 1]   # 正值 = 在靠近目标
    avg_speed = dpos_mag[window_start:window_end].mean()
    speed_thresh = _dynamic_speed_threshold(dist[window_end - 1], base_speed)
    return (dist_progress > min_progress) or (avg_speed > speed_thresh)


def detect_stall_point(state, action,
                       stall_window=15,
                       min_progress=0.10,
                       min_speed=0.05,
                       min_episode_len=20,
                       success_threshold=0.5,
                       partial_success_threshold=1.5) -> Tuple[int, Optional[str], Dict]:
    """
    修正版卡住检测。

    算法：
      1. min_dist < success_threshold → 已到达，保留全部
      2. 检查末尾 stall_window 步是否在推进
         - 是 → 保留全部（中途停顿后恢复的轨迹自动保留）
         - 否 → 从末尾往前扫描，找到最后一个 "在推进" 的窗口
                截断到该窗口结束位置

    Parameters:
        state:   (T, 15) — state[:,12] = dist_to_goal
        action:  (T, 11) — action[:,0:3] = dpos_world
        stall_window:   连续多少步算卡住窗口 (default 15 ≈ 4.5秒 @3.3Hz)
        min_progress:   窗口内 dist 最少减少量 (meters)
        min_speed:      窗口内平均 |dpos| 最低阈值 (meters/step，基准值)
        min_episode_len: 截断后最短保留长度
        success_threshold:         min_dist 低于此值 → 已到达 (meters)
        partial_success_threshold: final_dist 低于此值 → 部分成功 (meters)

    Returns:
        truncate_at: 保留 [0, truncate_at)。-1 = 丢弃整个 episode
        reason:      截断原因字符串，None = 不需要截断
        metadata:    详细统计字典
    """
    T = len(state)
    dist = state[:, 12]
    dpos_mag = np.linalg.norm(action[:, 0:3], axis=1)

    # ── 基础元数据 ──
    metadata = {
        'original_length': T,
        'min_dist': float(dist.min()),
        'final_dist': float(dist[-1]),
        'max_speed': float(dpos_mag.max()),
        'mean_speed': float(dpos_mag.mean()),
        'dist_at_start': float(dist[0]),
        'truncated': False,
        'truncated_length': 0,
        'truncated_frames': 0,
        'success': False,
        'partial_success': False,
        'truncation_reason': None,
        'stall_detected': False,
        'stall_length': 0,
        'mid_episode_stall': False,
    }

    min_dist = dist.min()

    # ── 1. 已到达目标 → 保留全部（但仍记录中途停顿）──
    if min_dist < success_threshold:
        metadata['success'] = True
        metadata['partial_success'] = True
        if T >= stall_window * 2:
            _detect_mid_stalls(dist, dpos_mag, stall_window, min_progress,
                               min_speed, metadata)
        return T, None, metadata

    # ── 2. 太短，无法做窗口检测 ──
    if T < stall_window + min_episode_len:
        if min_dist < partial_success_threshold:
            metadata['partial_success'] = True
        return T, None, metadata

    # ── 3. 检查末尾是否卡住 ──
    end_progressing = _is_window_progressing(
        dist, dpos_mag, T - stall_window, T, min_progress, min_speed)

    if end_progressing:
        # 末尾在推进 → 保留全部
        # 即使中途有停顿，只要最终恢复了就保留
        if min_dist < partial_success_threshold:
            metadata['partial_success'] = True

        # 可选：检测中途停顿（仅记录，不截断）
        _detect_mid_stalls(dist, dpos_mag, stall_window, min_progress, min_speed, metadata)

        return T, None, metadata

    # ── 4. 末尾卡住，从末尾往前扫描找到卡住起始点 ──
    stall_start = stall_window  # 最坏情况：整段（跳过初始窗口）都卡住

    for t in range(T - 1, stall_window, -1):
        ws = t - stall_window
        if _is_window_progressing(dist, dpos_mag, ws, t, min_progress, min_speed):
            # 这个窗口 [ws, t) 在推进。t 是窗口的最后一步，还属于推进阶段。
            # 卡住从 t 之后开始。但保守起见，截断点取 t（保留到 t-1）
            # 因为窗口 [ws, t) 推进 ≠ 步 t 本身推进（t 可能已经开始减速）
            stall_start = t
            break

    stall_length = T - stall_start
    metadata['stall_detected'] = True
    metadata['stall_length'] = stall_length

    # ── 5. 截断 ──
    truncate_at = stall_start

    if truncate_at < min_episode_len:
        metadata['truncated'] = False
        return -1, (f"stall too early: only {truncate_at} active steps, "
                    f"stall_length={stall_length}"), metadata

    final_dist = float(dist[truncate_at - 1])
    truncated_frames = T - truncate_at
    truncated_pct = truncated_frames / T * 100

    is_success = final_dist < success_threshold
    is_partial = final_dist < partial_success_threshold
    status = "success" if is_success else ("partial" if is_partial else "incomplete")

    reason = (f"tail stall ({status}): truncated {truncated_frames} frames "
              f"({truncated_pct:.0f}%), stall={stall_length} steps, "
              f"final_dist={final_dist:.2f}m")

    metadata.update({
        'truncated': True,
        'truncated_length': truncate_at,
        'truncated_frames': truncated_frames,
        'final_dist_after_trunc': final_dist,
        'truncation_reason': reason,
        'success': is_success,
        'partial_success': is_partial,
    })

    return truncate_at, reason, metadata


def _detect_mid_stalls(dist, dpos_mag, stall_window, min_progress, min_speed,
                       metadata):
    """
    检测中途停顿（仅记录到 metadata，不做截断）。
    用于了解数据中 EGO 中途整荡的频率和时长。
    """
    T = len(dist)
    if T < stall_window * 2:
        return

    stall_regions = []  # [(start, end), ...]
    in_stall = False
    stall_begin = 0

    for t in range(stall_window, T):
        ws = t - stall_window
        progressing = _is_window_progressing(
            dist, dpos_mag, ws, t, min_progress, min_speed)

        if not progressing and not in_stall:
            in_stall = True
            stall_begin = ws
        elif progressing and in_stall:
            in_stall = False
            stall_regions.append((stall_begin, t))

    # 不记录末尾卡住（那个由主函数处理）
    if stall_regions:
        # 排除最后一个 region 如果它延续到末尾
        if in_stall:
            pass  # 末尾的不记
        metadata['mid_episode_stall'] = True
        metadata['mid_stall_count'] = len(stall_regions)
        metadata['mid_stall_total_steps'] = sum(e - s for s, e in stall_regions)
        metadata['mid_stall_max_steps'] = max(e - s for s, e in stall_regions)


# ─────────────────────────────────────────────────────────────────────────────
# 扫描和检查
# ─────────────────────────────────────────────────────────────────────────────

def scan_zarr_files(input_dir, prefix="avoidance_dataset"):
    """扫描目录中的 zarr 文件"""
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)\.zarr$')
    files = []
    for name in os.listdir(input_dir):
        m = pattern.match(name)
        if m:
            num = int(m.group(1))
            full_path = os.path.join(input_dir, name)
            files.append((num, full_path, name))
    files.sort(key=lambda x: x[0])
    return files


def inspect_zarr(zarr_path):
    """检查 zarr 文件格式和维度"""
    root = zarr.open(str(zarr_path), mode='r')
    if 'data' not in root or 'meta' not in root:
        raise ValueError(f"不是 ReplayBuffer 格式: {zarr_path}")

    data = root['data']
    state_shape = data['state'].shape
    action_arr = data['action']
    pc_shape = data['point_cloud'].shape
    episode_ends = np.array(root['meta']['episode_ends'])

    action_ndim = action_arr.ndim
    if action_ndim == 2:
        horizon = 1
        action_dim = action_arr.shape[1]
    elif action_ndim == 3:
        horizon = action_arr.shape[1]
        action_dim = action_arr.shape[2]
    else:
        raise ValueError(f"未知 action 维度: {action_ndim}")

    return {
        'total_steps': state_shape[0],
        'state_dim': state_shape[1],
        'horizon': horizon,
        'action_dim': action_dim,
        'pc_points': pc_shape[1],
        'pc_channels': pc_shape[2],
        'num_episodes': len(episode_ends),
        'episode_ends': episode_ends,
        'action_ndim': action_ndim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主合并逻辑
# ─────────────────────────────────────────────────────────────────────────────

def merge(input_dir, output_path, prefix="avoidance_dataset",
          enable_stall_truncation=True,
          stall_window=15, min_progress=0.10, min_speed=0.05,
          min_episode_len=20,
          success_threshold=0.5, partial_success_threshold=1.5):
    """
    合并 zarr 文件，带修正的卡住检测。

    自然处理两种场景：
      - 中途停顿后恢复 → 保留全部（末尾在推进）
      - 末尾永久卡住 → 截断尾部卡住段
    """
    zarr_files = scan_zarr_files(input_dir, prefix)
    if not zarr_files:
        print(f"在 {input_dir} 中未找到 {prefix}_*.zarr 文件")
        return

    print("═" * 75)
    print("  merge_zarr_v8 — 合并工具（修正版卡住截断）" +
          (" [截断已启用]" if enable_stall_truncation else " [截断未启用]"))
    print("═" * 75)
    print(f"  输入目录: {input_dir}")
    print(f"  输出文件: {output_path}")
    print(f"  找到文件: {len(zarr_files)} 个")
    if enable_stall_truncation:
        print(f"  截断参数:")
        print(f"    stall_window    = {stall_window} steps (≈{stall_window/3.3:.1f}s @3.3Hz)")
        print(f"    min_progress    = {min_progress}m")
        print(f"    min_speed       = {min_speed}m/step (base, dynamic near goal)")
        print(f"    min_episode_len = {min_episode_len}")
        print(f"    success_thresh  = {success_threshold}m")
        print(f"    partial_thresh  = {partial_success_threshold}m")
    for num, path, name in zarr_files:
        print(f"    [{num:3d}] {name}")
    print("═" * 75)

    # ── Pass 1：扫描 + 截断决策 ──
    print("\n第1步: 扫描 + 截断决策...")

    file_plans = []     # (num, path, name, info, episode_plans) or None
    stats = {
        'total_original_frames': 0,
        'total_kept_frames': 0,
        'total_original_episodes': 0,
        'total_kept_episodes': 0,
        'n_truncated': 0,
        'n_dropped': 0,
        'n_success': 0,
        'n_partial': 0,
        'n_incomplete': 0,
        'n_mid_stall': 0,
    }
    truncation_log = []
    all_metadata = []

    action_ndim = None
    horizon = None

    for num, path, name in zarr_files:
        try:
            info = inspect_zarr(path)
        except Exception as e:
            print(f"  {name}: ✗ {e}")
            file_plans.append(None)
            continue

        # 格式一致性
        if action_ndim is None:
            action_ndim = info['action_ndim']
            horizon = info['horizon']
        elif action_ndim != info['action_ndim'] or horizon != info['horizon']:
            print(f"  {name}: 格式不匹配 (action_ndim={info['action_ndim']}, "
                  f"horizon={info['horizon']}), 跳过")
            file_plans.append(None)
            continue

        # 读取数据
        src = zarr.open(str(path), mode='r')
        state_all = np.array(src['data']['state'][:])
        action_all = np.array(src['data']['action'][:])
        ep_ends = info['episode_ends']
        ep_starts = np.concatenate([[0], ep_ends[:-1]]).astype(int)

        episode_plans = []

        for ei in range(len(ep_ends)):
            s = int(ep_starts[ei])
            e = int(ep_ends[ei])
            ep_len = e - s
            stats['total_original_frames'] += ep_len
            stats['total_original_episodes'] += 1

            if not enable_stall_truncation:
                # 不截断
                ep_dist = state_all[s:e, 12]
                md = {
                    'original_length': ep_len,
                    'min_dist': float(ep_dist.min()),
                    'final_dist': float(ep_dist[-1]),
                    'truncated': False,
                    'success': ep_dist.min() < success_threshold,
                    'partial_success': ep_dist.min() < partial_success_threshold,
                }
                episode_plans.append((s, e, e, None, md))
                stats['total_kept_frames'] += ep_len
                stats['total_kept_episodes'] += 1
                all_metadata.append(md)
                continue

            ep_state = state_all[s:e]
            # action: 如果是 3D (chunk-wise)，取第一个 step 作为当前帧 action
            ep_action = action_all[s:e] if action_ndim == 2 else action_all[s:e, 0, :]

            trunc_at, reason, md = detect_stall_point(
                ep_state, ep_action,
                stall_window=stall_window,
                min_progress=min_progress,
                min_speed=min_speed,
                min_episode_len=min_episode_len,
                success_threshold=success_threshold,
                partial_success_threshold=partial_success_threshold,
            )

            # 统计
            if md['success']:
                stats['n_success'] += 1
            elif md['partial_success']:
                stats['n_partial'] += 1
            else:
                stats['n_incomplete'] += 1
            if md.get('mid_episode_stall'):
                stats['n_mid_stall'] += 1

            if trunc_at == -1:
                episode_plans.append((s, e, -1, reason, md))
                stats['n_dropped'] += 1
                truncation_log.append(f"  DROP {name} ep{ei}: {reason}")
            elif trunc_at < ep_len:
                episode_plans.append((s, e, s + trunc_at, reason, md))
                stats['total_kept_frames'] += trunc_at
                stats['total_kept_episodes'] += 1
                stats['n_truncated'] += 1
                truncation_log.append(
                    f"  TRUNC {name} ep{ei}: {ep_len}→{trunc_at} frames, {reason}")
            else:
                episode_plans.append((s, e, e, reason, md))
                stats['total_kept_frames'] += ep_len
                stats['total_kept_episodes'] += 1

            all_metadata.append(md)

        file_plans.append((num, path, name, info, episode_plans))

        # 文件摘要
        kept = sum(1 for _, _, et, _, _ in episode_plans if et != -1)
        trunc_cnt = sum(1 for _, _, et, _, m in episode_plans
                        if et != -1 and m.get('truncated'))
        mid_stall_cnt = sum(1 for _, _, _, _, m in episode_plans
                            if m.get('mid_episode_stall'))
        print(f"  {name}: {info['num_episodes']} eps → kept={kept}, "
              f"truncated={trunc_cnt}, mid_stall={mid_stall_cnt}")

    # 截断报告
    n_orig = stats['total_original_episodes']
    n_kept = stats['total_kept_episodes']
    if enable_stall_truncation and n_orig > 0:
        print(f"\n  ┌─ 截断统计 ────────────────────────────────────────┐")
        print(f"  │  原始 episodes:      {n_orig:>6}")
        print(f"  │  保留 episodes:      {n_kept:>6}")
        print(f"  │  截断 episodes:      {stats['n_truncated']:>6} "
              f"({stats['n_truncated']/n_orig*100:.1f}%)")
        print(f"  │  丢弃 episodes:      {stats['n_dropped']:>6} "
              f"({stats['n_dropped']/n_orig*100:.1f}%)")
        print(f"  │  中途停顿 episodes:  {stats['n_mid_stall']:>6} "
              f"({stats['n_mid_stall']/max(n_kept,1)*100:.1f}% of kept)")
        print(f"  │  原始帧数:           {stats['total_original_frames']:>6}")
        print(f"  │  保留帧数:           {stats['total_kept_frames']:>6} "
              f"({stats['total_kept_frames']/max(stats['total_original_frames'],1)*100:.1f}%)")
        print(f"  ├─ 到达状态 ────────────────────────────────────────┤")
        print(f"  │  成功 (dist<{success_threshold}m):        {stats['n_success']:>6} "
              f"({stats['n_success']/max(n_kept,1)*100:.1f}%)")
        print(f"  │  部分成功 (dist<{partial_success_threshold}m):  {stats['n_partial']:>6} "
              f"({stats['n_partial']/max(n_kept,1)*100:.1f}%)")
        print(f"  │  未完成:              {stats['n_incomplete']:>6} "
              f"({stats['n_incomplete']/max(n_kept,1)*100:.1f}%)")
        print(f"  └──────────────────────────────────────────────────┘")

        if truncation_log:
            print(f"\n  截断详情 (前30条):")
            for line in truncation_log[:30]:
                print(line)
            if len(truncation_log) > 30:
                print(f"  ... 还有 {len(truncation_log) - 30} 条")

    # 过滤有效文件
    valid_plans = [fp for fp in file_plans if fp is not None]
    if not valid_plans:
        print("没有有效的 zarr 文件")
        return

    ref_info = valid_plans[0][3]
    state_dim = ref_info['state_dim']
    action_dim = ref_info['action_dim']
    pc_points = ref_info['pc_points']
    pc_channels = ref_info['pc_channels']
    total_frames = stats['total_kept_frames']

    print(f"\n  合计: {n_kept} episodes, {total_frames} frames")
    print(f"  维度: state({state_dim}), action({action_dim}), "
          f"pc({pc_points}×{pc_channels}), action_ndim={action_ndim}")

    # ── Pass 2：合并写入 ──
    print(f"\n第2步: 合并写入 → {output_path}")

    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    data_g = root.create_group('data')
    meta_g = root.create_group('meta')

    s_arr = data_g.zeros('state',
                         shape=(total_frames, state_dim),
                         chunks=(min(500, max(total_frames, 1)), state_dim),
                         dtype=np.float32)

    if action_ndim == 2:
        a_arr = data_g.zeros('action',
                             shape=(total_frames, action_dim),
                             chunks=(min(1000, max(total_frames, 1)), action_dim),
                             dtype=np.float32)
    else:
        a_arr = data_g.zeros('action',
                             shape=(total_frames, horizon, action_dim),
                             chunks=(min(100, max(total_frames, 1)), horizon, action_dim),
                             dtype=np.float32)

    p_arr = data_g.zeros('point_cloud',
                         shape=(total_frames, pc_points, pc_channels),
                         chunks=(min(20, max(total_frames, 1)), pc_points, pc_channels),
                         dtype=np.float32)

    all_episode_ends = []
    global_offset = 0
    CHUNK = 200

    for num, path, name, info, episode_plans in tqdm(valid_plans, desc="合并"):
        src = zarr.open(str(path), mode='r')

        for ep_start, ep_end_orig, ep_end_trunc, reason, md in episode_plans:
            if ep_end_trunc == -1:
                continue  # 丢弃

            n = ep_end_trunc - ep_start

            for cs in range(0, n, CHUNK):
                ce = min(cs + CHUNK, n)
                src_s = ep_start + cs
                src_e = ep_start + ce
                dst_s = global_offset + cs
                dst_e = global_offset + ce

                s_arr[dst_s:dst_e] = np.array(src['data']['state'][src_s:src_e])
                a_arr[dst_s:dst_e] = np.array(src['data']['action'][src_s:src_e])
                p_arr[dst_s:dst_e] = np.array(src['data']['point_cloud'][src_s:src_e])

            global_offset += n
            all_episode_ends.append(global_offset)

    meta_g.array('episode_ends', np.array(all_episode_ends, dtype=np.int64))

    # 元数据
    root.attrs.update({
        'version': 'v8',
        'num_episodes': len(all_episode_ends),
        'total_frames': total_frames,
        'horizon': int(horizon),
        'state_dim': int(state_dim),
        'action_dim': int(action_dim),
        'action_ndim': int(action_ndim),
        'state_format': '[pos_world(3), vel_world(3), acc_world(3), '
                        'goal_body(3), dist(1), sin_yaw(1), cos_yaw(1)]',
        'action_format': '[dpos_world(3), dvel_world(3), acc_world(3), '
                         'dyaw(1), yaw_rate(1)]',
        'point_cloud_format': '[xyz_body(3), color(3)]',
        'coordinate_frame': 'action=world, pointcloud=body, goal_body=body',
        'stall_truncation': enable_stall_truncation,
        'stall_params': {
            'stall_window': stall_window,
            'min_progress': min_progress,
            'min_speed': min_speed,
            'min_episode_len': min_episode_len,
            'success_threshold': success_threshold,
            'partial_success_threshold': partial_success_threshold,
        } if enable_stall_truncation else None,
        'truncation_stats': {
            'original_episodes': stats['total_original_episodes'],
            'kept_episodes': stats['total_kept_episodes'],
            'truncated_episodes': stats['n_truncated'],
            'dropped_episodes': stats['n_dropped'],
            'mid_stall_episodes': stats['n_mid_stall'],
            'original_frames': stats['total_original_frames'],
            'kept_frames': stats['total_kept_frames'],
            'success_count': stats['n_success'],
            'partial_count': stats['n_partial'],
            'incomplete_count': stats['n_incomplete'],
        } if enable_stall_truncation else None,
        'merged_from': [fp[2] for fp in valid_plans],
        'created': time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # ── 验证 ──
    print(f"\n第3步: 验证...")
    check = zarr.open(str(output_path), mode='r')
    check_ends = np.array(check['meta']['episode_ends'])

    print(f"  state:       {check['data']['state'].shape}")
    print(f"  action:      {check['data']['action'].shape}")
    print(f"  point_cloud: {check['data']['point_cloud'].shape}")
    print(f"  episodes:    {len(check_ends)}")
    print(f"  最后 end:    {check_ends[-1]}  (应={total_frames})")

    if int(check_ends[-1]) != total_frames:
        print(f"  ⚠ episode_ends 不一致!")
    else:
        print(f"  ✓ 数据完整")

    # episode 长度统计
    prev = 0
    lengths = []
    for e_val in check_ends:
        lengths.append(int(e_val) - prev)
        prev = int(e_val)
    lengths = np.array(lengths)

    print(f"\n  episode 长度统计:")
    print(f"    最短: {lengths.min()}  最长: {lengths.max()}")
    print(f"    均值: {lengths.mean():.1f}  中位: {np.median(lengths):.1f}")
    print(f"    < 50步: {(lengths < 50).sum()}  > 500步: {(lengths > 500).sum()}")

    # dist 分布
    all_state = check['data']['state'][:]
    dist_col = all_state[:, 12]
    print(f"\n  dist_to_goal 分布:")
    print(f"    min={dist_col.min():.2f}m  max={dist_col.max():.2f}m  "
          f"mean={dist_col.mean():.2f}m")

    # 末帧 dist 统计
    ep_starts_arr = np.concatenate([[0], check_ends[:-1]]).astype(int)
    final_dists = np.array([dist_col[int(check_ends[i]) - 1]
                            for i in range(len(check_ends))])

    n_strict = (final_dists < success_threshold).sum()
    n_part = (final_dists < partial_success_threshold).sum()

    print(f"\n  末帧 dist 统计:")
    print(f"    dist < {success_threshold}m (成功):     "
          f"{n_strict}/{len(check_ends)} ({n_strict/len(check_ends)*100:.1f}%)")
    print(f"    dist < {partial_success_threshold}m (部分成功): "
          f"{n_part}/{len(check_ends)} ({n_part/len(check_ends)*100:.1f}%)")
    print(f"    mean final dist: {final_dists.mean():.2f}m  "
          f"median: {np.median(final_dists):.2f}m")

    print(f"\n{'═' * 75}")
    print(f"  ✓ 合并完成!")
    print(f"  输出: {output_path}")
    print(f"  {len(all_episode_ends)} episodes, {total_frames} frames")
    if enable_stall_truncation:
        print(f"  截断: {stats['n_truncated']} truncated, "
              f"{stats['n_mid_stall']} mid-stall (kept), "
              f"{stats['n_dropped']} dropped")
    print(f"{'═' * 75}")


def main():
    parser = argparse.ArgumentParser(
        description='merge_zarr_v8: 合并工具（修正版卡住截断）')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='包含 avoidance_dataset_*.zarr 的目录')
    parser.add_argument('--output', type=str, required=True,
                        help='输出合并 zarr 路径')
    parser.add_argument('--prefix', type=str, default='avoidance_dataset',
                        help='zarr 文件前缀 (default: avoidance_dataset)')
    parser.add_argument('--enable_stall_truncation', action='store_true',
                        default=True,
                        help='启用卡住检测与截断 (default: True)')
    parser.add_argument('--no_stall_truncation', action='store_true',
                        help='禁用卡住截断')
    parser.add_argument('--stall_window', type=int, default=15,
                        help='卡住检测窗口 (steps, default 15 ≈ 4.5s)')
    parser.add_argument('--min_progress', type=float, default=0.10,
                        help='窗口内最小 dist 减少量 (m, default 0.10)')
    parser.add_argument('--min_speed', type=float, default=0.05,
                        help='窗口内最小平均速度 (m/step, default 0.05)')
    parser.add_argument('--min_episode_len', type=int, default=20,
                        help='截断后最短 episode 长度 (default 20)')
    parser.add_argument('--success_threshold', type=float, default=0.5,
                        help='成功阈值 (m, default 0.5)')
    parser.add_argument('--partial_success_threshold', type=float, default=1.5,
                        help='部分成功阈值 (m, default 1.5)')

    args = parser.parse_args()

    enable = args.enable_stall_truncation and not args.no_stall_truncation

    merge(args.input_dir, args.output, args.prefix,
          enable_stall_truncation=enable,
          stall_window=args.stall_window,
          min_progress=args.min_progress,
          min_speed=args.min_speed,
          min_episode_len=args.min_episode_len,
          success_threshold=args.success_threshold,
          partial_success_threshold=args.partial_success_threshold)


if __name__ == '__main__':
    main()