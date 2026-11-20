#!/usr/bin/env python3
"""
scan_ego_obstacle_margin.py — Phase ② 启动前诊断

扫描 EGO 训练数据 (zarr) 中每帧 / 每 episode 的 obstacle clearance 分布,
作为 Phase ② collision aux loss 的 d_safe / λ_col 起步参数依据.

理论依据:
  - point_cloud[t, :, :3] 是 body frame (drone center 为原点) 下的障碍物点云
  - drone clearance at frame t = min_i(||p_i||) for p_i in point_cloud[t]
  - 这是 "drone 中心到最近障碍点" 的距离 (真实物理 clearance, 不是 EGO 规划内部的 margin)

决策接入:
  P5(per-episode min margin) 是关键指标:
    P5 ≥ 1.0m → EGO 训练数据本身余量充分, Phase ② d_safe=1.0m 不会与 BC 冲突
                Phase ② λ_col 可起 0.1 (温和)
    P5 ∈ [0.5, 1.0)m → 部分演示擦边, d_safe=1.0m 会与 BC 冲突
                       Phase ② λ_col 起步 0.3-0.5 强 push
    P5 < 0.5m → 大量擦边演示, BC + collision aux loss 矛盾严重
                建议: d_safe 降到 0.75m + λ_col 0.5, 或考虑 Phase ④ DPPO 直接接管

用法:
    python3 scan_ego_obstacle_margin.py
    python3 scan_ego_obstacle_margin.py --zarr /custom/path.zarr --out report_dir/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import zarr
except ImportError:
    print("[ERR] 需要 zarr 包: pip install zarr")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--zarr',
                   default=os.path.expanduser(
                       '~/AUV/avoidance_dataset/merged_body_12.zarr'),
                   help='zarr 数据集路径')
    p.add_argument('--out', default='./ego_margin_scan',
                   help='输出目录 (默认 ./ego_margin_scan/)')
    p.add_argument('--eps', type=float, default=0.01,
                   help='||p|| < eps 的点视为无效/padding 过滤 (默认 0.01m)')
    p.add_argument('--max-episodes', type=int, default=None,
                   help='调试用: 仅处理前 N 个 episode (默认全部)')
    p.add_argument('--chunk-size', type=int, default=512,
                   help='每次从 zarr 读取的 frame 数 (默认 512)')
    return p.parse_args()


def open_zarr(path):
    """打开 zarr 并验证关键 array 存在"""
    if not os.path.exists(path):
        print(f"[ERR] zarr 不存在: {path}")
        sys.exit(1)
    z = zarr.open(path, mode='r')

    required = ['data/state', 'data/action', 'data/point_cloud', 'meta/episode_ends']
    for k in required:
        if k not in z:
            print(f"[ERR] zarr 缺少 key: {k}")
            print(f"      实际 keys: {list(z.tree())}")
            sys.exit(1)
    return z


def compute_frame_margins(point_cloud_chunk, eps=0.01):
    """
    对一个 chunk 的 point_cloud (T, N, C) 计算每帧的 min margin.

    Args:
        point_cloud_chunk: (T, N, C), C >= 3, 前 3 维 xyz body frame
        eps: 过滤 ||p|| < eps 的无效点

    Returns:
        margins: (T,) 每帧的 min distance from origin (drone center)
                 若一帧全是无效点, 该帧返回 inf (代表 drone 周围无障碍)
    """
    xyz = point_cloud_chunk[..., :3].astype(np.float32)  # (T, N, 3)
    norms = np.linalg.norm(xyz, axis=-1)  # (T, N)

    # 过滤无效点 (||p|| < eps)
    valid = norms > eps  # (T, N)

    # 对每帧, 在有效点中取最小. 全无效时设为 inf
    norms_masked = np.where(valid, norms, np.inf)
    margins = norms_masked.min(axis=-1)  # (T,)

    return margins, valid.sum(axis=-1)  # 也返回每帧有效点数


def percentile_stats(arr, name=""):
    """安全计算 percentile, 过滤 inf/nan"""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return {'name': name, 'count': 0}
    return {
        'name': name,
        'count': int(len(finite)),
        'min':  float(finite.min()),
        'p1':   float(np.percentile(finite, 1)),
        'p5':   float(np.percentile(finite, 5)),
        'p25':  float(np.percentile(finite, 25)),
        'p50':  float(np.percentile(finite, 50)),
        'p75':  float(np.percentile(finite, 75)),
        'p95':  float(np.percentile(finite, 95)),
        'p99':  float(np.percentile(finite, 99)),
        'max':  float(finite.max()),
        'mean': float(finite.mean()),
        'std':  float(finite.std()),
    }


def ascii_histogram(data, bins=20, width=50, title=""):
    """简单的 ascii 直方图"""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return f"{title}\n  (无数据)"
    hist, edges = np.histogram(finite, bins=bins)
    max_h = hist.max() if hist.max() > 0 else 1
    lines = [f"{title}"]
    for i, h in enumerate(hist):
        bar_len = int(width * h / max_h)
        bar = '█' * bar_len
        lines.append(f"  [{edges[i]:5.2f}, {edges[i+1]:5.2f}) "
                     f"{bar} {h}")
    return '\n'.join(lines)


def make_decision_recommendation(p5_episode_min, p25_episode_min,
                                 frame_margin_stats):
    """根据 P5 给出 Phase ② d_safe / λ_col 起步建议"""
    p5 = p5_episode_min
    p50_frame = frame_margin_stats.get('p50', float('nan'))

    if p5 >= 1.0:
        verdict = "GREEN"
        d_safe_rec = 1.0
        lambda_col_rec = 0.1
        reasoning = (
            f"EGO 训练数据 P5(per-episode min margin) = {p5:.2f}m ≥ 1.0m, "
            f"演示余量充分. d_safe=1.0m 不会与 BC 矛盾. "
            f"λ_col 起 {lambda_col_rec} 即可温和引导."
        )
    elif p5 >= 0.5:
        verdict = "YELLOW"
        d_safe_rec = 1.0
        lambda_col_rec = 0.3
        reasoning = (
            f"P5 = {p5:.2f}m ∈ [0.5, 1.0). 部分 EGO 演示擦边, "
            f"d_safe=1.0m 会与 BC 部分冲突. "
            f"λ_col 起 {lambda_col_rec} 给足 push, 扫参 {{0.3, 0.5, 1.0}} 找 sweet spot. "
            f"训练时监控 L_diffusion 是否仍能收敛 (G2-1)."
        )
    elif p5 >= 0.25:
        verdict = "ORANGE"
        d_safe_rec = 0.75
        lambda_col_rec = 0.5
        reasoning = (
            f"P5 = {p5:.2f}m < 0.5. 大量 EGO 擦边演示, "
            f"d_safe=1.0m 与 BC 严重矛盾. "
            f"降 d_safe 到 {d_safe_rec}m (drone 边缘 → 障碍 ≥ 0.5m), "
            f"λ_col 起 {lambda_col_rec}. 同时启动 R-2 应急预案 (Phase ④ DPPO 提前并行)."
        )
    else:
        verdict = "RED"
        d_safe_rec = 0.5
        lambda_col_rec = 1.0
        reasoning = (
            f"P5 = {p5:.2f}m < 0.25. EGO 数据本身存在碰撞或近碰撞演示, "
            f"BC 无法学到避障. d_safe={d_safe_rec}m + λ_col={lambda_col_rec} 是 BC 的极限尝试, "
            f"但更建议: (1) Phase ④ DPPO 立刻并行启动, "
            f"(2) 评估是否真的不重采数据 (MC-4 红线讨论)."
        )

    return {
        'verdict': verdict,
        'd_safe_recommendation': d_safe_rec,
        'lambda_col_recommendation': lambda_col_rec,
        'reasoning': reasoning,
    }


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[+] 打开 zarr: {args.zarr}")
    z = open_zarr(args.zarr)

    pc = z['data/point_cloud']     # (N_frames, N_points, C)
    state = z['data/state']         # (N_frames, 9)
    ep_ends = np.array(z['meta/episode_ends'])  # (N_episodes,)

    n_frames, n_points = pc.shape[0], pc.shape[1]
    n_channels = pc.shape[2] if pc.ndim == 3 else None
    n_episodes = len(ep_ends)

    print(f"    point_cloud shape: {pc.shape}")
    print(f"    state shape:       {state.shape}")
    print(f"    n_episodes:        {n_episodes}")
    print(f"    n_frames total:    {n_frames}")

    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)
        ep_ends = ep_ends[:n_episodes]
        n_frames = int(ep_ends[-1])
        print(f"    [DEBUG] 仅处理前 {n_episodes} episodes ({n_frames} frames)")

    # ---- 流式扫描 per-frame margin ----
    print(f"\n[+] 流式计算 per-frame margins (chunk size {args.chunk_size})")
    all_margins = np.zeros(n_frames, dtype=np.float32)
    all_valid_counts = np.zeros(n_frames, dtype=np.int32)
    t0 = time.time()

    for i in range(0, n_frames, args.chunk_size):
        end_i = min(i + args.chunk_size, n_frames)
        chunk = pc[i:end_i]  # 这里 zarr 流式读
        margins_chunk, valid_chunk = compute_frame_margins(chunk, eps=args.eps)
        all_margins[i:end_i] = margins_chunk
        all_valid_counts[i:end_i] = valid_chunk

        if (i // args.chunk_size) % 20 == 0:
            elapsed = time.time() - t0
            progress = (end_i / n_frames) * 100
            rate = end_i / max(elapsed, 1e-3)
            print(f"    {end_i}/{n_frames} frames ({progress:5.1f}%), "
                  f"{rate:.0f} frames/s, elapsed {elapsed:.0f}s")

    print(f"    完成. 总耗时 {time.time()-t0:.0f}s")

    # ---- per-episode aggregation ----
    print(f"\n[+] 计算 per-episode min margin (n_episodes={n_episodes})")
    ep_min_margins = np.zeros(n_episodes, dtype=np.float32)
    ep_mean_margins = np.zeros(n_episodes, dtype=np.float32)
    ep_n_frames = np.zeros(n_episodes, dtype=np.int32)

    start_idx = 0
    for ep_i, end_idx in enumerate(ep_ends):
        end_idx = int(end_idx)
        ep_margins = all_margins[start_idx:end_idx]
        # 仅在 finite margin 上聚合
        ep_margins_finite = ep_margins[np.isfinite(ep_margins)]
        if len(ep_margins_finite) > 0:
            ep_min_margins[ep_i] = float(ep_margins_finite.min())
            ep_mean_margins[ep_i] = float(ep_margins_finite.mean())
        else:
            ep_min_margins[ep_i] = float('inf')
            ep_mean_margins[ep_i] = float('inf')
        ep_n_frames[ep_i] = end_idx - start_idx
        start_idx = end_idx

    # ---- 统计 ----
    frame_stats = percentile_stats(all_margins, "per-frame margins")
    ep_min_stats = percentile_stats(ep_min_margins, "per-episode MIN margins")
    ep_mean_stats = percentile_stats(ep_mean_margins, "per-episode MEAN margins")

    # 阈值穿越统计
    thresholds = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    frame_below = {}
    ep_below = {}
    finite_frame = all_margins[np.isfinite(all_margins)]
    finite_ep = ep_min_margins[np.isfinite(ep_min_margins)]
    for t in thresholds:
        frame_below[t] = {
            'count': int((finite_frame < t).sum()),
            'share': float((finite_frame < t).mean()) if len(finite_frame) > 0 else 0.0,
        }
        ep_below[t] = {
            'count': int((finite_ep < t).sum()),
            'share': float((finite_ep < t).mean()) if len(finite_ep) > 0 else 0.0,
        }

    # ---- 决策 ----
    decision = make_decision_recommendation(
        ep_min_stats.get('p5', float('nan')),
        ep_min_stats.get('p25', float('nan')),
        frame_stats,
    )

    # ---- 写报告 ----
    report_md = Path(args.out) / 'ego_margin_scan_report.md'
    report_json = Path(args.out) / 'ego_margin_scan_stats.json'
    csv_path = Path(args.out) / 'ego_margin_per_episode.csv'

    with open(report_md, 'w') as f:
        f.write("# EGO 训练数据 Obstacle Margin 扫描报告\n\n")
        f.write(f"**Generated**: `scan_ego_obstacle_margin.py`\n")
        f.write(f"**Source**: `{args.zarr}`\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 0. TL;DR\n\n")
        f.write(f"- **n_episodes**: {n_episodes}\n")
        f.write(f"- **n_frames**: {n_frames}\n")
        f.write(f"- **per-episode min margin P5**: "
                f"**{ep_min_stats.get('p5', float('nan')):.3f} m**\n")
        f.write(f"- **per-episode min margin median**: "
                f"{ep_min_stats.get('p50', float('nan')):.3f} m\n")
        f.write(f"- **per-frame margin median**: "
                f"{frame_stats.get('p50', float('nan')):.3f} m\n\n")

        f.write(f"### 🚦 Phase ② 起步参数推荐: **{decision['verdict']}**\n\n")
        f.write(f"- `d_safe` = **{decision['d_safe_recommendation']} m**\n")
        f.write(f"- `λ_col` 起步 = **{decision['lambda_col_recommendation']}**\n\n")
        f.write(f"**Reasoning**: {decision['reasoning']}\n\n")

        f.write("## 1. Per-frame margin 分布\n\n")
        f.write("(每帧的 min ||p|| in point_cloud — drone center 到最近障碍点距离)\n\n")
        f.write("| 统计量 | 值 (m) |\n|---|---|\n")
        for k in ['min', 'p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99', 'max', 'mean', 'std']:
            v = frame_stats.get(k, float('nan'))
            f.write(f"| {k} | {v:.3f} |\n")
        f.write(f"| count (finite) | {frame_stats.get('count', 0)} |\n\n")

        f.write("```\n")
        # 自适应采样: 大数据集降采样防止 hist 计算慢, 小数据集全量
        n_finite_frame = int(np.isfinite(all_margins).sum())
        sample_step = max(1, n_finite_frame // 50000)  # 上限 50000 点参与 hist
        f.write(ascii_histogram(all_margins[np.isfinite(all_margins)][::sample_step],
                                bins=20,
                                title=f"per-frame margin (每 {sample_step} 帧抽样)"))
        f.write("\n```\n\n")

        f.write("## 2. Per-episode MIN margin 分布\n\n")
        f.write("(每个 episode 飞行过程中的最紧时刻 — **这是 BC 学到的行为下界**)\n\n")
        f.write("| 统计量 | 值 (m) |\n|---|---|\n")
        for k in ['min', 'p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99', 'max', 'mean', 'std']:
            v = ep_min_stats.get(k, float('nan'))
            f.write(f"| {k} | {v:.3f} |\n")
        f.write(f"| count (finite) | {ep_min_stats.get('count', 0)} |\n\n")

        f.write("```\n")
        f.write(ascii_histogram(ep_min_margins[np.isfinite(ep_min_margins)],
                                bins=20, title="per-episode MIN margin"))
        f.write("\n```\n\n")

        f.write("## 3. Per-episode MEAN margin 分布\n\n")
        f.write("(每个 episode 的平均 margin — 反映整体安全水平)\n\n")
        f.write("| 统计量 | 值 (m) |\n|---|---|\n")
        for k in ['min', 'p5', 'p25', 'p50', 'p75', 'p95', 'max', 'mean']:
            v = ep_mean_stats.get(k, float('nan'))
            f.write(f"| {k} | {v:.3f} |\n")
        f.write("\n")

        f.write("## 4. 阈值穿越统计\n\n")
        f.write("### 4.1 Per-frame margin 低于阈值的比例\n\n")
        f.write("| 阈值 (m) | n_frames < 阈值 | 占比 |\n|---|---|---|\n")
        for t in thresholds:
            f.write(f"| {t} | {frame_below[t]['count']} | "
                    f"{frame_below[t]['share']*100:.2f}% |\n")
        f.write("\n")

        f.write("### 4.2 Per-episode min margin 低于阈值的 episode 数\n\n")
        f.write("(这些 episode **永远擦边或近碰撞**, 整段飞行中存在 margin < 阈值的瞬间)\n\n")
        f.write("| 阈值 (m) | n_episodes < 阈值 | 占比 |\n|---|---|---|\n")
        for t in thresholds:
            f.write(f"| {t} | {ep_below[t]['count']} | "
                    f"{ep_below[t]['share']*100:.2f}% |\n")
        f.write("\n")

        f.write("## 5. 与 Phase ② d_safe 候选的对应关系\n\n")
        f.write("| d_safe | 与 EGO 数据冲突 episode 占比 | 解读 |\n")
        f.write("|---|---|---|\n")
        for ds in [0.5, 0.75, 1.0, 1.25]:
            conflict_share = ep_below[ds]['share'] if ds in ep_below else float('nan')
            if conflict_share < 0.05:
                interp = "✓ 几乎无冲突, λ_col 可温和"
            elif conflict_share < 0.20:
                interp = "🟡 部分冲突, λ_col 需要中等强度"
            elif conflict_share < 0.50:
                interp = "🟠 明显冲突, λ_col 需要强 push"
            else:
                interp = "🔴 严重冲突, 考虑 d_safe 下调或 DPPO 提前"
            f.write(f"| {ds}m | {conflict_share*100:.1f}% | {interp} |\n")
        f.write("\n")

        f.write("## 6. 决策接入 Phase ②\n\n")
        f.write(f"**Verdict: {decision['verdict']}**\n\n")
        f.write(f"修改设计文档 §5.2 collision proxy 起步参数:\n\n")
        f.write("```python\n")
        f.write(f"d_safe       = {decision['d_safe_recommendation']}  # m\n")
        f.write(f"lambda_col   = {decision['lambda_col_recommendation']}\n")
        f.write(f"k_neighbors  = 8  # 保持设计文档原值\n")
        f.write(f"tau          = 5.0  # 保持设计文档原值\n")
        f.write("```\n\n")

        f.write("## 7. 原始 JSON 数据\n\n")
        summary = {
            'meta': {
                'zarr_path': args.zarr,
                'n_episodes': n_episodes,
                'n_frames': n_frames,
                'eps_filter': args.eps,
            },
            'frame_margin_stats':  frame_stats,
            'ep_min_margin_stats':  ep_min_stats,
            'ep_mean_margin_stats': ep_mean_stats,
            'frame_below_thresholds': frame_below,
            'ep_below_thresholds':    ep_below,
            'decision': decision,
        }
        f.write("```json\n")
        f.write(json.dumps(summary, indent=2, default=str))
        f.write("\n```\n")

    # ---- JSON ----
    with open(report_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- per-episode CSV ----
    with open(csv_path, 'w') as f:
        f.write("episode_idx,n_frames,min_margin_m,mean_margin_m\n")
        for i in range(n_episodes):
            f.write(f"{i},{ep_n_frames[i]},"
                    f"{ep_min_margins[i]:.4f},{ep_mean_margins[i]:.4f}\n")

    print(f"\n[+] 报告已写入:")
    print(f"    markdown: {report_md}")
    print(f"    json:     {report_json}")
    print(f"    csv:      {csv_path}")
    print(f"\n关键数字:")
    print(f"  per-episode min margin P5  = {ep_min_stats.get('p5', float('nan')):.3f} m")
    print(f"  per-episode min margin P50 = {ep_min_stats.get('p50', float('nan')):.3f} m")
    print(f"  per-frame margin P50       = {frame_stats.get('p50', float('nan')):.3f} m")
    print(f"\n  >>> Verdict: {decision['verdict']} <<<")
    print(f"  推荐 d_safe = {decision['d_safe_recommendation']} m")
    print(f"  推荐 λ_col  = {decision['lambda_col_recommendation']}")
    print(f"  Reasoning: {decision['reasoning']}")


if __name__ == '__main__':
    main()