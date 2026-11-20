#!/usr/bin/env python3
"""
extract_anchors.py — Phase 1a Step 0: 从数据集提取 trajectory anchors
================================================================================
从训练数据中提取 K 个 anchor 轨迹（K-means 聚类），加一个直线锚点。
输出 anchor_bank.pt 供 Phase 1a 训练使用。

v8.3 §6.5 规格:
  - 提取每个 action chunk 的 (Δpx, Δpy, Δpz, Δyaw) × n_action_steps = 4×8 = 32d
  - K=8 K-means
  - 覆盖度验证: p90 < 0.3m → K=8, p90 < 0.5m → K=12, p90 < 0.8m → K=16
  - 加直线锚点 A_{K+1}
  - 保存 anchor_bank.pt

用法:
  python extract_anchors.py \
    --zarr_path /home/cjj/AUV/avoidance_dataset/merged_body_12.zarr \
    --output anchor_bank.pt \
    --n_action_steps 8 --K 8

输出:
  anchor_bank.pt: {
    'anchors':        (K+1, 32) float32 — 归一化前的 anchor 轨迹
    'anchors_full':   (K+1, n_action_steps, 11) float32 — 完整 11d anchor
    'K': int,
    'coverage': dict,
  }
"""

import argparse
import numpy as np
import torch
import zarr
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def extract_chunks(zarr_path, n_action_steps, horizon):
    """从数据集提取所有 action chunk 的 (Δpx, Δpy, Δpz, Δyaw) 特征"""
    root = zarr.open(zarr_path, mode='r')
    action = root['data/action'][:]  # (N, 11)
    episode_ends = root['meta/episode_ends'][:]
    ep_starts = np.concatenate([[0], episode_ends[:-1]]).astype(int)

    chunks_4d = []   # (Δpx, Δpy, Δpz, Δyaw) × n_action_steps = 32d
    chunks_11d = []  # 完整 11d × n_action_steps

    for i in range(len(episode_ends)):
        s = int(ep_starts[i])
        e = int(episode_ends[i])
        ep_len = e - s
        # 滑动窗口提取 chunks
        for t in range(ep_len - n_action_steps + 1):
            chunk = action[s + t : s + t + n_action_steps]  # (n_act, 11)
            # 4d 特征: Δpx[0], Δpy[1], Δpz[2], Δyaw[9]
            feat = chunk[:, [0, 1, 2, 9]].flatten()  # (32,)
            chunks_4d.append(feat)
            chunks_11d.append(chunk)

    chunks_4d = np.array(chunks_4d, dtype=np.float32)
    chunks_11d = np.array(chunks_11d, dtype=np.float32)
    print(f"  提取 {len(chunks_4d)} 个 action chunks")
    print(f"  4d feature shape: {chunks_4d.shape}")
    return chunks_4d, chunks_11d


def evaluate_coverage(chunks_4d, centers, K):
    """v8.3 §16.3 覆盖度验证"""
    dists = cdist(chunks_4d, centers).min(axis=1)  # 每个 chunk 到最近 anchor 的距离
    p50 = np.percentile(dists, 50)
    p90 = np.percentile(dists, 90)
    p99 = np.percentile(dists, 99)
    tail_ratio = p99 / (p50 + 1e-8)

    print(f"\n  覆盖度统计 (K={K}):")
    print(f"    p50 = {p50:.4f}")
    print(f"    p90 = {p90:.4f}")
    print(f"    p99 = {p99:.4f}")
    print(f"    tail_ratio (p99/p50) = {tail_ratio:.2f}")
    print(f"    max dist = {dists.max():.4f}")

    # 每个 anchor 的使用率
    assignments = cdist(chunks_4d, centers).argmin(axis=1)
    for k in range(K):
        cnt = (assignments == k).sum()
        print(f"    anchor {k}: {cnt} chunks ({cnt/len(chunks_4d)*100:.1f}%)")

    return {
        'p50': float(p50), 'p90': float(p90), 'p99': float(p99),
        'tail_ratio': float(tail_ratio), 'max_dist': float(dists.max()),
    }


def decide_K(coverage, current_K):
    """v8.3 §16.3 覆盖度决策"""
    p90 = coverage['p90']
    tail = coverage['tail_ratio']

    if p90 < 0.3 and tail < 3:
        print(f"\n  决策: ACCEPT K={current_K} (p90={p90:.3f} < 0.3)")
        return current_K
    elif p90 < 0.5:
        new_K = min(current_K + 4, 16)
        print(f"\n  决策: TRY K={new_K} (p90={p90:.3f} < 0.5)")
        return new_K
    elif p90 < 0.8:
        new_K = min(current_K + 8, 16)
        print(f"\n  决策: TRY K={new_K} (p90={p90:.3f} < 0.8)")
        return new_K
    else:
        print(f"\n  决策: DATA CHECK (p90={p90:.3f} > 0.8, 数据可能有问题)")
        return current_K


def make_straight_anchor(n_action_steps, mean_speed=0.15):
    """创建直线锚点: 前方匀速直飞, Δyaw=0"""
    anchor = np.zeros((n_action_steps, 11), dtype=np.float32)
    anchor[:, 0] = mean_speed  # Δpx > 0 (前进)
    return anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', required=True)
    parser.add_argument('--output', default='anchor_bank.pt')
    parser.add_argument('--n_action_steps', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--auto_K', action='store_true',
                        help='自动选择 K (8→12→16)')
    args = parser.parse_args()

    print("═" * 60)
    print("  Phase 1a Step 0: Anchor Extraction")
    print("═" * 60)

    # 提取 chunks
    chunks_4d, chunks_11d = extract_chunks(
        args.zarr_path, args.n_action_steps, args.horizon)

    # K-means
    K = args.K
    best_K = K
    best_km = None
    best_coverage = None

    K_candidates = [K]
    if args.auto_K:
        K_candidates = [8, 12, 16]

    for K_try in K_candidates:
        print(f"\n  K-means with K={K_try}...")
        km = KMeans(n_clusters=K_try, random_state=args.seed,
                    n_init=10, max_iter=300)
        km.fit(chunks_4d)
        coverage = evaluate_coverage(chunks_4d, km.cluster_centers_, K_try)

        if coverage['p90'] < 0.3 and coverage['tail_ratio'] < 3:
            best_K = K_try
            best_km = km
            best_coverage = coverage
            print(f"  → K={K_try} 通过覆盖度检查")
            break
        elif best_km is None or coverage['p90'] < best_coverage['p90']:
            best_K = K_try
            best_km = km
            best_coverage = coverage

    print(f"\n  最终 K = {best_K}")
    K = best_K
    km = best_km

    # 获取每个 anchor 的完整 11d 代表
    assignments = cdist(chunks_4d, km.cluster_centers_).argmin(axis=1)
    anchor_full = np.zeros((K, args.n_action_steps, 11), dtype=np.float32)
    for k in range(K):
        mask = (assignments == k)
        anchor_full[k] = chunks_11d[mask].mean(axis=0)

    # 加直线锚点
    straight = make_straight_anchor(args.n_action_steps,
                                     mean_speed=chunks_11d[:, :, 0].mean())
    anchor_full = np.concatenate([anchor_full, straight[np.newaxis]], axis=0)

    # 4d anchor bank (K+1, 32) — 用于距离计算
    anchors_4d = np.zeros((K + 1, 4 * args.n_action_steps), dtype=np.float32)
    anchors_4d[:K] = km.cluster_centers_
    anchors_4d[K] = straight[:, [0, 1, 2, 9]].flatten()

    # 保存
    save_dict = {
        'anchors_4d': torch.from_numpy(anchors_4d),         # (K+1, 32)
        'anchors_full': torch.from_numpy(anchor_full),       # (K+1, n_act, 11)
        'K': K,
        'K_plus_1': K + 1,
        'n_action_steps': args.n_action_steps,
        'coverage': best_coverage,
    }
    torch.save(save_dict, args.output)

    print(f"\n  保存: {args.output}")
    print(f"    anchors_4d: {anchors_4d.shape}")
    print(f"    anchors_full: {anchor_full.shape}")
    print(f"    K = {K} (+1 straight = {K+1})")
    print(f"    coverage p90 = {best_coverage['p90']:.4f}")
    print("═" * 60)


if __name__ == '__main__':
    main()