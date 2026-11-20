#!/usr/bin/env python3
"""
phase2_compare_results.py — Phase ② 闭环对比结果汇总
================================================================================
读取 phase2_closed_loop_compare.sh 产生的多 condition 飞行数据,
对每个 condition 跑 analyze_honest_baseline.py, 然后横向对比。

用法:
    python3 phase2_compare_results.py --run-root ~/ml_planner_log/phase2_closed_loop/<时间戳>

输出:
    <run-root>/phase2_comparison.md   — 横向对比报告
    每个 condition 子目录里也会有 analyze 脚本的单独报告

判定 (v1.2 §5.6):
    G2-3: Phase ② success rate ≥ Phase ① + 15%
    G2-4: Phase ② collision rate ≤ Phase ① × 0.3
"""

import argparse
import json
import os
import subprocess
import sys
from glob import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', required=True,
                   help='phase2_closed_loop_compare.sh 的输出根目录')
    p.add_argument('--analyzer', default='analyze_honest_baseline.py',
                   help='analyze_honest_baseline.py 路径')
    p.add_argument('--baseline-label', default='phase1a_r2',
                   help='作为 baseline 的 condition 标签 (G2-3/G2-4 对比基准)')
    p.add_argument('--out', default=None,
                   help='对比报告输出路径 (默认 <run-root>/phase2_comparison.md)')
    return p.parse_args()


def run_analyzer(analyzer, cond_dir, json_path):
    """对单个 condition 目录跑 analyze_honest_baseline.py"""
    md_path = os.path.join(cond_dir, 'analysis.md')
    cmd = [
        sys.executable, analyzer,
        '--dir', cond_dir,
        '--out', md_path,
        '--json', json_path,
    ]
    print(f"  运行: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  [WARN] analyzer 返回非零: {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [WARN] analyzer 超时")
        return False
    except Exception as e:
        print(f"  [WARN] analyzer 异常: {e}")
        return False


def main():
    args = parse_args()
    run_root = os.path.expanduser(args.run_root)
    if not os.path.isdir(run_root):
        print(f"[ERR] run-root 不存在: {run_root}")
        sys.exit(1)

    # 发现 condition 子目录 (每个含 diffusion_*.csv)
    cond_dirs = []
    for d in sorted(os.listdir(run_root)):
        full = os.path.join(run_root, d)
        if os.path.isdir(full) and glob(os.path.join(full, 'diffusion_*.csv')):
            cond_dirs.append((d, full))

    if not cond_dirs:
        print(f"[ERR] {run_root} 下没找到任何含 diffusion_*.csv 的 condition 子目录")
        sys.exit(1)

    print(f"[+] 发现 {len(cond_dirs)} 个 condition: {[c[0] for c in cond_dirs]}")

    # 逐 condition 跑 analyzer, 收集 JSON
    results = {}
    for label, cond_dir in cond_dirs:
        print(f"\n[+] 分析 condition: {label}")
        json_path = os.path.join(cond_dir, 'agg.json')
        if run_analyzer(args.analyzer, cond_dir, json_path):
            if os.path.exists(json_path):
                with open(json_path) as f:
                    results[label] = json.load(f)
            else:
                print(f"  [WARN] {label}: JSON 未生成")
        else:
            print(f"  [WARN] {label}: 分析失败, 跳过")

    if not results:
        print("[ERR] 没有任何 condition 分析成功")
        sys.exit(1)

    # 横向对比报告
    out_path = args.out or os.path.join(run_root, 'phase2_comparison.md')
    baseline = results.get(args.baseline_label)

    with open(out_path, 'w') as f:
        f.write("# Phase ② 闭环对比报告\n\n")
        f.write(f"**Run root**: `{run_root}`\n")
        f.write(f"**Conditions**: {len(results)}\n")
        f.write(f"**Baseline**: `{args.baseline_label}`\n\n")

        # ---- 主对比表 ----
        f.write("## 1. 关键指标对比\n\n")
        metrics = [
            ('success_rate', 'success rate', '%'),
            ('collision_rate', 'collision rate', '%'),
            ('mean_detour', 'mean detour', ''),
            ('mean_dxy_final', 'mean dxy_final (m)', ''),
            ('mean_end_raw_dpos_mean', '末段|raw_dpos| (m)', ''),
            ('mean_inference_ms_median', 'inference (ms)', ''),
            ('flights_with_data', '有效飞行数', ''),
        ]
        labels = list(results.keys())
        f.write("| 指标 | " + " | ".join(labels) + " |\n")
        f.write("|" + "|".join(['---'] * (len(labels) + 1)) + "|\n")
        for key, name, unit in metrics:
            row = [name]
            for lab in labels:
                v = results[lab].get(key, float('nan'))
                if unit == '%':
                    row.append(f"{v*100:.1f}%" if isinstance(v, (int, float)) else str(v))
                elif isinstance(v, float):
                    row.append(f"{v:.3f}")
                else:
                    row.append(str(v))
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")

        # ---- Gate 判定 ----
        f.write("## 2. Gate 判定 (v1.2 §5.6)\n\n")
        if baseline is None:
            f.write(f"⚠ 未找到 baseline condition `{args.baseline_label}`, 无法做 G2-3/G2-4 判定。\n\n")
        else:
            base_sr = baseline.get('success_rate', 0.0)
            base_cr = baseline.get('collision_rate', 1.0)
            f.write(f"baseline (`{args.baseline_label}`): "
                    f"success={base_sr:.1%}, collision={base_cr:.1%}\n\n")
            f.write("- **G2-3**: Phase ② success rate ≥ Phase ① + 15%\n")
            f.write("- **G2-4**: Phase ② collision rate ≤ Phase ① × 0.3\n\n")
            f.write("| Condition | success | G2-3 | collision | G2-4 | 综合 |\n")
            f.write("|---|---|---|---|---|---|\n")
            for lab in labels:
                if lab == args.baseline_label:
                    continue
                sr = results[lab].get('success_rate', 0.0)
                cr = results[lab].get('collision_rate', 1.0)
                g2_3 = sr >= base_sr + 0.15
                g2_4 = cr <= base_cr * 0.3
                verdict = '✓✓ 通过' if (g2_3 and g2_4) else (
                          '部分' if (g2_3 or g2_4) else '✗ 未通过')
                f.write(f"| {lab} | {sr:.1%} | {'✓' if g2_3 else '✗'} | "
                        f"{cr:.1%} | {'✓' if g2_4 else '✗'} | {verdict} |\n")
            f.write("\n")

        # ---- 结论与建议 ----
        f.write("## 3. 结论与下一步\n\n")
        if baseline is not None:
            base_sr = baseline.get('success_rate', 0.0)
            base_cr = baseline.get('collision_rate', 1.0)
            phase2_labels = [l for l in labels if l != args.baseline_label]
            best = None
            for lab in phase2_labels:
                sr = results[lab].get('success_rate', 0.0)
                cr = results[lab].get('collision_rate', 1.0)
                score = sr - cr  # 简单综合分
                if best is None or score > best[1]:
                    best = (lab, score, sr, cr)
            if best:
                lab, _, sr, cr = best
                f.write(f"- 表现最好的 Phase ② condition: **{lab}** "
                        f"(success={sr:.1%}, collision={cr:.1%})\n")
                if sr >= base_sr + 0.15 and cr <= base_cr * 0.3:
                    f.write(f"- ✓ G2-3 + G2-4 都过 → 用该 condition 的 λ 跑全量 300 epoch\n")
                elif sr > base_sr or cr < base_cr:
                    f.write(f"- ⚠ 有改善但未达 gate → collision loss 在 d_safe=0.75m 下作用有限\n")
                    f.write(f"  - 选项 A: 用该 λ 跑全量 300 epoch (21ep 可能欠拟合)\n")
                    f.write(f"  - 选项 B: 提高 d_safe 到 1.5m 重扫\n")
                    f.write(f"  - 选项 C: 接受现状, 直接推进 Phase ②.5 (diffusion-time guidance)\n")
                else:
                    f.write(f"- 🔴 Phase ② 未改善甚至变差\n")
                    f.write(f"  - 最可能原因: sweep 只 21 epoch, 欠拟合\n")
                    f.write(f"  - 建议: 选 λ=0.5 跑全量 300 epoch 再测; 或直接推进 Phase ②.5\n")
        f.write("\n")
        f.write("**注意**: sweep checkpoint 只训练了 21 epoch, "
                "若闭环表现差需先排除欠拟合 (跑全量 300 epoch 再判断)。\n")

    print(f"\n[+] 对比报告已写入: {out_path}")
    # 控制台速览
    print("\n" + "=" * 60)
    print("速览:")
    for lab in results:
        r = results[lab]
        print(f"  {lab}: success={r.get('success_rate',0):.1%}, "
              f"collision={r.get('collision_rate',0):.1%}, "
              f"detour={r.get('mean_detour',float('nan')):.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()