#!/usr/bin/env python3
"""
m1_verify_synthetic.py — M1 验证闸: 合成点云 vs 真实点云闭环 SR 对照
======================================================================
对照 Implementation Guide v1.0 任务 3:
  在不改 v9_1 ckpt 的前提下, 对比合成点云与真实渲染的闭环 SR。
  mean_gap < 15% → PASS, 合成可用于训练。

用法:
  python m1_verify_synthetic.py \
    --ckpt <v9_1_ckpt> \
    --worlds w4,w11,w12 \
    --obstacles-dir /path/to/layerA/ \
    --output-dir /root/autodl-tmp/AUV/phase_beta/stage0/

输出:
  m1_synthetic_gap.json — 含 per_env SR / gap / verdict / failure_features

放置: diffusion_policy_3d/model/common/m1_verify_synthetic.py
"""

import os, sys, json, argparse, time, subprocess, numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple

# ── 配置 ──
WORLD_MAP = {
    'w4':  'env_004_obs19',
    'w8':  'env_008_obs20',
    'w11': 'env_011_obs23',
    'w12': 'env_012_obs22',
    'w18': 'env_018_obs25',
}

SCRIPT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "catkin_ws0" / "scripts"

# ── 工具函数 ──

def load_obstacles_json(obstacles_dir: str, env_name: str) -> str:
    """找对应 .obstacles.json 文件"""
    fpath = Path(obstacles_dir) / f"{env_name}.obstacles.json"
    if not fpath.exists():
        raise FileNotFoundError(f"Obstacle file not found: {fpath}")
    return str(fpath)


def run_one_trial(world_name: str, ckpt_path: str, use_synthetic: bool,
                  obstacles_dir: str, trial_id: int = 0) -> Dict:
    """
    跑一次 trial (简化版 — 调现有 run_one_trial.sh 或 phase_alpha 闭环框架)
    实际部署: 用 ml_gazebo_fm.sh + launch 参数控制点云来源
    """
    # 占位: 真实部署需要实际 ROS/Gazebo 环境
    # 这里构建命令框架, 实际调试时填充
    result = {
        'world': world_name,
        'trial': trial_id,
        'use_synthetic': use_synthetic,
        'success': None,       # 需实际跑
        'collision': None,
        'timeout': None,
        'planning_count': None,
    }
    return result


def compare_pointclouds(real_pc: np.ndarray, synth_pc: np.ndarray) -> Dict:
    """
    对比一帧合成点云与真实点云的统计特征 (用于诊断 gap 来源)
    在同一无人机位姿 + 同一障碍场景下采集
    """
    stats = {}

    if real_pc.shape[0] > 0 and synth_pc.shape[0] > 0:
        # 点数
        stats['real_n'] = int(real_pc.shape[0])
        stats['synth_n'] = int(synth_pc.shape[0])

        # 深度范围
        real_depth = np.linalg.norm(real_pc[:, :3], axis=1) if real_pc.shape[1] >= 3 else real_pc
        synth_depth = np.linalg.norm(synth_pc[:, :3], axis=1) if synth_pc.shape[1] >= 3 else synth_pc

        stats['real_depth_mean'] = float(real_depth.mean())
        stats['synth_depth_mean'] = float(synth_depth.mean())
        stats['real_depth_std'] = float(real_depth.std())
        stats['synth_depth_std'] = float(synth_depth.std())

        # 地面点占比 (z < 0.2m after body-frame analysis)
        # 需要无人机 pose 参考 — 简化: 最低 10% 分位点的高度
        if real_pc.shape[1] >= 3:
            stats['real_z_p05'] = float(np.percentile(real_pc[:, 2], 5))
            stats['synth_z_p05'] = float(np.percentile(synth_pc[:, 2], 5))
            # 地面点: z close to body_origin_z (值取决于具体场景)
            stats['real_ground_ratio_est'] = float((np.abs(real_pc[:, 2]) < 0.5).mean())
            stats['synth_ground_ratio_est'] = float((np.abs(synth_pc[:, 2]) < 0.5).mean())

        # Chamfer distance (简化, 随机采样 1000 点)
        n_sample = min(1000, min(real_pc.shape[0], synth_pc.shape[0]))
        ridx = np.random.choice(real_pc.shape[0], n_sample, replace=False)
        sidx = np.random.choice(synth_pc.shape[0], n_sample, replace=False)
        r = real_pc[ridx, :3]
        s = synth_pc[sidx, :3]
        dist_r2s = np.min(np.linalg.norm(r[:, None] - s[None], axis=2), axis=1)
        dist_s2r = np.min(np.linalg.norm(s[:, None] - r[None], axis=2), axis=1)
        stats['chamfer_mean'] = float((dist_r2s.mean() + dist_s2r.mean()) / 2)

    return stats


def diagnose_gap(per_env_results: Dict, pc_stats: Dict) -> List[str]:
    """诊断 sim-to-sim gap 的可能来源"""
    features = []

    # 检查地面点占比
    for env_name, stats in pc_stats.items():
        rg = stats.get('real_ground_ratio_est', 0)
        sg = stats.get('synth_ground_ratio_est', 0)
        if abs(rg - sg) > 0.15:
            features.append(f"ground_ratio_mismatch_{env_name}: real={rg:.2f} synth={sg:.2f}")

    # 检查 Chamfer distance
    for env_name, stats in pc_stats.items():
        cd = stats.get('chamfer_mean', 0)
        if cd > 2.0:
            features.append(f"high_chamfer_{env_name}: {cd:.2f}m (large shape mismatch)")

    # 检查深度分布
    for env_name, stats in pc_stats.items():
        rd = stats.get('real_depth_mean', 0)
        sd = stats.get('synth_depth_mean', 0)
        if abs(rd - sd) / (abs(rd) + 1e-6) > 0.3:
            features.append(f"depth_mismatch_{env_name}: real={rd:.1f} synth={sd:.1f}")

    return features


class M1VerificationGate:
    """M1 验证闸主逻辑"""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pc_stats = {}

    def run(self):
        """执行完整 M1 验证: 先对比单帧点云, 再跑闭环"""
        print("═" * 65)
        print("  Phase β M1 验证闸 — 合成 vs 真实点云 SR 对照")
        print("═" * 65)
        print(f"  ckpt: {self.args.ckpt}")
        print(f"  worlds: {self.args.worlds}")
        print(f"  obstacles: {self.args.obstacles_dir}")
        print("═" * 65)

        # 收集待测环境
        env_names = []
        for w in self.args.worlds.split(','):
            w = w.strip()
            env_names.append(WORLD_MAP.get(w, w))

        results = {'per_env': {}, 'verdict': 'UNKNOWN', 'failure_features': []}

        # Step 1: 单帧点云对比 (如果真实点云数据可用)
        if self.args.real_pc_dir and Path(self.args.real_pc_dir).exists():
            print("\n── 单帧点云统计对比 ──")
            for env_name in env_names:
                obstacles = load_obstacles_json(self.args.obstacles_dir, env_name)
                # 需要对应的 real PC 文件 — 占位
                pc_stats = {}  # compare_pointclouds(real, synth)
                self.pc_stats[env_name] = pc_stats

        # Step 2: 闭环 SR 对照
        if self.args.run_closed_loop:
            print("\n── 闭环 SR 对照 ──")

            for env_name in env_names:
                world_file = str(Path(self.args.obstacles_dir) / f"{env_name}.world")
                obstacles = load_obstacles_json(self.args.obstacles_dir, env_name)

                # 真实渲染 (baseline)
                print(f"\n  {env_name}: 真实渲染...")
                sr_real = self._run_trials(env_name, world_file, obstacles, use_synthetic=False)

                # 合成点云
                print(f"  {env_name}: 合成点云...")
                sr_synth = self._run_trials(env_name, world_file, obstacles, use_synthetic=True)

                gap = abs(sr_real - sr_synth) if sr_real is not None and sr_synth is not None else None
                results['per_env'][env_name] = {
                    'sr_real': sr_real,
                    'sr_synth': sr_synth,
                    'gap_pp': round(gap * 100, 1) if gap is not None else None,
                }

                if sr_real is not None and sr_synth is not None:
                    print(f"    SR: real={sr_real:.0%}  synth={sr_synth:.0%}  gap={gap:.1%}")

        # Step 3: 判决
        gaps = [v['gap_pp'] for v in results['per_env'].values()
                if v['gap_pp'] is not None]
        if gaps:
            mean_gap = np.mean(gaps) / 100.0
            results['mean_sr_real'] = float(np.mean(
                [v['sr_real'] for v in results['per_env'].values() if v['sr_real'] is not None]))
            results['mean_sr_synth'] = float(np.mean(
                [v['sr_synth'] for v in results['per_env'].values() if v['sr_synth'] is not None]))
            results['mean_gap_pct'] = float(mean_gap * 100)

            if mean_gap < 0.15:
                results['verdict'] = 'PASS'
                print(f"\n✅ M1 PASS: mean_gap={mean_gap:.1%} < 15%")
            else:
                results['verdict'] = 'FAIL'
                results['failure_features'] = diagnose_gap(results['per_env'], self.pc_stats)
                print(f"\n❌ M1 FAIL: mean_gap={mean_gap:.1%} >= 15%")
                if results['failure_features']:
                    print(f"  诊断: {'; '.join(results['failure_features'])}")
        else:
            results['verdict'] = 'PENDING_DATA'
            print("\n⚠️  无闭环数据 — 需在实际 Gazebo 环境跑满 trial 后回填 SR")

        # ── 保存 ──
        results['timestamp'] = datetime.now().isoformat()
        results['ckpt'] = self.args.ckpt
        results['worlds_tested'] = env_names

        output_path = self.output_dir / 'm1_synthetic_gap.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {output_path}")

        return results['verdict']

    def _run_trials(self, env_name, world_file, obstacles_json, use_synthetic):
        """跑若干 trial 得 SR (占位 — 实际需 Gazebo 环境)"""
        # 这里设计框架, 实际由部署脚本填充
        print(f"    [占位] 需实际 Gazebo 环境跑 trial。部署命令:")
        if use_synthetic:
            synth_flag = f"--synth-obstacles {obstacles_json}"
        else:
            synth_flag = ""
        print(f"    bash ml_gazebo_fm.sh {self.args.ckpt} full 4 1.5 0 0 {synth_flag}")
        return None  # 实际返回 SR


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="M1 验证闸: 合成 vs 真实点云 SR 对照")
    parser.add_argument('--ckpt', required=True, help='v9_1 BC ckpt 路径')
    parser.add_argument('--worlds', default='w4,w11,w12',
                       help='测试环境 (逗号分隔, 如 w4,w11,w12)')
    parser.add_argument('--obstacles-dir', required=True,
                       help='.obstacles.json 所在目录')
    parser.add_argument('--output-dir', default='/tmp/m1_verify',
                       help='输出目录')
    parser.add_argument('--real-pc-dir', default=None,
                       help='真实点云样本目录 (可选, 用于单帧对比)')
    parser.add_argument('--run-closed-loop', action='store_true',
                       help='实际跑闭环 (需 Gazebo 环境)')
    parser.add_argument('--trials-per-env', type=int, default=5,
                       help='每环境 trial 数')
    parser.add_argument('--timeout', type=int, default=120,
                       help='每 trial 超时秒数')
    args = parser.parse_args()

    gate = M1VerificationGate(args)
    gate.run()


if __name__ == '__main__':
    main()
