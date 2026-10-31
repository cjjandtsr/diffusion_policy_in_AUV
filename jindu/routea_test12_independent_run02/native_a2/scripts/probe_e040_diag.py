#!/usr/bin/env python3
"""E040 阶段 0 离线收益结构诊断：rollout 数据收集（不训练）。

对 E038 step100 PPO ckpt 与 BC ckpt 各 rollout 64 eps（16 env × 4 waves，E038
训练侧配置），记录每 episode 的 per-step dist / d_soft / smooth / progress_terms
与结局（reached / collision / timeout + 碰撞步数 t_c），输出 JSON 供
analyze_e040_diag.py 计算 ΔV 与校准临界值（terminal 奖励参数化，多候选值扫描）。

设计：program/sheji/E040_RiskRewardCalibration_Design_v1_0.md §4
P2-5：terms/metrics 从 env 实例属性读（envs[i]._last_v14_terms/_last_v14_metrics），
不复用训练脚本模块级 _TERMS_COLLECTOR（并行 env 下交错无法对应 episode）。
采样配置与 E038 训练侧一致：eta=1.0 / sigma_clip 0.01-10.0 / ft_denoising_steps=5 /
max_ep_len=100 / reward v16 / 292worlds manifest / curriculum C2。

用法：
  python scripts/probe_e040_diag.py --ckpt e038_step100 --seed 42
  python scripts/probe_e040_diag.py --ckpt bc --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.eval_ddpm_bc_baseline import load_ddpm_policy
from scripts.eval_ddpm_paired import DEFAULT_BC_CKPT, load_ppo_policy
from scripts.train_ddpm_ppo_meanlogp_3 import (
    DEFAULT_ADMISSION_MANIFEST, _stack_obs_batch, build_envs)

E038_STEP100_CKPT = "outputs/ddpm_ppo_e038_nosafety/ddpm_ppo_step100.pt"

# E038 训练侧 rollout 采样配置（与 train_ddpm_ppo_meanlogp_3.py parse_args 默认一致）
ETA = 1.0
SIGMA_CLIP_MIN = 0.01
SIGMA_CLIP_MAX = 10.0
FT_DENOISING_STEPS = 5


def run_diag_wave(envs: list, policy: torch.nn.Module, device: str,
                  max_ep_len: int) -> list:
    """每 env 跑 1 episode（batched GPU 推理 + 并行 CPU step），记录 per-step 数据。

    与训练脚本 run_episodes_batched 同构，但不收集 transition/V_old——
    只记录 env 实例的 terms/metrics 快照（f.result() 返回后读，下一 wave 才覆盖，
    串行安全）。返回 per-env episode 记录列表（len = n_envs）。
    """
    n = len(envs)
    records = [[] for _ in range(n)]
    done = [False] * n
    obs_list = [e.reset()[0] for e in envs]

    with torch.no_grad():
        for _step_i in range(max_ep_len):
            active = [i for i in range(n) if not done[i]]
            if not active:
                break

            obs_b = _stack_obs_batch([obs_list[i] for i in active], device)
            action, _logp, _chain = policy.sample_ddpm_with_logprob(
                obs_b, eta=ETA,
                sigma_clip_min=SIGMA_CLIP_MIN,
                sigma_clip_max=SIGMA_CLIP_MAX,
                ft_denoising_steps=FT_DENOISING_STEPS)
            action = policy.normalizer['action'].unnormalize(action)  # F1：物理尺度

            with ThreadPoolExecutor(max_workers=len(active)) as ex:
                futs = {ex.submit(envs[i].step, action[k].cpu().numpy()): (k, i)
                        for k, i in enumerate(active)}
                for f in as_completed(futs):
                    k, i = futs[f]
                    next_obs, reward, terminated, truncated, _ = f.result()
                    terms = getattr(envs[i], '_last_v14_terms', None) or {}
                    metrics = getattr(envs[i], '_last_v14_metrics', None) or {}
                    records[i].append({
                        "dist": float(metrics.get("dist", -1.0)),
                        "d_soft_h": float(metrics.get("d_soft_h", -1.0)),
                        "smooth": float(terms.get("smooth", 0.0)),
                        # 注意：含 terminal 分量（reward_v16 reached→+r_reach / collision→+r_collision），
                        # analyze 时按当前常量扣除后参数化重算
                        "progress_terms": float(terms.get("progress", 0.0)),
                        "collision": bool(getattr(envs[i], '_collision_detected', False)),
                        "terminated": bool(terminated),
                        "reward": float(reward),
                    })
                    obs_list[i] = next_obs
                    if terminated or truncated:
                        done[i] = True

    episodes = []
    for i in range(n):
        rec = records[i]
        if not rec:
            episodes.append({"outcome": "timeout", "t_c": None, "n_steps": 0, "records": []})
            continue
        last = rec[-1]
        if last["terminated"] and not last["collision"]:
            outcome = "reached"
        elif last["collision"]:
            outcome = "collision"
        else:
            outcome = "timeout"
        t_c = (len(rec) - 1) if outcome == "collision" else None  # 0-indexed 碰撞步
        episodes.append({"outcome": outcome, "t_c": t_c, "n_steps": len(rec), "records": rec})
    return episodes


def build_envs_for_diag(n_env_parallel: int, max_ep_len: int) -> list:
    """按 E038 训练配置构建 env 池（292worlds manifest / v16 / C2 / max_ep_len）。"""
    ns = argparse.Namespace(
        admission_manifest=DEFAULT_ADMISSION_MANIFEST,
        reward_version="v16",
        n_env_parallel=n_env_parallel,
        max_ep_len=max_ep_len,
    )
    return build_envs(ns, _PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E040 阶段 0 诊断 rollout（离线收益结构数据收集）")
    p.add_argument("--ckpt", choices=["bc", "e038_step100"], required=True)
    p.add_argument("--n_env_parallel", type=int, default=16)
    p.add_argument("--n_waves", type=int, default=4, help="waves × n_envs = 总 episode 数（4×16=64）")
    p.add_argument("--max_ep_len", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="outputs/e040_diag")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    if args.ckpt == "bc":
        policy = load_ddpm_policy(DEFAULT_BC_CKPT, device)
        ckpt_tag = "bc"
    else:
        policy = load_ppo_policy(device, E038_STEP100_CKPT)
        ckpt_tag = "e038_step100"

    envs = build_envs_for_diag(args.n_env_parallel, args.max_ep_len)

    all_episodes: list = []
    t0 = time.time()
    for w in range(args.n_waves):
        eps = run_diag_wave(envs, policy, device, args.max_ep_len)
        all_episodes.extend(eps)
        n_reached = sum(1 for e in eps if e["outcome"] == "reached")
        n_coll = sum(1 for e in eps if e["outcome"] == "collision")
        print(f"[WAVE {w + 1}/{args.n_waves}] eps={len(eps)} "
              f"reached={n_reached} collision={n_coll} "
              f"elapsed={time.time() - t0:.0f}s")

    # terminal 常量当前值（E039 校准后；analyze 用其扣除并参数化重算）
    from diffusion_policy_3d.env.reward_v16 import REWARD_V16_DEFAULTS
    out = {
        "ckpt": ckpt_tag,
        "seed": args.seed,
        "config": {
            "n_env_parallel": args.n_env_parallel,
            "n_waves": args.n_waves,
            "max_ep_len": args.max_ep_len,
            "eta": ETA, "sigma_clip_min": SIGMA_CLIP_MIN, "sigma_clip_max": SIGMA_CLIP_MAX,
            "ft_denoising_steps": FT_DENOISING_STEPS,
            "reward_version": "v16",
            "gamma": 0.99,  # 训练 GAE gamma（DPPOTrainer L186），ΔV 分析用
            "r_reach_now": float(REWARD_V16_DEFAULTS["r_reach"]),
            "r_collision_now": float(REWARD_V16_DEFAULTS["r_collision"]),
            "admission_manifest": DEFAULT_ADMISSION_MANIFEST,
        },
        "episodes": all_episodes,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"diag_{ckpt_tag}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[SAVE] {out_path} ({len(all_episodes)} eps)")


if __name__ == "__main__":
    main()
