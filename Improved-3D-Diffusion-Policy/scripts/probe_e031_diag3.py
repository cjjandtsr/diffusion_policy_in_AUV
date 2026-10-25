#!/usr/bin/env python3
"""probe_e031_diag3.py — probe3: E031 训练主循环复刻 + R-A/R-B/R-D/R-E 验证 + A/B 修复实验。

规格来源:
  交接  program/jiaojie/DPPO_E031_KL衰减_交接_20260803.md  §probe3
  报告  program/research/dppo_failure_research_20260803/DPPO_失败根因与主流对照_20260803.md §5

行为（与训练逐行同构）:
  复刻 run_training 主循环（train_ddpm_ppo_meanlogp.py L777-886）:
  每轮 rollout 1 wave (16 eps, run_episodes_batched) → batch accumulation 到
  target_transitions=2048 → update_reward_stats → meritocracy advantage_weight →
  update() → 打印 → clear。参数与 E031 本地线一致（--logp_mean_norm --lr_actor 1e-3
  --K_epochs 2 --mode smoke --n_env_parallel 16 --n_eps_per_batch 16 --batched_rollout）。

每轮测量（报告 §5 六项）:
  1. reward 分量分解（r_progress/r_safety/r_smooth + d_soft_h/d_min/dist）—— 测 R-A
  2. raw advantage stats + ep_weights 分布（w 与 w² 两档，×2 失真直接量化）—— 测 R-B
  3. LoRA 权重 L1 漂移（逐轮）—— 测 R-E（收敛固定点 vs 噪声抖动）
  4. critic EV_pre（GAE 所用 V_old）/ EV_post（update 后）+ v_loss vs Var(returns) —— 测 R-D
  5. ep_sr 按 world id（本轮采样世界数 + 各世界成功率）—— 测 R-C
  6. kl/cf/nu/r/logp/ep_sr（训练既有）

变体（A/B 修复实验，验证后再动训练脚本）:
  baseline       当前代码路径（真实 DPPOTrainer.update()）—— 复现基线
  delta_rsafety  A/B-1: r_safety 改设计文档 delta 形式 tanh((d_soft_t-d_soft_{t-1})/0.5)
  single_merit   A/B-2: meritocracy 仅应用一次（去掉 mini-batch 内二次应用 L366-369）
  critic_budget  A/B-3: value 训练预算 5×actor（value-only 额外 epoch，对齐官方 50000×5）

用法:
  python scripts/probe_e031_diag3.py [variant ...] [--rounds 8]
  variant 缺省 = 全部 4 个顺序执行。
输出: stdout（行缓冲）+ logs/probe3_<variant>.log + 逐轮 jsonl（P1 教训: 探针日志保留不清理）
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from scripts.train_ddpm_ppo_meanlogp import (
    DPPOTrainer, ValueHead, PPORolloutBuffer, compute_gae,
    DEFAULT_ADMISSION_MANIFEST,
    build_envs, run_episodes_batched, load_ddpm_policy,
    global_to_obs_enc)
from diffusion_policy_3d.policy.lora_adapters import insert_lora
from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv

# ─────────────────────────────────────────────────────────────────────
# 配置（E031 本地线一致）
# ─────────────────────────────────────────────────────────────────────
BC_CKPT = "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/checkpoints/latest.ckpt"
REWARD_NORM = "outputs/ddpm_ppo/reward_norm_stats_v14.json"
LR_ACTOR, LR_VALUE = 1e-3, 1e-3
K_EPOCHS, MBS, TARGET_TRANS = 2, 256, 2048
N_ENV, N_EPS = 16, 16
MAX_EP_LEN = 100
K_INF, FT_STEPS = 10, 5
ETA, SCMIN, SCMAX = 1.0, 0.01, 10.0
LORA_R, LORA_ALPHA = 16, 16
LOG_DIR = os.path.join(PROJ, "logs")

VARIANTS = {
    "baseline":       dict(delta_safety=False, single_merit=False, value_epochs=None),
    "delta_rsafety":  dict(delta_safety=True,  single_merit=False, value_epochs=None),
    "single_merit":   dict(delta_safety=False, single_merit=True,  value_epochs=None),
    "critic_budget":  dict(delta_safety=False, single_merit=False, value_epochs=10),  # 5× actor K_epochs=2
    # H5 (2026-08-03): 组合修复 = delta reward + 去二次 meritocracy（决定性实验）
    "combined":       dict(delta_safety=True,  single_merit=True,  value_epochs=None),
}


def _snapshot(params):
    return torch.cat([p.detach().cpu().reshape(-1) for p in params])


def _l1_drift(prev, params):
    """per-param 平均 |Δ|（L1 / numel）。"""
    return float((_snapshot(params) - prev).abs().mean())


# ─────────────────────────────────────────────────────────────────────
# 环境钩子: reward 分量记录（全部变体）+ delta r_safety（A/B-1）
# ─────────────────────────────────────────────────────────────────────
_ORIG_COMPUTE = KinematicUAVEnv._compute_reward_v16
_ORIG_RESET = KinematicUAVEnv.reset
_DELTA_MODE = False

W_SAFETY, S_SAFETY = 3.0, 0.5  # 与 reward_v16 默认一致


def _install_env_hooks(delta_mode: bool) -> None:
    """类级 patch。reset 时重置 per-episode 记录与 delta 参考值。"""
    global _DELTA_MODE
    _DELTA_MODE = delta_mode

    def hooked_reset(self, *a, **kw):
        out = _ORIG_RESET(self, *a, **kw)
        self._probe_terms = []
        self._probe_metrics = []
        self._probe_d_soft_prev = None
        return out

    def hooked_compute(self, action, pos_t, yaw_t, vel_t):
        reward, terminated = _ORIG_COMPUTE(self, action, pos_t, yaw_t, vel_t)
        terms = dict(self._last_v14_terms or {})
        metrics = dict(self._last_v14_metrics or {})
        if _DELTA_MODE:
            # 设计文档 delta 形式: r_safety = w_safety·tanh((d_soft_t − d_soft_{t−1})/s_safety)
            # 首步无参考 → 保持绝对式（边界约定，写入日志可辨）
            d_soft = metrics.get("d_soft_h", -1.0)
            prev = getattr(self, "_probe_d_soft_prev", None)
            self._probe_d_soft_prev = d_soft
            if prev is not None and not getattr(self, "_collision_detected", False):
                r_safety_new = W_SAFETY * math.tanh((d_soft - prev) / S_SAFETY)
                reward = float(reward) - float(terms.get("safety", 0.0)) + r_safety_new
                terms["safety"] = r_safety_new
                terms["_delta_applied"] = 1.0
        self._probe_terms.append(terms)
        self._probe_metrics.append(metrics)
        return reward, terminated

    KinematicUAVEnv.reset = hooked_reset
    KinematicUAVEnv._compute_reward_v16 = hooked_compute


def _aggregate_components(envs):
    """本轮各 env 最后一集的分量均值（run_episodes_batched 每 env 每 wave 恰 1 集）。"""
    terms_all, metrics_all = {}, {}
    for e in envs:
        for t in getattr(e, "_probe_terms", []):
            for k, v in t.items():
                terms_all.setdefault(k, []).append(float(v))
        for m in getattr(e, "_probe_metrics", []):
            for k, v in m.items():
                metrics_all.setdefault(k, []).append(float(v))
    out = {}
    for k, v in terms_all.items():
        out[f"r_{k}"] = float(np.mean(v)) if v else 0.0
    for k in ("d_soft_h", "d_min", "dist"):
        v = metrics_all.get(k, [])
        out[k] = float(np.mean(v)) if v else -1.0
    return out


# ─────────────────────────────────────────────────────────────────────
# 更新前测量（只读, 与 update() 内 GAE 逐行同构: L304-332）
# ─────────────────────────────────────────────────────────────────────
def _measure_pre(value_head, buffer, trainer, advantage_weight, n_obs_steps):
    device = next(value_head.parameters()).device
    obs_all = torch.stack([b.to(device) for b in buffer.obs_list])
    rewards = torch.tensor(buffer.reward_list, device=device)
    dones = torch.tensor(buffer.done_list, device=device, dtype=torch.float32)
    V_old_all = torch.stack([b.to(device) for b in buffer.V_old_list])
    next_obs_all = torch.stack([b.to(device) for b in buffer.next_obs_list])
    with torch.no_grad():
        V_last = value_head(global_to_obs_enc(
            next_obs_all[-1:], n_obs_steps=n_obs_steps, obs_dim=192)).squeeze()
        if V_last.dim() == 0:
            V_last = V_last.unsqueeze(0)
        values = torch.cat([V_old_all, V_last], dim=0)
        advantages, returns = compute_gae(
            rewards, values, dones, trainer.gamma, trainer.gae_lambda,
            trainer.reward_mean, trainer.reward_std)

    # EV: GAE 所用 V_old 对 returns 的解释率
    ev_pre = 1.0 - ((V_old_all - returns) ** 2).mean() / (returns.var() + 1e-8)

    # advantage 管道逐级统计（R-B 核心: ×2 失真）
    z = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    aw = torch.tensor(advantage_weight, device=device, dtype=torch.float32)
    w1 = z * aw
    w2 = z * aw * aw

    return {
        "returns": returns,
        "ev_pre": float(ev_pre),
        "adv_raw_std": float(advantages.std()),
        "adv_z_std": float(z.std()),
        "adv_w1_std": float(w1.std()),
        "adv_w2_std": float(w2.std()),
        "aw_min": float(aw.min()),
        "aw_max": float(aw.max()),
        "aw_frac_lt05": float((aw < 0.5).float().mean()),
        "var_returns": float(returns.var()),
    }


def _measure_ev_post(value_head, buffer, returns, n_obs_steps):
    device = next(value_head.parameters()).device
    obs_all = torch.stack([b.to(device) for b in buffer.obs_list])
    with torch.no_grad():
        obs_encs = global_to_obs_enc(obs_all, n_obs_steps=n_obs_steps, obs_dim=192)
        V_pred = value_head(obs_encs).squeeze(-1)
    return float(1.0 - ((V_pred - returns) ** 2).mean() / (returns.var() + 1e-8))


# ─────────────────────────────────────────────────────────────────────
# ProbeTrainer — update() 变体（基类 L287-445 逐行复制，仅两处差异）
# ─────────────────────────────────────────────────────────────────────
class ProbeTrainer(DPPOTrainer):
    """update() 变体:
      - single_merit: 去掉 mini-batch 内第二次 meritocracy 应用（基类 L366-369）
      - value_epochs: 标准循环之后追加 value-only epoch（对齐官方 critic 预算）
    顺带捕获 actor/value 最后一次 grad norm（clip_grad_norm_ 返回值）。
    """

    def __init__(self, *a, single_merit=False, value_epochs=None, **kw):
        super().__init__(*a, **kw)
        self._probe_single_merit = single_merit
        self._probe_value_epochs = value_epochs
        self.last_grad_norms = (None, None)

    def update(self, buffer, model, scheduler, value_head,
               actor_optimizer, value_optimizer,
               horizon, action_dim, num_inference_steps,
               n_obs_steps, cfg_lambda, eta,
               sigma_clip_min, sigma_clip_max,
               ft_denoising_steps=None,
               K_epochs=4, mini_batch_size=64,
               advantage_weight=None):
        """Standard PPO on γ-weighted joint logp (scalar ratio, constant ε=0.2).

        γ_denoise weights early denoising steps less in the logp sum → the PPO
        ratio naturally focuses credit on later (more important) steps. BC reg
        uses per-step logp internally to anchor each step independently.
        """
        device = next(model.parameters()).device
        n_transitions = len(buffer)

        # Stack buffer
        obs_all = torch.stack([b.to(device) for b in buffer.obs_list])
        obs_uncond_all = torch.stack([b.to(device) for b in buffer.obs_uncond_list])
        rewards = torch.tensor(buffer.reward_list, device=device)
        dones = torch.tensor(buffer.done_list, device=device, dtype=torch.float32)
        V_old_all = torch.stack([b.to(device) for b in buffer.V_old_list])
        next_obs_all = torch.stack([b.to(device) for b in buffer.next_obs_list])

        # GAE
        with torch.no_grad():
            V_last = value_head(global_to_obs_enc(next_obs_all[-1:],
                                     n_obs_steps=n_obs_steps, obs_dim=192)).squeeze()
            if V_last.dim() == 0:
                V_last = V_last.unsqueeze(0)
            values = torch.cat([V_old_all, V_last], dim=0)
            advantages, returns = compute_gae(
                rewards, values, dones, self.gamma, self.gae_lambda,
                self.reward_mean, self.reward_std)

        obs_encs_192 = global_to_obs_enc(obs_all, n_obs_steps=n_obs_steps, obs_dim=192)

        # Joint logp: scalar per transition (γ-weighted sum of per-step logps)
        logp_old = torch.stack([b.to(device) for b in buffer.logp_old_list])  # (N,) scalars

        # FIX P1-2: Normalize advantages over FULL batch (REINFORCE++: per-minibatch norm is biased).
        # Move normalization BEFORE the mini-batch loop.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if advantage_weight is not None:
            advantages = advantages * torch.tensor(advantage_weight, device=device, dtype=torch.float32)

        actor_loss_total = 0.0
        value_loss_total = 0.0
        approx_kl_total = 0.0
        clipfrac_total = 0.0
        n_updates = 0

        for _ in range(K_epochs):
            kl_triggered = False  # FIX P1-4: epoch-level KL early-stop
            indices = torch.randperm(n_transitions, device=device)
            for start in range(0, n_transitions, mini_batch_size):
                idx = indices[start:start + mini_batch_size]
                if len(idx) < 2:
                    continue

                chains_batch = []
                for k in range(num_inference_steps + 1):
                    chains_batch.append(torch.stack(
                        [buffer.a_chain_list[i][k].to(device) for i in idx]))

                # Joint (γ-weighted) logp → (M,) scalar
                logp_new = self._recompute_joint_logp_batched(
                    chains_batch, obs_encs_192[idx],
                    model, scheduler,
                    obs_all[idx], obs_uncond_all[idx],
                    cfg_lambda, num_inference_steps,
                    eta, sigma_clip_min, sigma_clip_max,
                    ft_denoising_steps=ft_denoising_steps)  # (M,)

                logp_old_mb = logp_old[idx]  # (M,)

                # FIX P1-2: Advantage already normalized over full batch above
                adv_mb = advantages[idx]
                # probe3 (A/B-2): 仅应用一次 —— 去掉基类 L366-369 的二次 w 乘
                if advantage_weight is not None and not self._probe_single_merit:
                    w_mb = torch.tensor([advantage_weight[i] for i in idx.tolist()],
                                        device=device, dtype=torch.float32)
                    adv_mb = adv_mb * w_mb

                logp_diff = torch.clamp(logp_new - logp_old_mb, -20.0, 20.0)
                if self.logp_mean_norm:
                    # P0: 176 维求和 → 维度均值（官方 DDPO/DPPO 结构），KL/ratio 尺度正常化
                    logp_diff = logp_diff / (horizon * action_dim)
                ratio = torch.exp(logp_diff)  # (M,)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()
                    clipfrac = ((ratio < 1.0 - self.clip_epsilon) |
                                (ratio > 1.0 + self.clip_epsilon)).float().mean()

                # FIX P1-3/4: KL threshold 1.5× (standard PPO), epoch-level early-stop
                if approx_kl > 1.5 * self.target_kl:
                    kl_triggered = True
                    break

                V_pred = value_head(obs_encs_192[idx]).squeeze(-1)
                returns_batch = returns[idx]
                if returns_batch.dim() > 1:
                    returns_batch = returns_batch.squeeze(-1)
                value_loss = nn.functional.mse_loss(V_pred, returns_batch)

                # BC regularization: joint logp on BC chain
                bc_loss = torch.tensor(0.0, device=device)
                if self.w_bc > 0 and self.bc_model is not None:
                    x_T_batch = chains_batch[0].to(device)
                    with torch.no_grad():
                        bc_chains_batch = self._generate_deterministic_chain(
                            x_T_batch, self.bc_model.model, scheduler,
                            obs_all[idx], obs_uncond_all[idx],
                            cfg_lambda, num_inference_steps)
                    bc_logp = self._recompute_joint_logp_batched(
                        bc_chains_batch, obs_encs_192[idx],
                        model, scheduler,
                        obs_all[idx], obs_uncond_all[idx],
                        cfg_lambda, num_inference_steps,
                        eta, sigma_clip_min, sigma_clip_max,
                        ft_denoising_steps=ft_denoising_steps)  # (M,)
                    bc_loss = -bc_logp.mean()

                total_loss = actor_loss + 0.5 * value_loss + self.w_bc * bc_loss

                actor_optimizer.zero_grad()
                value_optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    agn = nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    vgn = nn.utils.clip_grad_norm_(value_head.parameters(), self.max_grad_norm)
                    self.last_grad_norms = (float(agn), float(vgn))
                actor_optimizer.step()
                value_optimizer.step()

                actor_loss_total += actor_loss.item()
                value_loss_total += value_loss.item()
                approx_kl_total += approx_kl.item()
                clipfrac_total += clipfrac.item()
                n_updates += 1

            # FIX P1-4: epoch-level KL early-stop — if any mini-batch triggered KL, skip remaining epochs
            if kl_triggered:
                break

        # probe3 (A/B-3): value-only 额外 epoch（对齐官方 critic 训练预算）
        extra = (self._probe_value_epochs or K_epochs) - K_epochs
        for _ in range(max(extra, 0)):
            indices = torch.randperm(n_transitions, device=device)
            for start in range(0, n_transitions, mini_batch_size):
                idx = indices[start:start + mini_batch_size]
                if len(idx) < 2:
                    continue
                V_pred = value_head(obs_encs_192[idx]).squeeze(-1)
                returns_batch = returns[idx]
                if returns_batch.dim() > 1:
                    returns_batch = returns_batch.squeeze(-1)
                vloss = nn.functional.mse_loss(V_pred, returns_batch)
                value_optimizer.zero_grad()
                vloss.backward()
                if self.max_grad_norm > 0:
                    self.last_grad_norms = (self.last_grad_norms[0],
                                            float(nn.utils.clip_grad_norm_(
                                                value_head.parameters(), self.max_grad_norm)))
                value_optimizer.step()

        if n_updates == 0:
            n_updates = 1

        return {
            'actor_loss': actor_loss_total / n_updates,
            'value_loss': value_loss_total / n_updates,
            'approx_kl': approx_kl_total / n_updates,
            'clipfrac': clipfrac_total / n_updates,
            'entropy': 0.0,
            'n_updates': n_updates,
        }


# ─────────────────────────────────────────────────────────────────────
# 单变体运行
# ─────────────────────────────────────────────────────────────────────
def run_variant(variant: str, rounds: int) -> None:
    cfg = VARIANTS[variant]
    logf = os.path.join(LOG_DIR, f"probe3_{variant}.log")
    jsonl = os.path.join(LOG_DIR, f"probe3_{variant}.jsonl")
    os.makedirs(LOG_DIR, exist_ok=True)
    out = open(jsonl, "w")

    def say(msg):
        print(msg, flush=True)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    say(f"===== probe3 variant={variant} {cfg} device={device} rounds={rounds} =====")

    policy = load_ddpm_policy(BC_CKPT, device=str(device))
    policy.eval()
    lora_info = insert_lora(policy.model, r=LORA_R, alpha=LORA_ALPHA, verbose=False)
    policy.model.requires_grad_(False)
    for p in lora_info["lora_params"]:
        p.requires_grad_(True)
    say(f"[{variant}] LoRA trainable: {sum(p.numel() for p in lora_info['lora_params']):,} params")

    value_head = ValueHead(input_dim=192).to(device).train()
    actor_opt = torch.optim.Adam(lora_info["lora_params"], lr=LR_ACTOR)
    value_opt = torch.optim.Adam(value_head.parameters(), lr=LR_VALUE)

    trainer = DPPOTrainer(clip_epsilon=0.2, target_kl=0.015, entropy_alpha=0.005,
                          max_grad_norm=0.5, gamma_denoise=0.8, logp_mean_norm=True)
    if cfg["single_merit"] or cfg["value_epochs"] is not None:
        trainer = ProbeTrainer(clip_epsilon=0.2, target_kl=0.015, entropy_alpha=0.005,
                               max_grad_norm=0.5, gamma_denoise=0.8, logp_mean_norm=True,
                               single_merit=cfg["single_merit"],
                               value_epochs=cfg["value_epochs"])
    if os.path.exists(REWARD_NORM):
        with open(REWARD_NORM) as f:
            rn = json.load(f)
        trainer.reward_mean = rn.get("total_mean", 0.0)
        trainer.reward_std = max(rn.get("total_std", 1.0), 1.0)
        say(f"[{variant}] reward_norm mean={trainer.reward_mean:.2f} std={trainer.reward_std:.2f}")

    args = argparse.Namespace(
        admission_manifest=DEFAULT_ADMISSION_MANIFEST,
        reward_version="v16",
        max_ep_len=MAX_EP_LEN, n_eps_per_batch=N_EPS, n_env_parallel=N_ENV,
        eta=ETA, sigma_clip_min=SCMIN, sigma_clip_max=SCMAX,
        ft_denoising_steps=FT_STEPS)
    _install_env_hooks(cfg["delta_safety"])
    envs = build_envs(args, PROJ)
    say(f"[{variant}] envs built: {len(envs)} (reward={getattr(envs[0], 'reward_version', '?')})")

    buffer = PPORolloutBuffer()
    ep_return_list, ep_success_list = [], []
    world_sr = {}                       # world -> [0/1]（R-C）
    n_updates_fired = 0
    lora_snapshot = _snapshot(lora_info["lora_params"])

    for rnd in range(1, rounds + 1):
        t_r = time.time()
        # ── rollout: 1 wave = N_ENV 集（与训练 update_step 语义一致）──
        trs_list, stats_list = run_episodes_batched(envs, policy, value_head, device, args)
        all_rewards, all_logps = [], []
        for i, (transitions, stats) in enumerate(zip(trs_list, stats_list)):
            for tr in transitions:
                buffer.add(*tr)
            all_rewards.append(stats[0])
            all_logps.append(stats[1])
            ep_return_list.append(stats[2])
            ep_success_list.append(stats[3])
            world_sr.setdefault(str(envs[i].world_id), []).append(int(stats[3]))
        n_eps, n_trans = len(ep_success_list), len(buffer)

        if n_trans < TARGET_TRANS:
            say(f"[R{rnd}/{rounds}] collect eps={n_eps} buffer={n_trans}/{TARGET_TRANS} "
                f"T={time.time() - t_r:.0f}s")
            continue

        # ── update 触发（L830-867 逐行同构）──
        trainer.update_reward_stats([float(r) for r in buffer.reward_list])
        ep_returns_arr = np.array(ep_return_list, dtype=np.float32)
        r_min, r_max = ep_returns_arr.min(), ep_returns_arr.max()
        if r_max > r_min + 0.01:
            ep_weights = 0.1 + 0.9 * (ep_returns_arr - r_min) / (r_max - r_min)
        else:
            ep_weights = np.ones_like(ep_returns_arr)
        advantage_weight = []
        ep_idx = 0
        for i in range(len(buffer.reward_list)):
            if i > 0 and buffer.done_list[i - 1]:
                ep_idx += 1
            advantage_weight.append(float(ep_weights[min(ep_idx, len(ep_weights) - 1)]))

        meas = _measure_pre(value_head, buffer, trainer, advantage_weight,
                            policy.n_obs_steps)

        upd_kw = dict(
            buffer=buffer, model=policy.model, scheduler=policy.noise_scheduler,
            value_head=value_head, actor_optimizer=actor_opt, value_optimizer=value_opt,
            horizon=policy.horizon, action_dim=policy.action_dim,
            num_inference_steps=K_INF, n_obs_steps=policy.n_obs_steps,
            cfg_lambda=policy.cfg_lambda, eta=ETA,
            sigma_clip_min=SCMIN, sigma_clip_max=SCMAX,
            ft_denoising_steps=FT_STEPS, K_epochs=K_EPOCHS,
            mini_batch_size=MBS, advantage_weight=advantage_weight)
        metrics = trainer.update(**upd_kw)

        l1 = _l1_drift(lora_snapshot, lora_info["lora_params"])
        lora_snapshot = _snapshot(lora_info["lora_params"])
        ev_post = _measure_ev_post(value_head, buffer, meas["returns"], policy.n_obs_steps)
        comp = _aggregate_components(envs)
        ep_sr = float(np.mean(ep_success_list)) if ep_success_list else 0.0
        mean_r = float(np.mean(all_rewards))
        mean_lp = float(np.mean(all_logps))
        n_updates_fired += 1

        gn = getattr(trainer, "last_grad_norms", (None, None))
        line = (f"[UPD {n_updates_fired}] rnd={rnd}/{rounds} kl={metrics['approx_kl']:.3e} "
                f"cf={metrics['clipfrac']:.3f} nu={metrics['n_updates']} "
                f"r={mean_r:.2f} logp={mean_lp:.2f} ep_sr={ep_sr:.3f} "
                f"| ev_pre={meas['ev_pre']:.3f} ev_post={ev_post:.3f} "
                f"v_loss={metrics['value_loss']:.2f} var_ret={meas['var_returns']:.2f} "
                f"| adv_raw_std={meas['adv_raw_std']:.2f} adv_z={meas['adv_z_std']:.2f} "
                f"adv_w1={meas['adv_w1_std']:.2f} adv_w2={meas['adv_w2_std']:.2f} "
                f"aw=[{meas['aw_min']:.2f},{meas['aw_max']:.2f},lo<0.5:{meas['aw_frac_lt05']:.2f}] "
                f"| l1={l1:.2e} gnorm={gn[0] if gn[0] else float('nan'):.2f}/"
                f"{gn[1] if gn[1] else float('nan'):.2f} "
                f"| r_prog={comp.get('r_progress', 0):+.2f} r_safe={comp.get('r_safety', 0):+.2f} "
                f"r_smoo={comp.get('r_smooth', 0):+.2f} "
                f"d_soft={comp.get('d_soft_h', -1):.2f} d_min={comp.get('d_min', -1):.2f} "
                f"dist={comp.get('dist', -1):.2f} "
                f"| eps={n_eps} worlds={len(world_sr)} T={time.time() - t_r:.0f}s")
        say(line)

        rec = {
            "variant": variant, "round": rnd, "n_updates_fired": n_updates_fired,
            "kl": metrics["approx_kl"], "cf": metrics["clipfrac"], "nu": metrics["n_updates"],
            "actor_loss": metrics["actor_loss"], "value_loss": metrics["value_loss"],
            "r": mean_r, "logp": mean_lp, "ep_sr": ep_sr,
            "ev_pre": meas["ev_pre"], "ev_post": ev_post,
            "var_returns": meas["var_returns"],
            "adv_raw_std": meas["adv_raw_std"], "adv_z_std": meas["adv_z_std"],
            "adv_w1_std": meas["adv_w1_std"], "adv_w2_std": meas["adv_w2_std"],
            "aw_min": meas["aw_min"], "aw_max": meas["aw_max"],
            "aw_frac_lt05": meas["aw_frac_lt05"],
            "l1_drift": l1,
            "grad_norm_actor": gn[0], "grad_norm_value": gn[1],
            **comp,
            "n_eps": n_eps, "n_worlds": len(world_sr),
            "T_s": round(time.time() - t_r),
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()

        buffer.clear()
        ep_return_list.clear()
        ep_success_list.clear()

    # ── world-level SR 摘要（R-C）──
    wl = sorted(world_sr.items(), key=lambda kv: -len(kv[1]))[:10]
    say(f"[{variant}] worlds sampled: {len(world_sr)} unique / {sum(len(v) for v in world_sr.values())} eps")
    for wid, srs in wl[:8]:
        say(f"  world {wid}: sr={np.mean(srs):.2f} n={len(srs)}")
    say(f"[{variant}] updates fired: {n_updates_fired}/{rounds} rounds | log={logf} jsonl={jsonl}")
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*",
                    default=list(VARIANTS.keys()))
    ap.add_argument("--rounds", type=int, default=8)
    args = ap.parse_args()
    for v in args.variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant: {v} (choose from {list(VARIANTS.keys())})")
    print(f"[probe3] variants={args.variants} rounds={args.rounds}", flush=True)
    for v in args.variants:
        run_variant(v, args.rounds)
    print("[probe3] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
