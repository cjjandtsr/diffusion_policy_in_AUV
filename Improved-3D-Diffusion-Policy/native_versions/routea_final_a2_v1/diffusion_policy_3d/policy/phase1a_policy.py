"""
phase1a_policy.py — Phase 1a: Anchor + Truncated Diffusion + CFG + Projection
================================================================================
继承 DiffusionPointcloudPolicy。

Zread R2 修正:
  P1-4 修复: truncated diffusion 的 t_start 从 truncation_ratio 计算
             （而非 scheduler.set_timesteps 的第一个步），确保 anchor 不被淹没

State 9d layout 说明 (对应 CFG dropout):
  [0:3] pos_world
  [3:6] goal_body     ← CFG dropout 目标
  [6]   dist          ← CFG dropout 目标
  [7]   sin_yaw
  [8]   cos_yaw
  所以 agent_pos[:, :, 3:7] = 0.0 零掉 goal_body(3) + dist(1)，正确。

放置: diffusion_policy_3d/policy/phase1a_policy.py
"""

import time
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import reduce
from termcolor import cprint

from diffusion_policy_3d.policy.diffusion_pointcloud_policy import DiffusionPointcloudPolicy
from diffusion_policy_3d.policy.anchor_head import AnchorHead
from diffusion_policy_3d.common.pytorch_util import dict_apply


class Phase1aPolicy(DiffusionPointcloudPolicy):

    def __init__(self,
                 # 基类参数
                 shape_meta, noise_scheduler, horizon, n_action_steps,
                 n_obs_steps, num_inference_steps=None, obs_as_global_cond=True,
                 diffusion_step_embed_dim=256, down_dims=(256, 512, 1024),
                 kernel_size=5, n_groups=8, condition_type="film",
                 use_down_condition=True, use_mid_condition=True,
                 use_up_condition=True, use_pc_color=False,
                 pointnet_type="pointnet", pointcloud_encoder_cfg=None,
                 point_downsample=False,
                 # Phase 1a 参数
                 anchor_bank_path=None,
                 anchor_loss_weight=0.5,
                 cfg_dropout_prob=0.15,
                 cfg_lambda=1.5,
                 truncated_steps=50,
                 truncation_ratio=1.0,
                 # Projection
                 use_projection=True,
                 z_min=1.0, z_max=5.0,
                 yaw_max_per_step=0.524,
                 yaw_max_accumulated=3.14159,
                 # Obstacle Projection (Week 4)
                 use_obstacle_projection=False,  # Phase ①: 默认关闭策略外修剪
                 obstacle_safe_margin=0.5,
                 obstacle_k_neighbors=20,
                 # Training phase
                 training_phase='joint',
                 # Phase ② Constraint-Aware BC: collision auxiliary loss
                 use_collision_loss=False,
                 collision_lambda=0.5,
                 collision_d_safe=0.75,
                 collision_beta=20.0,
                 collision_tau_softmin=5.0,
                 collision_k_nn=8,
                 collision_horizon_decay_tau=8.0,
                 **kwargs):
        super().__init__(
            shape_meta=shape_meta, noise_scheduler=noise_scheduler,
            horizon=horizon, n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps, num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims, kernel_size=kernel_size, n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            use_pc_color=use_pc_color, pointnet_type=pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            point_downsample=point_downsample, **kwargs)

        self.anchor_loss_weight = anchor_loss_weight
        self.cfg_dropout_prob = cfg_dropout_prob
        self.cfg_lambda = cfg_lambda
        self.truncated_steps = truncated_steps
        self.truncation_ratio = truncation_ratio
        self.training_phase = training_phase

        # Phase ② collision auxiliary loss
        self.use_collision_loss = use_collision_loss
        self.collision_lambda = collision_lambda
        self.collision_d_safe = collision_d_safe
        self.collision_beta = collision_beta
        self.collision_tau_softmin = collision_tau_softmin
        self.collision_k_nn = collision_k_nn
        self.collision_horizon_decay_tau = collision_horizon_decay_tau

        self.use_projection = use_projection
        self.z_min = z_min
        self.z_max = z_max
        self.yaw_max_per_step = yaw_max_per_step
        self.yaw_max_accumulated = yaw_max_accumulated
        self.use_obstacle_projection = use_obstacle_projection
        self.obstacle_safe_margin = obstacle_safe_margin
        self.obstacle_k_neighbors = obstacle_k_neighbors

        # Anchor bank 初始化为普通属性（后续会被 register_buffer 覆盖）
        self.anchor_bank_full = None
        self.n_anchors = 0
        if anchor_bank_path is not None:
            self._load_anchors(anchor_bank_path)

        # AnchorHead
        if self.n_anchors > 0:
            obs_feat_dim = self.obs_feature_dim * n_obs_steps
            self.anchor_head = AnchorHead(
                input_dim=obs_feat_dim, n_anchors=self.n_anchors)
            cprint(f'[Phase1a] AnchorHead: {obs_feat_dim}→{self.n_anchors}',
                   'cyan')
        else:
            self.anchor_head = None

        # 计算 truncated diffusion 的起始时间步
        T_full = noise_scheduler.config.num_train_timesteps
        self.t_trunc_start = min(int(T_full * truncation_ratio), T_full - 1)
        cprint(f'[Phase1a] truncated diffusion: t_start={self.t_trunc_start} '
               f'(ratio={truncation_ratio}, steps={truncated_steps})', 'cyan')
        cprint(f'[Phase1a] CFG dropout={cfg_dropout_prob}, λ={cfg_lambda}',
               'cyan')

        # 部署计时标志（默认关闭，部署时手动设置为 True）
        self._deploy_timing = False
        # 计时统计收集
        self._timing_stats = {'t_enc': [], 't_den': []}

    def _load_anchors(self, path):
        bank = torch.load(path, map_location='cpu')
        # 修复：删除已有的普通属性，避免与 register_buffer 冲突
        if hasattr(self, 'anchor_bank_full'):
            del self.anchor_bank_full
        self.register_buffer('anchor_bank_full',
                             bank['anchors_full'].float())
        self.n_anchors = bank['K_plus_1']
        cprint(f'[Phase1a] {self.n_anchors} anchors from {path}', 'green')

    # ──────────────────────────────────────────────────────────────
    # 辅助：确保 noise_scheduler 张量在模型设备上
    # ──────────────────────────────────────────────────────────────
    def _ensure_scheduler_device(self, device):
        """
        将 noise_scheduler 中的核心张量移动到指定设备。
        解决 diffusers DDPMScheduler 内部张量默认在 CPU 导致的设备不匹配问题。
        """
        scheduler = self.noise_scheduler
        for attr in ['alphas_cumprod', 'betas', 'alphas',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                     'log_one_minus_alphas_cumprod',
                     'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod',
                     'alpha_t', 'sigma_t']:
            if hasattr(scheduler, attr):
                t = getattr(scheduler, attr)
                if isinstance(t, torch.Tensor) and t.device != device:
                    setattr(scheduler, attr, t.to(device))

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _get_anchor_labels(self, nactions):
        """
        GT anchor labels: 在 normalized space 计算距离。
        nactions 已经归一化，anchor bank 也做同样归一化 → 一致。
        用 4d 特征 (Δpx, Δpy, Δpz, Δyaw) 计算距离。
        """
        if self.anchor_bank_full is None:
            return None

        B = nactions.shape[0]
        Da = self.action_dim
        To = self.n_obs_steps

        # action chunk 区域
        start = To - 1
        end = start + self.n_action_steps
        chunk = nactions[:, start:end, :].detach()  # (B, n_act, 11)

        # 归一化 anchor bank（与 nactions 同空间）
        anchors = self.anchor_bank_full.to(chunk.device)
        n_anchors_norm = self.normalizer['action'].normalize(
            anchors.reshape(-1, Da)).reshape(self.n_anchors, -1, Da)

        # 4d 特征距离
        idx_4d = [0, 1, 2, 9]
        chunk_4d = chunk[:, :, idx_4d].reshape(B, -1)
        anchor_4d = n_anchors_norm[:, :, idx_4d].reshape(self.n_anchors, -1)

        dists = torch.cdist(chunk_4d.unsqueeze(0),
                             anchor_4d.unsqueeze(0)).squeeze(0)
        return dists.argmin(dim=-1)

    def _get_anchor_trajectory(self, anchor_idx):
        """归一化的 anchor 轨迹，铺满 horizon"""
        B = anchor_idx.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        device = anchor_idx.device

        selected = self.anchor_bank_full.to(device)[anchor_idx]  # (B, n_act, 11)
        selected_norm = self.normalizer['action'].normalize(
            selected.reshape(-1, Da)).reshape(B, -1, Da)

        n_act = selected_norm.shape[1]
        traj = torch.zeros(B, T, Da, device=device, dtype=selected_norm.dtype)

        traj[:, :To-1, :] = selected_norm[:, :1, :].expand(-1, To-1, -1)
        end_pos = min(To - 1 + n_act, T)
        traj[:, To-1:end_pos, :] = selected_norm[:, :end_pos-(To-1), :]
        if end_pos < T:
            traj[:, end_pos:, :] = selected_norm[:, -1:, :].expand(
                -1, T - end_pos, -1)

        return traj

    # ──────────────────────────────────────────────────────────────
    # Anchored Truncated Sampling（Zread P1-4 修复版）
    # ──────────────────────────────────────────────────────────────

    def conditional_sample(self,
                           condition_data, condition_mask,
                           condition_data_pc=None, condition_mask_pc=None,
                           local_cond=None, global_cond=None,
                           global_cond_uncond=None,
                           anchor_idx=None,
                           generator=None, **kwargs):
        """
        Anchored Truncated Diffusion + CFG。

        Zread P1-4 修复:
          不用 scheduler.set_timesteps(truncated_steps) 取 timesteps[0] 作 t_start。
          那样会给出 t≈800（80% 噪声），anchor 被淹没。
          正确做法: t_start = T_full * truncation_ratio（默认 0.2 → t=200）
          然后在 [t_start, 0] 之间均匀分 truncated_steps 个步。
        """
        model = self.model
        scheduler = self.noise_scheduler
        B, T, Da = condition_data.shape
        device = condition_data.device
        dtype = condition_data.dtype

        # 确保 scheduler 张量在正确设备上
        self._ensure_scheduler_device(device)

        if anchor_idx is not None and self.anchor_bank_full is not None:
            # ── Anchored truncated diffusion ──
            anchor_traj = self._get_anchor_trajectory(anchor_idx)

            # 从 t_trunc_start 加噪声（低噪声，保留 anchor 信息）
            t_start = self.t_trunc_start
            noise = torch.randn(anchor_traj.shape, device=device, dtype=dtype)
            t_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)
            trajectory = scheduler.add_noise(anchor_traj, noise, t_tensor)

            # 从 t_start 到 0 均匀分 truncated_steps 个步
            timesteps = torch.linspace(
                t_start, 0, self.truncated_steps + 1,
                dtype=torch.long, device=device)[:-1]  # 不含 0
            timesteps = torch.clamp(timesteps, min=1)   # 避免 t=0

            # ★ 关键修复：通知 scheduler 推理步数，否则 scheduler.step 会因
            #   num_inference_steps=None 而抛出 ValueError
            scheduler.num_inference_steps = self.truncated_steps
        else:
            # Fallback: 标准采样
            trajectory = torch.randn(
                size=condition_data.shape, device=device, dtype=dtype)
            scheduler.set_timesteps(self.num_inference_steps)
            timesteps = scheduler.timesteps.to(device)

        # ── DEBUG: 打印 timesteps 状态 (已禁用，需要时取消注释) ──
        # print(f"[DEBUG] anchor mode timesteps 总数: {len(timesteps)}")
        # print(f"[DEBUG] timesteps 前10个: {timesteps[:10]}")
        # print(f"[DEBUG] timesteps 后5个: {timesteps[-5:]}")
        # print(f"[DEBUG] scheduler.num_inference_steps: {scheduler.num_inference_steps}")
        # print(f"[DEBUG] scheduler.timesteps 长度: {len(scheduler.timesteps) if scheduler.timesteps is not None else None}")
        # ────────────────────────────────

        # ── 去噪循环 ──
        use_cfg = (self.cfg_lambda > 0 and global_cond_uncond is not None)

        # ===== 部署性能优化：CFG batch 拼接 (Zread R3 reviewed) =====
        if use_cfg:
            assert global_cond.shape[0] == global_cond_uncond.shape[0], \
                (f"CFG batch mismatch: cond={global_cond.shape[0]} "
                 f"vs uncond={global_cond_uncond.shape[0]}")
            cond_batched = torch.cat([global_cond, global_cond_uncond], dim=0)
            local_cond_batched = (torch.cat([local_cond, local_cond], dim=0)
                                  if local_cond is not None else None)
        # ===========================================================

        for i, t in enumerate(timesteps):
            if i < 3 or i >= len(timesteps) - 2:
                ts = time.time()

            trajectory[condition_mask] = condition_data[condition_mask]
            t_batch = t.expand(B) if t.dim() == 0 else t

            if use_cfg:
                # batch 拼接 CFG：一次 forward 完成 cond + uncond
                traj_batched = torch.cat([trajectory, trajectory], dim=0)
                t_batched    = torch.cat([t_batch, t_batch], dim=0)
                pred_batched = model(sample=traj_batched, timestep=t_batched,
                                     local_cond=local_cond_batched,
                                     global_cond=cond_batched)
                pred_cond, pred_uncond = pred_batched.chunk(2, dim=0)
                model_output = pred_uncond + self.cfg_lambda * (
                    pred_cond - pred_uncond)
            else:
                model_output = model(sample=trajectory, timestep=t_batch,
                                     local_cond=local_cond,
                                     global_cond=global_cond)

            trajectory = scheduler.step(
                model_output, int(t.item()), trajectory).prev_sample

            if i < 3 or i >= len(timesteps) - 2:
                te = time.time()
                # print(f"[DEBUG] step {i:3d} | t={t.item():4d} | time={te-ts:.4f}s")
                ts = te

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # ──────────────────────────────────────────────────────────────
    # DPPO: DDIM stochastic sampling with log-probability
    # Design: DPPO_DDPM_RL_PostTraining_Design_v1_0.md §3.1
    # ──────────────────────────────────────────────────────────────

    def sample_ddpm_with_logprob(
        self,
        obs_dict,
        eta: float = 1.0,
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        num_inference_steps: int = None,
        ft_denoising_steps: int = None,
    ):
        """Official-DPPO DDIM stochastic sampling with exact
        log-probability (R7-2B: transition math executes the frozen
        official vendor runtime through OfficialPhase1aFacade; eta must
        be 1.0).

        Returns (action, logp, chain) for PPO training.

        Args:
            obs_dict: normalized observation dict
            eta: must be 1.0 (the official DDIM kernel semantics)
            ft_denoising_steps: if set, only last N steps contribute to logp
            min_sampling_denoising_std / min_logprob_denoising_std:
                the two INDEPENDENT std floors (sampling vs likelihood)
            num_inference_steps: if None, use self.num_inference_steps

        Returns:
            action: (B, horizon, 11) final denoised action chunk
            logp: (B,) γ-weighted joint log-probability (scalar, one PPO ratio)
            chain: list of (B, horizon, 11), length = K+1
                   chain[0]=x_T, chain[-1]=x_0=action
        """
        from diffusion_policy_3d.policy.dp_sampler import ddim_step_official
        from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
            make_official_cond,
        )
        from diffusion_policy_3d.common.pytorch_util import dict_apply

        device = self.device
        dtype = self.dtype
        B = 1
        # Resolve batch size from obs
        for v in obs_dict.values():
            if hasattr(v, 'shape'):
                B = v.shape[0]
                break

        T = self.horizon
        Da = self.action_dim
        K = num_inference_steps or self.num_inference_steps

        # 1. Encode observation
        nobs = self.normalizer.normalize(obs_dict)
        nobs = dict_apply(nobs, lambda x: x.to(device))
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        global_cond = self.obs_encoder(this_nobs).reshape(B, -1)

        # Unconditional encoding for CFG
        global_cond_uncond = None
        if self.cfg_lambda > 0:
            nobs_u = {k: v.clone() for k, v in nobs.items()}
            nobs_u['agent_pos'][:, :, 3:7] = 0.0
            this_u = dict_apply(
                nobs_u, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            global_cond_uncond = self.obs_encoder(this_u).reshape(B, -1)

        # 2. Setup scheduler
        scheduler = self.noise_scheduler
        self._ensure_scheduler_device(device)
        scheduler.set_timesteps(K)
        timesteps = scheduler.timesteps.to(device)

        # 3. Start from noise
        x_t = torch.randn(B, T, Da, device=device, dtype=dtype)
        chain = [x_t.clone()]

        # Determine which steps contribute to logp (last-step fine-tuning)
        if ft_denoising_steps is not None and ft_denoising_steps < K:
            logp_start_k = K - ft_denoising_steps
        else:
            logp_start_k = 0

        logp_steps = []  # per-step logp for γ-weighted joint computation

        # 4. Denoising loop — frozen official runtime (R7-2B): the
        #    backend owns epsilon prediction (CFG + sample→epsilon) and
        #    the facade's p_mean_var owns the transition; the project
        #    shell only injects the RNG noise (guide Step 7). Loop
        #    termination is decided HERE by the index range; the
        #    distribution has no t_prev terminal branch: the last step
        #    into valid timestep 0 is a real stochastic transition.
        cond = make_official_cond(global_cond, global_cond_uncond, device)
        for i in range(K):
            x_next, step_logp, _runtime = ddim_step_official(
                x_t=x_t, model=self.model, cond=cond, step_index=i,
                scheduler=scheduler, cfg_lambda=self.cfg_lambda, eta=eta,
                min_sampling_denoising_std=min_sampling_denoising_std,
                min_logprob_denoising_std=min_logprob_denoising_std,
                ft_denoising_steps=(
                    ft_denoising_steps if ft_denoising_steps is not None
                    else K
                ),
            )

            x_t = x_next
            if i >= logp_start_k:
                logp_steps.append(step_logp)
            chain.append(x_t.clone())

        # γ-weighted joint logp: Σ_k γ^k × step_logp_k (scalar per sample)
        # γ=1.0 → uniform sum (standard joint-chain). γ<1 → early steps discounted.
        # review partA P2-4: legacy-only value — the production RL path
        # recomputes per-dim logprobs via the kernel and takes gamma from
        # contract.gamma_denoising; this hardcoded 0.8 serves no live RL
        # consumer.
        if logp_steps:
            logp_per_step = torch.stack(logp_steps, dim=1)  # (B, ft_steps)
            ft_actual = logp_per_step.shape[1]
            ks = torch.arange(ft_actual, device=device, dtype=torch.float32)
            # FIX P0-4: γ^0=1 for latest denoising step, γ^(ft-1) for earliest
            weights = (0.8 ** (ft_actual - 1 - ks)).unsqueeze(0)
            joint_logp = (logp_per_step * weights).sum(dim=1)  # (B,) scalar
        else:
            joint_logp = torch.zeros(B, device=device)

        # chain[0]=x_T, chain[-1]=x_0=action
        return x_t, joint_logp, chain

    # ──────────────────────────────────────────────────────────────
    # Projection
    # ──────────────────────────────────────────────────────────────

    def _project_trajectory(self, action_pred, current_z=None,
                            point_cloud=None, current_pos=None,
                            current_yaw=None):
        """
        Projection 后处理: yaw 约束 + z 约束 + 障碍排斥

        action_pred: (B, H, 11) — dpos_world(3) + dvel(3) + acc(3) + dyaw(1) + yr(1)
        current_z: (B,) 当前高度（用于 z 投影）
        point_cloud: (N, 3) body-frame 点云（用于障碍排斥）
        current_pos: (B, 3) 当前 world-frame 位置（用于 delta→world 累积）
        current_yaw: (B,) 当前 yaw 角（用于点云 body→world 旋转）
        """
        if not self.use_projection:
            return action_pred

        traj = action_pred.clone()

        # ── 1. Δyaw 约束 ──
        dyaw = traj[..., 9]
        dyaw = torch.clamp(dyaw, -self.yaw_max_per_step,
                            self.yaw_max_per_step)
        yaw_accum = torch.cumsum(dyaw, dim=-1)
        over = torch.abs(yaw_accum) > self.yaw_max_accumulated
        if over.any():
            for b in range(traj.shape[0]):
                first = over[b].nonzero()
                if len(first) > 0:
                    idx = first[0].item()
                    sign = torch.sign(yaw_accum[b, idx])
                    if sign > 0:
                        dyaw[b, idx:] = torch.clamp(dyaw[b, idx:], max=0)
                    else:
                        dyaw[b, idx:] = torch.clamp(dyaw[b, idx:], min=0)
        traj[..., 9] = dyaw

        # ── 2. z 轴约束 ──
        if current_z is not None:
            dz = traj[..., 2]
            z = current_z + torch.cumsum(dz, dim=-1)
            z_c = torch.clamp(z, self.z_min, self.z_max)
            dz_new = z_c - torch.cat(
                [current_z.unsqueeze(-1), z_c[..., :-1]], dim=-1)
            traj[..., 2] = dz_new

        # ── 3. 障碍排斥投影 (Week 4) ──
        if (self.use_obstacle_projection
                and point_cloud is not None
                and current_pos is not None):
            traj = self._obstacle_project(traj, point_cloud, current_pos,
                                          current_yaw)

        return traj

    def _obstacle_project(self, traj, point_cloud, current_pos, current_yaw=None):
        """
        KD-Tree 障碍排斥: 将进入障碍安全边界的轨迹点投影到边界外。

        traj: (B, H, 11) — step-wise delta 位置在前 3 维
        point_cloud: (B, N, 3) body-frame 点云
        current_pos: (B, 3) world-frame 当前位置
        current_yaw: (B,) 当前 yaw 角（用于 body→world 旋转）

        原因: 模型输出轨迹可能穿过障碍物，需要后处理投影到安全边界外
        依据: v8.3 Week 4 规格: KD-Tree + safe_margin=0.5m
        坐标系: 点云 body frame → 用 R(yaw) 转到 world frame 后与轨迹比较
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return traj  # scipy 不可用时跳过

        B, H, D = traj.shape
        device = traj.device

        # 逐 batch 处理（每个 batch 的点云和 yaw 可能不同）
        cur_pos_np = current_pos.cpu().numpy()  # (B, 3)
        traj_np = traj.cpu().numpy().copy()

        if current_yaw is not None:
            cur_yaw_np = current_yaw.cpu().numpy()  # (B,)
        else:
            cur_yaw_np = np.zeros(B, dtype=np.float64)

        # 点云转 numpy
        if isinstance(point_cloud, torch.Tensor):
            pc_all = point_cloud.cpu().numpy()
        else:
            pc_all = np.asarray(point_cloud)
        if pc_all.ndim == 2:
            pc_all = pc_all[None, ...]  # (1, N, 3+)

        for b in range(B):
            # 取当前 batch 的点云 (body frame)
            pc_body = pc_all[b] if pc_all.shape[0] > 1 else pc_all[0]
            if pc_body.ndim > 2:
                pc_body = pc_body.reshape(-1, pc_body.shape[-1])
            pc_body = pc_body[:, :3].astype(np.float64)
            valid = np.isfinite(pc_body).all(axis=1)
            pc_body = pc_body[valid]

            if len(pc_body) < 5:
                continue  # 点太少，跳过

            # Fix 3: body frame → world frame
            # R(yaw) 旋转 + 平移
            yaw = cur_yaw_np[b]
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            R = np.array([[cos_y, -sin_y, 0],
                          [sin_y,  cos_y, 0],
                          [0,      0,     1]], dtype=np.float64)
            pc_world = (R @ pc_body.T).T + cur_pos_np[b][None, :]  # (N, 3) world frame

            # 构建 KD-Tree (world frame)
            kd_tree = cKDTree(pc_world)
            k = min(self.obstacle_k_neighbors, len(pc_world))

            # step-wise delta → world 位置 (cumsum)
            dpos = traj_np[b, :, :3]  # (H, 3)
            world_pos = np.zeros_like(dpos)
            world_pos[0] = cur_pos_np[b] + dpos[0]
            for t in range(1, H):
                world_pos[t] = world_pos[t-1] + dpos[t]

            # KD-Tree 查询每个轨迹点
            pt_dists, pt_idx = kd_tree.query(world_pos, k=k)
            mean_d = pt_dists.mean(axis=1) if k > 1 else pt_dists
            in_obs = mean_d < self.obstacle_safe_margin

            if not np.any(in_obs):
                continue

            # Opt 1: 简化排斥方向（不用 PCA，直接用轨迹点→最近点方向）
            # 原因: 全量 4096 点 PCA 约 1-2s CPU 开销，与推理时间目标冲突
            # 依据: zread R2 审查批准
            nearest = pt_idx[:, 0] if k > 1 else pt_idx
            for t in np.where(in_obs)[0]:
                npt = pc_world[nearest[t]]
                outward = world_pos[t] - npt
                out_norm = np.linalg.norm(outward)
                if out_norm < 1e-6:
                    direction = np.array([0.0, 0.0, 1.0])
                else:
                    direction = outward / out_norm
                world_pos[t] = npt + direction * (self.obstacle_safe_margin + 0.05)

            # world 位置 → 反算 step-wise delta
            traj_np[b, 0, :3] = world_pos[0] - cur_pos_np[b]
            for t in range(1, H):
                traj_np[b, t, :3] = world_pos[t] - world_pos[t-1]

        return torch.from_numpy(traj_np).to(device=device, dtype=traj.dtype)

    # ──────────────────────────────────────────────────────────────
    # predict_action (with optional obs_encoder / denoise timing)
    # ──────────────────────────────────────────────────────────────

    def predict_action(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        device = self.device
        nobs = dict_apply(nobs, lambda x: x.to(device))

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        dtype = self.dtype

        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(
                -1, *x.shape[2:]))

        # ── 计时：obs_encoder ──
        if self._deploy_timing and device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        global_cond = self.obs_encoder(this_nobs).reshape(B, -1)

        if self._deploy_timing and device.type == 'cuda':
            torch.cuda.synchronize()
        t_enc = time.time() - t0

        anchor_idx = None
        # ── 消融实验: 禁用 anchor_head，测试纯模型生成能力 ──
        # if self.anchor_head is not None:
        #     with torch.no_grad():
        #         anchor_idx = self.anchor_head(global_cond).argmax(dim=-1)

        global_cond_uncond = None
        if self.cfg_lambda > 0:
            nobs_u = {k: v.clone() for k, v in nobs.items()}
            nobs_u['agent_pos'][:, :, 3:7] = 0.0
            this_u = dict_apply(
                nobs_u, lambda x: x[:, :self.n_obs_steps, ...].reshape(
                    -1, *x.shape[2:]))
            global_cond_uncond = self.obs_encoder(this_u).reshape(B, -1)

        cond_data = torch.zeros(B, T, Da, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # ── 计时：去噪循环 ──
        if self._deploy_timing and device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        nsample = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=None, global_cond=global_cond,
            global_cond_uncond=global_cond_uncond,
            anchor_idx=anchor_idx, **self.kwargs)

        if self._deploy_timing and device.type == 'cuda':
            torch.cuda.synchronize()
        t_den = time.time() - t0

        if self._deploy_timing:
            self._timing_stats['t_enc'].append(t_enc * 1000)
            self._timing_stats['t_den'].append(t_den * 1000)

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        # Fix 4: 从原始 obs_dict 取坐标（不是归一化后的 nobs）
        # 原因: nobs['agent_pos'] 是归一化后的值，不是原始 world frame 坐标
        # 依据: zread R2 审查 Fix 4
        # agent_pos layout: pos_world(3) + goal_body(3) + dist(1) + sin_yaw(1) + cos_yaw(1)
        current_pos = obs_dict['agent_pos'][:, -1, :3]  # (B, 3) 原始 world frame
        if torch.is_tensor(current_pos):
            current_pos = current_pos.to(self.device)
        current_z = current_pos[:, 2]  # (B,)
        # 提取 yaw: sin_yaw=agent_pos[:,7], cos_yaw=agent_pos[:,8]
        sin_yaw = obs_dict['agent_pos'][:, -1, 7]
        cos_yaw = obs_dict['agent_pos'][:, -1, 8]
        if torch.is_tensor(sin_yaw):
            sin_yaw = sin_yaw.to(self.device)
            cos_yaw = cos_yaw.to(self.device)
        current_yaw = torch.atan2(sin_yaw, cos_yaw)  # (B,)
        # 点云也从 obs_dict 取（body frame，未归一化）
        pc_for_proj = obs_dict.get('point_cloud', None)
        if pc_for_proj is not None:
            pc_for_proj = pc_for_proj[:, -1]  # (B, N, 3+) → 取最新帧
            if torch.is_tensor(pc_for_proj):
                pc_for_proj = pc_for_proj.to(self.device)

        action = self._project_trajectory(
            action, current_z=current_z,
            point_cloud=pc_for_proj, current_pos=current_pos,
            current_yaw=current_yaw)
        return {'action_pred': action}

    def get_timing_stats(self):
        """返回推理计时统计汇总"""
        if not self._timing_stats['t_enc']:
            return None
        import numpy as np
        t_enc = np.array(self._timing_stats['t_enc'])
        t_den = np.array(self._timing_stats['t_den'])
        t_total = t_enc + t_den
        return {
            'n_calls': len(t_enc),
            't_enc_mean': t_enc.mean(),
            't_enc_std': t_enc.std(),
            't_den_mean': t_den.mean(),
            't_den_std': t_den.std(),
            't_total_mean': t_total.mean(),
            't_total_std': t_total.std(),
        }

    def print_timing_summary(self):
        """打印推理计时统计汇总"""
        stats = self.get_timing_stats()
        if stats is None:
            return
        print(f"\n[Timing Summary] n_calls={stats['n_calls']}")
        print(f"  obs_encoder: {stats['t_enc_mean']:.1f} ± {stats['t_enc_std']:.1f} ms")
        print(f"  denoise:     {stats['t_den_mean']:.1f} ± {stats['t_den_std']:.1f} ms")
        print(f"  total:       {stats['t_total_mean']:.1f} ± {stats['t_total_std']:.1f} ms")

    # ──────────────────────────────────────────────────────────────
    # compute_loss
    # ──────────────────────────────────────────────────────────────

    def compute_loss(self, batch):
        obs_dict = {k: v.clone() for k, v in batch['obs'].items()}

        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        dtype = self.dtype

        if self.training and self.cfg_dropout_prob > 0:
            drop = torch.rand(B, device=device) < self.cfg_dropout_prob
            if drop.any():
                obs_dict['agent_pos'][drop, :, 3:7] = 0.0

        nobs = self.normalizer.normalize(obs_dict)
        nobs = dict_apply(nobs, lambda x: x.to(device))

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        nactions = self.normalizer['action'].normalize(batch['action'])
        nactions = nactions.to(device)

        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        global_cond = self.obs_encoder(this_nobs).reshape(B, -1)

        trajectory = nactions
        self._ensure_scheduler_device(device)

        noise = torch.randn(trajectory.shape, device=device, dtype=dtype)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device).long()

        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        cond_data = torch.zeros(B, T, Da, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        noisy[cond_mask] = cond_data[cond_mask]

        pred = self.model(sample=noisy, timestep=timesteps,
                          local_cond=None, global_cond=global_cond)

        target = noise if self.noise_scheduler.config.prediction_type == 'epsilon' \
                 else trajectory
        loss_mask = ~cond_mask
        diff_loss = F.mse_loss(pred, target, reduction='none')
        diff_loss = (diff_loss * loss_mask.float()).mean()

        anchor_loss = torch.tensor(0.0, device=device)
        anchor_acc = 0.0
        if self.anchor_head is not None and self.training_phase != 'unet_warmup':
            logits = self.anchor_head(global_cond)
            labels = self._get_anchor_labels(nactions)
            if labels is not None:
                anchor_loss = F.cross_entropy(logits, labels)
                anchor_acc = (logits.argmax(1) == labels).float().mean().item()

        # Phase ②: Collision auxiliary loss (gated: only in joint phase)
        coll_loss = torch.tensor(0.0, device=device)
        coll_d_soft_p50 = 0.0
        coll_d_soft_p05 = 0.0
        coll_loss_ratio = 0.0
        if self.use_collision_loss and self.collision_lambda > 0 and self.training_phase == 'joint':
            from diffusion_policy_3d.model.common.collision_loss import collision_auxiliary_loss

            # Reconstruct clean x_0 prediction
            if self.noise_scheduler.config.prediction_type == 'epsilon':
                # Tweedie: x̂_0 = (x_t - √(1-ᾱ_t)·ε) / √(ᾱ_t)
                alpha_bar = self.noise_scheduler.alphas_cumprod[timesteps].to(device)
                alpha_bar = alpha_bar.reshape(B, 1, 1)
                pred_x0 = (noisy - torch.sqrt(1.0 - alpha_bar) * pred) / torch.sqrt(alpha_bar)
            else:
                # prediction_type == 'sample': model directly predicts x_0
                pred_x0 = pred

            # Unnormalize to physical coordinates (collision loss needs meters)
            pred_x0_action = self.normalizer['action'].unnormalize(pred_x0)

            # Current position (world frame)
            cur_pos = obs_dict['agent_pos'][:, -1, 0:3].to(device)

            # Point cloud (last frame, body frame)
            point_cloud = obs_dict['point_cloud'][:, -1].to(device)

            # [FIX 2026-08-14] 坐标系一致性 (V11 审计): 点云是 body frame,
            #   必须把 yaw 传给 collision_auxiliary_loss 做 body→world 变换
            #   (v1.2 §5.2: p_world = R(yaw) @ p_body + cur_pos)。
            #   agent_pos layout: pos(3) + goal_body(3) + dist(1) + sin_yaw(1) + cos_yaw(1)
            coll_sin_yaw = obs_dict['agent_pos'][:, -1, 7].to(device)
            coll_cos_yaw = obs_dict['agent_pos'][:, -1, 8].to(device)

            coll_loss_raw, coll_diag = collision_auxiliary_loss(
                pred_x0_action=pred_x0_action,
                point_cloud=point_cloud,
                cur_pos=cur_pos,
                d_safe=self.collision_d_safe,
                beta=self.collision_beta,
                tau_softmin=self.collision_tau_softmin,
                k_nn=self.collision_k_nn,
                horizon_decay_tau=self.collision_horizon_decay_tau,
                return_diagnostics=True,
                sin_yaw=coll_sin_yaw,
                cos_yaw=coll_cos_yaw,
            )
            coll_loss = self.collision_lambda * coll_loss_raw
            coll_d_soft_p50 = coll_diag.get('d_soft_p50', 0.0)
            coll_d_soft_p05 = coll_diag.get('d_soft_p05', 0.0)
            coll_loss_ratio = (coll_loss_raw / (diff_loss + 1e-8)).item() if diff_loss.item() > 0 else 0.0

        total = diff_loss + self.anchor_loss_weight * anchor_loss + coll_loss
        return total, {
            'bc_loss': diff_loss.item(),
            'anchor_loss': anchor_loss.item(),
            'anchor_acc': anchor_acc,
            'collision_loss': coll_loss.item() if isinstance(coll_loss, torch.Tensor) else coll_loss,
            'coll_loss_ratio': coll_loss_ratio,
            'coll_d_soft_p50': coll_d_soft_p50,
            'coll_d_soft_p05': coll_d_soft_p05,
            'total_loss': total.item(),
        }

    def set_training_phase(self, phase):
        self.training_phase = phase
        if phase == 'unet_warmup':
            for p in self.model.parameters(): p.requires_grad_(True)
            for p in self.obs_encoder.parameters(): p.requires_grad_(True)
            if self.anchor_head:
                for p in self.anchor_head.parameters(): p.requires_grad_(False)
        elif phase == 'anchor_only':
            for p in self.model.parameters(): p.requires_grad_(False)
            for p in self.obs_encoder.parameters(): p.requires_grad_(False)
            if self.anchor_head:
                for p in self.anchor_head.parameters(): p.requires_grad_(True)
        elif phase == 'joint':
            for p in self.parameters(): p.requires_grad_(True)
        cprint(f'[Phase1a] phase={phase}', 'cyan')

    def get_optimizer_groups(self, lr, anchor_lr_mult=5.0):
        groups = [
            {'params': list(self.model.parameters()), 'lr': lr},
            {'params': list(self.obs_encoder.parameters()), 'lr': lr},
        ]
        if self.anchor_head:
            groups.append({
                'params': list(self.anchor_head.parameters()),
                'lr': lr * anchor_lr_mult})
        return groups