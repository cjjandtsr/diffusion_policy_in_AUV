"""
phase1a_policy_fm.py — FM (Flow Matching) 版本: Anchor + OT-CFM + Euler + CFG + Projection
==========================================================================================
复制自 phase1a_policy.py，将 DDPM+DDIM diffusion 替换为 OT-CFM + Euler ODE。
AnchorHead / CFG / 安全引导模块 / Z/Yaw 投影 原封不动。

约定 (Lipman et al., 2023):
  t=0: 纯噪声 ε
  t=1: 干净数据
  插值: x_t = (1-t)*ε + t*data
  向量场: v = data - ε
  ODE: dx_t/dt = v_t, Euler 积分 t:0→1-sigma_min

放置: diffusion_policy_3d/policy/phase1a_policy_fm.py
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
# Phase ② collision auxiliary loss
from diffusion_policy_3d.model.common.collision_loss import (
    collision_auxiliary_loss)


class Phase1aPolicyFM(DiffusionPointcloudPolicy):

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
                 # ── FM 参数 (替换 DDPM truncated_steps / truncation_ratio) ──
                 num_fm_steps=4,
                 sigma_min=0.005,
                 fm_t_scale=50,
                 solver='euler',
                 fm_anchor_t_start=0.95,
                 # Projection (拆三独立开关 — Route B RL 前置)
                 use_projection=True,          # 兼容旧接口：若 False 跳过全部投影
                 use_yaw_projection=True,      # ★新增: yaw 回拉 (RL 设 False)
                 use_z_clamp=True,             # ★新增: z clamp safety envelope
                 z_min=1.0, z_max=5.0,
                 yaw_max_per_step=0.524,
                 yaw_max_accumulated=3.14159,
                 # Obstacle Projection (Week 4)
                 use_obstacle_projection=False,
                 obstacle_safe_margin=0.5,
                 obstacle_k_neighbors=20,
                 # Phase ② Constraint-Aware BC: collision auxiliary loss
                 use_collision_loss=False,
                 collision_lambda=0.5,
                 collision_d_safe=0.75,
                 collision_beta=20.0,
                 collision_tau_softmin=5.0,
                 collision_k_nn=8,
                 collision_horizon_decay_tau=8.0,
                 # ── 层次 1: SFP 稳定化项 + jerk 正则 ──
                 k_stab=0.0,                # SFP 稳定化增益; 0=禁用(退回原始 FM)
                 lambda_jerk=0.0,           # jerk 正则权重; 0=禁用
                 lambda_yaw_jerk=0.1,       # jerk 中 yaw 项相对权重(量纲分开)
                 jerk_smooth_dims=(0, 1, 2, 9),  # dpos_x/y/z + dyaw
                 # Training phase
                 training_phase='joint',
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
        self.training_phase = training_phase

        # ── FM 参数 ──
        self.num_fm_steps = num_fm_steps
        self.sigma_min = sigma_min
        self.fm_t_scale = fm_t_scale
        self.solver = solver
        self.fm_anchor_t_start = fm_anchor_t_start

        self.use_projection = use_projection
        self.use_yaw_projection = use_yaw_projection    # ★新增
        self.use_z_clamp = use_z_clamp                  # ★新增
        self.z_min = z_min
        self.z_max = z_max
        self.yaw_max_per_step = yaw_max_per_step
        self.yaw_max_accumulated = yaw_max_accumulated
        self.use_obstacle_projection = use_obstacle_projection
        self.obstacle_safe_margin = obstacle_safe_margin
        self.obstacle_k_neighbors = obstacle_k_neighbors

        # Phase ② collision auxiliary loss 参数
        self.use_collision_loss = use_collision_loss
        self.collision_lambda = collision_lambda
        self.collision_d_safe = collision_d_safe
        self.collision_beta = collision_beta
        self.collision_tau_softmin = collision_tau_softmin
        self.collision_k_nn = collision_k_nn
        self.collision_horizon_decay_tau = collision_horizon_decay_tau
        if use_collision_loss:
            cprint(f'[Phase2] collision aux loss ENABLED: '
                   f'λ_col={collision_lambda}, d_safe={collision_d_safe}m, '
                   f'β={collision_beta}, k={collision_k_nn}, '
                   f'τ_softmin={collision_tau_softmin}', 'green')

        # ── 层次 1: SFP 稳定化项 + jerk 正则 ──
        # 注册为 buffer 以便自动持久化到 checkpoint state_dict
        # （部署时 _load_phase1a_policy_fm 可从 sd 自动检测 k_stab/lambda_jerk）
        self.register_buffer('k_stab', torch.tensor(k_stab, dtype=torch.float32))
        self.register_buffer('lambda_jerk', torch.tensor(lambda_jerk, dtype=torch.float32))
        self.lambda_yaw_jerk = lambda_yaw_jerk
        self.jerk_smooth_dims = tuple(jerk_smooth_dims)
        if k_stab > 0:
            cprint(f'[Layer1] SFP stabilization ENABLED: k_stab={k_stab} '
                   f'(target_v = (traj-noise) - k*(x_t-traj))', 'yellow')
        if lambda_jerk > 0:
            cprint(f'[Layer1] jerk reg ENABLED: λ_jerk={lambda_jerk}, '
                   f'λ_yaw={lambda_yaw_jerk}, dims={jerk_smooth_dims} '
                   f'(L_jerk = mean(d2_pos²) + λ_yaw·mean(d2_yaw²))', 'yellow')

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

        cprint(f'[Phase1aFM] OT-CFM: num_fm_steps={num_fm_steps}, '
               f'sigma_min={sigma_min}, solver={solver}, '
               f'fm_t_scale={fm_t_scale}', 'cyan')
        cprint(f'[Phase1aFM] Anchor t_start={fm_anchor_t_start} '
               f'(v1.3 约定: t越大anchor权重越大)', 'cyan')
        cprint(f'[Phase1aFM] CFG dropout={cfg_dropout_prob}, λ={cfg_lambda}',
               'cyan')

        # 部署计时标志（默认关闭，部署时手动设置为 True）
        self._deploy_timing = False
        # 计时统计收集
        self._timing_stats = {'t_enc': [], 't_den': []}
        
    def _load_anchors(self, path):
        bank = torch.load(path, map_location='cpu')
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
        if self.anchor_bank_full is None:
            return None

        B = nactions.shape[0]
        Da = self.action_dim
        To = self.n_obs_steps

        start = To - 1
        end = start + self.n_action_steps
        chunk = nactions[:, start:end, :].detach()

        anchors = self.anchor_bank_full.to(chunk.device)
        n_anchors_norm = self.normalizer['action'].normalize(
            anchors.reshape(-1, Da)).reshape(self.n_anchors, -1, Da)

        idx_4d = [0, 1, 2, 9]
        chunk_4d = chunk[:, :, idx_4d].reshape(B, -1)
        anchor_4d = n_anchors_norm[:, :, idx_4d].reshape(self.n_anchors, -1)

        dists = torch.cdist(chunk_4d.unsqueeze(0),
                             anchor_4d.unsqueeze(0)).squeeze(0)
        return dists.argmin(dim=-1)

    def _get_anchor_trajectory(self, anchor_idx):
        B = anchor_idx.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        device = anchor_idx.device

        selected = self.anchor_bank_full.to(device)[anchor_idx]
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
    # FM Conditional Sampling (OT-CFM + Euler ODE)
    # ──────────────────────────────────────────────────────────────

    def conditional_sample_fm(self,
                               condition_data, condition_mask,
                               condition_data_pc=None, condition_mask_pc=None,
                               local_cond=None, global_cond=None,
                               global_cond_uncond=None,
                               anchor_idx=None,
                               init_noise=None,           # ★Route B RL: DSRL 传入 z
                               generator=None, **kwargs):
        """
        FM 标准约定 (Lipman et al., 2023):
          t=0: 纯噪声 ε
          t=1: 干净数据
          插值: x_t = (1-t)*ε + t*data
          向量场: v_t = data - ε
          ODE: dx_t/dt = v_t, 从 t=0 积分到 t=1-sigma_min
        """
        B, T, Da = condition_data.shape
        device = condition_data.device
        dtype = condition_data.dtype

        # Anchor 初始化
        # t_start 越大 → anchor 权重越大 (v1.3 约定)
        if init_noise is not None:
            # ★Route B RL: DSRL 外部传入 z 作为初始噪声（黑盒复用唯一侵入点）
            x_t = init_noise.to(device=device, dtype=dtype)
            t_start = 0.0
        elif anchor_idx is not None and self.anchor_bank_full is not None:
            anchor_traj = self._get_anchor_trajectory(anchor_idx)
            noise = torch.randn(anchor_traj.shape, device=device, dtype=dtype)
            t_start = self.fm_anchor_t_start
            x_t = (1 - t_start) * noise + t_start * anchor_traj
        else:
            noise = torch.randn(condition_data.shape, device=device, dtype=dtype)
            x_t = noise  # t=0, 纯噪声
            t_start = 0.0

        # 使用 linspace(N+1) 生成 N 个积分区间，保证到达 t=1-sigma_min
        t_end = 1.0 - self.sigma_min
        t_start = min(t_start, t_end - 1e-6)  # ensure t_start < t_end for forward integration
        timesteps = torch.linspace(t_start, t_end, self.num_fm_steps + 1,
                                   device=device)
        use_cfg = (self.cfg_lambda > 0 and global_cond_uncond is not None)

        if use_cfg:
            assert global_cond.shape[0] == global_cond_uncond.shape[0]

        for i in range(self.num_fm_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            x_t[condition_mask] = condition_data[condition_mask]

            # ⚠️ 必须是 float tensor (绕开 UNet forward() 中 dtype=torch.long 硬编码)
            t_input = (t * self.fm_t_scale).float()
            t_input = t_input.expand(B)

            if use_cfg:
                x_t_batched = torch.cat([x_t, x_t], dim=0)
                t_batched = torch.cat([t_input, t_input], dim=0)
                global_batched = torch.cat([global_cond, global_cond_uncond], dim=0)
                local_batched = (torch.cat([local_cond, local_cond], dim=0)
                                 if local_cond is not None else None)
                v_batched = self.model(sample=x_t_batched, timestep=t_batched,
                                       local_cond=local_batched,
                                       global_cond=global_batched)
                v_cond, v_uncond = v_batched.chunk(2, dim=0)
                v = v_uncond + self.cfg_lambda * (v_cond - v_uncond)
            else:
                v = self.model(sample=x_t, timestep=t_input,
                               local_cond=local_cond,
                               global_cond=global_cond)

            if self.solver == 'euler':
                x_t = x_t + v * dt  # 标准约定: 从噪声走向数据
            else:
                raise NotImplementedError(f"solver={self.solver}")

        x_t[condition_mask] = condition_data[condition_mask]
        return x_t

    # ──────────────────────────────────────────────────────────────
    # Projection
    # ──────────────────────────────────────────────────────────────

    def _project_trajectory(self, action_pred, current_z=None,
                            point_cloud=None, current_pos=None,
                            current_yaw=None):
        # 兼容旧接口: use_projection=False 跳过全部投影 (部署回归)
        if not self.use_projection:
            return action_pred

        traj = action_pred.clone()

        # ── yaw 回拉 (独立开关, RL 采样设 False) ──
        if self.use_yaw_projection:
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

        # ── z clamp (独立开关, RL 采样保持 True — safety envelope) ──
        if self.use_z_clamp and current_z is not None:
            dz = traj[..., 2]
            z = current_z.unsqueeze(-1) + torch.cumsum(dz, dim=-1)
            z_c = torch.clamp(z, self.z_min, self.z_max)
            dz_new = z_c - torch.cat(
                [current_z.unsqueeze(-1), z_c[..., :-1]], dim=-1)
            traj[..., 2] = dz_new

        # ── 障碍投影 (已有独立开关) ──
        if (self.use_obstacle_projection
                and point_cloud is not None
                and current_pos is not None):
            traj = self._obstacle_project(traj, point_cloud, current_pos,
                                          current_yaw)

        return traj

    def _obstacle_project(self, traj, point_cloud, current_pos, current_yaw=None):
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return traj

        B, H, D = traj.shape
        device = traj.device

        cur_pos_np = current_pos.cpu().numpy()
        traj_np = traj.cpu().numpy().copy()

        if current_yaw is not None:
            cur_yaw_np = current_yaw.cpu().numpy()
        else:
            cur_yaw_np = np.zeros(B, dtype=np.float64)

        if isinstance(point_cloud, torch.Tensor):
            pc_all = point_cloud.cpu().numpy()
        else:
            pc_all = np.asarray(point_cloud)
        if pc_all.ndim == 2:
            pc_all = pc_all[None, ...]

        for b in range(B):
            pc_body = pc_all[b] if pc_all.shape[0] > 1 else pc_all[0]
            if pc_body.ndim > 2:
                pc_body = pc_body.reshape(-1, pc_body.shape[-1])
            pc_body = pc_body[:, :3].astype(np.float64)
            valid = np.isfinite(pc_body).all(axis=1)
            pc_body = pc_body[valid]

            if len(pc_body) < 5:
                continue

            yaw = cur_yaw_np[b]
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            R = np.array([[cos_y, -sin_y, 0],
                          [sin_y,  cos_y, 0],
                          [0,      0,     1]], dtype=np.float64)
            pc_world = (R @ pc_body.T).T + cur_pos_np[b][None, :]

            kd_tree = cKDTree(pc_world)
            k = min(self.obstacle_k_neighbors, len(pc_world))

            dpos = traj_np[b, :, :3]
            world_pos = np.zeros_like(dpos)
            world_pos[0] = cur_pos_np[b] + dpos[0]
            for t in range(1, H):
                world_pos[t] = world_pos[t-1] + dpos[t]

            pt_dists, pt_idx = kd_tree.query(world_pos, k=k)
            mean_d = pt_dists.mean(axis=1) if k > 1 else pt_dists
            in_obs = mean_d < self.obstacle_safe_margin

            if not np.any(in_obs):
                continue

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

            traj_np[b, 0, :3] = world_pos[0] - cur_pos_np[b]
            for t in range(1, H):
                traj_np[b, t, :3] = world_pos[t] - world_pos[t-1]

        return torch.from_numpy(traj_np).to(device=device, dtype=traj.dtype)

    # ──────────────────────────────────────────────────────────────
    # predict_action
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

        if self._deploy_timing and device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        nsample = self.conditional_sample_fm(
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

        current_pos = obs_dict['agent_pos'][:, -1, :3]
        if torch.is_tensor(current_pos):
            current_pos = current_pos.to(self.device)
        current_z = current_pos[:, 2]
        sin_yaw = obs_dict['agent_pos'][:, -1, 7]
        cos_yaw = obs_dict['agent_pos'][:, -1, 8]
        if torch.is_tensor(sin_yaw):
            sin_yaw = sin_yaw.to(self.device)
            cos_yaw = cos_yaw.to(self.device)
        current_yaw = torch.atan2(sin_yaw, cos_yaw)
        pc_for_proj = obs_dict.get('point_cloud', None)
        if pc_for_proj is not None:
            pc_for_proj = pc_for_proj[:, -1]
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
    # compute_loss (含详细诊断，持续打印)
    # ──────────────────────────────────────────────────────────────

    def compute_loss(self, batch, global_step=None):
        obs_dict = {k: v.clone() for k, v in batch['obs'].items()}

        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        dtype = self.dtype

        if self.use_collision_loss and not getattr(self, '_phase2_logged', False):
            cprint(f"[Phase2FM] compute_loss: OT-CFM vector field prediction, "
                   f"x̂₀ = x_t + (1-t)*v̂ (FM analytic recovery), "
                   f"λ_col={self.collision_lambda}, d_safe={self.collision_d_safe}m",
                   'green')
            self._phase2_logged = True

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

        # ── OT-CFM loss (替换 DDPM noise/loss) ──
        noise = torch.randn(trajectory.shape, device=device, dtype=dtype)
        # t ~ U[sigma_min, 1]
        t = torch.rand(B, device=device) * (1.0 - self.sigma_min) + self.sigma_min
        # 插值: x_t = (1-t)*ε + t*data
        x_t = (1 - t.view(B, 1, 1)) * noise + t.view(B, 1, 1) * trajectory
        # 向量场目标: v = data - ε
        # ── 层次 1 (b): SFP 稳定化项 (选择 B, 真正的 -k(x_t - trajectory)) ──
        #   v_ξ(a,t) = ξ̇(t) - k·(a - ξ(t))  [SFP Eq.2; ξ=trajectory, a=x_t]
        #   on-trajectory: x_t - trajectory = -(1-t)·noise, 数值与简化一致,
        #   但保留拉回 demonstration 的物理意义 (Zread 选择 B)
        #   k_stab=0 → 严格退回 target_v = trajectory - noise
        if self.k_stab > 0:
            stab = -self.k_stab * (x_t - trajectory)        # (B, T, Da)
            target_v = (trajectory - noise) + stab
        else:
            target_v = trajectory - noise
        # ⚠️ timestep 必须是 float tensor (绕开 UNet forward() 中 dtype=torch.long)
        t_input = (t * self.fm_t_scale).float()
        assert t_input.shape == (B,), f"t_input shape must be (B,), got {t_input.shape}"

        cond_data = torch.zeros(B, T, Da, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        x_t[cond_mask] = cond_data[cond_mask]

        pred_v = self.model(sample=x_t, timestep=t_input,
                            local_cond=None, global_cond=global_cond)

        loss_mask = ~cond_mask
        diff_loss = F.mse_loss(pred_v, target_v, reduction='none')
        diff_loss = (diff_loss * loss_mask.float()).mean()

        anchor_loss = torch.tensor(0.0, device=device)
        anchor_acc = 0.0
        if self.anchor_head is not None and self.training_phase != 'unet_warmup':
            logits = self.anchor_head(global_cond)
            labels = self._get_anchor_labels(nactions)
            if labels is not None:
                anchor_loss = F.cross_entropy(logits, labels)
                anchor_acc = (logits.argmax(1) == labels).float().mean().item()

        # Phase ② collision auxiliary loss
        # Gating: 只在 use_collision_loss=True 且 training 且 training_phase='joint' 时生效
        #   Step 2 (anchor_only) 和 eval 时自动跳过
        collision_loss = torch.tensor(0.0, device=device)
        collision_diag = {}

        # ── 层次 1: need_x0 guard ──
        #   collision loss 和 jerk 正则都需要 pred_x0 (数据端解析恢复).
        #   按需计算一次, 两者共用, 避免无谓的 unnormalize.
        _collision_active = (self.use_collision_loss and self.training
                             and self.training_phase == 'joint')
        _jerk_active = (self.lambda_jerk > 0 and self.training
                        and self.training_phase == 'joint')
        need_x0 = _collision_active or _jerk_active

        pred_x0_action = None
        if need_x0:
            # FM: 从向量场预测恢复数据端 x̂_data = x_t + (1-t)*v̂
            # v1.3 约定: x_t = (1-t)*ε + t*data, v = data-ε
            # x_t + (1-t)*v = (1-t)*ε + t*data + (1-t)*(data-ε) = data ✓
            pred_x0_norm = x_t + (1 - t.view(B, 1, 1)) * pred_v
            # 反归一化到物理米制 (距离/jerk 计算需要物理坐标)
            pred_x0_action = self.normalizer['action'].unnormalize(pred_x0_norm)

        # ── 层次 1 (a): jerk 正则 (chunk 内二阶差分; dpos/dyaw 量纲分开) ──
        # 术语说明 (Zread R5): 严格 jerk = ‖d³x/dt³‖² (加速度的导数).
        #   此处用位置二阶差分 d²x ≈ x[i+1]-2x[i]+x[i-1] 作为代理 (实为加速度代理).
        #   二阶差分对 chunk 内 zigzag 已足够敏感; 若需严格 jerk 可改三阶差分
        #   d³x ≈ x[i+2]-3x[i+1]+3x[i]-x[i-1]. 论文写作时按"二阶差分平滑正则"描述.
        jerk_loss = torch.tensor(0.0, device=device)
        if _jerk_active:
            dims = self.jerk_smooth_dims
            pos_dims = [d for d in dims if d in (0, 1, 2)]   # 位置维(米)
            yaw_dims = [d for d in dims if d == 9]           # yaw维(弧度)
            # 二阶差分 = jerk 的离散代理 (沿 step / horizon 维)
            jerk_pos = torch.tensor(0.0, device=device)
            jerk_yaw = torch.tensor(0.0, device=device)
            if pos_dims:
                a = pred_x0_action[:, :, pos_dims]           # (B, T, |pos|)
                d1 = a[:, 1:] - a[:, :-1]                    # (B, T-1, *)
                d2 = d1[:, 1:] - d1[:, :-1]                  # (B, T-2, *)
                jerk_pos = (d2 ** 2).mean()
            if yaw_dims:
                ay = pred_x0_action[:, :, yaw_dims]          # (B, T, 1)
                d1y = ay[:, 1:] - ay[:, :-1]
                d2y = d1y[:, 1:] - d1y[:, :-1]
                jerk_yaw = (d2y ** 2).mean()
            # 量纲分开加权 (Zread: 米与弧度不可直接相加)
            jerk_loss = jerk_pos + self.lambda_yaw_jerk * jerk_yaw

        if _collision_active:
            # 当前 drone 位置 (world frame)
            cur_pos = obs_dict['agent_pos'][:, -1, 0:3].to(device)

            # 点云 (取最后一帧, 与 action 同增广帧)
            point_cloud = obs_dict['point_cloud'][:, -1].to(device)

            # 计算 collision auxiliary loss (含诊断)
            collision_loss, collision_diag = collision_auxiliary_loss(
                pred_x0_action=pred_x0_action,
                point_cloud=point_cloud,
                cur_pos=cur_pos,
                d_safe=self.collision_d_safe,
                beta=self.collision_beta,
                tau_softmin=self.collision_tau_softmin,
                k_nn=self.collision_k_nn,
                horizon_decay_tau=self.collision_horizon_decay_tau,
                return_diagnostics=True,
            )

        total = (diff_loss
                 + self.anchor_loss_weight * anchor_loss
                 + self.collision_lambda * collision_loss
                 + self.lambda_jerk * jerk_loss)          # 层次 1 (a)

        log_dict = {
            'bc_loss': diff_loss.item(),
            'anchor_loss': anchor_loss.item(),
            'anchor_acc': anchor_acc,
            'collision_loss': collision_loss.item(),
            'jerk_loss': jerk_loss.item(),               # 层次 1 (a)
            'total_loss': total.item(),
        }
        # 层次 1: jerk 诊断 (仅在生效时写, 便于 train 脚本聚合)
        if self.lambda_jerk > 0 and self.training and self.training_phase == 'joint':
            log_dict['jerk_active'] = 1.0
        for k, v in collision_diag.items():
            log_dict[f'coll_{k}'] = v
        return total, log_dict

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