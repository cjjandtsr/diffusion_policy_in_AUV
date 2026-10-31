"""
phase1a_policy_fm_v9.py — Phase α FM Policy: Graded Collision Loss + Sample Reweighting
========================================================================================
基于 phase1a_policy_fm.py（FM + Anchor + CFG + SFP + Projection），Phase α 扩展：

α1: 三级距离分级 collision loss（替代单级 d_safe=0.75m）
α2: SAFE-GIL 式样本重加权（min_dist → BC loss weight）

放置: diffusion_policy_3d/policy/phase1a_policy_fm_v9.py
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
# Phase ② collision auxiliary loss (原版，保留兼容)
from diffusion_policy_3d.model.common.collision_loss import (
    collision_auxiliary_loss)
# [P0-A] 坐标系修复版 collision loss（单级，保留兼容）
from diffusion_policy_3d.model.common.collision_loss_v9 import (
    collision_auxiliary_loss as collision_auxiliary_loss_v9)
# [Phase α Finetune] 三级分级 collision loss with z-downweight + linear hinge
from diffusion_policy_3d.model.common.collision_loss_v9_1 import (
    collision_auxiliary_loss_graded)


class Phase1aPolicyFMV9(DiffusionPointcloudPolicy):
    """Phase α Finetune (v9_1): z-downweighted collision loss + early avoidance enhancement"""

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
                 # ── FM 参数 ──
                 num_fm_steps=4,
                 sigma_min=0.005,
                 fm_t_scale=50,
                 solver='euler',
                 fm_anchor_t_start=0.95,
                 # Projection
                 use_projection=True,
                 z_min=1.0, z_max=5.0,
                 yaw_max_per_step=0.524,
                 yaw_max_accumulated=3.14159,
                 # Obstacle Projection
                 use_obstacle_projection=False,
                 obstacle_safe_margin=0.5,
                 obstacle_k_neighbors=20,
                 # ── Phase α Collision Loss (graded, v9_1 finetune) ──
                 use_collision_loss=False,
                 collision_lambda=0.5,
                 collision_eps_levels=(2.0, 1.5, 0.75),
                 collision_weights=(0.5, 2.0, 3.0),         # [Task 2] 上调预警/关注级
                 collision_hinge_types=('linear', 'quadratic', 'quadratic'),  # [Task 2] 预警级 linear
                 collision_z_weight=0.2,                     # [Task 1] z 降权
                 collision_beta=20.0,
                 collision_tau_softmin=5.0,
                 collision_k_nn=8,
                 collision_horizon_decay_tau=8.0,
                 collision_lambda_t_schedule='linear',
                 # ── Phase α2: sample reweighting ──
                 use_sample_reweight=False,
                 reweight_alpha=2.0,
                 reweight_tau=1.0,
                 # ── Collision lambda warmup ──
                 collision_warmup_epochs=50,
                 # ── 层次 1: SFP 稳定化项 + jerk 正则 ──
                 k_stab=0.0,
                 lambda_jerk=0.0,
                 lambda_yaw_jerk=0.1,
                 jerk_smooth_dims=(0, 1, 2, 9),
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
        self.z_min = z_min
        self.z_max = z_max
        self.yaw_max_per_step = yaw_max_per_step
        self.yaw_max_accumulated = yaw_max_accumulated
        self.use_obstacle_projection = use_obstacle_projection
        self.obstacle_safe_margin = obstacle_safe_margin
        self.obstacle_k_neighbors = obstacle_k_neighbors

        # ── Phase α Collision Loss (graded, v9_1 finetune) ──
        self.use_collision_loss = use_collision_loss
        self.collision_lambda = collision_lambda
        self.collision_eps_levels = collision_eps_levels
        self.collision_weights = collision_weights
        self.collision_hinge_types = collision_hinge_types    # [Task 2]
        self.collision_z_weight = collision_z_weight           # [Task 1]
        self.collision_beta = collision_beta
        self.collision_tau_softmin = collision_tau_softmin
        self.collision_k_nn = collision_k_nn
        self.collision_horizon_decay_tau = collision_horizon_decay_tau
        self.collision_lambda_t_schedule = collision_lambda_t_schedule
        self.collision_warmup_epochs = collision_warmup_epochs
        # warmup state: 由训练脚本通过 set_collision_warmup_progress 更新
        self._collision_warmup_progress = 0.0  # [0, 1]

        # ── Phase α2: sample reweighting ──
        self.use_sample_reweight = use_sample_reweight
        self.reweight_alpha = reweight_alpha
        self.reweight_tau = reweight_tau

        if use_collision_loss:
            eps_str = '/'.join(f'{e:.1f}' for e in collision_eps_levels)
            w_str = '/'.join(f'{w:.1f}' for w in collision_weights)
            h_str = '/'.join(collision_hinge_types)
            cprint(f'[Phaseα v9_1] Graded collision loss ENABLED: '
                   f'eps=[{eps_str}]m, w=[{w_str}], hinge=[{h_str}], '
                   f'z_weight={collision_z_weight}, λ_col={collision_lambda}, warmup={collision_warmup_epochs}ep',
                   'green')
        if use_sample_reweight:
            cprint(f'[Phaseα2] Sample reweight ENABLED: '
                   f'α={reweight_alpha}, τ={reweight_tau}m',
                   'yellow')

        # ── 层次 1: SFP 稳定化项 + jerk 正则 ──
        self.register_buffer('k_stab', torch.tensor(k_stab, dtype=torch.float32))
        self.register_buffer('lambda_jerk', torch.tensor(lambda_jerk, dtype=torch.float32))
        self.lambda_yaw_jerk = lambda_yaw_jerk
        self.jerk_smooth_dims = tuple(jerk_smooth_dims)
        if k_stab > 0:
            cprint(f'[Layer1] SFP stabilization ENABLED: k_stab={k_stab}', 'yellow')
        if lambda_jerk > 0:
            cprint(f'[Layer1] jerk reg ENABLED: λ_jerk={lambda_jerk}, '
                   f'λ_yaw={lambda_yaw_jerk}', 'yellow')

        # Anchor bank
        self.anchor_bank_full = None
        self.n_anchors = 0
        if anchor_bank_path is not None:
            self._load_anchors(anchor_bank_path)

        # AnchorHead
        if self.n_anchors > 0:
            obs_feat_dim = self.obs_feature_dim * n_obs_steps
            self.anchor_head = AnchorHead(
                input_dim=obs_feat_dim, n_anchors=self.n_anchors)
            cprint(f'[Phase1a] AnchorHead: {obs_feat_dim}→{self.n_anchors}', 'cyan')
        else:
            self.anchor_head = None

        cprint(f'[Phase1aFM-V9] OT-CFM: num_fm_steps={num_fm_steps}, '
               f'sigma_min={sigma_min}, solver={solver}, '
               f'fm_t_scale={fm_t_scale}', 'cyan')
        cprint(f'[Phase1aFM-V9] Anchor t_start={fm_anchor_t_start}', 'cyan')
        cprint(f'[Phase1aFM-V9] CFG dropout={cfg_dropout_prob}, λ={cfg_lambda}', 'cyan')

        self._deploy_timing = False
        self._timing_stats = {'t_enc': [], 't_den': []}

    def set_collision_warmup_progress(self, progress):
        """
        设置 collision lambda warmup 进度。

        Args:
            progress: float ∈ [0, 1], 0=warmup 开始, 1=warmup 完成
        """
        self._collision_warmup_progress = max(0.0, min(1.0, progress))

    def _get_effective_collision_lambda(self):
        """返回 warmup 后的有效 collision_lambda"""
        if self._collision_warmup_progress >= 1.0:
            return self.collision_lambda
        return self.collision_lambda * self._collision_warmup_progress

    def _load_anchors(self, path):
        bank = torch.load(path, map_location='cpu')
        if hasattr(self, 'anchor_bank_full'):
            del self.anchor_bank_full
        self.register_buffer('anchor_bank_full',
                             bank['anchors_full'].float())
        self.n_anchors = bank['K_plus_1']
        cprint(f'[Phase1a] {self.n_anchors} anchors from {path}', 'green')

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

    # ── Anchor helpers (unchanged from phase1a_policy_fm.py) ──

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

    # ── FM Conditional Sampling (unchanged from phase1a_policy_fm.py) ──

    def conditional_sample_fm(self,
                               condition_data, condition_mask,
                               condition_data_pc=None, condition_mask_pc=None,
                               local_cond=None, global_cond=None,
                               global_cond_uncond=None,
                               anchor_idx=None,
                               generator=None, **kwargs):
        B, T, Da = condition_data.shape
        device = condition_data.device
        dtype = condition_data.dtype

        if anchor_idx is not None and self.anchor_bank_full is not None:
            anchor_traj = self._get_anchor_trajectory(anchor_idx)
            noise = torch.randn(anchor_traj.shape, device=device, dtype=dtype)
            t_start = self.fm_anchor_t_start
            x_t = (1 - t_start) * noise + t_start * anchor_traj
        else:
            noise = torch.randn(condition_data.shape, device=device, dtype=dtype)
            x_t = noise
            t_start = 0.0

        t_end = 1.0 - self.sigma_min
        t_start = min(t_start, t_end - 1e-6)
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
                x_t = x_t + v * dt
            else:
                raise NotImplementedError(f"solver={self.solver}")

        x_t[condition_mask] = condition_data[condition_mask]
        return x_t

    # ── Projection (unchanged) ──

    def _project_trajectory(self, action_pred, current_z=None,
                            point_cloud=None, current_pos=None,
                            current_yaw=None):
        if not self.use_projection:
            return action_pred

        traj = action_pred.clone()

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

        if current_z is not None:
            dz = traj[..., 2]
            z = current_z + torch.cumsum(dz, dim=-1)
            z_c = torch.clamp(z, self.z_min, self.z_max)
            dz_new = z_c - torch.cat(
                [current_z.unsqueeze(-1), z_c[..., :-1]], dim=-1)
            traj[..., 2] = dz_new

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

    # ── predict_action (unchanged) ──

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
        # ── 消融实验: 禁用 anchor_head ──

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
        stats = self.get_timing_stats()
        if stats is None:
            return
        print(f"\n[Timing Summary] n_calls={stats['n_calls']}")
        print(f"  obs_encoder: {stats['t_enc_mean']:.1f} ± {stats['t_enc_std']:.1f} ms")
        print(f"  denoise:     {stats['t_den_mean']:.1f} ± {stats['t_den_std']:.1f} ms")
        print(f"  total:       {stats['t_total_mean']:.1f} ± {stats['t_total_std']:.1f} ms")

    # ═══════════════════════════════════════════════════════════════
    # compute_loss — Phase α 核心修改
    # ═══════════════════════════════════════════════════════════════

    def compute_loss(self, batch, global_step=None):
        obs_dict = {k: v.clone() for k, v in batch['obs'].items()}

        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        dtype = self.dtype

        # α2: 提取 min_dist 用于样本重加权
        min_dist = batch.get('min_dist', None)
        if min_dist is not None:
            min_dist = min_dist.to(device)

        if self.use_collision_loss and not getattr(self, '_phase_alpha_logged', False):
            eps_str = '/'.join(f'{e:.1f}' for e in self.collision_eps_levels)
            w_str = '/'.join(f'{w:.1f}' for w in self.collision_weights)
            h_str = '/'.join(self.collision_hinge_types)
            cprint(f"[Phaseα v9_1] compute_loss: Graded collision loss "
                   f"eps=[{eps_str}]m, w=[{w_str}], hinge=[{h_str}], "
                   f"z_weight={self.collision_z_weight}, λ_col={self.collision_lambda}, warmup_progress={self._collision_warmup_progress:.2f}",
                   'green')
            if self.use_sample_reweight:
                cprint(f"[Phaseα2] Sample reweight ENABLED: "
                       f"α={self.reweight_alpha}, τ={self.reweight_tau}m",
                       'yellow')
            self._phase_alpha_logged = True

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

        # ── OT-CFM loss ──
        noise = torch.randn(trajectory.shape, device=device, dtype=dtype)
        t = torch.rand(B, device=device) * (1.0 - self.sigma_min) + self.sigma_min
        x_t = (1 - t.view(B, 1, 1)) * noise + t.view(B, 1, 1) * trajectory

        if self.k_stab > 0:
            stab = -self.k_stab * (x_t - trajectory)
            target_v = (trajectory - noise) + stab
        else:
            target_v = trajectory - noise

        t_input = (t * self.fm_t_scale).float()
        assert t_input.shape == (B,), f"t_input shape must be (B,), got {t_input.shape}"

        cond_data = torch.zeros(B, T, Da, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        x_t[cond_mask] = cond_data[cond_mask]

        pred_v = self.model(sample=x_t, timestep=t_input,
                            local_cond=None, global_cond=global_cond)

        loss_mask = ~cond_mask
        diff_loss_per_sample = F.mse_loss(pred_v, target_v, reduction='none')
        diff_loss_per_sample = (diff_loss_per_sample * loss_mask.float()).mean(dim=[1, 2])  # (B,)

        # ── α2: SAFE-GIL 式样本重加权 ──
        sample_reweight_diag = {}
        if self.use_sample_reweight and min_dist is not None and self.training:
            w = 1.0 + self.reweight_alpha * torch.exp(-min_dist / self.reweight_tau)
            w = w / w.mean().clamp(min=1e-8)  # 归一化
            diff_loss = (diff_loss_per_sample * w).mean()
            sample_reweight_diag = {
                'reweight_w_mean': w.mean().item(),
                'reweight_w_max': w.max().item(),
                'reweight_w_min': w.min().item(),
            }
        else:
            diff_loss = diff_loss_per_sample.mean()

        anchor_loss = torch.tensor(0.0, device=device)
        anchor_acc = 0.0
        if self.anchor_head is not None and self.training_phase != 'unet_warmup':
            logits = self.anchor_head(global_cond)
            labels = self._get_anchor_labels(nactions)
            if labels is not None:
                anchor_loss = F.cross_entropy(logits, labels)
                anchor_acc = (logits.argmax(1) == labels).float().mean().item()

        # ── Collision loss (graded) ──
        collision_loss = torch.tensor(0.0, device=device)
        collision_diag = {}

        _collision_active = (self.use_collision_loss and self.training
                             and self.training_phase == 'joint')
        _jerk_active = (self.lambda_jerk > 0 and self.training
                        and self.training_phase == 'joint')
        need_x0 = _collision_active or _jerk_active

        pred_x0_action = None
        if need_x0:
            pred_x0_norm = x_t + (1 - t.view(B, 1, 1)) * pred_v
            pred_x0_action = self.normalizer['action'].unnormalize(pred_x0_norm)

        # ── Jerk 正则 ──
        jerk_loss = torch.tensor(0.0, device=device)
        if _jerk_active:
            dims = self.jerk_smooth_dims
            pos_dims = [d for d in dims if d in (0, 1, 2)]
            yaw_dims = [d for d in dims if d == 9]
            jerk_pos = torch.tensor(0.0, device=device)
            jerk_yaw = torch.tensor(0.0, device=device)
            if pos_dims:
                a = pred_x0_action[:, :, pos_dims]
                d1 = a[:, 1:] - a[:, :-1]
                d2 = d1[:, 1:] - d1[:, :-1]
                jerk_pos = (d2 ** 2).mean()
            if yaw_dims:
                ay = pred_x0_action[:, :, yaw_dims]
                d1y = ay[:, 1:] - ay[:, :-1]
                d2y = d1y[:, 1:] - d1y[:, :-1]
                jerk_yaw = (d2y ** 2).mean()
            jerk_loss = jerk_pos + self.lambda_yaw_jerk * jerk_yaw

        if _collision_active:
            cur_pos = obs_dict['agent_pos'][:, -1, 0:3].to(device)
            point_cloud = obs_dict['point_cloud'][:, -1].to(device)
            sin_yaw = obs_dict['agent_pos'][:, -1, 7].to(device)
            cos_yaw = obs_dict['agent_pos'][:, -1, 8].to(device)

            # ── α Finetune: 三级分级 collision loss (z-downweight + linear hinge) ──
            collision_loss, collision_diag = collision_auxiliary_loss_graded(
                pred_x0_action=pred_x0_action,
                point_cloud=point_cloud,
                cur_pos=cur_pos,
                sin_yaw=sin_yaw,
                cos_yaw=cos_yaw,
                eps_levels=self.collision_eps_levels,
                weights=self.collision_weights,
                hinge_types=self.collision_hinge_types,      # [Task 2]
                beta=self.collision_beta,
                tau_softmin=self.collision_tau_softmin,
                k_nn=self.collision_k_nn,
                horizon_decay_tau=self.collision_horizon_decay_tau,
                flow_t=t,
                lambda_t_schedule_type=self.collision_lambda_t_schedule,
                z_weight=self.collision_z_weight,             # [Task 1]
                return_diagnostics=True,
            )

        # ── 有效 collision_lambda (含 warmup) ──
        effective_lambda = self._get_effective_collision_lambda() if _collision_active else self.collision_lambda

        total = (diff_loss
                 + self.anchor_loss_weight * anchor_loss
                 + effective_lambda * collision_loss
                 + self.lambda_jerk * jerk_loss)

        log_dict = {
            'bc_loss': diff_loss.item(),
            'anchor_loss': anchor_loss.item(),
            'anchor_acc': anchor_acc,
            'collision_loss': collision_loss.item(),
            'jerk_loss': jerk_loss.item(),
            'total_loss': total.item(),
            'collision_lambda_eff': effective_lambda if _collision_active else 0.0,
        }
        # α2 reweight 诊断
        for k, v in sample_reweight_diag.items():
            log_dict[f'reweight_{k}'] = v
        # jerk 诊断
        if self.lambda_jerk > 0 and self.training and self.training_phase == 'joint':
            log_dict['jerk_active'] = 1.0
        # collision 诊断
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
        cprint(f'[Phase1a-V9] phase={phase}', 'cyan')

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
