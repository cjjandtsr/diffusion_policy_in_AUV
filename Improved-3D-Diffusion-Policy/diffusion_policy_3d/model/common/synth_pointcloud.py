"""
synth_pointcloud.py — CUDA 光线投射合成点云 (Phase β 阶段 0-B 核心)
====================================================================
从障碍几何 (.obstacles.json) + 无人机位姿合成 body-frame (4096,3) 点云。
走 CUDA 计算路径 (PyTorch tensor ops on GPU)，绕过容器无 /dev/dri 的限制。

对照 Implementation Guide v1.0 任务 1+2。

用法:
    from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
    synth = SynthPointCloud(obstacles_json_path, device='cuda:0')
    pc_body = synth.synthesize(drone_pos_world, drone_yaw)  # (4096, 3) float32
"""

import numpy as np
import torch
import torch.nn.functional as F
import json, math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
# Camera intrinsics — 从 D435i SDF 提取, 必须与真实 depth camera 一致
# ═══════════════════════════════════════════════════════════════

@dataclass
class DepthCameraIntrinsics:
    """RealSense D435i depth camera 内参 (匹配 Gazebo D435i.sdf depth sensor)"""
    width: int = 320
    height: int = 240
    hfov: float = 1.047198   # 60 degrees in radians

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / np.tan(self.hfov / 2.0)

    @property
    def fy(self) -> float:
        return self.fx  # square pixels

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def near(self) -> float:
        return 0.05   # 匹配 D435i.sdf depth clip near

    @property
    def far(self) -> float:
        return 10.0   # 匹配 D435i.sdf depth clip far / pointCloudCutoffMax


# Camera extrinsics (mount pose relative to base_link)
# iris_D435i.sdf: <pose>0.12 0 0 0 0 0</pose>
# D435i.sdf:    <pose>0 0 0.015 0 0 0</pose>
CAM_MOUNT_X = 0.12    # forward offset from body center
CAM_MOUNT_Y = 0.0
CAM_MOUNT_Z = 0.015   # vertical offset from body bottom

# Camera pitch: nose-down angle to bring near-field ground into FOV
# Real D435i mounted with slight downward tilt; without pitch, ground only
# visible from ~3.5m forward (at 2m cruise height). 8° brings closest ground
# to ~1.5m, matching real point cloud statistics.
DEFAULT_CAM_PITCH_DEG = 8.0  # nose-down degrees (positive = nose down)


# ═══════════════════════════════════════════════════════════════
# Ray-geometry intersection (analytical, batched on GPU)
# ═══════════════════════════════════════════════════════════════

@torch.jit.script
def intersect_rays_cylinder(
    ray_origin: torch.Tensor,   # (N_rays, 3)
    ray_dir: torch.Tensor,      # (N_rays, 3)
    cyl_center: torch.Tensor,   # (N_cyl, 2)  — xy only
    cyl_radius: torch.Tensor,   # (N_cyl,)
    cyl_z_min: torch.Tensor,    # (N_cyl,)
    cyl_z_max: torch.Tensor,    # (N_cyl,)
) -> torch.Tensor:
    """解析求解射线-圆柱交点 (无限高圆柱 + z 裁剪)

    返回: (N_rays, N_cyl)  hit distance, inf if miss

    公式: |ray_origin + t*ray_dir - cyl_center|² = r²
    解一元二次方程 at² + bt + c = 0
    其中 a = dx²+dy², b = 2*(dx*ox + dy*oy), c = ox²+oy² - r²
    """
    N_rays = ray_origin.shape[0]
    N_cyl = cyl_center.shape[0]
    device = ray_origin.device

    # 只解 xy 平面投影 (圆柱无限高)
    ox = ray_origin[:, 0:1] - cyl_center[:, 0].unsqueeze(0)  # (N_rays, N_cyl)
    oy = ray_origin[:, 1:2] - cyl_center[:, 1].unsqueeze(0)
    dx = ray_dir[:, 0:1].expand(-1, N_cyl)
    dy = ray_dir[:, 1:2].expand(-1, N_cyl)
    dz = ray_dir[:, 2:3].expand(-1, N_cyl)
    oz = ray_origin[:, 2:3].expand(-1, N_cyl)

    r = cyl_radius.unsqueeze(0)  # (1, N_cyl)

    a = dx*dx + dy*dy
    b = 2.0 * (dx*ox + dy*oy)
    c = ox*ox + oy*oy - r*r

    disc = b*b - 4.0*a*c
    mask = disc >= 0.0

    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sqrt_disc) / (2.0 * a + 1e-12)
    t2 = (-b + sqrt_disc) / (2.0 * a + 1e-12)

    # 取最小正解
    t_hit = torch.where(
        (t1 > 1e-6) & mask, t1,
        torch.where((t2 > 1e-6) & mask, t2,
                     torch.full_like(t1, float('inf')))
    )

    # z 裁剪: 交点必须在 cyl_z_min 和 cyl_z_max 之间
    z_hit = oz + t_hit * dz
    z_min = cyl_z_min.unsqueeze(0)
    z_max = cyl_z_max.unsqueeze(0)
    z_valid = (z_hit >= z_min) & (z_hit <= z_max)

    return torch.where(z_valid & mask, t_hit, torch.full_like(t_hit, float('inf')))


@torch.jit.script
def intersect_rays_box(
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    box_center: torch.Tensor,   # (N_box, 3)
    box_half_extents: torch.Tensor,  # (N_box, 3) — size_x/2, size_y/2, size_z/2
    box_yaws: torch.Tensor,     # (N_box,)
) -> torch.Tensor:
    """射线-AABB 求交 (OBB via rotation then AABB slab test)

    返回: (N_rays, N_box) hit distance, inf if miss
    """
    N_rays = ray_origin.shape[0]
    N_box = box_center.shape[0]
    device = ray_origin.device

    # 将射线转到 box local frame (反向旋转 yaw)
    cos_y = torch.cos(-box_yaws)  # (N_box,)
    sin_y = torch.sin(-box_yaws)

    # 对每条射线、每个 box: 平移 → 旋转
    ox = ray_origin[:, 0:1] - box_center[:, 0].unsqueeze(0)  # (N_rays, N_box)
    oy = ray_origin[:, 1:2] - box_center[:, 1].unsqueeze(0)
    oz = ray_origin[:, 2:3] - box_center[:, 2].unsqueeze(0)

    # 旋转 ray origin 到 box local frame
    ol_x = ox * cos_y.unsqueeze(0) + oy * sin_y.unsqueeze(0)
    ol_y = -ox * sin_y.unsqueeze(0) + oy * cos_y.unsqueeze(0)
    ol_z = oz

    dx = ray_dir[:, 0:1].expand(-1, N_box)
    dy = ray_dir[:, 1:2].expand(-1, N_box)
    dz = ray_dir[:, 2:3].expand(-1, N_box)

    dl_x = dx * cos_y.unsqueeze(0) + dy * sin_y.unsqueeze(0)
    dl_y = -dx * sin_y.unsqueeze(0) + dy * cos_y.unsqueeze(0)
    dl_z = dz

    he = box_half_extents  # (N_box, 3)

    # Slab test (AABB)
    inv_dx = 1.0 / (dl_x + 1e-12)
    inv_dy = 1.0 / (dl_y + 1e-12)
    inv_dz = 1.0 / (dl_z + 1e-12)

    tx1 = (-he[:, 0].unsqueeze(0) - ol_x) * inv_dx
    tx2 = (he[:, 0].unsqueeze(0) - ol_x) * inv_dx
    tmin_x = torch.min(tx1, tx2)
    tmax_x = torch.max(tx1, tx2)

    ty1 = (-he[:, 1].unsqueeze(0) - ol_y) * inv_dy
    ty2 = (he[:, 1].unsqueeze(0) - ol_y) * inv_dy
    tmin_y = torch.min(ty1, ty2)
    tmax_y = torch.max(ty1, ty2)

    tz1 = (-he[:, 2].unsqueeze(0) - ol_z) * inv_dz
    tz2 = (he[:, 2].unsqueeze(0) - ol_z) * inv_dz
    tmin_z = torch.min(tz1, tz2)
    tmax_z = torch.max(tz1, tz2)

    t_enter = torch.max(torch.max(tmin_x, tmin_y), tmin_z)
    t_exit = torch.min(torch.min(tmax_x, tmax_y), tmax_z)

    hit = (t_enter <= t_exit) & (t_exit >= 0.0) & (t_enter <= 1e-6)  # t_enter can be slightly negative
    # actually we want positive t
    hit = (t_enter <= t_exit) & (t_exit >= 1e-6)

    t_hit = torch.where(hit & (t_enter >= 1e-6), t_enter,
                        torch.where(hit & (t_exit >= 1e-6), t_exit,
                                    torch.full_like(t_enter, float('inf'))))

    return t_hit


@torch.jit.script
def intersect_rays_ground(
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    ground_z: float = 0.0,
) -> torch.Tensor:
    """射线-地平面求交 (z = ground_z 的无限平面)

    返回: (N_rays,) hit distance, inf if miss or looking away

    公式: t = (ground_z - ray_origin_z) / ray_dir_z
    """
    dz = ray_dir[:, 2]
    oz = ray_origin[:, 2]

    # 射线朝下且 origin 在地面之上 才可能 hit
    t = (ground_z - oz) / (dz + 1e-12)
    hit = (t > 1e-6) & (dz < 0.0) & (oz > ground_z)

    return torch.where(hit, t, torch.full_like(t, float('inf')))


# ═══════════════════════════════════════════════════════════════
# Main synthesis class
# ═══════════════════════════════════════════════════════════════

class SynthPointCloud:
    """CUDA 光线投射合成 body-frame 点云

    Args:
        obstacles_json: .obstacles.json 路径 (含圆柱/方块几何)
        device: 'cuda:0' | 'cpu'
        cam_intrinsics: depth camera 内参
        noise_cfg: 噪声配置 (None=不加噪声)
    """

    def __init__(
        self,
        obstacles_json: str,
        device: str = 'cuda:0',
        cam_intrinsics: Optional[DepthCameraIntrinsics] = None,
        noise_cfg: Optional[Dict] = None,
    ):
        self.device = torch.device(device)
        self.cam = cam_intrinsics or DepthCameraIntrinsics()

        # 加载障碍几何
        with open(obstacles_json) as f:
            data = json.load(f)
        self.obstacles_data = data
        self._build_geometry(data)

        # 噪声配置
        self.noise_cfg = noise_cfg or {
            'depth_noise_sigma': 0.005,   # 乘性噪声 std (0.5%)
            'dropout_rate': 0.05,         # 随机点缺失率
            'ground_ratio_min': 0.20,     # domain rand: 地面点占比下界
            'ground_ratio_max': 0.70,     # domain rand: 地面点占比上界
        }

        # 预计算射线方向 (camera local frame), 在 _build_rays() 中填充
        self._ray_dirs_cam = None  # (H*W, 3) normalized, in camera frame

    def _build_geometry(self, data: dict):
        """从 .obstacles.json 构建场景几何"""
        obstacles = data.get('obstacles', [])

        cyl_centers, cyl_radii, cyl_z_mins, cyl_z_maxs = [], [], [], []
        box_centers, box_half_exts, box_yaws = [], [], []

        for obs in obstacles:
            shape = obs.get('shape', 'cylinder')
            if shape == 'cylinder':
                cyl_centers.append([obs['x'], obs['y']])
                cyl_radii.append(obs['radius'])
                # Cylinder sits on ground, height from z=0
                cyl_z_mins.append(0.0)
                cyl_z_maxs.append(obs['height'])
            elif shape == 'box':
                box_centers.append([obs['x'], obs['y'], obs.get('size_z', 4.0) / 2.0])
                box_half_exts.append([
                    obs.get('size_x', 2.0) / 2.0,
                    obs.get('size_y', 2.0) / 2.0,
                    obs.get('size_z', 4.0) / 2.0,
                ])
                box_yaws.append(obs.get('yaw', 0.0))

        self.n_cylinders = len(cyl_centers)
        self.n_boxes = len(box_centers)

        if self.n_cylinders > 0:
            self.cyl_center = torch.tensor(cyl_centers, device=self.device, dtype=torch.float32)
            self.cyl_radius = torch.tensor(cyl_radii, device=self.device, dtype=torch.float32)
            self.cyl_z_min = torch.tensor(cyl_z_mins, device=self.device, dtype=torch.float32)
            self.cyl_z_max = torch.tensor(cyl_z_maxs, device=self.device, dtype=torch.float32)
        else:
            self.cyl_center = torch.zeros((0, 2), device=self.device)
            self.cyl_radius = torch.zeros(0, device=self.device)
            self.cyl_z_min = torch.zeros(0, device=self.device)
            self.cyl_z_max = torch.zeros(0, device=self.device)

        if self.n_boxes > 0:
            self.box_center = torch.tensor(box_centers, device=self.device, dtype=torch.float32)
            self.box_half_ext = torch.tensor(box_half_exts, device=self.device, dtype=torch.float32)
            self.box_yaw = torch.tensor(box_yaws, device=self.device, dtype=torch.float32)
        else:
            self.box_center = torch.zeros((0, 3), device=self.device)
            self.box_half_ext = torch.zeros((0, 3), device=self.device)
            self.box_yaw = torch.zeros(0, device=self.device)

    def _build_rays(self, device=None):
        """按 depth camera 内参生成归一化射线方向 (camera frame)

        射线: d_cam = normalize([(u-cx)/fx, (v-cy)/fy, 1.0])
        所有射线起点在 camera origin (0,0,0)
        """
        if device is None:
            device = self.device
        if self._ray_dirs_cam is not None and self._ray_dirs_cam.device == device:
            return self._ray_dirs_cam

        W, H = self.cam.width, self.cam.height
        fx, fy = self.cam.fx, self.cam.fy
        cx, cy = self.cam.cx, self.cam.cy

        u = torch.arange(W, device=device, dtype=torch.float32)  # (W,)
        v = torch.arange(H, device=device, dtype=torch.float32)  # (H,)
        uu, vv = torch.meshgrid(u, v, indexing='xy')  # (W, H)
        uu = uu.T.reshape(-1)  # (H*W,)
        vv = vv.T.reshape(-1)

        dx = (uu - cx) / fx
        dy = (vv - cy) / fy
        dz = torch.ones_like(dx)

        dirs = torch.stack([dx, dy, dz], dim=-1)  # (H*W, 3)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

        self._ray_dirs_cam = dirs
        return dirs

    def _cam_to_world_pose(self, drone_pos_world, drone_yaw, drone_pitch=0.0):
        """计算 camera 在 world frame 的位姿 + camera→world 旋转矩阵

        坐标系约定 (与项目 body-frame 定义一致):
          body  frame: +x=forward, +y=left, +z=up
          camera frame: +x=right, +y=down, +z=forward (OpenCV/ROS convention)

        camera→body 映射:
          cam_x(right)  → body_y(neg) → body_y = -cam_x
          cam_y(down)   → body_z(neg) → body_z = -cam_y
          cam_z(forward)→ body_x(pos) → body_x =  cam_z

        body→world: R(yaw) rotation around z-axis + pos_world

        pitch: nose-down rotation around body y-axis (positive = nose down).
          Applied between R_cam2body and R_body2world, so pitch rotates the
          camera's forward axis toward the ground in body frame.

        camera 相对 body 的 offset: (CAM_MOUNT_X, 0, CAM_MOUNT_Z) in body frame
        """
        cos_y = np.cos(drone_yaw)
        sin_y = np.sin(drone_yaw)
        cos_p = np.cos(drone_pitch)
        sin_p = np.sin(drone_pitch)

        # camera 在 body frame 的位置
        cam_body = np.array([CAM_MOUNT_X, CAM_MOUNT_Y, CAM_MOUNT_Z])

        # body→world 旋转 (仅 yaw)
        R_body2world = np.array([[cos_y, -sin_y, 0],
                                  [sin_y,  cos_y, 0],
                                  [0,      0,     1]])

        # camera→body 旋转 (camera +z → body +x, camera +x → body -y, camera -y → body +z)
        R_cam2body = np.array([[ 0,  0,  1],   # cam_z → body_x (forward)
                                [-1,  0,  0],   # cam_x → body_y (left, negated)
                                [ 0, -1,  0]])  # cam_y → body_z (up, negated)

        # pitch: nose-down rotation around body y-axis
        #   body_x' =  cos(p)*body_x + sin(p)*body_z  (forward tips down)
        #   body_y' =  body_y
        #   body_z' = -sin(p)*body_x + cos(p)*body_z  (up tilts forward)
        R_pitch = np.array([[ cos_p, 0, sin_p],
                             [     0, 1,     0],
                             [-sin_p, 0, cos_p]])

        # camera→world = body→world @ pitch @ camera→body
        R_cam2world = R_body2world @ R_pitch @ R_cam2body

        # camera world position = drone_pos + R_body2world @ cam_body
        cam_world = drone_pos_world + R_body2world @ cam_body

        return torch.tensor(cam_world, device=self.device, dtype=torch.float32), \
               torch.tensor(R_cam2world, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def _generate_with_stats(
        self,
        drone_pos_world: np.ndarray,   # (3,)  world frame
        drone_yaw: float,               # radians
        num_points: int = 4096,
        add_noise: bool = True,
        ground_z: float = 0.0,
        drone_pitch: float = None,      # radians, nose-down positive. None → DEFAULT_CAM_PITCH_DEG
        rng_seed: Optional[int] = None,  # P0-2: episode-local RNG seed for reproducibility
    ):
        """合成一帧 (num_points, 3) body-frame 点云

        Args:
            drone_pos_world: 无人机当前位置 (world frame, x,y,z)
            drone_yaw: 当前偏航角 (rad)
            num_points: 输出点数 (默认 4096)
            add_noise: 是否加传感器噪声
            ground_z: 地面高度 (world frame)
            drone_pitch: 相机俯仰角 (rad, nose-down positive). None 使用默认 8°

        Returns:
            (num_points, 3) float32 body-frame point cloud
        """
        # ── P0-2: Episode-local RNG ──
        if rng_seed is not None:
            _torch_gen = torch.Generator(device=self.device).manual_seed(rng_seed & 0xFFFFFFFF)
            _np_rng = np.random.default_rng(rng_seed & 0xFFFFFFFF)
        else:
            _torch_gen = None
            _np_rng = np.random.default_rng()

        if drone_pitch is None:
            drone_pitch = math.radians(DEFAULT_CAM_PITCH_DEG)

        # Step 1: 生成射线 (camera frame)
        ray_dirs_cam = self._build_rays(self.device)  # (H*W, 3)
        N_rays = ray_dirs_cam.shape[0]

        # Step 2: camera → world transform
        cam_pos_world, R_cam2world = self._cam_to_world_pose(drone_pos_world, drone_yaw, drone_pitch)

        # 射线转 world frame
        ray_origin_world = cam_pos_world.unsqueeze(0).expand(N_rays, -1)  # (N_rays, 3)
        ray_dir_world = (R_cam2world @ ray_dirs_cam.T).T  # (N_rays, 3)

        # Step 3: 射线-几何求交 (取每个 ray 的最近 hit)
        t_min = torch.full((N_rays,), float('inf'), device=self.device)
        hit_points = torch.zeros((N_rays, 3), device=self.device)

        # 3a. 圆柱
        if self.n_cylinders > 0:
            t_cyl = intersect_rays_cylinder(
                ray_origin_world, ray_dir_world,
                self.cyl_center, self.cyl_radius, self.cyl_z_min, self.cyl_z_max
            )  # (N_rays, N_cyl)
            t_cyl_min, _ = t_cyl.min(dim=1)  # (N_rays,)
            update = t_cyl_min < t_min
            t_min = torch.where(update, t_cyl_min, t_min)

        # 3b. 方块
        if self.n_boxes > 0:
            t_box = intersect_rays_box(
                ray_origin_world, ray_dir_world,
                self.box_center, self.box_half_ext, self.box_yaw
            )
            t_box_min, _ = t_box.min(dim=1)
            update = t_box_min < t_min
            t_min = torch.where(update, t_box_min, t_min)

        # 3c. 地面平面 (ground_z=0)
        t_ground = intersect_rays_ground(ray_origin_world, ray_dir_world, ground_z)
        update = t_ground < t_min
        t_min = torch.where(update, t_ground, t_min)

        # 计算 hit points (world frame)
        hit_mask = torch.isfinite(t_min) & (t_min > 1e-6)
        hit_points = ray_origin_world + t_min.unsqueeze(-1) * ray_dir_world

        # 只保留有效 hit (在深度 camera 的 near/far 范围内)
        hit_mask = hit_mask & (t_min >= self.cam.near) & (t_min <= self.cam.far)
        valid_hit_rays = int(hit_mask.sum().item())
        hit_points = hit_points[hit_mask]  # (N_hit, 3)

        if hit_points.shape[0] == 0:
            # 合法量程外无返回：例如固定 8° 下俯、10 m 量程的 D435i 在
            # 高度约 5.2 m 以上看不到地面。用各相机射线的 far-plane
            # 截止点表达“量程内自由”，不能把所有点堆在 [far,far,far]
            # 形成一个虚假的远处点簇。metadata 仍保留真实 hit_count=0。
            import warnings
            warnings.warn(
                f"synth: no ray hit (pos={drone_pos_world}, yaw={drone_yaw:.2f}); "
                f"using max-range censoring", RuntimeWarning)
            far_world = ray_origin_world + float(self.cam.far) * ray_dir_world
            cos_y = np.cos(drone_yaw)
            sin_y = np.sin(drone_yaw)
            R_world2body = torch.tensor(
                [[cos_y, sin_y, 0], [-sin_y, cos_y, 0], [0, 0, 1]],
                device=self.device, dtype=torch.float32,
            )
            body_origin = torch.tensor(
                drone_pos_world, device=self.device, dtype=torch.float32
            )
            far_pts = (R_world2body @ (far_world - body_origin).T).T
            if far_pts.shape[0] > num_points:
                idx = torch.randperm(
                    far_pts.shape[0], device=self.device, generator=_torch_gen
                )[:num_points]
                far_pts = far_pts[idx]
            elif far_pts.shape[0] < num_points:
                repeats = (num_points + far_pts.shape[0] - 1) // far_pts.shape[0]
                far_pts = far_pts.repeat((repeats, 1))[:num_points]
            return (
                far_pts.cpu().numpy().astype(np.float32),
                int(N_rays),
                0,
                True,
            )

        # Step 4: world → body frame
        # R_body2world = R(yaw), body_origin = drone_pos_world
        cos_y = np.cos(drone_yaw)
        sin_y = np.sin(drone_yaw)
        R_world2body = torch.tensor(
            [[cos_y,  sin_y, 0],
             [-sin_y, cos_y, 0],
             [0,      0,     1]],
            device=self.device, dtype=torch.float32
        )
        body_origin = torch.tensor(drone_pos_world, device=self.device, dtype=torch.float32)
        pc_body = (R_world2body @ (hit_points - body_origin.unsqueeze(0)).T).T

        # Step 5: 下采样到 num_points —— uniform random, 与真实管线一致
        # (ml_planner_gazebo.py L988 / bag2zarr_body_v5.py L105: np.random.choice;
        #  iDP3 encoder 内部不采样, 裁决表 #4。uniform random 自然保留地面点占比,
        #  不像 FPS/voxel 会重分布 —— 对维持 48-76% 地面占比有利)
        if pc_body.shape[0] > num_points:
            idx = torch.randperm(pc_body.shape[0], device=self.device,
                                 generator=_torch_gen)[:num_points]
            pc_body = pc_body[idx]
        elif pc_body.shape[0] < num_points:
            # 不足时 tile 重复填充 (与真实管线 <4096 填充一致)
            n_extra = num_points - pc_body.shape[0]
            idx = torch.randint(0, pc_body.shape[0], (n_extra,), device=self.device,
                                generator=_torch_gen)
            pc_body = torch.cat([pc_body, pc_body[idx]], dim=0)

        # Step 6: 噪声 (可选)
        if add_noise:
            pc_body = self._apply_noise(pc_body, torch_gen=_torch_gen, np_rng=_np_rng)

        return (
            pc_body.cpu().numpy().astype(np.float32),
            int(N_rays),
            valid_hit_rays,
            False,
        )

    def synthesize(
        self,
        drone_pos_world: np.ndarray,
        drone_yaw: float,
        num_points: int = 4096,
        add_noise: bool = True,
        ground_z: float = 0.0,
        drone_pitch: float = None,
        rng_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Compatibility wrapper; formal RL uses generate_with_metadata."""

        points, _, _, _ = self._generate_with_stats(
            drone_pos_world,
            drone_yaw,
            num_points=num_points,
            add_noise=add_noise,
            ground_z=ground_z,
            drone_pitch=drone_pitch,
            rng_seed=rng_seed,
        )
        return points

    def generate_with_metadata(
        self,
        drone_pos_world: np.ndarray,
        drone_yaw: float,
        num_points: int = 4096,
        add_noise: bool = True,
        ground_z: float = 0.0,
        drone_pitch: float = None,
        rng_seed: Optional[int] = None,
        *,
        episode_identity_hash: str,
        decision_index: int,
    ):
        """Generate bytes and immutable ray-quality metadata in one call."""

        from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import (
            PointCloudGenerationMeta,
            canonical_array_hash,
        )

        points, total_rays, valid_hit_rays, fallback = self._generate_with_stats(
            drone_pos_world,
            drone_yaw,
            num_points=num_points,
            add_noise=add_noise,
            ground_z=ground_z,
            drone_pitch=drone_pitch,
            rng_seed=rng_seed,
        )
        meta = PointCloudGenerationMeta(
            total_rays=total_rays,
            valid_hit_rays=valid_hit_rays,
            no_ray_hit_fraction=1.0 - valid_hit_rays / total_rays,
            used_far_plane_fallback=fallback,
            finite=bool(np.isfinite(points).all()),
            episode_identity_hash=episode_identity_hash,
            decision_index=decision_index,
            raw_array_hash=canonical_array_hash(points),
        )
        return points, meta

    def _apply_noise(self, pc: torch.Tensor,
                     torch_gen: Optional[torch.Generator] = None,
                     np_rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        """施加传感器噪声 + 随机点缺失 + domain randomization

        缺失语义修正 (P0-2): 真实传感器缺失 = 该射线无返回 = 点不存在。
        因此缺失点必须真删除, 而非搬到原点 (旧实现 pc[~keep]*=0.01 会造一簇
        (0,0,0) 幽灵点, 被 PointNet 误判为机体中心有障碍)。
        删除后点数不足 num_points 时, 用 tile 重复有效点补足 ——
        与真实管线 (ml_planner_gazebo.py: <4096 tile 重复填充) 完全一致。
        """
        cfg = self.noise_cfg
        N = pc.shape[0]
        if N == 0:
            return pc
        num_points = N  # 维持输出点数与输入一致 (上游已下采样到 num_points)

        # 乘性深度噪声 (作用于点坐标, 等价于沿视线方向的深度扰动近似)
        if cfg.get('depth_noise_sigma', 0) > 0:
            sigma = cfg['depth_noise_sigma']
            noise = torch.randn(N, device=self.device, generator=torch_gen) * sigma
            pc = pc * (1.0 + noise.unsqueeze(-1))

        # 随机点缺失: 真删除缺失点, 再 tile 补足到 num_points
        dropout_rate = cfg.get('dropout_rate', 0.0)
        if dropout_rate > 0:
            if cfg.get('domain_rand', False):
                # P0-2: 使用 episode-local RNG 替代全局 np.random.uniform
                if np_rng is not None:
                    dropout_rate = float(np_rng.uniform(0.0, dropout_rate * 2))
                else:
                    dropout_rate = float(np.random.uniform(0.0, dropout_rate * 2))
            keep = torch.rand(N, device=self.device, generator=torch_gen) > dropout_rate
            pc_kept = pc[keep]  # 真删除
            n_kept = pc_kept.shape[0]
            if n_kept == 0:
                # 极端: 全部缺失, 退回原始 pc (不引入假点)
                return pc
            if n_kept < num_points:
                # tile 重复有效点补足 (与真实管线 <4096 填充一致)
                idx = torch.randint(0, n_kept, (num_points - n_kept,), device=self.device,
                                    generator=torch_gen)
                pc = torch.cat([pc_kept, pc_kept[idx]], dim=0)
            else:
                pc = pc_kept

        return pc

    def set_obstacles(self, obstacles_json: str):
        """热切换障碍配置 (用于不同 test world)"""
        with open(obstacles_json) as f:
            data = json.load(f)
        self.obstacles_data = data
        self._build_geometry(data)
