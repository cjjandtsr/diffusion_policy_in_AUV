#!/usr/bin/env python3
"""
test_kinematic.py — 运动学闭环环境单元测试
用法: python -m diffusion_policy_3d.env.test_kinematic [--real-synth]
  无参数: MockSynthPC (快速验证逻辑, 无需 GPU/障碍文件)
  --real-synth: 真实 SynthPointCloud (服务器上需 GPU + obstacles.json)
"""

import sys, os, math
import numpy as np

# 加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diffusion_policy_3d.env.kinematic_uav_env import (
    KinematicUAVEnv, WORLD_ENV_MAP, WORLDS_DIR, DEFAULT_GOAL, SPAWN_OFFSET,
)
from diffusion_policy_3d.env.traj_utils import action_to_world, normalize_angle, POS, VEL, YAW
from diffusion_policy_3d.env.collision_geom import (
    ObstacleSpec, point_to_oriented_rect_distance, compute_obstacle_clearance, load_obstacles,
)


# ═══════════════════════════════════════════════════════════════
# MockSynthPC — 快速验证逻辑 (无需 GPU)
# ═══════════════════════════════════════════════════════════════

class MockSynthPC:
    """生成随机点云 (不依赖 GPU/ROS)."""
    def __init__(self, obstacles_json=None, device='cpu'):
        self.obstacles_json = obstacles_json

    def synthesize(self, drone_pos, drone_yaw, num_points=4096, add_noise=True):
        rng = np.random.default_rng(42)
        # 前方 8m 半球 (模拟障碍感知)
        c, s = np.cos(drone_yaw), np.sin(drone_yaw)
        r = rng.uniform(0.5, 8.0, num_points)
        theta = rng.uniform(-np.pi / 2, np.pi / 2, num_points)
        phi = rng.uniform(-np.pi / 4, np.pi / 4, num_points)
        # body frame → world
        bx = r * np.cos(theta) * np.cos(phi)
        by = r * np.cos(theta) * np.sin(phi)
        bz = r * np.sin(phi)
        wx = c * bx - s * by + drone_pos[0]
        wy = s * bx + c * by + drone_pos[1]
        wz = bz + drone_pos[2]
        return np.stack([wx, wy, wz], axis=-1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# 测试 1: 几何碰撞 (纯数学, 无 GPU 依赖)
# ═══════════════════════════════════════════════════════════════

def test_geometry():
    """纯几何函数: 侧距/穿透/飞越豁免."""
    print("=== Test 1: 几何碰撞 ===")

    # 造一个圆柱障碍
    obs = ObstacleSpec(name="test_cyl", shape="cylinder",
                       x=10.0, y=0.0, z=1.5, radius=0.5, height=3.0)

    # 1a) 无人机在侧方 2m (无障碍)
    clr, name, flyover = compute_obstacle_clearance(8.0, 0.0, 2.0, [obs])
    print(f"  侧距2m: clr={clr:.3f}, flyover={flyover}  (expect ~1.5)")
    assert clr > 1.0, f"Expected clearance >1.0, got {clr:.3f}"

    # 1b) 穿透 0.5m (无人机在障碍内)
    clr, name, flyover = compute_obstacle_clearance(10.0, 0.0, 1.5, [obs])
    print(f"  穿透: clr={clr:.3f}  (expect < 0)")
    assert clr < 0.0, f"Expected negative clearance, got {clr:.3f}"

    # 1c) 飞越豁免: 无人机在障碍正上方 (底部 > 障碍顶部 + 0.05)
    obs_top = 1.5 + 3.0 / 2.0  # 3.0
    drone_z = obs_top + 0.2     # 3.2, 底部=3.1 > 3.0-0.05 → flyover
    clr, name, flyover = compute_obstacle_clearance(10.0, 0.0, drone_z, [obs])
    print(f"  飞越: clr={clr:.3f}, flyover={flyover}  (expect flyover=True)")
    assert flyover, f"Expected flyover=True, got {flyover}"

    # 1d) box 障碍
    obs_box = ObstacleSpec(name="test_box", shape="box",
                           x=5.0, y=0.0, z=1.0, yaw=0.0,
                           size_x=2.0, size_y=1.0, size_z=3.0)
    # 无人机在前方 2m (无障碍)
    clr, _, _ = compute_obstacle_clearance(2.0, 0.0, 1.0, [obs_box])
    print(f"  box侧距: clr={clr:.3f}  (expect > 0)")
    assert clr > 0.0, f"Expected positive clearance, got {clr:.3f}"

    print("  [PASS] test_geometry\n")


# ═══════════════════════════════════════════════════════════════
# 测试 2: 坐标系 (pos/goal/yaw)
# ═══════════════════════════════════════════════════════════════

def test_coordinate():
    """坐标系: pos(8,0,2)/goal(16,0,2) → dist=8, yaw=0 时 goal 在前方."""
    print("=== Test 2: 坐标系 ===")

    pos = np.array([8.0, 0.0, 2.0], dtype=np.float32)
    goal = np.array([16.0, 0.0, 2.0], dtype=np.float32)
    yaw = 0.0
    s9 = KinematicUAVEnv._build_state(pos, yaw, goal)

    # s9 = [pos_w(3), goal_body(3), dist, sin_yaw, cos_yaw]
    dist = np.linalg.norm(goal - pos)
    print(f"  pos={pos}, goal={goal}")
    print(f"  s9={s9}")
    print(f"  dist={s9[6]:.3f} (expect {dist:.3f})")
    assert abs(s9[6] - dist) < 0.01, f"dist mismatch: {s9[6]} vs {dist}"
    # goal_body: yaw=0 → goal 在正前方 (body x = dx, body y = dy)
    assert abs(s9[3] - 8.0) < 0.01, f"goal_body_x: {s9[3]} vs 8"
    assert abs(s9[4] - 0.0) < 0.01, f"goal_body_y: {s9[4]} vs 0"

    # 左转 90° → goal 在右侧
    yaw2 = np.pi / 2
    s9_2 = KinematicUAVEnv._build_state(pos, yaw2, goal)
    print(f"  yaw=π/2 → goal_body=({s9_2[3]:.3f}, {s9_2[4]:.3f}) (expect (0, -8))")
    assert abs(s9_2[3] - 0.0) < 0.3, f"goal_body_x at 90°: {s9_2[3]}"
    assert abs(s9_2[4] - (-8.0)) < 0.3, f"goal_body_y at 90°: {s9_2[4]}"

    print("  [PASS] test_coordinate\n")


# ═══════════════════════════════════════════════════════════════
# 测试 3: 完美执行 (action_to_world → 运动学转移)
# ═══════════════════════════════════════════════════════════════

def test_perfect_execution():
    """完美执行: 前进 1m → Δx≈1, 左转 90° → yaw≈π/2."""
    print("=== Test 3: 完美执行 ===")

    # 前进 1m (8 步, 每步 0.3s, action 是 delta)
    action = np.zeros((8, 11), dtype=np.float32)
    action[:, 0] = 0.125  # dpos_x = 1.0/8 per step
    action[:, 3] = 0.125 / 0.3  # dvel_x = dpos/dt

    pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    vel = np.zeros(3, dtype=np.float32)
    yaw = 0.0

    world_wp = action_to_world(action, pos, vel, yaw)
    # 第 8 步的最终位置
    last_pos = world_wp[7, POS]
    print(f"  前进1m: start=(0,0,2) → end=({last_pos[0]:.3f},{last_pos[1]:.3f},{last_pos[2]:.3f})")
    assert abs(last_pos[0] - 1.0) < 0.02, f"x={last_pos[0]:.3f}, expected ~1.0"
    assert abs(last_pos[1]) < 0.02, f"y={last_pos[1]:.3f}, expected ~0"
    assert abs(last_pos[2] - 2.0) < 0.02, f"z={last_pos[2]:.3f}, expected ~2"

    # 左转 90° (纯旋转, 无位移)
    action2 = np.zeros((8, 11), dtype=np.float32)
    action2[:, 9] = np.pi / 2 / 8  # dyaw per step
    action2[:, 10] = (np.pi / 2 / 8) / 0.3  # yaw_rate

    world_wp2 = action_to_world(action2, np.array([0.0, 0.0, 2.0]),
                                np.zeros(3), 0.0)
    last_yaw = world_wp2[7, YAW]
    print(f"  左转90°: end_yaw={last_yaw:.4f} (expect {np.pi/2:.4f})")
    assert abs(last_yaw - np.pi / 2) < 0.05, f"yaw={last_yaw:.4f}"

    # 前进 + 左转组合
    action3 = np.zeros((8, 11), dtype=np.float32)
    action3[:, 0] = 0.05   # 小前进
    action3[:, 9] = 0.1    # 小转

    world_wp3 = action_to_world(action3,
                                np.array([5.0, 0.0, 2.0]),
                                np.array([0.2, 0.0, 0.0]), 0.0)
    print(f"  组合: end_pos={world_wp3[7,POS]}, end_yaw={world_wp3[7,YAW]:.3f}")
    assert abs(world_wp3[7, POS][0] - 5.0) > 0.1, "should move forward"

    print("  [PASS] test_perfect_execution\n")


# ═══════════════════════════════════════════════════════════════
# 测试 4: env reset + step shape (MockSynthPC)
# ═══════════════════════════════════════════════════════════════

def test_env_shape():
    """reset/step 返回正确 shape."""
    print("=== Test 4: env reset/step shapes ===")

    synth_pc = MockSynthPC()
    env = KinematicUAVEnv(
        world_id=4,
        synth_pc=synth_pc,
        rl_action_steps=8,
        disturb_enable=False,
        mid_state_init_ratio=0.0,
        seed=123,
    )

    # obstacle json 必须存在 (几何碰撞需要)
    if not os.path.exists(env.obstacles_json):
        print(f"  [SKIP] obstacles.json not found: {env.obstacles_json}")
        return

    obs, info = env.reset()
    print(f"  reset obs['point_cloud']={obs['point_cloud'].shape} (expect (8,4096,3))")
    print(f"  reset obs['agent_pos']={obs['agent_pos'].shape} (expect (8,9))")
    assert obs['point_cloud'].shape == (8, 4096, 3), f"pc shape: {obs['point_cloud'].shape}"
    assert obs['agent_pos'].shape == (8, 9), f"state shape: {obs['agent_pos'].shape}"

    # step with random action
    action = np.random.randn(8, 11).astype(np.float32) * 0.01
    obs2, reward, terminated, truncated, info = env.step(action)
    print(f"  step reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
    print(f"  step obs['point_cloud']={obs2['point_cloud'].shape}")
    assert obs2['point_cloud'].shape == (8, 4096, 3)
    assert obs2['agent_pos'].shape == (8, 9)

    env.close()
    print("  [PASS] test_env_shape\n")


# ═══════════════════════════════════════════════════════════════
# 测试 5: 8 帧填充 + 滚动更新
# ═══════════════════════════════════════════════════════════════

def test_history_roll():
    """reset 后 8 帧相同, step 后旧帧被推出."""
    print("=== Test 5: 8-frame history ===")

    synth_pc = MockSynthPC()
    env = KinematicUAVEnv(
        world_id=4, synth_pc=synth_pc, rl_action_steps=8,
        disturb_enable=False, mid_state_init_ratio=0.0, seed=456,
    )
    if not os.path.exists(env.obstacles_json):
        print(f"  [SKIP] obstacles.json not found")
        return

    obs, _ = env.reset()
    pc0 = obs['point_cloud']
    state0 = obs['agent_pos']

    # reset 后 8 帧应完全相同 (无人机未动)
    for i in range(7):
        assert np.allclose(pc0[i], pc0[i+1], atol=1e-5), f"pc frame {i} != {i+1}"
    print(f"  reset: 8 frames identical ✓")

    # step 后, 前 7 帧应被推出
    action = np.zeros((8, 11), dtype=np.float32)
    action[:, 0] = 0.1
    obs2, _, _, _, _ = env.step(action)
    pc1 = obs2['point_cloud']
    # 第 0 帧 = 旧第 1 帧 (被 roll)
    assert np.allclose(pc0[1], pc1[0], atol=1e-5), "roll: pc0[1] != pc1[0]"
    print(f"  step: history rolled correctly ✓")

    env.close()
    print("  [PASS] test_history_roll\n")


# ═══════════════════════════════════════════════════════════════
# 测试 6: 扰动注入
# ═══════════════════════════════════════════════════════════════

def test_disturb():
    """人为扰动开启后, 位置被推偏."""
    print("=== Test 6: 人为扰动 ===")

    # disturb_prob=1.0 → 每步必扰
    synth_pc = MockSynthPC()
    env = KinematicUAVEnv(
        world_id=4, synth_pc=synth_pc, rl_action_steps=8,
        disturb_enable=True, disturb_prob=1.0, disturb_pos_scale=1.0,
        exec_noise_scale=0.0,
        mid_state_init_ratio=0.0, seed=789,
    )
    if not os.path.exists(env.obstacles_json):
        print(f"  [SKIP] obstacles.json not found")
        return

    env.reset()
    pos_before = env._drone_pos.copy()

    # 零 action (完美执行应几乎不动), 但扰动一定会推偏
    action = np.zeros((8, 11), dtype=np.float32)
    env.step(action)
    pos_after = env._drone_pos.copy()
    delta = np.linalg.norm(pos_after - pos_before)
    print(f"  disturb: Δpos={delta:.3f}m (expect > 0.1, scale=1.0)")
    assert delta > 0.1, f"disturb too small: {delta:.3f}"

    env.close()
    print("  [PASS] test_disturb\n")


# ═══════════════════════════════════════════════════════════════
# 测试 7: 高度门控
# ═══════════════════════════════════════════════════════════════

def test_height_gate():
    """高度门控 h(w) ∈ [h_min, 1]."""
    print("=== Test 7: 高度门控 ===")

    # 不依赖障碍文件: 直接测 _compute_phi
    synth_pc = MockSynthPC()
    env = KinematicUAVEnv(
        world_id=4, synth_pc=synth_pc, rl_action_steps=8,
        h_min=0.3, disturb_enable=False, mid_state_init_ratio=0.0, seed=1,
    )
    if not os.path.exists(env.obstacles_json):
        # 使用自定义障碍手动测
        env.obstacles = []
    env._drone_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    env._drone_yaw = 0.0
    env.goal_w = np.array([8.0, 0.0, 2.0], dtype=np.float32)

    phi = env._compute_phi()
    print(f"  φ(无障碍)={phi:.3f} (should be negative, dominated by -dist)")
    assert phi < -1.0, f"φ should be negative (dominated by -dist), got {phi:.3f}"

    # 在障碍旁 (贴障, w_avoid ≈ 1)
    # 造一个接近的障碍
    env.obstacles = [ObstacleSpec(name="near", shape="cylinder",
                                   x=0.5, y=0.0, z=2.0, radius=0.3, height=4.0)]
    phi2 = env._compute_phi()
    print(f"  φ(贴障)={phi2:.3f}")
    # 贴障时 w_avoid → 1, progress 趋零 → h(w) → h_min=0.3 (非零)
    # φ 主要由高度项贡献: h_min * |z-z_target|
    # z_target = (1+5)/2 = 3.0, z=2.0 → |z-3|=1, h_min=0.3 → -0.3
    assert abs(phi2) < 5.0, f"贴障 φ should not blow up, got {phi2:.3f}"

    env.close()
    print("  [PASS] test_height_gate\n")

    
# ═══════════════════════════════════════════════════════════════
# 测试 9: 碰撞检测严密性 (drone_radius 阈值 + 路径采样防穿墙)
# ═══════════════════════════════════════════════════════════════
#
# 把以下函数插入 test_kinematic.py (放在 test_real_synth 之前),
# 并在 main() 的 tests 列表里加入 test_collision_robustness。
#
# 验证目标 (纯数学碰撞比物理引擎更需严密):
#   漏洞1: 无人机有体积 (drone_radius=0.35), 中心离表面 < 半径即撞,
#          而非中心穿表面 (质点判据 min_clr<0 会漏 0.35m).
#   漏洞2: 离散 step 从 A 跳到 B, 薄障碍可能被跨过 (tunneling),
#          需查 A→B 路径所有 waypoint.
#   飞越豁免: drone_radius 阈值不得破坏垂直越障 (水平半径 ≠ 垂直间隙).

def test_collision_robustness():
    """碰撞检测严密性: drone_radius 阈值 + 穿墙 + 飞越豁免不破坏."""
    print("=== Test 9: 碰撞检测严密性 ===")

    synth_pc = MockSynthPC()
    env = KinematicUAVEnv(
        world_id=4, synth_pc=synth_pc, rl_action_steps=8,
        disturb_enable=False, mid_state_init_ratio=0.0, seed=1,
    )
    print(f"  drone_radius = {env.drone_radius}")
    # ── 漏洞1: drone_radius 阈值 ──
    # cyl 中心(0,0) 半径0.5. 无人机中心(0.7,0,2): 表面距=0.7-0.5=0.2 < 0.35 → 撞
    env.obstacles = [ObstacleSpec(name="cyl", shape="cylinder",
                                   x=0.0, y=0.0, z=1.5, radius=0.5, height=3.0)]
    env._drone_pos = np.array([0.7, 0.0, 2.0], dtype=np.float32)
    hit = env._check_collision()
    print(f"  漏洞1 机身半径: 表面距0.2 < r=0.35 → 撞={hit} (expect True)")
    assert hit, f"中心离表面0.2 < drone_radius0.35 应判撞, got {hit}"

    # 对照: 表面距0.5 > 0.35 → 不撞
    env._drone_pos = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    hit_far = env._check_collision()
    print(f"  对照: 表面距0.5 > r=0.35 → 撞={hit_far} (expect False)")
    assert not hit_far, f"表面距0.5 > drone_radius 不应判撞, got {hit_far}"

    # ── 漏洞2: 路径采样防穿墙 ──
    # 薄墙 box 中心(5,0) x向厚0.2 (x∈[4.9,5.1]). A(4,0,2)→B(6,0,2) 两端都在墙外,
    # 但路径穿过 x=5. 只查终点不撞, 查路径应撞.
    env.obstacles = [ObstacleSpec(name="wall", shape="box",
                                   x=5.0, y=0.0, z=1.5,
                                   size_x=0.2, size_y=4.0, size_z=3.0)]
    # 造 A→B 路径 (8 waypoint, x 从 4.25 到 6.0)
    path = np.zeros((8, 11), dtype=np.float32)
    for k in range(8):
        path[k, POS] = [4.0 + 2.0 * (k + 1) / 8, 0.0, 2.0]
    env._drone_pos = np.array([6.0, 0.0, 2.0], dtype=np.float32)  # 终点墙外

    hit_end = env._check_collision()  # 不传路径, 只查终点
    print(f"  漏洞2 穿墙: 只查终点(6,0,2) → 撞={hit_end} (expect False, 终点在墙外)")
    assert not hit_end, f"终点在墙外只查终点不应撞, got {hit_end}"

    hit_path = env._check_collision(path_wp=path)  # 查路径
    print(f"  查路径(穿过x=5的墙) → 撞={hit_path} (expect True)")
    assert hit_path, f"路径穿墙应判撞, got {hit_path}"

    # ── 飞越豁免不被 drone_radius 破坏 ──
    # cyl 中心(0,0) 半径0.5 顶3.0. 无人机正上方 z=3.5 (底部3.4 > 顶3.0) → 飞越合法
    env.obstacles = [ObstacleSpec(name="cyl", shape="cylinder",
                                   x=0.0, y=0.0, z=1.5, radius=0.5, height=3.0)]
    env._drone_pos = np.array([0.0, 0.0, 3.5], dtype=np.float32)
    hit_fly = env._check_collision()
    print(f"  飞越豁免: 柱正上方z=3.5 → 撞={hit_fly} (expect False, 垂直越障合法)")
    assert not hit_fly, f"飞越不应被 drone_radius 误判, got {hit_fly}"

    # 对照: z=2.9 在柱内 (非飞越) → 撞
    env._drone_pos = np.array([0.0, 0.0, 2.9], dtype=np.float32)
    hit_in = env._check_collision()
    print(f"  对照: z=2.9 在柱内 → 撞={hit_in} (expect True)")
    assert hit_in, f"柱内应判撞, got {hit_in}"

    env.close()
    print("  [PASS] test_collision_robustness\n")


# ═══════════════════════════════════════════════════════════════
# 测试 10: 多 world 池模式
# ═══════════════════════════════════════════════════════════════

def test_multi_world():
    """world_pool 模式: 随机切 world + obstacles/synth_pc 同源."""
    print("=== Test 10: 多 world 池 ===")

    # 用真实障碍文件 (至少需要2个world)
    import os
    from diffusion_policy_3d.env.kinematic_uav_env import scan_world_pool, TRAIN_WORLDS_DIR
    pool = scan_world_pool(str(TRAIN_WORLDS_DIR))
    if len(pool) < 2:
        print(f"  [SKIP] 训练world不足: {len(pool)} < 2")
        return
    # 只用前3个测试 (快)
    mini_pool = pool[:3]
    print(f"  mini_pool: {[name for name, _ in mini_pool]}")

    # 用 MockSynthPC 当作 synth_pc_pool
    synth_pc_pool = {name: MockSynthPC(json_path) for name, json_path in mini_pool}
    env = KinematicUAVEnv(
        world_pool=mini_pool, synth_pc_pool=synth_pc_pool,
        rl_action_steps=8, disturb_enable=False, mid_state_init_ratio=0.0, seed=42,
    )

    # reset 多次, 观察 world_id 变化
    seen = set()
    for i in range(10):
        obs, info = env.reset(seed=42 + i)
        wid = info['world_id']
        seen.add(wid)
        # 验证: synth_pc 与 world 同源
        synth_json = getattr(env.synth_pc, 'obstacles_json', 'unknown')
        print(f"  reset {i}: world={wid}, synth_pc={synth_json}")
        # 验证 obs shape
        assert obs['point_cloud'].shape == (8, 4096, 3)
        assert obs['agent_pos'].shape == (8, 9)

        # step 一次验证不崩溃
        action = np.zeros((8, 11), dtype=np.float32)
        _, _, _, _, _ = env.step(action)

    env.close()
    print(f"  seen worlds: {seen}")
    assert len(seen) > 1, f"多world模式应跨多world, got {len(seen)}"
    print("  [PASS] test_multi_world\n")


# ═══════════════════════════════════════════════════════════════
# 测试 8: 真实 SynthPointCloud (需要 GPU + obstacles.json)
# ═══════════════════════════════════════════════════════════════

def test_real_synth():
    """用真实 SynthPointCloud 跑 reset + step, 检查 obs 合理."""
    print("=== Test 8: 真实 SynthPointCloud ===")
    try:
        import torch
        from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
    except ImportError as e:
        print(f"  [SKIP] torch not available: {e}")
        return

    world_id = 4
    if world_id not in WORLD_ENV_MAP:
        print(f"  [SKIP] world {world_id} not in WORLD_ENV_MAP")
        return

    obstacles_json = str(WORLDS_DIR / f"{WORLD_ENV_MAP[world_id]}.obstacles.json")
    if not os.path.exists(obstacles_json):
        print(f"  [SKIP] obstacles.json not found: {obstacles_json}")
        return

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"  device={device}, obstacles={obstacles_json}")

    synth_pc = SynthPointCloud(obstacles_json, device=device)
    env = KinematicUAVEnv(
        world_id=world_id, synth_pc=synth_pc, rl_action_steps=8,
        disturb_enable=False, mid_state_init_ratio=0.0, seed=42,
    )

    obs, _ = env.reset()
    pc = obs['point_cloud']
    s9 = obs['agent_pos']
    print(f"  reset: pc={pc.shape}, agent_pos={s9.shape}")
    print(f"  agent_pos[0]={s9[0]}")
    print(f"  pc range: [{pc.min():.1f}, {pc.max():.1f}]")

    # 检查 SPAWN_OFFSET (synth_pc 输入坐标是否正确)
    drone_pos = env._drone_pos
    print(f"  drone_pos (world): {drone_pos}")
    # 在已知位置产新点云, 检查障碍是否在合理位置
    test_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    pc2 = synth_pc.synthesize(test_pos + SPAWN_OFFSET, 0.0, num_points=4096, add_noise=False)
    print(f"  pc at origin: shape={pc2.shape}, range=[{pc2.min():.1f}, {pc2.max():.1f}]")

    # 跑几步
    for i in range(5):
        action = np.random.randn(8, 11).astype(np.float32) * 0.01
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"  step {i}: terminated, reason={info.get('done_reason')}")
            break
    print(f"  steps completed, drone_pos={env._drone_pos}")

    env.close()
    print("  [PASS] test_real_synth\n")


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    use_real = '--real-synth' in sys.argv

    print("=" * 60)
    print("KinematicUAVEnv 单元测试")
    print("=" * 60)

    # 1-7: 纯数学/逻辑 (无需 GPU, 可本地跑)
    tests = [
        test_geometry,
        test_coordinate,
        test_perfect_execution,
        test_env_shape,
        test_history_roll,
        test_disturb,
        test_height_gate,
        test_collision_robustness,
        test_multi_world,
    ]
    results = {}
    for t in tests:
        try:
            t()
            results[t.__name__] = 'PASS'
        except Exception as e:
            results[t.__name__] = f'FAIL: {e}'
            print(f"  [FAIL] {t.__name__}: {e}\n")

    # 8: 真实 synth (需要 GPU + 障碍文件)
    if use_real:
        try:
            test_real_synth()
            results['test_real_synth'] = 'PASS'
        except Exception as e:
            results['test_real_synth'] = f'FAIL: {e}'
            print(f"  [FAIL] test_real_synth: {e}\n")
    else:
        print("=== Test 8: 真实 SynthPointCloud === [SKIP] (use --real-synth)\n")

    # 汇总
    print("=" * 60)
    passed = sum(1 for v in results.values() if v == 'PASS')
    failed = sum(1 for v in results.values() if 'FAIL' in str(v))
    print(f"Results: {passed} passed, {failed} failed")
    for name, result in results.items():
        status = '✓' if result == 'PASS' else '✗'
        print(f"  {status} {name}: {result}")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
