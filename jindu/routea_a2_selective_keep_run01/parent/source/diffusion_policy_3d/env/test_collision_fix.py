"""碰撞检测修复验证 — 漏洞1(drone_radius) + 漏洞2(穿墙) + 飞越豁免不破坏"""
import numpy as np, json, tempfile, os
from pathlib import Path

class MockSynthPC:
    def synthesize(self, pos, yaw, num_points=4096, add_noise=True):
        return np.random.randn(num_points, 3).astype(np.float32) * 2.0

# 障碍: cylinder 中心(0,0) 半径0.5 高3; box 中心(5,0) 2x0.2x3 (薄墙,y向薄)
obs_data = {"obstacles": [
    {"name":"cyl","shape":"cylinder","x":0.0,"y":0.0,"z":1.5,"radius":0.5,"height":3.0},
    {"name":"wall","shape":"box","x":5.0,"y":0.0,"z":1.5,"size_x":0.2,"size_y":4.0,"size_z":3.0},
]}
tmpdir = tempfile.mkdtemp()
d = os.path.join(tmpdir, "PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA")
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "env_004_obs19.obstacles.json"), "w") as f:
    json.dump(obs_data, f)

import testpkg.kinematic_uav_env as kenv
kenv.WORLDS_DIR = Path(d)
env = kenv.KinematicUAVEnv(world_id=4, synth_pc=MockSynthPC(), seed=1)
env.reset()
print(f"drone_radius = {env.drone_radius}")

# ── 漏洞1: 中心离cyl表面0.2 (<0.35机身半径) 应判撞 ──
# cyl中心(0,0)半径0.5, 无人机在(0.7,0,2): 中心距0.7, 表面距0.7-0.5=0.2
env._drone_pos = np.array([0.7, 0.0, 2.0], np.float32)
hit = env._check_collision()  # 不传路径, 只查当前点
print(f"\n漏洞1 机身半径: 表面距0.2<0.35 → 撞={hit} (期望True), 原因={env._done_reason if hit else 'none'}")
assert hit, "中心离表面0.2 < drone_radius0.35 应判撞"

# 对照: 中心离表面0.5 (>0.35) 不撞
env._drone_pos = np.array([1.0, 0.0, 2.0], np.float32)  # 表面距1.0-0.5=0.5
hit2 = env._check_collision()
print(f"  对照: 表面距0.5>0.35 → 撞={hit2} (期望False)")
assert not hit2, "表面距0.5 > 0.35 不应判撞"

# ── 漏洞2: 穿墙 — A(4,0,2)和B(6,0,2)都在wall(x=5,厚0.2)外, 但路径穿过 ──
import testpkg.traj_utils as tu
# 造path: 从(4,0,2)直线到(6,0,2), 8步
path = np.zeros((8, 11), np.float32)
for k in range(8):
    x = 4.0 + (6.0-4.0)*(k+1)/8  # 4.25, 4.5, ..., 6.0
    path[k, 0:3] = [x, 0, 2]
env._drone_pos = np.array([6.0, 0.0, 2.0], np.float32)  # 终点在墙外
# 终点单独查: 不撞 (x=6 离墙x=5表面 6-5.1=0.9>0.35)
hit_end_only = env._check_collision()
print(f"\n漏洞2 穿墙: 只查终点(6,0,2) → 撞={hit_end_only} (期望False, 终点在墙外)")
# 查路径: 中间点x=5穿墙, 应撞
hit_path = env._check_collision(path_wp=path)
print(f"  查路径(穿过x=5的墙) → 撞={hit_path} (期望True), 原因={env._done_reason if hit_path else 'none'}")
assert not hit_end_only, "终点在墙外, 只查终点不撞"
assert hit_path, "路径穿墙应判撞"

# ── 飞越豁免不破坏: 无人机在cyl正上方z=3.5 (顶3.0+margin) ──
env._drone_pos = np.array([0.0, 0.0, 3.5], np.float32)  # XY在柱内但飞越
hit_fly = env._check_collision()
print(f"\n飞越豁免: cyl正上方z=3.5 → 撞={hit_fly} (期望False, 垂直越障合法)")
assert not hit_fly, "飞越不应被drone_radius误判"

# 对照: 飞越时真撞顶 (z=3.0刚好顶, 垂直间隙<0)
env._drone_pos = np.array([0.0, 0.0, 2.9], np.float32)  # 在柱顶内部
hit_top = env._check_collision()
print(f"  对照: z=2.9在柱内 → 撞={hit_top} (期望True)")

print("\n=== 碰撞修复验证通过 ===")
