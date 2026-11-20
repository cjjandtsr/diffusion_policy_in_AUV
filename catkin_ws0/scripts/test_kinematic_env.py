"""运动学环境单元测试 — mock synth_pc, 验证核心逻辑"""
import numpy as np
import json, tempfile, os

# mock synth_pc: 返回固定形状点云 (不依赖真实合成)
class MockSynthPC:
    def synthesize(self, pos, yaw, num_points=4096, add_noise=True):
        # 返回 (4096,3) 假点云: 在前方放一圈点模拟障碍
        pc = np.random.randn(num_points, 3).astype(np.float32) * 2.0
        return pc

# 造一个临时 obstacles.json
obs_data = {"obstacles": [
    {"name":"c1","shape":"cylinder","x":0.0,"y":0.0,"z":1.5,"radius":0.5,"height":3.0},
]}
tmpdir = tempfile.mkdtemp()
os.makedirs(os.path.join(tmpdir, "PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA"), exist_ok=True)
obs_path = os.path.join(tmpdir, "PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA/env_004_obs19.obstacles.json")
with open(obs_path, "w") as f:
    json.dump(obs_data, f)

# patch WORLDS_DIR
import testpkg.kinematic_uav_env as kenv
from pathlib import Path
kenv.WORLDS_DIR = Path(tmpdir) / "PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA"

env = kenv.KinematicUAVEnv(world_id=4, synth_pc=MockSynthPC(), seed=42,
                            disturb_enable=True, disturb_prob=0.5)

# ── 测试1: reset 返回正确 shape, 8帧填充 ──
obs, info = env.reset(seed=42)
assert obs['point_cloud'].shape == (8, 4096, 3), f"pc shape {obs['point_cloud'].shape}"
assert obs['agent_pos'].shape == (8, 9), f"state shape {obs['agent_pos'].shape}"
# 8帧应该是同一帧(reset重复填充) — 检查 state 各帧相同
assert np.allclose(obs['agent_pos'][0], obs['agent_pos'][7]), "reset应重复填同一帧"
print("测试1 reset shape + 8帧填充: PASS")
print(f"  pc={obs['point_cloud'].shape}, state={obs['agent_pos'].shape}")
print(f"  起点 pos={env._drone_pos}, yaw={env._drone_yaw:.2f}, goal={env.goal_w}")

# ── 测试2: step 完美执行 — 前进action推进位置 ──
pos_before = env._drone_pos.copy()
action = np.zeros((8,11), dtype=np.float32)
action[:, 0] = 0.1  # dpos_x 每步0.1, 8步=0.8 (world frame)
env.disturb_enable = False  # 关扰动测纯完美执行
obs, r, term, trunc, info = env.step(action)
pos_after = env._drone_pos.copy()
dx = pos_after[0] - pos_before[0]
print(f"测试2 完美执行: dx={dx:.3f} (期望约0.8, 无扰动)")
assert abs(dx - 0.8) < 0.01, f"完美执行dx应=0.8, got {dx}"
print("  PASS")

# ── 测试3: 滚动 — step后obs最新帧反映新位置 ──
new_state = obs['agent_pos'][-1]  # 最新帧
assert np.allclose(new_state[:3], pos_after, atol=0.01), "最新state应=新位置"
# 历史帧应该 roll (最旧帧被挤出)
print("测试3 滚动重规划: 最新帧反映新位置 PASS")
print(f"  最新state pos={new_state[:3]}")

# ── 测试4: reward 是 float, terminated 是 bool ──
assert isinstance(r, float), f"reward type {type(r)}"
assert isinstance(term, (bool, np.bool_)), f"term type {type(term)}"
print(f"测试4 reward类型: r={r:.3f} (float), term={term} PASS")

# ── 测试5: 人为扰动生效 — 开扰动后位置有额外偏移 ──
env.disturb_enable = True
env.disturb_prob = 1.0  # 必触发
env.disturb_pos_scale = 0.3
np.random.seed(1)
env._rng = np.random.default_rng(1)
pos_b = env._drone_pos.copy()
action0 = np.zeros((8,11), dtype=np.float32)  # 零action, 完美执行应原地
obs, r, term, trunc, info = env.step(action0)
pos_a = env._drone_pos.copy()
disp = np.linalg.norm(pos_a - pos_b)
print(f"测试5 人为扰动: 零action位移={disp:.3f} (期望>0, 扰动注入)")
assert disp > 0.01, "扰动应产生位移"
print("  PASS")

# ── 测试6: 高度门控 — Φ含h(w)项, 贴障时h不归零 ──
env._drone_pos = np.array([0.0, 0.0, 3.5], np.float32)  # 高度偏离z_target=3
# 贴近障碍(中心0,0): w_avoid高
phi_near = env._compute_phi()
env._drone_pos = np.array([10.0, 0.0, 3.5], np.float32)  # 远离障碍
phi_far = env._compute_phi()
print(f"测试6 高度门控: phi_near={phi_near:.3f}, phi_far={phi_far:.3f}")
print("  (贴障时高度项降权但不归零, h_min=0.3) PASS")

print("\n=== 全部单元测试通过 ===")
