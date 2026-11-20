#!/usr/bin/env python3
"""
bag2zarr_v5.py — v5: 15维state + world step-wise delta action (逐帧存储, 多文件输出)

═══════════════════════════════════════════════════════════════════════════
  关键特性:
  
  1. 存储格式: per-frame 逐帧存储 (与 iDP3 ReplayBuffer 兼容)
     state(N_frames, 15), action(N_frames, 11), point_cloud(N_frames, 4096, 6)
  
  2. action 表示: step-wise delta in world frame
     action[t] = cmd_pos_w[t+1] - cmd_pos_w[t]  (不做 R(-yaw) 旋转)
  
  3. 输出: 每个 episode 独立保存为一个 .zarr 文件 (类似 v4)
     文件名: {prefix}_{num}.zarr, 序号自动递增
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import zarr
import os
import re
import time
import argparse
import gc
from sensor_msgs import point_cloud2

# 常量
NUM_POINTS = 4096
TRUNCATE_DIST = 0.5
MIN_STEPS = 15
DPZ_OUTLIER_THRESH = 1.0
STATE_DIM = 15
ACTION_DIM = 11

R_CAM2BODY = np.array([[ 0,  0,  1],
                        [-1,  0,  0],
                        [ 0, -1,  0]], dtype=np.float64)


def quat_to_yaw(quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def Rz(angle):
    c, s = np.cos(float(angle)), np.sin(float(angle))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def normalize_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def extract_odom_state(odom_msg):
    """提取 odom: [pos(3), vel_body(3), quat_wxyz(4)] = 10维"""
    s = np.zeros(10, dtype=np.float32)
    s[0] = odom_msg.pose.pose.position.x
    s[1] = odom_msg.pose.pose.position.y
    s[2] = odom_msg.pose.pose.position.z
    s[3] = odom_msg.twist.twist.linear.x
    s[4] = odom_msg.twist.twist.linear.y
    s[5] = odom_msg.twist.twist.linear.z
    s[6] = odom_msg.pose.pose.orientation.w
    s[7] = odom_msg.pose.pose.orientation.x
    s[8] = odom_msg.pose.pose.orientation.y
    s[9] = odom_msg.pose.pose.orientation.z
    return s

def extract_action(cmd_msg):
    """提取 cmd: [pos(3), vel(3), acc(3), yaw, yaw_rate] = 11维 (world)"""
    a = np.zeros(11, dtype=np.float32)
    a[0] = cmd_msg.position.x
    a[1] = cmd_msg.position.y
    a[2] = cmd_msg.position.z
    a[3] = cmd_msg.velocity.x
    a[4] = cmd_msg.velocity.y
    a[5] = cmd_msg.velocity.z
    a[6] = cmd_msg.acceleration.x
    a[7] = cmd_msg.acceleration.y
    a[8] = cmd_msg.acceleration.z
    a[9] = cmd_msg.yaw
    a[10] = cmd_msg.yaw_dot
    return a

def extract_goal(goal_msg):
    return np.array([
        goal_msg.pose.position.x,
        goal_msg.pose.position.y,
        goal_msg.pose.position.z,
    ], dtype=np.float32)

def process_pointcloud(pc_msg):
    try:
        gen = point_cloud2.read_points(pc_msg, field_names=("x","y","z"), skip_nans=True)
        pts = np.array(list(gen), dtype=np.float32)
        if len(pts) == 0:
            return None
        # 转换到 body 坐标系
        pts_body = (R_CAM2BODY @ pts[:, :3].T).T
        out = np.zeros((len(pts_body), 6), dtype=np.float32)
        out[:, :3] = pts_body
        return out
    except Exception as e:
        print(f"    点云处理出错: {e}")
        return None

def resample_pc(pc, target_n=NUM_POINTS):
    if pc is None or len(pc) == 0:
        return np.zeros((target_n, 6), dtype=np.float32)
    n = len(pc)
    if n == target_n:
        return pc
    idx = np.random.choice(n, target_n, replace=(n < target_n))
    return pc[idx]


def process_episode(ep_data, ep_index):
    """
    处理一个 episode, 返回逐帧数据:
      states:  (T', 15)
      actions: (T', 11)
      pcs:     (T', 4096, 6)
    或 None
    """
    states_raw  = ep_data['states']      # list of (10,) odom
    actions_raw = ep_data['actions']     # list of (11,) cmd (world raw)
    goals_raw   = ep_data['goals']       # list of (3,)
    pc_msgs     = ep_data['pc_msgs']
    T = len(states_raw)

    # 1. 截断: 到达目标后停止
    truncate_at = T
    for i in range(T):
        dist = np.linalg.norm(states_raw[i][0:3] - goals_raw[i])
        if dist < TRUNCATE_DIST:
            truncate_at = i
            break

    if truncate_at < MIN_STEPS:
        print(f"    ep[{ep_index}]: 有效步数 {truncate_at} < {MIN_STEPS}, 跳过")
        return None

    if truncate_at < T:
        pct = (T - truncate_at) / T * 100
        print(f"    ep[{ep_index}]: {T} → {truncate_at} 步 (去除 {pct:.0f}% 悬停)")

    T = truncate_at

    # 2. 预计算 world 速度、加速度、yaw
    dt = 0.3
    pos_list = []
    vel_w_list = []
    yaw_list = []

    for t in range(T):
        pos = states_raw[t][0:3].astype(np.float64)
        vel_body = states_raw[t][3:6].astype(np.float64)
        quat = states_raw[t][6:10].astype(np.float64)
        yaw = quat_to_yaw(quat)
        vel_world = Rz(yaw) @ vel_body
        pos_list.append(pos)
        vel_w_list.append(vel_world)
        yaw_list.append(yaw)

    acc_w_list = [np.zeros(3, dtype=np.float64)]
    for t in range(1, T):
        acc_w_list.append((vel_w_list[t] - vel_w_list[t-1]) / dt)

    # 3. 构建逐帧 state (15维) 和 action (11维)
    # action[t] 需要 cmd[t+1], 所以有效帧数 = T-1
    T_valid = T - 1

    if T_valid < MIN_STEPS:
        print(f"    ep[{ep_index}]: action有效帧 {T_valid} < {MIN_STEPS}, 跳过")
        return None

    out_states  = np.zeros((T_valid, STATE_DIM), dtype=np.float32)
    out_actions = np.zeros((T_valid, ACTION_DIM), dtype=np.float32)
    out_pcs     = []

    n_outlier = 0

    for t in range(T_valid):
        # ── state (15维) ──
        pos_w = pos_list[t]
        vel_w = vel_w_list[t]
        acc_w = acc_w_list[t]
        yaw = yaw_list[t]
        goal_w = goals_raw[t].astype(np.float64)
        goal_rel = goal_w - pos_w
        goal_body = Rz(-yaw) @ goal_rel

        out_states[t, 0:3]  = pos_w
        out_states[t, 3:6]  = vel_w
        out_states[t, 6:9]  = acc_w
        out_states[t, 9:12] = goal_body
        out_states[t, 12]   = np.linalg.norm(goal_rel)
        out_states[t, 13]   = np.sin(yaw)
        out_states[t, 14]   = np.cos(yaw)

        # ── action (11维, world step-wise delta) ──
        cmd_t  = actions_raw[t]    # (11,) = [pos, vel, acc, yaw, yr] in world
        cmd_t1 = actions_raw[t+1]  # next frame cmd

        dpos_world = cmd_t1[0:3] - cmd_t[0:3]   # world 一步位移
        dvel_world = cmd_t1[3:6] - cmd_t[3:6]   # world 速度变化
        acc_world  = cmd_t[6:9]                  # world 加速度
        dyaw       = normalize_angle(cmd_t1[9] - cmd_t[9])  # step-wise Δyaw
        yr         = cmd_t[10]                   # yaw rate

        # 异常值检测
        if np.abs(dpos_world[2]) > DPZ_OUTLIER_THRESH:
            n_outlier += 1
            dpos_world[2] = np.clip(dpos_world[2], -DPZ_OUTLIER_THRESH, DPZ_OUTLIER_THRESH)

        out_actions[t, 0:3] = dpos_world
        out_actions[t, 3:6] = dvel_world
        out_actions[t, 6:9] = acc_world
        out_actions[t, 9]   = dyaw
        out_actions[t, 10]  = yr

        # ── point_cloud ──
        pc = process_pointcloud(pc_msgs[t])
        pc = resample_pc(pc, NUM_POINTS)
        out_pcs.append(pc)

    out_pcs = np.stack(out_pcs)  # (T_valid, 4096, 6)

    if n_outlier > 0:
        print(f"    ep[{ep_index}]: {n_outlier} 帧 |Δpz|>{DPZ_OUTLIER_THRESH} 被 clamp")

    # 统计
    dp_mag = np.sqrt(np.sum(out_actions[:, 0:3]**2, axis=1))
    dy_std = out_actions[:, 1].std()
    print(f"    ✓ ep[{ep_index}]: {T_valid} 帧, |Δpos_w|={dp_mag.mean():.3f}±{dp_mag.std():.3f}m, "
          f"Δy_w std={dy_std:.4f}")

    return out_states, out_actions, out_pcs


def save_episode_zarr(states, actions, pcs, output_path):
    """保存单个 episode 的逐帧数据到 zarr 文件"""
    N = len(states)
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    data_g = root.create_group('data')
    meta_g = root.create_group('meta')

    data_g.array('state', states, chunks=(min(500, N), STATE_DIM), dtype=np.float32)
    data_g.array('action', actions, chunks=(min(500, N), ACTION_DIM), dtype=np.float32)
    data_g.array('point_cloud', pcs, chunks=(min(20, N), NUM_POINTS, 6), dtype=np.float32)
    meta_g.array('episode_ends', np.array([N], dtype=np.int64))

    root.attrs.update({
        'num_episodes': 1,
        'total_timesteps': N,
        'state_format': '[pos_world(3), vel_world(3), acc_world(3), goal_body(3), dist(1), sin_yaw(1), cos_yaw(1)]',
        'state_dim': STATE_DIM,
        'action_format': '[dpos_world(3), dvel_world(3), acc_world(3), dyaw(1), yaw_rate(1)]',
        'action_note': 'v5: per-frame step-wise delta, world frame, NO R(-yaw) rotation',
        'action_dim': ACTION_DIM,
        'point_cloud_format': '[xyz_body(3), color(3)]',
        'coordinate_frame': 'action=world, pointcloud=body, goal_body=body',
        'sync_freq': '3.3Hz',
        'dt': 0.3,
        'version': 'v5',
        'created': time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def get_next_number(output_dir, prefix):
    """获取下一个可用的序号"""
    if not os.path.exists(output_dir):
        return 1
    max_num = 0
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)\.zarr$')
    for name in os.listdir(output_dir):
        m = pattern.match(name)
        if m:
            num = int(m.group(1))
            if num > max_num:
                max_num = num
    return max_num + 1


def convert_one_bag(bag_path, output_dir, prefix, start_num):
    """处理单个 bag 文件，返回 (saved_episodes, next_num)"""
    import rosbag

    print(f"\n{'─'*60}")
    print(f"  处理: {os.path.basename(bag_path)}")
    print(f"{'─'*60}")

    try:
        bag = rosbag.Bag(bag_path, 'r')
    except Exception as e:
        print(f"  ✗ 无法打开: {e}")
        return 0, start_num

    current_ep = None
    in_avoidance = False
    episode_saved = 0
    ep_index = 0
    sync_count = 0
    current_num = start_num

    for topic, msg, t in bag.read_messages(topics=['/sync_data', '/current_phase']):
        ts = t.to_sec()

        if topic == '/current_phase':
            phase = msg.data
            if phase == "avoidance_start" and not in_avoidance:
                if current_ep is not None:
                    print("  ⚠ 未结束的片段被丢弃")
                current_ep = {
                    'timestamps': [], 'states': [], 'actions': [],
                    'goals': [], 'pc_msgs': [],
                }
                in_avoidance = True
                sync_count = 0
                print(f"    avoidance_start @ t={ts:.1f}")

            elif phase == "avoidance_end" and in_avoidance:
                print(f"    avoidance_end   @ t={ts:.1f}  ({sync_count} 步)")
                if current_ep and len(current_ep['states']) >= MIN_STEPS:
                    result = process_episode(current_ep, ep_index)
                    if result is not None:
                        states, actions, pcs = result
                        out_name = f"{prefix}_{current_num}.zarr"
                        out_path = os.path.join(output_dir, out_name)
                        save_episode_zarr(states, actions, pcs, out_path)
                        episode_saved += 1
                        print(f"      保存: {out_name} ({len(states)} 帧)")
                        current_num += 1
                    ep_index += 1
                else:
                    n = len(current_ep['states']) if current_ep else 0
                    print(f"    ep[{ep_index}]: 仅 {n} 步, 跳过")
                current_ep = None
                in_avoidance = False
            continue

        if topic == '/sync_data' and in_avoidance and current_ep is not None:
            odom_state = extract_odom_state(msg.odom)
            action = extract_action(msg.pos_cmd)
            goal = extract_goal(msg.goal)
            current_ep['timestamps'].append(ts)
            current_ep['states'].append(odom_state)
            current_ep['actions'].append(action)
            current_ep['goals'].append(goal)
            current_ep['pc_msgs'].append(msg.pointcloud)
            sync_count += 1

    # 处理未关闭的片段
    if current_ep and len(current_ep['states']) >= MIN_STEPS:
        print(f"    ⚠ 未收到 avoidance_end, 处理最后片段 ({sync_count} 步)")
        result = process_episode(current_ep, ep_index)
        if result is not None:
            states, actions, pcs = result
            out_name = f"{prefix}_{current_num}.zarr"
            out_path = os.path.join(output_dir, out_name)
            save_episode_zarr(states, actions, pcs, out_path)
            episode_saved += 1
            print(f"      保存: {out_name} ({len(states)} 帧)")
            current_num += 1

    bag.close()
    gc.collect()
    print(f"  ✓ 完成: {episode_saved} 条轨迹")
    return episode_saved, current_num


def main():
    parser = argparse.ArgumentParser(
        description='bag2zarr v5: 15维state + world step-wise delta action (逐帧存储，多文件输出)')
    parser.add_argument('--bags_dir', type=str, required=True,
                        help='包含 .bag 文件的目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出 zarr 目录')
    parser.add_argument('--prefix', type=str, default='avoidance_dataset',
                        help='输出文件前缀')
    parser.add_argument('--keep-bags', action='store_true',
                        help='处理后保留 bag 文件')
    parser.add_argument('--truncate_dist', type=float, default=0.1,
                        help='到达目标距离阈值')
    args = parser.parse_args()

    global TRUNCATE_DIST
    TRUNCATE_DIST = args.truncate_dist

    import glob
    bag_files = sorted(glob.glob(os.path.join(args.bags_dir, '*.bag')))
    if not bag_files:
        print(f"未找到 .bag 文件: {args.bags_dir}")
        return

    def sort_key(f):
        m = re.search(r'(\d+)', os.path.basename(f))
        return int(m.group(1)) if m else 0
    bag_files.sort(key=sort_key)

    os.makedirs(args.output_dir, exist_ok=True)
    start_num = get_next_number(args.output_dir, args.prefix)

    print("═" * 60)
    print("  bag2zarr v5 — 逐帧存储 + world step-wise delta (多文件输出)")
    print("═" * 60)
    print(f"  输入:     {args.bags_dir} ({len(bag_files)} 个 bag)")
    print(f"  输出:     {args.output_dir}")
    print(f"  起始编号: {start_num}")
    print(f"  prefix:   {args.prefix}")
    print(f"  state:    {STATE_DIM}维 [pos_w, vel_w, acc_w, goal_b, dist, sin/cos]")
    print(f"  action:   {ACTION_DIM}维 [dpos_w, dvel_w, acc_w, dyaw, yr]")
    print(f"  格式:     逐帧 (N, dim) — 兼容 iDP3 ReplayBuffer")
    print(f"  截断距离: {TRUNCATE_DIST}m")
    print("═" * 60)

    total_saved = 0
    current_num = start_num

    for i, bag_path in enumerate(bag_files):
        print(f"\n[{i+1}/{len(bag_files)}] {os.path.basename(bag_path)}")
        try:
            saved, current_num = convert_one_bag(
                bag_path, args.output_dir, args.prefix, current_num)
            total_saved += saved
            if saved > 0 and not args.keep_bags:
                os.remove(bag_path)
                print(f"  🗑 已删除: {os.path.basename(bag_path)}")
        except Exception as e:
            print(f"  ✗ 异常: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'═'*60}")
    print(f"  完成! 共保存 {total_saved} 条轨迹")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()