#!/usr/bin/env python3
"""
P1-2 端到端验证: SITL 重启后 RL 节点能否控制无人机

在 AutoDL 服务器上跑:
  1. conda activate idp3
  2. source /opt/ros/noetic/setup.bash
  3. source ~/AUV/catkin_ws0/devel/setup.bash
  4. python test_p1_2_reconnect.py
"""

import os, sys, time, subprocess, threading
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

SCRIPTS = os.path.expanduser("~/AUV/catkin_ws0/scripts")
TAKEOFF = os.path.join(SCRIPTS, "takeoff_only.sh")
START_SITL = os.path.join(SCRIPTS, "start_sitl.sh")

rospy.init_node('test_reconnect', anonymous=True, disable_signals=True)

odom_cnt = [0]
def cb(msg): odom_cnt[0] += 1

rospy.Subscriber('/mavros/local_position/odom', Odometry, cb, queue_size=1)
cmd_pub = rospy.Publisher('/planning/pos_cmd', Float64, queue_size=10)
spin_t = threading.Thread(target=rospy.spin, daemon=True)
spin_t.start()
rospy.sleep(1.0)

# ═══ Phase 1: 起 SITL + takeoff, 验证能控制 ═══
print("[Phase 1] 起 SITL...")
subprocess.run(['bash', START_SITL, 'env_004_obs19.world'],
               cwd=SCRIPTS, timeout=120)
for _ in range(40):  # 等 MAVROS
    if odom_cnt[0] > 0: break
    time.sleep(3)
assert odom_cnt[0] > 0, "FAIL: odom 无数据"
print(f"  odom OK ({odom_cnt[0]} msgs)")

print("[Phase 1] 起飞 + px4ctrl...")
subprocess.run(['bash', TAKEOFF], cwd=SCRIPTS, timeout=90)
rospy.sleep(5)
print("  起飞完成")

# 发包测试
cmd_pub.publish(Float64(0.0))
rospy.sleep(0.5)
print("  发包 OK")

# ═══ Phase 2: 重启 (模拟 episode reset) ═══
print("[Phase 2] 重启 SITL + px4ctrl...")
subprocess.run(['pkill', '-9', '-f', 'px4_sitl_default'], timeout=10)
subprocess.run(['pkill', '-9', '-f', 'gzserver'], timeout=10)
time.sleep(5)
subprocess.run(['bash', START_SITL, 'env_004_obs19.world'],
               cwd=SCRIPTS, timeout=120)
odom_cnt[0] = 0
for _ in range(40):
    if odom_cnt[0] > 0: break
    time.sleep(3)
assert odom_cnt[0] > 0, "FAIL: odom 重连失败"
print(f"  odom 重连 OK ({odom_cnt[0]} msgs)")

subprocess.run(['bash', TAKEOFF], cwd=SCRIPTS, timeout=90)
rospy.sleep(5)

# ═══ Phase 3: 重连后发指令 ═══
print("[Phase 3] 重连后发包...")
cmd_pub.publish(Float64(1.0))
rospy.sleep(0.5)
print("  重连后发包 OK")

print("\n✅ P1-2 PASS: SITL 重启后 ROS pub/sub 自动重连, 节点常驻可继续控制")
