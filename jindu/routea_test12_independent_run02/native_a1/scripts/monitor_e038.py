#!/usr/bin/env python3
"""E038 正式训练监测器（2026-08-03 创建，配套 logs/e038_local_3000.log）。

判据（E038 预注册，设计 v2.0 §4.6 + STATUS E037 段；G1 后 r_safety 已删）：
- KL < 1e-3                    → 塌缩（失败 A 分支）
- ep_sr 连续 >=10 updates <0.10 → 探针门判据（E038 风险登记：若仍 SR≈0 →
                                  问题在环境难度/稀疏性 → 转 F5/r_reach/reward_norm，
                                  不再改 shaping）
- r_prog 连 7 负               → "绕圈不完成"行为监控（r_prog≈0 + d_soft 稳 + SR 低）
- d_soft > 10.5                → 漂移行为监控（远离任务区）
- 进程死亡（无 TRAINING COMPLETE）→ 崩溃告警
- 每 100 steps 里程碑（save_freq=100 checkpoint 对齐）

E038 探针门（L4 教训）：前 30-60 updates 即判据 horizon，窗口内触发判据即 abort，
不默认跑满 3000 updates / ~95h。判据行以上述 VIOLATION/WATCH 输出为准。

用法：python scripts/monitor_e038.py [--poll 240] [--pid PID]
stdout 逐行事件 → 配 Monitor 工具（persistent）。
"""
import argparse
import os
import re
import subprocess
import sys
import time

LOG = "logs/e038_local_3000.log"
PID = 0  # 启动时以真实 python PID 覆盖（nvidia-smi/ps 为准，勿用 $! 包装 bash）

# stderr（synth 警告）无缓冲会插入 stdout 行中间 → 数值一律用数字字符集截断
NUM = r"[-+0-9.eE]+"
UPD_RE = re.compile(
    r"\[ *(\d+)/\d+\] a_loss=\S+ v_loss=\S+ kl=(" + NUM + r") cf=(" + NUM + r") nu=\S+ "
    r"r=\S+ logp=\S+ ep_sr=(" + NUM + r") N=\S+ steps=\S+ T=\S+s"
)
# E038: r_safe 列已删（F6）→ 只抓 r_prog / d_soft
COMP_RE = re.compile(
    r"\[comp\] r_prog=(" + NUM + r") r_smoo=\S+ d_soft=(" + NUM + r")"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=240)
    ap.add_argument("--pid", type=int, default=PID)
    ap.add_argument("--stall", type=int, default=1500,
                    help="超过 N 秒无新 UPD 且进程存活 → 停滞告警（python -u 后日志是实时的）")
    args = ap.parse_args()

    if args.pid <= 0:
        print("ERROR: --pid 必须为真实 python PID（ps/nvidia-smi 查询；0/$! 会令崩溃检测失效）", flush=True)
        sys.exit(2)

    last_step = 0
    last_comp_upto = 0  # 已配对处理到的 [comp] 行索引（每 update 恰一条，按日志顺序与 UPD 对齐）
    streak_bad_sr = 0
    rprog_hist = []  # (step, r_prog) 滑动窗口，供连负判据
    last_activity_ts = time.time()  # 最后"有新输出"时刻（UPD 或日志文件增长）
    last_stall_emit = 0.0
    last_size = -1

    while True:
        try:
            text = open(LOG, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            text = ""

        # 停滞判据按"无新输出"（设计 v2.0 §4.6）：日志文件增长也算活跃
        # （首轮 buffer 填充期无 UPD 属正常，不触发假 STALL）
        size = os.path.getsize(LOG) if os.path.exists(LOG) else 0
        if size != last_size:
            last_size = size
            last_activity_ts = time.time()

        ups = [(int(m[1]), float(m[2]), float(m[4])) for m in UPD_RE.finditer(text)]  # (step, kl, ep_sr)
        comps = [(float(m[1]), float(m[2])) for m in COMP_RE.finditer(text)]  # (r_prog, d_soft)

        new = [u for u in ups if u[0] > last_step]
        for i, (step, kl, epsr) in enumerate(new):
            # P1 修复（code-reviewer）：分量判据只在有新 update 时评估一次——
            # 旧实现每轮询重复 append 同一 (step, r_prog) → 单次负值 7 份拷贝假触发
            # "连 7 负"，且 d_soft WATCH 无新 update 也重复打印。comp 按索引配对，
            # 缺 comp 行（崩溃/交错）时跳过该次分量判据，索引照常推进自愈。
            if last_comp_upto + i < len(comps):
                rp, d = comps[last_comp_upto + i]
                rprog_hist.append((step, rp))
                rprog_hist = [(s, v) for s, v in rprog_hist if s > step - 2000]
                if d > 10.5:
                    print(f"WATCH step {step}: d_soft={d:.2f} > 10.5 (漂移行为)", flush=True)
                recent = [v for _, v in rprog_hist[-7:]]
                if len(recent) >= 7 and all(v < 0.0 for v in recent):
                    print(f"WATCH step {step}: r_prog 连 7 负 "
                          f"({recent[-1]:+.3f}) → 绕圈不完成信号", flush=True)
                    rprog_hist.clear()
            if kl < 1e-3:
                print(f"VIOLATION step {step}: KL={kl:.2e} < 1e-3 (collapse, 失败A)", flush=True)
            streak_bad_sr = streak_bad_sr + 1 if epsr < 0.10 else 0
            if streak_bad_sr >= 10:
                print(f"VIOLATION step {step}: ep_sr={epsr:.3f} <0.10 x{streak_bad_sr} "
                      f"(探针门判据 → abort，转 F5/r_reach/reward_norm)", flush=True)
            if step % 100 == 0:
                print(f"MILESTONE step {step}: kl={kl:.2e} ep_sr={epsr:.3f} (ckpt saved)", flush=True)
            last_step = step
            last_activity_ts = time.time()
        last_comp_upto += len(new)

        # 停滞检测：进程存活但超过 --stall 秒无任何新输出（每小时最多告警一次）
        idle = time.time() - last_activity_ts
        alive = subprocess.run(["kill", "-0", str(args.pid)], capture_output=True).returncode == 0
        if alive and idle > args.stall and time.time() - last_stall_emit > 3600:
            print(f"STALL: no output for {int(idle)}s (process alive, last UPD step {last_step})", flush=True)
            last_stall_emit = time.time()

        alive = subprocess.run(["kill", "-0", str(args.pid)], capture_output=True).returncode == 0
        if not alive:
            if "TRAINING COMPLETE" in text:
                print("E038 COMPLETE: 3000 updates done — paired eval + 判读", flush=True)
            elif "Traceback" in text:
                print("E038 CRASHED: traceback in log", flush=True)
            else:
                print("E038 PROCESS EXITED (unexpected)", flush=True)
            sys.exit(0)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
