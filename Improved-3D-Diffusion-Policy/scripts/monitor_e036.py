#!/usr/bin/env python3
"""E036 正式训练监测器（2026-08-03 创建，配套 logs/e036_local_3000.log）。

判据（来自 H5 预注册，notes.md / STATUS.md 2026-08-03 段）：
- KL < 1e-3                    → 塌缩（失败 A 分支，噪声控制 H5b）
- ep_sr 连续 >=10 updates <0.10 → H2 触发（w_safety 重平衡）
- d_soft > 10.5                → 吸引子漂移警戒（probe 尺度漂移伴随 d_soft 7-11）
- 进程死亡（无 TRAINING COMPLETE）→ 崩溃告警
- 每 500 steps 里程碑（save_freq checkpoint 对齐）

用法：python scripts/monitor_e036.py [--poll 240] [--pid PID]
stdout 逐行事件 → 配 Monitor 工具（persistent）。
"""
import argparse
import re
import subprocess
import sys
import time

LOG = "logs/e036_local_3000.log"
PID = 224703  # 2026-08-03 无缓冲重启（PYTHONUNBUFFERED=1）；旧 141455 块缓冲弃用；注意 $! 可能捕获包装 bash，实际 python PID 以 nvidia-smi/ps 为准

# stderr（synth 警告）无缓冲会插入 stdout 行中间 → 数值一律用数字字符集截断
NUM = r"[-+0-9.eE]+"
UPD_RE = re.compile(
    r"\[ *(\d+)/\d+\] a_loss=\S+ v_loss=\S+ kl=(" + NUM + r") cf=(" + NUM + r") nu=\S+ "
    r"r=\S+ logp=\S+ ep_sr=(" + NUM + r") N=\S+ steps=\S+ T=\S+s"
)
COMP_RE = re.compile(r"\[comp\] r_prog=\S+ r_safe=\S+ r_smoo=\S+ d_soft=(" + NUM + r")")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=240)
    ap.add_argument("--pid", type=int, default=PID)
    ap.add_argument("--stall", type=int, default=1500,
                    help="超过 N 秒无新 UPD 且进程存活 → 停滞告警（python -u 后日志是实时的）")
    args = ap.parse_args()

    last_step = 0
    streak_bad_sr = 0
    last_new_ts = time.time()
    last_stall_emit = 0.0

    while True:
        try:
            text = open(LOG, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            text = ""

        ups = [(int(m[1]), float(m[2]), float(m[4])) for m in UPD_RE.finditer(text)]  # (step, kl, ep_sr)
        comps = [float(m[1]) for m in COMP_RE.finditer(text)]  # d_soft (唯一捕获组)

        new = [u for u in ups if u[0] > last_step]
        for step, kl, epsr in new:
            if kl < 1e-3:
                print(f"VIOLATION step {step}: KL={kl:.2e} < 1e-3 (collapse, 失败A)", flush=True)
            streak_bad_sr = streak_bad_sr + 1 if epsr < 0.10 else 0
            if streak_bad_sr >= 10:
                print(f"VIOLATION step {step}: ep_sr={epsr:.3f} <0.10 x{streak_bad_sr} (H2 trigger)", flush=True)
            if step % 500 == 0:
                print(f"MILESTONE step {step}: kl={kl:.2e} ep_sr={epsr:.3f} (checkpoint saved)", flush=True)
            last_step = step
            last_new_ts = time.time()

        if comps and last_step:
            d = comps[-1]
            if d > 10.5:
                print(f"WATCH step {last_step}: d_soft={d:.2f} > 10.5 (attractor drift)", flush=True)

        # 停滞检测：进程存活但超过 --stall 秒无新 UPD（每小时最多告警一次）
        idle = time.time() - last_new_ts
        alive = subprocess.run(["kill", "-0", str(args.pid)], capture_output=True).returncode == 0
        if alive and idle > args.stall and time.time() - last_stall_emit > 3600:
            print(f"STALL: no new UPD for {int(idle)}s (process alive, last step {last_step})", flush=True)
            last_stall_emit = time.time()

        alive = subprocess.run(["kill", "-0", str(args.pid)], capture_output=True).returncode == 0
        if not alive:
            if "TRAINING COMPLETE" in text:
                print("E036 COMPLETE: 3000 updates done — run_eval + 判读", flush=True)
            elif "Traceback" in text:
                print("E036 CRASHED: traceback in log", flush=True)
            else:
                print("E036 PROCESS EXITED (unexpected)", flush=True)
            sys.exit(0)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
