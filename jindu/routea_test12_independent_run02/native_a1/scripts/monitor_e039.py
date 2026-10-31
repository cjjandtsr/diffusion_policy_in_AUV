#!/usr/bin/env python3
"""E039 正式训练监测器（2026-08-04 创建，配套 logs/e039_local_3000.log）。

判据（E039 设计 v1.1 §6 预注册；E038 教训：单信号 VIOLATION 会误杀校准重启）：

- 单独判负: KL < 1e-3（策略塌缩）——全窗有效
- 组合判负（漂移三要素，三要素同时成立才判负）:
    (i)  r_prog 连 7 负
    (ii) ep_sr 连 10 < 0.10
    (iii) d_soft 持续 > 10.5 且上升（机械定义: 最近 7 个 d_soft 全部 >10.5
          且窗口末值 > 窗口首值）
- 判据覆盖 0-200 全衰减窗（漂移 onset 实测 step 94，探针门不得止于 60）
- 校准重启区分规则只适用前 60 updates（§6.2）: d_soft 短暂下降（接近目标区）/
  ep_sr 波动（尝试更难场景）为预期信号 → 豁免组合判负的对应分量
  （0-60 内组合判负退化为 r_prog 连 7 负单分量）
- F5 交互（§4.3）: 日志出现 "[F5] ACTIVATED (E039b)" → ep_sr 分量判据移交
  F5 自身机制接管（组合判负退化为 r_prog + d_soft 两分量）
- 停止判据参考: 首个 500-ckpt paired eval 起 SR 连续 3 个 eval 不涨（外部 eval
  执行，monitor 只在 500 步里程碑提示）

用法：python scripts/monitor_e039.py [--poll 240] [--pid PID]
stdout 逐行事件 → 配 Monitor 工具（persistent）。
"""
import argparse
import os
import re
import subprocess
import sys
import time

LOG = "logs/e039_local_3000.log"
PID = 0  # 启动时以真实 python PID 覆盖（nvidia-smi/ps 为准，勿用 $! 包装 bash）

# stderr（synth 警告）无缓冲会插入 stdout 行中间 → 数值一律用数字字符集截断
NUM = r"[-+0-9.eE]+"
# E039 新格式: 行尾追加 w_bc=... f5=...（可选组，兼容旧格式行）
UPD_RE = re.compile(
    r"\[ *(\d+)/\d+\] a_loss=\S+ v_loss=\S+ kl=(" + NUM + r") cf=(" + NUM + r") nu=\S+ "
    r"r=\S+ logp=\S+ ep_sr=(" + NUM + r") N=\S+ steps=\S+ T=\S+s"
    r"(?: w_bc=(" + NUM + r") f5=(\S+))?"
)
COMP_RE = re.compile(
    r"\[comp\] r_prog=(" + NUM + r") r_smoo=\S+ d_soft=(" + NUM + r")"
)
F5_RE = re.compile(r"\[F5\] ACTIVATED")

# 组合判负窗口（预注册 §6.1）
W_RPROG = 7   # r_prog 连 7 负
W_EPSR = 10   # ep_sr 连 10 < 0.10
W_DSOFT = 7   # d_soft 全部 >10.5 且末值 > 首值
D_SOFT_LINE = 10.5
KL_COLLAPSE = 1e-3
# 校准重启区分规则只适用前 60 updates（§6.2）
CALIB_WINDOW = 60


def combo_violation(step: int, f5_active: bool,
                    rprog_hist, epsr_hist, dsoft_hist) -> str:
    """三要素组合判负。返回触发描述字符串；未触发返回空串。

    0-60: 豁免 d_soft + ep_sr 分量（校准重启区分规则）→ 只查 r_prog。
    F5 激活后: ep_sr 分量移交 F5 机制 → 只查 r_prog + d_soft。
    """
    rp_ok = (len(rprog_hist) >= W_RPROG
             and all(v < 0.0 for _, v in rprog_hist[-W_RPROG:]))
    eps_ok = (len(epsr_hist) >= W_EPSR
              and all(e < 0.10 for _, e in epsr_hist[-W_EPSR:]))
    ds_ok = (len(dsoft_hist) >= W_DSOFT
             and all(d > D_SOFT_LINE for _, d in dsoft_hist[-W_DSOFT:])
             and dsoft_hist[-1][1] > dsoft_hist[-W_DSOFT][1])
    if step <= CALIB_WINDOW:
        # 校准重启区分规则（§6.2）: d_soft 短暂下降 / ep_sr 波动为预期信号
        if rp_ok:
            return (f"r_prog 连 {W_RPROG} 负 "
                    f"({rprog_hist[-1][1]:+.3f}) [0-{CALIB_WINDOW} 窗，d_soft/ep_sr 豁免]")
        return ""
    if f5_active:
        if rp_ok and ds_ok:
            return (f"r_prog 连 {W_RPROG} 负 + d_soft 持续 >{D_SOFT_LINE} 上升 "
                    f"(F5 接管 ep_sr 分量)")
        return ""
    if rp_ok and eps_ok and ds_ok:
        return (f"r_prog 连 {W_RPROG} 负 + ep_sr 连 {W_EPSR} <0.10 "
                f"+ d_soft 持续 >{D_SOFT_LINE} 上升")
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=240)
    ap.add_argument("--pid", type=int, default=PID)
    ap.add_argument("--stall", type=int, default=1500,
                    help="超过 N 秒无新输出且进程存活 → 停滞告警（python -u 后日志是实时的）")
    args = ap.parse_args()

    if args.pid <= 0:
        print("ERROR: --pid 必须为真实 python PID（ps/nvidia-smi 查询；0/$! 会令崩溃检测失效）", flush=True)
        sys.exit(2)

    last_step = 0
    last_comp_upto = 0  # 已配对处理到的 [comp] 行索引（每 update 恰一条，按日志顺序与 UPD 对齐）
    rprog_hist = []  # (step, r_prog) 滑动窗口（连负判据）
    epsr_hist = []   # (step, ep_sr) 滑动窗口（失败主导批次判据）
    dsoft_hist = []  # (step, d_soft) 滑动窗口（漂移行为判据）
    f5_active = False
    wbc_off_reported = False
    last_activity_ts = time.time()  # 最后"有新输出"时刻（UPD 或日志文件增长）
    last_stall_emit = 0.0
    last_size = -1

    while True:
        try:
            text = open(LOG, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            text = ""

        # 停滞判据按"无新输出"（E038 同规格）：日志文件增长也算活跃
        # （首轮 buffer 填充期无 UPD 属正常，不触发假 STALL）
        size = os.path.getsize(LOG) if os.path.exists(LOG) else 0
        if size != last_size:
            last_size = size
            last_activity_ts = time.time()

        if F5_RE.search(text) and not f5_active:
            f5_active = True
            print("F5 ACTIVE (E039b): ep_sr 分量判据移交 F5 自身机制接管"
                  "（组合判负 = r_prog + d_soft 两分量）", flush=True)

        # 组: 1=step 2=kl 3=cf 4=ep_sr 5=w_bc(可选) 6=f5(可选)
        ups = [(int(m[1]), float(m[2]), float(m[4]), (m[6] or ""), m[5])
               for m in UPD_RE.finditer(text)]  # (step, kl, ep_sr, f5, w_bc)
        comps = [(float(m[1]), float(m[2])) for m in COMP_RE.finditer(text)]  # (r_prog, d_soft)

        new = [u for u in ups if u[0] > last_step]
        for i, (step, kl, epsr, f5tag, wbc) in enumerate(new):
            # 分量数据只在有新 update 时按索引配对（E038 code-reviewer P1 教训:
            # 重复 append 同一值会造成假触发）
            if last_comp_upto + i < len(comps):
                rp, d = comps[last_comp_upto + i]
                rprog_hist.append((step, rp))
                epsr_hist.append((step, epsr))
                dsoft_hist.append((step, d))
                rprog_hist = [(s, v) for s, v in rprog_hist if s > step - 2000]
                epsr_hist = [(s, v) for s, v in epsr_hist if s > step - 2000]
                dsoft_hist = [(s, v) for s, v in dsoft_hist if s > step - 2000]
                # WATCH（信息性，单分量）
                if d > D_SOFT_LINE:
                    print(f"WATCH step {step}: d_soft={d:.2f} > {D_SOFT_LINE} (漂移行为)", flush=True)
                rp_recent = [v for _, v in rprog_hist[-W_RPROG:]]
                if len(rp_recent) >= W_RPROG and all(v < 0.0 for v in rp_recent):
                    print(f"WATCH step {step}: r_prog 连 {W_RPROG} 负 "
                          f"({rp_recent[-1]:+.3f}) → 绕开/后退信号", flush=True)
                eps_recent = [v for _, v in epsr_hist[-W_EPSR:]]
                if len(eps_recent) >= W_EPSR and all(v < 0.10 for v in eps_recent):
                    print(f"WATCH step {step}: ep_sr 连 {W_EPSR} <0.10 "
                          f"({eps_recent[-1]:.3f}) → 失败主导批次", flush=True)
                # VIOLATION（组合判负，预注册 §6.1/§6.2）
                viol = combo_violation(step, f5_active, rprog_hist, epsr_hist, dsoft_hist)
                if viol:
                    print(f"VIOLATION step {step}: {viol} → abort 判读（判负路径 §3.5）",
                          flush=True)
            if kl < KL_COLLAPSE and f5tag != "skip":
                # f5=skip 更新不重算 logp → approx_kl=0.0 恒打印，非塌缩（F5 假阳性防护）
                print(f"VIOLATION step {step}: KL={kl:.2e} < {KL_COLLAPSE:.0e} "
                      f"(collapse, 单独判负)", flush=True)
            if step % 100 == 0:
                print(f"MILESTONE step {step}: kl={kl:.2e} ep_sr={epsr:.3f} "
                      f"f5={f5tag or 'off'}"
                      + ("" if step < 500 else " (500-ckpt 起: SR 连 3 eval 不涨 → 停训判据启动)"),
                      flush=True)
            if not wbc_off_reported and wbc and float(wbc) == 0.0:
                wbc_off_reported = True
                print(f"MILESTONE step {step}: w_bc=0.0000 (T_decay=200 衰减完成，"
                      f"锚定期结束 → 放手期；KL 窗 + clip 维持)", flush=True)
            last_step = step
            last_activity_ts = time.time()
        last_comp_upto += len(new)

        # 停滞检测：进程存活但超过 --stall 秒无任何新输出（每小时最多告警一次）
        idle = time.time() - last_activity_ts
        alive = subprocess.run(["kill", "-0", str(args.pid)], capture_output=True).returncode == 0
        if alive and idle > args.stall and time.time() - last_stall_emit > 3600:
            print(f"STALL: no output for {int(idle)}s (process alive, last UPD step {last_step})",
                  flush=True)
            last_stall_emit = time.time()

        alive = subprocess.run(["kill", "-0", str(args.pid)], capture_output=True).returncode == 0
        if not alive:
            if "TRAINING COMPLETE" in text:
                print("E039 COMPLETE: 3000 updates done — paired eval + 行为学 + 判读", flush=True)
            elif "Traceback" in text:
                print("E039 CRASHED: traceback in log", flush=True)
            else:
                print("E039 PROCESS EXITED (unexpected)", flush=True)
            sys.exit(0)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
