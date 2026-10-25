#!/usr/bin/env python3
"""E040 正式训练监测器（2026-08-04 创建，配套 logs/e040_local_3000.log）。

判据（E040 设计 v1.0 §6 预注册；波动免疫原则——持续性签名判负，单点波动不判负）：

- KL 塌缩单独判负: **连 2 批** KL < 1e-3 —— E038 step 89 单点 dip 6.29e-4 回弹实证
  （健康训练中真实发生），单点触发会误杀；E039 签名首批高于阈值（S3 1.46e-3 →
  S6 4.5e-4 → S9 3.3e-4），连 3 批会漏判 → 连 2 批覆盖 E031/E038/E039 全历史场景
- 组合判负（漂移三要素，三要素同时成立才判负）:
    (i)  r_prog 连 7 负
    (ii) ep_sr 连 10 < 0.10
    (iii) d_soft 持续 > 10.5 且上升（最近 7 个全部 >10.5 且末值 > 首值）
- learn-to-crash 判负（E040 新增，[crash] 行数据源）: 最近 5 批滚动——
    碰撞率 = Σn_coll/Σn_eps > 0.633（BC 基线 0.422 + 裕量 max(0.05, 0.5×0.422)）
    且 平均碰撞步数 = Σ(n_coll×t_c_mean)/Σn_coll < 20（按碰撞数加权，无碰撞批不拉低）
    单批 16 eps 碰撞率噪声 σ≈0.05+ → 必须滚动，单批判定会假触发
- **判负确认期（用户裁决 2026-08-04 补充）：所有判负信号须连续成立 >100 步
  （updates）才确认判负，不足 100 步一律视为波动**——KL / 三要素 / learn-to-crash
  三类信号各自计数：每连续一个 update 判据成立 +1，回弹（判据不成立）清零；
  计数 ≥100 才打印 VIOLATION；确认期每 25 步报告进度
- 0-60 校准豁免（§6.2）: d_soft 短暂下降 / ep_sr 波动 / 碰撞率上升（探索重启
  代价）为预期信号 → 组合判负退化为 r_prog 单分量；learn-to-crash 0-60 不判负；
  **KL 不豁免**（塌缩可在任何步发生）
- 判据覆盖 0-200 全衰减窗（漂移 onset 实测 step 94，探针门不得止于 60）
- 停止判据参考: 首个 500-ckpt paired eval 起 SR 连续 3 个 eval 不涨（外部 eval
  执行，monitor 只在 500 步里程碑提示）

用法：python scripts/monitor_e040.py [--poll 240] [--pid PID]
stdout 逐行事件 → 配 Monitor 工具（persistent）。
"""
import argparse
import os
import re
import subprocess
import sys
import time

LOG = "logs/e040_local_3000.log"
PID = 0  # 启动时以真实 python PID 覆盖（nvidia-smi/ps 为准，勿用 $! 包装 bash）

# stderr（synth 警告）无缓冲会插入 stdout 行中间 → 数值一律用数字字符集截断
NUM = r"[-+0-9.eE]+"
# E038/E040 格式（无 w_bc/f5 字段；E040 副本不含 anchor/F5）
UPD_RE = re.compile(
    r"\[ *(\d+)/\d+\] a_loss=\S+ v_loss=\S+ kl=(" + NUM + r") cf=(" + NUM + r") nu=\S+ "
    r"r=\S+ logp=\S+ ep_sr=(" + NUM + r") N=\S+ steps=\S+ T=\S+s"
)
COMP_RE = re.compile(
    r"\[comp\] r_prog=(" + NUM + r") r_smoo=\S+ d_soft=(" + NUM + r")"
)
CRASH_RE = re.compile(
    r"\[crash\] n_coll=(\d+) n_eps=(\d+) t_c_mean=(" + NUM + r")"
)

# 组合判负窗口（预注册 §6.1）
W_RPROG = 7   # r_prog 连 7 负
W_EPSR = 10   # ep_sr 连 10 < 0.10
W_DSOFT = 7   # d_soft 全部 >10.5 且末值 > 首值
D_SOFT_LINE = 10.5
# KL 塌缩: 连 2 批 <1e-3（波动免疫——E038 step 89 单点 dip 6.29e-4 回弹实证排除
# 单批触发；E039 签名 S3 1.46e-3→S6 4.5e-4→S9 3.3e-4 首批高于阈值，连 3 批会漏判；
# E031 持续锁定 1e-4 级 → 连 2 批覆盖全部历史场景）
W_KL = 2
KL_COLLAPSE = 1e-3
# learn-to-crash（预注册 §6.1，诊断报告 §4）: 最近 5 批滚动
CRASH_WINDOW = 5
CRASH_RATE_LINE = 0.633   # BC 基线 0.422 + 裕量 max(0.05, 0.5×0.422)=0.211（0.453 是 E038 step100 诊断率，非基线）
CRASH_TC_LINE = 20.0      # 平均碰撞步数 < 20（早期终止 = 自尽 hacks）
# 校准重启区分规则只适用前 60 updates（§6.2）
CALIB_WINDOW = 60
# 判负确认期（用户裁决 2026-08-04）：信号须连续成立 >100 步才确认判负，
# 不足即视为波动；回弹清零。确认期进度每 25 步报告一次。
CONFIRM_STEPS = 100
CONFIRM_REPORT_EVERY = 25


def combo_violation(step: int, rprog_hist, epsr_hist, dsoft_hist) -> str:
    """三要素组合判负。返回触发描述字符串；未触发返回空串。

    0-60: 豁免 d_soft + ep_sr 分量（校准重启区分规则）→ 只查 r_prog。
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
    if rp_ok and eps_ok and ds_ok:
        return (f"r_prog 连 {W_RPROG} 负 + ep_sr 连 {W_EPSR} <0.10 "
                f"+ d_soft 持续 >{D_SOFT_LINE} 上升")
    return ""


def learn_to_crash_violation(step: int, crash_hist) -> str:
    """learn-to-crash 判负（§6.1 第三条）。返回触发描述字符串；未触发返回空串。

    滚动窗口（最近 CRASH_WINDOW 批）: 碰撞率按批数加权，t_c 按碰撞数加权——
    无碰撞批的 t_c_mean=0.0 不拉低平均碰撞步数。0-60 窗豁免（探索重启代价）。
    """
    if step <= CALIB_WINDOW or len(crash_hist) < CRASH_WINDOW:
        return ""
    n_coll = sum(c[1] for c in crash_hist[-CRASH_WINDOW:])
    n_eps = sum(c[2] for c in crash_hist[-CRASH_WINDOW:])
    if n_coll == 0:
        return ""
    rate = n_coll / n_eps
    tc_w = sum(c[1] * c[3] for c in crash_hist[-CRASH_WINDOW:]) / n_coll
    if rate > CRASH_RATE_LINE and tc_w < CRASH_TC_LINE:
        return (f"碰撞率 {rate:.3f} > {CRASH_RATE_LINE:.3f} "
                f"且 平均碰撞步数 {tc_w:.1f} < {CRASH_TC_LINE:.0f} (learn-to-crash)")
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
    last_comp_upto = 0   # 已配对处理到的 [comp] 行索引（每 update 恰一条，按日志顺序与 UPD 对齐）
    last_crash_upto = 0  # 已配对处理到的 [crash] 行索引（[crash] 行独立于 [comp] 行）
    rprog_hist = []  # (step, r_prog) 滑动窗口（连负判据）
    epsr_hist = []   # (step, ep_sr) 滑动窗口（失败主导批次判据）
    dsoft_hist = []  # (step, d_soft) 滑动窗口（漂移行为判据）
    kl_hist = []     # (step, kl) 滑动窗口（KL 连 2 批塌缩判据）
    crash_hist = []  # (step, n_coll, n_eps, t_c_mean) 滑动窗口（learn-to-crash 判据）
    # 判负确认期状态（须连续成立 >CONFIRM_STEPS 步才确认判负，回弹清零）
    combo_cnt = 0; combo_desc = ""
    crash_cnt = 0; crash_desc = ""
    kl_cnt = 0;    kl_desc = ""
    combo_fired = crash_fired = kl_fired = False
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

        # 组: 1=step 2=kl 3=cf 4=ep_sr
        ups = [(int(m[1]), float(m[2]), float(m[4]))
               for m in UPD_RE.finditer(text)]  # (step, kl, ep_sr)
        comps = [(float(m[1]), float(m[2])) for m in COMP_RE.finditer(text)]  # (r_prog, d_soft)
        crashes = [(int(m[1]), int(m[2]), float(m[3])) for m in CRASH_RE.finditer(text)]

        new = [u for u in ups if u[0] > last_step]
        for i, (step, kl, epsr) in enumerate(new):
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
                # 组合判负（预注册 §6.1/§6.2 + 100 步确认期，用户裁决 2026-08-04）
                viol = combo_violation(step, rprog_hist, epsr_hist, dsoft_hist)
                if viol and not combo_fired:
                    combo_cnt += 1; combo_desc = viol
                    if combo_cnt == 1:
                        print(f"WATCH step {step}: 组合判负信号进入确认期"
                              f"（{viol}）→ 须连续 {CONFIRM_STEPS} 步确认，不足视为波动",
                              flush=True)
                    elif combo_cnt % CONFIRM_REPORT_EVERY == 0:
                        print(f"WATCH step {step}: 组合判负确认中 {combo_cnt}/{CONFIRM_STEPS}",
                              flush=True)
                    if combo_cnt >= CONFIRM_STEPS:
                        combo_fired = True
                        print(f"VIOLATION step {step}: {combo_desc}"
                              f"（连续 {combo_cnt} 步确认，非波动）→ abort 判读（判负路径 §3.5）",
                              flush=True)
                elif combo_cnt:
                    print(f"WATCH step {step}: 组合判负信号回弹（{combo_cnt} 步后）"
                          f"→ 视为波动，确认期重置", flush=True)
                    combo_cnt = 0; combo_desc = ""
            # [crash] 行独立配对（数据源: train_ddpm_ppo_meanlogp_3_e040.py）
            if last_crash_upto + i < len(crashes):
                nc, ne, tc = crashes[last_crash_upto + i]
                crash_hist.append((step, nc, ne, tc))
                crash_hist = [c for c in crash_hist if c[0] > step - 2000]
                ltc = learn_to_crash_violation(step, crash_hist)
                if ltc and not crash_fired:
                    crash_cnt += 1; crash_desc = ltc
                    if crash_cnt == 1:
                        print(f"WATCH step {step}: learn-to-crash 信号进入确认期"
                              f"（{ltc}）→ 须连续 {CONFIRM_STEPS} 步确认，不足视为波动",
                              flush=True)
                    elif crash_cnt % CONFIRM_REPORT_EVERY == 0:
                        print(f"WATCH step {step}: learn-to-crash 确认中 {crash_cnt}/{CONFIRM_STEPS}",
                              flush=True)
                    if crash_cnt >= CONFIRM_STEPS:
                        crash_fired = True
                        print(f"VIOLATION step {step}: {crash_desc}"
                              f"（连续 {crash_cnt} 步确认，非波动）→ abort 判读（判负路径 §3.5）",
                              flush=True)
                elif crash_cnt:
                    print(f"WATCH step {step}: learn-to-crash 信号回弹（{crash_cnt} 步后）"
                          f"→ 视为波动，确认期重置", flush=True)
                    crash_cnt = 0; crash_desc = ""
                else:
                    # 滚动窗口近触发线预警（防事件洪泛 + 不敏感原则：per-batch
                    # 碰撞观测每批必打 3000 条 → 只在滚动 5 批贴近触发线时 WATCH；
                    # 预警线 = 触发线缓冲：rate>0.58（线 0.633）或 t_c<22（线 20））
                    w = crash_hist[-CRASH_WINDOW:]
                    if len(w) == CRASH_WINDOW:
                        nc_w = sum(c[1] for c in w)
                        ne_w = sum(c[2] for c in w)
                        if nc_w > 0:
                            rate_w = nc_w / ne_w
                            tc_w = sum(c[1] * c[3] for c in w) / nc_w
                            if rate_w > 0.58 or tc_w < 22:
                                print(f"WATCH step {step}: 滚动{CRASH_WINDOW}批 "
                                      f"碰撞率 {rate_w:.3f} (线{CRASH_RATE_LINE:.3f}) "
                                      f"t_c {tc_w:.1f} (线<{CRASH_TC_LINE:.0f}) 近触发线",
                                      flush=True)
            # KL 塌缩: 连 2 批 <1e-3 为信号（波动免疫；0-60 窗 KL 不豁免）
            # + 100 步确认期（用户裁决 2026-08-04）：信号须连续成立 >100 步才判负
            kl_hist.append((step, kl))
            kl_hist = [(s, k) for s, k in kl_hist if s > step - 2000]
            kl_sig = len(kl_hist) >= W_KL and all(k < KL_COLLAPSE for _, k in kl_hist[-W_KL:])
            if kl_sig and not kl_fired:
                kl_cnt += 1
                if kl_cnt == 1:
                    kl_desc = (f"KL 连 {W_KL} 批 < {KL_COLLAPSE:.0e} "
                               f"({kl_hist[-W_KL][1]:.2e}→{kl:.2e})")
                    print(f"WATCH step {step}: KL 塌缩信号进入确认期（{kl_desc}）"
                          f"→ 须连续 {CONFIRM_STEPS} 步确认，不足视为波动", flush=True)
                elif kl_cnt % CONFIRM_REPORT_EVERY == 0:
                    print(f"WATCH step {step}: KL 塌缩确认中 {kl_cnt}/{CONFIRM_STEPS}"
                          f" (kl={kl:.2e})", flush=True)
                if kl_cnt >= CONFIRM_STEPS:
                    kl_fired = True
                    print(f"VIOLATION step {step}: {kl_desc}"
                          f"（连续 {kl_cnt} 步确认，非波动；collapse, 单独判负）"
                          f"→ abort 判读（判负路径 §3.5）", flush=True)
            elif kl_cnt:
                print(f"WATCH step {step}: KL 信号回弹（{kl_cnt} 步后, kl={kl:.2e}）"
                      f"→ 视为波动，确认期重置", flush=True)
                kl_cnt = 0; kl_desc = ""
            elif kl < KL_COLLAPSE:
                print(f"WATCH step {step}: KL={kl:.2e} < {KL_COLLAPSE:.0e} 单批 "
                      f"（待 {W_KL} 连确认，防 E038 step 89 式单点 dip 误杀）", flush=True)
            if step % 100 == 0:
                print(f"MILESTONE step {step}: kl={kl:.2e} ep_sr={epsr:.3f}"
                      + ("" if step < 500 else " (500-ckpt 起: SR 连 3 eval 不涨 → 停训判据启动)"),
                      flush=True)
            last_step = step
            last_activity_ts = time.time()
        last_comp_upto += len(new)
        # 配对自愈（P2-1）：advance 不越过已读 [crash] 行数——若某轮 poll 恰落在
        # UPD 打印与 [crash] 打印之间（flush 边界），缺行跳过，下一轮索引自然补齐，
        # 不会永久错位（E038 缺行自愈同款，code-reviewer P2）
        last_crash_upto += min(len(new), len(crashes) - last_crash_upto)

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
                print("E040 COMPLETE: 3000 updates done — paired eval + 行为学 + 判读", flush=True)
            elif "Traceback" in text:
                print("E040 CRASHED: traceback in log", flush=True)
            else:
                print("E040 PROCESS EXITED (unexpected)", flush=True)
            sys.exit(0)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
