#!/usr/bin/env python3
"""E040 阶段 0 诊断分析：ΔV 网格 + 校准临界值 + 决策树结论。

输入：probe_e040_diag.py 输出的 diag_{ckpt}_seed{s}.json（e038_step100 + bc）
输出：outputs/e040_diag/analyze_report.json + console 摘要

计算（设计 program/sheji/E040_RiskRewardCalibration_Design_v1_0.md §4；
P2-1/P2-4 定稿）：
- per-step 净收益 per_t = (progress_terms_t − terminal_now_t) + smooth_t
  （terminal_now：最后步 reached→+r_reach_now / collision→+r_collision_now，当前常量；
  非末步为 0）
- V_i(rc, rr) = Σ_t γ^t·per_t + rr·γ^T·I(reached_i) + rc·γ^t_c·I(collision_i)
  ——terminal 参数化，逐轨迹 V 再平均（P2-4）
- 分帧（P1-1）：approach 段（穿障推进）/ drift 段（开阔地驻留）/ other；
  穿障类 = 含 approach 段的 episode；漂移类 = 无 approach 且含 drift 段
- ΔV(rc, rr) = E[V_漂移类](rc) − E[V_穿障类](rc, rr)
- 裕量判据（P2-1）：ΔV ≤ −0.2·|E[V_漂移类]|（收益差 20% 相对余量）
- 决策树（§5）：路径 A 固定 rr=100 降权 rc∈(−20,0]；路径 B 固定 rc=−20 强化 rr∈{150,200}；互斥

用法：
  python scripts/analyze_e040_diag.py [--diag-dir outputs/e040_diag] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

GAMMA = 0.99
# 决策网格（E040 设计 §4/§5）
RC_GRID = [-50.0, -40.0, -30.0, -20.0, -15.0, -10.0, -5.0, 0.0]
RR_GRID = [100.0, 150.0, 200.0]
RC_CURRENT = -20.0   # E039 校准后当前值（决策树从当前值出发）
RR_CURRENT = 100.0
# 分帧参数（P1-1）
TOL = 0.05        # Δdist 阈值（m/步）
MIN_RUN = 3       # 最小连续步数
OPEN_THRESH = 8.0 # d_soft 开阔地阈值（m，与漂移判据 10.5 接近但宽松——探测段判定）
# 裕量（P2-1）
MARGIN_FRAC = 0.2


def segment_behavior(dist: list, d_soft: list, tol: float = TOL,
                     min_run: int = MIN_RUN, open_thresh: float = OPEN_THRESH) -> list:
    """per-step 行为段分类。t=1..n-1 步的标签：
    approach：Δ_t = dist[t-1]−dist[t] ≥ tol（穿障推进）
    drift：|Δ_t| ≤ tol 且 d_soft[t] > open_thresh（开阔地驻留）
    other：其余（含向远离目标移动）
    连续同类步合并为段；run < min_run 的段降级为 other。
    返回 [{"type": ..., "steps": [t..]}, ...]（steps 是 t≥1 的步索引）。"""
    n = len(dist)
    if n < 2:
        return []
    labels = ["other"] * n
    for t in range(1, n):
        delta = dist[t - 1] - dist[t]
        if delta >= tol:
            labels[t] = "approach"
        elif abs(delta) <= tol and d_soft[t] > open_thresh:
            labels[t] = "drift"
        else:
            labels[t] = "other"
    segs = []
    cur_type, cur_start = labels[1], 1
    for t in range(2, n + 1):
        cur_label = labels[t] if t < n else None
        if cur_label != cur_type:
            if t - cur_start >= min_run:
                segs.append({"type": cur_type, "steps": list(range(cur_start, t))})
            else:
                segs.append({"type": "other", "steps": list(range(cur_start, t))})
            cur_type, cur_start = cur_label, t if t < n else t
    return segs


def load_diag(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def episode_value(ep: dict, r_collision: float, r_reach: float,
                  r_reach_now: float, r_collision_now: float) -> float:
    """逐轨迹 V（P2-4）：净收益 + 参数化 terminal。γ^0 给第一步（与 GAE 同向）。"""
    recs = ep["records"]
    n = len(recs)
    v = 0.0
    for t, r in enumerate(recs):
        terminal_now = 0.0
        if t == n - 1:
            if ep["outcome"] == "reached":
                terminal_now = r_reach_now
            elif ep["outcome"] == "collision":
                terminal_now = r_collision_now
        per_t = (r["progress_terms"] - terminal_now) + r["smooth"]
        v += (GAMMA ** t) * per_t
    if ep["outcome"] == "reached":
        v += r_reach * (GAMMA ** (n - 1))
    elif ep["outcome"] == "collision":
        v += r_collision * (GAMMA ** ep["t_c"])
    return v


def per_ckpt_stats(data: dict, ckpt_tag: str) -> dict:
    eps = data["episodes"]
    cfg = data["config"]
    r_reach_now, r_collision_now = cfg["r_reach_now"], cfg["r_collision_now"]
    n = len(eps)
    outcomes = {"reached": 0, "collision": 0, "timeout": 0}
    t_c_list: list = []
    approach_eps, drift_eps, other_eps = 0, 0, 0
    approach_outcomes = {"reached": 0, "collision": 0, "timeout": 0}
    seg_returns = {"approach": [], "drift": []}

    for ep in eps:
        outcomes[ep["outcome"]] += 1
        if ep["t_c"] is not None:
            t_c_list.append(ep["t_c"])
        recs = ep["records"]
        dist = [r["dist"] for r in recs]
        d_soft = [r["d_soft_h"] for r in recs]
        segs = segment_behavior(dist, d_soft)
        types = {s["type"] for s in segs}
        has_approach = "approach" in types
        has_drift = "drift" in types
        if has_approach:
            approach_eps += 1
            approach_outcomes[ep["outcome"]] += 1
        elif has_drift:
            drift_eps += 1
        else:
            other_eps += 1
        # 段级收益（P1-1 参考量）
        for s in segs:
            if s["type"] in ("approach", "drift"):
                t0 = s["steps"][0]
                ret = 0.0
                for t in s["steps"]:
                    terminal_now = 0.0
                    if t == len(recs) - 1 and ep["outcome"] == "reached":
                        terminal_now = r_reach_now
                    elif t == len(recs) - 1 and ep["outcome"] == "collision":
                        terminal_now = r_collision_now
                    ret += (GAMMA ** (t - t0)) * ((recs[t]["progress_terms"] - terminal_now) + recs[t]["smooth"])
                seg_returns[s["type"]].append(ret)

    return {
        "ckpt": ckpt_tag,
        "n_eps": n,
        "outcomes": outcomes,
        "collision_rate": outcomes["collision"] / n if n else 0.0,
        "t_c_list": t_c_list,
        "t_c_mean": float(np.mean(t_c_list)) if t_c_list else None,
        "t_c_min": min(t_c_list) if t_c_list else None,
        "class_eps": {"approach": approach_eps, "drift": drift_eps, "other": other_eps},
        "approach_outcomes": approach_outcomes,
        "segment_returns": {
            "approach": {"n": len(seg_returns["approach"]),
                         "mean": float(np.mean(seg_returns["approach"])) if seg_returns["approach"] else None},
            "drift": {"n": len(seg_returns["drift"]),
                      "mean": float(np.mean(seg_returns["drift"])) if seg_returns["drift"] else None},
        },
    }


def seg_return(ep: dict, steps: list, r_reach_now: float, r_collision_now: float) -> float:
    """段级净收益（不含 terminal）：段内步相对段起点折扣。"""
    recs = ep["records"]
    t0 = steps[0]
    ret = 0.0
    for t in steps:
        terminal_now = 0.0
        if t == len(recs) - 1:
            if ep["outcome"] == "reached":
                terminal_now = r_reach_now
            elif ep["outcome"] == "collision":
                terminal_now = r_collision_now
        ret += (GAMMA ** (t - t0)) * ((recs[t]["progress_terms"] - terminal_now) + recs[t]["smooth"])
    return ret


def delta_v_grid(eps: list, r_reach_now: float, r_collision_now: float) -> dict:
    """穿障尝试 vs 漂移收益（P1-1 修正：纯漂移 episode 不存在——诊断数据漂移类=0，
    行为是混合的，改为段级对比）：
    - 穿障尝试收益（逐 episode 对齐）：
      approach_ep 的 V_尝试(rc, rr) = Σ approach 段收益 + terminal_i(rc, rr)
    - 漂移收益：drift 段收益均值（无 terminal；≈0 即 E038 判读实证）
    - ΔV(rc, rr) = E[V_漂移] − E[V_穿障尝试]
    参考量：approach 类 episode 完整 V 均值（逐轨迹，P2-4）。"""
    approach_ep_list = []   # (approach_seg_ret_sum, terminal_rc_rr 延迟算, outcome, t_c/T)
    drift_rets = []
    ep_vals_current = []    # approach 类完整 episode V（参考，rc=-20/rr=100 当前配置）

    for ep in eps:
        recs = ep["records"]
        dist = [r["dist"] for r in recs]
        d_soft = [r["d_soft_h"] for r in recs]
        segs = segment_behavior(dist, d_soft)
        app_seg_sum = 0.0
        has_approach = False
        for s in segs:
            if s["type"] == "approach":
                has_approach = True
                app_seg_sum += seg_return(ep, s["steps"], r_reach_now, r_collision_now)
            elif s["type"] == "drift":
                drift_rets.append(seg_return(ep, s["steps"], r_reach_now, r_collision_now))
        if has_approach:
            approach_ep_list.append({"seg_sum": app_seg_sum, "ep": ep})
            ep_vals_current.append(episode_value(ep, RC_CURRENT, RR_CURRENT, r_reach_now, r_collision_now))

    n_app = len(approach_ep_list)
    e_drift = float(np.mean(drift_rets)) if drift_rets else None

    def E_V_attempt(rc, rr):
        if not approach_ep_list:
            return None
        vals = []
        for it in approach_ep_list:
            ep = it["ep"]
            term = 0.0
            if ep["outcome"] == "reached":
                term = rr * (GAMMA ** (ep["n_steps"] - 1))
            elif ep["outcome"] == "collision":
                term = rc * (GAMMA ** ep["t_c"])
            vals.append(it["seg_sum"] + term)
        return float(np.mean(vals))

    grid = {}
    for rc in RC_GRID:
        for rr in RR_GRID:
            e_a = E_V_attempt(rc, rr)
            grid[f"rc={rc:.0f},rr={rr:.0f}"] = {
                "E_V_attempt": e_a, "E_V_drift": e_drift,
                "deltaV": (e_drift - e_a) if (e_a is not None and e_drift is not None) else None,
            }
    return {
        "n_approach": n_app, "n_drift_segs": len(drift_rets),
        "E_V_drift_ref": e_drift,  # 裕量判据参考值（与 rc/rr 无关：drift 段无 terminal）
        "approach_outcomes": {"reached": sum(1 for it in approach_ep_list if it["ep"]["outcome"] == "reached"),
                              "collision": sum(1 for it in approach_ep_list if it["ep"]["outcome"] == "collision"),
                              "timeout": sum(1 for it in approach_ep_list if it["ep"]["outcome"] == "timeout")},
        "episode_V_current_mean": float(np.mean(ep_vals_current)) if ep_vals_current else None,
        "grid": grid, "current": grid[f"rc={RC_CURRENT:.0f},rr={RR_CURRENT:.0f}"],
    }


def episode_v_grid(eps: list, r_reach_now: float, r_collision_now: float) -> dict:
    """Episode 级口径（设计 §4 原公式，P2-4 逐轨迹 V）：穿障类 episode 的完整 V
    （含非 approach 步的初始飞行/盘旋/停滞拖累）+ 参数化 terminal。
    E[V(漂移)] 参照段级 E_drift（漂移类 episode 不存在，无法按 episode 计算）。
    与 delta_v_grid（段级，P1-1 审核后口径）并列输出——诊断报告双口径呈现，
    口径分歧 = 非 approach 拖累归属（共同成本 vs 穿障特有成本），由用户裁决主口径。"""
    approach_eps = []
    for ep in eps:
        recs = ep["records"]
        segs = segment_behavior([r["dist"] for r in recs], [r["d_soft_h"] for r in recs])
        if "approach" in {s["type"] for s in segs}:
            approach_eps.append(ep)
    drift_rets = []
    for ep in eps:
        recs = ep["records"]
        for s in segment_behavior([r["dist"] for r in recs], [r["d_soft_h"] for r in recs]):
            if s["type"] == "drift":
                t0 = s["steps"][0]
                ret = 0.0
                for t in s["steps"]:
                    terminal_now = 0.0
                    if t == len(recs) - 1:
                        if ep["outcome"] == "reached":
                            terminal_now = r_reach_now
                        elif ep["outcome"] == "collision":
                            terminal_now = r_collision_now
                    ret += (GAMMA ** (t - t0)) * ((recs[t]["progress_terms"] - terminal_now) + recs[t]["smooth"])
                drift_rets.append(ret)
    e_drift = float(np.mean(drift_rets)) if drift_rets else None
    grid = {}
    for rc in RC_GRID:
        for rr in RR_GRID:
            if not approach_eps:
                grid[f"rc={rc:.0f},rr={rr:.0f}"] = {"E_V_episode": None, "E_V_drift": e_drift, "deltaV": None}
                continue
            vals = [episode_value(ep, rc, rr, r_reach_now, r_collision_now) for ep in approach_eps]
            m = float(np.mean(vals))
            grid[f"rc={rc:.0f},rr={rr:.0f}"] = {
                "E_V_episode": m, "E_V_drift": e_drift,
                "deltaV": (e_drift - m) if e_drift is not None else None,
            }
    return {"n_approach": len(approach_eps), "n_drift_segs": len(drift_rets),
            "grid": grid, "current": grid[f"rc={RC_CURRENT:.0f},rr={RR_CURRENT:.0f}"]}


def decision_tree(dvg: dict) -> dict:
    """§5 决策树：路径 A（降 rc，rr 固定 100）→ 无解则路径 B（升 rr，rc 固定 −20）。
    裕量判据（P2-1）：ΔV ≤ −0.2·|E[V_drift]|。
    搜索方向（P1-2 定稿）：沿 −50 → 当前 → 更低方向；首个满足裕量的 rc = 临界值（最小改动）。
    若 −50（E038 配置）自身已满足 → 失衡不成立 → H1 证伪（0 成本止损，§3.1/§4 输出项 1）。"""
    ref = dvg["E_V_drift_ref"] if dvg["E_V_drift_ref"] is not None else 0.0
    thresh = -MARGIN_FRAC * abs(ref)

    # 路径 A：rr=100，rc 沿 RC_GRID 升序（−50→−40→…→0）扫描，首个满足裕量即临界值
    pathA = {"feasible": False, "r_collision_suggested": None, "deltaV_at": None}
    for rc in RC_GRID:
        cell = dvg["grid"].get(f"rc={rc:.0f},rr={RR_CURRENT:.0f}")
        if cell and cell["deltaV"] is not None and cell["deltaV"] <= thresh:
            pathA = {"feasible": True, "r_collision_suggested": rc,
                     "deltaV_at": cell["deltaV"], "threshold": thresh}
            break
    # 路径 B：rc 固定 −20，rr 升权（150, 200）
    pathB = {"feasible": False, "r_reach_suggested": None, "deltaV_at": None}
    if not pathA["feasible"]:
        for rr in sorted([x for x in RR_GRID if x > RR_CURRENT]):
            cell = dvg["grid"].get(f"rc={RC_CURRENT:.0f},rr={rr:.0f}")
            if cell and cell["deltaV"] is not None and cell["deltaV"] <= thresh:
                pathB = {"feasible": True, "r_reach_suggested": rr,
                         "deltaV_at": cell["deltaV"], "threshold": thresh}
                break

    if pathA["feasible"] and pathA["r_collision_suggested"] == RC_GRID[0]:
        conclusion = ("H1 证伪：E038 配置 (rc=−50) 下 ΔV 已满足裕量 → 穿障期望收益不劣于漂移，"
                      "失衡不成立 → 校准不是解药 → 0 成本止损，转 F5/训练动力学候选（§8.3）")
    elif pathA["feasible"]:
        conclusion = (f"路径 A：r_collision = {pathA['r_collision_suggested']:.0f} "
                      f"(rr 保持 100；沿 −50→当前→更低方向首个满足裕量，ΔV = "
                      f"{pathA['deltaV_at']:.2f} ≤ {thresh:.2f})")
    elif pathB["feasible"]:
        conclusion = (f"路径 B：r_reach = {pathB['r_reach_suggested']:.0f} "
                      f"(rc 保持 −20)，ΔV = {pathB['deltaV_at']:.2f} ≤ {thresh:.2f}")
    else:
        conclusion = "校准无解（ΔV 全网格未转正）→ H1 证伪，转 F5/训练动力学候选（§8.3）"
    return {"pathA": pathA, "pathB": pathB, "margin_threshold": thresh, "conclusion": conclusion}


def main() -> None:
    p = argparse.ArgumentParser(description="E040 阶段 0 诊断分析（ΔV + 临界值 + 决策树）")
    p.add_argument("--diag-dir", default="outputs/e040_diag")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/e040_diag/analyze_report.json")
    args = p.parse_args()

    paths = {
        "e038_step100": os.path.join(args.diag_dir, f"diag_e038_step100_seed{args.seed}.json"),
        "bc": os.path.join(args.diag_dir, f"diag_bc_seed{args.seed}.json"),
    }
    report = {"gamma": GAMMA, "margin_rule": "ΔV ≤ -0.2·|E[V_drift]|", "per_ckpt": {}}
    for tag, path in paths.items():
        if not os.path.exists(path):
            print(f"[SKIP] {path} 不存在（先跑 probe_e040_diag.py）")
            continue
        data = load_diag(path)
        report["per_ckpt"][tag] = per_ckpt_stats(data, tag)

    # 决策树基于 e038_step100（漂移态，校准目标）；穿障类 n<5 时用 BC 补充（报告中注明）
    e038 = report["per_ckpt"].get("e038_step100")
    bc = report["per_ckpt"].get("bc")
    if e038 and bc:
        e038_data = load_diag(paths["e038_step100"])
        dvg_e038 = delta_v_grid(e038_data["episodes"],
                                e038_data["config"]["r_reach_now"],
                                e038_data["config"]["r_collision_now"])
        report["deltaV_e038"] = dvg_e038
        # episode 级口径（设计 §4 原公式）并列输出——诊断报告双口径呈现，主口径由用户裁决
        report["deltaV_e038_episode"] = episode_v_grid(
            e038_data["episodes"], e038_data["config"]["r_reach_now"],
            e038_data["config"]["r_collision_now"])
        report["decision_tree"] = decision_tree(dvg_e038)
        if dvg_e038["n_approach"] < 5:
            dvg_bc = delta_v_grid(load_diag(paths["bc"])["episodes"],
                                  load_diag(paths["bc"])["config"]["r_reach_now"],
                                  load_diag(paths["bc"])["config"]["r_collision_now"])
            report["deltaV_bc_supplement"] = dvg_bc
            report["decision_tree"]["sample_warning"] = (
                f"e038 穿障类 n={dvg_e038['n_approach']} < 5，穿障收益估计并入 BC 参考")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    # ── console 摘要（结论先行）──
    print("\n" + "=" * 70)
    print("E040 阶段 0 诊断摘要")
    print("=" * 70)
    for tag in ("e038_step100", "bc"):
        s = report["per_ckpt"].get(tag)
        if not s:
            continue
        print(f"\n[{tag}] n={s['n_eps']}  outcomes={s['outcomes']}  "
              f"碰撞率={s['collision_rate']:.3f}  t_c mean={s['t_c_mean']}"
              f"(min {s['t_c_min']})")
        print(f"  行为段分类: 穿障类={s['class_eps']['approach']} 漂移类={s['class_eps']['drift']} "
              f"其他={s['class_eps']['other']}")
        print(f"  穿障类结局: {s['approach_outcomes']}")
        print(f"  段收益均值: approach={s['segment_returns']['approach']['mean']} "
              f"(n={s['segment_returns']['approach']['n']}), "
              f"drift={s['segment_returns']['drift']['mean']} (n={s['segment_returns']['drift']['n']})")
    dvg = report.get("deltaV_e038")
    if dvg:
        e038_cell = dvg["grid"].get(f"rc={RC_GRID[0]:.0f},rr={RR_CURRENT:.0f}")
        cur = dvg["current"]
        print(f"\n[ΔV @ E038 配置 rc=-50, rr=100] E[V_穿障]={e038_cell['E_V_attempt']:.2f}  "
              f"E[V_漂移]={e038_cell['E_V_drift']:.2f}  ΔV={e038_cell['deltaV']:.2f}"
              f"  (H1 判读: ΔV<0 → 穿障期望收益≥漂移 → 失衡前提不成立)")
        print(f"[ΔV @ 当前常量 rc={RC_CURRENT:.0f}, rr={RR_CURRENT:.0f}] "
              f"E[V_穿障]={cur['E_V_attempt']:.2f}  E[V_漂移]={cur['E_V_drift']:.2f}  "
              f"ΔV={cur['deltaV']:.2f}")
    dvg_ep = report.get("deltaV_e038_episode")
    if dvg_ep:
        e038_ep = dvg_ep["grid"].get(f"rc={RC_GRID[0]:.0f},rr={RR_CURRENT:.0f}")
        cur_ep = dvg_ep["current"]
        print(f"[episode 级口径: ΔV @ E038 配置 rc=-50] E[V_episode]={e038_ep['E_V_episode']:.2f}  "
              f"ΔV={e038_ep['deltaV']:.2f}  (穿障尝试整集含拖累；ΔV>0 → 整集净亏于漂移)")
        print(f"[episode 级口径: ΔV @ 当前常量 rc=-20] E[V_episode]={cur_ep['E_V_episode']:.2f}  "
              f"ΔV={cur_ep['deltaV']:.2f}")
    dt = report.get("decision_tree")
    if dt:
        print(f"\n[决策树] {dt['conclusion']}")
        if "sample_warning" in dt:
            print(f"  ⚠ {dt['sample_warning']}")
    print(f"\n[SAVE] {args.out}")


if __name__ == "__main__":
    main()
