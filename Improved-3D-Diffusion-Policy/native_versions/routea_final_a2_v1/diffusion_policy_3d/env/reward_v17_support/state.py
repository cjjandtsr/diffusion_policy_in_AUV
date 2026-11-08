"""Episode-local reward state. No PPO or diffusion dependencies."""


def create_reward_history(initial_yaw=0., episode_id=0):
    return dict(episode_id=episode_id, initial_yaw=float(initial_yaw), steps=[],
                caps=dict(onset=0., future_clearance=0., escape=0., flip=0.),
                trap_state="FREE", enter_count=0, exit_count=0, segment_id=0,
                entry_clearance=None, segment_paid=False, eligible_cells=[],
                support_paid=False, episode_steps=0, term_sums={}, previous_phi=None,
                density_hash=None, eligible_above_demo=[], side_flip_count=0)


def positive_budget(value, step_cap, used, episode_cap):
    return max(0., min(value, step_cap, max(0., episode_cap - used)))


def update_trap(h, rows, cfg, physical_safe, episode_end, reached):
    window = rows[-cfg["trap_window"]:]
    full = len(window) == cfg["trap_window"]
    progress = sum(r["progress"] for r in window)
    eligible = full and progress < cfg["epsilon_p"] and min(r["d_min"] for r in window) < cfg["d_trap"]
    old = h["trap_state"]
    recovery = (full and h["entry_clearance"] is not None and physical_safe
                and progress > cfg["epsilon_rec"]
                and rows[-1]["d_soft"] - h["entry_clearance"] > cfg["epsilon_d"])
    escape = 0.
    if old in ("FREE", "CANDIDATE"):
        h["enter_count"] = h["enter_count"] + 1 if eligible else 0
        h["trap_state"] = "CANDIDATE" if eligible else "FREE"
        if h["enter_count"] >= cfg["N_enter"]:
            h.update(trap_state="TRAPPED", exit_count=0, segment_id=h["segment_id"] + 1,
                     entry_clearance=rows[-1]["d_soft"], segment_paid=False)
    else:
        h["exit_count"] = h["exit_count"] + 1 if recovery else 0
        h["trap_state"] = "RECOVERING" if recovery else "TRAPPED"
        if h["exit_count"] >= cfg["N_exit"]:
            if physical_safe and (not episode_end or reached) and not h["segment_paid"]:
                escape = min(cfg["w_escape"], max(0., cfg["escape_episode_cap"] - h["caps"]["escape"]))
                h["caps"]["escape"] += escape
                h["segment_paid"] = True
            h.update(trap_state="FREE", enter_count=0, exit_count=0, entry_clearance=None)
    stall = cfg["w_stall"] if old in ("TRAPPED", "RECOVERING") or h["trap_state"] == "TRAPPED" else 0.
    active = old in ("TRAPPED", "RECOVERING") or h["trap_state"] in ("TRAPPED", "RECOVERING")
    return escape - stall, dict(trap_active=active, trap_recovered=escape > 0,
                               trap_escape_paid=escape, trap_stall_raw=stall,
                               trap_enter_count=h["enter_count"], trap_exit_count=h["exit_count"],
                               trap_segment_id=h["segment_id"], trap_state=h["trap_state"])


def guard_cost(excess_length, context, rows, cfg):
    window = rows[-cfg["loop_window"]:]
    ready = len(window) == cfg["loop_window"]
    qualified = ready and all(r["guard_context"] for r in window)
    signs = [1 if r["progress"] > 0 else -1 for r in window
             if abs(r["progress"]) > cfg["loop_sign_epsilon"]]
    changes = sum(a != b for a, b in zip(signs, signs[1:]))
    loop = qualified and changes >= cfg["loop_sign_changes"] and sum(r["progress"] for r in window) < cfg["loop_progress"]
    excess = max(0., excess_length - cfg["eta_eff"]) if context else 0.
    efficiency, loop_cost = cfg["w_eff"] * excess, cfg["w_loop"] * float(loop)
    cost = efficiency + loop_cost
    return -min(cfg["guard_step_cap"], cost), dict(
        guard_context=bool(context), guard_detour_opportunity=excess > 0,
        guard_loop_window_ready=ready, guard_loop_window_qualified=qualified,
        loop_sign_changes=changes, loop_detected=bool(loop),
        guard_efficiency_cost=efficiency, guard_loop_cost=loop_cost,
        guard_pre_cap=cost, guard_cap_hit=cost > cfg["guard_step_cap"])
