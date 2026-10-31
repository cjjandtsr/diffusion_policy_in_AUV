"""Reward v1.7 / design v1.4. Scalar rewards; no geometry autograd or repair.

The authoritative TransitionFacts must be formed before calling this module.
Configuration requires explicit calibration values; density is frozen BC data.
"""

import copy
import math

import numpy as np

from .reward_v17_support.config import resolve_config
from .reward_v17_support.density import validate_density, support_cell, probability
from .reward_v17_support.geometry import path_geometry, direct_path, free_potential, validate_obstacles
from .reward_v17_support.state import create_reward_history, positive_budget, update_trap, guard_cost

TERM_NAMES = ("terminal", "progress", "clearance", "ground", "overflight", "climb",
              "smoothness", "pbrs_free", "onset", "future_clearance", "trap", "flip",
              "support_escrow", "guard")


def _metadata():
    return dict(reward_schema_version="v17", altitude_termination_enabled=True,
                altitude_exceed_status="active", altitude_exceed_count=0,
                altitude_exceed_rate=0., invalid_state=False, training_usable=True)


def _bounded_square(x):
    if x <= 1:
        return x * x / (1. + x * x)
    return 1. / (1. + (1. / x) ** 2)


def _history_finite(value):
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, dict):
        return all(_history_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_history_finite(v) for v in value)
    return bool(np.isfinite(value).all())


def invalid_result(history, cfg, phi=None):
    terms = dict.fromkeys(TERM_NAMES, 0.)
    terms["terminal"] = -cfg["r_invalid"]
    if phi is None:
        phi = history.get("previous_phi")
    valid = phi is not None and math.isfinite(float(phi))
    if valid:
        terms["pbrs_free"] = -float(phi)
    metrics = _metadata()
    metrics.update(invalid_state=True, training_usable=False, pbrs_valid=valid,
                   terminated=True, done_reason="invalid_state", raw_done_reason="invalid_state")
    return float(sum(terms.values())), terms, metrics, copy.deepcopy(history)


def compute_reward_v17(s_t, executed_path, s_tp1, transition_facts, obstacle_specs,
                       goal_world, episode_start_world, geometry_context, history,
                       offline_density, cfg):
    c = resolve_config(cfg)
    h = copy.deepcopy(history)
    if h.get("support_paid"):
        raise ValueError("episode already settled; reset reward history before another transition")
    if h.get("density_hash") != (offline_density or {}).get("content_hash") or h.get("density_hash") is None:
        validate_density(offline_density)
        h["density_hash"] = offline_density["content_hash"]
    validate_obstacles(obstacle_specs)
    p = np.asarray(executed_path, dtype=np.float64)
    goal = np.asarray(goal_world, dtype=np.float64)
    start = np.asarray(episode_start_world, dtype=np.float64)
    if p.shape != (c["H_e"] + 1, 3) or goal.shape != (3,) or start.shape != (3,):
        raise ValueError("v17 requires actual (H_e+1,3) executed path and xyz references")
    if not all(np.isfinite(v).all() for v in (p, goal, start, s_t["pos"], s_tp1["pos"],
                                              s_t["yaw"], s_tp1["yaw"])):
        return invalid_result(h, c)
    if not _history_finite(h):
        return invalid_result(h, c)
    if not np.array_equal(p[0], s_t["pos"]) or not np.array_equal(p[-1], s_tp1["pos"]):
        raise ValueError("executed path endpoints must equal the actual before/after state")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _compute(s_t, p, s_tp1, transition_facts, obstacle_specs, goal, start,
                            geometry_context, h, offline_density, c)
    except (FloatingPointError, OverflowError):
        return invalid_result(history, c)


def _compute(s_t, p, s_tp1, facts, obstacles, goal, start, context, h, density, c):
    terms, metrics = dict.fromkeys(TERM_NAMES, 0.), _metadata()
    g = path_geometry(p, obstacles, context, c)
    if (g["collision"] and not (facts.collided or facts.ground_crash)) or (
        g["out_of_bounds"] and not (facts.out_of_bounds or facts.collided or facts.ground_crash)
    ):
        raise ValueError("transition facts disagree with continuous physical geometry")
    failure = facts.physical_failure
    safe = g["physical_safe"] and not failure
    reached = facts.done_reason == "reached"
    ended = bool(facts.terminated or facts.truncated)
    progress = float(np.linalg.norm(goal - p[0]) - np.linalg.norm(goal - p[-1]))
    delta = np.diff(p, axis=0)
    climbs = np.maximum(delta[:, 2], 0.)
    climb_raw = float(np.sum(1. - 1. / (1. + climbs)))
    smooth_raw = float(np.sum(np.abs(np.diff(delta, axis=0))))
    contact_failure = bool(facts.collided or facts.ground_crash)
    terms["terminal"] = c["r_reach"] if reached else (
        -c["r_collision"] if contact_failure else (-c["r_invalid"] if failure else 0.)
    )
    terms["progress"] = c["w_goal"] * progress
    terms["climb"] = -c["w_climb"] * climb_raw
    terms["smoothness"] = -c["w_smooth"] * smooth_raw
    clearance_raw = max(0., (c["d_safe"] - g["generic_clearance"]) / c["s_c"]) ** 2
    # Minimum over the legal subset of a continuous path. If it crosses the
    # contact height, that boundary belongs to the legal subset (strict <).
    legal_ground_min = max(g["ground_clearance"], c["ground_contact"])
    has_legal_ground = float(np.max(p[:, 2]) - context.ground_z) >= c["ground_contact"]
    ground_raw = max(0., (c["ground_safe"] - legal_ground_min) / (c["ground_safe"] - c["ground_contact"])) ** 2 if has_legal_ground else 0.
    terms["clearance"], terms["ground"] = -c["w_c"] * clearance_raw, -c["w_ground"] * ground_raw
    clearances = g["overflight_clearances"]
    low_raw = max((max(0., (c["c_min"] - v) / c["s_low"]) ** 2 for v in clearances), default=0.)
    high_raw = max((_bounded_square(max(0., v - c["c_opt"]) / c["s_high"]) for v in clearances), default=0.)
    terms["overflight"] = -c["w_low"] * low_raw - c["w_high"] * high_raw

    phi = h["previous_phi"]
    if phi is None:
        phi = free_potential(p[0], goal, obstacles, context, c)
    phi_next = 0. if facts.terminated else free_potential(p[-1], goal, obstacles, context, c)
    terms["pbrs_free"] = c["gamma_pbrs"] * phi_next - phi
    h["previous_phi"] = phi_next

    horizontal = goal[:2] - p[0, :2]
    norm = np.hypot(*horizontal)
    lateral = float(np.mean((p[1:, :2] - p[0, :2]) @ [-horizontal[1] / norm, horizontal[0] / norm])) if norm > 0 else 0.
    yaw = abs(math.atan2(math.sin(s_tp1["yaw"] - s_t["yaw"]), math.cos(s_tp1["yaw"] - s_t["yaw"])))
    q = direct_path(p, goal)
    direct = path_geometry(q, obstacles, context, c) if q is not None else None
    baseline = [row for row in h["steps"][-c["n_base"]:] if row["risk"] < c["low_risk_threshold"]]
    mode = "history" if len(baseline) >= 3 else "contrast"
    if direct is None:
        mode = "invalid"
    if mode == "history":
        amplitude = max(0., abs(lateral) - np.mean([abs(r["lateral"]) for r in baseline])
                        + c["kappa"] * (yaw - np.mean([r["yaw"] for r in baseline])) - c["eta_o"])
    elif mode == "contrast":
        direct_lateral = float(np.mean((q[1:, :2] - p[0, :2]) @ [-horizontal[1] / norm, horizontal[0] / norm]))
        amplitude = max(0., abs(lateral) - abs(direct_lateral) - c["eta_o"])
    else:
        amplitude = 0.
    improvement = max(0., g["d_soft"] - direct["d_soft"]) if direct is not None else 0.
    onset_raw = c["w_o"] * g["risk"] * amplitude * min(improvement, c["d_improve_cap"]) if safe else 0.
    terms["onset"] = positive_budget(onset_raw, c["onset_step_cap"], h["caps"]["onset"], c["onset_episode_cap"])
    h["caps"]["onset"] += terms["onset"]

    row = dict(progress=progress, d_min=g["d_min"], d_soft=g["d_soft"], risk=g["risk"],
               lateral=lateral, yaw=yaw, guard_context=False)
    rows = h["steps"] + [row]
    window = rows[-c["trap_window"]:]
    window_progress = sum(r["progress"] for r in window)
    full = len(window) == c["trap_window"]
    fc_raw = c["w_f"] * math.tanh((g["d_soft"] - c["d_ref"]) / c["s_f"]) if full and window_progress > -c["epsilon_back"] else 0.
    fc = max(-c["fc_step_cap"], min(c["fc_step_cap"], fc_raw))
    terms["future_clearance"] = min(fc, 0.) if failure else (positive_budget(fc, c["fc_step_cap"], h["caps"]["future_clearance"], c["fc_episode_cap"]) if fc > 0 else fc)
    h["caps"]["future_clearance"] += max(0., terms["future_clearance"])
    terms["trap"], trap_metrics = update_trap(h, rows, c, safe, ended, reached)
    metrics.update(trap_metrics)

    side = 1 if lateral > c["epsilon_m"] else (-1 if lateral < -c["epsilon_m"] else 0)
    previous_side = h["steps"][-1].get("side", 0) if h["steps"] else 0
    flip = side != 0 and previous_side != 0 and side != previous_side
    h["side_flip_count"] += int(flip)
    if flip and g["risk"] > c["risk_active_threshold"] and g["d_soft"] <= c["clearance_recovered"]:
        terms["flip"] = -min(c["w_flip"], max(0., -c["flip_episode_floor"] - h["caps"]["flip"]))
        h["caps"]["flip"] += -terms["flip"]
    row["side"] = side

    cell = support_cell(p[-1], start, goal, context.ground_z, h["initial_yaw"])
    density_value = probability(density, cell)
    eligible = safe and full and g["d_soft"] > c["d_sup_safe"] and window_progress > c["epsilon_sup_prog"] and density_value < density["tau_D"]
    key = ",".join(map(str, cell))
    if eligible and key not in h["eligible_cells"]:
        h["eligible_cells"].append(key)
        if p[-1, 2] - context.ground_z > density["demo_height_max"]:
            h["eligible_above_demo"].append(key)
    if reached:
        terms["support_escrow"] = c["w_sup_ep"] * min(len(h["eligible_cells"]), c["K_max"])
        h["support_paid"] = True

    guard_context = (safe and direct is not None and direct["physical_safe"]
                     and h["trap_state"] == "FREE" and g["related_count"] == 0
                     and g["risk"] < c["rho"] and g["d_soft"] > c["d_guard"]
                     and direct["risk"] < c["rho"] and direct["d_soft"] > c["d_guard"])
    row["guard_context"] = bool(guard_context)
    terms["guard"], guard_metrics = guard_cost(g["excess_length"], guard_context, rows, c)
    metrics.update(guard_metrics)
    heights = p[:, 2] - context.ground_z
    threshold = density["demo_height_max"]
    above_seconds = 0.
    for z0, z1 in zip(heights[:-1], heights[1:]):
        if z0 > threshold and z1 > threshold:
            fraction = 1.
        elif z0 <= threshold and z1 <= threshold:
            fraction = 0.
        else:
            fraction = (max(z0, z1) - threshold) / abs(z1 - z0)
        above_seconds += c["traj_dt"] * fraction
    paid_high = sum(key in h["eligible_above_demo"] for key in h["eligible_cells"][:c["K_max"]]) if reached else 0
    metrics.update(demo_height_max=threshold, goal_above_demo_height=goal[2] - context.ground_z > threshold,
                   above_demo_height=bool(np.max(heights) > threshold),
                   max_ground_relative_height=float(np.max(heights)),
                   max_goal_relative_height=float(np.max(p[:, 2] - goal[2])),
                   above_demo_dwell_seconds=above_seconds,
                   above_demo_unknown_seconds=above_seconds if not full else 0.,
                   above_demo_stall_seconds=above_seconds if full and window_progress <= c["epsilon_p"] else 0.,
                   support_eligible_above_demo=len(h["eligible_above_demo"]),
                   support_paid_above_demo=c["w_sup_ep"] * paid_high,
                   side_flip_count=h["side_flip_count"])
    metrics.update(terminated=facts.terminated, truncated=facts.truncated, done_reason=facts.done_reason,
                   altitude_exceed_count=int(facts.altitude_exceed),
                   altitude_exceed_rate=float(facts.altitude_exceed),
                   d_soft_h=g["d_soft"], d_min=g["d_min"], risk_gate=g["risk"],
                   counterfactual_valid=direct is not None, counterfactual_collision=bool(direct and direct["collision"]),
                   counterfactual_risk=direct["risk"] if direct else None,
                   counterfactual_d_soft=direct["d_soft"] if direct else None,
                   actual_collision=g["collision"], onset_mode=mode, history_n_valid=len(baseline),
                   onset_history=mode == "history", onset_contrast=mode == "contrast", onset_invalid=mode == "invalid",
                   onset_amplitude=float(amplitude), onset_improvement=improvement, onset_raw=float(onset_raw),
                   onset_step_cap_hit=onset_raw > c["onset_step_cap"], onset_episode_cap_hit=h["caps"]["onset"] >= c["onset_episode_cap"],
                   fc_raw=fc_raw, fc_cap_hit=abs(fc_raw) > c["fc_step_cap"],
                   guard_context_unknown=direct is None, guard_excess_length_m=g["excess_length"],
                   guard_chord_m=g["chord_length"], executed_length_m=g["path_length"],
                   legacy_lateral_proxy=abs(lateral) + c["kappa"] * yaw,
                   ground_clearance=g["ground_clearance"], max_z=float(np.max(p[:, 2])),
                   raw_progress=progress, raw_climb=climb_raw, raw_smoothness=smooth_raw,
                   raw_clearance=clearance_raw, raw_ground=ground_raw, raw_overflight_low=low_raw,
                   raw_overflight_high=high_raw, overflight_encounter_count=len(clearances),
                   overflight_clearance_min=min(clearances) if clearances else None,
                   positive_climb_m=float(climbs.sum()), support_eligible=eligible,
                   support_cell_density=density_value, support_eligible_cells=len(h["eligible_cells"]),
                   support_paid=terms["support_escrow"] > 0, frozen_density_hash=density["content_hash"],
                   pbrs_valid=True, phi_current=phi, phi_next=phi_next, gamma_pbrs=c["gamma_pbrs"],
                   H_e=c["H_e"], traj_dt=c["traj_dt"])
    h["steps"] = rows[-max(c["n_base"], c["trap_window"], c["loop_window"]):]
    if ended:
        h.update(trap_state="FREE", enter_count=0, exit_count=0, entry_clearance=None)
    h["episode_steps"] += 1
    for name, value in terms.items():
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite reward term {}".format(name))
        h["term_sums"][name] = h["term_sums"].get(name, 0.) + value
        metrics["episode_sum_" + name] = h["term_sums"][name]
    metrics = {key: value.item() if isinstance(value, np.generic) else value for key, value in metrics.items()}
    return float(sum(terms.values())), terms, metrics, h
