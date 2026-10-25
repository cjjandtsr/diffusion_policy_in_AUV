"""Deterministic train/development replay and the finite 27-candidate protocol.

Inputs are complete executed episodes, never aggregate training metrics. Fixture
records must identify themselves; no fixture is counted as an empirical episode.
"""

from collections import Counter
import math

import numpy as np

from ..collision_geom import ObstacleSpec
from ..reward_v17 import compute_reward_v17, create_reward_history, resolve_config
from .geometry import GeometryContext, path_geometry
from .state import guard_cost, update_trap

FORBIDDEN = {"w4", "w8", "w11", "w12", "w18"}


def _source(row, fixture=False):
    allowed = {"fixture"} if fixture else {"rl_train", "rl_dev"}
    if row.get("split") not in allowed or row.get("world_id") in FORBIDDEN:
        raise ValueError("evaluation/unknown source cannot enter reward calibration")


def _weighted_quantile(values, weights, fraction):
    ordered = sorted(zip(values, weights))
    target, cumulative = fraction * sum(weights), 0.
    for value, weight in ordered:
        cumulative += weight
        if cumulative + 1e-15 >= target:
            return float(value)
    return float(ordered[-1][0])


def anchor_scale(empirical, fixtures):
    for r in empirical:
        _source(r)
    for r in fixtures:
        _source(r, fixture=True)
    episodes = {r["episode_id"] for r in empirical}
    worlds = {r["world_id"] for r in empirical}
    use_empirical = len(episodes) >= 100 and len(worlds) >= 2

    def distribution(rows):
        if not rows:
            raise ValueError("calibration requires nonempty geometrically measured fixture anchors")
        sizes = Counter(r["episode_id"] for r in rows)
        values = [float(r["d_soft"]) for r in rows]
        if any(not math.isfinite(v) or v <= 0 or v >= 100 for v in values):
            raise ValueError("anchor includes invalid/contact/no-obstacle sentinel clearance")
        weights = [1. / (len(sizes) * sizes[r["episode_id"]]) for r in rows]
        q25, median, q75 = (_weighted_quantile(values, weights, q) for q in (.25, .5, .75))
        scale = max(median - q25, q75 - median) / math.atanh(.5)
        return values, weights, median, scale

    values, weights, median, scale = distribution(empirical if use_empirical else fixtures)
    if use_empirical and scale <= 0:
        use_empirical = False
        values, weights, median, scale = distribution(fixtures)
    if scale <= 0:
        raise ValueError("degenerate fixture anchor scale")
    trials, chosen = [], None
    for k in (1, 2, 4):
        mass = sum(w for v, w in zip(values, weights) if abs(math.tanh((v - median) / (k * scale))) >= .95)
        trials.append(dict(k_s=k, saturation_mass=mass))
        if mass <= .05 + 1e-12:
            chosen = k
            break
    return dict(status="resolved" if chosen else "unresolved", d_ref=median,
                s_f=chosen * scale if chosen else None, k_s=chosen, trials=trials,
                calibration_source="empirical" if use_empirical else "fixture",
                empirical_episode_count=len(episodes), empirical_world_count=len(worlds),
                empirical_window_count=len(empirical), used_window_count=len(values))


def select_height_weights(features, comparisons, signal_checks):
    if not features or not comparisons or not signal_checks:
        raise ValueError("height calibration requires complete fixtures, ordering and signal checks")
    for row in features.values():
        for values in (row, row.get("undiscounted", row)):
            if any(not math.isfinite(values[key]) for key in ("base", "low", "high", "climb")):
                raise ValueError("calibration features must be finite")
            if any(values[key] < 0 for key in ("low", "high", "climb")):
                raise ValueError("calibration cost features must be nonnegative")
    high = max(r["high"] for r in features.values())
    climb = max(r["climb"] for r in features.values())
    if high <= 0 or climb <= 0:
        raise ValueError("height calibration features must include overflight and climb")
    candidates, selected = [], None
    for budget in (4., 8., 16.):
        for low in (2.5, 5., 10.):
            for a in (.5, .25, .75):
                wh, wc = a * budget / high, (1. - a) * budget / climb
                returns = {key: r["base"] - low * r["low"] - wh * r["high"] - wc * r["climb"] for key, r in features.items()}
                margins = {better + ">" + worse: returns[better] - returns[worse] for better, worse in comparisons}
                failed = [name for name, margin in margins.items() if margin <= 0]
                undiscounted = {key: r.get("undiscounted", r) for key, r in features.items()}
                undiscounted_returns = {key: r["base"] - low * r["low"] - wh * r["high"] - wc * r["climb"] for key, r in undiscounted.items()}
                raw_margins = {better + ">" + worse: undiscounted_returns[better] - undiscounted_returns[worse] for better, worse in comparisons}
                raw_failed = [name for name, margin in raw_margins.items() if margin <= 0]
                candidate = dict(B=budget, a=a, w_low=low, w_high=wh, w_climb=wc,
                                 margins=margins, minimum_margin=min(margins.values()),
                                 undiscounted_margins=raw_margins, discounted_returns=returns,
                                 undiscounted_returns=undiscounted_returns,
                                 failed_undiscounted_orderings=raw_failed,
                                 failed_orderings=failed, signals_pass=all(signal_checks.values()))
                # DPPO optimizes the configured discounted return.  The
                # gamma=1 replay is retained as a stress diagnostic, but is
                # not a second objective that can veto the gamma=.99 result.
                candidate["accepted"] = not failed and candidate["signals_pass"]
                candidates.append(candidate)
                if candidate["accepted"] and selected is None:
                    selected = candidate
    return dict(status="resolved" if selected else "unresolved", selected=selected,
                candidates=candidates, signal_checks=signal_checks,
                feature_maxima=dict(high=high, climb=climb),
                selection_objective="discounted_return",
                undiscounted_role="reported_stress_diagnostic_not_acceptance_gate")


def replay_episodes(episodes, cfg, density):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import TransitionFacts
    features, anchors, records = {}, [], {}
    for e in episodes:
        fixture = e.get("split") == "fixture"
        _source(e, fixture)
        eid = str(e["episode_id"])
        if eid in features:
            raise ValueError("duplicate replay episode identity")
        obs = [ObstacleSpec.from_dict(o) for o in e["obstacles"]]
        context = GeometryContext(**e["geometry_context"])
        start, goal = np.array(e["start"], dtype=float), np.array(e["goal"], dtype=float)
        history = create_reward_history(e["initial_yaw"], eid)
        position, yaw = start.copy(), float(e["initial_yaw"])
        rows, anchor_rows = [], []
        feature = dict(base=0., low=0., high=0., climb=0.,
                       undiscounted=dict(base=0., low=0., high=0., climb=0.))
        if not e["steps"] or len(e["steps"]) > cfg["max_episode_steps"]:
            raise ValueError("replay episode must have 1..60 actual transitions")
        for t, step in enumerate(e["steps"]):
            path = np.asarray(step["executed_path"], dtype=np.float64)
            if path.shape != (5, 3) or not np.array_equal(path[0], position):
                raise ValueError("replay path shape/continuity mismatch")
            g = path_geometry(path, obs, context, cfg)
            last = path[-1]
            reached = np.linalg.norm(last - goal) < .5 and abs(last[2] - goal[2]) < .3
            raw = g["raw_done_reason"] if not g["physical_safe"] else ("reached" if reached else "none")
            facts = TransitionFacts.resolve(collided=g["collision"], early_death=g["out_of_bounds"],
                                             reached=reached, time_limit=t + 1 >= cfg["max_episode_steps"], raw_done_reason=raw)
            next_yaw = float(step["end_yaw"])
            result = compute_reward_v17(dict(pos=position, yaw=yaw), path, dict(pos=last, yaw=next_yaw),
                                         facts, obs, goal, start, context, history, density, cfg)
            total, terms, metrics, history = result
            if not metrics["training_usable"]:
                raise ValueError("invalid numeric replay episode {}".format(eid))
            factor = cfg["gamma_pbrs"] ** t
            # Candidate height weights do not alter the fixed trajectory/history.
            feature["base"] += factor * (total - terms["overflight"] - terms["climb"])
            feature["undiscounted"]["base"] += total - terms["overflight"] - terms["climb"]
            for key, metric in (("low", "raw_overflight_low"), ("high", "raw_overflight_high"), ("climb", "raw_climb")):
                feature[key] += factor * metrics[metric]
                feature["undiscounted"][key] += metrics[metric]
            rows.append(dict(terms=terms, metrics=metrics, discount=factor,
                             critic_bootstrap="not_available_in_geometry_replay" if facts.truncated else "zero_if_terminal"))
            if g["physical_safe"] and g["related_count"] > 0 and 0 < g["d_soft"] < 100:
                anchor_rows.append(dict(episode_id=eid, world_id=e["world_id"], role=e["role"], split=e["split"], d_soft=g["d_soft"]))
            position, yaw = last, next_yaw
            if facts.terminated or facts.truncated:
                if t != len(e["steps"]) - 1:
                    raise ValueError("replay has data after authoritative terminal")
            elif t == len(e["steps"]) - 1:
                raise ValueError("incomplete/truncated legacy trajectory cannot be full-return fixture")
        if facts.done_reason != e["expected_done_reason"]:
            raise ValueError("replay expected outcome differs from current physical truth")
        if facts.done_reason == "reached":
            anchors.extend(anchor_rows)
        features[eid] = feature
        records[eid] = rows
    return features, anchors, records


def deterministic_signal_checks(cfg, density, with_evidence=False):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import TransitionFacts
    context = GeometryContext()
    goal, start = np.array([4., 0., 1.]), np.array([0., 0., 1.])
    facts = TransitionFacts.resolve(collided=False, early_death=False, reached=False, time_limit=False)

    def run(path, obs=(), history=None):
        path = np.array(path, dtype=float)
        return compute_reward_v17(dict(pos=path[0], yaw=0.), path, dict(pos=path[-1], yaw=0.),
                                   facts, obs, goal, start, context, history or create_reward_history(), density, cfg)

    straight = [[0, 0, 1], [.2, .2, 1], [.4, .4, 1], [.6, .6, 1], [.8, .8, 1]]
    zig = [[0, 0, 1], [.2, .2, 1], [.4, -.2, 1], [.6, .2, 1], [.8, 0, 1]]
    obstacle = ObstacleSpec("signal_box", "box", .8, 0, 1, size_x=.15, size_y=.15, size_z=2)
    avoid = [[0, 0, 1], [.2, .6, 1], [.4, .6, 1], [.6, .6, 1], [.8, .6, 1]]
    contrast = run(avoid, [obstacle])
    h = create_reward_history()
    h["steps"] = [dict(progress=.1, d_min=2., d_soft=2., risk=0., lateral=0., yaw=0., guard_context=True)] * 5
    historical = run(avoid, [obstacle], h)
    fc_near, fc_far = historical[1]["future_clearance"], run(straight, (), h)[1]["future_clearance"]
    capped = create_reward_history(); capped["caps"]["onset"] = cfg["onset_episode_cap"]
    loop_rows = [dict(progress=p, guard_context=True) for p in [.1, -.1] * 4]
    checks = dict(onset_contrast=np.float32(contrast[1]["onset"]) > 0,
                onset_history=np.float32(historical[1]["onset"]) > 0,
                future_clearance_contrast=np.float32(fc_near) != np.float32(fc_far),
                necessary_avoidance_guard=contrast[1]["guard"] == 0,
                clear_diagonal_guard=run(straight)[1]["guard"] == 0,
                clear_detour_guard=np.float32(run(zig)[1]["guard"]) < 0,
                guard_onset_budget_independent=run(zig)[1]["guard"] == run(zig, history=capped)[1]["guard"],
                loop_signal=guard_cost(0., True, loop_rows, cfg)[0] < 0)
    checks["no_obstacle_turn_onset_zero"] = run(zig)[1]["onset"] == 0.
    trap = create_reward_history()
    stuck = [dict(progress=0., d_min=.3, d_soft=.3)] * cfg["trap_window"]
    update_trap(trap, stuck, cfg, True, False, False)
    candidate_only = trap["trap_state"] == "CANDIDATE"
    update_trap(trap, stuck, cfg, True, False, False)
    checks["trap_stable_detection"] = candidate_only and trap["trap_state"] == "TRAPPED"
    alternating = [dict(progress=.1 * (-1)**i, d_min=.3, d_soft=.3) for i in range(cfg["trap_window"])]
    alternate_trap = create_reward_history()
    for _ in range(cfg["N_enter"]):
        update_trap(alternate_trap, alternating, cfg, True, False, False)
    checks["trap_alternating_detection"] = alternate_trap["trap_state"] == "TRAPPED"
    recover = [dict(progress=.12, d_min=1., d_soft=1.)] * cfg["trap_window"]
    update_trap(trap, recover, cfg, True, False, False)
    update_trap(trap, stuck, cfg, True, False, False)
    checks["trap_chatter_no_payment"] = trap["caps"]["escape"] == 0 and trap["exit_count"] == 0
    for _ in range(cfg["N_exit"]):
        _, recovery_metrics = update_trap(trap, recover, cfg, True, False, False)
    paid = trap["caps"]["escape"]
    update_trap(trap, recover, cfg, True, False, False)
    checks["trap_escape_once"] = paid == cfg["w_escape"] and trap["caps"]["escape"] == paid
    checks = {key: bool(value) for key, value in checks.items()}
    signals = dict(onset_contrast=contrast[1]["onset"], onset_history=historical[1]["onset"],
                   future_clearance_near=fc_near, future_clearance_far=fc_far,
                   clear_detour_guard=run(zig)[1]["guard"], trap_escape=recovery_metrics["trap_escape_paid"])
    evidence = {key: dict(float64=float(v), buffer_float32=float(np.float32(v))) for key, v in signals.items()}
    return (checks, evidence) if with_evidence else checks
