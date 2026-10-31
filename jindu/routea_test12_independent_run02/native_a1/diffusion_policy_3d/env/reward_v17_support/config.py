"""Explicit v1.4 contract values; calibration is never silently invented."""

import math


DEFAULTS = dict(
    H_e=4, traj_dt=.3, max_episode_steps=60, w_goal=2., r_reach=100.,
    r_collision=50., r_invalid=50., d_safe=.45, w_c=2.5, w_ground=20.,
    ground_contact=.2, ground_safe=1., collision_margin_xy=.1,
    c_min=.1, sigma_z=.3, w_smooth=.01, d_risk=.9, s_d=.15,
    risk_active_threshold=.35, softmin_beta=8., time_decay=.15,
    lambda_f=.4, C_f=2., free_cone_angle=30., free_ray_step=.1, free_ray_range=10.,
    free_ray_clearance=.2, n_base=10, low_risk_threshold=.2,
    kappa=.3, eta_o=.03, onset_step_cap=.35, onset_episode_cap=8.,
    w_f=.35, epsilon_back=.05, fc_step_cap=.35, fc_episode_cap=10.,
    trap_window=6, epsilon_p=.25, d_trap=.65, epsilon_rec=.5,
    epsilon_d=.15, N_enter=2, N_exit=2, w_stall=.1, w_escape=3.,
    escape_episode_cap=6., w_flip=.35, flip_episode_floor=-5.,
    epsilon_m=.1, clearance_recovered=.9, d_sup_safe=.75,
    epsilon_sup_prog=.2, w_sup_ep=.5, K_max=8,
    rho=.2, d_guard=1., w_eff=.02, eta_eff=.04, w_loop=1.,
    guard_step_cap=2., loop_window=8, loop_sign_changes=4,
    loop_progress=.2, loop_sign_epsilon=1e-4,
)
REQUIRED = ("gamma_pbrs", "d_ref", "s_f", "w_low", "w_high", "w_climb")
DERIVED = ("s_c", "s_low", "h_free", "c_opt", "s_high", "d_improve_cap", "w_o")


def resolve_config(config):
    source = dict(config or {})
    missing = [k for k in REQUIRED if k not in source]
    if missing:
        raise ValueError("v17 calibration values required: {}".format(missing))
    unknown = set(source) - set(DEFAULTS) - set(REQUIRED) - set(DERIVED)
    if unknown:
        raise ValueError("unknown v17 reward fields: {}".format(sorted(unknown)))
    c = dict(DEFAULTS, **source)
    for key, value in c.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("reward config {} must be finite numeric".format(key))
    if (c["H_e"], c["traj_dt"], c["max_episode_steps"], c["c_min"], c["collision_margin_xy"], c["ground_contact"]) != (4, .3, 60, .1, .1, .2):
        raise ValueError("v17/H4 physical and horizon protocol mismatch")
    if not 0 < c["gamma_pbrs"] <= 1 or not 0 < c["rho"] < 1:
        raise ValueError("invalid gamma/risk probability")
    if not c["ground_safe"] > c["ground_contact"] or not c["d_safe"] > .1:
        raise ValueError("invalid barrier scale")
    if not c["epsilon_rec"] > c["epsilon_p"] >= 0:
        raise ValueError("trap recovery hysteresis mismatch")
    for key in ("s_f", "s_d", "softmin_beta", "C_f", "free_ray_step", "free_ray_range", "free_ray_clearance"):
        if c[key] <= 0:
            raise ValueError("{} must be positive".format(key))
    if not c["free_ray_step"] <= c["C_f"] <= c["free_ray_range"] or not 0 < c["free_cone_angle"] < 90:
        raise ValueError("invalid free-cone sampling range/angle")
    for key in ("H_e", "max_episode_steps", "n_base", "trap_window", "N_enter", "N_exit", "K_max", "loop_window", "loop_sign_changes"):
        if c[key] < 1 or int(c[key]) != c[key]:
            raise ValueError("{} must be a positive integer".format(key))
        c[key] = int(c[key])
    for key in c:
        if (key.startswith("w_") or key.endswith("cap") or key in ("sigma_z", "d_ref", "eta_eff", "eta_o", "loop_sign_epsilon")) and c[key] < 0:
            raise ValueError("{} must be nonnegative".format(key))
    if c["flip_episode_floor"] > 0:
        raise ValueError("flip floor must not be positive")
    derived = dict(s_c=c["d_safe"] - .1, s_low=c["c_min"],
                   h_free=max(c["c_min"], 2 * c["sigma_z"]), d_improve_cap=c["d_safe"])
    derived.update(c_opt=c["c_min"] + derived["h_free"], s_high=derived["h_free"], w_o=.25 / c["d_safe"])
    for key, value in derived.items():
        if key in source and not math.isclose(source[key], value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("derived reward field conflict: {}".format(key))
    c.update(derived)
    return c
