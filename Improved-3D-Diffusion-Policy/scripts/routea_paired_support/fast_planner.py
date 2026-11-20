"""Fast-Planner native position/yaw B-spline sampling (no EGO yaw rule)."""
import numpy as np


def clamp_eval(spline, value):
    """Evaluate inside the spline's own valid range.

    Upstream `NonUniformBspline::evaluateDeBoor` clamps every query into that
    spline's range, `ub = min(max(u_(p_), u), u_(m_ - p_))`, and `traj_server`
    relies on it: position, velocity, acceleration, yaw and yaw-rate each have
    their own knots. scipy returns NaN outside a spline's range instead, so the
    clamp has to be explicit. The yaw spline's range end `seg_num * (duration /
    seg_num)` can sit 1 ulp below the position duration, which is enough to turn
    every terminal sample into a NaN yaw without this.
    """
    return spline(min(float(spline.t[-spline.k-1]), max(float(spline.t[spline.k]), value)))


def sample_reference(message, now, position=None, yaw=None, yaw_rate=0.):
    """Match upstream traj_server's native spline and terminal hover semantics."""
    from scipy.interpolate import BSpline
    degree = int(message.order)
    knots = np.asarray(message.knots, dtype=np.float64)
    points = np.array([[p.x, p.y, p.z] for p in message.pos_pts])
    position = BSpline(knots, points, degree, extrapolate=False)
    velocity, acceleration = position.derivative(), position.derivative(2)
    # NonUniformBspline initializes uniform knots as (i - degree)*interval.
    yaw_points = np.asarray(message.yaw_pts, dtype=np.float64)
    yaw_knots = (np.arange(len(yaw_points) + degree + 1) - degree)*message.yaw_dt
    yaw = BSpline(yaw_knots, yaw_points, degree, extrapolate=False)
    yaw_rate = yaw.derivative()
    begin, end = knots[degree], knots[-degree-1]
    elapsed = now-message.start_time.to_sec()
    rows = []
    for step in range(1, 9):
        t = elapsed+step*.3
        local = min(end, begin+t)
        if t < 0:
            raise ValueError('Fast-Planner trajectory starts in the future')
        terminal = begin+t >= end
        rows.append(np.r_[clamp_eval(position, local),
                          np.zeros(3) if terminal else clamp_eval(velocity, local),
                          np.zeros(3) if terminal else clamp_eval(acceleration, local),
                          clamp_eval(yaw, local-begin), clamp_eval(yaw_rate, local-begin)])
    return np.asarray(rows, dtype=np.float32)


from .ego import EgoPolicy


class FastPolicy(EgoPolicy):
    policy_kind = 'fast'
    service_module = 'plan_manage.srv'
    service_name = '/fast_planner_node/evaluation_step'
    parameter_namespace = '/fast_planner_node'
    reference_sampler = staticmethod(sample_reference)

    def after_step(self, context):
        result = super().after_step(context)
        result['action_origin'] = 'fast_native_position_yaw_bspline'
        return result
