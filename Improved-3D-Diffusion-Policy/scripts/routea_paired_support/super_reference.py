"""Sample official SUPER polynomial coefficients; never reshape its path."""
import numpy as np


def polynomial_state(coefficients, durations, order, elapsed):
    """Upstream Piece::getCoeffMat exports descending powers in physical seconds."""
    durations = np.asarray(durations, dtype=float)
    elapsed = min(max(float(elapsed), 0.), float(durations.sum()))
    piece = min(int(np.searchsorted(np.cumsum(durations), elapsed, side='right')),
                len(durations)-1)
    local = elapsed - durations[:piece].sum()
    coeff = np.asarray(coefficients).reshape(-1, order+1)[piece]
    return [np.polyval(np.polyder(coeff, derivative), local) for derivative in range(3)]


def sample_reference(message, now):
    elapsed = now-message.start_WT_pos.to_sec()
    rows = []
    for step in range(1, 9):
        t = elapsed + step*.3
        xyz = np.array([polynomial_state(c, message.time_pos, message.order_pos, t)
                        for c in [message.coef_pos_x, message.coef_pos_y, message.coef_pos_z]])
        yaw = polynomial_state(message.coef_yaw, message.time_yaw, message.order_yaw,
                               now-message.start_WT_yaw.to_sec()+step*.3)
        rows.append(np.r_[xyz[:, 0], xyz[:, 1], xyz[:, 2], yaw[:2]])
    return np.asarray(rows, dtype=np.float32)
