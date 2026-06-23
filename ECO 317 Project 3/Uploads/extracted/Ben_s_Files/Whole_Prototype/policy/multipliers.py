"""Fiscal multipliers and drag horizon."""

import numpy as np


def compute_multipliers(irf, idx, beta=0.99, g_ss=0.2, y_ss=1.0):
    """Compute impact and cumulative fiscal multipliers.

    The multiplier is defined as the output response divided by the
    initial spending impulse (normalised by steady-state ratios).

    Impact multiplier:     dy(0) / dg(0)
    Cumulative multiplier: sum[ beta^t dy(t) ] / sum[ beta^t dg(0) ]
    """
    y = irf[:, idx["y_hat"]] * y_ss
    T = len(y)

    # Use the initial output impulse to infer the spending shock size.
    # For government spending shocks the initial b_hat response proxies
    # the spending impulse; for tax shocks we use the output response
    # relative to g_ss directly.
    g0 = irf[0, idx["b_hat"]] * g_ss
    if abs(g0) < 1e-12:
        # Fallback: use the output response itself as scale
        g0 = y[0] if abs(y[0]) > 1e-12 else 1.0

    impact = y[0] / g0 if abs(g0) > 1e-9 else np.nan

    # Cumulative: discounted sum of output / discounted spending impulse
    discount = beta ** np.arange(T)
    cum_num = np.sum(discount * y)
    # The spending impulse is a one-time shock; its discounted sum
    # equals g0 * sum(beta^t) only if spending persists. Use the
    # actual spending path if available, otherwise spread the impulse.
    cum_den = g0 * np.sum(discount)  # present value of a unit permanent shock
    # Better: use the actual initial impulse only at t=0
    cum_den = g0  # single-period impulse
    if abs(cum_den) < 1e-12:
        cum_den = 1.0

    cumulative = cum_num / cum_den if abs(cum_den) > 1e-9 else np.nan

    drag = next((t for t in range(1, T) if y[t] < 0), None)
    return {
        "impact_multiplier": impact,
        "cumulative_multiplier": cumulative,
        "fiscal_drag_horizon": drag,
    }
