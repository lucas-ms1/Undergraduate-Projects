"""Steady-state computations for a stylized medium-scale DSGE model."""


def compute_steady_state(params):
    beta = params["beta"]
    alpha = params["alpha"]
    delta = params["delta"]
    gy_ratio = params["gy_ratio"]
    ig_ratio = params["ig_ratio"]
    debt_y_ratio = params["debt_y_ratio"]

    y = 1.0
    r_k = 1.0 / beta - (1.0 - delta)
    k = alpha / max(r_k, 1e-6)
    i = delta * k
    if i / y > 0.45:
        i = ig_ratio * y
        k = i / delta
    g = gy_ratio * y
    c = y - i - g
    w = (1.0 - alpha) * y
    l = 1.0
    b = debt_y_ratio * y

    resource_gap = y - (c + i + g)

    return {
        "Y": y,
        "C": c,
        "I": i,
        "G": g,
        "K": k,
        "L": l,
        "W": w,
        "Rk": r_k,
        "B": b,
        "resource_gap": resource_gap,
    }
