"""
solow/model.py
==============
Core Solow Growth Model with Harrod-neutral technological progress.

Production function in effective-worker units:
    y = f(k) = k**alpha

Course-note transition equation:
    k_{t+1} = s*f(k_t) + [1 - (n + g + delta)]*k_t

Steady state:
    k* = [s / (n + g + delta)]**[1 / (1 - alpha)]

Golden Rule:
    s_gold = alpha
    k_gold = [alpha / (n + g + delta)]**[1 / (1 - alpha)]
"""

import numpy as np


def steady_state_k(s: float, n: float, g: float, delta: float, alpha: float) -> float:
    """
    Analytically solve for steady-state capital per effective worker.

    k* = [s / (n + g + delta)]**[1 / (1 - alpha)]
    """
    breakeven = n + g + delta
    if breakeven <= 0:
        return np.inf
    return (s / breakeven) ** (1 / (1 - alpha))


def golden_rule(n: float, g: float, delta: float, alpha: float) -> dict:
    """
    Compute the Golden Rule saving rate and associated steady-state values.

    At the Golden Rule, MPK = n + g + delta, so s_gold = alpha.
    """
    breakeven = n + g + delta
    s_gold = alpha
    k_gold = (alpha / breakeven) ** (1 / (1 - alpha)) if breakeven > 0 else np.inf
    y_gold = k_gold ** alpha
    c_gold = y_gold - breakeven * k_gold
    return dict(s_gold=s_gold, k_gold=k_gold, y_gold=y_gold, c_gold=c_gold)


def simulate_transition(
    s: float,
    n: float,
    g: float,
    delta: float,
    alpha: float,
    k0: float | None = None,
    T: int = 200,
) -> dict:
    """
    Simulate Solow transition dynamics in effective-worker units.

    The law of motion is:
        k_{t+1} = s*k_t**alpha + [1 - (n + g + delta)]*k_t

    This is intentionally consistent with steady_state_k() and the
    Solow diagram break-even curve.
    """
    if T <= 0:
        raise ValueError("T must be a positive integer.")

    breakeven_rate = n + g + delta
    k_star = steady_state_k(s, n, g, delta, alpha)

    if k0 is None:
        k0 = 0.5 * k_star if np.isfinite(k_star) else 1.0

    k = np.zeros(T)
    k[0] = max(float(k0), 1e-8)

    for t in range(T - 1):
        k[t + 1] = s * k[t] ** alpha + (1 - breakeven_rate) * k[t]

    y = k ** alpha
    c = (1 - s) * y
    investment = s * y
    breakeven = breakeven_rate * k

    return dict(
        t=np.arange(T),
        k=k,
        y=y,
        c=c,
        investment=investment,
        breakeven=breakeven,
        k_star=k_star,
        y_star=k_star ** alpha if np.isfinite(k_star) else np.inf,
    )


def solow_diagram_curves(
    s: float,
    n: float,
    g: float,
    delta: float,
    alpha: float,
    k_max: float | None = None,
    n_points: int = 500,
) -> dict:
    """
    Compute curves for the classic Solow diagram:
      - actual investment: s*f(k)
      - break-even investment: (n + g + delta)*k
    """
    k_star = steady_state_k(s, n, g, delta, alpha)

    if k_max is None:
        k_max = 2.0 * k_star if np.isfinite(k_star) else 20.0

    k_grid = np.linspace(0, k_max, n_points)
    sf_k = s * k_grid ** alpha
    breakeven_k = (n + g + delta) * k_grid

    return dict(
        k_grid=k_grid,
        sf_k=sf_k,
        breakeven_k=breakeven_k,
        k_star=k_star,
        y_star=k_star ** alpha if np.isfinite(k_star) else np.inf,
    )


def simulate_capital_destruction(
    s: float,
    n: float,
    g: float,
    delta: float,
    alpha: float,
    destruction_frac: float = 0.5,
    T_before: int = 50,
    T_after: int = 150,
) -> dict:
    """
    Simulate a sudden capital-destruction event followed by Solow recovery.

    The economy starts at k*, runs for T_before periods, loses
    destruction_frac of its capital at destruction_period, and then
    transitions back toward k*.
    """
    if T_before < 1 or T_after < 1:
        raise ValueError("T_before and T_after must both be positive.")

    breakeven_rate = n + g + delta
    k_star = steady_state_k(s, n, g, delta, alpha)
    T = T_before + T_after

    k = np.zeros(T)
    k[0] = k_star

    for t in range(T - 1):
        if t == T_before - 1:
            k[t + 1] = k[t] * (1 - destruction_frac)
        else:
            k[t + 1] = s * k[t] ** alpha + (1 - breakeven_rate) * k[t]

    y = k ** alpha
    c = (1 - s) * y

    growth_rate = np.zeros(T)
    growth_rate[1:] = (y[1:] - y[:-1]) / y[:-1] * 100

    return dict(
        t=np.arange(T),
        k=k,
        y=y,
        c=c,
        growth_rate=growth_rate,
        destruction_period=T_before,
        k_star=k_star,
    )
