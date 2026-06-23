"""Simple GE closure for the advanced model-extension sandbox."""

from __future__ import annotations

import numpy as np


def solve_interest_rate_fixed_point(excess_asset_supply_fn, r_init: float = 0.02, max_iter: int = 40, step: float = 0.25):
    """Damped fixed-point update on interest rate using excess asset supply."""
    r = float(r_init)
    history = []
    for it in range(1, int(max_iter) + 1):
        excess = float(excess_asset_supply_fn(r))
        history.append({"iteration": it, "r": r, "excess_assets": excess})
        if abs(excess) < 1e-4:
            break
        r = r - float(step) * excess
        r = float(np.clip(r, -0.02, 0.15))
    return {"r_star": r, "history": history, "converged": abs(history[-1]["excess_assets"]) < 1e-4 if history else False}
