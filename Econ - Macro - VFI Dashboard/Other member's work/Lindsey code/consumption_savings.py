"""
Model 1: Stochastic Consumption-Savings
========================================
Household chooses next-period assets a' to maximise expected discounted
CRRA utility subject to a two-state Markov income shock.

Bellman:
    V(a, y) = max_{a'} { u(c) + beta * sum_{y'} P(y'|y) V(a', y') }

Budget constraint:
    c = (1+r)*a + y - a',   c > 0

State:   (a, y)
Control: a'
Residual: c (from budget constraint)

Notation follows model_lockin_sheet.tex  Section 3 and the
cross-model coding conventions table.
"""

import numpy as np
from config import MODEL1_DEFAULTS


def utility(c: np.ndarray, sigma: float) -> np.ndarray:
    """CRRA utility.  Returns -inf where c <= 0."""
    raise NotImplementedError


def build_grids(params: dict) -> dict:
    """Return dict with keys 'a_grid' and 'y_vals'."""
    raise NotImplementedError


def feasibility_mask(a_grid: np.ndarray, y_vals: np.ndarray,
                     r: float) -> np.ndarray:
    """Boolean array marking (a, y, a') triples where c > 0."""
    raise NotImplementedError


def solve(params: dict | None = None) -> dict:
    """Solve Model 1 via VFI.

    Parameters
    ----------
    params : dict, optional
        Overrides for MODEL1_DEFAULTS.

    Returns
    -------
    dict with keys (standardised across all models):
        value_function : np.ndarray, shape (n_a, 2)
        policy_indices : np.ndarray, shape (n_a, 2)
        policy_levels  : np.ndarray, shape (n_a, 2)   – optimal a'
        c_policy       : np.ndarray, shape (n_a, 2)
        grid           : np.ndarray, shape (n_a,)      – the asset grid
        diagnostics    : dict with 'iterations', 'final_error', 'converged'
    """
    raise NotImplementedError
