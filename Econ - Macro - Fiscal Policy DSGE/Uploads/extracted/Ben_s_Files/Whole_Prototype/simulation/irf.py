"""Impulse response calculations with financing-rule feedback."""

import numpy as np

from policy.financing import financing_channel


def compute_irf(A, B, idx, sidx, shock_col, rule, horizon=40, shock_size=1.0):
    """Compute impulse responses to a fiscal shock under a financing rule.

    The financing rule adds an ADDITIVE feedback each period: the
    financing instrument (e.g. labour tax) adjusts in proportion to
    the current debt deviation, which stabilises debt without creating
    explosive multiplicative dynamics.

    Parameters
    ----------
    A, B       : reduced-form transition and shock-loading matrices.
    idx, sidx  : variable and shock index dicts.
    shock_col  : name of the initial shock (key into sidx).
    rule       : financing rule string.
    horizon    : number of quarters.
    shock_size : size of the initial impulse (std-dev units).
    """
    n = A.shape[0]
    m = B.shape[1]
    x = np.zeros((horizon, n))

    # Initial shock vector
    eps0 = np.zeros(m)
    eps0[sidx[shock_col]] = shock_size
    x[0] = B @ eps0

    # Financing feedback: instrument adjusts proportional to debt
    fin_col_name, fin_sign = financing_channel(rule)
    fin_strength = 0.15   # feedback intensity (moderate)

    for t in range(1, horizon):
        x[t] = A @ x[t - 1]

        # Additive financing feedback: if debt is high, the financing
        # instrument fires to bring it back down.
        if fin_col_name is not None and fin_col_name in sidx:
            debt_dev = x[t, idx["b_hat"]]
            feedback = np.zeros(m)
            feedback[sidx[fin_col_name]] = fin_sign * fin_strength * debt_dev
            x[t] += B @ feedback

    return x
