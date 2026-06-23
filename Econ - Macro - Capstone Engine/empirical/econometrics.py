"""
empirical/econometrics.py
=========================
Econometric models for the Empirical Data Suite.

Provides flexible OLS regression on live FRED/Yahoo data with standard
regression diagnostics (coefficients, t-stats, R², etc.).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class OLSResult:
    """Container for OLS regression results."""
    coefficients: np.ndarray
    std_errors: np.ndarray
    t_stats: np.ndarray
    p_values: np.ndarray
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residuals: np.ndarray
    fitted: np.ndarray
    variable_names: list[str]
    dependent_name: str

    def summary_df(self) -> pd.DataFrame:
        """Return a DataFrame summary table."""
        rows = []
        for i, name in enumerate(self.variable_names):
            rows.append({
                "Variable": name,
                "Coefficient": f"{self.coefficients[i]:.6f}",
                "Std. Error": f"{self.std_errors[i]:.6f}",
                "t-stat": f"{self.t_stats[i]:.3f}",
                "p-value": f"{self.p_values[i]:.4f}",
                "Sig.": _sig_stars(self.p_values[i]),
            })
        return pd.DataFrame(rows)


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "."
    return ""


def run_ols(
    y: pd.Series,
    X: pd.DataFrame,
    add_constant: bool = True,
) -> OLSResult:
    """
    Run OLS regression: y = Xβ + ε

    Parameters
    ----------
    y : dependent variable (pd.Series)
    X : independent variables (pd.DataFrame, each column is a regressor)
    add_constant : whether to prepend a constant column

    Returns
    -------
    OLSResult with all standard diagnostics
    """
    # Align and drop NaN
    data = pd.concat([y.rename("__y__"), X], axis=1).dropna()

    if len(data) == 0:
        raise ValueError(
            "No observations remain after aligning and dropping missing values. "
            "The selected series may not overlap in time — try a different combination."
        )

    y_arr = data["__y__"].values
    X_arr = data.drop(columns="__y__").values
    var_names = list(data.drop(columns="__y__").columns)

    if add_constant:
        X_arr = np.column_stack([np.ones(len(X_arr)), X_arr])
        var_names = ["Constant"] + var_names

    n, k = X_arr.shape

    if n <= k:
        raise ValueError(
            f"Not enough observations ({n}) for the number of regressors ({k}). "
            "Try fewer lags or a wider date range."
        )

    # OLS: β = (X'X)^{-1} X'y
    try:
        XtX_inv = np.linalg.inv(X_arr.T @ X_arr)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_arr.T @ X_arr)

    beta = XtX_inv @ (X_arr.T @ y_arr)
    fitted = X_arr @ beta
    resid = y_arr - fitted

    # Variance of residuals
    s2 = (resid @ resid) / (n - k)
    var_beta = s2 * XtX_inv
    se = np.sqrt(np.diag(var_beta))

    # t-stats and p-values
    t_stats = beta / se
    from scipy import stats as sp_stats
    p_values = 2 * sp_stats.t.sf(np.abs(t_stats), df=n - k)

    # R²
    ss_res = resid @ resid
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k) if (n - k) > 0 else 0.0

    return OLSResult(
        coefficients=beta,
        std_errors=se,
        t_stats=t_stats,
        p_values=p_values,
        r_squared=r2,
        adj_r_squared=adj_r2,
        n_obs=n,
        residuals=resid,
        fitted=fitted,
        variable_names=var_names,
        dependent_name=y.name or "y",
    )


def build_regression_data(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    lags: int = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Prepare regression data with optional lagged regressors.

    If lags > 0, adds columns "{x_col}_lag{i}" for i in 1..lags.
    """
    data = df[[y_col] + x_cols].copy().dropna()
    y = data[y_col]
    X = data[x_cols].copy()

    if lags > 0:
        for col in x_cols:
            for lag in range(1, lags + 1):
                X[f"{col}_lag{lag}"] = data[col].shift(lag)
        X = X.dropna()
        y = y.loc[X.index]

    return y, X
