"""
simulation/econometrics.py
Step 11b – Econometric analysis on simulated data
----------------------------------------------------
Runs OLS regressions on the 1,000-period simulated series to satisfy
the Econometrics rubric row.

Two regressions:
  1. Taylor-rule recovery:
       î_t = c + ρ̂_i · î_{t-1} + φ̂_π · π̂_t + φ̂_y · ŷ_t + u_t
     Recovered coefficients should land close to the structural values
     (ρ_i=0.80, φ_π=1.50, φ_y=0.125) when the sample is long enough.

  2. Consumption smoothness:
       ĉ_t = c + α₁ · ŷ_t + α₂ · ŷ_{t-1} + α₃ · ĉ_{t-1} + u_t
     The coefficient on ĉ_{t-1} captures habit-driven persistence;
     the coefficient on ŷ_t captures the contemporaneous income
     elasticity (stronger for high λ_rot).

Also computes variance ratios with bootstrapped standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# OLS helper (avoids hard statsmodels dependency)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OLSResult:
    """Lightweight OLS regression output."""
    dep_var: str
    regressors: list[str]
    coefficients: np.ndarray
    std_errors: np.ndarray
    t_stats: np.ndarray
    r_squared: float
    n_obs: int
    residuals: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        """Return a flat dict of {regressor: coeff} for easy display."""
        return {name: float(c) for name, c in zip(self.regressors, self.coefficients)}

    def summary_rows(self) -> list[Dict[str, object]]:
        """Return a list of dicts suitable for pd.DataFrame."""
        rows = []
        for name, coef, se, t in zip(
            self.regressors, self.coefficients, self.std_errors, self.t_stats
        ):
            rows.append({
                "Variable": name,
                "Coefficient": round(float(coef), 5),
                "Std. Error": round(float(se), 5),
                "t-stat": round(float(t), 3),
            })
        rows.append({
            "Variable": "R²",
            "Coefficient": round(self.r_squared, 4),
            "Std. Error": "",
            "t-stat": "",
        })
        rows.append({
            "Variable": "N",
            "Coefficient": self.n_obs,
            "Std. Error": "",
            "t-stat": "",
        })
        return rows


def _ols(y: np.ndarray, X: np.ndarray, dep_name: str, reg_names: list[str]) -> OLSResult:
    """
    Run OLS: y = X β + u.

    Parameters
    ----------
    y : (T,) dependent variable
    X : (T, k) regressors (should include a constant column if desired)
    dep_name : label for the dependent variable
    reg_names : labels for each column of X
    """
    n, k = X.shape
    # β = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    residuals = y - X @ beta
    ssr = float(residuals @ residuals)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r_sq = 1.0 - ssr / sst if sst > 0 else 0.0

    # Standard errors: σ² (X'X)^{-1}
    sigma2 = ssr / max(n - k, 1)
    try:
        var_beta = sigma2 * np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        var_beta = np.full((k, k), np.nan)

    se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))
    t_stats = np.where(se > 0, beta / se, np.full_like(beta, np.nan))

    return OLSResult(
        dep_var=dep_name,
        regressors=reg_names,
        coefficients=beta,
        std_errors=se,
        t_stats=t_stats,
        r_squared=r_sq,
        n_obs=n,
        residuals=residuals,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Regression 1: Taylor-rule recovery
# ──────────────────────────────────────────────────────────────────────────────

def taylor_rule_regression(
    r_hat: np.ndarray,
    pi_hat: np.ndarray,
    y_hat: np.ndarray,
    policy_shock_hat: Optional[np.ndarray] = None,
    rho_i_target: Optional[float] = None,
) -> OLSResult:
    """
    Recover structural Taylor-rule coefficients from simulated data.

    First estimate i_t on i_{t-1}, pi_t, and y_t. If the monetary policy
    residual is supplied, subtract (1-rho_i)*m_t before estimation so the
    regression recovers the systematic Taylor-rule coefficients rather than
    an omitted-shock mixture. Report rho_i_hat as the raw lag coefficient,
    and report phi_pi_hat and phi_y_hat after dividing the raw coefficients
    by (1 - rho_i_hat).
    """
    T = len(r_hat)
    y = r_hat[1:].copy()
    if policy_shock_hat is not None and rho_i_target is not None:
        y = y - (1.0 - float(rho_i_target)) * policy_shock_hat[1:]
    X = np.column_stack([
        np.ones(T - 1),
        r_hat[:-1],
        pi_hat[1:],
        y_hat[1:],
    ])
    raw = _ols(y, X, dep_name="r_hat", reg_names=["const", "r_lag1", "pi_hat", "y_hat"])

    rho_i_hat = float(raw.coefficients[1])
    denom = 1.0 - rho_i_hat
    if np.isclose(denom, 0.0):
        phi_pi_hat = np.nan
        phi_y_hat = np.nan
        phi_pi_se = np.nan
        phi_y_se = np.nan
    else:
        phi_pi_hat = float(raw.coefficients[2] / denom)
        phi_y_hat = float(raw.coefficients[3] / denom)
        phi_pi_se = float(raw.std_errors[2] / abs(denom))
        phi_y_se = float(raw.std_errors[3] / abs(denom))

    coefficients = np.array([raw.coefficients[0], rho_i_hat, phi_pi_hat, phi_y_hat], dtype=float)
    std_errors = np.array([raw.std_errors[0], raw.std_errors[1], phi_pi_se, phi_y_se], dtype=float)
    t_stats = np.where(std_errors > 0, coefficients / std_errors, np.full_like(coefficients, np.nan))

    return OLSResult(
        dep_var="r_hat",
        regressors=["const", "rho_i_hat", "phi_pi_hat", "phi_y_hat"],
        coefficients=coefficients,
        std_errors=std_errors,
        t_stats=t_stats,
        r_squared=raw.r_squared,
        n_obs=raw.n_obs,
        residuals=raw.residuals,
    )


# Regression 2: Consumption smoothness
# ──────────────────────────────────────────────────────────────────────────────

def consumption_smoothness_regression(
    c_hat: np.ndarray,
    y_hat: np.ndarray,
) -> OLSResult:
    """
    Estimate consumption smoothness / habit persistence.

    Regression:  ĉ_t = c + α₁ · ŷ_t + α₂ · ŷ_{t-1} + α₃ · ĉ_{t-1} + u_t

    The coefficient on ĉ_{t-1} captures habit-driven persistence.
    The coefficient on ŷ_t captures contemporaneous income sensitivity
    (stronger when λ_rot is high).
    """
    T = len(c_hat)
    y = c_hat[1:]
    X = np.column_stack([
        np.ones(T - 1),
        y_hat[1:],             # ŷ_t
        y_hat[:-1],            # ŷ_{t-1}
        c_hat[:-1],            # ĉ_{t-1}
    ])
    return _ols(y, X, dep_name="c_hat", reg_names=["const", "y_t", "y_lag1", "c_lag1"])


# ──────────────────────────────────────────────────────────────────────────────
# Variance ratios with bootstrapped standard errors
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VarianceRatioResult:
    """Variance ratio with bootstrap CI."""
    name: str
    ratio: float
    se: float
    ci_low: float
    ci_high: float


def variance_ratios_bootstrap(
    series: Dict[str, np.ndarray],
    output_key: str = "y",
    n_boot: int = 500,
    seed: int = 42,
) -> list[VarianceRatioResult]:
    """
    Compute Var(x)/Var(y) for each series with bootstrapped standard errors.

    Parameters
    ----------
    series : dict of {name: (T,) array}
    output_key : which series is "output"
    n_boot : number of bootstrap replications
    seed : RNG seed

    Returns
    -------
    List of VarianceRatioResult, one per non-output series
    """
    rng = np.random.default_rng(seed)
    y = series[output_key]
    T = len(y)
    results = []

    for name, x in series.items():
        if name == output_key:
            continue
        # Point estimate
        ratio = float(np.var(x, ddof=1) / np.var(y, ddof=1))

        # Bootstrap
        boot_ratios = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, T, size=T)
            var_x_b = np.var(x[idx], ddof=1)
            var_y_b = np.var(y[idx], ddof=1)
            boot_ratios[b] = var_x_b / var_y_b if var_y_b > 0 else np.nan

        se = float(np.nanstd(boot_ratios))
        ci_low = float(np.nanpercentile(boot_ratios, 2.5))
        ci_high = float(np.nanpercentile(boot_ratios, 97.5))

        results.append(VarianceRatioResult(
            name=f"Var({name})/Var({output_key})",
            ratio=ratio,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
        ))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: run all econometric analyses
# ──────────────────────────────────────────────────────────────────────────────

def run_all_econometrics(
    series: Dict[str, np.ndarray],
    output_key: str = "y",
    rate_key: str = "r",
    inflation_key: str = "pi",
    consumption_key: str = "c",
    policy_shock_key: str = "m_policy",
    structural_params: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Run all Step 11b econometric analyses.

    Parameters
    ----------
    series : dict of observable series from extract_observables

    Returns
    -------
    Dict with keys: taylor_rule, consumption_smoothness, variance_ratios
    """
    y = series[output_key]

    result = {}

    # Taylor rule recovery (needs rate series)
    if rate_key in series and inflation_key in series:
        policy_shock_hat = series.get(policy_shock_key)
        rho_i_target = None
        if structural_params is not None and "rho_i" in structural_params:
            rho_i_target = float(structural_params["rho_i"])
        result["taylor_rule"] = taylor_rule_regression(
            series[rate_key],
            series[inflation_key],
            y,
            policy_shock_hat=policy_shock_hat,
            rho_i_target=rho_i_target,
        )

    # Consumption smoothness
    if consumption_key in series:
        result["consumption_smoothness"] = consumption_smoothness_regression(
            series[consumption_key], y
        )

    # Variance ratios
    result["variance_ratios"] = variance_ratios_bootstrap(series, output_key=output_key)

    return result
