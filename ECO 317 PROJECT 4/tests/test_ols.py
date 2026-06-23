import os
import sys
import traceback

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from empirical.econometrics import build_regression_data, run_ols  # noqa: E402


FAILURES = []


def check(name, func):
    try:
        func()
        print(f"PASS  {name}")
    except Exception:
        print(f"FAIL  {name}")
        print(traceback.format_exc())
        FAILURES.append(name)


def data():
    np.random.seed(42)
    n = 500
    x = np.random.randn(n)
    noise = np.random.randn(n) * 0.5
    y = 2.0 + 3.0 * x + noise
    return n, x, y


def test_simple_regression():
    n, x, y = data()
    result = run_ols(pd.Series(y, name="y"), pd.DataFrame({"x": x}), add_constant=True)
    assert abs(result.coefficients[0] - 2.0) < 0.3, f"Intercept: {result.coefficients[0]}"
    assert abs(result.coefficients[1] - 3.0) < 0.3, f"Slope: {result.coefficients[1]}"
    assert result.r_squared > 0.85, f"R2 too low: {result.r_squared}"
    assert abs(result.t_stats[0]) > 5, f"Intercept t-stat too small: {result.t_stats[0]}"
    assert abs(result.t_stats[1]) > 10, f"Slope t-stat too small: {result.t_stats[1]}"
    assert result.p_values[0] < 0.001
    assert result.p_values[1] < 0.001
    assert result.variable_names == ["Constant", "x"]
    assert result.dependent_name == "y"
    assert result.n_obs == n
    summary = result.summary_df()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 2


def test_multiple_regressors():
    np.random.seed(42)
    n = 500
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y2 = 1.0 + 2.0 * x1 - 0.5 * x2 + np.random.randn(n) * 0.3
    result = run_ols(pd.Series(y2, name="y"), pd.DataFrame({"x1": x1, "x2": x2}))
    assert abs(result.coefficients[1] - 2.0) < 0.3
    assert abs(result.coefficients[2] - (-0.5)) < 0.3


def test_no_constant():
    _, x, y = data()
    result = run_ols(pd.Series(y, name="y"), pd.DataFrame({"x": x}), add_constant=False)
    assert "Constant" not in result.variable_names
    assert len(result.coefficients) == 1


def test_lagged_regressors():
    np.random.seed(42)
    df = pd.DataFrame({
        "y": np.random.randn(100),
        "x1": np.random.randn(100),
        "x2": np.random.randn(100),
    })
    y_out, X_out = build_regression_data(df, "y", ["x1", "x2"], lags=2)
    assert "x1_lag1" in X_out.columns
    assert "x1_lag2" in X_out.columns
    assert "x2_lag1" in X_out.columns
    assert "x2_lag2" in X_out.columns
    assert len(y_out) == len(X_out)
    assert len(y_out) <= 98


def test_collinearity_handling():
    _, x, y = data()
    result = run_ols(pd.Series(y, name="y"), pd.DataFrame({"x": x, "x_copy": x}))
    assert np.all(np.isfinite(result.coefficients))


def main():
    check("Simple regression (y = 2 + 3x)", test_simple_regression)
    check("Multiple regressors", test_multiple_regressors)
    check("No constant", test_no_constant)
    check("Lagged regressors", test_lagged_regressors)
    check("Collinearity handling", test_collinearity_handling)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
