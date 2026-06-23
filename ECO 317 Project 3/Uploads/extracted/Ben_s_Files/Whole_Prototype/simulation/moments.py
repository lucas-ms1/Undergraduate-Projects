"""Business-cycle moments from simulated series."""

import numpy as np
import pandas as pd


def _acf1(x):
    return np.corrcoef(x[1:], x[:-1])[0, 1] if len(x) > 2 else np.nan


def compute_moments(sim_data, idx):
    names = ["y_hat", "c_hat", "i_hat", "l_hat", "pi_hat"]
    series = {name: sim_data[:, idx[name]] for name in names}
    y = series["y_hat"]

    out = {}
    for name in names:
        x = series[name]
        out[f"var_{name}"] = float(np.var(x))
        out[f"corr_{name}_y"] = float(np.corrcoef(x, y)[0, 1])
        out[f"acf1_{name}"] = float(_acf1(x))
    return out, pd.DataFrame(series)
