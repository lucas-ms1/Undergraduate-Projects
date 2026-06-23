"""
dsge/calibration.py
Step 4 – Baseline calibration
-------------------------------
Returns a mutable copy of the baseline parameter dictionary.  Slider
overrides from the Streamlit sidebar are merged on top of these defaults
before the steady-state solver or the model builder sees them.
"""

from __future__ import annotations
from copy import deepcopy
from config import BASELINE_PARAMS


def baseline_parameters() -> dict:
    """Return a mutable copy of the baseline calibration values."""
    return deepcopy(BASELINE_PARAMS)


def override_parameters(overrides: dict | None = None) -> dict:
    """
    Return baseline parameters with user overrides merged in.

    Parameters
    ----------
    overrides : dict of {param_name: value} from Streamlit sliders

    Returns
    -------
    Full parameter dictionary ready for steady_state.solve_steady_state()
    """
    params = baseline_parameters()
    if overrides:
        params.update(overrides)
    return params
