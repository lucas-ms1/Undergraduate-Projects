"""Calibration utilities."""

from copy import deepcopy

from config import BASELINE_PARAMS


def get_baseline_parameters():
    """Return a mutable copy of baseline calibration values."""
    return deepcopy(BASELINE_PARAMS)
