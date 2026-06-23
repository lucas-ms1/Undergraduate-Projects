"""
Grid construction helpers.
==========================
Will provide:
    - linear_grid(x_min, x_max, n)
    - optional curved/log-spaced grids
    - clipping / bounds-check utilities

Implementation deferred to Step 4.
"""


def linear_grid(x_min: float, x_max: float, n: int):
    raise NotImplementedError


def clip_to_grid(value, x_min: float, x_max: float):
    raise NotImplementedError
