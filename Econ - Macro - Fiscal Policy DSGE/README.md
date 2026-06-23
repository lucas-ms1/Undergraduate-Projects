# ECO 317 Project 3 – AI-Assisted Fiscal Policy Dashboard

A Streamlit application that solves a medium-scale Smets–Wouters (2007) style
DSGE model, simulates unconditional business-cycle dynamics, and runs fiscal
policy experiments with multiple financing rules.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
ECO 317 Project 3/
├── app.py                  # Streamlit entry point (two tabs + sidebar)
├── config.py               # Baseline parameters, slider bounds, constants
├── requirements.txt
├── README.md
├── assets/style.css        # Dark/navy theme with EB Garamond
├── dsge/
│   ├── calibration.py      # Baseline parameter dict
│   ├── steady_state.py     # Analytical steady-state solver
│   └── model.py            # Linearized equations → Γ₀, Γ₁, Ψ, Π
├── solvers/
│   ├── state_space.py      # Build canonical form from parameters
│   └── rational_expectations.py  # QZ / Blanchard-Kahn solver
├── simulation/
│   ├── simulate.py         # 1,000-period stochastic simulation
│   ├── moments.py          # Variances, correlations, autocorrelations
│   ├── empirical.py        # FRED data pull + HP-filter moments
│   ├── econometrics.py     # OLS regressions on simulated data
│   └── irf.py              # 40-quarter deterministic impulse responses
├── policy/
│   ├── shocks.py           # Gc, GI, τL cut, τK cut impulse definitions
│   ├── financing.py        # Lump-sum, τC, τL, τK, Gc-cut rules
│   └── multipliers.py      # Impact/cumulative multipliers, drag horizon
├── utils/
│   └── summaries.py        # f-string commentary generators
└── tests/
    ├── test_steady_state.py
    ├── test_blanchard_kahn.py
    ├── test_moments.py
    ├── test_multipliers.py
    └── ...
```

## Git workflow

- Feature branches per step (e.g., `step-5a-private-block`)
- No direct pushes to `main`
- PRs require one teammate review
- Branch must pass `pytest` before merge
