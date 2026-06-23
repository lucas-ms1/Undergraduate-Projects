# Econ - Macro - Fiscal Policy DSGE

Streamlit dashboard for fiscal policy experiments in a medium-scale DSGE model. The app solves a Smets-Wouters-style linearized model, simulates business-cycle dynamics, compares model and empirical moments, and computes fiscal impulse responses under alternative financing rules.

## Contents

- `app.py`: main Streamlit dashboard.
- `config.py`: baseline parameters, constants, and slider bounds.
- `dsge/`: calibration, steady-state, and model-equation code.
- `solvers/`: state-space and rational-expectations solvers.
- `simulation/`: stochastic simulation, moments, empirical comparison, and IRFs.
- `policy/`: fiscal shocks, financing rules, and multiplier calculations.
- `tests/`: model, solver, simulation, and policy checks.
- `Submission/` and `Uploads/`: submission copy and supporting uploaded material.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Test

```bash
pytest
```

Some empirical-data functionality expects a local `FRED_API_KEY` environment variable.
