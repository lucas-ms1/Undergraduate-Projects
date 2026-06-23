# Econ - Macro - Model Solver

Standalone Streamlit dashboard for solving three infinite-horizon stochastic macroeconomic models with value function iteration.

## Models

- Stochastic consumption-savings with CRRA utility and income risk.
- Stochastic Robinson Crusoe capital-accumulation model with productivity risk.
- Endogenous labor supply model with wage risk.

## Contents

- `app.py`: Streamlit entry point.
- `config.py`: model defaults.
- `models/`: model-specific state, reward, and feasibility logic.
- `solvers/`: value function iteration routines.
- `simulation/`: simulations, forecasts, and moments.
- `analysis/`: diagnostics, plots, calibration, and welfare helpers.
- `tests/`: validation and sanity checks.
- `plan.tex` and `plan.pdf`: project plan artifacts.

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
