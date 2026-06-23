# ECO 317 Project 2 – AI-Assisted Macroeconomic Modeling Dashboard

Interactive Streamlit dashboard featuring three infinite-horizon stochastic
models solved by Value Function Iteration (VFI).

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit entry point |
| `config.py` | Default parameters for all three models |
| `models/` | Model-specific definitions (utility, constraints, feasibility) |
| `solvers/` | Generic VFI engine |
| `simulation/` | Simulation, forecasting, and moment calculation |
| `utils/` | Shared numerical helpers (grids, Markov, interpolation) |
| `tests/` | Validation and sanity checks |
| `assets/` | CSS styling |

## Models

1. **Stochastic Consumption-Savings** – CRRA utility, exogenous income shock
2. **Stochastic Robinson Crusoe** – Capital accumulation, TFP shock
3. **Endogenous Labor Supply** – Labor/leisure choice, wage shock
