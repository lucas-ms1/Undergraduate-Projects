# ECO 317 Project 3 Dashboard

Streamlit dashboard for a stylized medium-scale DSGE fiscal-policy assignment with:
- QZ-based rational-expectations diagnostics
- 1,000-period unconditional simulation and moments
- 40-quarter fiscal IRFs across shocks and financing rules
- Impact/cumulative multipliers and fiscal drag horizon

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

- `app.py`: UI + tabs + plots + commentary
- `dsge/`: calibration, steady state, model assembly
- `solvers/`: state-space and QZ diagnostics
- `simulation/`: simulation, moments, IRFs
- `policy/`: shocks, financing rules, multipliers
- `utils/`: dynamic narrative summaries
- `tests/`: sanity tests
