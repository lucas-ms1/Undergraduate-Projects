# Econ - Macro - VFI Dashboard

Course project folder for an interactive value-function-iteration dashboard. It includes the final project version, optional enhanced versions, older planning files, and teammate contributions.

## Models

- Stochastic consumption-savings.
- Stochastic Robinson Crusoe production.
- Endogenous labor supply.

## Contents

- `app.py`: main Streamlit entry point for the root copy.
- `models/`, `solvers/`, `simulation/`, `utils/`: core model and numerical code.
- `analysis/`: diagnostics, plots, welfare, calibration, and scoring helpers.
- `tests/`: pytest validation checks.
- `Final Project - Lucas, Rida, Lindsey, Aidan, Ben/`: final team submission copy.
- `Bells and whistles v2/`: enhanced version with additional analysis modules.
- `Other member's work/`: teammate draft code and supporting files.
- `old/`: earlier plans and lock-in sheets.

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
