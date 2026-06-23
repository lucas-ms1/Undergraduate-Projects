# Econ - Macro - Capstone Engine

Capstone Streamlit app that combines the ECO 317 macro projects into a single dashboard with module navigation.

## Modules

- Empirical Data Suite.
- Long-Run Growth: Solow Model.
- Micro-Founded VFI Models.
- DSGE and Fiscal Policy.

## Contents

- `master_app.py`: main Streamlit launcher.
- `pages/`: module pages for empirical analysis, Solow, VFI, and DSGE fiscal policy.
- `solow/`: Solow model logic.
- `vfi/`: imported value-function-iteration models.
- `dsge_engine/`: DSGE, fiscal policy, simulation, and solver code.
- `empirical/`: empirical data fetching and transformations.
- `tests/`: checks for imports, model pieces, and page syntax.
- `ASSIGNMENTRUBRIC/`: assignment and grading documents.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run master_app.py
```

## Test

```bash
pytest
```

Some empirical-data functionality expects a local `FRED_API_KEY` environment variable.
