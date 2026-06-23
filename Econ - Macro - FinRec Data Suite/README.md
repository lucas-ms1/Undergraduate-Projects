# Econ - Macro - FinRec Data Suite

Local-first Streamlit application for macro, market, and news data workflows. This is the standalone FinRec data-suite repository imported into the undergraduate projects collection.

## Contents

- `streamlit_app.py`: main Streamlit app.
- `src/finrec/providers/`: market, macro, and news provider interfaces.
- `src/finrec/recipes/`: transformation and modeling recipes.
- `src/finrec/storage/`: SQLite-backed job and artifact storage.
- `src/finrec/ui/`: Streamlit UI helpers.
- `tests/`: smoke and unit tests.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
streamlit run streamlit_app.py
```

If using the Financial Modeling Prep provider, copy `.env.example` to `.env` and set `FINREC_FMP_API_KEY` locally. Do not commit `.env`.

## Test

```bash
pytest
```
