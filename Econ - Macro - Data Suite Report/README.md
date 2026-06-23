# Econ - Macro - Data Suite Report

Streamlit and Python project for collecting, organizing, and reporting macroeconomic and financial data. The folder also includes the written report artifacts from the coursework submission.

## Contents

- `streamlit_app.py`: main Streamlit app.
- `src/finrec/`: provider, recipe, storage, and visualization code.
- `legacy_pages/`: earlier Streamlit page layout.
- `tests/`: pytest checks for recipes, storage, and dataset merge behavior.
- `main.tex`, `main.pdf`, and related files: report build artifacts.
- `FEATURE_SUMMARY.md`: feature inventory and implementation notes.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
streamlit run streamlit_app.py
```

If using optional data providers, copy `.env.example` to `.env` and add local API keys there. Do not commit `.env`.

## Optional Extras

```bash
pip install -e ".[forecast]"
pip install -e ".[market]"
pip install -e ".[macro]"
pip install -e ".[news]"
pip install -e ".[econ]"
```
