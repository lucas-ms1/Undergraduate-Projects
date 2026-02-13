# finrec-app (local)

A local-first Streamlit app skeleton with:
- pluggable data/news providers (stubbed initially)
- background-ish jobs (threaded worker for now)
- SQLite for job status + logs
- saved CSV artifacts per job

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"

cp .env.example .env
streamlit run streamlit_app.py
```

## Notes

* This is intentionally minimal: providers are stubs that generate synthetic data.
* Later steps can swap the job runner to a true background worker process and implement real providers.

## Optional extras

Install feature-specific dependencies as needed:

- Forecasting (ETS/ARIMA + ML lags): `pip install -e ".[forecast]"`
- Market data (yfinance): `pip install -e ".[market]"`
- Macro data (FRED via pandas-datareader): `pip install -e ".[macro]"`
- News (GDELT via requests): `pip install -e ".[news]"`
- Econometrics (statsmodels): `pip install -e ".[econ]"`
