from __future__ import annotations

import pandas as pd
import streamlit as st

from finrec.recipes.registry import get_recipe_registry


def _get_singletons():
    storage = st.session_state.finrec_storage
    runner = st.session_state.finrec_runner
    return storage, runner


st.set_page_config(page_title="Analysis", layout="wide")
st.title("Analysis — Recipes")

storage, runner = _get_singletons()
recipe_reg = get_recipe_registry()

jobs = storage.list_jobs(limit=500)
completed = [j for j in jobs if j.status == "SUCCEEDED" and j.output_path]

if not completed:
    st.info("No completed artifacts yet. Run a Data Pull job first.")
    st.stop()

label_to_job = {f"{j.job_id} — {j.kind} — {j.provider_kind}:{j.provider_id}": j for j in completed}
job_label = st.selectbox("Input artifact (completed job)", list(label_to_job.keys()))
job = label_to_job[job_label]

st.write(f"**Input path:** `{job.output_path}`")

recipes = recipe_reg.list()
recipe_label_to_id = {f"{r.meta.name} ({r.meta.id})": r.meta.id for r in recipes}
recipe_label = st.selectbox("Recipe", list(recipe_label_to_id.keys()))
recipe_id = recipe_label_to_id[recipe_label]
recipe = recipe_reg.get(recipe_id)

with st.expander("Recipe description", expanded=False):
    st.write(recipe.meta.description)

st.markdown("### Parameters")
params: dict = {}

if recipe_id == "log_returns":
    params["price_col"] = st.text_input("price_col", value="close")
    params["out_col"] = st.text_input("out_col", value="log_return")
elif recipe_id == "rolling_zscore":
    params["value_col"] = st.text_input("value_col", value="value")
    params["window"] = st.number_input("window", min_value=2, max_value=365, value=12, step=1)
    params["prefix"] = st.text_input("prefix", value="roll")
elif recipe_id == "sma":
    params["value_col"] = st.text_input("value_col", value="close")
    params["window"] = st.number_input("window", min_value=2, max_value=500, value=20, step=1)
    params["out_col"] = st.text_input("out_col", value=f"sma_{int(params['window'])}")
elif recipe_id == "ema":
    params["value_col"] = st.text_input("value_col", value="close")
    params["span"] = st.number_input("span", min_value=2, max_value=500, value=20, step=1)
    params["out_col"] = st.text_input("out_col", value=f"ema_{int(params['span'])}")
elif recipe_id == "rsi":
    params["price_col"] = st.text_input("price_col", value="close")
    params["window"] = st.number_input("window", min_value=2, max_value=200, value=14, step=1)
    params["out_col"] = st.text_input("out_col", value=f"rsi_{int(params['window'])}")
elif recipe_id == "macd":
    params["price_col"] = st.text_input("price_col", value="close")
    params["fast"] = st.number_input("fast", min_value=2, max_value=200, value=12, step=1)
    params["slow"] = st.number_input("slow", min_value=3, max_value=300, value=26, step=1)
    params["signal"] = st.number_input("signal", min_value=2, max_value=200, value=9, step=1)
    params["prefix"] = st.text_input("prefix", value="macd")
elif recipe_id == "rolling_vol":
    params["price_col"] = st.text_input("price_col", value="close")
    params["window"] = st.number_input("window", min_value=2, max_value=500, value=20, step=1)
    params["annualize"] = st.checkbox("annualize", value=True)
    params["periods_per_year"] = st.number_input(
        "periods_per_year",
        min_value=1,
        max_value=10000,
        value=252,
        step=1,
    )
    params["out_col"] = st.text_input("out_col", value=f"vol_{int(params['window'])}")
elif recipe_id == "ols":
    params["y_col"] = st.text_input("y_col", value="value")
    params["x_cols"] = st.text_input("x_cols (comma-separated)", value="")
    params["add_constant"] = st.checkbox("add_constant", value=True)
    st.caption("Tip: build a merged dataset in the Datasets page first, then run OLS on that dataset.")
elif recipe_id == "ar1":
    params["y_col"] = st.text_input("y_col", value="value")
elif recipe_id == "lp_irf":
    params["y_col"] = st.text_input("y_col", value="value")
    params["shock_col"] = st.text_input("shock_col", value="shock")
    params["controls"] = st.text_input("controls (comma-separated)", value="")
    params["horizons"] = st.number_input("horizons", min_value=1, max_value=60, value=12, step=1)
    params["ci_level"] = st.selectbox("ci_level", [0.9, 0.95, 0.99], index=1)
    params["add_constant"] = st.checkbox("add_constant", value=True)
elif recipe_id == "news_sentiment":
    params["ts_col"] = st.text_input("ts_col", value="ts")
    params["title_col"] = st.text_input("title_col", value="title")
    params["snippet_col"] = st.text_input("snippet_col", value="snippet")
    params["prefix"] = st.text_input("prefix", value="sent")
    params["batch_size"] = st.number_input("batch_size", min_value=1, max_value=128, value=16, step=1)
    st.caption('Run this on a GDELT news artifact to produce a daily sentiment time series (with a "date" column).')
else:
    st.caption("No UI params for this recipe yet; using empty dict.")

st.markdown("### Run")
col1, col2 = st.columns([1, 2])
with col1:
    run_btn = st.button("Run recipe job", type="primary")
with col2:
    st.caption("This submits a background job that loads the CSV, runs the recipe, and writes a new CSV artifact.")

if run_btn:

    def job_fn(ctx):
        ctx.log("INFO", f"Loading input CSV: {job.output_path}")
        df_in = pd.read_csv(job.output_path)

        ctx.log("INFO", f"Running recipe: {recipe.meta.id} with params={params}")
        df_out = recipe.run(df_in, params=params, ctx=ctx)

        ctx.log("INFO", f"Recipe produced dataframe: shape={df_out.shape}")
        return df_out

    recipe_job_id = runner.submit(
        kind="recipe_run",
        provider_kind="recipe",
        provider_id=recipe.meta.id,
        request={"input_job_id": job.job_id, "input_path": job.output_path, "params": params},
        fn=job_fn,
    )
    st.success(f"Submitted recipe job: {recipe_job_id}")
    st.info("Go to Jobs page to watch status/logs; Preview page to view the new artifact.")

