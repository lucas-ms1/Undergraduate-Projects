from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from finrec.jobs.runner import JobRunner
from finrec.providers.registry import get_registry
from finrec.recipes.registry import get_recipe_registry
from finrec.storage.sqlite import SQLiteStorage


@dataclass(frozen=True)
class ProviderFetchSpec:
    kind: str
    provider_id: str
    request: dict


def submit_provider_fetch(*, runner: JobRunner, kind: str, provider_id: str, request: dict) -> str:
    reg = get_registry()
    provider = reg.get(kind, provider_id)

    def job_fn(ctx):
        ctx.log("INFO", f"Executing provider fetch: {provider.meta.kind}:{provider.meta.id}")
        df = provider.fetch(request, ctx=ctx)
        ctx.log("INFO", f"Provider returned dataframe: shape={df.shape}")
        return df

    return runner.submit(
        kind="provider_fetch",
        provider_kind=kind,
        provider_id=provider_id,
        request=request,
        fn=job_fn,
    )


def submit_recipe_run(
    *,
    runner: JobRunner,
    input_job_id: str,
    input_path: str,
    recipe_id: str,
    params: dict,
) -> str:
    recipe_reg = get_recipe_registry()
    recipe = recipe_reg.get(recipe_id)

    def job_fn(ctx):
        ctx.log("INFO", f"Loading input CSV: {input_path}")
        df_in = pd.read_csv(input_path)
        ctx.log("INFO", f"Running recipe: {recipe.meta.id} with params={params}")
        df_out = recipe.run(df_in, params=params, ctx=ctx)
        ctx.log("INFO", f"Recipe produced dataframe: shape={df_out.shape}")
        return df_out

    return runner.submit(
        kind="recipe_run",
        provider_kind="recipe",
        provider_id=recipe.meta.id,
        request={"input_job_id": input_job_id, "input_path": input_path, "params": params},
        fn=job_fn,
    )


def get_jobs_by_id(storage: SQLiteStorage, *, limit: int = 1000) -> Dict[str, Any]:
    jobs = storage.list_jobs(limit=limit)
    return {j.job_id: j for j in jobs}

