from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from finrec.providers.utils.optional import require_optional
from finrec.recipes.base import Recipe, RecipeMeta

_PIPELINE: Any | None = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        transformers = require_optional("transformers", extra_hint="ml")
        # Common FinBERT checkpoint; can be swapped later.
        _PIPELINE = transformers.pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return _PIPELINE


def _label_to_signed_score(label: str, score: float) -> float:
    lab = label.strip().upper()
    if "POS" in lab:
        return float(score)
    if "NEG" in lab:
        return -float(score)
    return 0.0


@dataclass
class FinBERTDailySentimentRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="news_sentiment",
        name="News sentiment (FinBERT, daily)",
        description="Scores each article using FinBERT then aggregates to daily sentiment time series.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        ts_col = str(params.get("ts_col", "ts"))
        title_col = str(params.get("title_col", "title"))
        snippet_col = str(params.get("snippet_col", "snippet"))
        out_prefix = str(params.get("prefix", "sent"))
        batch_size = int(params.get("batch_size", 16))

        if ts_col not in df.columns and "date" not in df.columns:
            raise ValueError(f"Expected '{ts_col}' or 'date' column in input. Columns: {list(df.columns)}")
        if title_col not in df.columns:
            raise ValueError(f"Expected '{title_col}' column in input. Columns: {list(df.columns)}")

        pipe = _get_pipeline()

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. ts_col={ts_col}, title_col={title_col}, snippet_col={snippet_col}, "
            f"batch_size={batch_size}",
        )

        titles = df[title_col].astype(str).fillna("")
        snippets = df[snippet_col].astype(str).fillna("") if snippet_col in df.columns else ""
        text = (titles + ". " + snippets).str.strip()

        # Determine date per row
        if "date" in df.columns and df["date"].astype(str).str.len().gt(0).any():
            dates = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
        else:
            dates = pd.to_datetime(df[ts_col], errors="coerce").dt.date.astype(str)

        results: list[dict[str, Any]] = []
        texts = text.tolist()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            preds = pipe(batch)
            for j, pred in enumerate(preds):
                idx = i + j
                label = str(pred.get("label", ""))
                score = float(pred.get("score", 0.0))
                results.append(
                    {
                        "date": dates.iloc[idx],
                        "label": label,
                        "score": score,
                        f"{out_prefix}_signed": _label_to_signed_score(label, score),
                    }
                )

        scored = pd.DataFrame(results)
        scored = scored[scored["date"].astype(str).str.len() > 0]

        agg = scored.groupby("date", as_index=False).agg(
            n_articles=("label", "count"),
            **{
                f"{out_prefix}_mean": (f"{out_prefix}_signed", "mean"),
                f"{out_prefix}_sum": (f"{out_prefix}_signed", "sum"),
            },
        )

        # label shares
        def _share(df_g: pd.DataFrame, needle: str) -> float:
            return float((df_g["label"].astype(str).str.upper().str.contains(needle)).mean())

        shares = (
            scored.groupby("date")
            .apply(lambda g: pd.Series({"pos_share": _share(g, "POS"), "neg_share": _share(g, "NEG")}))
            .reset_index()
        )
        out = agg.merge(shares, on="date", how="left").sort_values("date").reset_index(drop=True)

        ctx.log("INFO", f"[{self.meta.id}] Done. days={len(out)}")
        return out

