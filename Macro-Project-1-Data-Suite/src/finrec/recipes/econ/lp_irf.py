from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.providers.utils.optional import require_optional
from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class LocalProjectionIRFRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="lp_irf",
        name="Local Projection IRF",
        description=(
            "Computes a simple Local Projection impulse response of y to a shock: "
            "for each horizon h, regress y_{t+h} on shock_t (and optional controls)."
        ),
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        sm = require_optional("statsmodels.api", extra_hint="econ")

        y_col = str(params.get("y_col", "value"))
        shock_col = str(params.get("shock_col", "shock"))
        controls_raw = str(params.get("controls", "")).strip()
        horizons = int(params.get("horizons", 12))
        add_constant = bool(params.get("add_constant", True))
        ci_level = float(params.get("ci_level", 0.95))

        if y_col not in df.columns:
            raise ValueError(f"Expected y_col '{y_col}' not found. Columns: {list(df.columns)}")
        if shock_col not in df.columns:
            raise ValueError(f"Expected shock_col '{shock_col}' not found. Columns: {list(df.columns)}")

        controls = [c.strip() for c in controls_raw.split(",") if c.strip()]
        missing = [c for c in controls if c not in df.columns]
        if missing:
            raise ValueError(f"Missing control columns: {missing}. Columns: {list(df.columns)}")

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. y_col={y_col}, shock_col={shock_col}, controls={controls}, horizons={horizons}",
        )

        y = pd.to_numeric(df[y_col], errors="coerce")
        shock = pd.to_numeric(df[shock_col], errors="coerce")
        C = df[controls].apply(pd.to_numeric, errors="coerce") if controls else None

        # Critical value for normal approx (avoid scipy dependency)
        # 95% -> 1.96, 90% -> 1.645, etc.
        def _zcrit(level: float) -> float:
            # Basic mapping for common levels; fallback to 1.96.
            common = {0.9: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
            return common.get(round(level, 2), 1.959963984540054)

        z = _zcrit(ci_level)

        rows: list[dict] = []
        for h in range(horizons + 1):
            y_fwd = y.shift(-h)
            parts = [y_fwd.rename("y_fwd"), shock.rename("shock")]
            if C is not None:
                parts.append(C)
            data = pd.concat(parts, axis=1).dropna()

            if len(data) < 25:
                ctx.log("WARNING", f"[{self.meta.id}] Horizon {h}: insufficient data after dropna (n={len(data)})")
                continue

            Y = data["y_fwd"]
            X = data.drop(columns=["y_fwd"])
            if add_constant:
                X = sm.add_constant(X, has_constant="add")

            res = sm.OLS(Y, X).fit()

            if "shock" not in res.params.index:
                raise RuntimeError("Internal error: expected 'shock' coefficient not found.")

            beta = float(res.params["shock"])
            se = float(res.bse["shock"])
            rows.append(
                {
                    "horizon": h,
                    "irf": beta,
                    "std_err": se,
                    "ci_low": beta - z * se,
                    "ci_high": beta + z * se,
                    "nobs": float(res.nobs),
                }
            )

        out = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
        ctx.log("INFO", f"[{self.meta.id}] Done. horizons_computed={len(out)}")
        return out

