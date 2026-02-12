from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from finrec.recipes.base import Recipe
from finrec.recipes.timeseries.log_returns import LogReturnsRecipe
from finrec.recipes.timeseries.rolling_zscore import RollingZScoreRecipe
from finrec.recipes.technical.sma import SimpleMovingAverageRecipe
from finrec.recipes.technical.ema import ExponentialMovingAverageRecipe
from finrec.recipes.technical.rsi import RSIRecipe
from finrec.recipes.technical.macd import MACDRecipe
from finrec.recipes.technical.rolling_vol import RollingVolatilityRecipe
from finrec.recipes.econ.ols import OLSRecipe
from finrec.recipes.econ.ar1 import AR1Recipe
from finrec.recipes.econ.lp_irf import LocalProjectionIRFRecipe
from finrec.recipes.news.sentiment_finbert import FinBERTDailySentimentRecipe


@dataclass
class RecipeRegistry:
    _recipes: Dict[str, Recipe]

    def register(self, recipe: Recipe) -> None:
        self._recipes[recipe.meta.id] = recipe

    def list(self) -> List[Recipe]:
        return list(self._recipes.values())

    def get(self, recipe_id: str) -> Recipe:
        if recipe_id not in self._recipes:
            raise KeyError(f"Recipe not found: {recipe_id}")
        return self._recipes[recipe_id]


_RECIPE_REGISTRY: RecipeRegistry | None = None


def get_recipe_registry() -> RecipeRegistry:
    global _RECIPE_REGISTRY
    if _RECIPE_REGISTRY is None:
        reg = RecipeRegistry(_recipes={})
        # Register minimal recipes (Prompt #2)
        reg.register(LogReturnsRecipe())
        reg.register(RollingZScoreRecipe())
        # Technical indicators
        reg.register(SimpleMovingAverageRecipe())
        reg.register(ExponentialMovingAverageRecipe())
        reg.register(RSIRecipe())
        reg.register(MACDRecipe())
        reg.register(RollingVolatilityRecipe())
        # Econometric models (optional dependency: statsmodels)
        reg.register(OLSRecipe())
        reg.register(AR1Recipe())
        reg.register(LocalProjectionIRFRecipe())
        # News processing (optional dependency: transformers/torch)
        reg.register(FinBERTDailySentimentRecipe())
        _RECIPE_REGISTRY = reg
    return _RECIPE_REGISTRY

