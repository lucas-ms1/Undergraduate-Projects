from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from finrec.providers.base import Provider, ProviderKind
from finrec.providers.utils.optional import optional_import


@dataclass
class ProviderRegistry:
    _providers: Dict[str, Provider]

    @staticmethod
    def _key(kind: ProviderKind, provider_id: str) -> str:
        return f"{kind}:{provider_id}"

    def register(self, provider: Provider) -> None:
        key = self._key(provider.meta.kind, provider.meta.id)
        self._providers[key] = provider

    def list(self, kind: ProviderKind) -> List[Provider]:
        return [p for p in self._providers.values() if p.meta.kind == kind]

    def get(self, kind: ProviderKind, provider_id: str) -> Provider:
        key = self._key(kind, provider_id)
        if key not in self._providers:
            raise KeyError(f"Provider not found: {key}")
        return self._providers[key]


_REGISTRY: ProviderRegistry | None = None


def _try_register_real_providers(reg: ProviderRegistry) -> None:
    """
    Best-effort: real providers only appear if their optional dependencies are installed.
    This must never raise and must never break the app.
    """
    # Market: yfinance
    try:
        if optional_import("yfinance") is not None:
            from finrec.providers.market.yfinance import YFinanceProvider

            reg.register(YFinanceProvider())
    except Exception:
        pass

    # Market: Financial Modeling Prep (FMP) via requests
    try:
        if optional_import("requests") is not None:
            from finrec.providers.market.fmp import FMPProvider

            reg.register(FMPProvider())
    except Exception:
        pass

    # Macro: FRED via pandas-datareader
    try:
        if optional_import("pandas_datareader.data") is not None:
            from finrec.providers.macro.fred import FREDProvider

            reg.register(FREDProvider())
    except Exception:
        pass

    # News: GDELT doc API (requests)
    try:
        if optional_import("requests") is not None:
            from finrec.providers.news.gdelt import GDELTProvider

            reg.register(GDELTProvider())
    except Exception:
        pass


def get_registry() -> ProviderRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        reg = ProviderRegistry(_providers={})
        _try_register_real_providers(reg)

        _REGISTRY = reg
    return _REGISTRY

