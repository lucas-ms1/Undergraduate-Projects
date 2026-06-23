from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional


def optional_import(module_name: str) -> Optional[ModuleType]:
    """
    Import a module if available. Returns None if not installed.
    Never raises ImportError.
    """
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def require_optional(module_name: str, extra_hint: str) -> ModuleType:
    """
    Import a module or raise a friendly error indicating which optional extra to install.
    """
    mod = optional_import(module_name)
    if mod is None:
        extras = ["dev"]
        hint = (extra_hint or "").strip()
        if hint and hint not in extras:
            extras.append(hint)
        extras_str = ",".join(extras)
        raise ImportError(
            f"Optional dependency '{module_name}' is not installed. "
            f'Install with: pip install -e ".[{extras_str}]"'
        )
    return mod

