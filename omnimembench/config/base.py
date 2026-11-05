"""Configuration base classes and utilities."""

from __future__ import annotations

import copy
from typing import Any


class AttributeDict(dict):
    """Dictionary subclass that allows attribute-style access."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")
        if isinstance(val, dict) and not isinstance(val, AttributeDict):
            val = AttributeDict(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")


class MemoryConfig(AttributeDict):
    """Configuration object for a memory method.

    Constructed from a base config dict with optional overrides.
    """

    def __init__(self, base: dict, overrides: dict | None = None):
        merged = copy.deepcopy(base)
        if overrides:
            _deep_merge(merged, overrides)
        super().__init__(merged)


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge patch into base, modifying base in place."""
    for k, v in patch.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base
