"""Unified configuration system for memory methods.

Usage:
    from omnimembench.config import get_memory_config, list_memory_methods

    config = get_memory_config("NaiveRAG")
    config = get_memory_config("AUGUSTUS", overrides={"recall": {"max_depth": 5}})

    print(list_memory_methods())
"""

from omnimembench.config.base import AttributeDict, MemoryConfig
from omnimembench.config.memory_configs import (
    MEMORY_CONFIGS,
    get_memory_config,
    list_memory_methods,
)

__all__ = [
    "AttributeDict",
    "MemoryConfig",
    "MEMORY_CONFIGS",
    "get_memory_config",
    "list_memory_methods",
]
