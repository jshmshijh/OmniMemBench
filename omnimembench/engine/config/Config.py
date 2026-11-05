"""Engine-level configuration bridge.

This module provides the MemoryConfig class used by memory implementations
to access their configuration as attribute-style objects.
"""

import logging
import os

import yaml


class AttributeDict:
    """Recursively converts a dictionary to attribute-accessible object."""

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, AttributeDict(value))
            else:
                setattr(self, key, value)


class MemoryConfig:
    """Wrapper that memory implementations receive as self.config."""

    def __init__(self, arg_config=None):
        self.args = self._load_arg_config(arg_config)
        gpu = getattr(getattr(self.args, "global_config", None), "usable_gpu", None)
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    def _load_arg_config(self, arg_config):
        if isinstance(arg_config, ArgConfig):
            return arg_config
        return ArgConfig(arg_config)


class ArgConfig:
    """Loads config from dict or YAML file path."""

    def __init__(self, obj=None):
        if obj is None:
            raise ValueError("No parameter obj provided.")
        elif isinstance(obj, dict):
            self.add_args(obj)
        elif isinstance(obj, str):
            with open(obj, "r", encoding="utf-8") as f:
                try:
                    config = yaml.safe_load(f)
                    logging.info("Configuration loaded: %s", config)
                    self.add_args(config)
                except yaml.YAMLError as exc:
                    raise ValueError(f"Failed to load config: {exc}")
        else:
            raise TypeError(f"Unrecognized config type: {type(obj)}")

    def add_args(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, AttributeDict(value))
            else:
                setattr(self, key, value)
