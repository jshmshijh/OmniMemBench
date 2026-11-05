#!/usr/bin/env python3
"""Resolver registry: source bucket -> resolver instance.

Resolvers are added here as they are implemented. The orchestrator asks this
registry which resolver (if any) handles a given media path's source bucket.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from sources.base import Resolver  # type: ignore
from sources.esc50 import Esc50Resolver  # type: ignore
from sources.flickr30k import Flickr30kResolver  # type: ignore
from sources.mmmu import MmmuResolver, mmmupro_resolvers  # type: ignore
from sources.omnibench import MmauResolver, OmniBenchResolver  # type: ignore
from sources.musicbench import MusicBenchResolver  # type: ignore
from sources.chartqa import ChartQAResolver, LibriSpeechResolver  # type: ignore
from sources.mvbench import MVBenchResolver  # type: ignore

# Instantiate each resolver once.
_RESOLVERS: list[Resolver] = [
    Esc50Resolver(),
    Flickr30kResolver(),
    MmmuResolver(),
    *mmmupro_resolvers(),
    MmauResolver(),
    OmniBenchResolver(),
    MusicBenchResolver(),
    ChartQAResolver(),
    LibriSpeechResolver(),
    MVBenchResolver(),
]

# bucket prefix -> resolver
_BY_BUCKET: dict[str, Resolver] = {}
for _r in _RESOLVERS:
    for _b in _r.buckets:
        _BY_BUCKET[_b] = _r


def resolver_for_bucket(bucket: str) -> Resolver | None:
    return _BY_BUCKET.get(bucket)


def resolver_by_name(name: str) -> Resolver | None:
    for r in _RESOLVERS:
        if r.name == name:
            return r
    return None


def all_resolvers() -> list[Resolver]:
    return list(_RESOLVERS)


def implemented_buckets() -> set[str]:
    return set(_BY_BUCKET)
