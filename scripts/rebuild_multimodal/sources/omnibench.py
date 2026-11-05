#!/usr/bin/env python3
"""MMAU + OmniBench resolvers (parquet-embedded media, byte-exact by ``path``).

MMAU:  multimodal/audio/mmau/audio/{uuid}.wav              (lmms-lab/mmau)
       audio column path == "{uuid}.wav" == referenced basename.
OmniBench (two prefixes, same source m-a-p/OmniBench):
       multimodal/omni/omnibench/{audio,image}/{name}
       multimodal/unified/OmniBench/{audio,image}/{name}   (percent-encoded names)
       audio/image column path == referenced basename (after url-decode).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from sources.parquet_images import ParquetMediaResolver  # type: ignore


class MmauResolver(ParquetMediaResolver):
    name = "mmau"
    buckets = ("multimodal/audio/mmau",)
    HF_REPO = "lmms-lab/mmau"
    # Canonical full MMAU test set (70 shards, uuid-named audio embedded).
    # Index bounded to referenced uuids; audio column path == "{uuid}.wav".


class OmniBenchResolver(ParquetMediaResolver):
    # Handles BOTH the old (omni/omnibench) and new (unified/OmniBench) prefixes;
    # both draw from the same m-a-p/OmniBench parquets and match by path basename.
    name = "omnibench"
    buckets = ("multimodal/omni/omnibench", "multimodal/unified/OmniBench")
    HF_REPO = "m-a-p/OmniBench"
