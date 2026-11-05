#!/usr/bin/env python3
"""MMMU + MMMU_Pro resolvers (parquet-embedded images, byte-exact by ``path``).

MMMU:      multimodal/unified/MMMU/images/{id}_{i}.png            (MMMU/MMMU)
MMMU_Pro:  multimodal/image/mmmupro_{4,10}options/{id}_image_{i}.png  (MMMU/MMMU_Pro)

The parquet image column's ``path`` equals the referenced basename, so we simply
scan the relevant parquets and write the matching bytes.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from sources.parquet_images import ParquetImageResolver  # type: ignore
from sources.base import hf_list_files  # type: ignore
from crosswalk import SourceRecord  # type: ignore


class MmmuResolver(ParquetImageResolver):
    name = "mmmu"
    buckets = ("multimodal/unified/MMMU",)
    HF_REPO = "MMMU/MMMU"
    # All parquets (all subjects/splits). Full set is ~1.5GB; correctness over minimization.


class MmmuProResolver(ParquetImageResolver):
    """One instance per option config. ``config_dir`` selects the parquet subset.

    MMMU_Pro was re-released with drifted ids/filenames, so match on question text.
    """
    HF_REPO = "MMMU/MMMU_Pro"
    question_join = True

    def __init__(self, name: str, bucket: str, config_dir: str):
        self.name = name
        self.buckets = (bucket,)
        self.config_dir = config_dir

    def parquet_files(self, records: list[SourceRecord]) -> list[str]:
        return [
            f for f in hf_list_files(self.HF_REPO)
            if f.endswith(".parquet") and f.startswith(self.config_dir + "/")
        ]


def mmmupro_resolvers() -> list[MmmuProResolver]:
    return [
        MmmuProResolver("mmmupro_4options", "multimodal/image/mmmupro_4options", "standard (4 options)"),
        MmmuProResolver("mmmupro_10options", "multimodal/image/mmmupro_10options", "standard (10 options)"),
    ]
