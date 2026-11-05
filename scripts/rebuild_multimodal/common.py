#!/usr/bin/env python3
"""Shared paths and helpers for the multimodal rebuild pipeline.

The benchmark data under ``data/*/benchmark_*.json`` references media by relative
paths like ``multimodal/omni/omnibench/audio/3_038.mp3``. The original media files
and source datasets were deleted; only the relative paths + captions remain.

These helpers locate the repo layout and the crosswalk jsonl files that still map
each relative path back to its source-dataset record.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# scripts/rebuild_multimodal/common.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
DATASETS_DIR = REPO_ROOT / "crosswalk" / "datasets"
# The directory that ``multimodal/...`` relative paths are resolved against.
# Media land at ``<out_root>/multimodal/...`` (i.e. the tree lives at repo/multimodal/).
DEFAULT_OUT_ROOT = REPO_ROOT
PIPELINE_DIR = Path(__file__).resolve().parent

# Matches any "multimodal/..." relative path inside a JSON string value.
MM_PATH_RE = re.compile(r'multimodal/[^"\\\s]+')

# All benchmark tiers, canonical first.
TIERS = ["128k", "256k", "512k", "1m"]

# HuggingFace dataset repo for crosswalk files
HF_DATASET_REPO = os.environ.get(
    "OMNIMEMBENCH_HF_REPO", "jshmsjh/OmniMemBench"
)


def ensure_crosswalk():
    """Download crosswalk files from HuggingFace if not present locally."""
    if DATASETS_DIR.is_dir() and any(DATASETS_DIR.rglob("*.jsonl")):
        return
    print(f"Crosswalk not found at {DATASETS_DIR}. Downloading from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(REPO_ROOT),
        allow_patterns=["crosswalk/**"],
    )
    print(f"Crosswalk downloaded to {DATASETS_DIR}")


def source_bucket(rel_path: str) -> str:
    """Return the coarse source key (first 3 path segments) for a media path.

    e.g. ``multimodal/audio/mmau/audio/x.wav`` -> ``multimodal/audio/mmau``.
    This is the grouping used across the ledger and resolver registry.
    """
    parts = rel_path.split("/")
    return "/".join(parts[:3])
