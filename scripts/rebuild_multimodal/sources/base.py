#!/usr/bin/env python3
"""Resolver framework for the multimodal rebuild.

A *resolver* knows how to turn a crosswalk :class:`SourceRecord` into an actual
media file on disk. It has two responsibilities:

* ``prepare(source_root, records)`` — ensure the source dataset is available
  locally under ``source_root`` (download / extract), once per run. May be a
  no-op if the user points ``source_root`` at an already-downloaded copy.
* ``materialize(record, dst_path, source_root)`` — place the concrete bytes for
  one referenced media path at ``dst_path`` (copy / convert / PIL-export).

Resolvers are registered by the source bucket(s) they handle (see
``common.source_bucket``). The orchestrator (``rebuild.py``) groups refs by
bucket and dispatches accordingly.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Iterable

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
import common  # type: ignore
from crosswalk import SourceRecord  # type: ignore


class Resolver:
    """Base class. Subclasses set ``name`` and ``buckets`` and implement materialize."""

    name: str = "base"
    buckets: tuple[str, ...] = ()

    def default_source_root(self) -> Path:
        return common.PIPELINE_DIR / "_sources" / self.name

    def prepare(self, source_root: Path, records: list[SourceRecord]) -> None:
        """Ensure source data exists under ``source_root``. Default: no-op."""

    def materialize(self, record: SourceRecord, dst_path: Path, source_root: Path) -> bool:
        raise NotImplementedError


# --- small shared helpers -------------------------------------------------

def copy_file(src: Path, dst: Path) -> bool:
    """Copy ``src`` -> ``dst`` (creating parents). Returns True if a non-empty file lands."""
    if not src.is_file() or src.stat().st_size == 0:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst.is_file() and dst.stat().st_size > 0


def find_first(candidates: Iterable[Path]) -> Path | None:
    for c in candidates:
        if c.is_file() and c.stat().st_size > 0:
            return c
    return None


def download(url: str, dst: Path, prefix: str = "") -> None:
    """Download ``url`` -> ``dst`` via curl with progress bar. Resumes partial
    downloads (``curl -C -``) so interrupted transfers continue. Raises on failure."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    fname = dst.name
    if len(fname) > 40:
        fname = fname[:20] + "..." + fname[-15:]
    print(f"{prefix} downloading {fname}", file=sys.stderr, flush=True)
    subprocess.run(
        ["curl", "-fSL", "-C", "-", "--retry", "5", "--retry-delay", "2",
         "--retry-connrefused", "--progress-bar", "-o", str(tmp), url],
        check=True,
    )
    tmp.replace(dst)


# --- HuggingFace access ---

import os

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def hf_resolve_url(repo: str, filename: str, repo_type: str = "dataset", revision: str = "main") -> str:
    """Build the raw ``resolve`` URL for a repo file (spaces etc. url-encoded)."""
    seg = "datasets/" if repo_type == "dataset" else ("" if repo_type == "model" else f"{repo_type}s/")
    quoted = urllib.parse.quote(filename)
    return f"{HF_ENDPOINT}/{seg}{repo}/resolve/{revision}/{quoted}"


def hf_download(repo: str, filename: str, dst: Path, repo_type: str = "dataset", prefix: str = "") -> None:
    """curl a single file from a HF repo to ``dst`` (skips if already present)."""
    if dst.is_file() and dst.stat().st_size > 0:
        return
    download(hf_resolve_url(repo, filename, repo_type), dst, prefix or f"[hf:{repo}]")


def hf_list_files(repo: str, repo_type: str = "dataset") -> list[str]:
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(repo, repo_type=repo_type)
