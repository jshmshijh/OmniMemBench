#!/usr/bin/env python3
"""Flickr30K resolver — copy by original filename.

Referenced path:   multimodal/unified/flickr30k/flickr30k-images/{filename}.jpg
Crosswalk ``other``: {"filename": "2280437733.jpg", "img_id": 4487, ...}
Source:            nlphuji/flickr30k on HF — ``flickr30k-images.zip`` holds all images
                   named by their original flickr filename (byte-exact match).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from sources.base import Resolver, copy_file, find_first, hf_download  # type: ignore
from crosswalk import SourceRecord  # type: ignore

HF_REPO = "nlphuji/flickr30k"
IMAGES_ZIP = "flickr30k-images.zip"


class Flickr30kResolver(Resolver):
    name = "flickr30k"
    buckets = ("multimodal/unified/flickr30k",)

    def _images_dir(self, source_root: Path) -> Path | None:
        for cand in (source_root / "flickr30k-images", source_root / "flickr30k_images", source_root):
            if cand.is_dir() and any(cand.glob("*.jpg")):
                return cand
        return None

    def prepare(self, source_root: Path, records: list[SourceRecord]) -> None:
        if self._images_dir(source_root):
            return
        source_root.mkdir(parents=True, exist_ok=True)
        zip_path = source_root / IMAGES_ZIP
        hf_download(HF_REPO, IMAGES_ZIP, zip_path, prefix="[flickr30k]")
        print(f"[flickr30k] extracting {IMAGES_ZIP} (~4.4GB)", file=sys.stderr)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(source_root)
        if not self._images_dir(source_root):
            raise RuntimeError(f"[flickr30k] images dir not found after extracting into {source_root}")

    def materialize(self, record: SourceRecord, dst_path: Path, source_root: Path) -> bool:
        filename = record.other.get("filename") or Path(record.rel_path).name
        img_dir = self._images_dir(source_root)
        if img_dir is None:
            return False
        # Images may sit directly in img_dir or in a nested folder; try both.
        src = find_first([img_dir / filename, source_root / "flickr30k-images" / filename])
        if src is None:
            hits = list(source_root.rglob(filename))
            src = find_first(hits)
        return copy_file(src, dst_path) if src else False
