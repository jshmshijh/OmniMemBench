#!/usr/bin/env python3
"""ESC-50 resolver — the simplest source (copy-by-original-filename).

Referenced path:   multimodal/unified/esc50/audio/{filename}.wav
Crosswalk ``other``: {"filename": "2-152964-A-31.wav", "audio_path": "audio/..."}
Source:            https://github.com/karolpiczak/ESC-50  (audio/ holds all 2000 wavs
                   named exactly like ``other.filename``).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from sources.base import Resolver, copy_file, find_first, download  # type: ignore
from crosswalk import SourceRecord  # type: ignore

ESC50_ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


class Esc50Resolver(Resolver):
    name = "esc50"
    buckets = ("multimodal/unified/esc50",)

    def _audio_dir(self, source_root: Path) -> Path | None:
        # Accept either <root>/audio or the extracted <root>/ESC-50-master/audio.
        for cand in (source_root / "audio", source_root / "ESC-50-master" / "audio"):
            if cand.is_dir():
                return cand
        return None

    def prepare(self, source_root: Path, records: list[SourceRecord]) -> None:
        if self._audio_dir(source_root):
            return
        source_root.mkdir(parents=True, exist_ok=True)
        zip_path = source_root / "esc50-master.zip"
        if not zip_path.is_file() or zip_path.stat().st_size == 0:
            download(ESC50_ZIP_URL, zip_path, prefix="[esc50]")
        print(f"[esc50] extracting {zip_path.name}", file=sys.stderr)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(source_root)
        if not self._audio_dir(source_root):
            raise RuntimeError(f"[esc50] audio/ not found after extracting into {source_root}")

    def materialize(self, record: SourceRecord, dst_path: Path, source_root: Path) -> bool:
        filename = record.other.get("filename") or Path(record.rel_path).name
        audio_dir = self._audio_dir(source_root)
        if audio_dir is None:
            return False
        src = find_first([audio_dir / filename])
        return copy_file(src, dst_path) if src else False
