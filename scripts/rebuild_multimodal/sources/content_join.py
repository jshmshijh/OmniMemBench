#!/usr/bin/env python3
"""Base resolver for sources that must be joined by CONTENT (not filename).

Some sources were re-numbered during benchmark construction so the referenced
filename no longer maps to any source id (ChartQA ``{split}_{n}.png``, LibriSpeech
``librispeech_{n}.wav``). But the crosswalk ``other`` preserves stable content
(the question+answer, or the transcript), so we join on that instead of the name.

Subclasses set ``name``/``buckets``/``HF_REPO`` and implement ``other_key`` and
``row_key`` (which must produce equal keys for the same item), and may override
``parquet_files``, ``media_bytes`` and ``convert``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from sources.base import Resolver, hf_download, hf_list_files  # type: ignore
from crosswalk import SourceRecord  # type: ignore


def _media_columns(table) -> list[str]:
    import pyarrow as pa
    cols = []
    for field in table.schema:
        t = field.type
        if pa.types.is_struct(t) and any(t.field(i).name == "bytes" for i in range(t.num_fields)):
            cols.append(field.name)
    return cols


class ContentJoinResolver(Resolver):
    HF_REPO: str = ""

    def parquet_files(self, records: list[SourceRecord]) -> list[str]:
        return [f for f in hf_list_files(self.HF_REPO) if f.endswith(".parquet")]

    # --- subclass hooks -----------------------------------------------------
    def other_key(self, other: dict):
        raise NotImplementedError

    def row_key(self, row: dict):
        raise NotImplementedError

    def media_bytes(self, row: dict) -> bytes | None:
        media = getattr(self, "_media_cols", None)
        col = media[0] if media else None
        v = row.get(col) if col else None
        return v.get("bytes") if isinstance(v, dict) else None

    def convert(self, data: bytes, dst_path: Path) -> bytes:
        """Optional format conversion (e.g. flac->wav). Default: identity."""
        return data

    # --- driver -------------------------------------------------------------
    def prepare(self, source_root: Path, records: list[SourceRecord]) -> None:
        import pyarrow.parquet as pq

        needed = {self.other_key(r.other) for r in records}
        needed.discard(None)
        self._index: dict = {}
        for pf in self.parquet_files(records):
            dst = source_root / pf
            hf_download(self.HF_REPO, pf, dst, prefix=f"[{self.name}]")
            table = pq.read_table(dst)
            self._media_cols = _media_columns(table)
            for row in table.to_pylist():
                k = self.row_key(row)
                if k in needed and k not in self._index:
                    b = self.media_bytes(row)
                    if b:
                        self._index[k] = b
            print(f"[{self.name}] scanned {pf}: index={len(self._index)}/{len(needed)}",
                  file=sys.stderr)

    def materialize(self, record: SourceRecord, dst_path: Path, source_root: Path) -> bool:
        data = getattr(self, "_index", {}).get(self.other_key(record.other))
        if not data:
            return False
        data = self.convert(data, dst_path)
        if not data:
            return False
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(data)
        return dst_path.is_file() and dst_path.stat().st_size > 0
