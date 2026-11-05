#!/usr/bin/env python3
"""Base resolver for HF datasets that embed media (image/audio) in parquet files.

Media columns store ``{"bytes": ..., "path": "<filename>"}``. Columns are detected
dynamically (any struct column carrying a ``bytes`` field), so this handles
``image_1..image_7`` (MMMU/MMMU_Pro), a single ``audio`` (MMAU), or ``audio``+
``image`` (OmniBench) uniformly.

Two match modes:

* **path match** (default): the parquet media ``path`` equals the referenced
  basename (MMMU, MMAU, OmniBench). Byte-exact. Referenced names are url-decoded
  first (some OmniBench paths are percent-encoded).
* **question join** (``question_join=True``): source re-released with drifted
  ids/filenames (MMMU_Pro), so join on the original question text (kept in the
  crosswalk ``other``) plus the image index. Verified 100% question overlap.

Subclasses set ``name``/``buckets``/``HF_REPO`` and optionally ``question_join``
and ``parquet_files()``.
"""
from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from sources.base import Resolver, hf_download, hf_list_files  # type: ignore
from crosswalk import SourceRecord  # type: ignore

_IDX_RE = re.compile(r"_(?:image_)?(\d+)\.\w+$")


def _decode(name: str) -> str:
    return urllib.parse.unquote(name)


def _media_columns(table) -> list[str]:
    """Return columns whose pyarrow type is a struct containing a ``bytes`` field."""
    import pyarrow as pa

    cols = []
    for field in table.schema:
        t = field.type
        if pa.types.is_struct(t) and any(t.field(i).name == "bytes" for i in range(t.num_fields)):
            cols.append(field.name)
    return cols


class ParquetImageResolver(Resolver):
    HF_REPO: str = ""
    question_join: bool = False

    def parquet_files(self, records: list[SourceRecord]) -> list[str]:
        """Repo-relative parquet paths to download+scan. Default: all in the repo."""
        return [f for f in hf_list_files(self.HF_REPO) if f.endswith(".parquet")]

    def _image_index(self, record: SourceRecord) -> int | None:
        """Which image_K column the referenced path corresponds to (1-based)."""
        base = _decode(Path(record.rel_path).name)
        for k in range(1, 8):
            v = record.other.get(f"image_{k}")
            if v and Path(str(v)).name == base:
                return k
        m = _IDX_RE.search(base)
        return int(m.group(1)) if m else None

    def prepare(self, source_root: Path, records: list[SourceRecord]) -> None:
        import pyarrow.parquet as pq

        needed_paths = {_decode(Path(r.rel_path).name) for r in records}
        needed_qk: set[tuple[str, int]] = set()
        if self.question_join:
            for r in records:
                k = self._image_index(r)
                if k:
                    needed_qk.add(((r.other.get("question") or "").strip(), k))

        self._by_path: dict[str, bytes] = {}
        self._by_qk: dict[tuple[str, int], bytes] = {}

        for pf in self.parquet_files(records):
            dst = source_root / pf
            hf_download(self.HF_REPO, pf, dst, prefix=f"[{self.name}]")
            table = pq.read_table(dst)
            media_cols = _media_columns(table)
            if not media_cols:
                continue
            cols = (["question"] if self.question_join else []) + media_cols
            for row in table.select(cols).to_pylist():
                q = (row.get("question") or "").strip() if self.question_join else None
                for k, c in enumerate(media_cols, start=1):
                    img = row.get(c)
                    if not (isinstance(img, dict) and img.get("bytes")):
                        continue
                    p = img.get("path")
                    if p in needed_paths:
                        self._by_path[p] = img["bytes"]
                    if self.question_join and (q, k) in needed_qk:
                        self._by_qk[(q, k)] = img["bytes"]
            print(f"[{self.name}] scanned {pf}: by_path={len(self._by_path)} "
                  f"by_qk={len(self._by_qk)}", file=sys.stderr)

    def materialize(self, record: SourceRecord, dst_path: Path, source_root: Path) -> bool:
        base = _decode(Path(record.rel_path).name)
        data = getattr(self, "_by_path", {}).get(base)
        if data is None and self.question_join:
            k = self._image_index(record)
            q = (record.other.get("question") or "").strip()
            if k:
                data = getattr(self, "_by_qk", {}).get((q, k))
        if not data:
            return False
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(data)
        return dst_path.is_file() and dst_path.stat().st_size > 0


# Back-compat alias: the class now handles audio too.
ParquetMediaResolver = ParquetImageResolver
