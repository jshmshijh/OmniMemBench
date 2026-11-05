#!/usr/bin/env python3
"""MusicBench resolver — copy by within-tar relative path (streaming extract).

Referenced path:   multimodal/unified/MusicBench/{audio_path}
Crosswalk ``other``: {"audio_path": "FMACaps_eval_set/audiodata/051124_3.wav"} or
                     {"audio_path": "datashare/data/{yt}.wav"}
Source:            amaai-lab/MusicBench — ``FMACaps_eval_set.tar.gz`` (284MB) and
                   ``MusicBench.tar.gz`` (~16GB, holds ``datashare/``).

The 16GB tar is downloaded but only the referenced members are extracted (streamed),
so disk use stays small.
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from sources.base import Resolver, copy_file, hf_download  # type: ignore
from crosswalk import SourceRecord  # type: ignore

HF_REPO = "amaai-lab/MusicBench"
TAR_FOR_PREFIX = {"FMACaps_eval_set": "FMACaps_eval_set.tar.gz", "datashare": "MusicBench.tar.gz"}


def _norm(name: str) -> str:
    return name.lstrip("./")


class MusicBenchResolver(Resolver):
    name = "musicbench"
    buckets = ("multimodal/unified/MusicBench",)

    def _rel(self, record: SourceRecord) -> str:
        # Prefer the crosswalk audio_path; fall back to the path after the bucket.
        ap = record.other.get("audio_path")
        if ap:
            return ap
        return record.rel_path.split("multimodal/unified/MusicBench/", 1)[-1]

    def prepare(self, source_root: Path, records: list[SourceRecord]) -> None:
        needed = {self._rel(r) for r in records}
        # Which tars are actually needed (by top-level prefix).
        tars = {TAR_FOR_PREFIX[p] for p in TAR_FOR_PREFIX
                if any(rel.split("/", 1)[0] == p for rel in needed)}
        for tar_name in sorted(tars):
            missing = {rel for rel in needed if not (source_root / rel).is_file()}
            if not missing:
                continue
            tar_path = source_root / tar_name
            hf_download(HF_REPO, tar_name, tar_path, prefix="[musicbench]")
            print(f"[musicbench] streaming {tar_name}: extracting up to {len(missing)} members",
                  file=sys.stderr)
            got = 0
            with tarfile.open(tar_path, "r|gz") as tf:  # streaming (no random access)
                for m in tf:
                    if not m.isfile():
                        continue
                    rel = _norm(m.name)
                    if rel in missing:
                        dst = source_root / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with tf.extractfile(m) as fsrc:
                            dst.write_bytes(fsrc.read())
                        got += 1
            print(f"[musicbench] extracted {got} from {tar_name}", file=sys.stderr)

    def materialize(self, record: SourceRecord, dst_path: Path, source_root: Path) -> bool:
        src = source_root / self._rel(record)
        return copy_file(src, dst_path) if src.is_file() else False
