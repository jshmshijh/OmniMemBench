#!/usr/bin/env python3
"""MVBench resolver — per-task source zip, extract referenced video members.

Referenced path:   multimodal/video/mvbench/{task}/{video}
Crosswalk ``other``: {"video": "video_7958.mp4"} (or a webm / no-ext name).
Source:            OpenGVLab/MVBench — per-source video zips. Each task draws from
                   a specific zip (canonical VideoChat2/MVBench mapping below).

Only the zips for referenced tasks are downloaded, and only the referenced members
are extracted (zip random-access), so disk stays bounded even for perception.zip
(~10GB). A global basename index across the downloaded zips is used as a fallback
in case a task's video lives in a different zip than the canonical mapping says.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from sources.base import Resolver, hf_download  # type: ignore
from crosswalk import SourceRecord  # type: ignore

HF_REPO = "OpenGVLab/MVBench"

# task (json stem) -> source video zip (stem). Canonical MVBench mapping.
TASK_ZIP = {
    "action_antonym": "ssv2_video",
    "action_count": "perception",
    "action_localization": "sta",
    "character_order": "perception",
    "moving_attribute": "clevrer",
    "moving_count": "clevrer",
    "moving_direction": "clevrer",
    "object_interaction": "star",
    "object_shuffle": "perception",
    "scene_transition": "scene_qa",
    "state_change": "perception",
    "counterfactual_inference": "clevrer",
    "object_existence": "clevrer",
    "action_prediction": "star",
    "action_sequence": "star",
}

# MVBench's curated zips are incomplete: their task jsons reference videos that are
# NOT packaged in the corresponding zip (e.g. action_sequence.json points at Charades
# videos beyond STAR's subset in star.zip). For those we fall back to the ORIGINAL
# dataset, fetching only the needed members via HTTP-range (remotezip).
ORIGINAL_ZIP = {
    "star": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip",
    "clevrer": "http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip",
}


class MVBenchResolver(Resolver):
    name = "mvbench"
    buckets = ("multimodal/video/mvbench",)

    @staticmethod
    def _task(rel_path: str) -> str:
        return rel_path.split("/")[3]

    @staticmethod
    def _video(record: SourceRecord) -> str:
        return record.other.get("video") or Path(record.rel_path).name

    def prepare(self, source_root: Path, records: list[SourceRecord]) -> None:
        needed_zips = {TASK_ZIP[self._task(r.rel_path)] for r in records
                       if self._task(r.rel_path) in TASK_ZIP}
        self._zips: dict[str, zipfile.ZipFile] = {}
        self._by_name: dict[str, tuple[str, str]] = {}   # basename -> (zip, member)
        self._by_stem: dict[str, tuple[str, str]] = {}    # stem -> (zip, member)
        for z in sorted(needed_zips):
            zpath = source_root / f"{z}.zip"
            hf_download(HF_REPO, f"video/{z}.zip", zpath, prefix="[mvbench]")
            zf = zipfile.ZipFile(zpath)
            self._zips[z] = zf
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                base = Path(member).name
                self._by_name.setdefault(base, (z, member))
                self._by_stem.setdefault(Path(base).stem, (z, member))
            print(f"[mvbench] indexed {z}.zip ({len(zf.namelist())} members)", file=sys.stderr)

    def _locate(self, video: str, task: str) -> tuple[str, str] | None:
        # Prefer the canonically-mapped zip, then fall back to any zip by name/stem.
        want_zip = TASK_ZIP.get(task)
        cands = [self._by_name.get(video), self._by_stem.get(Path(video).stem)]
        cands = [c for c in cands if c]
        if want_zip:
            for c in cands:
                if c[0] == want_zip:
                    return c
        return cands[0] if cands else None

    def _fallback_bytes(self, video: str, task: str) -> bytes | None:
        """Fetch a video missing from MVBench's zip from the ORIGINAL dataset
        (Charades / CLEVRER) via HTTP-range, so only the needed member is read."""
        url = ORIGINAL_ZIP.get(TASK_ZIP.get(task, ""))
        if not url:
            return None
        cache = getattr(self, "_remote", None)
        if cache is None:
            cache = self._remote = {}
        if url not in cache:
            try:
                from remotezip import RemoteZip
            except ImportError:
                print("[mvbench] remotezip not installed; cannot fetch original-source videos",
                      file=sys.stderr)
                cache[url] = None
                return None
            print(f"[mvbench] opening original source {url}", file=sys.stderr)
            rz = RemoteZip(url)
            idx: dict[str, str] = {}
            for n in rz.namelist():
                if not n.endswith("/"):
                    idx.setdefault(Path(n).name, n)
                    idx.setdefault(Path(n).stem, n)
            cache[url] = (rz, idx)
        entry = cache[url]
        if not entry:
            return None
        rz, idx = entry
        member = idx.get(video) or idx.get(Path(video).stem)
        return rz.read(member) if member else None

    def materialize(self, record: SourceRecord, dst_path: Path, source_root: Path) -> bool:
        video, task = self._video(record), self._task(record.rel_path)
        hit = self._locate(video, task)
        if hit:
            zname, member = hit
            data = self._zips[zname].read(member)
        else:
            data = self._fallback_bytes(video, task)
        if not data:
            return False
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(data)
        return dst_path.is_file() and dst_path.stat().st_size > 0
