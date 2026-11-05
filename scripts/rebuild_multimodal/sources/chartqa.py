#!/usr/bin/env python3
"""ChartQA + LibriSpeech resolvers (content-join; names were re-numbered).

ChartQA:     multimodal/unified/ChartQA/images/{split}_{n}.png   (HuggingFaceM4/ChartQA)
             join on (question, answer, human_or_machine) -> row 'image' bytes.
LibriSpeech: multimodal/unified/librispeech_test_clean/audio/librispeech_{n}.wav
             (openslr/librispeech_asr, test.clean) join on transcript
             (crosswalk 'answer' == row 'text'); FLAC bytes converted to WAV.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

_PIPE = str(Path(__file__).resolve().parents[1])
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from sources.content_join import ContentJoinResolver  # type: ignore


def _norm_label(v):
    if isinstance(v, list):
        v = v[0] if v else ""
    return str(v).strip()


class ChartQAResolver(ContentJoinResolver):
    name = "chartqa"
    buckets = ("multimodal/unified/ChartQA",)
    HF_REPO = "HuggingFaceM4/ChartQA"

    def other_key(self, other: dict):
        return ((other.get("question") or "").strip(),
                _norm_label(other.get("answer")),
                str(other.get("human_or_machine")))

    def row_key(self, row: dict):
        return ((row.get("query") or "").strip(),
                _norm_label(row.get("label")),
                str(row.get("human_or_machine")))


class LibriSpeechResolver(ContentJoinResolver):
    name = "librispeech"
    buckets = ("multimodal/unified/librispeech_test_clean",)
    HF_REPO = "openslr/librispeech_asr"

    def parquet_files(self, records):
        # Only the test.clean split (matches the benchmark's ~2.6k utterances).
        from sources.base import hf_list_files  # type: ignore
        return [f for f in hf_list_files(self.HF_REPO)
                if f.endswith(".parquet") and "test.clean" in f]

    def _txt(self, s: str) -> str:
        return " ".join((s or "").upper().split())

    def other_key(self, other: dict):
        return self._txt(other.get("answer"))

    def row_key(self, row: dict):
        return self._txt(row.get("text"))

    def convert(self, data: bytes, dst_path: Path) -> bytes:
        # Source audio is FLAC; the benchmark path is .wav -> transcode.
        import soundfile as sf
        arr, sr = sf.read(io.BytesIO(data))
        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV")
        return buf.getvalue()
