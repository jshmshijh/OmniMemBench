#!/usr/bin/env python3
"""Build the crosswalk: ``multimodal/...`` relative path -> source-dataset record.

The crosswalk dataset files retain, for every media path, the original source record
in the ``other`` field (original filename / within-dataset relative path / HF id).
This is the bridge that lets us recover which source file each benchmark path came from.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore
else:
    from . import common


@dataclass(frozen=True)
class SourceRecord:
    rel_path: str          # multimodal/... path as referenced by benchmarks
    dataset: str | None    # value of the top-level "dataset" field
    other: dict            # original source record (parsed from "other")
    source_jsonl: str      # crosswalk file this came from (basename)


def _coerce_other(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def load_crosswalk(datasets_dir: Path | None = None) -> dict[str, SourceRecord]:
    """Return ``{rel_path: SourceRecord}`` over all crosswalk jsonl files.

    If the same rel_path appears in multiple files, the first one wins (verified
    consistent for the referenced set). ``scenario_tagged`` is preferred over
    ``original`` when both are present, since it is the newer unified export.
    """
    datasets_dir = datasets_dir or common.DATASETS_DIR
    index: dict[str, SourceRecord] = {}

    # scenario_tagged first (newer/unified), then original, then anything else.
    def sort_key(p: Path) -> int:
        parent = p.parent.name
        return {"scenario_tagged": 0, "original": 1}.get(parent, 2)

    files = sorted(datasets_dir.rglob("*.jsonl"), key=sort_key)
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            other = _coerce_other(d.get("other"))
            mods = d.get("modalities") or {}
            for urls in mods.values():
                if not isinstance(urls, list):
                    continue
                for u in urls:
                    if isinstance(u, str) and u.startswith("multimodal/") and u not in index:
                        index[u] = SourceRecord(
                            rel_path=u,
                            dataset=d.get("dataset"),
                            other=other,
                            source_jsonl=f.name,
                        )
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-refs", action="store_true",
                    help="Load all benchmark refs and report crosswalk coverage.")
    args = ap.parse_args(argv)

    cross = load_crosswalk()
    print(f"crosswalk indexed {len(cross)} unique media paths", file=sys.stderr)

    if args.check_refs:
        import collect_refs  # type: ignore
        refs = set(collect_refs.collect_refs())
        keys = set(cross)
        covered = refs & keys
        missing = refs - keys
        print(f"refs={len(refs)}  covered={len(covered)} ({len(covered)/max(1,len(refs)):.1%})  "
              f"missing={len(missing)}", file=sys.stderr)
        if missing:
            from collections import Counter
            for k, v in Counter(common.source_bucket(m) for m in missing).most_common():
                print(f"  MISSING {v:6d}  {k}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
