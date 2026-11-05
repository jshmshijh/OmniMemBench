#!/usr/bin/env python3
"""Collect the worklist of referenced ``multimodal/...`` paths from the benchmarks.

Scans ``data/*/benchmark_*.json`` and extracts every distinct media relative path,
optionally filtered by run id (folder) and tier. This is the authoritative list of
what needs to be rebuilt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore
else:
    from . import common


def _benchmark_files(runs: list[str] | None, tiers: list[str] | None) -> list[Path]:
    files: list[Path] = []
    run_dirs = sorted(p for p in common.DATA_DIR.iterdir() if p.is_dir())
    if runs:
        wanted = set(runs)
        run_dirs = [p for p in run_dirs if p.name in wanted]
    tier_set = set(tiers) if tiers else None
    for d in run_dirs:
        for f in sorted(d.glob("benchmark_*.json")):
            tier = f.stem.split("benchmark_", 1)[-1]
            if tier_set is None or tier in tier_set:
                files.append(f)
    return files


def collect_refs(runs: list[str] | None = None, tiers: list[str] | None = None) -> list[str]:
    """Return sorted unique ``multimodal/...`` paths referenced by the selected benchmarks."""
    refs: set[str] = set()
    for f in _benchmark_files(runs, tiers):
        text = f.read_text(encoding="utf-8")
        for m in common.MM_PATH_RE.findall(text):
            refs.add(m)
    return sorted(refs)


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", help="Comma-separated run ids, e.g. 00001,00007. Default: all.")
    ap.add_argument("--tier", help=f"Comma-separated tiers ({'/'.join(common.TIERS)}). Default: all.")
    ap.add_argument("--out", help="Write refs (one path per line) to this file. Default: stdout summary only.")
    ap.add_argument("--json", action="store_true", help="With --out, write a JSON array instead of newline list.")
    args = ap.parse_args(argv)

    refs = collect_refs(_parse_csv(args.runs), _parse_csv(args.tier))

    # Summary by source bucket to stderr so --out stdout stays clean.
    from collections import Counter
    by = Counter(common.source_bucket(r) for r in refs)
    print(f"collected {len(refs)} unique media refs", file=sys.stderr)
    for k, v in by.most_common():
        print(f"  {v:6d}  {k}", file=sys.stderr)

    if args.out:
        out = Path(args.out)
        if args.json:
            out.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            out.write_text("\n".join(refs) + "\n", encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
