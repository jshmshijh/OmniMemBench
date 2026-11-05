#!/usr/bin/env python3
"""Orchestrator: rebuild the local ``multimodal/`` tree from source datasets.

Flow:  collect refs -> look up crosswalk -> group by source bucket ->
       per source: prepare() then materialize() each file -> verify -> report.

Idempotent: an existing non-empty destination file is skipped. Sources without an
implemented resolver, or files that cannot be located, are recorded in the report
rather than aborting the run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore
    import collect_refs  # type: ignore
    from crosswalk import load_crosswalk, SourceRecord  # type: ignore
    import sources  # type: ignore
else:  # pragma: no cover
    from . import common, collect_refs, sources
    from .crosswalk import load_crosswalk, SourceRecord

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 20) == 0:
                print(f"\r  {desc}: {i}/{total} ({100*i//total}%)", end="", file=sys.stderr)
            yield item
        if total:
            print(f"\r  {desc}: {total}/{total} (100%)", file=sys.stderr)


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_source_roots(items: list[str] | None) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--source-root must be NAME=PATH, got: {it}")
        name, path = it.split("=", 1)
        roots[name.strip()] = Path(path).expanduser().resolve()
    return roots


def _fmt_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", help="Comma-separated run ids (folders). Default: all.")
    ap.add_argument("--tier", help=f"Comma-separated tiers ({'/'.join(common.TIERS)}). Default: all.")
    ap.add_argument("--sources", help="Comma-separated resolver names to run. Default: all implemented.")
    ap.add_argument("--out-root", default=str(common.DEFAULT_OUT_ROOT),
                    help="Dir that multimodal/ paths resolve against; media land at "
                         "<out-root>/multimodal/... . Default: repo root.")
    ap.add_argument("--source-root", action="append",
                    help="Override a resolver's source dir: NAME=PATH (repeatable).")
    ap.add_argument("--report", default=str(common.PIPELINE_DIR / "rebuild_report.json"),
                    help="Path to write the JSON coverage report.")
    ap.add_argument("--no-prepare", action="store_true",
                    help="Skip prepare() (assume source data already present).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be built without downloading or writing media.")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root).resolve()
    source_roots = _parse_source_roots(args.source_root)
    want_sources = set(_parse_csv(args.sources) or [])

    # --- Phase 1: Collect references ---
    print("Phase 1/3: Collecting multimodal references from benchmark data...", file=sys.stderr)
    refs = collect_refs.collect_refs(_parse_csv(args.runs), _parse_csv(args.tier))
    print(f"  Found {len(refs)} media references", file=sys.stderr)

    # --- Phase 2: Load crosswalk ---
    print("Phase 2/3: Loading crosswalk mappings...", file=sys.stderr)
    common.ensure_crosswalk()
    cross = load_crosswalk()
    matched = sum(1 for r in refs if r in cross)
    print(f"  Crosswalk matched: {matched}/{len(refs)}", file=sys.stderr)

    # Group refs by source bucket.
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for r in refs:
        by_bucket[common.source_bucket(r)].append(r)

    # --- Overview ---
    print(f"\nPhase 3/3: {'Checking' if args.dry_run else 'Downloading & materializing'} "
          f"media files...", file=sys.stderr)
    print(f"  Total: {len(refs)} files across {len(by_bucket)} sources", file=sys.stderr)
    print(f"  Output: {out_root}/multimodal/", file=sys.stderr)
    print("", file=sys.stderr)

    report: dict = {
        "refs_total": len(refs),
        "out_root": str(out_root),
        "dry_run": args.dry_run,
        "buckets": {},
        "no_resolver_buckets": [],
        "not_in_crosswalk": [],
    }
    grand = {"found": 0, "skipped": 0, "missing_source": 0, "no_resolver": 0, "no_crosswalk": 0}
    global_done = 0
    global_total = len(refs)
    start_time = time.time()

    sorted_buckets = sorted(by_bucket, key=lambda b: -len(by_bucket[b]))
    for bucket_idx, bucket in enumerate(sorted_buckets, 1):
        paths = by_bucket[bucket]
        resolver = sources.resolver_for_bucket(bucket)
        if resolver is None:
            report["no_resolver_buckets"].append({"bucket": bucket, "count": len(paths)})
            grand["no_resolver"] += len(paths)
            global_done += len(paths)
            print(f"  [{bucket_idx}/{len(sorted_buckets)}] {bucket}: SKIP (no resolver, {len(paths)} files)",
                  file=sys.stderr)
            continue
        if want_sources and resolver.name not in want_sources:
            global_done += len(paths)
            continue

        source_root = source_roots.get(resolver.name, resolver.default_source_root())
        records = [cross[p] for p in paths if p in cross]

        stats = {"resolver": resolver.name, "source_root": str(source_root),
                 "total": len(paths), "found": 0, "skipped": 0,
                 "missing_source": 0, "no_crosswalk": 0, "missing_examples": []}

        # --- Prepare phase (download source data) ---
        if not args.dry_run and not args.no_prepare:
            print(f"  [{bucket_idx}/{len(sorted_buckets)}] {resolver.name}: "
                  f"preparing source data ({len(records)} items)...", file=sys.stderr)
            resolver.prepare(source_root, records)

        # --- Materialize phase ---
        desc = f"{resolver.name}"
        for p in tqdm(paths, desc=f"  [{bucket_idx}/{len(sorted_buckets)}] {desc}",
                      unit="file", file=sys.stderr):
            dst = out_root / p
            if dst.is_file() and dst.stat().st_size > 0:
                stats["skipped"] += 1
                grand["skipped"] += 1
                global_done += 1
                continue
            rec = cross.get(p)
            if rec is None:
                stats["no_crosswalk"] += 1
                grand["no_crosswalk"] += 1
                report["not_in_crosswalk"].append(p)
                global_done += 1
                continue
            if args.dry_run:
                global_done += 1
                continue
            ok = False
            try:
                ok = resolver.materialize(rec, dst, source_root)
            except Exception as e:
                if len(stats["missing_examples"]) < 10:
                    stats["missing_examples"].append(f"{p}  ERR: {e}")
            if ok:
                stats["found"] += 1
                grand["found"] += 1
            else:
                stats["missing_source"] += 1
                grand["missing_source"] += 1
                if len(stats["missing_examples"]) < 10:
                    stats["missing_examples"].append(p)
            global_done += 1

        report["buckets"][bucket] = stats
        elapsed = time.time() - start_time
        rate = global_done / elapsed if elapsed > 0 else 0
        eta = (global_total - global_done) / rate if rate > 0 else 0
        print(f"    -> found={stats['found']} skipped={stats['skipped']} "
              f"missing={stats['missing_source']} "
              f"[{global_done}/{global_total} total, ETA {int(eta)}s]",
              file=sys.stderr)

    # --- Summary ---
    elapsed = time.time() - start_time
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"REBUILD COMPLETE ({elapsed:.1f}s)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Downloaded:    {grand['found']}", file=sys.stderr)
    print(f"  Already exist: {grand['skipped']}", file=sys.stderr)
    print(f"  Missing:       {grand['missing_source']}", file=sys.stderr)
    print(f"  No crosswalk:  {grand['no_crosswalk']}", file=sys.stderr)
    print(f"  No resolver:   {grand['no_resolver']}", file=sys.stderr)
    total_ok = grand["found"] + grand["skipped"]
    print(f"  Success rate:  {total_ok}/{global_total} "
          f"({100*total_ok/global_total:.1f}%)" if global_total else "", file=sys.stderr)

    # Show disk usage if any files were written
    mm_dir = out_root / "multimodal"
    if mm_dir.is_dir():
        import subprocess
        result = subprocess.run(["du", "-sh", str(mm_dir)], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Disk usage:    {result.stdout.split()[0]}", file=sys.stderr)

    report["summary"] = grand
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report: {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
