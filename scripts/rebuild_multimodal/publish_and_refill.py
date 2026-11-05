#!/usr/bin/env python3
"""Publish the rebuilt media and refill benchmark URLs (optional, for raw-mode eval).

The evaluation harness (``benchmark/run/run_bench.py``, ``benchmark/longctx/utils.py``)
passes each media ``url`` straight to Gemini as a ``fileData.fileUri`` and, in raw
mode, REQUIRES an ``http`` URL. The benchmarks currently carry relative
``multimodal/...`` paths (stripped by ``scripts/normalize_dataset_urls.py``), so raw
mode fails until they point at a reachable URL.

This tool:
  1. serves / hosts the rebuilt ``multimodal/`` tree via a chosen backend, and
  2. rewrites every ``multimodal/...`` occurrence in ``data/*/benchmark_*.json`` to
     the reachable URL (recursively, like ``normalize_dataset_urls.py``; idempotent;
     originals backed up to ``*.bak`` unless disabled).

Backends:
  * ``localhttp`` (default): assume ``<out-root>`` is served at ``--base-url``
    (e.g. ``python -m http.server`` rooted there). URL = ``{base_url}/multimodal/...``.
    NOTE: The LLM API must be able to REACH ``base_url`` — an internal-only host
    works for local encoders but NOT for the external API path.
  * ``prefix``: pure rewrite to ``{base_url}/multimodal/...`` with no hosting (use when
    you host the tree yourself, e.g. object storage / CDN).

Only relative ``multimodal/...`` paths that resolve to an existing rebuilt file are
rewritten (missing media are left untouched and reported), so it is safe to run
before the full rebuild finishes.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore
    import collect_refs  # type: ignore
else:
    from . import common, collect_refs

_REL_RE = re.compile(r"(?<![\w/])multimodal/[^\"'\\\s]+")


def _rewrite_value(value, base_url, out_root, stats):
    if isinstance(value, dict):
        return {k: _rewrite_value(v, base_url, out_root, stats) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_value(v, base_url, out_root, stats) for v in value]
    if isinstance(value, str):
        def repl(m):
            rel = m.group(0)
            if not (out_root / rel).is_file():
                stats["missing"] += 1
                return rel
            stats["rewritten"] += 1
            return f"{base_url.rstrip('/')}/{rel}"
        return _REL_RE.sub(repl, value)
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True,
                    help="Reachable root under which the multimodal/ tree is served, "
                         "e.g. http://host:8000 (files at {base}/multimodal/...).")
    ap.add_argument("--backend", choices=["localhttp", "prefix"], default="localhttp")
    ap.add_argument("--out-root", default=str(common.DEFAULT_OUT_ROOT),
                    help="Dir containing the rebuilt multimodal/ tree. Default: repo root.")
    ap.add_argument("--runs", help="Comma-separated run ids. Default: all.")
    ap.add_argument("--tier", help="Comma-separated tiers. Default: all.")
    ap.add_argument("--no-backup", action="store_true", help="Do not write *.bak files.")
    ap.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    ap.add_argument("--serve", action="store_true",
                    help="(localhttp) print the http.server command to run; does not spawn it.")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root).resolve()

    if args.serve and args.backend == "localhttp":
        print(f"# serve the tree so {args.base_url}/multimodal/... resolves:\n"
              f"cd {out_root} && python -m http.server {args.base_url.rsplit(':', 1)[-1]}")
        return 0

    runs = [x.strip() for x in args.runs.split(",")] if args.runs else None
    tiers = [x.strip() for x in args.tier.split(",")] if args.tier else None

    files: list[Path] = []
    run_dirs = sorted(p for p in common.DATA_DIR.iterdir() if p.is_dir())
    if runs:
        run_dirs = [p for p in run_dirs if p.name in set(runs)]
    for d in run_dirs:
        for f in sorted(d.glob("benchmark_*.json")):
            tier = f.stem.split("benchmark_", 1)[-1]
            if not tiers or tier in set(tiers):
                files.append(f)

    total = {"rewritten": 0, "missing": 0, "files": 0}
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        stats = {"rewritten": 0, "missing": 0}
        new = _rewrite_value(data, args.base_url, out_root, stats)
        total["rewritten"] += stats["rewritten"]
        total["missing"] += stats["missing"]
        if stats["rewritten"] and not args.dry_run:
            if not args.no_backup:
                shutil.copyfile(f, f.with_suffix(f.suffix + ".bak"))
            f.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            total["files"] += 1

    action = "would rewrite" if args.dry_run else "rewrote"
    print(f"{action}: {total['rewritten']} url(s) across {total['files']} file(s); "
          f"{total['missing']} still-missing media left as relative paths", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
