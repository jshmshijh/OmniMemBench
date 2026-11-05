#!/usr/bin/env python3
"""Download the OmniMemBench benchmark files from Hugging Face."""

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO = os.environ.get("OMNIMEMBENCH_HF_REPO") or "jshmsjh/OmniMemBench"
TIERS = ["128k", "256k", "512k", "1m"]
METADATA = ["index.jsonl", "croissant.json", "DATASHEET.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", default=",".join(TIERS),
                        help="Comma-separated context tiers to download (default: all four)")
    parser.add_argument("--no_crosswalk", action="store_true",
                        help="Skip crosswalk/, which is only needed to rebuild media")
    parser.add_argument("--revision", default=None,
                        help="Dataset branch, tag, or commit to download (default: main)")
    parser.add_argument("--repo_id", default=DEFAULT_REPO, help="Hugging Face dataset repository")
    parser.add_argument("--local_dir", type=Path, default=REPO_ROOT,
                        help="Destination directory (default: the repository root)")
    args = parser.parse_args()

    tiers = [tier.strip().lower() for tier in args.tiers.split(",") if tier.strip()]
    unknown = sorted(set(tiers) - set(TIERS))
    if not tiers or unknown:
        parser.error(f"--tiers must be a subset of {','.join(TIERS)}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub is required: pip install huggingface_hub")

    patterns = [f"data/*/benchmark_{tier}.json" for tier in tiers] + METADATA
    if not args.no_crosswalk:
        patterns.append("crosswalk/**")

    print(f"Downloading {','.join(tiers)} from {args.repo_id} to {args.local_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=patterns,
    )
    print(f"Done. Benchmark files are in {args.local_dir / 'data'}")


if __name__ == "__main__":
    main()
