#!/usr/bin/env python3
"""
04_filter.py — Remove QAs that failed qa_audit, embed_audit, or uniqueness_audit
from base benchmark and all tier benchmarks.
"""

import json
import argparse
from pathlib import Path


def load_json(p: Path) -> dict:
    return json.loads(p.read_text("utf-8"))


def load_jsonl(p: Path) -> list:
    rows = []
    if not p.exists():
        return rows
    for line in p.read_text("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def collect_bad_ids(audit_dir: Path, run_id: str) -> set:
    bad = set()

    # 01_qa_audit: overall_verdict != "pass"
    qa_audit = audit_dir / "01_qa_audit" / run_id / "qa_audit_results.jsonl"
    for r in load_jsonl(qa_audit):
        if r.get("overall_verdict") != "pass":
            bad.add(r.get("question_id", ""))

    # 02_embed_audit: overall_verdict != "pass"
    embed_audit = audit_dir / "02_embed_audit" / run_id / "embed_audit_results.jsonl"
    for r in load_jsonl(embed_audit):
        if r.get("overall_verdict") != "pass":
            bad.add(r.get("question_id", ""))

    # 03_uniqueness_audit: is_unique != True
    uniq_audit = audit_dir / "03_uniqueness_audit" / run_id / "uniqueness_audit_results.jsonl"
    for r in load_jsonl(uniq_audit):
        if r.get("is_unique") is not True:
            bad.add(r.get("question_id", ""))

    bad.discard("")
    return bad


def filter_benchmark(data: dict, bad_ids: set) -> dict:
    out = dict(data)
    if "QAs" in out:
        out["QAs"] = [qa for qa in out["QAs"] if qa.get("question_id") not in bad_ids]
    return out


def main():
    parser = argparse.ArgumentParser(description="Filter bad QAs from benchmarks")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    audit_dir = Path(__file__).resolve().parent / "output"
    output_dir = args.output_dir or (audit_dir / "04_filtered" / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    bad_ids = collect_bad_ids(audit_dir, args.run_id)
    print(f"Bad QA IDs to remove: {len(bad_ids)}")

    # base benchmark
    bench_dir = base / "session" / "output" / "05_benchmark_assembly" / args.run_id
    tier_dir = base / "session" / "output" / "09_tier_assembly" / args.run_id

    filtered_count = 0
    for src in [bench_dir / "benchmark.json"] + sorted(tier_dir.glob("benchmark_*.json")):
        if not src.exists():
            continue
        data = load_json(src)
        original = len(data.get("QAs", []))
        filtered = filter_benchmark(data, bad_ids)
        remaining = len(filtered.get("QAs", []))
        out_path = output_dir / src.name
        out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), "utf-8")
        print(f"  {src.name}: {original} → {remaining} QAs ({original - remaining} removed)")
        filtered_count += 1

    print(f"\nFiltered {filtered_count} files → {output_dir}")


if __name__ == "__main__":
    main()