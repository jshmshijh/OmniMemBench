#!/usr/bin/env python3
import argparse, json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GA_ROOT = BASE_DIR.parent
DEFAULT_TIER_DIR = GA_ROOT / "evaluate" / "output" / "04_filtered"
RESULT_FILENAME = "benchmark_128k_none_longctx_results.jsonl"


def find_runs(tier_dir: Path) -> list[str]:
    return sorted(d.name for d in tier_dir.iterdir()
                  if d.is_dir() and (d / "benchmark_128k.json").exists())


def load_jsonl(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        return rows
    for line in p.read_text("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def should_filter(row: dict) -> tuple[bool, str, float]:
    if row.get("status") != "ok":
        return False, "", 0
    judge = (row.get("judge") or {}).get("result") or {}
    category = (row.get("category") or "").upper()

    if category == "SUM":
        nr = judge.get("nugget_recall")
        if isinstance(nr, (int, float)) and nr > 0.85:
            return True, "mm_leakage_sum", nr
        return False, "", nr or 0

    score = judge.get("score")
    if isinstance(score, (int, float)) and score == 1:
        return True, "mm_leakage", score
    return False, "", score or 0


def build_qa_index_map(benchmark_path: Path) -> dict[int, str]:
    bm = json.loads(benchmark_path.read_text("utf-8"))
    return {i: qa.get("question_id", "") for i, qa in enumerate(bm.get("QAs", []))}


def filter_benchmark(src: Path, remove_qids: set[str], dst: Path) -> tuple[int, int, dict]:
    bm = json.loads(src.read_text("utf-8"))
    original = len(bm.get("QAs", []))
    bm["QAs"] = [qa for qa in bm.get("QAs", []) if qa.get("question_id") not in remove_qids]
    kept_by_type = dict(Counter(qa.get("type") for qa in bm["QAs"]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(bm, ensure_ascii=False, indent=2), "utf-8")
    return original, len(bm["QAs"]), kept_by_type


def process_run(run_id: str, tier_dir: Path, inter_dir: Path, output_dir: Path) -> dict:
    result_file = inter_dir / run_id / RESULT_FILENAME
    benchmark_128k = tier_dir / run_id / "benchmark_128k.json"

    if not result_file.exists():
        return {"run_id": run_id, "status": "skip", "reason": "no eval results"}
    if not benchmark_128k.exists():
        return {"run_id": run_id, "status": "skip", "reason": "no benchmark_128k"}

    idx_map = build_qa_index_map(benchmark_128k)
    rows = load_jsonl(result_file)

    filter_list = []
    remove_qids = set()
    for row in rows:
        qi = row.get("qa_index")
        if qi is None:
            continue
        hit, reason, score = should_filter(row)
        if hit:
            qid = idx_map.get(qi, f"unknown_{qi}")
            filter_list.append({
                "question_id": qid, "qa_index": qi,
                "reason": reason, "score": score,
                "category": row.get("category", ""),
            })
            remove_qids.add(qid)

    fl_path = inter_dir / run_id / "mm_leakage_filter_list.jsonl"
    fl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in filter_list) + "\n"
        if filter_list else "", "utf-8")

    src_dir = tier_dir / run_id
    dst_dir = output_dir / run_id
    processed = {}
    for src_file in sorted(src_dir.glob("benchmark_*.json")):
        dst_file = dst_dir / src_file.name
        orig, kept, kept_by_type = filter_benchmark(src_file, remove_qids, dst_file)
        processed[src_file.name] = {"original": orig, "kept": kept, "removed": orig - kept, "kept_by_type": kept_by_type}

    return {
        "run_id": run_id, "status": "ok",
        "total_eval": len(rows),
        "filtered": len(filter_list),
        "benchmarks": processed,
        "num_runs": 1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier_dir", type=Path, default=DEFAULT_TIER_DIR)
    args = p.parse_args()

    inter_dir = BASE_DIR / "_intermediate"
    output_dir = BASE_DIR / "output"
    runs = find_runs(args.tier_dir)

    # bm_name -> {original, kept, removed, kept_by_type, num_runs}
    bm_summary: dict[str, dict] = {}

    for rid in runs:
        result = process_run(rid, args.tier_dir, inter_dir, output_dir)
        if result["status"] == "skip":
            continue
        for bm_name, bm_stat in result["benchmarks"].items():
            s = bm_summary.setdefault(bm_name, {"original": 0, "kept": 0, "removed": 0, "kept_by_type": {}, "num_runs": 0})
            s["original"] += bm_stat["original"]
            s["kept"]     += bm_stat["kept"]
            s["removed"]  += bm_stat["removed"]
            s["num_runs"] += 1
            for t, c in bm_stat["kept_by_type"].items():
                s["kept_by_type"][t] = s["kept_by_type"].get(t, 0) + c

    print("\n" + "=" * 60)
    print("=" * 60)
    for bm_name, s in sorted(bm_summary.items()):
        type_info = " ".join(f"{t}:{c}" for t, c in sorted(s["kept_by_type"].items()))
        print(f"[{bm_name}]")
    print("\n" + "-" * 60)
    all_kept = sum(s["kept"] for s in bm_summary.values())
    all_original = sum(s["original"] for s in bm_summary.values())
    all_removed = sum(s["removed"] for s in bm_summary.values())
    all_runs = sum(s["num_runs"] for s in bm_summary.values())
    if all_original:
        pass


if __name__ == "__main__":
    main()