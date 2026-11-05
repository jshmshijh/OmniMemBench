#!/usr/bin/env python3
import json
import shutil
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SRC_DIR = BASE_DIR / "output"

DST_DIR = BASE_DIR / "output_03"

EXCLUDE_RUN_IDS = {
    "00002", "00009", "00010", "00014", "00021", "00025", "00026", "00028",
    "00032", "00033", "00035", "00038", "00045", "00050", "00052", "00056",
    "00057", "00064", "00085", "00094", "00105", "00107",
}


def find_runs(src_dir: Path) -> list[str]:
    return sorted(d.name for d in src_dir.iterdir() if d.is_dir())


def copy_run(run_id: str, src_dir: Path, dst_dir: Path) -> dict[str, dict]:
    src_run = src_dir / run_id
    dst_run = dst_dir / run_id
    dst_run.mkdir(parents=True, exist_ok=True)

    stats = {}
    for src_file in sorted(src_run.glob("benchmark_*.json")):
        dst_file = dst_run / src_file.name
        shutil.copy2(src_file, dst_file)

        bm = json.loads(src_file.read_text("utf-8"))
        qas = bm.get("QAs", [])
        kept_by_type = dict(Counter(qa.get("type") for qa in qas))
        stats[src_file.name] = {"kept": len(qas), "kept_by_type": kept_by_type}

    return stats


def main():
    all_runs = find_runs(SRC_DIR)
    kept_runs = [r for r in all_runs if r not in EXCLUDE_RUN_IDS]
    excluded_runs = [r for r in all_runs if r in EXCLUDE_RUN_IDS]

    print()

    DST_DIR.mkdir(parents=True, exist_ok=True)

    # bm_name -> {kept, removed(runs), kept_by_type, num_runs}
    bm_summary: dict[str, dict] = {}

    for run_id in kept_runs:
        run_stats = copy_run(run_id, SRC_DIR, DST_DIR)
        for bm_name, stat in run_stats.items():
            s = bm_summary.setdefault(bm_name, {"kept": 0, "kept_by_type": {}, "num_runs": 0})
            s["kept"] += stat["kept"]
            s["num_runs"] += 1
            for t, c in stat["kept_by_type"].items():
                s["kept_by_type"][t] = s["kept_by_type"].get(t, 0) + c

    bm_original: dict[str, int] = {}
    for run_id in all_runs:
        for src_file in sorted((SRC_DIR / run_id).glob("benchmark_*.json")):
            bm = json.loads(src_file.read_text("utf-8"))
            bm_original[src_file.name] = bm_original.get(src_file.name, 0) + len(bm.get("QAs", []))

    print("=" * 60)
    for bm_name, s in sorted(bm_summary.items()):
        original = bm_original.get(bm_name, s["kept"])
        removed = original - s["kept"]
        type_info = " ".join(f"{t}:{c}" for t, c in sorted(s["kept_by_type"].items()))
        print(f"[{bm_name}]")

    print("\n" + "-" * 60)
    all_kept = sum(s["kept"] for s in bm_summary.values())
    all_original = sum(bm_original.values())
    all_removed = all_original - all_kept
    all_file_count = sum(s["num_runs"] for s in bm_summary.values())
    if all_original:
        pass


if __name__ == "__main__":
    main()
