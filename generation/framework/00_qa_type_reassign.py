#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

BUDGET = {
    "MH": 0.15,
    "TR": 0.15,
    "CR": 0.15,
    "ADV": 0.15,
    "UPD": 0.15,
    "REF": 0.15,
    "SH": 0.10
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    return rows


def assign_types(suitability_file: Path, output_file: Path):
    questions = _read_jsonl(suitability_file)

    # Filter out questions with API failures and empty top_types
    valid_questions = [q for q in questions if q.get("api_status") == "success" and q.get("top_types")]

    total = len(valid_questions)
    print(f"Total valid questions: {total} (out of {len(questions)} total)")

    quota = {t: int(total * p) for t, p in BUDGET.items()}
    quota["MH"] += total - sum(quota.values())

    counts = defaultdict(int)

    # Create a mapping from question_id to assignment
    assignments_map = {}

    valid_questions.sort(key=lambda q: q["top_types"][0]["score"], reverse=True)

    for q in valid_questions:
        assigned_type = None
        assigned_subtype = None

        for choice in q["top_types"]:
            t = choice["type"]
            if counts[t] < quota[t]:
                assigned_type = t
                assigned_subtype = choice["subtype"]
                counts[t] += 1
                break

        if not assigned_type:
            t = max(quota.keys(), key=lambda x: quota[x] - counts[x])
            assigned_type = t
            assigned_subtype = "Fallback"
            counts[t] += 1

        assignments_map[q["question_id"]] = {
            "assigned_type": assigned_type,
            "assigned_subtype": assigned_subtype
        }

    # Write output in JSONL format, preserving all original fields
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for q in questions:
            # Start with the original question data
            output_line = dict(q)

            # Add assignment if available
            if q["question_id"] in assignments_map:
                output_line.update(assignments_map[q["question_id"]])
            else:
                # For failed questions, mark them
                output_line["assigned_type"] = None
                output_line["assigned_subtype"] = None

            f.write(json.dumps(output_line, ensure_ascii=False) + "\n")


    # Also save statistics separately
    stats_file = output_file.parent / f"{output_file.stem}_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({"statistics": dict(counts), "total": total}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    default_output_root = base_dir / "output" / "00_qa_type_assign"

    parser = argparse.ArgumentParser(description="Reassign QA types by budget quota.")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=default_output_root)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    input_file = args.input or (args.output_root / args.run_id / "generated_qa_assignments.jsonl")
    output_file = args.output or (args.output_root / args.run_id / "qa_with_assigned_types.jsonl")

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    assign_types(input_file, output_file)
