#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

# Predefined categories
CATEGORIES = [
    "Work & Study",
    "Life Services",
    "Health & Fitness",
    "Entertainment & Culture",
    "Social & Relationships",
    "Knowledge & Information",
    "Creation & Hobbies"
]

GA_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = GA_ROOT / "preprocess/datasets/scenario_tagged"
DEFAULT_OUTPUT_ROOT = GA_ROOT / "filter/qa"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample QA data for generative agents.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing scenario_tagged *.jsonl files (non-recursive).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory to place outputs under.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="00001",
        help="Output subdirectory name under output_root (e.g. 00001).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Number of entries to sample (capped by available data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If omitted, uses system time.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=CATEGORIES,
        help="Category to filter by. If omitted, selects one randomly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_size <= 0:
        raise SystemExit("--sample_size must be a positive integer.")

    if args.seed is None:
        random.seed()
    else:
        random.seed(args.seed)

    input_dir = args.input_dir
    output_dir = args.output_root / args.run_id

    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = [f for f in input_dir.glob("*.jsonl") if f.is_file()]

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(jsonl_files)} jsonl files:")
    for f in jsonl_files:
        print(f"  - {f.name}")

    all_data = []
    for jsonl_file in jsonl_files:
        print(f"\nReading {jsonl_file.name}...")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    all_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse line {line_num} in {jsonl_file.name}: {e}")

    print(f"\nTotal data entries: {len(all_data)}")

    selected_category = args.category if args.category is not None else random.choice(CATEGORIES)
    print(f"\nSelected category: {selected_category}")

    filtered_data = [entry for entry in all_data if selected_category in entry.get("categories", [])]
    print(f"Data entries in '{selected_category}': {len(filtered_data)}")

    if len(filtered_data) == 0:
        print(f"Warning: No data found for category '{selected_category}'")
        raise SystemExit(1)

    sample_size = min(args.sample_size, len(filtered_data))
    sampled_data = random.sample(filtered_data, sample_size)

    print(f"Sampled {sample_size} entries")

    output_file = output_dir / "sampled_data.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in sampled_data:
            entry.pop("other", None)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nSaved sampled data to: {output_file}")

    dataset_counts = {}
    for entry in sampled_data:
        dataset = entry.get("dataset", "Unknown")
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    print("\nDataset distribution in sample:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset}: {count}")


if __name__ == "__main__":
    main()
