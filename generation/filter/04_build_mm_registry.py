#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def extract_caption(item: Dict[str, Any]) -> str:
    if item.get("caption"):
        return str(item["caption"])
    caps = item.get("captions", [])
    if isinstance(caps, list) and caps:
        first = caps[0]
        if isinstance(first, dict) and first.get("caption"):
            return str(first["caption"])
    return ""


def build_registry(input_path: Path):
    registry: Dict[str, Dict[str, str]] = {}
    records: List[Dict[str, Any]] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            qid = f"Q{line_num}"

            modalities = record.get("modalities", {})
            if not isinstance(modalities, dict):
                records.append(record)
                continue

            for modality_type in ("audio", "video", "image"):
                items = modalities.get(modality_type, [])
                if not isinstance(items, list):
                    continue
                for idx, item in enumerate(items):
                    if not isinstance(item, dict) or "url" not in item:
                        continue

                    mm_id = f"{qid}_{modality_type}_{idx}"
                    url = str(item["url"])
                    caption = extract_caption(item)

                    registry[mm_id] = {
                        "url": url,
                        "caption": caption,
                        "type": modality_type,
                    }

                    item["mm_id"] = mm_id

            records.append(record)

    return registry, records


def validate_registry(registry: Dict[str, Dict[str, str]]) -> List[str]:
    errors = []
    for mm_id, entry in registry.items():
        if not entry.get("url"):
            errors.append(f"{mm_id}: URL ...")
        if not entry.get("caption"):
            errors.append(f"{mm_id}: caption ...")
    return errors


def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(
        "--input", type=Path, required=True,
        help="",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"...")

    output_dir = args.output_dir or input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = output_dir / "mm_registry.json"
    data_path = output_dir / "sampled_data_with_mmid.jsonl"

    print("=" * 60)
    print("=" * 60)
    print(f"      {data_path}")

    registry, records = build_registry(input_path)

    errors = validate_registry(registry)
    if errors:
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            pass

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    with open(data_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    type_counts: Dict[str, int] = {}
    for entry in registry.values():
        t = entry["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()