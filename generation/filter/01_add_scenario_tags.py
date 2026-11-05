#!/usr/bin/env python3

import json
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))
from call_gemini import call_gemini


GA_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = GA_ROOT / "preprocess/datasets/original"
OUTPUT_DIR = GA_ROOT / "preprocess/datasets/scenario_tagged"

MAX_WORKERS = 100

CATEGORY_DEFINITIONS = """
1. Work & Study: workplace tasks, programming, academic research, coursework, skill training
2. Life Services: shopping, dining, transportation, housework, booking services
3. Health & Fitness: exercise, diet management, medical consultation, mental health, habit tracking
4. Entertainment & Culture: movies/music, gaming, reading, art appreciation, travel experiences
5. Social & Relationships: interpersonal communication, emotional support, family relations, anniversaries, social events
6. Knowledge & Information: encyclopedia, scientific principles, history/culture, current news, concept explanation
7. Creation & Hobbies: writing, painting, photography, crafts, collecting, DIY projects
"""

SENTINEL = None


def build_modality_description(modalities: Dict[str, Any]) -> str:
    descriptions = []

    if "image" in modalities and modalities["image"]:
        descriptions.append(f"Images ({len(modalities['image'])} URLs):\n" +
                          "\n".join(modalities["image"]))

    if "audio" in modalities and modalities["audio"]:
        descriptions.append(f"Audio ({len(modalities['audio'])} URLs):\n" +
                          "\n".join(modalities["audio"]))

    if "video" in modalities and modalities["video"]:
        descriptions.append(f"Video ({len(modalities['video'])} URLs):\n" +
                          "\n".join(modalities["video"]))

    return "\n".join(descriptions) if descriptions else "No media content"


def build_prompt(question: str, modalities: Dict[str, Any]) -> str:
    modality_desc = build_modality_description(modalities)

    prompt = f"""You are a conversation scenario classifier. Analyze the following QA dialogue (may include text, images, audio, video) and classify it into appropriate categories.

【Category Definitions】
{CATEGORY_DEFINITIONS}

【Input】
Question: {question}

Modalities:
{modality_desc}

【Requirements】
- Output ONLY a valid JSON object with this format: {{"categories": ["category_name1", "category_name2", ...]}}
- Select ALL applicable categories based on the question and provided content
- Return category names exactly as defined above
- If no categories apply, return an empty list: {{"categories": []}}
- Return ONLY the JSON, no additional text

Please analyze and classify:"""

    return prompt


def parse_response(response: str) -> List[str]:
    try:
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```.*?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        data = json.loads(text)

        if isinstance(data, dict) and "categories" in data:
            categories = data["categories"]
            if isinstance(categories, list):
                return categories

    except json.JSONDecodeError as e:
        pass

    return []


def classify_record(record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    record_id = record.get("id", "unknown")

    try:
        question = record.get("question", "")
        modalities = record.get("modalities", {})

        if not question:
            record["categories"] = []
            return record_id, record

        prompt = build_prompt(question, modalities)

        success, response = call_gemini(
            prompt=prompt,
            temperature=0.3,
            max_tokens=2048,
            max_retries=3
        )

        if success:
            categories = parse_response(response)
            record["categories"] = categories
        else:
            record["categories"] = []

        return record_id, record

    except Exception as e:
        record["categories"] = []
        return record_id, record


def writer_thread(output_file: Path, result_queue: Queue, total_count: int):
    processed = 0
    pbar = tqdm(total=total_count, desc="  Writing", unit="record")

    with open(output_file, 'a', encoding='utf-8') as f:
        while True:
            item = result_queue.get()

            if item is SENTINEL:
                break

            record_id, record = item
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            processed += 1
            pbar.update(1)

    pbar.close()


def load_processed_ids(output_file: Path) -> Set[str]:
    if not output_file.exists():
        return set()

    processed_ids = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        processed_ids.add(data.get("id", ""))
                    except:
                        pass
    except Exception as e:
        print(f"  Warning: Failed to load processed IDs: {e}")

    return processed_ids


def process_jsonl_file(input_file: Path, output_file: Path) -> Dict[str, Any]:
    stats = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "file": input_file.name
    }

    print(f"\nProcessing: {input_file.name}")

    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    stats["total"] = len(records)
    print(f"  Total records: {stats['total']}")

    processed_ids = load_processed_ids(output_file)
    print(f"  Already processed: {len(processed_ids)}")

    unprocessed_records = [r for r in records if r.get("id") not in processed_ids]
    stats["skipped"] = len(processed_ids)
    stats["processed"] = len(unprocessed_records)

    if not unprocessed_records:
        print(f"  ✓ All records already processed")
        return stats

    print(f"  To process: {len(unprocessed_records)}")

    result_queue: Queue = Queue(maxsize=MAX_WORKERS * 2)

    writer = Thread(
        target=writer_thread,
        args=(output_file, result_queue, len(unprocessed_records)),
        daemon=True
    )
    writer.start()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(classify_record, record): record.get("id")
            for record in unprocessed_records
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="  Classifying", unit="record"):
            try:
                record_id, record = future.result()
                result_queue.put((record_id, record))
                time.sleep(0.05)
            except Exception as e:
                print(f"  Exception in thread: {e}")

    result_queue.put(SENTINEL)
    writer.join()

    print(f"  ✓ Saved to: {output_file.name}")

    return stats


def main():
    print("=" * 80)
    print("Adding Scenario Classification Tags to Multimodal Data")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(INPUT_DIR.glob("*.jsonl"))

    if not jsonl_files:
        print(f"✗ No JSONL files found in {INPUT_DIR}")
        return

    print(f"\nFound {len(jsonl_files)} JSONL files to process")

    all_stats = []
    for input_file in jsonl_files:
        output_file = OUTPUT_DIR / input_file.name
        stats = process_jsonl_file(input_file, output_file)
        all_stats.append(stats)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_records = 0
    total_processed = 0
    total_skipped = 0

    for stats in all_stats:
        total_records += stats["total"]
        total_processed += stats["processed"]
        total_skipped += stats["skipped"]
        print(f"{stats['file']:40s} | Total: {stats['total']:6d} | Processed: {stats['processed']:6d} | Skipped: {stats['skipped']:6d}")

    print(f"\n{'Total':40s} | {total_records:6d} | Processed: {total_processed:6d} | Skipped: {total_skipped:6d}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
