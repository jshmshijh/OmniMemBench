#!/usr/bin/env python3
"""
Add scenario classification tags to unified multimodal data.
Classifies each record's media content into predefined categories using an LLM.
"""

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
from call_gemini import call_gemini_multimodal


GA_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = Path("./local_data/unified")
LOCAL_DATA_ROOT = Path("./local_data")
OUTPUT_DIR = GA_ROOT / "preprocess/datasets/scenario_tagged"

MAX_WORKERS = 100

CATEGORY_DEFINITIONS = """
1. Work & Study: workplace tasks, programming, academic research, coursework, skill training, accounting, science, education, math, computer science
2. Life Services: shopping, dining, transportation, housework, booking services, mobile apps, daily utilities, navigation
3. Health & Fitness: exercise, diet management, medical consultation, mental health, habit tracking, wellness, sports
4. Entertainment & Culture: movies, music, gaming, reading, art appreciation, travel experiences, leisure activities, singing, performance
5. Social & Relationships: interpersonal communication, emotional support, family relations, anniversaries, social events
6. Knowledge & Information: encyclopedia, scientific principles, history/culture, current news, concept explanation, charts, data analysis, diagrams
7. Creation & Hobbies: writing, painting, photography, crafts, collecting, DIY projects, creative works, drawing
"""

SENTINEL = None


def resolve_local_path(dataset: str, relative_path: str) -> Path:
    return LOCAL_DATA_ROOT / dataset / "processed" / relative_path


def get_mime_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
        '.gif': 'image/gif', '.webp': 'image/webp',
        '.mp4': 'video/mp4', '.webm': 'video/webm', '.avi': 'video/x-msvideo', '.mov': 'video/quicktime',
        '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ogg': 'audio/ogg', '.m4a': 'audio/mp4', '.flac': 'audio/flac',
    }
    return mime_types.get(ext, 'application/octet-stream')


def get_modality_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in {'.mp4', '.webm', '.avi', '.mov', '.mkv'}:
        return 'video'
    elif ext in {'.mp3', '.wav', '.ogg', '.m4a', '.flac'}:
        return 'audio'
    return 'image'


def build_media_list(media_urls: Dict[str, List[str]]) -> List[Dict[str, str]]:
    media = []
    for mod_type, urls in media_urls.items():
        for url in urls:
            ext = Path(url.split('?')[0]).suffix.lower()
            media.append({"url": url, "mime_type": get_mime_type(f"f{ext}"), "modality_type": mod_type})
    return media


def build_classification_prompt() -> str:
    return f"""Analyze the provided multimodal content and classify it into appropriate scenario categories.

Category Definitions:
{CATEGORY_DEFINITIONS}

Requirements:
- Output ONLY a valid JSON object: {{"categories": ["category_name1", ...]}}
- Select ALL applicable categories
- Return category names exactly as defined above
- If none apply, return: {{"categories": []}}
- Return ONLY the JSON, no additional text

Please analyze and classify:"""


def parse_classification_response(response: str) -> List[str]:
    try:
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```.*?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("categories"), list):
            return data["categories"]
    except json.JSONDecodeError:
        pass
    return []


def process_single_record(record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    record_id = record.get("id", "unknown")

    try:
        modalities = record.get("modalities", {})
        media_urls = {"image": [], "audio": [], "video": []}
        for mod_type, paths in modalities.items():
            if not paths:
                continue
            for relative_path in paths:
                local_path = resolve_local_path(record.get("dataset", ""), relative_path)
                if local_path.exists():
                    media_urls[mod_type].append(str(local_path))

        media = build_media_list(media_urls)
        categories = []
        if media:
            try:
                success, response = call_gemini_multimodal(
                    prompt=build_classification_prompt(),
                    media=media, temperature=0.3, max_tokens=500, max_retries=3, max_time=120
                )
                if success:
                    categories = parse_classification_response(response)
            except Exception as e:
                print(f"  Error classifying {record_id}: {e}")

        time.sleep(0.1)

        record["local_paths"] = record.get("modalities", {})
        record["modalities"] = media_urls
        record["categories"] = categories
        return record_id, record

    except Exception as e:
        print(f"  Error processing {record_id}: {e}")
        record["local_paths"] = record.get("modalities", {})
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
            _, record = item
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
                        processed_ids.add(json.loads(line.strip()).get("id", ""))
                    except:
                        pass
    except Exception as e:
        print(f"  Warning: Failed to load processed IDs: {e}")
    return processed_ids


def process_jsonl_file(input_file: Path, output_file: Path) -> Dict[str, Any]:
    stats = {"total": 0, "processed": 0, "skipped": 0, "file": input_file.name}

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

    unprocessed = [r for r in records if r.get("id") not in processed_ids]
    stats["skipped"] = len(processed_ids)
    stats["processed"] = len(unprocessed)

    if not unprocessed:
        print(f"  All records already processed")
        return stats

    print(f"  To process: {len(unprocessed)}")

    result_queue: Queue = Queue(maxsize=MAX_WORKERS * 2)
    writer = Thread(target=writer_thread, args=(output_file, result_queue, len(unprocessed)), daemon=True)
    writer.start()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_record, r): r.get("id") for r in unprocessed}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Processing", unit="record"):
            try:
                result_queue.put(future.result())
            except Exception as e:
                print(f"  Exception: {e}")

    result_queue.put(SENTINEL)
    writer.join()
    print(f"  Saved to: {output_file.name}")
    return stats


def main():
    print("=" * 80)
    print("Adding Scenario Classification Tags to Unified Multimodal Data")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(INPUT_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {INPUT_DIR}")
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
    total_r = total_p = total_s = 0
    for s in all_stats:
        total_r += s["total"]; total_p += s["processed"]; total_s += s["skipped"]
        print(f"{s['file']:40s} | Total: {s['total']:6d} | Processed: {s['processed']:6d} | Skipped: {s['skipped']:6d}")
    print(f"\n{'Total':40s} | {total_r:6d} | Processed: {total_p:6d} | Skipped: {total_s:6d}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
