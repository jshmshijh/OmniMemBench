#!/usr/bin/env python3

import os
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent / "preprocess"))
from call_gemini import call_gemini


def load_jsonl_data(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def load_prompt_template(template_file: str) -> str:
    with open(template_file, 'r', encoding='utf-8') as f:
        return f.read()


def check_resume_progress(output_file: Path) -> int:
    if not output_file.exists():
        return 0
    count = 0
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def normalize_resume_progress(output_file: Path, batch_size: int) -> int:
    if not output_file.exists():
        return 0

    success_map: Dict[int, str] = {}
    bad_lines: List[str] = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines.append(line)
                continue
            qid = str(obj.get("question_id", "")).strip()
            if not qid.startswith("Q"):
                bad_lines.append(line)
                continue
            try:
                qnum = int(qid[1:])
            except Exception:
                bad_lines.append(line)
                continue
            if obj.get("api_status") == "success":
                success_map.setdefault(qnum, line)
            else:
                bad_lines.append(line)

    ok_lines: List[str] = []
    qnum = 1
    while qnum in success_map:
        ok_lines.append(success_map[qnum])
        qnum += 1

    keep = len(ok_lines)
    if batch_size > 0:
        keep = (keep // batch_size) * batch_size

    total = 0
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    if keep == total:
        return keep

    dropped = ok_lines[keep:] + bad_lines
    dropped_file = output_file.parent / f"{output_file.stem}_dropped_failed.jsonl"
    with open(dropped_file, "w", encoding="utf-8") as f:
        for line in dropped:
            f.write(line if line.endswith("\n") else line + "\n")

    tmp = output_file.with_suffix(output_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for line in ok_lines[:keep]:
            f.write(line if line.endswith("\n") else line + "\n")
    tmp.replace(output_file)

    return keep


def extract_mm_items(data: Dict) -> List[Dict]:
    if 'mm_items' in data and data['mm_items']:
        return data['mm_items']

    mm_items = []
    modalities = data.get('modalities', {})
    if not isinstance(modalities, dict):
        return mm_items

    for modality_type in ('audio', 'video', 'image'):
        for item in modalities.get(modality_type, []):
            if not isinstance(item, dict):
                continue
            mm_id = item.get('mm_id', '')
            caption = ''
            if 'captions' in item and item['captions']:
                caption = item['captions'][0].get('caption', '')
            elif 'caption' in item:
                caption = item.get('caption', '')
            if mm_id and caption:
                mm_items.append({"mm_id": mm_id, "type": modality_type, "caption": caption})

    return mm_items


def format_batch_for_prompt(batch_data: List[Dict], batch_start_idx: int) -> str:
    formatted_lines = []

    for local_idx, data in enumerate(batch_data):
        global_idx = batch_start_idx + local_idx + 1
        question_id = f"Q{global_idx}"

        mm_items = extract_mm_items(data)

        q_str = f"- {question_id}:"
        if mm_items:
            q_str += "\n  Multimodal Items:"
            for item in mm_items:
                q_str += f"\n    - {item['mm_id']} [{item['type']}]: {item['caption']}"
        else:
            q_str += "\n  Multimodal Items: (none)"

        hint = data.get('hint', '')
        if hint:
            q_str += f"\n  Hint: {hint}"

        relations = data.get('relations', '')
        if relations:
            q_str += f"\n  Relations: {relations}"

        formatted_lines.append(q_str)

    return "\n\n".join(formatted_lines)


def build_qa_type_prompt(template: str, batch_data: List[Dict], batch_start_idx: int) -> str:
    questions_section = format_batch_for_prompt(batch_data, batch_start_idx)
    return template.replace("{questions_list}", questions_section)


def parse_qa_type_response(response: str) -> Dict:
    try:
        text = response.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "", 1).strip()
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"assignments": [], "parse_error": f"JSON parsing failed: {str(e)[:200]}"}


def save_batch_results(
    batch_results: Dict,
    batch_data: List[Dict],
    batch_idx: int,
    batch_start_idx: int,
    output_file: Path,
    api_success: bool
):
    assignments = batch_results.get('assignments', [])
    assignment_map = {a.get('question_id', ''): a for a in assignments}

    with open(output_file, 'a', encoding='utf-8') as f:
        for local_idx, data in enumerate(batch_data):
            global_idx = batch_start_idx + local_idx
            question_id = f"Q{global_idx + 1}"

            mm_items = extract_mm_items(data)
            modalities_list = list({item['type'] for item in mm_items})

            output_line = {
                "question_id": question_id,
                "modalities": modalities_list,
                "mm_items": mm_items,
                "categories": data.get('categories', []),
            }

            hint = data.get('hint', '')
            if hint:
                output_line['hint'] = hint
            relations = data.get('relations', '')
            if relations:
                output_line['relations'] = relations

            if api_success and question_id in assignment_map:
                assignment = assignment_map[question_id]
                output_line['top_types'] = assignment.get('top_types', [])
                output_line['api_status'] = 'success'
            else:
                output_line['top_types'] = []
                output_line['api_status'] = 'failed'
                output_line['error'] = batch_results.get('parse_error', 'API call failed or not found in response')

            f.write(json.dumps(output_line, ensure_ascii=False) + '\n')


def process_batch_with_retry(
    batch_data: List[Dict],
    batch_idx: int,
    batch_start_idx: int,
    prompt_template: str,
    output_file: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    api_retries: int,
    max_retries: int = 300
) -> Tuple[bool, str]:
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            prompt = build_qa_type_prompt(prompt_template, batch_data, batch_start_idx)

            success, response = call_gemini(
                prompt=prompt, model=model, temperature=temperature,
                max_tokens=max_tokens, max_retries=api_retries
            )
            elapsed = time.time() - t0

            if not success:
                last_error = f"API call failed ({elapsed:.2f}s): {response}"
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                else:
                    save_batch_results({"assignments": [], "error": last_error},
                                      batch_data, batch_idx, batch_start_idx, output_file, False)
                    return False, last_error

            batch_results = parse_qa_type_response(response)

            if 'parse_error' in batch_results:
                last_error = batch_results['parse_error']
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                else:
                    save_batch_results(batch_results, batch_data, batch_idx, batch_start_idx, output_file, False)
                    return False, last_error

            save_batch_results(batch_results, batch_data, batch_idx, batch_start_idx, output_file, True)
            return True, f"Success ({elapsed:.2f}s)"

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30))
                continue

    save_batch_results({"assignments": [], "error": last_error},
                      batch_data, batch_idx, batch_start_idx, output_file, False)
    return False, last_error


def main():
    base_dir = Path(__file__).parent
    default_qa_root = base_dir.parent / "filter" / "qa"

    parser = argparse.ArgumentParser(description="QA Type Assignment - Based on Multimodal Content Groups")
    parser.add_argument("--qa_root", type=Path, default=default_qa_root)
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--template_file", type=Path, default=base_dir / "prompt" / "00_qa_type_assign.txt")
    parser.add_argument("--output_root", type=Path, default=base_dir / "output" / "00_qa_type_assign")
    parser.add_argument("--output_file", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--batch_retries", type=int, default=3)
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "your-model-name"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--api_retries", type=int, default=20)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    if args.input:
        input_file = args.input
    else:
        mmid_file = args.qa_root / args.run_id / "sampled_data_with_mmid.jsonl"
        fallback_file = args.qa_root / args.run_id / "sampled_data_captioned.jsonl"
        input_file = mmid_file if mmid_file.exists() else fallback_file

    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_file or (output_dir / "generated_qa_assignments.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("QA Type Assignment - Based on Multimodal Content Groups")
    print("=" * 80)

    if not input_file.exists():
        return
    if not args.template_file.exists():
        return

    input_data = load_jsonl_data(str(input_file))
    prompt_template = load_prompt_template(str(args.template_file))
    total = len(input_data)

    processed_count = 0
    if not args.no_resume:
        processed_count = normalize_resume_progress(output_file, batch_size=args.batch_size)
        if processed_count > 0:
            pass

    batch_size = args.batch_size
    total_batches = (total + batch_size - 1) // batch_size
    start_batch_idx = processed_count // batch_size

    print("=" * 80)

    success_count = 0
    failed_count = 0

    for batch_idx in range(start_batch_idx, total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total)
        batch_data = input_data[batch_start:batch_end]

        print(f"\n[Batch {batch_idx + 1}/{total_batches}] Q{batch_start + 1}-Q{batch_end}...")

        success, message = process_batch_with_retry(
            batch_data, batch_idx, batch_start, prompt_template, output_file,
            args.model, args.temperature, args.max_tokens, args.api_retries, args.batch_retries
        )

        if success:
            success_count += len(batch_data)
        else:
            failed_count += len(batch_data)

    print("\n" + "=" * 80)

    final_count = check_resume_progress(output_file)

    if final_count != total or failed_count > 0:
        if final_count != total:
            pass
        if failed_count > 0:
            pass
        raise SystemExit(1)
    else:
        pass


if __name__ == "__main__":
    main()
