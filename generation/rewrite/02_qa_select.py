#!/usr/bin/env python3

import os
import json
import sys
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

sys.path.append(str(Path(__file__).parent.parent / "preprocess"))
from call_gemini import call_gemini

TYPE_CONFIGS = {
    "MH": "02_qa_rewrite_mh.txt",
    "CR": "02_qa_rewrite_cr.txt",
    "ADV": "02_qa_rewrite_adv.txt",
    "REF": "02_qa_rewrite_ref.txt",
    "TR": "02_qa_rewrite_tr.txt",
    "UPD": "02_qa_rewrite_upd.txt",
    "SH": "02_qa_rewrite_sh.txt",
}

file_lock = threading.Lock()

DEFAULT_RETRIES = 20
DEFAULT_API_RETRIES = 20
RETRY_WAIT_CAP_S = 30


def _retry_sleep(attempt: int, max_wait_s: int) -> None:
    wait_s = min(2 ** attempt, max(1, int(max_wait_s)))
    time.sleep(wait_s)


def load_jsonl_data(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def load_json_data(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt_template(template_file: str) -> str:
    with open(template_file, 'r', encoding='utf-8') as f:
        return f.read()


def extract_sections(prompt_text: str) -> Dict[str, str]:
    sections = {}
    principles_match = re.search(r'## KEY PRINCIPLES\s*\n(.*?)(?=\n## |\Z)', prompt_text, re.DOTALL)
    sections['principles'] = principles_match.group(1).strip() if principles_match else ""
    subtypes_match = re.search(r'## \w+ SUBTYPES\s*\n(.*?)(?=\n## |\Z)', prompt_text, re.DOTALL)
    sections['subtypes'] = subtypes_match.group(1).strip() if subtypes_match else ""
    examples_match = re.search(r'## EXAMPLES\s*\n(.*?)(?=\n## |\Z)', prompt_text, re.DOTALL)
    sections['examples'] = examples_match.group(1).strip() if examples_match else ""
    requirements_match = re.search(r'## REQUIREMENTS\s*\n(.*?)(?=\n## |\Z)', prompt_text, re.DOTALL)
    sections['requirements'] = requirements_match.group(1).strip() if requirements_match else ""
    return sections


def format_persona_info(persona_data: Dict) -> str:
    parts = []
    if 'persona_summary' in persona_data:
        parts.append(persona_data['persona_summary'])
    if 'conversation_style' in persona_data:
        parts.append(persona_data['conversation_style'])
    return "\n".join(parts)


def build_caption_lookup(questions_file: Path) -> Dict[str, str]:
    if not questions_file.exists():
        return {}
    lookup: Dict[str, str] = {}
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("question_id")
            if not qid:
                continue
            ctx = obj.get("multimodal_context", [])
            if isinstance(ctx, list) and ctx:
                lookup[str(qid)] = "\n".join([str(x) for x in ctx if x])
    return lookup


def build_question_to_session(sessions_data: Dict) -> Dict[str, str]:
    """question_id -> session_id"""
    mapping: Dict[str, str] = {}
    for s in sessions_data.get('sessions', []):
        sid = s.get('session_id', '')
        for qid in s.get('questions_involved', []):
            mapping[str(qid)] = sid
    return mapping


def build_session_events(events_file: Path, all_sessions: List[Dict]) -> Dict[str, List[str]]:
    if not events_file.exists():
        return {}
    events_data = load_json_data(str(events_file))
    event_lookup: Dict[str, str] = {}
    for e in events_data.get('graph', []):
        eid = e.get('id', '')
        sub_event = e.get('sub-event', '')
        date = e.get('date', '')
        if eid and sub_event:
            event_lookup[eid] = f'{eid}: "{sub_event}" ({date})'
    session_events: Dict[str, List[str]] = {}
    for s in all_sessions:
        sid = s['session_id']
        eids = s.get('events_involved', [])
        session_events[sid] = [event_lookup[eid] for eid in eids if eid in event_lookup]
    return session_events


def format_sessions_for_select(
    question_session_id: str,
    all_sessions: List[Dict],
    session_events: Dict[str, List[str]],
) -> str:
    session_map = {s['session_id']: s for s in all_sessions}
    current = session_map.get(question_session_id)
    if not current:
        return "N/A"
    current_date = current['date']
    available = sorted(
        [s for s in all_sessions if s['date'] <= current_date],
        key=lambda x: x['date'],
    )
    lines = [f"Question session: {question_session_id} ({current['date']})", ""]
    for s in available:
        sid = s['session_id']
        marker = " [CURRENT]" if sid == question_session_id else ""
        lines.append(f"- {sid} ({s['date']}, {s['time_of_day']}){marker}: {s['session_summary']}")
        if sid in session_events and session_events[sid]:
            for ev in session_events[sid]:
                lines.append(f"    Event: {ev}")
    return "\n".join(lines)


def build_validation_prompt(
    template: str,
    task_type: str,
    sections: Dict[str, str],
    persona_info: str,
    original_question: str,
    original_answer: str,
    caption: str,
    rewrites_json: str,
    available_sessions: str = "",
) -> str:
    prompt = template.replace('{task_type}', task_type)
    prompt = prompt.replace('{persona_info}', persona_info)
    prompt = prompt.replace('{original_question}', original_question)
    prompt = prompt.replace('{original_answer}', original_answer)
    prompt = prompt.replace('{caption}', caption)
    prompt = prompt.replace('{rewrites_json}', rewrites_json)
    prompt = prompt.replace('{extracted_principles}', sections['principles'])
    prompt = prompt.replace('{extracted_subtypes}', sections['subtypes'])
    prompt = prompt.replace('{extracted_examples}', sections['examples'])
    prompt = prompt.replace('{extracted_requirements}', sections['requirements'])
    prompt = prompt.replace('{available_sessions}', available_sessions)
    return prompt


def parse_validation_response(response: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    try:
        text = response.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "", 1).strip()
        if text.startswith("```"):
            text = text.replace("```", "", 1).strip()
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].strip()
        data = json.loads(text)
        best_id = data.get('best_rewrite_id')
        reasoning = data.get('reasoning', '')
        if best_id is None or not isinstance(best_id, int) or best_id < 1 or best_id > 5:
            return None, None, f"Invalid best_rewrite_id: {best_id}"
        return best_id, reasoning, None
    except json.JSONDecodeError as e:
        return None, None, f"JSON parse error: {str(e)[:100]}"


def get_processed_ids(output_file: Path) -> set:
    if not output_file.exists():
        return set()
    processed = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    qid = data.get('original_question_id')
                    if qid:
                        processed.add(qid)
                except:
                    pass
    return processed


def save_validated_result(result: Dict, output_file: Path):
    with file_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def process_single_validation(
    rewrite_data: Dict,
    validation_template: str,
    sections: Dict[str, str],
    persona_info: str,
    task_type: str,
    caption_lookup: Dict[str, str],
    output_file: Path,
    label: str,
    available_sessions: str = "",
    max_retries: int = DEFAULT_RETRIES,
    api_retries: int = DEFAULT_API_RETRIES,
    max_retry_wait_s: int = RETRY_WAIT_CAP_S,
) -> Tuple[bool, str, str]:
    question_id = rewrite_data.get('original_question_id', 'Unknown')

    rewrites = rewrite_data.get('rewrites', [])
    if not rewrites:
        return False, question_id, task_type


    original_question = rewrite_data.get('original_question', '')
    original_answer = rewrite_data.get('original_answer', '')
    caption = caption_lookup.get(question_id, "N/A")
    rewrites_json = json.dumps(rewrites, ensure_ascii=False, indent=2)

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            prompt = build_validation_prompt(
                template=validation_template, task_type=task_type,
                sections=sections, persona_info=persona_info,
                original_question=original_question,
                original_answer=original_answer,
                caption=caption, rewrites_json=rewrites_json,
                available_sessions=available_sessions,
            )

            success, response = call_gemini(
                prompt=prompt, model=os.environ.get("LLM_MODEL", "your-model-name"),
                temperature=0.3, max_tokens=32768, max_retries=api_retries,
            )

            if not success:
                last_error = f"API failed: {response}"
                if attempt < max_retries:
                    _retry_sleep(attempt, max_retry_wait_s)
                    continue
                else:
                    print(f"  [{label}] ✗ {question_id}: {last_error}")
                    return False, question_id, task_type

            best_id, reasoning, error = parse_validation_response(response)

            if error:
                last_error = error
                if attempt < max_retries:
                    _retry_sleep(attempt, max_retry_wait_s)
                    continue
                else:
                    print(f"  [{label}] ✗ {question_id}: {last_error}")
                    return False, question_id, task_type

            best_rewrite = rewrites[best_id - 1]

            output = {
                "original_question_id": question_id,
                "original_question": original_question,
                "original_answer": original_answer,
                "modalities": rewrite_data.get('modalities', []),
                "best_rewrite": best_rewrite,
                "selection_reasoning": reasoning
            }

            save_validated_result(output, output_file)
        except Exception:
            if attempt < max_retries:
                _retry_sleep(attempt, max_retry_wait_s)
                continue

    print(f"  [{label}] ✗ {question_id}: {last_error}")
    return False, question_id, task_type


def collect_all_tasks(
    types_to_process: List[str],
    rewrite_dir: Path,
    prompt_dir: Path,
    output_dir: Path,
    validation_template: str,
    persona_info: str,
    caption_lookup: Dict[str, str],
    question_to_session: Dict[str, str],
    all_sessions: List[Dict],
    session_events: Dict[str, List[str]],
) -> List[Dict]:
    all_tasks = []

    for task_type in types_to_process:
        rewrite_file = rewrite_dir / f"{task_type.lower()}_rewrites.jsonl"
        prompt_file = prompt_dir / TYPE_CONFIGS[task_type]
        output_file = output_dir / f"{task_type.lower()}_validated.jsonl"

        if not rewrite_file.exists():
            continue
        if not prompt_file.exists():
            raise SystemExit(1)

        rewrite_data_list = load_jsonl_data(str(rewrite_file))
        valid_data = [d for d in rewrite_data_list if d.get('api_status') == 'success']
        processed_ids = get_processed_ids(output_file)

        to_process = [d for d in valid_data if d.get('original_question_id') not in processed_ids]

        print(f"  {task_type}: {len(valid_data)} ...")

        if not to_process:
            continue

        prompt_text = load_prompt_template(str(prompt_file))
        sections = extract_sections(prompt_text)

        for data in to_process:
            qid = data.get('original_question_id', '')
            sid = question_to_session.get(str(qid), '')
            available_sessions_text = ""
            if sid:
                available_sessions_text = format_sessions_for_select(
                    sid, all_sessions, session_events,
                )

            all_tasks.append({
                "rewrite_data": data,
                "validation_template": validation_template,
                "sections": sections,
                "persona_info": persona_info,
                "task_type": task_type,
                "caption_lookup": caption_lookup,
                "output_file": output_file,
                "available_sessions": available_sessions_text,
            })

    return all_tasks


def main():
    parser = argparse.ArgumentParser(description='QA Rewrite Validation')
    parser.add_argument('--type', type=str, choices=list(TYPE_CONFIGS.keys()))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--workers', type=int, default=50)
    parser.add_argument('--max_retries', type=int, default=DEFAULT_RETRIES)
    parser.add_argument('--api_retries', type=int, default=DEFAULT_API_RETRIES)
    parser.add_argument('--retry_wait_cap_s', type=int, default=RETRY_WAIT_CAP_S)
    parser.add_argument('--run_id', type=str, default='00001')
    parser.add_argument('--questions_file', type=Path, default=None)
    parser.add_argument('--source_file', type=Path, default=None)
    args = parser.parse_args()

    types_to_process = [args.type] if args.type else list(TYPE_CONFIGS.keys())

    base_dir = Path(__file__).parent
    step2_output_root = base_dir.parent / "framework" / "output"
    rewrite_dir = base_dir / "output" / "02_qa_rewrite" / args.run_id
    prompt_dir = base_dir / "prompt"
    output_dir = base_dir / "output" / "03_qa_validated" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    persona_file = step2_output_root / "01_persona_gen_caption" / args.run_id / "generated_persona_caption.json"
    persona_data = load_json_data(str(persona_file))
    persona_info = format_persona_info(persona_data)

    questions_file = args.questions_file or (
        step2_output_root / "00_qa_type_assign" / args.run_id / "qa_with_assigned_types.jsonl"
    )
    caption_lookup = build_caption_lookup(questions_file)

    sessions_file = step2_output_root / "03_session_gen_stage1" / args.run_id / "generated_sessions_stage1.json"
    events_file = step2_output_root / "02_event_gen_caption" / args.run_id / "generated_events.json"
    sessions_data = load_json_data(str(sessions_file))
    all_sessions = sessions_data.get('sessions', [])
    question_to_session = build_question_to_session(sessions_data)
    session_events = build_session_events(events_file, all_sessions)

    validation_template_file = prompt_dir / "03_qa_select.txt"
    if not validation_template_file.exists():
        raise SystemExit(1)
    validation_template = load_prompt_template(str(validation_template_file))

    print("=" * 80)
    print("QA Rewrite Validation")
    print("=" * 80)

    all_tasks = collect_all_tasks(
        types_to_process, rewrite_dir, prompt_dir, output_dir,
        validation_template, persona_info, caption_lookup,
        question_to_session, all_sessions, session_events,
    )

    if not all_tasks:
        return


    stats = defaultdict(lambda: {"success": 0, "failed": 0})

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_validation,
                t["rewrite_data"], t["validation_template"], t["sections"],
                t["persona_info"], t["task_type"], t["caption_lookup"],
                t["output_file"],
                f"{t['task_type']}:{idx+1}/{len(all_tasks)}",
                t["available_sessions"],
                args.max_retries, args.api_retries, args.retry_wait_cap_s,
            ): t["task_type"]
            for idx, t in enumerate(all_tasks)
        }

        for future in as_completed(futures):
            task_type = futures[future]
            try:
                success, _, _ = future.result()
                stats[task_type]["success" if success else "failed"] += 1
            except Exception as e:
                stats[task_type]["failed"] += 1

    print("\n" + "=" * 80)
    total_success = total_failed = 0
    for t in types_to_process:
        s, f = stats[t]["success"], stats[t]["failed"]
        total_success += s
        total_failed += f
        if s or f:
            pass

    if total_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
