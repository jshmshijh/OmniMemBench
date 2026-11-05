#!/usr/bin/env python3

import os
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.append(str(Path(__file__).parent.parent / "preprocess"))
from call_gemini import call_gemini, call_gemini_with_usage, call_gemini_multimodal

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

token_stats = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "call_count": 0
}
token_stats_lock = threading.Lock()

DEFAULT_QUESTION_RETRIES = 20
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


def filter_questions_by_type(questions: List[Dict], question_type: str) -> List[Dict]:
    return [q for q in questions if q.get('assigned_type') == question_type]


def group_questions_by_type(questions: List[Dict]) -> Dict[str, List[Dict]]:
    groups = {}
    for question in questions:
        qtype = question.get('assigned_type')
        if qtype:
            groups.setdefault(qtype, []).append(question)
    return groups


def format_persona_info(persona_data: Dict) -> str:
    parts = []
    if 'persona_summary' in persona_data:
        parts.append(f"**Persona Summary:**\n{persona_data['persona_summary']}")
    if 'conversation_style' in persona_data:
        parts.append(f"\n**Conversation Style:**\n{persona_data['conversation_style']}")
    return "\n".join(parts)


def format_multimodal_context(mm_items: List[Dict]) -> str:
    if not mm_items:
        return "N/A"
    lines = []
    for item in mm_items:
        mm_id = item.get("mm_id", "unknown")
        mm_type = item.get("type", "unknown")
        caption = item.get("caption", "")
        lines.append(f"{mm_id} [{mm_type}]: {caption}")
    return "\n".join(lines)


def extract_media_from_question(question_data: Dict) -> List[Dict[str, str]]:
    media = []
    for item in question_data.get("mm_items", []):
        url = item.get("url") or item.get("file_uri")
        if url:
            media.append({
                "url": url,
                "mime_type": item.get("mime_type", "image/jpeg"),
                "modality_type": item.get("type", "image"),
            })
    return media


def load_sessions_mapping(sessions_file: str) -> tuple[Dict, Dict]:
    sessions_data = load_json_data(sessions_file)
    question_to_session = {}
    session_info = {}
    for session in sessions_data['sessions']:
        sid = session['session_id']
        session_info[sid] = session
        for qid in session.get('questions_involved', []):
            question_to_session[qid] = sid
    return question_to_session, session_info


def load_events_by_session(events_file: str, all_sessions: List[Dict]) -> Dict[str, List[str]]:
    events_data = load_json_data(events_file)
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


def format_available_sessions(
    current_session_id: str,
    session_info: Dict,
    all_sessions: List[Dict],
    session_events: Optional[Dict[str, List[str]]] = None,
) -> str:
    current_session = session_info[current_session_id]
    current_date = current_session['date']
    available = [s for s in all_sessions if s['date'] <= current_date]
    available.sort(key=lambda x: x['date'])

    lines = [f"Current question belongs to: Session {current_session_id} ({current_session['date']}, {current_session['time_of_day']})"]
    lines.append("\nAvailable sessions (current and previous):")
    for s in available:
        sid = s['session_id']
        marker = "[CURRENT]" if sid == current_session_id else ""
        lines.append(f"- {sid} ({s['date']}, {s['time_of_day']}): {marker} {s['session_summary']}")
        if session_events and sid in session_events and session_events[sid]:
            for ev in session_events[sid]:
                lines.append(f"    Event: {ev}")
    lines.append(f"\nYou can ONLY use sessions {available[0]['session_id']}-{current_session_id} for evidence.")
    return "\n".join(lines)


def build_rewrite_prompt(
    template: str,
    persona_info: str,
    question_data: Dict,
    sessions_context: str,
    current_session_id: str,
) -> str:
    question_id = question_data.get('question_id', '')
    original_question = question_data.get('question_text', '')
    original_answer = question_data.get('original_answer', 'NA')
    modalities = question_data.get('modalities', [])

    mm_items = question_data.get('mm_items', [])
    if not mm_items:
        multimodal_context = question_data.get('multimodal_context', [])
        caption_str = "\n".join(multimodal_context) if multimodal_context else "N/A"
    else:
        caption_str = format_multimodal_context(mm_items)

    modalities_str = json.dumps(modalities, ensure_ascii=False)

    prompt = template.replace('{persona_info}', persona_info)
    prompt = prompt.replace('{available_sessions}', sessions_context)
    prompt = prompt.replace('{question_session_id}', current_session_id)
    prompt = prompt.replace('{question_id}', question_id)
    prompt = prompt.replace('{original_question}', original_question)
    prompt = prompt.replace('{original_answer}', original_answer)
    prompt = prompt.replace('{modalities}', modalities_str)
    prompt = prompt.replace('{caption}', caption_str)

    return prompt


def parse_rewrite_response(response: str) -> Dict:
    try:
        text = response.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "", 1).strip()
        if text.startswith("```"):
            text = text.replace("```", "", 1).strip()
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "parse_error": f"JSON parsing failed: {str(e)[:200]}",
            "raw_response": response[:500]
        }


def load_success_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()
    ids: set[str] = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("api_status") != "success":
                continue
            qid = obj.get("original_question_id") or obj.get("question_id")
            if qid:
                ids.add(str(qid))
    return ids


def save_rewrite_result(
    result: Dict,
    output_file: Path,
    api_success: bool,
    error_msg: Optional[str] = None
):
    output_line = dict(result)
    if not api_success:
        output_line['api_status'] = 'failed'
        if error_msg:
            output_line['error'] = error_msg
    else:
        output_line['api_status'] = 'success'

    with file_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output_line, ensure_ascii=False) + '\n')


def process_question_with_retry(
    question_data: Dict,
    prompt_template: str,
    persona_info: str,
    output_file: Path,
    question_idx: int,
    total_questions: int,
    question_to_session: Dict,
    session_info: Dict,
    all_sessions: List[Dict],
    session_events: Optional[Dict[str, List[str]]] = None,
    max_retries: int = DEFAULT_QUESTION_RETRIES,
    api_retries: int = DEFAULT_API_RETRIES,
    max_retry_wait_s: int = RETRY_WAIT_CAP_S,
    track_tokens: bool = False,
    use_multimodal: bool = False,
) -> tuple[bool, str]:
    question_id = question_data.get('question_id', 'Unknown')

    if question_id not in question_to_session:
        error_msg = f"Question {question_id} not found in any session"
        save_rewrite_result(
            {"original_question_id": question_id},
            output_file, api_success=False, error_msg=error_msg
        )
        return False, question_id

    current_session_id = question_to_session[question_id]
    sessions_context = format_available_sessions(
        current_session_id, session_info, all_sessions, session_events,
    )

    media = extract_media_from_question(question_data) if use_multimodal else []

    try:
        mm_tag = f", ..." if media else ""
    except:
        pass

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            prompt = build_rewrite_prompt(
                prompt_template, persona_info, question_data,
                sessions_context, current_session_id,
            )

            model = os.environ.get("LLM_MODEL", "your-model-name")

            if media:
                success, response = call_gemini_multimodal(
                    prompt=prompt, media=media, model=model,
                    temperature=0.8, max_tokens=32768,
                    max_retries=api_retries, max_time=300,
                    clean_json=True,
                )
            elif track_tokens:
                success, response, usage = call_gemini_with_usage(
                    prompt=prompt, model=model,
                    temperature=0.8, max_tokens=32768, max_retries=api_retries,
                )
                if success and usage:
                    with token_stats_lock:
                        token_stats["prompt_tokens"] += usage["prompt_tokens"]
                        token_stats["completion_tokens"] += usage["completion_tokens"]
                        token_stats["total_tokens"] += usage["total_tokens"]
                        token_stats["call_count"] += 1
            else:
                success, response = call_gemini(
                    prompt=prompt, model=model,
                    temperature=0.8, max_tokens=32768, max_retries=api_retries,
                )

            if not success:
                last_error = f"API call failed: {response}"
                if attempt < max_retries:
                    _retry_sleep(attempt, max_retry_wait_s)
                    continue
                else:
                    save_rewrite_result(
                        {"original_question_id": question_id},
                        output_file, api_success=False, error_msg=last_error
                    )
                    return False, question_id

            result = parse_rewrite_response(response)

            if 'parse_error' in result:
                last_error = result['parse_error']
                if attempt < max_retries:
                    _retry_sleep(attempt, max_retry_wait_s)
                    continue
                else:
                    save_rewrite_result(result, output_file, api_success=False, error_msg=last_error)
                    return False, question_id

            if 'rewrites' not in result or not isinstance(result['rewrites'], list):
                last_error = "Invalid response format: missing 'rewrites' field"
                if attempt < max_retries:
                    _retry_sleep(attempt, max_retry_wait_s)
                    continue
                else:
                    save_rewrite_result(result, output_file, api_success=False, error_msg=last_error)
                    return False, question_id

            result.setdefault("original_question_id", question_id)
            save_rewrite_result(result, output_file, api_success=True)
            return True, question_id

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                _retry_sleep(attempt, max_retry_wait_s)
                continue

    save_rewrite_result(
        {"original_question_id": question_id},
        output_file, api_success=False, error_msg=last_error
    )
    return False, question_id


def main():
    parser = argparse.ArgumentParser(description='QA Rewrite - ...')
    parser.add_argument('--type', type=str, choices=list(TYPE_CONFIGS.keys()))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--track-tokens', action='store_true')
    parser.add_argument('--multimodal', action='store_true',
                        help="")
    parser.add_argument('--question_retries', type=int, default=DEFAULT_QUESTION_RETRIES)
    parser.add_argument('--api_retries', type=int, default=DEFAULT_API_RETRIES)
    parser.add_argument('--retry_wait_cap_s', type=int, default=RETRY_WAIT_CAP_S)
    parser.add_argument('--run_id', type=str, default='00001')
    args = parser.parse_args()

    types_to_process = [args.type] if args.type else list(TYPE_CONFIGS.keys())

    base_dir = Path(__file__).parent
    step2_output_root = base_dir.parent / "framework" / "output"

    assigned_questions_file = step2_output_root / "00_qa_type_assign" / args.run_id / "qa_with_assigned_types.jsonl"
    persona_file = step2_output_root / "01_persona_gen_caption" / args.run_id / "generated_persona_caption.json"
    events_file = step2_output_root / "02_event_gen_caption" / args.run_id / "generated_events.json"
    sessions_file = step2_output_root / "03_session_gen_stage1" / args.run_id / "generated_sessions_stage1.json"
    prompt_dir = base_dir / "prompt"

    output_dir = base_dir / "output" / "02_qa_rewrite" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("=" * 80)
    print(f"  - Questions: {assigned_questions_file}")
    print(f"  - Persona: {persona_file}")
    print(f"  - Events: {events_file}")
    print(f"  - Sessions: {sessions_file}")
    if args.track_tokens:
        pass
    if args.multimodal:
        pass
    else:
        pass

    all_questions = load_jsonl_data(str(assigned_questions_file))
    persona_data = load_json_data(str(persona_file))
    persona_info = format_persona_info(persona_data)

    question_to_session, session_info = load_sessions_mapping(str(sessions_file))
    sessions_data = load_json_data(str(sessions_file))
    all_sessions = sessions_data['sessions']

    session_events: Optional[Dict[str, List[str]]] = None
    if events_file.exists():
        session_events = load_events_by_session(str(events_file), all_sessions)
        total_events = sum(len(v) for v in session_events.values())
    else:
        pass


    grouped_questions = group_questions_by_type(all_questions)
    for qtype, questions in grouped_questions.items():
        print(f"  - {qtype}: {len(questions)} ...")

    prompt_templates: Dict[str, str] = {}
    output_files: Dict[str, Path] = {}
    for qtype in types_to_process:
        template_file = prompt_dir / TYPE_CONFIGS[qtype]
        if not template_file.exists():
            raise SystemExit(1)
        prompt_templates[qtype] = load_prompt_template(str(template_file))
        output_files[qtype] = output_dir / f"{qtype.lower()}_rewrites.jsonl"

    tasks = []
    total_skipped = 0
    for qtype in types_to_process:
        questions = grouped_questions.get(qtype, [])
        if not questions:
            continue
        ofile = output_files[qtype]
        processed_ids = load_success_ids(ofile)
        total_skipped += len(processed_ids)
        for q in questions:
            if str(q.get("question_id", "")) not in processed_ids:
                tasks.append((q, prompt_templates[qtype], ofile))


    if not tasks:
        return

    print("=" * 80)

    total_success = 0
    total_failed = 0

    with ThreadPoolExecutor(max_workers=50) as executor:
        future_map = {
            executor.submit(
                process_question_with_retry,
                q, tmpl, persona_info, ofile,
                idx + 1, len(tasks),
                question_to_session, session_info, all_sessions,
                session_events,
                args.question_retries, args.api_retries, args.retry_wait_cap_s,
                args.track_tokens,
                args.multimodal,
            ): q
            for idx, (q, tmpl, ofile) in enumerate(tasks)
        }

        for future in as_completed(future_map):
            try:
                success, question_id = future.result()
                if success:
                    total_success += 1
                else:
                    total_failed += 1
            except Exception as e:
                total_failed += 1

    print("\n" + "=" * 80)

    if args.track_tokens and token_stats["call_count"] > 0:
        pass

    if total_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
