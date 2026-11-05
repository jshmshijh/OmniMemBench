#!/usr/bin/env python3

import os
import json
import sys
import time
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent / "preprocess"))
from call_gemini import call_gemini


def load_jsonl(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_evidence_by_session(qa_data: List[Dict], mm_registry: Dict) -> Dict[str, List[Dict]]:
    session_evidence = {}
    for q in qa_data:
        rewrite = q.get("best_rewrite", q)
        qid = rewrite.get("rewrite_id", q.get("question_id", "unknown"))
        original_qid = q.get("question_id", f"Q{qid}")
        question_text = q.get("question", "")

        for hop_idx, hop in enumerate(rewrite.get("evidence_chain", []), start=1):
            sid = hop.get("session_id")
            if not sid:
                continue
            evidence_item = {
                "evidence_id": f"{original_qid}_hop{hop_idx}",
                "type": hop.get("type", "text"),
                "content": hop.get("evidence", ""),
                "question": question_text,
            }
            mm_id = hop.get("mm_id")
            if mm_id:
                evidence_item["mm_id"] = mm_id
                caption = mm_registry.get(mm_id, {}).get("caption", "")
                if caption:
                    evidence_item["caption"] = caption

            session_evidence.setdefault(sid, []).append(evidence_item)

    return session_evidence


_SANITIZE_PROMPT = """\
You are a content sanitizer for a multimodal memory benchmark.

Your task: convert each media caption into a **safe context hint** that preserves the DOMAIN and SCENARIO direction but removes all perceptual details and specific objects.

KEEP: academic/professional domain, activity type, scenario/setting type, emotional tone
REMOVE: specific objects, components, labels, spatial layout, colors, shapes, materials, quantities, text content, sounds, appearances

The goal is to give a dialogue planner enough domain context to build a natural scene, without revealing what the media actually depicts.

Examples:
- "A schematic diagram of a gas turbine cycle with compressor, combustion chamber, and turbine at specific pressures and temperatures" → "related to a thermodynamics engineering concept"
- "A brown dog running on a sandy beach" → "related to pets and outdoor activities"
- "A person using a hairdryer to blow dust off a pink stick" → "related to an unconventional use of a household appliance"
- "Sheet music for a jazz piano piece in C minor" → "related to a music composition the user is working on"
- "Two children playing in a park with red balloons" → "related to a family outing"
- "A messy kitchen counter with spilled coffee and scattered papers" → "related to a household situation"
- "A Python code snippet showing a recursive Fibonacci implementation with memoization" → "related to a computer science programming concept"
- "An X-ray image showing a hairline fracture in the left radius bone" → "related to a medical imaging case"
- "A candlestick chart showing Tesla stock price movements over 3 months" → "related to a financial markets analysis"

Now sanitize the following captions. Return ONLY a JSON object mapping each id to its sanitized hint, no other text.

{items}
"""

_hint_cache: Dict[str, str] = {}


def sanitize_mm_captions(
    evidence_list: List[Dict],
    mm_registry: Dict,
) -> Dict[str, str]:
    to_sanitize: Dict[str, str] = {}
    eid_to_mmid: Dict[str, str] = {}

    for e in evidence_list:
        if not _is_mm_type(e.get("type", "")):
            continue
        mm_id = e.get("mm_id")
        if not mm_id:
            continue
        eid_to_mmid[e["evidence_id"]] = mm_id
        if mm_id in _hint_cache:
            continue
        caption = mm_registry.get(mm_id, {}).get("caption", "")
        if caption:
            to_sanitize[mm_id] = caption

    if to_sanitize:
        items_text = "\n".join(f'- id="{mid}": "{cap}"' for mid, cap in to_sanitize.items())
        prompt = _SANITIZE_PROMPT.replace("{items}", items_text)

        success, response = call_gemini(
            prompt=prompt, model=os.environ.get("LLM_MODEL", "your-model-name"),
            temperature=0.3, max_tokens=4096, max_retries=3
        )
        if success:
            try:
                hints = extract_json_from_response(response)
                for mid, hint in hints.items():
                    _hint_cache[mid] = hint
            except ValueError:
                pass

        for mid in to_sanitize:
            if mid not in _hint_cache:
                _hint_cache[mid] = "related to multimedia content shared by the user"

    result = {}
    for eid, mid in eid_to_mmid.items():
        result[eid] = _hint_cache.get(mid, "related to multimedia content shared by the user")
    return result


def format_previous_summaries(previous_results: List[Dict], window_size: int = 5) -> str:
    if not previous_results:
        return "No previous sessions."
    lines = []
    boundary = max(0, len(previous_results) - window_size)
    for r in previous_results[:boundary]:
        lines.append(f"- {r['session_id']}: {r.get('summary', '(no summary)')}")
    if boundary > 0:
        lines.append("")
    for r in previous_results[boundary:]:
        lines.append(f"【{r['session_id']}】")
        lines.append(r.get("summary", "(no summary)"))
        lines.append("")
    return "\n".join(lines)


def format_session_outline(session: Dict) -> str:
    parts = [
        f"Session ID: {session['session_id']}",
        f"Date: {session['date']}",
    ]
    if session.get('time_of_day'):
        parts.append(f"Time: {session['time_of_day']}")
    if session.get('events_involved'):
        parts.append(f"Events: {', '.join(session['events_involved'])}")
    if session.get('trigger'):
        parts.append(f"Trigger: {session['trigger']}")
    return "\n".join(parts)


def _is_mm_type(t: str) -> bool:
    return t.lower() in {"image", "audio", "video"}


def format_evidence_list(evidence: List[Dict], context_hints: Dict[str, str] = None) -> str:
    if not evidence:
        return "No evidence to embed in this session."
    if context_hints is None:
        context_hints = {}
    lines = []
    for e in evidence:
        mm_tag = f" (mm_id={e['mm_id']})" if e.get('mm_id') else ""
        question_str = f"\n  question: \"{e['question']}\"" if e.get("question") else ""

        if _is_mm_type(e.get("type", "")):
            caption_str = f"\n  caption: \"{e['caption']}\"" if e.get("caption") else ""
            hint = context_hints.get(e["evidence_id"], "")
            hint_str = f"\n  context_hint: \"{hint}\"" if hint else ""
            lines.append(f"- {e['evidence_id']} [{e['type']}]{mm_tag}{caption_str}{hint_str}{question_str}")
        else:
            lines.append(f"- {e['evidence_id']} [{e['type']}]{mm_tag}: {e['content']}{question_str}")
    return "\n".join(lines)


def extract_json_from_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{.*\}', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError("...")


def _to_int(value) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def validate_result(result: Dict, evidence_ids: set) -> List[str]:
    errors = []
    placement = result.get("evidence_placement", {})
    if not isinstance(placement, dict):
        return ["evidence_placement ..."]

    phases = result.get("dialogue_flow", {}).get("phases", [])
    valid_phase_ids: set[int] = set()
    if not isinstance(phases, list):
        errors.append("dialogue_flow.phases ...")
        phases = []
    for idx, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            errors.append("phase is not a dict")
            continue
        pid_i = _to_int(phase.get("phase_id"))
        if pid_i is None:
            errors.append("phase has no valid phase_id")
            continue
        valid_phase_ids.add(pid_i)

    placed = set(placement.keys())
    missing = evidence_ids - placed
    extra = placed - evidence_ids
    if missing:
        errors.append(f"...")
    if extra:
        errors.append(f"...")

    for eid, phase_id in placement.items():
        pid_i = _to_int(phase_id)
        if pid_i is None:
            errors.append(f"evidence_placement[{eid}] ...")
            continue
        if valid_phase_ids and pid_i not in valid_phase_ids:
            errors.append(f"evidence_placement[{eid}] ...")
    return errors


def process_session(
    current_session: Dict,
    previous_results: List[Dict],
    evidence: List[Dict],
    context_hints: Dict[str, str],
    prompt_template: str,
    output_dir: Path,
    window_size: int = 5,
    max_retries: int = 3
) -> Tuple[bool, Dict, float]:

    start_time = time.time()
    session_id = current_session['session_id']
    result_file = output_dir / f"{session_id}.json"
    raw_file = output_dir / f"{session_id}_raw.txt"

    if result_file.exists():
        result = load_json(str(result_file))
        elapsed = time.time() - start_time
        return True, result, elapsed

    prompt = prompt_template.replace("{previous_summaries}", format_previous_summaries(previous_results, window_size))
    prompt = prompt.replace("{session_outline}", format_session_outline(current_session))
    prompt = prompt.replace("{evidence_list}", format_evidence_list(evidence, context_hints))
    prompt = prompt.replace("{num_evidence}", str(len(evidence)))
    prompt = prompt.replace("{session_id}", session_id)

    evidence_ids = {e["evidence_id"] for e in evidence}
    evidence_mm_map = {e["evidence_id"]: e.get("mm_id") for e in evidence}
    evidence_question_map = {e["evidence_id"]: e.get("question", "") for e in evidence}
    evidence_caption_map = {e["evidence_id"]: e.get("caption", "") for e in evidence}

    for attempt in range(1, max_retries + 1):
        try:
            success, response = call_gemini(
                prompt=prompt, model=os.environ.get("LLM_MODEL", "your-model-name"),
                temperature=0.7, max_tokens=32768, max_retries=5
            )
            if not success:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                elapsed = time.time() - start_time
                return False, {"error": f"API..."}, elapsed

            raw_file.write_text(response, encoding='utf-8')
            result = extract_json_from_response(response)

            if evidence:
                errors = validate_result(result, evidence_ids)
                if errors:
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue

            for phase in result.get("dialogue_flow", {}).get("phases", []):
                for ev in phase.get("evidence", []):
                    eid = ev.get("evidence_id")
                    if eid and evidence_mm_map.get(eid):
                        ev["mm_id"] = evidence_mm_map[eid]
                    if eid and eid in context_hints:
                        ev["context_hint"] = context_hints[eid]
                    if eid and evidence_question_map.get(eid):
                        ev["question"] = evidence_question_map[eid]
                    if eid and evidence_caption_map.get(eid):
                        ev["caption"] = evidence_caption_map[eid]

            result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
            elapsed = time.time() - start_time
            return True, result, elapsed

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            elapsed = time.time() - start_time
            return False, {"error": str(e)}, elapsed

    elapsed = time.time() - start_time
    return False, {"error": "max retries exceeded"}, elapsed


def main():
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Session Dialogue Planning")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--sessions_file", type=Path, default=None)
    parser.add_argument("--qa_file", type=Path, default=None)
    parser.add_argument("--registry_file", type=Path, default=None)
    parser.add_argument("--template_file", type=Path, default=base_dir / "prompt" / "01_session_dialogue_planning_prompt.txt")
    parser.add_argument("--output_root", type=Path, default=base_dir / "output" / "01_session_dialogue_planning")
    args = parser.parse_args()

    sessions_file = args.sessions_file or (
        base_dir.parent / "framework" / "output" / "03_session_gen_stage1" / args.run_id / "generated_sessions_stage1.json"
    )
    qa_file = args.qa_file or (
        base_dir.parent / "rewrite" / "output" / "04_final_qa" / args.run_id / "final_qa_aggregated.jsonl"
    )
    registry_file = args.registry_file or (
        base_dir.parent / "filter" / "qa" / args.run_id / "mm_registry.json"
    )
    template_file = args.template_file

    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = output_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    final_output = output_dir / "all_session_plans.json"

    print("=" * 60)
    print("Session Dialogue Planning")
    print("=" * 60)

    for name, path in [("sessions_file", sessions_file), ("qa_file", qa_file),
                       ("template_file", Path(template_file)), ("registry_file", registry_file)]:
        if not path.exists():
            raise SystemExit(f"{name} not found: {path}")

    sessions_data = load_json(str(sessions_file))
    qa_data = load_jsonl(str(qa_file))
    mm_registry = load_json(str(registry_file))
    prompt_template = Path(template_file).read_text(encoding='utf-8')
    sessions = sessions_data['sessions']

    session_evidence = group_evidence_by_session(qa_data, mm_registry)

    total_evidence = sum(len(v) for v in session_evidence.values())
    for sid, evs in sorted(session_evidence.items()):
        print(f"  {sid}: {len(evs)} evidence")

    previous_results = []
    failed = []
    processing_times = []

    for i, session in enumerate(sessions):
        sid = session['session_id']
        evidence = session_evidence.get(sid, [])

        context_hints = sanitize_mm_captions(evidence, mm_registry)
        if context_hints:
            print(f"  [sanitize] {sid}: {len(context_hints)} mm hints generated")

        print(f"[{i+1}/{len(sessions)}] {sid} ({len(evidence)} evidence)...", end=" ")

        success, result, elapsed = process_session(
            current_session=session, previous_results=previous_results,
            evidence=evidence, context_hints=context_hints,
            prompt_template=prompt_template,
            output_dir=sessions_dir, window_size=5
        )

        if success:
            n_phases = len(result.get("dialogue_flow", {}).get("phases", []))
            print(f"✓ ({n_phases} phases, {elapsed:.2f}s)")
            previous_results.append(result)
            processing_times.append(elapsed)
        else:
            print(f"✗ {result.get('error', '?')}")
            failed.append(sid)
            break

    aggregated = {
        "total_sessions": len(sessions),
        "completed": len(previous_results),
        "session_plans": previous_results,
    }
    final_output.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding='utf-8')

    total_time = sum(processing_times)

    print(f"\n{'=' * 60}")
    if failed:
        pass

    if failed or len(previous_results) != len(sessions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
