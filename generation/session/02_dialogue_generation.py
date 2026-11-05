#!/usr/bin/env python3

import os
import json
import sys
import time
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent / "preprocess"))
from call_gemini import call_gemini


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def _is_mm_type(t: str) -> bool:
    return t.lower() in {"image", "audio", "video"}


def format_context(previous_dialogues: List[Dict], window_size: int = 3) -> str:
    if not previous_dialogues:
        return "No previous sessions."
    lines = []
    boundary = max(0, len(previous_dialogues) - window_size)
    for d in previous_dialogues[:boundary]:
        lines.append(f"- {d['session_id']} ({d.get('date', '?')}): {d.get('summary', '')}")
    if boundary > 0:
        lines.append("")
    for d in previous_dialogues[boundary:]:
        lines.append(f"【{d['session_id']}】")
        lines.append(f"Date: {d.get('date', '?')}")
        lines.append(f"Summary: {d.get('summary', '')}")
        lines.append("Dialogue:")
        for r in d.get('dialogues', []):
            lines.append(f"  [User] {r.get('user', '')}")
            lines.append(f"  [Assistant] {r.get('assistant', '')}")
            if r.get('mm'):
                for m in r['mm']:
                    lines.append(f"    ({m['type']})")
        lines.append("")
    return "\n".join(lines)


def format_session_info(session: Dict) -> str:
    parts = [f"Session ID: {session['session_id']}", f"Date: {session['date']}"]
    for key in ['time_of_day', 'trigger']:
        if session.get(key):
            parts.append(f"{key.replace('_', ' ').title()}: {session[key]}")
    return "\n".join(parts)


def format_dialogue_plan(plan: Dict) -> str:
    lines = []
    flow = plan.get("dialogue_flow", {})
    lines.append(f"Session Vibe: {flow.get('session_vibe', 'N/A')}")
    lines.append(f"\nPhases ({len(flow.get('phases', []))}):")

    for phase in flow.get("phases", []):
        pid = phase.get("phase_id", "?")
        lines.append(f"\n--- Phase {pid} ---")
        lines.append(f"Description: {phase.get('description', '')}")
        evidence = phase.get("evidence", [])
        if evidence:
            lines.append("Evidence to embed:")
            for e in evidence:
                mm_tag = f" (mm_id={e['mm_id']})" if e.get('mm_id') else ""
                question_str = f"\n    question: \"{e['question']}\"" if e.get("question") else ""

                if _is_mm_type(e.get("type", "")):
                    caption = e.get("caption", "")
                    caption_str = f"\n    caption: \"{caption}\"" if caption else ""
                    line = f"  • {e['evidence_id']} [{e['type']}]{mm_tag}{caption_str}{question_str}"
                else:
                    line = f"  • {e['evidence_id']} [{e['type']}]{mm_tag}: {e['content']}{question_str}"
                lines.append(line)
                if e.get('integration_hint'):
                    lines.append(f"    Hint: {e['integration_hint']}")

    lines.append(f"\nSession Plan Summary: {plan.get('summary', '')}")
    return "\n".join(lines)


def format_persona(persona: Dict) -> str:
    parts = []
    if persona.get('persona_summary'):
        parts.append(f"Name & Role: {persona['persona_summary']}")
    if persona.get('conversation_style'):
        parts.append(f"Conversation Style: {persona['conversation_style']}")
    return "\n".join(parts) if parts else "No persona information."


def get_plan_evidence_ids(plan: Dict) -> set:
    ids = set()
    for phase in plan.get("dialogue_flow", {}).get("phases", []):
        for e in phase.get("evidence", []):
            ids.add(e["evidence_id"])
    return ids


def build_evidence_mm_map(plan: Dict) -> Dict[str, str]:
    mm_map = {}
    for phase in plan.get("dialogue_flow", {}).get("phases", []):
        for e in phase.get("evidence", []):
            eid = e.get("evidence_id")
            mm_id = e.get("mm_id")
            if eid and mm_id:
                mm_map[eid] = mm_id
    return mm_map


def build_expected_mm_types(plan: Dict) -> Dict[str, str]:
    expected: Dict[str, str] = {}
    for phase in plan.get("dialogue_flow", {}).get("phases", []):
        for e in phase.get("evidence", []):
            eid = e.get("evidence_id")
            etype = str(e.get("type", "")).lower()
            if eid and etype in {"image", "audio", "video"}:
                expected[eid] = etype
    return expected


def extract_mm_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [t.lower() for t in re.findall(r"<\s*(video|image|audio)\s*>", text, flags=re.IGNORECASE)]


def validate_dialogue(result: Dict, expected_evidence: set, expected_mm_types: Dict[str, str]) -> List[str]:
    errors = []
    coverage = result.get("evidence_coverage", {})
    if not isinstance(coverage, dict):
        errors.append("evidence_coverage ...")
        coverage = {}
    covered = set(coverage.keys())
    missing = expected_evidence - covered
    extra = covered - expected_evidence
    if missing:
        errors.append(f"...")
    if extra:
        errors.append(f"evidence_coverage ...")

    dialogues = result.get("dialogues", [])
    if not isinstance(dialogues, list) or not dialogues:
        errors.append("...")
        return errors

    mm_covered: set[str] = set()
    for ridx, r in enumerate(dialogues, start=1):
        if not isinstance(r, dict):
            errors.append(f"dialogues[{ridx}] ...")
            continue

        if not str(r.get("user", "")).strip():
            errors.append(f"dialogues[{ridx}] ...")
        if not str(r.get("assistant", "")).strip():
            errors.append(f"dialogues[{ridx}] ...")

        user_text = r.get("user", "")
        tags = extract_mm_tags(user_text)

        mm_items = r.get("mm")
        if mm_items is None:
            mm_items = []
        if not isinstance(mm_items, list):
            errors.append(f"dialogues[{ridx}].mm ...")
            continue

        if tags and not mm_items:
            errors.append(f"dialogues[{ridx}] user ...")
        if mm_items and not tags:
            errors.append(f"dialogues[{ridx}] ...")
        if tags and mm_items and len(tags) != len(mm_items):
            errors.append(
                f"dialogues[{ridx}] ..."
            )

        for midx, mm_item in enumerate(mm_items, start=1):
            if not isinstance(mm_item, dict):
                errors.append(f"dialogues[{ridx}].mm[{midx}] ...")
                continue

            mm_type = str(mm_item.get("type", "")).lower()
            if mm_type not in {"video", "image", "audio"}:
                errors.append(f"dialogues[{ridx}].mm[{midx}] type invalid")
            elif midx <= len(tags) and tags[midx - 1] != mm_type:
                errors.append(
                    f"dialogues[{ridx}] mm type mismatch"
                )

            eid = mm_item.get("evidence_id")
            if not isinstance(eid, str) or not eid:
                errors.append(f"dialogues[{ridx}].mm[{midx}] ...")
                continue
            mm_covered.add(eid)

            expected_type = expected_mm_types.get(eid)
            if expected_type is None:
                errors.append(f"dialogues[{ridx}].mm[{midx}] evidence_id={eid} ...")
            elif mm_type and mm_type != expected_type:
                errors.append(
                    f"dialogues[{ridx}].mm[{midx}] type={mm_type} ..."
                )

    missing_mm = set(expected_mm_types.keys()) - mm_covered
    if missing_mm:
        errors.append(f"...")
    return errors


def process_session(
    current_session: Dict,
    session_plan: Dict,
    persona: Dict,
    previous_dialogues: List[Dict],
    prompt_template: str,
    output_dir: Path,
    mm_registry: Dict,
    window_size: int = 3,
    max_retries: int = 3
) -> Tuple[bool, Dict]:

    session_id = current_session['session_id']
    result_file = output_dir / f"{session_id}.json"
    raw_file = output_dir / f"{session_id}_raw.txt"

    if result_file.exists():
        return True, load_json(str(result_file))

    expected_evidence = get_plan_evidence_ids(session_plan)
    evidence_mm_map = build_evidence_mm_map(session_plan)
    expected_mm_types = build_expected_mm_types(session_plan)

    prompt = prompt_template.replace("{context}", format_context(previous_dialogues, window_size))
    prompt = prompt.replace("{persona}", format_persona(persona))
    prompt = prompt.replace("{session_info}", format_session_info(current_session))
    prompt = prompt.replace("{dialogue_plan}", format_dialogue_plan(session_plan))
    prompt = prompt.replace("{session_id}", session_id)

    if not expected_mm_types:
        prompt += (
            "\n\n**HARD CONSTRAINT**: This session has NO multimodal evidence. "
            "You MUST NOT generate any <image>, <audio>, or <video> tags or mm items in the dialogue."
        )

    for attempt in range(1, max_retries + 1):
        try:
            success, response = call_gemini(
                prompt=prompt, model=os.environ.get("LLM_MODEL", "your-model-name"),
                temperature=0.8, max_tokens=32768, max_retries=5
            )
            if not success:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return False, {"error": f"API..."}

            raw_file.write_text(response, encoding='utf-8')
            result = extract_json_from_response(response)

            if expected_evidence:
                errors = validate_dialogue(result, expected_evidence, expected_mm_types)
                if errors:
                    print(f"    ⚠ {errors}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    return False, {"error": f"validation failed: {errors}"}

            for dialogue_round in result.get('dialogues', []):
                for mm_item in dialogue_round.get('mm', []):
                    eid = mm_item.get('evidence_id')
                    mm_id = evidence_mm_map.get(eid)
                    if eid and mm_id:
                        mm_item["mm_id"] = mm_id
                        registry_entry = mm_registry.get(mm_id, {})
                        if registry_entry.get("caption"):
                            mm_item["description"] = registry_entry["caption"]

            for eid, coverage_info in result.get('evidence_coverage', {}).items():
                if evidence_mm_map.get(eid):
                    coverage_info["mm_id"] = evidence_mm_map[eid]

            result['session_id'] = session_id
            result['date'] = current_session.get('date', '')

            result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
            return True, result

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return False, {"error": str(e)}

    return False, {"error": "max retries exceeded"}


def main():
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Dialogue Generation")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--sessions_file", type=Path, default=None)
    parser.add_argument("--plans_file", type=Path, default=None)
    parser.add_argument("--persona_file", type=Path, default=None)
    parser.add_argument("--registry_file", type=Path, default=None)
    parser.add_argument("--template_file", type=Path, default=base_dir / "prompt" / "02_dialogue_generation_prompt_template.txt")
    parser.add_argument("--output_root", type=Path, default=base_dir / "output" / "02_dialogue_generation")
    args = parser.parse_args()

    sessions_file = args.sessions_file or (
        base_dir.parent / "framework" / "output" / "03_session_gen_stage1" / args.run_id / "generated_sessions_stage1.json"
    )
    plans_file = args.plans_file or (base_dir / "output" / "01_session_dialogue_planning" / args.run_id / "all_session_plans.json")
    persona_file = args.persona_file or (
        base_dir.parent / "framework" / "output" / "01_persona_gen_caption" / args.run_id / "generated_persona_caption.json"
    )
    registry_file = args.registry_file or (
        base_dir.parent / "filter" / "qa" / args.run_id / "mm_registry.json"
    )
    template_file = args.template_file

    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = output_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    final_output = output_dir / "all_dialogues.json"

    print("=" * 60)
    print("Dialogue Generation")
    print("=" * 60)

    for name, path in [("sessions_file", sessions_file), ("plans_file", plans_file),
                       ("persona_file", persona_file), ("template_file", Path(template_file)),
                       ("registry_file", registry_file)]:
        if not path.exists():
            raise SystemExit(f"{name} not found: {path}")

    sessions_data = load_json(str(sessions_file))
    plans_data = load_json(str(plans_file))
    persona = load_json(str(persona_file))
    mm_registry = load_json(str(registry_file))
    prompt_template = Path(template_file).read_text(encoding='utf-8')

    sessions = sessions_data['sessions']
    plan_map = {p['session_id']: p for p in plans_data.get('session_plans', [])}


    previous_dialogues = []
    failed = []

    for i, session in enumerate(sessions):
        sid = session['session_id']
        plan = plan_map.get(sid)
        if not plan:
            print(f"\n[{i+1}/{len(sessions)}] {sid} — ...")
            continue

        n_evidence = len(get_plan_evidence_ids(plan))
        n_phases = len(plan.get("dialogue_flow", {}).get("phases", []))
        print(f"\n[{i+1}/{len(sessions)}] {sid} ({n_phases} phases, {n_evidence} evidence)...", end=" ")

        success, result = process_session(
            current_session=session, session_plan=plan, persona=persona,
            previous_dialogues=previous_dialogues, prompt_template=prompt_template,
            output_dir=sessions_dir, mm_registry=mm_registry, window_size=3
        )

        if success:
            n_rounds = len(result.get('dialogues', []))
            print(f"✓ ({n_rounds} rounds)")
            previous_dialogues.append(result)
        else:
            print(f"✗ {result.get('error', '?')}")
            failed.append(sid)
            break

    aggregated = {
        "multi_session_dialogues": [
            {"session_id": d["session_id"], "date": d.get("date", ""), "dialogues": d.get("dialogues", [])}
            for d in previous_dialogues
        ]
    }
    final_output.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding='utf-8')

    coverage_map = {}
    for d in previous_dialogues:
        for eid, info in d.get("evidence_coverage", {}).items():
            coverage_map[eid] = {**info, "session_id": d["session_id"]}
    coverage_file = output_dir / "evidence_coverage.json"
    coverage_file.write_text(json.dumps(coverage_map, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"\n{'=' * 60}")
    if failed:
        pass

    if failed or len(previous_dialogues) != len(sessions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
