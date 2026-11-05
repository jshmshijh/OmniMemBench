#!/usr/bin/env python3

import os
import json
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List

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
    raise ValueError("Cannot extract JSON from response")


def format_event_graph(events_data: dict) -> str:
    lines = []
    # story lines
    story_lines = events_data.get("story_lines", [])
    if story_lines:
        lines.append("Story Lines:")
        for sl in story_lines:
            lines.append(f"  - {sl}")
        lines.append("")

    # events
    graph = events_data.get("graph", [])
    lines.append(f"Events ({len(graph)}):")
    for ev in graph:
        eid = ev.get("id", "?")
        desc = ev.get("sub-event", "")
        date = ev.get("date", "")
        caused_by = ev.get("caused_by", [])
        attached = ev.get("attached_question", [])
        lines.append(f"  {eid} [{date}]: {desc}")
        if caused_by:
            lines.append(f"    caused_by: {caused_by}")
        if attached:
            lines.append(f"    attached_questions: {attached}")
    return "\n".join(lines)


def format_session_summaries(sessions_data: dict, plans_data: dict) -> str:
    lines = []
    sessions = sessions_data.get("sessions", [])
    plan_map = {p["session_id"]: p for p in plans_data.get("session_plans", [])}

    for s in sessions:
        sid = s.get("session_id", "?")
        date = s.get("date", "")
        summary = s.get("session_summary", "")
        lines.append(f"[{sid}] {date}: {summary}")

        # evidence placement from plan
        plan = plan_map.get(sid, {})
        for phase in plan.get("dialogue_flow", {}).get("phases", []):
            for ev in phase.get("evidence", []):
                ev_type = ev.get("type", "text")
                if ev_type != "text":
                    lines.append(f"  (multimodal: {ev_type} — {ev.get('content', '')[:80]})")
        lines.append("")

    return "\n".join(lines)


def extract_session_dialogues(dialogues_data: dict, session_ids: List[str]) -> str:
    lines = []
    for s in dialogues_data.get("multi_session_dialogues", []):
        sid = s.get("session_id", "")
        if sid not in session_ids:
            continue
        lines.append(f"=== {sid} (date: {s.get('date', '')}) ===")
        for r in s.get("dialogues", []):
            rnd = r.get("round", "?")
            lines.append(f"  [Round {rnd}] User: {r.get('user', '')}")
            lines.append(f"  [Round {rnd}] Assistant: {r.get('assistant', '')}")
            for mm in r.get("mm", []):
                lines.append(f"    ({mm.get('type', '?')}: {mm.get('description', '')})")
        lines.append("")
    return "\n".join(lines)


def run_step1(
    events_data: dict,
    sessions_data: dict,
    plans_data: dict,
    prompt_template: str,
    max_retries: int = 5,
) -> dict:
    prompt = prompt_template.replace("{event_graph}", format_event_graph(events_data))
    prompt = prompt.replace("{session_summaries}", format_session_summaries(sessions_data, plans_data))

    for attempt in range(1, max_retries + 1):
        success, response = call_gemini(
            prompt=prompt,
            model=os.environ.get("LLM_MODEL", "your-model-name"),
            temperature=0.7,
            max_tokens=32768,
            max_retries=5,
        )
        if not success:
            print(f"  Step1 attempt {attempt} API failed: {response}")
            continue
        try:
            result = extract_json_from_response(response)
            questions = result.get("questions", [])
            if len(questions) >= 3:
                return result
            print(f"  Step1 attempt {attempt}: only {len(questions)} questions, retrying")
        except ValueError as e:
            print(f"  Step1 attempt {attempt} parse error: {e}")

    raise RuntimeError("Step1 failed after max retries")


def run_step2(
    question_skeleton: dict,
    dialogues_data: dict,
    prompt_template: str,
    max_retries: int = 5,
) -> dict:
    session_ids = question_skeleton.get("session_ids", [])
    session_text = extract_session_dialogues(dialogues_data, session_ids)

    prompt = prompt_template.replace("{question_skeleton}", json.dumps(question_skeleton, indent=2, ensure_ascii=False))
    prompt = prompt.replace("{session_dialogues}", session_text)

    for attempt in range(1, max_retries + 1):
        success, response = call_gemini(
            prompt=prompt,
            model=os.environ.get("LLM_MODEL", "your-model-name"),
            temperature=0.5,
            max_tokens=32768,
            max_retries=5,
        )
        if not success:
            print(f"    Step2 attempt {attempt} API failed: {response}")
            continue
        try:
            result = extract_json_from_response(response)
            nuggets = result.get("nuggets", [])
            if len(nuggets) >= 3:
                return result
            print(f"    Step2 attempt {attempt}: only {len(nuggets)} nuggets, retrying")
        except ValueError as e:
            print(f"    Step2 attempt {attempt} parse error: {e}")

    raise RuntimeError(f"Step2 failed for {question_skeleton.get('question_id', '?')}")


def main():
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="ES Question Generation")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--events_file", type=Path, default=None)
    parser.add_argument("--sessions_file", type=Path, default=None)
    parser.add_argument("--plans_file", type=Path, default=None)
    parser.add_argument("--dialogues_file", type=Path, default=None)
    parser.add_argument("--step1_template", type=Path, default=base_dir / "prompt" / "04_es_step1_prompt.txt")
    parser.add_argument("--step2_template", type=Path, default=base_dir / "prompt" / "04_es_step2_prompt.txt")
    parser.add_argument("--output_root", type=Path, default=base_dir / "output" / "04_es_generation")
    parser.add_argument("--max_retries", type=int, default=5)
    args = parser.parse_args()

    events_file = args.events_file or (
        base_dir.parent / "framework" / "output" / "02_event_gen_caption" / args.run_id / "generated_events.json"
    )
    sessions_file = args.sessions_file or (
        base_dir / "output" / "03_post_hoc_summary" / args.run_id / "post_hoc_summaries.json"
    )
    plans_file = args.plans_file or (
        base_dir / "output" / "01_session_dialogue_planning" / args.run_id / "all_session_plans.json"
    )
    dialogues_file = args.dialogues_file or (
        base_dir / "output" / "02_dialogue_generation" / args.run_id / "all_dialogues.json"
    )

    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output = output_dir / "es_questions.jsonl"

    print("=" * 60)
    print("ES Question Generation")
    print("=" * 60)

    for name, path in [("events", events_file), ("sessions", sessions_file),
                       ("plans", plans_file), ("dialogues", dialogues_file),
                       ("step1_template", args.step1_template), ("step2_template", args.step2_template)]:
        if not path.exists():
            raise SystemExit(f"{name} not found: {path}")

    events_data = load_json(str(events_file))
    sessions_data = load_json(str(sessions_file))
    plans_data = load_json(str(plans_file))
    dialogues_data = load_json(str(dialogues_file))
    step1_template = args.step1_template.read_text(encoding="utf-8")
    step2_template = args.step2_template.read_text(encoding="utf-8")

    n_sessions = len(dialogues_data.get("multi_session_dialogues", []))
    n_events = len(events_data.get("graph", []))
    print(f"Loaded: {n_events} events, {n_sessions} sessions")

    # Step 1: generate skeletons
    print("\n--- Step 1: Skeleton Generation ---")
    step1_result = run_step1(events_data, sessions_data, plans_data, step1_template, args.max_retries)
    skeletons = step1_result.get("questions", [])
    print(f"Generated {len(skeletons)} SUM question skeletons")

    # save intermediate
    step1_file = output_dir / "step1_skeletons.json"
    step1_file.write_text(json.dumps(step1_result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Step 2: generate nuggets per question
    print("\n--- Step 2: Nugget Generation ---")
    es_questions = []
    for i, skeleton in enumerate(skeletons):
        qid = skeleton.get("question_id", f"SUM_{i+1}")
        print(f"  [{i+1}/{len(skeletons)}] {qid} ({skeleton.get('subtype', '?')})...", end=" ")

        try:
            step2_result = run_step2(skeleton, dialogues_data, step2_template, args.max_retries)
            n_nuggets = len(step2_result.get("nuggets", []))
            print(f"✓ ({n_nuggets} nuggets)")

            # assemble final question
            es_q = {
                "question_id": qid,
                "type": "SUM",
                "subtype": skeleton.get("subtype", ""),
                "question": step2_result.get("question", skeleton.get("question", "")),
                "ideal_summary": step2_result.get("ideal_summary", skeleton.get("ideal_summary", "")),
                "nuggets": step2_result.get("nuggets", []),
                "session_ids": skeleton.get("session_ids", []),
                "event_ids": skeleton.get("event_ids", []),
            }
            es_questions.append(es_q)

        except RuntimeError as e:
            print(f"✗ {e}")
            continue

    # save as jsonl
    with open(final_output, "w", encoding="utf-8") as f:
        for q in es_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Done: {len(es_questions)}/{len(skeletons)} ES questions generated")
    print(f"Output: {final_output}")

    if len(es_questions) == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
