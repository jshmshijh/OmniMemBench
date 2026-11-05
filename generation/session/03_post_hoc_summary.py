#!/usr/bin/env python3
"""
Post-hoc Session Summary:
"""

import os
import json
import sys
import re
import argparse
from pathlib import Path

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


def format_dialogue(session: dict) -> str:
    lines = []
    for r in session.get("dialogues", []):
        rnd = r.get("round", "?")
        lines.append(f"[Round {rnd}] User: {r.get('user', '')}")
        lines.append(f"[Round {rnd}] Assistant: {r.get('assistant', '')}")
        for mm in r.get("mm", []):
            lines.append(f"  ({mm.get('type', '?')}: {mm.get('description', '')})")
    return "\n".join(lines)


def summarize_session(
    session: dict,
    prompt_template: str,
    max_retries: int = 5,
) -> dict:
    sid = session.get("session_id", "?")
    date = session.get("date", "")
    dialogue_text = format_dialogue(session)

    prompt = prompt_template.replace("{session_id}", sid)
    prompt = prompt.replace("{date}", date)
    prompt = prompt.replace("{dialogue_text}", dialogue_text)

    for attempt in range(1, max_retries + 1):
        success, response = call_gemini(
            prompt=prompt,
            model=os.environ.get("LLM_MODEL", "your-model-name"),
            temperature=0.3,
            max_tokens=32768,
            max_retries=5,
        )
        if not success:
            print(f"    attempt {attempt} API failed: {response}")
            continue
        try:
            result = extract_json_from_response(response)
            if result.get("session_summary"):
                return result
            print(f"    attempt {attempt}: empty summary, retrying")
        except ValueError as e:
            print(f"    attempt {attempt} parse error: {e}")

    raise RuntimeError(f"Failed to summarize {sid}")


def main():
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Post-hoc Session Summary")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--dialogues_file", type=Path, default=None)
    parser.add_argument("--prompt_template", type=Path,
                        default=base_dir / "prompt" / "03_post_hoc_summary_prompt.txt")
    parser.add_argument("--output_root", type=Path,
                        default=base_dir / "output" / "03_post_hoc_summary")
    parser.add_argument("--max_retries", type=int, default=5)
    args = parser.parse_args()

    dialogues_file = args.dialogues_file or (
        base_dir / "output" / "02_dialogue_generation" / args.run_id / "all_dialogues.json"
    )

    if not dialogues_file.exists():
        raise SystemExit(f"dialogues not found: {dialogues_file}")
    if not args.prompt_template.exists():
        raise SystemExit(f"prompt not found: {args.prompt_template}")

    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output = output_dir / "post_hoc_summaries.json"

    dialogues_data = load_json(str(dialogues_file))
    prompt_template = args.prompt_template.read_text(encoding="utf-8")
    sessions = dialogues_data.get("multi_session_dialogues", [])

    print("=" * 60)
    print("Post-hoc Session Summary")
    print(f"Sessions: {len(sessions)}")
    print("=" * 60)

    summaries = []
    for i, session in enumerate(sessions):
        sid = session.get("session_id", "?")
        print(f"  [{i+1}/{len(sessions)}] {sid}...", end=" ")
        try:
            result = summarize_session(session, prompt_template, args.max_retries)
            summaries.append({
                "session_id": result.get("session_id", sid),
                "date": result.get("date", session.get("date", "")),
                "session_summary": result["session_summary"],
                "key_entities": result.get("key_entities", []),
                "has_multimodal": result.get("has_multimodal", False),
            })
            print("✓")
        except RuntimeError as e:
            print(f"✗ {e}")
            continue

    output = {"sessions": summaries}
    final_output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone: {len(summaries)}/{len(sessions)} sessions summarized")
    print(f"Output: {final_output}")

    if len(summaries) < len(sessions):
        print("⚠ Some sessions failed, ES generation may have incomplete coverage")


if __name__ == "__main__":
    main()
