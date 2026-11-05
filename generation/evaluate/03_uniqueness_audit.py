#!/usr/bin/env python3
"""
03_uniqueness_audit.py — QA answer uniqueness audit
Given full dialogue context, judge whether each question has exactly one reasonable answer.
"""

import os
import json
import argparse
import time
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocess"))
from call_gemini import call_gemini

PROMPT_TEMPLATE = """\
You are a benchmark quality auditor. Your task is to determine whether a question has exactly ONE reasonable answer given the dialogue context.

## Dialogue Context
{dialogue_context}

## Question
{question}

## Reference Answer
{answer}

## Instructions
Based on the dialogue context above, judge whether this question has a unique reasonable answer, or whether the context supports multiple genuinely different answers.

- Answers that differ only in phrasing/format but are semantically equivalent count as the SAME answer (e.g., "3 days" vs "three days", "NYC" vs "New York City").
- Only flag as ambiguous if the context genuinely supports a DIFFERENT answer with different semantic meaning.

## Output Format (JSON only, no markdown)
{{
  "question_id": "{question_id}",
  "is_unique": true or false,
  "reason": "brief explanation",
  "alternative_answer": "if ambiguous, provide the alternative answer; otherwise null"
}}
"""


def build_dialogue_text(sessions: list) -> str:
    """Flatten all sessions into plain text, using captions for multimodal content."""
    parts = []
    for session in sessions:
        sid = session.get("session_id", "?")
        date = session.get("date", "")
        header = f"[Session {sid}"
        if date:
            header += f" — {date}"
        header += "]"
        parts.append(header)

        for d in session.get("dialogues", []):
            user_text = d.get("user", "")
            asst_text = d.get("assistant", "")
            # inject captions for multimodal items
            mm = d.get("mm") or []
            if mm:
                captions = []
                for item in mm:
                    mod = item.get("type", "media")
                    cap = item.get("caption") or item.get("description") or ""
                    mm_id = ""
                    for k in ("mm_id", "media_id", "evidence_id"):
                        v = item.get(k)
                        if isinstance(v, str) and v.strip():
                            mm_id = v.strip()
                            break
                    line = f"[{mod}"
                    if mm_id:
                        line += f" id={mm_id}"
                    line += f"] {cap}" if cap else "]"
                    captions.append(line)
                if captions:
                    user_text = "\n".join(captions) + "\n" + user_text

            if user_text.strip():
                parts.append(f"User: {user_text.strip()}")
            if asst_text.strip():
                parts.append(f"Assistant: {asst_text.strip()}")
        parts.append("")
    return "\n".join(parts)


def audit_one(qa: dict, dialogue_text: str, model: str, max_retries: int = 3) -> dict:
    qid = qa.get("question_id", "")
    question = qa.get("question", "")
    answer = qa.get("full_answer") or qa.get("answer") or ""

    prompt = PROMPT_TEMPLATE.format(
        dialogue_context=dialogue_text,
        question=question,
        answer=answer,
        question_id=qid,
    )
    for attempt in range(max_retries):
        success, resp = call_gemini(prompt=prompt, model=model, temperature=0.1, max_tokens=4096, max_retries=3)
        if not success:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"question_id": qid, "error": resp, "is_unique": None}
        try:
            result = json.loads(resp)
            result["question_id"] = qid
            return result
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"question_id": qid, "error": f"JSON parse failed: {resp[:200]}", "is_unique": None}
    return {"question_id": qid, "error": "max retries", "is_unique": None}


def main():
    parser = argparse.ArgumentParser(description="QA answer uniqueness audit")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "your-model-name"))
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=60)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    benchmark_file = args.benchmark or (base / "session" / "output" / "05_benchmark_assembly" / args.run_id / "benchmark.json")
    output_dir = args.output_dir or (Path(__file__).resolve().parent / "output" / "03_uniqueness_audit" / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "uniqueness_audit_results.jsonl"

    dataset = json.loads(benchmark_file.read_text("utf-8"))
    qas = dataset.get("QAs", [])
    sessions = dataset.get("multi_session_dialogues", [])
    dialogue_text = build_dialogue_text(sessions)

    print(f"Uniqueness Audit: {len(qas)} QAs, {len(sessions)} sessions")
    print(f"Dialogue context length: {len(dialogue_text)} chars")
    print(f"Model: {args.model}, Workers: {args.workers}")

    # resume
    done_ids: set[str] = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line).get("question_id", ""))
                    except Exception:
                        pass
        print(f"  Completed: {len(done_ids)}, Remaining: {len(qas) - len(done_ids)}")

    todo = [qa for qa in qas if qa.get("question_id", "") not in done_ids]

    if not todo:
        print("  All done, skipping.")
    else:
        write_lock = threading.Lock()
        with open(output_file, "a", encoding="utf-8") as out:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(audit_one, qa, dialogue_text, args.model, args.max_retries): qa for qa in todo}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Auditing"):
                    result = future.result()
                    with write_lock:
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out.flush()

    # summarize
    all_results = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    total = len(all_results)
    unique = sum(1 for r in all_results if r.get("is_unique") is True)
    ambiguous = sum(1 for r in all_results if r.get("is_unique") is False)
    error = sum(1 for r in all_results if r.get("is_unique") is None)
    ambiguous_ids = [r["question_id"] for r in all_results if r.get("is_unique") is False]

    summary = {
        "total": total,
        "unique": unique,
        "ambiguous": ambiguous,
        "error": error,
        "unique_rate": round(unique / total * 100, 2) if total else 0,
        "ambiguous_ids": ambiguous_ids,
    }
    summary_file = output_dir / "uniqueness_audit_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")

    print(f"\nTotal: {total}  Unique: {unique}  Ambiguous: {ambiguous}  Error: {error}")
    print(f"Unique rate: {summary['unique_rate']}%")
    if ambiguous_ids:
        print(f"Ambiguous IDs: {ambiguous_ids}")
    print(f"Summary → {summary_file}")


if __name__ == "__main__":
    main()
