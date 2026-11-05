#!/usr/bin/env python3

import os
import json
import argparse
import time
import sys
import threading
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocess"))
from call_gemini import call_gemini, call_gemini_multimodal

# ── Mime helpers ─────────────────────────────────────────────────────────────

_SUFFIX_MAP = {
    ".jpg": ("image", "image/jpeg"), ".jpeg": ("image", "image/jpeg"),
    ".png": ("image", "image/png"), ".webp": ("image", "image/webp"),
    ".mp3": ("audio", "audio/mp3"), ".wav": ("audio", "audio/wav"),
    ".m4a": ("audio", "audio/mp4"), ".ogg": ("audio", "audio/ogg"),
    ".mp4": ("video", "video/mp4"), ".webm": ("video", "video/webm"),
}
_MODALITY_MIME = {"image": "image/jpeg", "audio": "audio/wav", "video": "video/mp4"}


def _infer_media_info(url: str, modality: str) -> Dict[str, str]:
    suffix = Path(urlparse(url).path).suffix.lower()
    _, mime = _SUFFIX_MAP.get(suffix, (modality, _MODALITY_MIME.get(modality, "application/octet-stream")))
    return {"url": url, "mime_type": mime, "modality_type": modality}


def _is_http_url(v: str) -> bool:
    return isinstance(v, str) and urlparse(v).scheme.lower() in {"http", "https"}


# ── Session / evidence helpers ───────────────────────────────────────────────

def build_session_map(dialogues_data: dict) -> Dict[str, dict]:
    return {s["session_id"]: s for s in dialogues_data.get("multi_session_dialogues", []) if s.get("session_id")}


def get_qa_session_ids(qa: dict) -> List[str]:
    sids = set()
    for e in qa.get("evidence_chain", []):
        if e.get("session_id"):
            sids.add(e["session_id"])
    return sorted(sids)


def format_evidence_summary(chain: list) -> str:
    if not chain:
        return "(empty)"
    parts = []
    for e in chain:
        meta = []
        if e.get("session_id"):
            meta.append(f"session={e['session_id']}")
        if e.get("timestamp"):
            meta.append(f"time={e['timestamp']}")
        meta.append(f"type={e.get('type', 'text')}")
        if e.get("mm_id"):
            meta.append(f"mm_id={e['mm_id']}")
        eid = e.get("evidence_id", "")
        header = f"[{eid}] ({', '.join(meta)})" if eid else f"({', '.join(meta)})"
        content = e.get("evidence", "") or e.get("content", "")
        parts.append(f"{header} {content}" if content else header)
    return "\n".join(parts)


# ── Build flattened dialogue text + collect media URLs ───────────────────────

def flatten_sessions(session_ids: List[str], session_map: Dict[str, dict]):
    lines = []
    media_list = []
    media_counter = 0

    for sid in session_ids:
        session = session_map.get(sid)
        if not session:
            continue
        date = session.get("date", "?")
        lines.append(f"\n=== Session {sid} — {date} ===")

        for dialog in session.get("dialogues", []):
            user_text = dialog.get("user", "") or ""
            asst_text = dialog.get("assistant", "") or ""
            mm_items = dialog.get("mm") or []

            lines.append(f"User: {user_text}")
            for item in mm_items:
                modality = item.get("type", "").lower()
                url = (item.get("url") or "").strip()
                if _is_http_url(url) and modality in ("image", "audio", "video"):
                    media_counter += 1
                    tag = f"[{modality}_{media_counter}]"
                    lines.append(f"  {tag} (attached {modality})")
                    media_list.append(_infer_media_info(url, modality))

            if asst_text.strip():
                lines.append(f"Assistant: {asst_text}")

    return "\n".join(lines), media_list


# ── Audit prompt ─────────────────────────────────────────────────────────────

AUDIT_PROMPT = """\
You are a strict benchmark quality auditor. A QA item has been embedded into multi-session dialogues. The dialogue text is shown below, and any referenced multimodal files (images/audio/video) are attached to this message. Your job is to verify the embedding quality.

## Dialogue Context
{dialogue_text}

## QA Item
- **Question ID**: {question_id}
- **Task Type**: {task_type}
- **Question**: {question}
- **Expected Answer**: {answer}
- **Evidence Chain**:
{evidence_summary}

## Evaluation Criteria
Rate each dimension as "pass" or "fail" with a brief reason:

1. **evidence_presence**: Is each piece of evidence actually present (explicitly or implicitly) in the dialogue? For multimodal evidence, verify the attached media content matches. List any missing evidence IDs.
2. **no_answer_leakage**: Could someone answer the question correctly from the dialogue WITHOUT needing to reason or connect information across sessions? This includes:
   - The answer being directly stated in a single obvious place.
   - Descriptive language in the dialogue (e.g., user or assistant describing, reacting to, or characterizing shared media) that reveals information needed to answer the question, bypassing the need for multimodal analysis or cross-session reasoning.
   - For example, if the question asks about the tempo of an audio file, and the dialogue text already describes it as "lively and upbeat", that is leakage even if the answer also requires listening to the audio.
3. **media_consistency**: Does the dialogue text contradict the actual multimodal content? Check whether any user message, assistant response, or summary describes, reacts to, or characterizes the media in a way that conflicts with what the attached files actually contain. Also check whether referring expressions in the question unambiguously point to the correct media target — if a distractor media in the dialogue matches the referring expression equally well, that is a fail. If the dialogue never describes media content (only shares it neutrally), this is a pass.

## Important Context
- Evidence items come from multi-session dialogues. Each session has a date and session_id.
- Items sharing the same session_id belong to the same conversation and share natural co-reference.
- Cross-session references can be resolved by comparing session dates.
- For multimodal evidence, the actual media files are attached. Judge whether they match the described evidence content.

## Output Format (JSON only, no markdown)
{{
  "question_id": "{question_id}",
  "evidence_presence": {{"verdict": "pass"|"fail", "reason": "...", "missing_evidence_ids": []}},
  "no_answer_leakage": {{"verdict": "pass"|"fail", "reason": "..."}},
  "media_consistency": {{"verdict": "pass"|"fail", "reason": "..."}},
  "overall_verdict": "pass"|"fail",
  "overall_reason": "..."
}}

Rules:
- overall_verdict is "pass" only if ALL three dimensions pass.
- For multi-hop/cross-session questions, individual evidence pieces in different sessions is NOT leakage.
- Be strict but fair. Respond with JSON only.
"""


# ── Audit one QA ─────────────────────────────────────────────────────────────

def audit_one(qa: dict, session_map: dict, model: str, max_retries: int = 3) -> dict:
    qid = qa.get("question_id", "")
    session_ids = get_qa_session_ids(qa)
    dialogue_text, media_list = flatten_sessions(session_ids, session_map)

    prompt = AUDIT_PROMPT.format(
        dialogue_text=dialogue_text if dialogue_text.strip() else "(no relevant sessions found)",
        question_id=qid,
        task_type=qa.get("task_type", ""),
        question=qa.get("question", ""),
        answer=qa.get("full_answer", "") or qa.get("answer", ""),
        evidence_summary=format_evidence_summary(qa.get("evidence_chain", [])),
    )

    for attempt in range(max_retries):
        if media_list:
            success, resp = call_gemini_multimodal(
                prompt=prompt, media=media_list, model=model,
                temperature=0.1, max_tokens=20480, max_retries=3,
                max_time=300, clean_json=True,
            )
        else:
            success, resp = call_gemini(
                prompt=prompt, model=model,
                temperature=0.1, max_tokens=20480, max_retries=3,
            )

        if not success:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"question_id": qid, "error": resp, "overall_verdict": "error"}
        try:
            result = json.loads(resp)
            result["question_id"] = qid
            return result
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"question_id": qid, "error": f"JSON parse failed: {resp[:200]}", "overall_verdict": "error"}
    return {"question_id": qid, "error": "max retries", "overall_verdict": "error"}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dialogue embedding quality audit (multimodal)")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--qa_file", type=Path, default=None)
    parser.add_argument("--dialogues_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "your-model-name"))
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    qa_file = args.qa_file or (base / "rewrite" / "output" / "04_final_qa" / args.run_id / "final_qa_aggregated.jsonl")
    dialogues_file = args.dialogues_file or (base / "session" / "output" / "02_dialogue_generation" / args.run_id / "all_dialogues.json")
    output_dir = args.output_dir or (Path(__file__).resolve().parent / "output" / "02_embed_audit" / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "embed_audit_results.jsonl"

    qa_items = []
    with open(qa_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_items.append(json.loads(line))

    dialogues_data = json.loads(dialogues_file.read_text("utf-8"))
    session_map = build_session_map(dialogues_data)

    print(f"Embed Audit: {len(qa_items)} QAs, {len(session_map)} sessions")
    print(f"Output: {output_file}")
    print(f"Model: {args.model}, Workers: {args.workers}")

    done_ids: set[str] = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line).get("question_id", ""))
                    except Exception:
                        pass

    todo = [qa for qa in qa_items if qa.get("question_id", "") not in done_ids]

    if not todo:
        pass
    else:
        write_lock = threading.Lock()
        with open(output_file, "a", encoding="utf-8") as out:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(audit_one, qa, session_map, args.model, args.max_retries): qa
                    for qa in todo
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Embed Auditing"):
                    result = future.result()
                    with write_lock:
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out.flush()

    all_results = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    qa_type_map = {item["question_id"]: item.get("task_type", "?") for item in qa_items}
    dims = ["evidence_presence", "no_answer_leakage", "media_consistency"]

    type_stats: Dict[str, Dict] = {}
    for r in all_results:
        t = qa_type_map.get(r.get("question_id", ""), "?")
        if t not in type_stats:
            type_stats[t] = {"total": 0, "overall_pass": 0, **{d: 0 for d in dims}}
        type_stats[t]["total"] += 1
        if r.get("overall_verdict") == "pass":
            type_stats[t]["overall_pass"] += 1
        for d in dims:
            if isinstance(r.get(d), dict) and r[d].get("verdict") == "pass":
                type_stats[t][d] += 1

    print(f"\n{'='*80}")
    print(f"{'Type':<6} {'Total':>5} {'Pass':>5} {'Rate':>6}  | {'Presence':>9} {'NoLeak':>8} {'MediaOK':>8}")
    print(f"{'-'*80}")
    grand = {"total": 0, "overall_pass": 0, **{d: 0 for d in dims}}
    for t in sorted(type_stats):
        s = type_stats[t]
        n = s["total"]
        rate = f"{s['overall_pass']/n*100:.1f}%" if n else "N/A"
        dr = [f"{s[d]/n*100:.0f}%" if n else "N/A" for d in dims]
        print(f"{t:<6} {n:>5} {s['overall_pass']:>5} {rate:>6}  | {dr[0]:>9} {dr[1]:>8} {dr[2]:>8}")
        for k in grand:
            grand[k] += s[k]
    n = grand["total"]
    rate = f"{grand['overall_pass']/n*100:.1f}%" if n else "N/A"
    dr = [f"{grand[d]/n*100:.0f}%" if n else "N/A" for d in dims]
    print(f"{'-'*80}")
    print(f"{'ALL':<6} {n:>5} {grand['overall_pass']:>5} {rate:>6}  | {dr[0]:>9} {dr[1]:>8} {dr[2]:>8}")
    print(f"{'='*80}")

    summary = {
        "total": n, "pass": grand["overall_pass"],
        "fail": n - grand["overall_pass"] - sum(1 for r in all_results if r.get("overall_verdict") == "error"),
        "error": sum(1 for r in all_results if r.get("overall_verdict") == "error"),
        "pass_rate": round(grand["overall_pass"] / n * 100, 2) if n else 0,
        "by_type": type_stats,
        "by_dimension": {d: round(grand[d] / n * 100, 2) if n else 0 for d in dims},
    }
    (output_dir / "embed_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nSummary → {output_dir / 'embed_audit_summary.json'}")


if __name__ == "__main__":
    main()
