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


# ── Extract media from evidence chain ────────────────────────────────────────

def extract_media(evidence_chain: list, mm_registry: dict = None) -> List[Dict[str, str]]:
    """Extract media URLs from evidence chain for multimodal audit.
    
    Uses mm_registry to resolve mm_id -> url when the evidence item
    does not contain a direct url field.
    """
    if mm_registry is None:
        mm_registry = {}
    media_list = []
    for e in evidence_chain:
        modality = (e.get("type") or "").lower()
        if modality not in ("image", "audio", "video"):
            continue

        mm_id = (e.get("mm_id") or "").strip()
        url = (e.get("url") or "").strip()

        # Fallback: resolve url from mm_registry via mm_id
        if not _is_http_url(url) and mm_id and mm_id in mm_registry:
            url = (mm_registry[mm_id].get("url") or "").strip()

        if not _is_http_url(url):
            continue

        info = _infer_media_info(url, modality)
        info["mm_id"] = mm_id
        media_list.append(info)
    return media_list


# ── Prompt ───────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are a strict benchmark quality auditor. Given a QA item from a memory benchmark, judge its quality.

## QA Item
- **Question ID**: {question_id}
- **Task Type**: {task_type} ({task_subtype})
- **Question**: {question}
- **Short Answer**: {short_answer}
- **Full Answer**: {full_answer}
- **Evidence Chain**:
{evidence_chain}

## Evaluation Criteria
Rate each dimension as "pass" or "fail" with a brief reason:

1. **question_clarity**: Is the question clear, specific, and unambiguous? Would a human understand exactly what is being asked?
2. **evidence_sufficiency**: Does the evidence chain contain enough information to derive the answer? Is any critical piece missing? 
   - Focus on: Are the rules, criteria, and factual premises sufficient? Is multimodal evidence described adequately?
   - Do NOT flag: The question may use referring expressions (e.g., "urban digital art section") that will appear in the dialogue but not in the evidence chain. This is expected and not a deficiency.
3. **answer_correctness**: Is the answer correct given the evidence chain?{mm_instruction}

## Important Context
- Evidence items come from multi-session dialogues. Each item has a session_id and timestamp indicating when and where the conversation occurred.
- Items sharing the same session_id belong to the same conversation session and naturally share co-reference (e.g., "the track" refers to a track mentioned earlier in that session).
- Relative time expressions (e.g., "18 days ago", "last month") are anchored by the item's timestamp.
- Cross-session references (e.g., "the track we reviewed last month") can be resolved by comparing timestamps across sessions.
- Do NOT flag missing information that is resolvable from session metadata (timestamps, session grouping).

## Output Format (JSON only, no markdown)
{{
  "question_id": "{question_id}",
  "question_clarity": {{"verdict": "pass"|"fail", "reason": "..."}},
  "evidence_sufficiency": {{"verdict": "pass"|"fail", "reason": "..."}},
  "answer_correctness": {{"verdict": "pass"|"fail", "reason": "..."}},
  "overall_verdict": "pass"|"fail",
  "overall_reason": "..."
}}

Rules:
- overall_verdict is "pass" only if ALL three dimensions pass.
- Be strict but fair. Flag genuine issues, not stylistic preferences.
- Respond with JSON only.
"""

MM_INSTRUCTION = """
   - CRITICAL: The actual multimodal files (images/audio/video) from the evidence chain are attached to this message. For answer_correctness, you MUST perceive the actual media yourself, use it together with the text evidence to independently derive the correct answer, and then check whether the provided answer matches. Do NOT trust the text descriptions of media content in the evidence chain — they may contain errors. The attached media is the sole ground truth for any media-related facts. Each attached media corresponds to the mm_id referenced in the evidence chain."""


def format_evidence_chain(chain: list) -> str:
    if not chain:
        return "  (empty)"
    lines = []
    for i, e in enumerate(chain, 1):
        meta = []
        if e.get("session_id"):
            meta.append(f"session={e['session_id']}")
        if e.get("timestamp"):
            meta.append(f"time={e['timestamp']}")
        meta.append(f"type={e.get('type', 'text')}")
        if e.get("mm_id"):
            meta.append(f"mm_id={e['mm_id']}")
        header = ", ".join(meta)
        modality = (e.get("type") or "").lower()
        if modality in ("image", "audio", "video"):
            mm_id = e.get("mm_id", "unknown")
            content = f"[Refer to attached media: mm_id={mm_id}]"
        else:
            content = e.get("evidence", "") or e.get("content", "")
        lines.append(f"  {i}. [{header}] {content}")
    return "\n".join(lines)


def audit_one(item: dict, model: str, mm_registry: dict, max_retries: int = 3) -> dict:
    chain = item.get("evidence_chain", [])
    media_list = extract_media(chain, mm_registry)

    prompt = PROMPT_TEMPLATE.format(
        question_id=item.get("question_id", ""),
        task_type=item.get("task_type", ""),
        task_subtype=item.get("task_subtype", ""),
        question=item.get("question", ""),
        short_answer=item.get("short_answer", ""),
        full_answer=item.get("full_answer", ""),
        evidence_chain=format_evidence_chain(chain),
        mm_instruction=MM_INSTRUCTION if media_list else "",
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
            return {"question_id": item.get("question_id"), "error": resp, "overall_verdict": "error"}
        try:
            result = json.loads(resp)
            result["question_id"] = item.get("question_id", "")
            return result
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"question_id": item.get("question_id"), "error": f"JSON parse failed: {resp[:200]}", "overall_verdict": "error"}
    return {"question_id": item.get("question_id"), "error": "max retries", "overall_verdict": "error"}


def main():
    parser = argparse.ArgumentParser(description="QA quality audit via LLM judge (multimodal)")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--mm_registry", type=Path, default=None,
                        help="Path to mm_registry.json for resolving mm_id -> url")
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "your-model-name"))
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    input_file = args.input or (base / "rewrite" / "output" / "04_final_qa" / args.run_id / "final_qa_aggregated.jsonl")
    output_dir = args.output_dir or (Path(__file__).resolve().parent / "output" / "01_qa_audit" / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "qa_audit_results.jsonl"

    # Load mm_registry
    registry_path = args.mm_registry or (base / "filter" / "qa" / args.run_id / "mm_registry.json")
    mm_registry = {}
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            mm_registry = json.load(f)
        print(f"Loaded mm_registry: {len(mm_registry)} entries from {registry_path}")
    else:
        print(f"WARNING: mm_registry not found at {registry_path}, media URLs will not be resolved")

    items = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    mm_count = sum(1 for item in items if extract_media(item.get("evidence_chain", []), mm_registry))
    print(f"QA Audit: {len(items)} items ({mm_count} with media) from {input_file}")
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

    todo = [item for item in items if item.get("question_id", "") not in done_ids]

    if not todo:
        pass
    else:
        write_lock = threading.Lock()
        with open(output_file, "a", encoding="utf-8") as out:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(audit_one, item, args.model, mm_registry, args.max_retries): item for item in todo}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Auditing"):
                    result = future.result()
                    with write_lock:
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out.flush()

    all_results = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    qa_type_map = {item["question_id"]: item.get("task_type", "?") for item in items}
    dims = ["question_clarity", "evidence_sufficiency", "answer_correctness"]

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
    print(f"{'Type':<6} {'Total':>5} {'Pass':>5} {'Rate':>6}  | {'Clarity':>8} {'Evidence':>8} {'Answer':>8}")
    print(f"{'-'*80}")
    grand = {"total": 0, "overall_pass": 0, **{d: 0 for d in dims}}
    for t in sorted(type_stats):
        s = type_stats[t]
        n = s["total"]
        rate = f"{s['overall_pass']/n*100:.1f}%" if n else "N/A"
        dr = [f"{s[d]/n*100:.0f}%" if n else "N/A" for d in dims]
        print(f"{t:<6} {n:>5} {s['overall_pass']:>5} {rate:>6}  | {dr[0]:>8} {dr[1]:>8} {dr[2]:>8}")
        for k in grand:
            grand[k] += s[k]
    n = grand["total"]
    rate = f"{grand['overall_pass']/n*100:.1f}%" if n else "N/A"
    dr = [f"{grand[d]/n*100:.0f}%" if n else "N/A" for d in dims]
    print(f"{'-'*80}")
    print(f"{'ALL':<6} {n:>5} {grand['overall_pass']:>5} {rate:>6}  | {dr[0]:>8} {dr[1]:>8} {dr[2]:>8}")
    print(f"{'='*80}")

    summary = {
        "total": n, "pass": grand["overall_pass"],
        "fail": n - grand["overall_pass"] - sum(1 for r in all_results if r.get("overall_verdict") == "error"),
        "error": sum(1 for r in all_results if r.get("overall_verdict") == "error"),
        "pass_rate": round(grand["overall_pass"] / n * 100, 2) if n else 0,
        "by_type": type_stats,
        "by_dimension": {d: round(grand[d] / n * 100, 2) if n else 0 for d in dims},
    }
    (output_dir / "qa_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nSummary → {output_dir / 'qa_audit_summary.json'}")


if __name__ == "__main__":
    main()
