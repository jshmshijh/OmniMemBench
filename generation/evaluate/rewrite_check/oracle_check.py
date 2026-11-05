#!/usr/bin/env python3

import json
import time
import re
import sys
import argparse
import subprocess
import threading
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


sys.path.append(str(Path(__file__).parent.parent.parent / "preprocess"))
from call_gemini import call_gemini

API_KEY: Optional[str] = None
BASE_URL = "https://localhost:8000/v1"
MODEL = os.environ.get("LLM_MODEL", "your-model-name")
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

QA_TYPES = ["MH", "CR", "ADV", "REF", "TR", "UPD", "SH"]

DEFAULT_WORKERS = 20
DEFAULT_RETRIES = 10
DEFAULT_API_RETRIES = 5
RETRY_WAIT_CAP_S = 30

file_lock = threading.Lock()


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_mime_type(url: str, modality_type: str) -> str:
    url_lower = url.lower().split("?")[0]
    if modality_type == "audio":
        if url_lower.endswith(".mp3"):   return "audio/mp3"
        if url_lower.endswith(".wav"):   return "audio/wav"
        if url_lower.endswith(".m4a"):   return "audio/mp4"
        if url_lower.endswith(".ogg"):   return "audio/ogg"
        return "audio/wav"
    if modality_type == "image":
        if url_lower.endswith(".jpg") or url_lower.endswith(".jpeg"): return "image/jpeg"
        if url_lower.endswith(".png"):   return "image/png"
        if url_lower.endswith(".webp"):  return "image/webp"
        if url_lower.endswith(".gif"):   return "image/gif"
        return "image/jpeg"
    if modality_type == "video":
        if url_lower.endswith(".mp4"):   return "video/mp4"
        if url_lower.endswith(".webm"):  return "video/webm"
        if url_lower.endswith(".avi"):   return "video/x-msvideo"
        if url_lower.endswith(".mkv"):   return "video/x-matroska"
        return "video/mp4"
    return "application/octet-stream"


def build_evidence_parts(evidence_chain: List[Dict]) -> List[Dict]:
    parts: List[Dict] = [{"text": "## Evidence\n"}]

    for i, ev in enumerate(evidence_chain, 1):
        ev_type = ev.get("type", "text")
        session_id = ev.get("session_id", "")
        timestamp = ev.get("timestamp", "")

        header = f"\n### Evidence {i} [{ev_type.upper()}]"
        if session_id:
            header += f" (Session: {session_id}"
            if timestamp:
                header += f", {timestamp}"
            header += ")"
        parts.append({"text": header + "\n"})

        if ev_type == "text":
            content = ev.get("evidence", ev.get("content", ""))
            parts.append({"text": content})

        elif ev_type in ("image", "video", "audio"):
            url = ev.get("url", "")
            caption = ev.get("caption", "")

            if url:
                mime = get_mime_type(url, ev_type)
                parts.append({"fileData": {"mimeType": mime, "fileUri": url}})
                if caption:
                    parts.append({"text": f"[Description: {caption}]"})
            elif caption:
                parts.append({"text": f"[{ev_type.upper()} — Description: {caption}]"})
            else:
                parts.append({"text": f"[{ev_type.upper()} — no content available]"})

    return parts


def call_gemini_multimodal(
    parts: List[Dict],
    model: str = MODEL,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = DEFAULT_API_RETRIES,
) -> Tuple[bool, str]:
    if not API_KEY:
        return False, "Missing API key. Provide --api_key or set LLM_API_KEY."

    for attempt in range(max_retries):
        try:
            payload = {
                "model": model,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
                "contents": [{"parts": parts, "role": "user"}],
            }

            curl_command = [
                "curl", "--location",
                f"{BASE_URL}/chat/completions",
                "--header", "Content-Type: application/json",
                "--header", f"Authorization: Bearer {API_KEY}",
                "--data", json.dumps(payload),
                "--silent", "--show-error",
                "--max-time", "300",
            ]

            result = subprocess.run(
                curl_command, capture_output=True, text=True, timeout=310
            )

            if result.returncode != 0:
                raise RuntimeError(f"curl failed: {result.stderr}")

            resp = json.loads(result.stdout)

            if "candidates" in resp and resp["candidates"]:
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
                if not text:
                    raise ValueError("Empty response")
                text = text.strip()
                text = re.sub(r"^```json\s*\n?", "", text)
                text = re.sub(r"^```\s*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
                return True, text
            else:
                raise ValueError(f"Unexpected response: {resp}")

        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return False, "Request timeout"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return False, str(e)

    return False, "Max retries exceeded"


ANSWER_INSTRUCTION = """
## Question
{question}

## Task
Based **solely** on the evidence above, answer the question concisely and accurately.

Respond in JSON format:
{{
  "answer": "<your concise answer>",
  "reasoning": "<brief explanation citing which evidence supports your answer>"
}}
"""

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a model's answer is correct.

Question: {question}

Expected answer (short): {expected_short}
Expected answer (full): {expected_full}

Model's answer: {model_answer}

Evaluate the model's answer:
- "correct": matches the expected answer in substance
- "partial": partially correct (contains some but not all key information)
- "incorrect": wrong or clearly missing key information

Respond in JSON:
{{
  "result": "correct" or "partial" or "incorrect",
  "reasoning": "<one sentence explanation>"
}}
"""


def parse_json_response(response: str) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        return json.loads(response), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {str(e)[:100]}"


def _retry_sleep(attempt: int, max_wait_s: int) -> None:
    time.sleep(min(2 ** attempt, max(1, max_wait_s)))


def process_one_qa(
    qa_record: Dict,
    run_id: str,
    output_file: Path,
    idx: int,
    total: int,
    max_retries: int = DEFAULT_RETRIES,
    api_retries: int = DEFAULT_API_RETRIES,
    max_retry_wait_s: int = RETRY_WAIT_CAP_S,
) -> Tuple[bool, str]:
    question_id = qa_record.get("original_question_id", "Unknown")
    best = qa_record.get("best_rewrite", {})
    question = best.get("rewritten_question", qa_record.get("original_question", ""))
    expected_short = best.get("short_answer", "")
    expected_full = best.get("full_answer", best.get("short_answer", ""))
    qa_type = best.get("task_type", "Unknown")
    evidence_chain = best.get("evidence_chain", [])

    has_multimodal = any(
        ev.get("type") in ("image", "video", "audio") for ev in evidence_chain
    )

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            evidence_parts = build_evidence_parts(evidence_chain)
            answer_parts = evidence_parts + [
                {"text": ANSWER_INSTRUCTION.format(question=question)}
            ]

            ok, answer_resp = call_gemini_multimodal(
                parts=answer_parts,
                temperature=0.3,
                max_tokens=2048,
                max_retries=api_retries,
            )

            if not ok:
                last_error = f"Answer API failed: {answer_resp}"
                if attempt < max_retries:
                    _retry_sleep(attempt, max_retry_wait_s)
                    continue
                break

            answer_data, err = parse_json_response(answer_resp)
            if err or not answer_data:
                model_answer = answer_resp[:500]
                answer_reasoning = ""
            else:
                model_answer = answer_data.get("answer", answer_resp[:500])
                answer_reasoning = answer_data.get("reasoning", "")

            # ---- Step 2: Judge ----
            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=question,
                expected_short=expected_short,
                expected_full=expected_full,
                model_answer=model_answer,
            )

            ok2, judge_resp = call_gemini(
                prompt=judge_prompt,
                model=MODEL,
                temperature=0.1,
                max_tokens=512,
                max_retries=api_retries,
            )

            if not ok2:
                last_error = f"Judge API failed: {judge_resp}"
                if attempt < max_retries:
                    _retry_sleep(attempt, max_retry_wait_s)
                    continue
                break

            judge_data, err2 = parse_json_response(judge_resp)
            if err2 or not judge_data:
                last_error = f"Judge parse failed: {err2}"
                if attempt < max_retries:
                    _retry_sleep(attempt, max_retry_wait_s)
                    continue
                break

            judge_result = judge_data.get("result", "")
            if judge_result not in ("correct", "partial", "incorrect"):
                last_error = f"Invalid judge result: {judge_result}"
                if attempt < max_retries:
                    _retry_sleep(attempt, max_retry_wait_s)
                    continue
                break

            record = {
                "run_id": run_id,
                "question_id": question_id,
                "qa_type": qa_type,
                "question": question,
                "expected_short": expected_short,
                "expected_full": expected_full,
                "model_answer": model_answer,
                "answer_reasoning": answer_reasoning,
                "judge_result": judge_result,
                "judge_reasoning": judge_data.get("reasoning", ""),
                "has_multimodal": has_multimodal,
                "evidence_count": len(evidence_chain),
            }
            with file_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            status = "✓" if judge_result == "correct" else ("△" if judge_result == "partial" else "✗")
            print(f"  [{idx}/{total}] {status} {question_id} ({qa_type}) → {judge_result}")
            return True, question_id

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                _retry_sleep(attempt, max_retry_wait_s)
                continue

    fail_record = {
        "run_id": run_id,
        "question_id": question_id,
        "qa_type": qa_type,
        "status": "failed",
        "error": last_error,
    }
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")

    print(f"  [{idx}/{total}] ✗ {question_id} FAILED: {last_error}")
    return False, question_id


def load_validated_for_run(validated_dir: Path) -> List[Dict]:
    records = []
    for qa_type in QA_TYPES:
        fpath = validated_dir / f"{qa_type.lower()}_validated.jsonl"
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    return records


def get_processed_ids(output_file: Path) -> set:
    if not output_file.exists():
        return set()
    ids = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("question_id")
                if qid:
                    ids.add(qid)
            except Exception:
                pass
    return ids


def compute_summary(results_file: Path, run_id: str) -> Dict:
    by_type: Dict[str, Dict] = {}
    total = correct = partial = incorrect = failed = 0

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("status") == "failed":
                failed += 1
                continue

            judge = obj.get("judge_result", "")
            qa_type = obj.get("qa_type", "Unknown")
            total += 1

            t = by_type.setdefault(qa_type, {"total": 0, "correct": 0, "partial": 0, "incorrect": 0})
            t["total"] += 1

            if judge == "correct":
                correct += 1
                t["correct"] += 1
            elif judge == "partial":
                partial += 1
                t["partial"] += 1
            else:
                incorrect += 1
                t["incorrect"] += 1

    for t in by_type.values():
        n = t["total"]
        t["accuracy"] = round(t["correct"] / n, 4) if n > 0 else 0.0
        t["accuracy_with_partial"] = round((t["correct"] + t["partial"] * 0.5) / n, 4) if n > 0 else 0.0

    return {
        "run_id": run_id,
        "total_evaluated": total,
        "failed": failed,
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "accuracy_with_partial": round((correct + partial * 0.5) / total, 4) if total > 0 else 0.0,
        "by_type": by_type,
    }


def process_run(
    run_id: str,
    validated_base_dir: Path,
    output_base_dir: Path,
    workers: int,
    max_retries: int,
    api_retries: int,
    max_retry_wait_s: int,
) -> Optional[Dict]:
    validated_dir = validated_base_dir / run_id
    if not validated_dir.exists():
        return None

    output_dir = output_base_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"
    summary_file = output_dir / "summary.json"

    print(f"\n{'='*60}")
    print(f"run_id: {run_id}")
    print(f"{'='*60}")

    all_records = load_validated_for_run(validated_dir)
    if not all_records:
        return None

    processed_ids = get_processed_ids(results_file)
    if processed_ids:
        to_process = [r for r in all_records if r.get("original_question_id") not in processed_ids]
    else:
        to_process = all_records
    if not to_process:
        pass
    else:

        success_count = failed_count = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_one_qa,
                    record, run_id, results_file,
                    len(processed_ids) + idx + 1, len(all_records),
                    max_retries, api_retries, max_retry_wait_s,
                ): record
                for idx, record in enumerate(to_process)
            }
            for future in as_completed(futures):
                try:
                    ok, _ = future.result()
                    if ok:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1


    summary = compute_summary(results_file, run_id)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    for qa_type, stats in sorted(summary["by_type"].items()):
        print(f"    {qa_type}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    return summary


def resolve_run_ids(run_id_arg: str, validated_base_dir: Path) -> List[str]:
    if run_id_arg.lower() == "all":
        dirs = sorted([d.name for d in validated_base_dir.iterdir() if d.is_dir()])
        if not dirs:
            raise SystemExit(f"...")
        return dirs
    return [r.strip() for r in run_id_arg.split(",") if r.strip()]


def main():
    global API_KEY

    parser = argparse.ArgumentParser(description="Oracle Check: ...")
    parser.add_argument("--run_id", type=str, default="all",
                        help="")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"...")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_RETRIES,
                        help=f"...")
    parser.add_argument("--api_retries", type=int, default=DEFAULT_API_RETRIES,
                        help=f"...")
    parser.add_argument("--retry_wait_cap_s", type=int, default=RETRY_WAIT_CAP_S,
                        help=f"...")
    parser.add_argument("--api_key", type=str, default=None,
                        help="")
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE,
                        help=f".env ...")
    args = parser.parse_args()

    load_env_file(args.env_file)
    API_KEY = args.api_key or os.environ.get("LLM_API_KEY")
    if not API_KEY:
        raise SystemExit("Missing API key. Provide --api_key, set LLM_API_KEY, or configure it in .env.")

    base_dir = Path(__file__).parent
    step3_dir = base_dir.parent.parent / "rewrite"
    validated_base_dir = step3_dir / "output" / "03_qa_validated"
    output_base_dir = base_dir / "output"

    print("=" * 60)
    print("Oracle Check")
    print("=" * 60)

    run_ids = resolve_run_ids(args.run_id, validated_base_dir)
    print(f"run_ids: {run_ids}\n")

    all_summaries = []
    for run_id in run_ids:
        summary = process_run(
            run_id=run_id,
            validated_base_dir=validated_base_dir,
            output_base_dir=output_base_dir,
            workers=args.workers,
            max_retries=args.max_retries,
            api_retries=args.api_retries,
            max_retry_wait_s=args.retry_wait_cap_s,
        )
        if summary:
            all_summaries.append(summary)

    if len(all_summaries) > 1:
        combined_total = sum(s["total_evaluated"] for s in all_summaries)
        combined_correct = sum(s["correct"] for s in all_summaries)
        combined_partial = sum(s["partial"] for s in all_summaries)
        combined_summary = {
            "run_ids": run_ids,
            "total_evaluated": combined_total,
            "correct": combined_correct,
            "partial": combined_partial,
            "incorrect": sum(s["incorrect"] for s in all_summaries),
            "failed": sum(s["failed"] for s in all_summaries),
            "accuracy": round(combined_correct / combined_total, 4) if combined_total > 0 else 0.0,
            "accuracy_with_partial": round(
                (combined_correct + combined_partial * 0.5) / combined_total, 4
            ) if combined_total > 0 else 0.0,
            "per_run": all_summaries,
        }
        combined_file = output_base_dir / "combined_summary.json"
        combined_file.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(combined_summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("Combined Summary")
        print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"{'='*60}")

    total_failed = sum(s.get("failed", 0) for s in all_summaries)
    if total_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
