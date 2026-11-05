#!/usr/bin/env python3
"""
Long-context API evaluation for OmniMemBench datasets.

This script bypasses omnimembench/engine: it feeds the full conversation history (optionally including
raw omni media parts) directly to the API, then evaluates QA answers.

Input dataset (collection/*.json) expected keys (best-effort):
- character_profile: {persona_summary, conversation_style}
- multi_session_dialogues: [{session_id, date, dialogues:[{round,user,assistant,input_image,image_caption,image_id,input_audio,audio_caption,audio_id}]}]
- QAs: [{type, question, answer, full_answer, session_id, clue}]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


DEFAULT_BASE_URL = "https://localhost:8000/v1"
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "your-model-name")


def _is_http_url(value: str) -> bool:
    if not isinstance(value, str):
        return False
    scheme = urlparse(value).scheme.lower()
    return scheme in {"http", "https"}


def _suffix_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
    except Exception:
        path = url
    return Path(path).suffix.lower()


def infer_modality_and_mime(url: str) -> Tuple[str, str]:
    """
    Returns:
      (modality_type, mime_type) where modality_type in {"image","audio","video","unknown"}.
    """
    suffix = _suffix_from_url(url)

    if suffix in {".jpg", ".jpeg"}:
        return "image", "image/jpeg"
    if suffix == ".png":
        return "image", "image/png"
    if suffix == ".webp":
        return "image", "image/webp"
    if suffix == ".gif":
        return "image", "image/gif"

    if suffix == ".mp3":
        return "audio", "audio/mp3"
    if suffix == ".wav":
        return "audio", "audio/wav"
    if suffix == ".m4a":
        return "audio", "audio/mp4"
    if suffix == ".ogg":
        return "audio", "audio/ogg"

    if suffix == ".mp4":
        return "video", "video/mp4"
    if suffix == ".webm":
        return "video", "video/webm"
    if suffix == ".avi":
        return "video", "video/x-msvideo"
    if suffix == ".mkv":
        return "video", "video/x-matroska"

    return "unknown", "application/octet-stream"


def parse_input_form(spec: str) -> Dict[str, str]:
    """
    Parse per-modality input form.

    Examples:
      - "raw" -> all modalities raw
      - "caption" -> all modalities caption
      - "image=raw,audio=caption,video=raw"
    """
    spec = (spec or "").strip()
    if not spec:
        return {"image": "raw", "audio": "raw", "video": "raw"}

    if spec in {"raw", "caption"}:
        return {"image": spec, "audio": spec, "video": spec}

    result: Dict[str, str] = {"image": "raw", "audio": "raw", "video": "raw"}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --input_form item: {item!r}. Expected k=v.")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if key not in {"image", "audio", "video"}:
            raise ValueError(f"Invalid modality in --input_form: {key!r}")
        if value not in {"raw", "caption"}:
            raise ValueError(f"Invalid mode in --input_form for {key}: {value!r}")
        result[key] = value
    return result


def _dump_http_error_context(err: str) -> str:
    """
    LLM sometimes wraps upstream errors; keep a short readable snippet.
    """
    s = (err or "").strip()
    if len(s) <= 2000:
        return s
    return s[:2000] + " ...<truncated>"


def input_form_tag(input_form: Dict[str, str]) -> str:
    """
    Make a short stable tag for filenames, e.g.:
      {"image":"raw","audio":"caption","video":"raw"} -> "imgr_audc_vidr"
    """
    def one(mod: str, short: str) -> str:
        mode = (input_form.get(mod) or "raw").lower()
        return f"{short}{'r' if mode == 'raw' else 'c'}"

    return "_".join([one("image", "imgr"), one("audio", "aud"), one("video", "vid")]).replace("imgrr", "imgr")


def build_filedata_part(url: str, mime_type: str) -> Dict[str, Any]:
    return {"fileData": {"mimeType": mime_type, "fileUri": url}}


def clean_model_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = re.sub(r"^```json\s*\n?", "", text)
    if text.startswith("```"):
        text = re.sub(r"^```\s*\n?", "", text)
    if text.endswith("```"):
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def llm_api_chat_native(
    *,
    api_key: str,
    base_url: str,
    model: str,
    contents: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: int = 120,
    max_retries: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns:
      (assistant_text, raw_response_json)
    """
    last_error: Optional[str] = None
    keep_payload = os.environ.get("KEEP_PAYLOAD", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    for attempt in range(max_retries):
        tmp_payload_path: Optional[str] = None
        try:
            payload: Dict[str, Any] = {
                "model": model,
                "contents": contents,
            }
            # Do NOT pass OpenAI-style top-level `temperature`/`max_tokens`, and also avoid
            # `generationConfig` here because some provider routes reject it.
            # If you want to tune decoding, add it only after confirming the upstream schema.
            _ = (temperature, max_tokens)

            # Payload can be very large for long-context / omni parts.
            # Avoid OS argv length limits by writing JSON to a temp file.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(payload, f, ensure_ascii=False)
                tmp_payload_path = f.name

            curl_command = [
                "curl",
                "--location",
                f"{base_url.rstrip('/')}/chat/completions",
                "--header",
                "Content-Type: application/json",
                "--header",
                f"Authorization: Bearer {api_key}",
                "--data-binary",
                f"@{tmp_payload_path}",
                "--silent",
                "--show-error",
                "--max-time",
                str(int(timeout_seconds)),
                "--write-out",
                "\nHTTPSTATUS:%{http_code}",
            ]

            run_result = subprocess.run(
                curl_command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 10,
            )

            if run_result.returncode != 0:
                last_error = f"curl failed: {run_result.stderr}"
                raise RuntimeError(last_error)

            stdout = run_result.stdout
            if "\nHTTPSTATUS:" not in stdout:
                last_error = f"Unexpected curl output (missing HTTPSTATUS): {stdout[:2000]}"
                raise RuntimeError(last_error)

            body, status_line = stdout.rsplit("\nHTTPSTATUS:", 1)
            status_code = int(status_line.strip()) if status_line.strip().isdigit() else None

            try:
                response_json = json.loads(body)
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}; body head: {body[:1000]}"
                raise RuntimeError(last_error)

            if status_code is not None and status_code >= 400:
                payload_hint = ""
                if tmp_payload_path:
                    payload_hint = f" (payload: {tmp_payload_path})"
                last_error = f"HTTP {status_code}{payload_hint}: {response_json}"
                raise RuntimeError(last_error)

            if "candidates" not in response_json or not response_json["candidates"]:
                last_error = f"Unexpected response format (no candidates): {response_json}"
                raise RuntimeError(last_error)

            content = response_json["candidates"][0].get("content", {})
            parts = content.get("parts") or []
            if not parts or "text" not in parts[0]:
                last_error = f"Unexpected response format (no text parts): {response_json}"
                raise RuntimeError(last_error)

            return clean_model_text(parts[0]["text"]), response_json

        except subprocess.TimeoutExpired:
            last_error = "Request timeout"
        except Exception as e:
            last_error = _dump_http_error_context(str(e))
        finally:
            if tmp_payload_path and not keep_payload:
                try:
                    os.unlink(tmp_payload_path)
                except Exception:
                    pass

        if attempt < max_retries - 1:
            sleep_seconds = min(2 * (2**attempt), 60)
            time.sleep(sleep_seconds)
            continue
        raise RuntimeError(f"LLM call failed after {max_retries} attempts. Last error: {last_error}")

    raise RuntimeError(f"LLM call failed. Last error: {last_error}")


def normalize_answer(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match(pred: str, gt: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0


def token_f1(pred: str, gt: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gt_tokens = normalize_answer(gt).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common: Dict[str, int] = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gt_tokens:
        if common.get(t, 0) > 0:
            common[t] -= 1
            overlap += 1
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(gt_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


@dataclass
class DialogueMedia:
    modality: str  # image|audio|video|unknown
    url: str
    caption: str
    media_id: str
    mime_type: str


def extract_dialogue_media(dialog: Dict[str, Any]) -> List[DialogueMedia]:
    media: List[DialogueMedia] = []

    def add_items(
        *,
        urls_key: str,
        caps_key: str,
        ids_key: str,
        forced_modality: Optional[str] = None,
    ) -> None:
        urls = dialog.get(urls_key) or []
        caps = dialog.get(caps_key) or []
        ids = dialog.get(ids_key) or []
        if not isinstance(urls, list) or not urls:
            return
        for idx, url in enumerate(urls):
            if not isinstance(url, str) or not url:
                continue
            caption = caps[idx] if idx < len(caps) and isinstance(caps[idx], str) else ""
            media_id = ids[idx] if idx < len(ids) and isinstance(ids[idx], str) else ""

            inferred_modality, mime_type = infer_modality_and_mime(url)
            modality = forced_modality or inferred_modality
            media.append(
                DialogueMedia(
                    modality=modality,
                    url=url,
                    caption=caption,
                    media_id=media_id,
                    mime_type=mime_type,
                )
            )

    add_items(urls_key="input_image", caps_key="image_caption", ids_key="image_id")
    add_items(urls_key="input_audio", caps_key="audio_caption", ids_key="audio_id", forced_modality="audio")
    return media


def extract_qa_media(qa: Dict[str, Any]) -> List[DialogueMedia]:
    media: List[DialogueMedia] = []
    caption_value = qa.get("image_caption")
    caption = caption_value.strip() if isinstance(caption_value, str) else ""
    url = qa.get("question_image")
    if isinstance(url, str) and url.strip():
        u = url.strip()
        inferred_modality, mime_type = infer_modality_and_mime(u)
        media.append(
            DialogueMedia(
                modality="image" if inferred_modality == "unknown" else inferred_modality,
                url=u,
                caption=caption,
                media_id="",
                mime_type=mime_type,
            )
        )
    return media


def build_caption_block(items: List[DialogueMedia], enabled_modalities: Dict[str, str]) -> str:
    lines: List[str] = []
    for item in items:
        if item.modality not in {"image", "audio", "video"}:
            continue
        if enabled_modalities.get(item.modality, "raw") != "caption":
            continue
        if not item.caption and not item.media_id:
            continue
        header = item.modality
        if item.media_id:
            header += f"_id: {item.media_id}"
        lines.append(f"{header}")
        if item.caption:
            lines.append(f"{item.modality}_caption: {item.caption}")
    return "\n".join(lines).strip()


def build_raw_parts(items: List[DialogueMedia], enabled_modalities: Dict[str, str]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for item in items:
        if item.modality not in {"image", "audio", "video"}:
            continue
        if enabled_modalities.get(item.modality, "raw") != "raw":
            continue
        if not _is_http_url(item.url):
            continue
        parts.append(build_filedata_part(item.url, item.mime_type))
    return parts


def build_contents_from_dataset(
    dataset: Dict[str, Any],
    input_form: Dict[str, str],
    sys_prompt: str,
    include_persona: bool = True,
) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []

    if sys_prompt or include_persona:
        persona = dataset.get("character_profile") or {}
        persona_summary = persona.get("persona_summary", "")
        conversation_style = persona.get("conversation_style", "")

        prompt_parts: List[str] = []
        if sys_prompt:
            prompt_parts.append(sys_prompt.strip())
        if include_persona:
            if persona_summary:
                prompt_parts.append(f"Persona summary:\n{persona_summary}".strip())
            if conversation_style:
                prompt_parts.append(f"Conversation style:\n{conversation_style}".strip())
        if prompt_parts:
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": "\n\n".join(prompt_parts).strip()}],
                }
            )

    sessions = dataset.get("multi_session_dialogues") or []
    for session in sessions:
        for dialog in session.get("dialogues") or []:
            if not isinstance(dialog, dict):
                continue
            user_text = dialog.get("user", "") or ""
            assistant_text = dialog.get("assistant", "") or ""

            media_items = extract_dialogue_media(dialog)
            raw_parts = build_raw_parts(media_items, input_form)
            caption_block = build_caption_block(media_items, input_form)

            merged_user_text = user_text.strip()
            if caption_block:
                merged_user_text = f"{caption_block}\n{merged_user_text}".strip()

            if raw_parts or merged_user_text:
                user_parts: List[Dict[str, Any]] = []
                user_parts.extend(raw_parts)  # media first
                if merged_user_text:
                    user_parts.append({"text": merged_user_text})
                contents.append({"role": "user", "parts": user_parts})

            if assistant_text.strip():
                contents.append({"role": "model", "parts": [{"text": assistant_text.strip()}]})

    return contents


def load_dataset(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_judge_prompt_template(path: Path) -> str:
    template = load_text_file(path)
    if not template:
        raise FileNotFoundError(f"Judge prompt template not found: {path}")
    return template


def format_judge_prompt(template: str, *, question: str, ground_truth: str, model_output: str) -> str:
    prompt = template
    prompt = prompt.replace("{{question}}", question)
    prompt = prompt.replace("{{ground_truth}}", ground_truth)
    prompt = prompt.replace("{{model_output}}", model_output)
    return prompt


def parse_judge_json(text: str) -> Dict[str, Any]:
    cleaned = clean_model_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        # Try to extract a JSON object substring
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def append_jsonl(path: Path, row: Dict[str, Any], lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_indices(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            idx = row.get("qa_index")
            if isinstance(idx, int):
                done.add(idx)
    return done


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (k - f) * (xs[c] - xs[f])


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    errors = [r for r in rows if r.get("status") != "ok"]

    ems: List[float] = []
    f1s: List[float] = []
    latencies: List[float] = []
    judge_scores: List[float] = []
    judge_latencies: List[float] = []

    # per-category accumulators
    cat_ems: Dict[str, List[float]] = {}
    cat_f1s: Dict[str, List[float]] = {}
    cat_judge_scores: Dict[str, List[float]] = {}
    cat_counts: Dict[str, int] = {}

    for r in ok:
        m = r.get("metrics") or {}
        em = float(m["EM"]) if isinstance(m.get("EM"), (int, float)) else None
        f1 = float(m["F1"]) if isinstance(m.get("F1"), (int, float)) else None
        if em is not None:
            ems.append(em)
        if f1 is not None:
            f1s.append(f1)
        if isinstance(r.get("latency_seconds"), (int, float)):
            latencies.append(float(r["latency_seconds"]))
        j = r.get("judge") or {}
        jr = j.get("result") if isinstance(j, dict) else None
        jscore = float(jr["score"]) if isinstance(jr, dict) and isinstance(jr.get("score"), (int, float)) else None
        if jscore is not None:
            judge_scores.append(jscore)
        if isinstance(j, dict) and isinstance(j.get("latency_seconds"), (int, float)):
            judge_latencies.append(float(j["latency_seconds"]))

        cat = str(r.get("category") or "").strip() or "unknown"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if em is not None:
            cat_ems.setdefault(cat, []).append(em)
        if f1 is not None:
            cat_f1s.setdefault(cat, []).append(f1)
        if jscore is not None:
            cat_judge_scores.setdefault(cat, []).append(jscore)

    all_cats = sorted(set(list(cat_counts.keys())))
    per_category: Dict[str, Any] = {}
    for cat in all_cats:
        per_category[cat] = {
            "count": cat_counts.get(cat, 0),
            "EM_mean": _mean(cat_ems.get(cat, [])),
            "F1_mean": _mean(cat_f1s.get(cat, [])),
            "judge_score_mean": _mean(cat_judge_scores.get(cat, [])),
            "judge_score_p50": _percentile(cat_judge_scores.get(cat, []), 50) if cat_judge_scores.get(cat) else None,
        }

    return {
        "counts": {"total": len(rows), "ok": len(ok), "error": len(errors)},
        "metrics": {
            "EM_mean": _mean(ems),
            "F1_mean": _mean(f1s),
            "judge_score_mean": _mean(judge_scores) if judge_scores else None,
            "judge_score_p50": _percentile(judge_scores, 50) if judge_scores else None,
            "judge_score_p95": _percentile(judge_scores, 95) if judge_scores else None,
        },
        "latency_seconds": {
            "answer_mean": _mean(latencies),
            "answer_p50": _percentile(latencies, 50),
            "answer_p95": _percentile(latencies, 95),
            "judge_mean": _mean(judge_latencies) if judge_latencies else None,
            "judge_p50": _percentile(judge_latencies, 50) if judge_latencies else None,
            "judge_p95": _percentile(judge_latencies, 95) if judge_latencies else None,
        },
        "per_category": per_category,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_sys_prompt_file = (script_dir.parents[2] / "benchmark" / "prompt" / "sys_prompt.txt").resolve()
    default_judge_prompt_file = (script_dir.parents[2] / "omnimembench" / "engine" / "evaluate" / "llm_judge.txt").resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, help="Path to a single collection JSON (default mode).")
    parser.add_argument("--collection_dir", default=None, help="Directory containing collection/*.json to run.")
    parser.add_argument("--all", action="store_true", help="If set with --collection_dir, run all JSON files in that dir.")

    parser.add_argument(
        "--input_form",
        default="image=raw,audio=raw,video=raw",
        help="Per-modality input form: raw|caption or 'image=raw,audio=caption,video=raw'.",
    )

    parser.add_argument("--api_key", default=None, help="LLM API key. If not set, uses env LLM_API_KEY.")
    parser.add_argument(
        "--api_key_file",
        default=None,
        help="Optional file containing LLM API key (first non-empty line). Used if --api_key and env are absent.",
    )
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="LLM compatible-mode base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="(Currently unused) Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="(Currently unused) Max tokens.")
    parser.add_argument("--timeout_seconds", type=int, default=120, help="Per-request timeout seconds.")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries per request.")

    parser.add_argument("--sys_prompt_file", default=str(default_sys_prompt_file), help="System prompt text file (optional).")
    parser.add_argument("--no_persona", action="store_true", help="Do not prepend persona_summary / conversation_style.")

    parser.add_argument("--qa_idx", type=int, default=None, help="Run only one QA index (0-based). Default runs all QAs.")
    parser.add_argument("--limit_qas", type=int, default=None, help="Limit number of QAs from the start.")
    parser.add_argument("--dry_run", action="store_true", help="Build request(s) and write payload preview without calling the API.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL results (skip completed qa_index).")
    parser.add_argument("--max_workers", type=int, default=10, help="Parallel workers (QA-level). Default 10.")
    parser.add_argument("--print_every", type=int, default=1, help="Print progress every N completed QAs. Default 1.")
    parser.add_argument(
        "--heartbeat_seconds",
        type=int,
        default=30,
        help="Print progress heartbeat every N seconds when no QA finishes. Set 0 to disable. Default 30.",
    )

    parser.add_argument(
        "--enable_judge",
        dest="enable_judge",
        action="store_true",
        default=True,
        help="Enable LLM-as-judge evaluation. Default enabled.",
    )
    parser.add_argument(
        "--disable_judge",
        dest="enable_judge",
        action="store_false",
        help="Disable LLM-as-judge.",
    )
    parser.add_argument("--judge_model", default=None, help="Judge model name (defaults to --model).")
    parser.add_argument("--judge_prompt_file", default=str(default_judge_prompt_file), help="Judge prompt template file.")

    parser.add_argument("--output_dir", default=str((script_dir / "output_longctx").resolve()), help="Output directory.")
    parser.add_argument("--run_tag", default=None, help="Optional tag appended to output filenames. Default derived from --input_form.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not api_key and args.api_key_file:
        try:
            for line in Path(args.api_key_file).read_text(encoding="utf-8").splitlines():
                candidate = line.strip()
                if candidate:
                    api_key = candidate
                    break
        except FileNotFoundError:
            raise SystemExit(f"--api_key_file not found: {args.api_key_file}")
        except Exception as e:
            raise SystemExit(f"Failed to read --api_key_file: {e}")
    if not args.dry_run and not api_key:
        raise SystemExit("Missing API key. Provide --api_key or set LLM_API_KEY.")

    input_form = parse_input_form(args.input_form)
    tag = (args.run_tag or input_form_tag(input_form)).strip()
    if not tag:
        tag = input_form_tag(input_form)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys_prompt = load_text_file(Path(args.sys_prompt_file)) if args.sys_prompt_file else ""

    judge_template = ""
    if args.enable_judge:
        judge_template = load_judge_prompt_template(Path(args.judge_prompt_file))

    data_files: List[Path] = []
    if args.data_path:
        data_files = [Path(args.data_path).resolve()]
    elif args.collection_dir:
        d = Path(args.collection_dir).resolve()
        if not d.exists():
            raise SystemExit(f"--collection_dir does not exist: {d}")
        if args.all:
            data_files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
        else:
            # default: run the first JSON file
            data_files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".json"])[:1]
    else:
        raise SystemExit("Provide --data_path or --collection_dir.")

    for data_path in data_files:
        dataset = load_dataset(data_path)
        base_contents = build_contents_from_dataset(
            dataset,
            input_form=input_form,
            sys_prompt=sys_prompt,
            include_persona=not args.no_persona,
        )

        qas: List[Dict[str, Any]] = dataset.get("QAs") or dataset.get("human-annotated QAs") or []
        if not isinstance(qas, list):
            raise SystemExit(f"Invalid QAs format in {data_path.name}")

        if args.qa_idx is not None:
            if args.qa_idx < 0 or args.qa_idx >= len(qas):
                raise SystemExit(f"--qa_idx out of range: {args.qa_idx} (QAs={len(qas)})")
            qas = [qas[args.qa_idx]]
            qa_offset = args.qa_idx
        else:
            qa_offset = 0

        if args.limit_qas is not None:
            qas = qas[: max(0, int(args.limit_qas))]

        run_id = data_path.stem
        jsonl_path = output_dir / f"{run_id}_{tag}_longctx_results.jsonl"
        meta_path = output_dir / f"{run_id}_{tag}_longctx_meta.json"
        out_path = output_dir / f"{run_id}_{tag}_longctx_results.json"

        done_indices: set[int] = load_done_indices(jsonl_path) if args.resume and not args.dry_run else set()
        if done_indices:
            print(f"Resume enabled: found {len(done_indices)} completed QAs in {jsonl_path}")

        started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        meta = {
            "run_id": run_id,
            "tag": tag,
            "started_at": started_at,
            "data_path": str(data_path),
            "model": args.model,
            "base_url": args.base_url,
            "input_form": input_form,
            "enable_judge": bool(args.enable_judge),
            "judge_model": (args.judge_model or args.model) if args.enable_judge else None,
            "timeout_seconds": int(args.timeout_seconds),
            "max_retries": int(args.max_retries),
            "max_workers": int(args.max_workers),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Meta: {meta_path}")
        print(f"Streaming results to: {jsonl_path}")

        write_lock = Lock()

        def process_one(qa_index: int, qa: Dict[str, Any]) -> Dict[str, Any]:
            question = str(qa.get("question", "") or "")
            short_gt = str(qa.get("answer", "") or "")
            full_gt = str(qa.get("full_answer", "") or "")
            category = str(qa.get("type", qa.get("point", "")) or "")

            qa_media = extract_qa_media(qa)
            qa_raw_parts = build_raw_parts(qa_media, input_form)
            qa_caption_block = build_caption_block(qa_media, input_form)
            merged_question = question.strip()
            if qa_caption_block:
                merged_question = f"{qa_caption_block}\n{merged_question}".strip()
            qa_parts: List[Dict[str, Any]] = []
            qa_parts.extend(qa_raw_parts)
            if merged_question:
                qa_parts.append({"text": merged_question})

            contents = base_contents + [{"role": "user", "parts": qa_parts}]

            t0 = time.perf_counter()
            pred, raw_resp = llm_api_chat_native(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                contents=contents,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            latency = time.perf_counter() - t0
            usage = raw_resp.get("usage") if isinstance(raw_resp, dict) else None

            item: Dict[str, Any] = {
                "run_id": run_id,
                "data_path": str(data_path),
                "qa_index": qa_index,
                "category": category,
                "question": question,
                "system_answer": pred,
                "original_answer": short_gt,
                "judge_answer": full_gt or short_gt,
                "metrics": {"EM": exact_match(pred, short_gt), "F1": token_f1(pred, short_gt)},
                "latency_seconds": latency,
                "usage": usage,
                "model": args.model,
                "input_form": input_form,
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "status": "ok",
            }

            if args.enable_judge:
                judge_prompt = format_judge_prompt(
                    judge_template,
                    question=question,
                    ground_truth=full_gt or short_gt,
                    model_output=pred,
                )
                judge_contents = [{"role": "user", "parts": [{"text": judge_prompt}]}]
                jt0 = time.perf_counter()
                judge_text, judge_raw = llm_api_chat_native(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.judge_model or args.model,
                    contents=judge_contents,
                    temperature=0.0,
                    max_tokens=512,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                )
                jlat = time.perf_counter() - jt0
                try:
                    judge_json = parse_judge_json(judge_text)
                except Exception as e:
                    judge_json = {"error": str(e), "raw_text": judge_text}
                item["judge"] = {
                    "model": args.judge_model or args.model,
                    "latency_seconds": jlat,
                    "usage": judge_raw.get("usage") if isinstance(judge_raw, dict) else None,
                    "result": judge_json,
                }

            return item

        # Dry-run: write one preview request and move on.
        if args.dry_run:
            for idx, qa in enumerate(qas):
                qa_index = qa_offset + idx
                question = str(qa.get("question", "") or "")
                contents = base_contents + [{"role": "user", "parts": [{"text": question}]}]
                payload_preview = {
                    "model": args.model,
                    "contents": contents,
                }
                out_preview = output_dir / f"{run_id}_qa{qa_index}_dry_request.json"
                with open(out_preview, "w", encoding="utf-8") as f:
                    json.dump(payload_preview, f, ensure_ascii=False, indent=2)
                print(f"Wrote dry-run request: {out_preview}")
                break
            continue

        jobs: List[Tuple[int, Dict[str, Any]]] = []
        for idx, qa in enumerate(qas):
            qa_index = qa_offset + idx
            if qa_index in done_indices:
                continue
            if not isinstance(qa, dict):
                continue
            jobs.append((qa_index, qa))

        if not jobs:
            print("No pending QAs to run.")
        else:
            print(f"Running {len(jobs)} QAs with max_workers={args.max_workers} ...")

        completed_count = 0
        ok_count = 0
        err_count = 0
        t_start = time.perf_counter()
        last_heartbeat = t_start
        with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
            future_map = {ex.submit(process_one, qa_index, qa): qa_index for qa_index, qa in jobs}
            pending = set(future_map.keys())

            def emit_heartbeat(now: float) -> None:
                elapsed = now - t_start
                rate = completed_count / elapsed if elapsed > 0 else 0.0
                eta = (len(jobs) - completed_count) / rate if rate > 0 else None
                msg = (
                    f"[{completed_count}/{len(jobs)}] "
                    f"pending={len(pending)} ok={ok_count} err={err_count} "
                    f"elapsed={elapsed:.0f}s"
                )
                if eta is not None:
                    msg += f" eta={eta:.0f}s"
                print(msg)

            heartbeat = max(0, int(args.heartbeat_seconds))
            while pending:
                done, pending = wait(
                    pending,
                    timeout=heartbeat if heartbeat > 0 else None,
                    return_when=FIRST_COMPLETED,
                )
                now = time.perf_counter()
                if not done:
                    if heartbeat > 0 and now - last_heartbeat >= heartbeat:
                        emit_heartbeat(now)
                        last_heartbeat = now
                    continue

                for fut in done:
                    qa_index = future_map[fut]
                    try:
                        item = fut.result()
                    except Exception as e:
                        item = {
                            "run_id": run_id,
                            "data_path": str(data_path),
                            "qa_index": qa_index,
                            "status": "error",
                            "error": str(e),
                            "traceback": traceback.format_exc(limit=20),
                            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        }
                    append_jsonl(jsonl_path, item, write_lock)
                    completed_count += 1
                    if item.get("status") == "ok":
                        ok_count += 1
                    else:
                        err_count += 1

                    if args.print_every > 0 and (completed_count % max(1, int(args.print_every)) == 0):
                        msg = f"[{completed_count}/{len(jobs)}] QA {qa_index} saved"
                        if item.get("status") == "ok":
                            em = item.get("metrics", {}).get("EM")
                            f1 = item.get("metrics", {}).get("F1")
                            lat = item.get("latency_seconds")
                            msg += f" latency={lat:.2f}s EM={em} F1={f1:.3f}"
                            if args.enable_judge:
                                score = (item.get("judge", {}).get("result", {}) or {}).get("score")
                                jlat = item.get("judge", {}).get("latency_seconds")
                                if score is not None:
                                    msg += f" judge={score}"
                                if isinstance(jlat, (int, float)):
                                    msg += f" judge_latency={jlat:.2f}s"
                        else:
                            msg += f" ERROR={item.get('error')}"
                        print(msg)

                if heartbeat > 0 and now - last_heartbeat >= heartbeat:
                    emit_heartbeat(now)
                    last_heartbeat = now

        meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        rows = load_jsonl_rows(jsonl_path)
        meta["qa_count"] = len(rows)
        meta["summary"] = summarize_results(rows)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": rows}, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
