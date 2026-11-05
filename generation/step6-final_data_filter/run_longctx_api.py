#!/usr/bin/env python3
"""
Long-context API evaluation for OmniMemBench.
Supports nugget-based evaluation for SUM questions.
Auto-detects Gemini vs OpenAI format based on model name.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, time, traceback
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

DEFAULT_BASE_URL = "https://localhost:8000/v1"
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "your-model-name")
MM_INLINE_TAG_RE = re.compile(r"<\s*(image|audio|video)\s*>", flags=re.IGNORECASE)
SUPPORTED_MM_TYPES = {"image", "audio", "video"}

# ── Mime / modality ──────────────────────────────────────────────────────────

_SUFFIX_MAP = {
    ".jpg": ("image", "image/jpeg"), ".jpeg": ("image", "image/jpeg"),
    ".png": ("image", "image/png"), ".webp": ("image", "image/webp"), ".gif": ("image", "image/gif"),
    ".mp3": ("audio", "audio/mp3"), ".wav": ("audio", "audio/wav"),
    ".m4a": ("audio", "audio/mp4"), ".ogg": ("audio", "audio/ogg"),
    ".mp4": ("video", "video/mp4"), ".webm": ("video", "video/webm"),
    ".avi": ("video", "video/x-msvideo"), ".mkv": ("video", "video/x-matroska"),
}
_DEFAULT_MIME = {"image": "image/jpeg", "audio": "audio/wav", "video": "video/mp4"}


def _is_http_url(v: str) -> bool:
    return isinstance(v, str) and urlparse(v).scheme.lower() in {"http", "https"}


def _is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()


def infer_modality_and_mime(url: str) -> Tuple[str, str]:
    suffix = Path(urlparse(url).path).suffix.lower()
    return _SUFFIX_MAP.get(suffix, ("unknown", "application/octet-stream"))


# ── Input form parsing ───────────────────────────────────────────────────────

def parse_input_form(spec: str) -> Dict[str, str]:
    spec = (spec or "").strip()
    if not spec or spec in {"raw", "caption", "none"}:
        v = spec or "raw"
        return {"image": v, "audio": v, "video": v}
    result = {"image": "raw", "audio": "raw", "video": "raw"}
    for item in spec.split(","):
        k, _, v = item.strip().partition("=")
        k, v = k.strip().lower(), v.strip().lower()
        if k not in result or v not in {"raw", "caption", "none"}:
            raise ValueError(f"Invalid --input_form: {item!r}")
        result[k] = v
    return result


def input_form_tag(form: Dict[str, str]) -> str:
    shorts = {"image": "img", "audio": "aud", "video": "vid"}
    return "_".join(f"{shorts[m]}{form.get(m,'raw')[0]}" for m in ["image", "audio", "video"])


# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_model_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ── Gemini parts → OpenAI content conversion ─────────────────────────────────

_GEMINI_ROLE_TO_OPENAI = {"USER": "user", "MODEL": "assistant", "user": "user", "assistant": "assistant"}

_MODALITY_FROM_MIME = {
    "image": "image_url", "audio": "input_audio", "video": "video_url",
}


def _gemini_part_to_openai(part: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single Gemini part to OpenAI content block."""
    if "text" in part:
        return {"type": "text", "text": part["text"]}
    if "fileData" in part:
        fd = part["fileData"]
        mime = fd.get("mimeType", "")
        url = fd.get("fileUri", "")
        modality = mime.split("/")[0] if "/" in mime else "unknown"
        if modality == "image":
            return {"type": "image_url", "image_url": {"url": url}}
        elif modality == "audio":
            return {"type": "input_audio", "input_audio": {"data": url, "format": mime.split("/")[-1]}}
        elif modality == "video":
            # OpenAI doesn't have a standard video type; use image_url as fallback
            # or a custom type depending on the provider
            return {"type": "video_url", "video_url": {"url": url}}
        else:
            return {"type": "text", "text": f"[unsupported media: {mime}]"}
    return {"type": "text", "text": "[unknown part]"}


def contents_to_openai_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini contents list to OpenAI messages list."""
    messages = []
    for item in contents:
        role = _GEMINI_ROLE_TO_OPENAI.get(item.get("role", ""), "user")
        parts = item.get("parts", [])
        # If all parts are text-only, merge into a single string
        if all("text" in p and len(p) == 1 for p in parts):
            content = "\n".join(p["text"] for p in parts)
        else:
            content = [_gemini_part_to_openai(p) for p in parts]
        messages.append({"role": role, "content": content})
    return messages


# ── LLM API (native Gemini protocol via curl) ──────────────────────────

def llm_api_chat_native(
    *, api_key: str, base_url: str, model: str, contents: List[Dict[str, Any]],
    temperature: Optional[float] = None, max_tokens: Optional[int] = None,
    timeout_seconds: int = 120, max_retries: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    is_gemini = _is_gemini_model(model)
    last_error = None
    for attempt in range(max_retries):
        try:
            if is_gemini:
                payload = {"model": model, "contents": contents}
            else:
                payload = {
                    "model": model,
                    "messages": contents_to_openai_messages(contents),
                }
                if temperature is not None:
                    payload["temperature"] = temperature
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens

            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            url = base_url.rstrip("/") + "/chat/completions"

            result = subprocess.run(
                ["curl", "-s", "-X", "POST", url,
                 "-H", "Content-Type: application/json",
                 "-H", f"Authorization: {api_key}",
                 "--data-binary", "@-"],
                input=body, capture_output=True,
                timeout=timeout_seconds + 10,
            )

            if result.returncode != 0:
                raise RuntimeError(f"curl failed: {result.stderr.decode()}")

            resp = json.loads(result.stdout.decode("utf-8"))

            if "candidates" in resp:
                parts = resp["candidates"][0].get("content", {}).get("parts", [])
                if not parts or "text" not in parts[0]:
                    raise RuntimeError(f"No text in response: {json.dumps(resp)[:500]}")
                return clean_model_text(parts[0]["text"]), resp
            elif "choices" in resp:
                content = resp["choices"][0]["message"]["content"]
                if not content:
                    raise RuntimeError(f"Empty content in response: {json.dumps(resp)[:500]}")
                return clean_model_text(content), resp
            else:
                full_body = result.stdout.decode("utf-8", errors="replace")
                try:
                    err_obj = json.loads(full_body)
                    err_msg = json.dumps(err_obj, ensure_ascii=False, indent=2)
                except Exception:
                    err_msg = full_body
                raise RuntimeError(f"Unexpected response format:\n{err_msg}")

        except subprocess.TimeoutExpired:
            last_error = "timeout"
        except Exception as e:
            last_error = str(e)[:2000]

        if attempt < max_retries - 1:
            time.sleep(min(2 * (2 ** attempt), 60))

    raise RuntimeError(f"API failed after {max_retries} attempts: {last_error}")


# ── Metrics ──────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = re.sub(r"[\t\r\n]+", " ", str(s or "").lower())
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred: str, gt: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0


def token_f1(pred: str, gt: str) -> float:
    p_tok, g_tok = normalize_answer(pred).split(), normalize_answer(gt).split()
    if not p_tok and not g_tok: return 1.0
    if not p_tok or not g_tok: return 0.0
    common = {}
    for t in p_tok: common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g_tok:
        if common.get(t, 0) > 0: common[t] -= 1; overlap += 1
    prec = overlap / len(p_tok)
    rec = overlap / len(g_tok)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


# ── Dialogue media extraction ────────────────────────────────────────────────

@dataclass
class DialogueMedia:
    modality: str; url: str; caption: str; media_id: str; mime_type: str


def extract_dialogue_media(dialog: Dict[str, Any]) -> List[DialogueMedia]:
    mm_value = dialog.get("mm")
    if mm_value is None:
        return []
    if not isinstance(mm_value, list):
        raise ValueError("`mm` must be a list")
    media = []
    for idx, item in enumerate(mm_value):
        modality = item.get("type", "").strip().lower()
        if modality not in SUPPORTED_MM_TYPES:
            raise ValueError(f"Unsupported mm[{idx}].type: {item.get('type')!r}")
        url = (item.get("url") or "").strip()
        caption = (item.get("caption") or item.get("description") or "").strip()
        media_id = ""
        for k in ("mm_id", "media_id", "evidence_id"):
            v = item.get(k)
            if isinstance(v, str) and v.strip(): media_id = v.strip(); break
        _, inferred_mime = infer_modality_and_mime(url) if url else ("unknown", "application/octet-stream")
        mime = inferred_mime if inferred_mime != "application/octet-stream" else _DEFAULT_MIME[modality]
        media.append(DialogueMedia(modality=modality, url=url, caption=caption, media_id=media_id, mime_type=mime))
    return media


def extract_qa_media(qa: Dict[str, Any]) -> List[DialogueMedia]:
    url = qa.get("question_image")
    if not (isinstance(url, str) and url.strip()):
        return []
    u = url.strip()
    mod, mime = infer_modality_and_mime(u)
    caption = (qa.get("image_caption") or "").strip()
    return [DialogueMedia(modality="image" if mod == "unknown" else mod, url=u, caption=caption, media_id="", mime_type=mime)]


# ── Content building ─────────────────────────────────────────────────────────

def _filedata_part(url: str, mime: str) -> Dict[str, Any]:
    return {"fileData": {"mimeType": mime, "fileUri": url}}


def _caption_for_item(item: DialogueMedia) -> str:
    parts = []
    header = item.modality + (f"_id: {item.media_id}" if item.media_id else "")
    parts.append(header)
    if item.caption: parts.append(f"{item.modality}_caption: {item.caption}")
    return "\n".join(parts) if (item.caption or item.media_id) else ""


def build_user_parts(*, user_text: str, media_items: List[DialogueMedia],
                     enabled: Dict[str, str], ref: str) -> List[Dict[str, Any]]:
    text = user_text or ""
    matches = list(MM_INLINE_TAG_RE.finditer(text))
    tags = [m.group(1).lower() for m in matches]

    if media_items and not tags:
        raise ValueError(f"{ref}: mm items but no inline tags in user text")
    if tags and len(tags) != len(media_items):
        raise ValueError(f"{ref}: {len(tags)} tags vs {len(media_items)} mm items")

    if not matches:
        return [{"text": text.strip()}] if text.strip() else []

    parts, cursor = [], 0
    for idx, m in enumerate(matches):
        prefix = text[cursor:m.start()]
        if prefix: parts.append({"text": prefix})
        item = media_items[idx]
        mode = enabled.get(item.modality, "raw")
        if mode == "raw":
            if not _is_http_url(item.url):
                raise ValueError(f"{ref}: mm[{idx}] needs http url for raw mode")
            parts.append(_filedata_part(item.url, item.mime_type))
        elif mode == "caption":
            cb = _caption_for_item(item)
            if not cb: raise ValueError(f"{ref}: mm[{idx}] no caption for caption mode")
            parts.append({"text": cb})
        # mode == "none": skip
        cursor = m.end()
    suffix = text[cursor:]
    if suffix: parts.append({"text": suffix})
    return parts


def build_raw_parts(items: List[DialogueMedia], enabled: Dict[str, str]) -> List[Dict[str, Any]]:
    return [_filedata_part(i.url, i.mime_type) for i in items
            if i.modality in SUPPORTED_MM_TYPES and enabled.get(i.modality) == "raw" and _is_http_url(i.url)]


def build_caption_block(items: List[DialogueMedia], enabled: Dict[str, str]) -> str:
    return "\n".join(_caption_for_item(i) for i in items
                     if i.modality in SUPPORTED_MM_TYPES and enabled.get(i.modality) == "caption"
                     and _caption_for_item(i)).strip()


def build_contents(dataset: Dict[str, Any], input_form: Dict[str, str],
                   sys_prompt: str, include_persona: bool = True) -> List[Dict[str, Any]]:
    contents = []
    # system/persona preamble
    preamble = []
    if sys_prompt: preamble.append(sys_prompt.strip())
    if include_persona:
        persona = dataset.get("character_profile") or {}
        for k in ("persona_summary", "conversation_style"):
            v = persona.get(k, "")
            if v: preamble.append(f"{k.replace('_',' ').title()}:\n{v}".strip())
    if preamble:
        contents.append({"role": "USER", "parts": [{"text": "\n\n".join(preamble)}]})

    # dialogue sessions
    for s_idx, session in enumerate(dataset.get("multi_session_dialogues") or []):
        sid = str(session.get("session_id", f"session#{s_idx+1}"))
        date = str(session.get("date", "")).strip()
        if date:
            contents.append({"role": "USER", "parts": [{"text": f"[Session {sid} — {date}]"}]})
        for d_idx, dialog in enumerate(session.get("dialogues") or []):
            ref = f"{sid}.dialogues[{d_idx}]"
            user_text = dialog.get("user", "") or ""
            asst_text = dialog.get("assistant", "") or ""
            media = extract_dialogue_media(dialog)
            user_parts = build_user_parts(user_text=user_text, media_items=media, enabled=input_form, ref=ref)
            if user_parts: contents.append({"role": "USER", "parts": user_parts})
            if asst_text.strip(): contents.append({"role": "MODEL", "parts": [{"text": asst_text.strip()}]})
    return contents


# ── Judge: standard ──────────────────────────────────────────────────────────

def format_judge_prompt(template: str, *, question: str, ground_truth: str, model_output: str) -> str:
    return template.replace("{{question}}", question).replace("{{ground_truth}}", ground_truth).replace("{{model_output}}", model_output)


def parse_judge_json(text: str) -> Dict[str, Any]:
    cleaned = clean_model_text(text)
    try: return json.loads(cleaned)
    except Exception: pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m: return json.loads(m.group(0))
    raise ValueError(f"Cannot parse judge JSON: {cleaned[:200]}")


# ── Judge: SUM nugget-based ──────────────────────────────────────────────────

def format_nugget_judge_prompt(template: str, *, question: str, model_output: str,
                               nuggets: List[Dict[str, Any]]) -> str:
    lines = []
    for n in nuggets:
        nid = n.get("nugget_id", "?")
        content = n.get("content", "")
        lines.append(f"  {nid}: {content}")
    nuggets_list = "\n".join(lines)
    return (template
            .replace("{{question}}", question)
            .replace("{{model_output}}", model_output)
            .replace("{{nuggets_list}}", nuggets_list))


def compute_nugget_metrics(coverage: Dict[str, int], nuggets: List[Dict[str, Any]]) -> Dict[str, Any]:
    total, covered = 0, 0
    mm_total, mm_covered = 0, 0
    for n in nuggets:
        nid = n.get("nugget_id", "?")
        v = coverage.get(nid, 0)
        total += 1; covered += v
        if n.get("source_type") == "multimodal":
            mm_total += 1; mm_covered += v
    result = {"nugget_recall": covered / total if total else 0.0, "nugget_total": total, "nugget_covered": covered}
    if mm_total > 0:
        result["mm_nugget_recall"] = mm_covered / mm_total
    return result


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def load_text(p: Path) -> str:
    try: return p.read_text(encoding="utf-8").strip()
    except FileNotFoundError: return ""

def append_jsonl(path: Path, row: Dict, lock: Lock):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_done_indices(path: Path) -> set:
    done = set()
    if not path.exists(): return done
    for line in open(path, encoding="utf-8"):
        try:
            row = json.loads(line.strip())
            if row.get("status") == "ok" and isinstance(row.get("qa_index"), int):
                done.add(row["qa_index"])
        except Exception: pass
    return done

def load_jsonl_rows(path: Path) -> List[Dict]:
    if not path.exists(): return []
    rows = []
    for line in open(path, encoding="utf-8"):
        try: rows.append(json.loads(line.strip()))
        except Exception: pass
    return rows


# ── Result summarization ─────────────────────────────────────────────────────

def _mean(vals): return sum(vals) / len(vals) if vals else None

def _pct(vals, p):
    if not vals: return None
    xs = sorted(vals); k = (len(xs)-1) * p / 100.0
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (k-f)*(xs[c]-xs[f]) if f != c else xs[f]


def summarize_results(rows: List[Dict]) -> Dict:
    def summarize(subset):
        ok = [r for r in subset if r.get("status") == "ok"]
        ems = [r["metrics"]["EM"] for r in ok if isinstance(r.get("metrics", {}).get("EM"), (int, float))]
        f1s = [r["metrics"]["F1"] for r in ok if isinstance(r.get("metrics", {}).get("F1"), (int, float))]
        lats = [r["latency_seconds"] for r in ok if isinstance(r.get("latency_seconds"), (int, float))]

        judge_scores = []
        for r in ok:
            j = (r.get("judge") or {}).get("result") or {}
            if isinstance(j.get("score"), (int, float)):
                judge_scores.append(j["score"])

        nugget_recalls = []
        for r in ok:
            j = (r.get("judge") or {}).get("result") or {}
            if isinstance(j.get("nugget_recall"), (int, float)):
                nugget_recalls.append(j["nugget_recall"])

        metrics = {"EM_mean": _mean(ems), "F1_mean": _mean(f1s)}
        if judge_scores: metrics["judge_score_mean"] = _mean(judge_scores)
        if nugget_recalls: metrics["nugget_recall_mean"] = _mean(nugget_recalls)

        return {
            "counts": {"total": len(subset), "ok": len(ok), "error": len(subset) - len(ok)},
            "metrics": metrics,
            "latency": {"mean": _mean(lats), "p50": _pct(lats, 50), "p95": _pct(lats, 95)},
        }

    summary = summarize(rows)
    grouped = {}
    for r in rows:
        t = str(r.get("category") or "__uncategorized__").strip() or "__uncategorized__"
        grouped.setdefault(t, []).append(r)
    summary["by_type"] = {t: summarize(g) for t, g in sorted(grouped.items())}
    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    default_sys_prompt = (script_dir.parents[1] / "benchmark" / "prompt" / "sys_prompt.txt").resolve()
    default_judge_prompt = (script_dir.parents[1] / "omnimembench" / "engine" / "evaluate" / "llm_judge.txt").resolve()
    default_nugget_prompt = (script_dir.parents[1] / "omnimembench" / "engine" / "evaluate" / "llm_judge_nugget.txt").resolve()

    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data_path", default=None)
    p.add_argument("--collection_dir", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--input_form", default="image=raw,audio=raw,video=raw")
    # api
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_key_file", default=None)
    p.add_argument("--base_url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--timeout_seconds", type=int, default=120)
    p.add_argument("--max_retries", type=int, default=5)
    # prompt
    p.add_argument("--sys_prompt_file", default=str(default_sys_prompt))
    p.add_argument("--no_persona", action="store_true")
    # judge
    p.add_argument("--enable_judge", dest="enable_judge", action="store_true", default=True)
    p.add_argument("--disable_judge", dest="enable_judge", action="store_false")
    p.add_argument("--judge_model", default=None)
    p.add_argument("--judge_prompt_file", default=str(default_judge_prompt))
    p.add_argument("--nugget_judge_prompt_file", default=str(default_nugget_prompt))
    # execution
    p.add_argument("--qa_idx", type=int, default=None)
    p.add_argument("--limit_qas", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_workers", type=int, default=10)
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--heartbeat_seconds", type=int, default=30)
    # output
    p.add_argument("--output_dir", default=str((script_dir / "output_longctx").resolve()))
    p.add_argument("--run_tag", default=None)
    args = p.parse_args()

    # resolve API key
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not api_key and args.api_key_file:
        for line in Path(args.api_key_file).read_text(encoding="utf-8").splitlines():
            if line.strip(): api_key = line.strip(); break
    if not args.dry_run and not api_key:
        raise SystemExit("Missing API key")

    input_form = parse_input_form(args.input_form)
    tag = (args.run_tag or input_form_tag(input_form)).strip() or input_form_tag(input_form)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys_prompt = load_text(Path(args.sys_prompt_file)) if args.sys_prompt_file else ""
    judge_template = load_text(Path(args.judge_prompt_file)) if args.enable_judge else ""
    nugget_judge_template = load_text(Path(args.nugget_judge_prompt_file)) if args.enable_judge else ""

    # log format detection
    fmt = "Gemini" if _is_gemini_model(args.model) else "OpenAI"
    print(f"Model: {args.model} → {fmt} format")

    # collect data files
    if args.data_path:
        data_files = [Path(args.data_path).resolve()]
    elif args.collection_dir:
        d = Path(args.collection_dir).resolve()
        jsons = sorted(f for f in d.iterdir() if f.suffix.lower() == ".json")
        data_files = jsons if args.all else jsons[:1]
    else:
        raise SystemExit("Provide --data_path or --collection_dir")

    for data_path in data_files:
        dataset = load_json(data_path)
        base_contents = build_contents(dataset, input_form, sys_prompt, not args.no_persona)

        qas = dataset.get("QAs") or dataset.get("human-annotated QAs") or []
        qa_offset = 0
        if args.qa_idx is not None:
            if not (0 <= args.qa_idx < len(qas)):
                raise SystemExit(f"--qa_idx out of range: {args.qa_idx}")
            qas, qa_offset = [qas[args.qa_idx]], args.qa_idx
        if args.limit_qas is not None:
            qas = qas[:max(0, args.limit_qas)]

        run_id = data_path.stem
        jsonl_path = output_dir / f"{run_id}_{tag}_longctx_results.jsonl"
        summary_path = output_dir / f"{run_id}_{tag}_longctx_summary.json"
        meta_path = output_dir / f"{run_id}_{tag}_longctx_meta.json"

        done_indices = load_done_indices(jsonl_path) if args.resume and not args.dry_run else set()
        if done_indices:
            print(f"Resume: {len(done_indices)} completed QAs found")

        meta = {
            "run_id": run_id, "tag": tag, "data_path": str(data_path),
            "model": args.model, "api_format": fmt, "input_form": input_form,
            "enable_judge": args.enable_judge,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        write_lock = Lock()

        def process_one(qa_index: int, qa: Dict) -> Dict:
            question = str(qa.get("question", "") or "")
            short_gt = str(qa.get("answer", "") or "")
            full_gt = str(qa.get("full_answer", "") or "")
            category = str(qa.get("type", qa.get("point", "")) or "")
            nuggets = qa.get("nuggets") or []
            is_sum = category.upper() == "SUM" and nuggets

            # build question content
            qa_media = extract_qa_media(qa)
            qa_parts = build_raw_parts(qa_media, input_form)
            merged_q = question.strip()
            caption_blk = build_caption_block(qa_media, input_form)
            if caption_blk: merged_q = f"{caption_blk}\n{merged_q}"
            if merged_q: qa_parts.append({"text": merged_q})

            contents = base_contents + [{"role": "user", "parts": qa_parts}]

            # get model answer
            t0 = time.perf_counter()
            pred, raw_resp = llm_api_chat_native(
                api_key=api_key, base_url=args.base_url, model=args.model,
                contents=contents, temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds, max_retries=args.max_retries,
            )
            latency = time.perf_counter() - t0

            item = {
                "run_id": run_id, "qa_index": qa_index, "category": category,
                "question": question, "system_answer": pred,
                "original_answer": short_gt, "judge_answer": full_gt or short_gt,
                "metrics": {"EM": exact_match(pred, short_gt), "F1": token_f1(pred, short_gt)},
                "latency_seconds": latency,
                "usage": (raw_resp.get("usage") or raw_resp.get("usageMetadata")) if isinstance(raw_resp, dict) else None,
                "model": args.model, "input_form": input_form,
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "ok",
            }

            # judge
            if args.enable_judge:
                jmodel = args.judge_model or args.model

                if is_sum and nugget_judge_template:
                    jp = format_nugget_judge_prompt(
                        nugget_judge_template, question=question, model_output=pred, nuggets=nuggets)
                else:
                    jp = format_judge_prompt(
                        judge_template, question=question,
                        ground_truth=full_gt or short_gt, model_output=pred)

                jt0 = time.perf_counter()
                judge_text, judge_raw = llm_api_chat_native(
                    api_key=api_key, base_url=args.base_url, model=jmodel,
                    contents=[{"role": "USER", "parts": [{"text": jp}]}],
                    timeout_seconds=args.timeout_seconds, max_retries=args.max_retries,
                )
                jlat = time.perf_counter() - jt0

                try:
                    judge_json = parse_judge_json(judge_text)
                except Exception as e:
                    judge_json = {"error": str(e), "raw_text": judge_text}

                if is_sum:
                    coverage = {k: int(v) for k, v in judge_json.items()
                                if isinstance(v, (int, float)) and not k.startswith("error")}
                    nugget_metrics = compute_nugget_metrics(coverage, nuggets)
                    judge_result = {"nugget_coverage": coverage, **nugget_metrics}
                else:
                    judge_result = judge_json

                item["judge"] = {
                    "model": jmodel, "latency_seconds": jlat,
                    "usage": judge_raw.get("usage") if isinstance(judge_raw, dict) else None,
                    "result": judge_result,
                }

            return item

        # dry run
        if args.dry_run:
            for idx, qa in enumerate(qas):
                qi = qa_offset + idx
                contents = base_contents + [{"role": "user", "parts": [{"text": qa.get("question", "")}]}]
                # dry run output shows the actual format that would be sent
                if _is_gemini_model(args.model):
                    dry_payload = {"model": args.model, "contents": contents}
                else:
                    dry_payload = {"model": args.model, "messages": contents_to_openai_messages(contents)}
                out = output_dir / f"{run_id}_qa{qi}_dry_request.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(dry_payload, f, ensure_ascii=False, indent=2)
                print(f"Dry-run ({fmt}): {out}")
                break
            continue

        # build job list
        jobs = [(qa_offset + i, qa) for i, qa in enumerate(qas)
                if (qa_offset + i) not in done_indices and isinstance(qa, dict)]
        if not jobs:
            print("No pending QAs."); continue
        print(f"Running {len(jobs)} QAs, workers={args.max_workers}")

        completed, ok_cnt, err_cnt = 0, 0, 0
        t_start = time.perf_counter()
        last_hb = t_start
        hb = max(0, args.heartbeat_seconds)

        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            fut_map = {ex.submit(process_one, qi, qa): qi for qi, qa in jobs}
            pending = set(fut_map)

            while pending:
                done, pending = wait(pending, timeout=hb or None, return_when=FIRST_COMPLETED)
                now = time.perf_counter()

                if not done:
                    if hb and now - last_hb >= hb:
                        elapsed = now - t_start
                        print(f"[{completed}/{len(jobs)}] pending={len(pending)} ok={ok_cnt} err={err_cnt} {elapsed:.0f}s")
                        last_hb = now
                    continue

                for fut in done:
                    qi = fut_map[fut]
                    try:
                        item = fut.result()
                    except Exception as e:
                        item = {"run_id": run_id, "qa_index": qi, "status": "error",
                                "error": str(e), "traceback": traceback.format_exc(limit=10),
                                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")}

                    append_jsonl(jsonl_path, item, write_lock)
                    completed += 1
                    if item.get("status") == "ok": ok_cnt += 1
                    else: err_cnt += 1

                    if args.print_every and completed % args.print_every == 0:
                        msg = f"[{completed}/{len(jobs)}] QA {qi}"
                        if item.get("status") == "ok":
                            m = item.get("metrics", {})
                            msg += f" EM={m.get('EM')} F1={m.get('F1',0):.3f} lat={item.get('latency_seconds',0):.1f}s"
                            jr = (item.get("judge") or {}).get("result") or {}
                            if "nugget_recall" in jr:
                                msg += f" nugget_recall={jr['nugget_recall']:.3f}"
                            elif "score" in jr:
                                msg += f" judge={jr['score']}"
                        else:
                            msg += f" ERR={item.get('error','')}"
                        print(msg)

                if hb and now - last_hb >= hb:
                    print(f"[{completed}/{len(jobs)}] pending={len(pending)} ok={ok_cnt} err={err_cnt}")
                    last_hb = now

        # write summary
        rows = load_jsonl_rows(jsonl_path)
        meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["qa_count"] = len(rows)
        meta["summary"] = summarize_results(rows)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "summary": meta["summary"]}, f, ensure_ascii=False, indent=2)
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()