#!/usr/bin/env python3
"""
OmniMemBench evaluation utilities.
Shared by run_longctx.py (and future eval scripts).
"""
from __future__ import annotations

import base64, json, re, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from omnimembench.engine.evaluate.scoring import qa_score, compute_nugget_metrics

# ── Constants ─────────────────────────────────────────────────────────────────

MM_INLINE_TAG_RE = re.compile(r"<\s*(image|audio|video)\s*>", flags=re.IGNORECASE)
SUPPORTED_MM_TYPES = {"image", "audio", "video"}

_SUFFIX_MAP = {
    ".jpg": ("image", "image/jpeg"), ".jpeg": ("image", "image/jpeg"),
    ".png": ("image", "image/png"), ".webp": ("image", "image/webp"), ".gif": ("image", "image/gif"),
    ".mp3": ("audio", "audio/mp3"), ".wav": ("audio", "audio/wav"),
    ".m4a": ("audio", "audio/mp4"), ".ogg": ("audio", "audio/ogg"),
    ".flac": ("audio", "audio/flac"),
    ".mp4": ("video", "video/mp4"), ".webm": ("video", "video/webm"),
    ".avi": ("video", "video/x-msvideo"), ".mkv": ("video", "video/x-matroska"),
}
_DEFAULT_MIME = {"image": "image/jpeg", "audio": "audio/wav", "video": "video/mp4"}
_GEMINI_ROLE_TO_OPENAI = {"USER": "user", "MODEL": "assistant", "user": "user", "assistant": "assistant"}


# ── Small helpers ─────────────────────────────────────────────────────────────

def _is_http_url(v: str) -> bool:
    return isinstance(v, str) and urlparse(v).scheme.lower() in {"http", "https"}


def _is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()


def infer_modality_and_mime(url: str) -> Tuple[str, str]:
    suffix = Path(urlparse(url).path).suffix.lower()
    return _SUFFIX_MAP.get(suffix, ("unknown", "application/octet-stream"))


def clean_model_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ── Input form ────────────────────────────────────────────────────────────────

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
            raise ValueError(f"Invalid input_form: {item!r}")
        result[k] = v
    return result


def input_form_tag(form: Dict[str, str]) -> str:
    shorts = {"image": "img", "audio": "aud", "video": "vid"}
    return "_".join(f"{shorts[m]}{form.get(m,'raw')[0]}" for m in ["image", "audio", "video"])


# ── Metrics ───────────────────────────────────────────────────────────────────

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
    prec, rec = overlap / len(p_tok), overlap / len(g_tok)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


# ── Media extraction ─────────────────────────────────────────────────────────

@dataclass
class DialogueMedia:
    modality: str; url: str; caption: str; media_id: str; mime_type: str


def extract_dialogue_media(dialog: Dict[str, Any]) -> List[DialogueMedia]:
    mm_value = dialog.get("mm")
    if mm_value is None: return []
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
    if not (isinstance(url, str) and url.strip()): return []
    u = url.strip()
    mod, mime = infer_modality_and_mime(u)
    caption = (qa.get("image_caption") or "").strip()
    return [DialogueMedia(modality="image" if mod == "unknown" else mod, url=u, caption=caption, media_id="", mime_type=mime)]


# ── Content building ─────────────────────────────────────────────────────────

def _filedata_part(url: str, mime: str, media_root: Optional[Path] = None) -> Dict[str, Any]:
    if _is_http_url(url):
        return {"fileData": {"mimeType": mime, "fileUri": url}}
    path = Path(url).expanduser()
    if not path.is_absolute():
        path = (Path(media_root) if media_root is not None else PROJECT_ROOT) / path
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return {"inlineData": {"mimeType": mime, "data": data}}


def _caption_for_item(item: DialogueMedia) -> str:
    parts = []
    header = item.modality + (f"_id: {item.media_id}" if item.media_id else "")
    parts.append(header)
    if item.caption: parts.append(f"{item.modality}_caption: {item.caption}")
    return "\n".join(parts) if (item.caption or item.media_id) else ""


def build_user_parts(*, user_text: str, media_items: List[DialogueMedia],
                     enabled: Dict[str, str], ref: str,
                     media_root: Optional[Path] = None) -> List[Dict[str, Any]]:
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
            parts.append(_filedata_part(item.url, item.mime_type, media_root))
        elif mode == "caption":
            cb = _caption_for_item(item)
            if not cb: raise ValueError(f"{ref}: mm[{idx}] no caption for caption mode")
            parts.append({"text": cb})
        cursor = m.end()
    suffix = text[cursor:]
    if suffix: parts.append({"text": suffix})
    return parts


def build_raw_parts(items: List[DialogueMedia], enabled: Dict[str, str],
                    media_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    return [_filedata_part(i.url, i.mime_type, media_root) for i in items
            if i.modality in SUPPORTED_MM_TYPES and enabled.get(i.modality) == "raw"]


def build_caption_block(items: List[DialogueMedia], enabled: Dict[str, str]) -> str:
    return "\n".join(_caption_for_item(i) for i in items
                     if i.modality in SUPPORTED_MM_TYPES and enabled.get(i.modality) == "caption"
                     and _caption_for_item(i)).strip()


def build_contents(dataset: Dict[str, Any], input_form: Dict[str, str],
                   sys_prompt: str, include_persona: bool = True,
                   media_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    contents = []
    preamble = []
    if sys_prompt: preamble.append(sys_prompt.strip())
    if include_persona:
        persona = dataset.get("character_profile") or {}
        for k in ("persona_summary", "conversation_style"):
            v = persona.get(k, "")
            if v: preamble.append(f"{k.replace('_',' ').title()}:\n{v}".strip())
    if preamble:
        contents.append({"role": "USER", "parts": [{"text": "\n\n".join(preamble)}]})
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
            user_parts = build_user_parts(user_text=user_text, media_items=media, enabled=input_form,
                                         ref=ref, media_root=media_root)
            if user_parts: contents.append({"role": "USER", "parts": user_parts})
            if asst_text.strip(): contents.append({"role": "MODEL", "parts": [{"text": asst_text.strip()}]})
    return contents


# ── Gemini → OpenAI format conversion ────────────────────────────────────────

def _gemini_part_to_openai(part: Dict[str, Any]) -> Dict[str, Any]:
    if "text" in part:
        return {"type": "text", "text": part["text"]}
    if "inlineData" in part:
        data = part["inlineData"]
        mime, encoded = data["mimeType"], data["data"]
        modality, media_format = mime.split("/", 1)
        if modality == "audio":
            return {"type": "input_audio", "input_audio": {"data": encoded, "format": media_format}}
        key = "image_url" if modality == "image" else "video_url"
        return {"type": key, key: {"url": f"data:{mime};base64,{encoded}"}}
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
            return {"type": "video_url", "video_url": {"url": url}}
        else:
            return {"type": "text", "text": f"[unsupported media: {mime}]"}
    return {"type": "text", "text": "[unknown part]"}


def contents_to_openai_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages = []
    for item in contents:
        role = _GEMINI_ROLE_TO_OPENAI.get(item.get("role", ""), "user")
        parts = item.get("parts", [])
        if all("text" in p and len(p) == 1 for p in parts):
            content = "\n".join(p["text"] for p in parts)
        else:
            content = [_gemini_part_to_openai(p) for p in parts]
        messages.append({"role": role, "content": content})
    return messages


# ── API call ──────────────────────────────────────────────────────────────────

def llm_chat(
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
                payload = {"model": model, "messages": contents_to_openai_messages(contents)}
                if temperature is not None: payload["temperature"] = temperature
                if max_tokens is not None: payload["max_tokens"] = max_tokens

            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            url = base_url.rstrip("/") + "/chat/completions"
            authorization = api_key
            if not is_gemini and not api_key.lower().startswith("bearer "):
                authorization = f"Bearer {api_key}"
            result = subprocess.run(
                ["curl", "-s", "-X", "POST", url,
                 "-H", "Content-Type: application/json",
                 "-H", f"Authorization: {authorization}",
                 "--data-binary", "@-"],
                input=body, capture_output=True, timeout=timeout_seconds + 10,
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
                    raise RuntimeError(f"Empty content: {json.dumps(resp)[:500]}")
                return clean_model_text(content), resp
            else:
                raise RuntimeError(f"Unexpected response: {result.stdout.decode()[:500]}")
        except subprocess.TimeoutExpired:
            last_error = "timeout"
        except Exception as e:
            last_error = str(e)[:2000]
        if attempt < max_retries - 1:
            time.sleep(min(2 * (2 ** attempt), 60))
    raise RuntimeError(f"API failed after {max_retries} attempts: {last_error}")


# ── Judge ─────────────────────────────────────────────────────────────────────

def format_judge_prompt(template: str, *, question: str, ground_truth: str, model_output: str) -> str:
    return (template.replace("{{question}}", question)
            .replace("{{ground_truth}}", ground_truth)
            .replace("{{model_output}}", model_output))


def format_nugget_judge_prompt(template: str, *, question: str, model_output: str,
                               nuggets: List[Dict[str, Any]]) -> str:
    lines = [f"  {n.get('nugget_id','?')}: {n.get('content','')}" for n in nuggets]
    return (template.replace("{{question}}", question)
            .replace("{{model_output}}", model_output)
            .replace("{{nuggets_list}}", "\n".join(lines)))


def parse_judge_json(text: str) -> Dict[str, Any]:
    cleaned = clean_model_text(text)
    try: return json.loads(cleaned)
    except Exception: pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m: return json.loads(m.group(0))
    raise ValueError(f"Cannot parse judge JSON: {cleaned[:200]}")


# ── IO helpers ────────────────────────────────────────────────────────────────

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

def load_done_indices(path: Path, judge_only: bool = False) -> set:
    done = set()
    if not path.exists(): return done
    latest = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line.strip())
                qi = row.get("qa_index")
                if isinstance(qi, int):
                    latest[qi] = row
            except Exception: pass
    for qi, row in latest.items():
        if row.get("status") != "ok":
            continue
        if judge_only:
            if "judge" in row:
                done.add(qi)
        else:
            done.add(qi)
    return done
# ── END CHANGED ──────────────────────────────────────────────────────────────

def load_jsonl_rows(path: Path, dedup: bool = True) -> List[Dict]:
    if not path.exists(): return []
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try: rows.append(json.loads(line.strip()))
            except Exception: pass
    if dedup:
        seen = {}
        for r in rows:
            qi = r.get("qa_index")
            if qi is not None: seen[qi] = r
            else: seen[id(r)] = r
        rows = list(seen.values())
    return rows


def _mean(vals): return sum(vals) / len(vals) if vals else None

def _pct(vals, p):
    if not vals: return None
    xs = sorted(vals); k = (len(xs)-1) * p / 100.0
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (k-f)*(xs[c]-xs[f]) if f != c else xs[f]


def summarize_rows(rows: List[Dict]) -> Dict:
    def _sum(subset):
        ok = [r for r in subset if r.get("status") == "ok"]
        ems  = [r["metrics"]["EM"]  for r in ok if isinstance((r.get("metrics") or {}).get("EM"),  (int, float))]
        f1s  = [r["metrics"]["F1"]  for r in ok if isinstance((r.get("metrics") or {}).get("F1"),  (int, float))]
        lats = [r["latency_seconds"] for r in ok if isinstance(r.get("latency_seconds"), (int, float))]
        judge_scores, nugget_recalls, qa_scores = [], [], []
        judge_missing_indices = []
        for r in ok:
            jr = (r.get("judge") or {}).get("result") or {}
            try:
                if str(r.get("category", "")).upper() == "SUM":
                    score = compute_nugget_metrics(jr.get('nugget_coverage'), r.get('nuggets') or [])['nugget_recall']
                    nugget_recalls.append(score)
                else:
                    score = qa_score(jr)
                    judge_scores.append(score)
                qa_scores.append(score)
            except (ValueError, KeyError, TypeError, AttributeError):
                qi = r.get("qa_index")
                if qi is not None: judge_missing_indices.append(qi)
        # ── END NEW ──
        metrics: Dict[str, Any] = {"EM_mean": _mean(ems), "F1_mean": _mean(f1s)}
        metrics["overall_score"] = _mean(qa_scores)
        if judge_scores: metrics["judge_score_mean"] = _mean(judge_scores)
        if nugget_recalls: metrics["nugget_recall_mean"] = _mean(nugget_recalls)
        result = {
            "counts": {"total": len(subset), "ok": len(ok), "error": len(subset) - len(ok),
                       "valid_judged": len(qa_scores), "unjudged_or_invalid": len(ok) - len(qa_scores)},
            "metrics": metrics,
            "latency": {"mean": _mean(lats), "p50": _pct(lats, 50), "p95": _pct(lats, 95)},
        }
        if judge_missing_indices:
            result["judge_missing"] = len(judge_missing_indices)
            result["judge_missing_indices"] = judge_missing_indices
        return result
    summary = _sum(rows)
    grouped: Dict[str, list] = {}
    for r in rows:
        t = str(r.get("category") or "__uncategorized__").strip() or "__uncategorized__"
        grouped.setdefault(t, []).append(r)
    summary["by_type"] = {t: _sum(g) for t, g in sorted(grouped.items())}
    return summary
# ── END CHANGED ──────────────────────────────────────────────────────────────
