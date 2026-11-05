#!/usr/bin/env python3
"""
Unified LLM client for the generation pipeline.
Supports both Gemini-native and OpenAI-compatible endpoints via curl.
"""

import base64
import json
import time
import re
import subprocess
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any


GA_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = GA_ROOT / ".env"
BASE_URL = "https://localhost:8000/v1"
MODEL = "your-model-name"


def _load_env_file(env_file: Path = DEFAULT_ENV_FILE) -> None:
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


_load_env_file()
BASE_URL = os.environ.get("LLM_API_BASE") or BASE_URL
MODEL = os.environ.get("LLM_MODEL") or MODEL


def _is_native(model: str) -> bool:
    return "gemini" in model.lower()


def _build_native_payload(
    contents: list,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    return {
        "model": model,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
        "contents": contents,
    }


def _build_openai_payload(
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }


def _text_payload(prompt: str, model: str, temperature: float, max_tokens: int) -> dict:
    if _is_native(model):
        contents = [{"parts": [{"text": prompt}], "role": "user"}]
        return _build_native_payload(contents, model, temperature, max_tokens)
    else:
        messages = [{"role": "user", "content": prompt}]
        return _build_openai_payload(messages, model, temperature, max_tokens)


def _history_payload(messages: list, model: str, temperature: float, max_tokens: int) -> dict:
    if _is_native(model):
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append({"parts": [{"text": msg["content"]}], "role": role})
        return _build_native_payload(contents, model, temperature, max_tokens)
    else:
        return _build_openai_payload(messages, model, temperature, max_tokens)


def _local_media(url: str) -> Optional[str]:
    if url.startswith(("http://", "https://")):
        return None
    path = Path(url).expanduser()
    if not path.is_absolute():
        path = Path(os.environ.get("MEDIA_ROOT") or GA_ROOT.parent) / path
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _media_part_native(url: str, mime_type: str) -> dict:
    data = _local_media(url)
    if data is None:
        return {"fileData": {"mimeType": mime_type, "fileUri": url}}
    return {"inlineData": {"mimeType": mime_type, "data": data}}


def _media_part_openai(url: str, modality_type: str, mime_type: str) -> dict:
    data = _local_media(url)
    if data is not None:
        if modality_type == "audio":
            return {"type": "input_audio", "input_audio": {"data": data, "format": Path(url).suffix.lstrip(".").lower()}}
        url = f"data:{mime_type};base64,{data}"
    if modality_type == "video":
        return {"type": "video_url", "video_url": {"url": url}}
    if modality_type == "audio":
        return {"type": "input_audio", "input_audio": {"url": url}}
    return {"type": "image_url", "image_url": {"url": url}}


def _multimodal_payload(
    prompt: str,
    media: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    if _is_native(model):
        parts = [_media_part_native(m["url"], m["mime_type"]) for m in media]
        parts.append({"text": prompt})
        contents = [{"parts": parts, "role": "user"}]
        return _build_native_payload(contents, model, temperature, max_tokens)
    else:
        content = [_media_part_openai(m["url"], m["modality_type"], m["mime_type"]) for m in media]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return _build_openai_payload(messages, model, temperature, max_tokens)


def _extract_response(response_json: dict, model: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
    usage = None

    if _is_native(model):
        candidates = response_json.get("candidates", [])
        if not candidates:
            return None, None
        text = candidates[0]["content"]["parts"][0]["text"]
        if "usage" in response_json:
            u = response_json["usage"]
            usage = {
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0),
                "total_tokens": u.get("total_tokens", 0),
            }
    else:
        choices = response_json.get("choices", [])
        if not choices:
            return None, None
        text = choices[0]["message"]["content"]
        if "usage" in response_json:
            u = response_json["usage"]
            usage = {
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0),
                "total_tokens": u.get("total_tokens", 0),
            }

    return text, usage


def _curl_call(payload: dict, api_key: str, max_time: int = 300) -> Tuple[int, str, str]:
    body = json.dumps(payload, ensure_ascii=False)
    cmd = [
        "curl", "--location",
        f"{BASE_URL}/chat/completions",
        "--header", "Content-Type: application/json",
        "--header", f"Authorization: Bearer {api_key}",
        "--data-binary", "@-",
        "--silent", "--show-error",
        "--max-time", str(max_time),
    ]
    result = subprocess.run(
        cmd, input=body, capture_output=True, text=True, timeout=max_time + 10,
    )
    return result.returncode, result.stdout, result.stderr


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = re.sub(r"^```json\s*\n?", "", text)
    if text.startswith("```"):
        text = re.sub(r"^```\s*\n?", "", text)
    if text.endswith("```"):
        text = re.sub(r"\n?```$", "", text)
    text = re.sub(r",\s*\]", "]", text)
    text = re.sub(r",\s*\}", "}", text)
    return text


def _call_core(
    payload: dict,
    model: str,
    max_retries: int = 3,
    clean_json: bool = True,
    max_time: int = 300,
) -> Tuple[bool, str, Optional[Dict[str, int]]]:
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return False, "Missing API key. Set LLM_API_KEY or place it in .env.", None

    for attempt in range(max_retries):
        try:
            rc, stdout, stderr = _curl_call(payload, api_key, max_time)
            if rc != 0:
                raise RuntimeError(f"curl failed: {stderr}")

            response_json = json.loads(stdout)
            text, usage = _extract_response(response_json, model)

            if text is None:
                raise RuntimeError(f"Unexpected response format: {response_json}")
            if not text.strip():
                raise RuntimeError("Empty response from API")

            if clean_json:
                text = _clean_json_text(text)

            return True, text, usage

        except (subprocess.TimeoutExpired, json.JSONDecodeError, RuntimeError, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return False, f"Error after {max_retries} retries: {e}", None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return False, f"Error after {max_retries} retries: {e}", None

    return False, "Max retries exceeded", None


def call_gemini(
    prompt: str,
    model: str = MODEL,
    temperature: float = 0.9,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> Tuple[bool, str]:
    payload = _text_payload(prompt, model, temperature, max_tokens)
    success, text, _ = _call_core(payload, model, max_retries)
    return success, text


def call_gemini_with_usage(
    prompt: str,
    model: str = MODEL,
    temperature: float = 0.9,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> Tuple[bool, str, Optional[Dict[str, int]]]:
    payload = _text_payload(prompt, model, temperature, max_tokens)
    return _call_core(payload, model, max_retries)


def call_gemini_with_history(
    messages: list,
    model: str = MODEL,
    temperature: float = 0.9,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> Tuple[bool, str]:
    payload = _history_payload(messages, model, temperature, max_tokens)
    success, text, _ = _call_core(payload, model, max_retries, clean_json=False)
    return success, text


def call_gemini_multimodal(
    prompt: str,
    media: List[Dict[str, str]],
    model: str = MODEL,
    temperature: float = 0.9,
    max_tokens: int = 4096,
    max_retries: int = 3,
    max_time: int = 120,
    clean_json: bool = False,
) -> Tuple[bool, str]:
    """
    Args:
        prompt: Text prompt
        media: [{"url": "...", "mime_type": "image/jpeg", "modality_type": "image"}, ...]
    """
    payload = _multimodal_payload(prompt, media, model, temperature, max_tokens)
    success, text, _ = _call_core(payload, model, max_retries, clean_json=clean_json, max_time=max_time)
    return success, text
