"""Unified LLM client with OpenAI-compatible API support.

Handles retries, token counting, and Gemini native protocol as a fallback.
Works with any OpenAI-compatible endpoint (vLLM, Ollama, OpenRouter, etc.).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    run_id: str = ""
    timestamp: int = 0
    raw: dict = field(default_factory=dict)


class TokenCounter:
    """Thread-safe global token counter."""

    def __init__(self):
        self._lock = threading.Lock()
        self._totals = {"input": 0, "output": 0, "calls": 0}

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._totals["input"] += input_tokens
            self._totals["output"] += output_tokens
            self._totals["calls"] += 1

    def get(self) -> dict[str, int]:
        with self._lock:
            return dict(self._totals)

    def reset(self) -> None:
        with self._lock:
            self._totals = {"input": 0, "output": 0, "calls": 0}


_global_counter = TokenCounter()


def get_token_counter() -> TokenCounter:
    return _global_counter


class BaseLLM(ABC):
    """Base class for LLM backends."""

    def __init__(self, config: dict | Any):
        if isinstance(config, dict):
            from omnimembench.config.base import AttributeDict
            self.config = AttributeDict(config)
        else:
            self.config = config

    def reset(self) -> None:
        pass

    @abstractmethod
    def run(self, messages: list[dict]) -> LLMResponse:
        pass

    def fast_run(self, query: str) -> str:
        resp = self.run([{"role": "user", "content": query}])
        return resp.text


class OpenAILLM(BaseLLM):
    """OpenAI-compatible API client with retry logic."""

    def __init__(self, config: dict | Any):
        super().__init__(config)
        self._is_gemini_native = "gemini" in (getattr(self.config, "name", "") or "").lower() and getattr(self.config, "gemini_native", False)
        if not self._is_gemini_native:
            self.client = OpenAI(
                api_key=getattr(self.config, "api_key", "") or "EMPTY",
                base_url=getattr(self.config, "base_url", "http://localhost:8000/v1"),
            )

    def run(self, messages: list[dict], **kwargs) -> LLMResponse:
        if self._is_gemini_native:
            return self._run_gemini_native(messages)
        return self._run_openai(messages, **kwargs)

    def _run_openai(self, messages: list[dict], **kwargs) -> LLMResponse:
        max_retries = 10
        temperature = getattr(self.config, "temperature", 0.0)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.name,
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )
                text = response.choices[0].message.content or ""
                inp = response.usage.prompt_tokens if response.usage else 0
                out = response.usage.completion_tokens if response.usage else 0
                _global_counter.add(inp, out)
                return LLMResponse(
                    text=text,
                    input_tokens=inp,
                    output_tokens=out,
                    run_id=response.id or "",
                    timestamp=response.created or 0,
                )
            except Exception as e:
                err_str = str(e)
                if "insufficient_quota" in err_str:
                    raise
                if "429" in err_str or "rate_limit" in err_str.lower():
                    wait = min(2**attempt + random.uniform(0, 2), 60)
                    print(f"[RateLimit] attempt {attempt+1}/{max_retries}, waiting {wait:.1f}s")
                    time.sleep(wait)
                    continue
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2**attempt, 16))

    def _run_gemini_native(self, messages: list[dict]) -> LLMResponse:
        """Gemini native protocol via curl (for endpoints not OpenAI-compatible)."""
        contents = _openai_to_gemini_contents(messages)
        payload = {"model": self.config.name, "contents": contents}
        url = getattr(self.config, "base_url", "").rstrip("/") + "/chat/completions"

        max_retries = 5
        for attempt in range(max_retries):
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    json.dump(payload, f, ensure_ascii=False)
                    tmp_path = f.name
                try:
                    result = subprocess.run(
                        [
                            "curl", "-s", "-X", "POST", url,
                            "-H", "Content-Type: application/json",
                            "-H", f"Authorization: Bearer {self.config.api_key}",
                            "--data", f"@{tmp_path}",
                            "--max-time", "120",
                        ],
                        capture_output=True, text=True, timeout=130,
                    )
                finally:
                    os.unlink(tmp_path)

                if result.returncode != 0:
                    raise RuntimeError(f"curl failed: {result.stderr}")

                resp = json.loads(result.stdout)
                inp_tok, out_tok = 0, 0

                if "candidates" in resp:
                    text = resp["candidates"][0]["content"]["parts"][0]["text"]
                    usage = resp.get("usageMetadata", {})
                    inp_tok = usage.get("promptTokenCount", 0)
                    out_tok = usage.get("candidatesTokenCount", 0)
                elif "choices" in resp:
                    text = resp["choices"][0]["message"]["content"]
                    usage = resp.get("usage", {})
                    inp_tok = usage.get("prompt_tokens", 0)
                    out_tok = usage.get("completion_tokens", 0)
                else:
                    raise ValueError(f"Unexpected response: {json.dumps(resp)[:500]}")

                _global_counter.add(inp_tok, out_tok)
                return LLMResponse(
                    text=text,
                    input_tokens=inp_tok,
                    output_tokens=out_tok,
                    run_id=resp.get("id", ""),
                    timestamp=resp.get("created", 0),
                    raw=resp,
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2**attempt, 16))


class LocalVLLM(BaseLLM):
    """Local vLLM backend (in-process inference)."""

    def __init__(self, config: dict | Any):
        super().__init__(config)
        from vllm import LLM, SamplingParams
        self.model = LLM(self.config.name)
        self.sampling_params = SamplingParams(temperature=getattr(self.config, "temperature", 0.0))

    def run(self, messages: list[dict]) -> LLMResponse:
        result = self.model.chat(messages, self.sampling_params)[0]
        text = result.outputs[0].text
        return LLMResponse(text=text)


def create_llm(config: dict | Any) -> BaseLLM:
    """Factory to create an LLM instance from config.

    Config must have a 'method' field: 'APILLM' or 'LocalVLLM'.
    """
    if isinstance(config, dict):
        from omnimembench.config.base import AttributeDict
        config = AttributeDict(config)

    method = getattr(config, "method", "APILLM")
    if method == "LocalVLLM":
        return LocalVLLM(config)
    return OpenAILLM(config)


# --- Helpers ---

def _openai_to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Convert OpenAI message format to Gemini contents format."""
    role_map = {"user": "user", "assistant": "model", "system": "user"}
    contents = []
    for msg in messages:
        role = role_map.get(msg.get("role", "user"), "user")
        raw = msg.get("content", "")
        if isinstance(raw, str):
            parts = [{"text": raw}]
        elif isinstance(raw, list):
            parts = []
            for block in raw:
                if isinstance(block, str):
                    parts.append({"text": block})
                elif block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    parts.append({"fileData": {"mimeType": "image/jpeg", "fileUri": url}})
                elif block.get("type") == "input_audio":
                    aud = block.get("input_audio", {})
                    fmt = aud.get("format", "wav")
                    parts.append({"fileData": {"mimeType": f"audio/{fmt}", "fileUri": aud.get("data", "")}})
                elif block.get("type") == "video_url":
                    url = block.get("video_url", {}).get("url", "")
                    parts.append({"fileData": {"mimeType": "video/mp4", "fileUri": url}})
                else:
                    parts.append({"text": str(block)})
        else:
            parts = [{"text": str(raw)}]
        contents.append({"role": role, "parts": parts})
    return contents
