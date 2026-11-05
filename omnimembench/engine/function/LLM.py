import json, os, random, subprocess, tempfile, time, threading
from abc import ABC, abstractmethod
from openai import OpenAI


class BaseLLM(ABC):
    def __init__(self, config):
        self.config = config

    def reset(self):
        pass

    @abstractmethod
    def fast_run(self, query):
        pass


def _messages_to_contents(messages):
    """Convert OpenAI messages list to Gemini contents format."""
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


class APILLM(BaseLLM):
    # Thread-safe class-level token & call counter
    _token_lock = threading.Lock()
    _total_tokens = {'input': 0, 'output': 0, 'calls': 0}

    @classmethod
    def reset_token_counter(cls):
        with cls._token_lock:
            cls._total_tokens = {'input': 0, 'output': 0, 'calls': 0}

    @classmethod
    def get_token_counter(cls):
        with cls._token_lock:
            return dict(cls._total_tokens)

    @classmethod
    def _add_tokens(cls, inp, out):
        with cls._token_lock:
            cls._total_tokens['input'] += inp
            cls._total_tokens['output'] += out
            cls._total_tokens['calls'] += 1

    def __init__(self, config):
        super().__init__(config)
        self.is_gemini = 'gemini' in (self.config.name or '').lower()
        if not self.is_gemini:
            self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def parse_response(self, response):
        return {
            'run_id': response.id,
            'time_stamp': response.created,
            'result': response.choices[0].message.content,
            'input_token': response.usage.prompt_tokens,
            'output_token': response.usage.completion_tokens
        }

    def _run_gemini(self, message_list):
        contents = _messages_to_contents(message_list)
        payload = {"model": self.config.name, "contents": contents}
        url = self.config.base_url.rstrip("/") + "/chat/completions"

        max_retries = 5
        for attempt in range(max_retries):
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(payload, f, ensure_ascii=False)
                    tmp_path = f.name
                try:
                    result = subprocess.run(
                        ["curl", "-s", "-X", "POST", url,
                         "-H", "Content-Type: application/json",
                         "-H", f"Authorization: Bearer {self.config.api_key}",
                         "--data", f"@{tmp_path}",
                         "--max-time", "120"],
                        capture_output=True, text=True, timeout=130)
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

                self._add_tokens(inp_tok, out_tok)
                return {
                    'run_id': resp.get('id', ''),
                    'time_stamp': resp.get('created', 0),
                    'result': text,
                    'input_token': inp_tok,
                    'output_token': out_tok,
                }
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 16))

    def run(self, message_list):
        if self.is_gemini:
            return self._run_gemini(message_list)
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.name,
                    messages=message_list,
                    temperature=self.config.temperature,
                    extra_body={'chat_template_kwargs': {'enable_thinking': False}}
                )
                parsed = self.parse_response(response)
                self._add_tokens(parsed['input_token'], parsed['output_token'])
                return parsed
            except Exception as e:
                err_str = str(e)
                if 'insufficient_quota' in err_str:
                    raise
                if '429' in err_str or 'rate_limit' in err_str.lower() or 'RateLimitError' in type(e).__name__:
                    wait = min(2 ** attempt + random.uniform(0, 2), 60)
                    print(f"[RateLimit] attempt {attempt+1}/{max_retries}, waiting {wait:.1f}s ...", flush=True)
                    time.sleep(wait)
                    continue
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 16))

    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        return response['result']


class LocalVLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        from vllm import LLM, SamplingParams
        self.model = LLM(config.name)
        self.sampling_params = SamplingParams(temperature=config.temperature)

    def run(self, message_list):
        return self.model.chat(message_list, self.sampling_params)[0]

    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        return response.outputs[0].text