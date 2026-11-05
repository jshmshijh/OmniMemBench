import logging, json, random, string, re, sys, os
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message="System prompt modified")

import subprocess, tempfile, urllib.request, base64, warnings, time, threading
import numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnimembench.engine.config.Config import MemoryConfig
from omnimembench.engine.memory.SlidingWindow import SlidingWindow
from omnimembench.engine.memory.NaiveRAG import NaiveRAG
from omnimembench.engine.memory.GenerativeAgents import GenerativeAgents
from omnimembench.engine.memory.MultimodalRAG import MultimodalRAG
from omnimembench.engine.memory.NeuralGraph import NeuralGraph
from omnimembench.engine.memory.AUGUSTUS import AUGUSTUS
from omnimembench.engine.memory.UniversalRAG import UniversalRAG
from omnimembench.engine.memory.MemGPT import MemGPT
from omnimembench.engine.memory.Reflexion import Reflexion
from omnimembench.engine.memory.AMem import AMem
from omnimembench.engine.memory.Mem0 import Mem0
from omnimembench.engine.memory.LangMem import LangMem
from omnimembench.engine.memory.HippoRAG import HippoRAG
from omnimembench.config import get_memory_config, MEMORY_CONFIGS






os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DIALOG_DIR = os.path.join(DATA_DIR, "dialog")  # default (legacy) dialog dir
IMAGE_DIR = os.path.join(DATA_DIR, "image")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result_debug")
STORE_STATE_DIR = os.path.join(PROJECT_ROOT, "store_state")


def inject_llm_config(api_key, base_url, model_name):
    """Inject LLM config into memory configs at runtime via environment variables."""
    os.environ.setdefault('LLM_API_KEY', api_key)
    os.environ.setdefault('LLM_API_BASE', base_url)
    os.environ.setdefault('LLM_MODEL', model_name)


def inject_external_baseline_config(memory_name, api_key, base_url, model_name, proxy_base_url=None):
    """Inject LLM config for external baselines - now handled via env vars."""
    pass

PROMPT_DIR = os.path.join(SCRIPT_DIR, "prompt")
ENV_PATHS = [os.path.join(PROJECT_ROOT, ".env"), os.path.join(PROJECT_ROOT, "env")]

# Runtime-overridable: set by --data_root / --tier at startup
_DATA_ROOT = None   # e.g. /OmniMemBench/data  (new-style per-sample dirs)
_TIER      = None   # e.g. benchmark_128k

USE_GEMINI_NATIVE = False

# ── Memory config mapping ───────────────────────────────────────────────────
MEMORY_CONFIG_MAP = {name: MEMORY_CONFIGS[name] for name in MEMORY_CONFIGS}

# Memory methods that use full context (no retrieval step)
NO_RETRIEVAL_MEMORIES = {"Reflexion", "MemGPT"}

# ── Prompt loading ───────────────────────────────────────────────────────────
_PROMPT_CACHE = {}

def load_prompt_file(filename):
    if filename in _PROMPT_CACHE:
        return _PROMPT_CACHE[filename]
    prompt_path = os.path.join(PROMPT_DIR, filename)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception:
        content = ""
    _PROMPT_CACHE[filename] = content
    return content

for _f in ["sys_prompt.txt", "ar_prompt.txt", "cd_prompt.txt", "vs_prompt.txt"]:
    load_prompt_file(_f)

# ── Prompt templates ─────────────────────────────────────────────────────────
SystemPrompt = load_prompt_file("sys_prompt.txt") or (
    "Your task is to answer questions in a concise manner with the help of memory content.\n"
    "When the question is: \"What did the charity race raise awareness for?\", you should not answer "
    "in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: "
    "\"Mental health.\", as this is more concise.\n")

TextMsgPrompt = "\nThe following is the relevant conversation history:\n\n{memory_context}\n"
MsgStartPromptWOMemory = "\nThe following is the relevant conversation history:\n\n"

MMMemoryDialoguePrompt = "\n{textual_context}\nimage:\nimage_id: {image_id}\nimage_content:\n"
MMMemoryAudioPrompt = "\n{textual_context}\naudio:\naudio_id: {audio_id}\naudio_content:\n"
MMMemoryVideoPrompt = "\n{textual_context}\nvideo:\nvideo_id: {video_id}\nvideo_content:\n"

DialogueAgentPrompt = (
    "Your task is to answer the question about the conversation between {speaker_a} and {speaker_b} "
    "based on the provided conversation history.\n"
    "Answer the question to the best of your ability using the available information. "
    "Use reasoning to infer answers when the information is not explicitly stated but can be deduced.\n"
    "Please only provide the content of the answer, without including introductory phrases like 'answer:'.\n"
    "For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.\n"
    "Generate answers primarily concise, yet complete enough to accurately answer the questions.\n\n"
    "The current question is as follows:\n{observation} {format_constraint}\n")

DialogueAgentPromptImage = "\nHere is the attached image of the question:\n"

# ── Utilities ────────────────────────────────────────────────────────────────

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_local_env_files():
    for env_path in ENV_PATHS:
        if not os.path.exists(env_path): continue
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"): continue
                if line.startswith("export "): line = line[7:].strip()
                if "=" not in line: continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value

def resolve_media_path(path):
    if path.startswith(("http://", "https://")):
        return path
    if os.path.isabs(path):
        return path
    if path.startswith("../image/"):
        return os.path.normpath(os.path.join(IMAGE_DIR, path.replace("../image/", "")))
    return os.path.normpath(os.path.join(IMAGE_DIR, path))

def path_to_url(path):
    if path.startswith(("http://", "https://", "file://")):
        return path
    return f"file://{resolve_media_path(path)}"

def encode_image(image_path):
    if image_path.startswith(("http://", "https://")):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name
        try:
            urllib.request.urlretrieve(image_path, tmp_path)
            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        finally:
            os.unlink(tmp_path)
    final_path = resolve_media_path(image_path)
    with open(final_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_audio_format(url):
    lower = url.lower().split('?')[0]
    if lower.endswith('.mp3'): return 'mp3'
    if lower.endswith('.m4a'): return 'mp4'
    if lower.endswith('.ogg'): return 'ogg'
    return 'wav'

def get_mime_type(url, modality):
    lower = url.lower().split('?')[0]
    _map = {
        'image': {'.png': 'image/png', '.webp': 'image/webp', '.gif': 'image/gif'},
        'audio': {'.mp3': 'audio/mp3', '.m4a': 'audio/mp4', '.ogg': 'audio/ogg'},
        'video': {'.webm': 'video/webm', '.avi': 'video/x-msvideo', '.mkv': 'video/x-matroska'},
    }
    defaults = {'image': 'image/jpeg', 'audio': 'audio/wav', 'video': 'video/mp4'}
    for ext, mime in _map.get(modality, {}).items():
        if lower.endswith(ext): return mime
    return defaults.get(modality, 'application/octet-stream')

def get_available_datasets():
    """Return dataset names depending on which data layout is active."""
    if _DATA_ROOT and _TIER:
        datasets = []
        if os.path.isdir(_DATA_ROOT):
            for entry in sorted(os.listdir(_DATA_ROOT)):
                sample_dir = os.path.join(_DATA_ROOT, entry)
                if os.path.isdir(sample_dir):
                    tier_file = os.path.join(sample_dir, f"{_TIER}.json")
                    if os.path.exists(tier_file):
                        datasets.append(entry)
        return datasets
    datasets = []
    if os.path.exists(DIALOG_DIR):
        for fn in os.listdir(DIALOG_DIR):
            if fn.endswith('.json') and '_results_' not in fn and '_evaluate_result_' not in fn:
                datasets.append(fn.replace('.json', ''))
    return sorted(datasets)

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ── VLM Agent (unified: OpenAI SDK + Gemini native, with token tracking) ────

def _is_gemini_model(name):
    return 'gemini' in (name or '').lower()

class VLMAgent:
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

    def __init__(self, model_name=None, seed=None):
        self.model_name = model_name or OPENAI_MODEL
        self.seed = seed
        self.is_gemini = _is_gemini_model(self.model_name)
        if not self.is_gemini:
            self.client = OpenAI(api_key=OPENAI_APIKEY, base_url=OPENAI_APIBASE)

    def _run_openai(self, message_list):
        api_params = {'model': self.model_name, 'messages': message_list, 'temperature': 0.0}
        if self.seed is not None:
            api_params['seed'] = self.seed
        for attempt in range(10):
            try:
                resp = self.client.chat.completions.create(**api_params)
                inp = resp.usage.prompt_tokens if resp.usage else 0
                out = resp.usage.completion_tokens if resp.usage else 0
                self._add_tokens(inp, out)
                return resp.choices[0].message.content
            except Exception as e:
                err_str = str(e)
                if '429' in err_str or 'rate_limit' in err_str.lower() or 'RateLimitError' in type(e).__name__:
                    if attempt == 9: raise
                    wait = min(2 ** attempt + random.uniform(0, 2), 60)
                    print(f"[VLMAgent RateLimit] attempt {attempt+1}/10, error: {err_str[:100]}, waiting {wait:.1f}s ...", flush=True)
                    time.sleep(wait)
                    continue
                print(f"[VLMAgent Error] attempt {attempt+1}/10, error: {err_str[:200]}", flush=True)
                if attempt == 9: raise
                time.sleep(min(2 ** attempt, 16))

    def _run_gemini(self, parts):
        payload = {
            "model": self.model_name,
            "contents": [{"role": "user", "parts": parts}]
        }
        for attempt in range(10):
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(payload, f); tmp_path = f.name
                try:
                    result = subprocess.run(
                        ["curl", "--location", f"{OPENAI_APIBASE}/chat/completions",
                         "--header", "Content-Type: application/json",
                         "--header", f"Authorization: Bearer {OPENAI_APIKEY}",
                         "--data", f"@{tmp_path}", "--silent", "--show-error", "--max-time", "120"],
                        capture_output=True, text=True, timeout=130)
                finally:
                    os.unlink(tmp_path)
                if result.returncode != 0:
                    raise RuntimeError(f"curl failed: {result.stderr}")
                resp = json.loads(result.stdout)
                err_code = resp.get('error', {}).get('code', '') if isinstance(resp, dict) else ''
                err_msg = str(resp.get('error', {}).get('message', '')) if isinstance(resp, dict) else ''
                if '429' in str(err_code) or 'rate_limit' in err_msg.lower() or 'RateLimitError' in err_msg:
                    if attempt == 9:
                        raise RuntimeError(f"RateLimit after 10 attempts: {err_msg}")
                    wait = min(2 ** attempt + random.uniform(0, 2), 60)
                    print(f"[VLMAgent RateLimit] attempt {attempt+1}/10, waiting {wait:.1f}s ...", flush=True)
                    time.sleep(wait)
                    continue
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
                return text
            except Exception as e:
                if attempt == 9: raise
                time.sleep(min(2 ** attempt, 16))

    def _build_mm_flow_openai(self, memory_dict_list):
        flow = []
        for md in memory_dict_list:
            text, img, aud, vid = md.get('text',''), md.get('image'), md.get('audio'), md.get('video')
            flow.append({"type": "text", "text": text})
            if img and img.get('path'):
                flow.append({"type": "text", "text": f"image_id: {img.get('img_id', '')}"})
                flow.append({"type": "image_url", "image_url": {"url": path_to_url(img['path'])}})
            if aud and aud.get('url'):
                flow.append({"type": "text", "text": f"audio_id: {aud.get('audio_id', '')}"})
                flow.append({"type": "input_audio", "input_audio": {"data": path_to_url(aud['url']), "format": get_audio_format(aud['url'])}})
            if vid and vid.get('url'):
                flow.append({"type": "text", "text": f"video_id: {vid.get('video_id', '')}"})
                flow.append({"type": "video_url", "video_url": {"url": path_to_url(vid['url'])}})
        return flow

    def _build_mm_flow_gemini(self, memory_dict_list):
        parts = []
        for md in memory_dict_list:
            text, img, aud, vid = md.get('text',''), md.get('image'), md.get('audio'), md.get('video')
            parts.append({"text": text})
            if img and img.get('path'):
                parts.append({"text": f"image_id: {img.get('img_id', '')}"})
                parts.append({"fileData": {"mimeType": get_mime_type(img['path'], 'image'), "fileUri": img['path']}})
            if aud and aud.get('url'):
                parts.append({"text": f"audio_id: {aud.get('audio_id', '')}"})
                parts.append({"fileData": {"mimeType": get_mime_type(aud['url'], 'audio'), "fileUri": aud['url']}})
            if vid and vid.get('url'):
                parts.append({"text": f"video_id: {vid.get('video_id', '')}"})
                parts.append({"fileData": {"mimeType": get_mime_type(vid['url'], 'video'), "fileUri": vid['url']}})
        return parts

    def fast_run_with_mm_memory(self, memory_dict_list, query_prompt, query_img=None):
        if self.is_gemini:
            return self._fast_run_mm_gemini(memory_dict_list, query_prompt, query_img)
        return self._fast_run_mm_openai(memory_dict_list, query_prompt, query_img)

    def _append_query_img_openai(self, content, query_img):
        if not query_img: return
        p = query_img.get('path') if isinstance(query_img, dict) else None
        if not p: raise ValueError(f"query_img provided but path missing: {query_img}")
        content.append({"type":"text","text":DialogueAgentPromptImage})
        content.append({"type":"image_url","image_url":{"url":path_to_url(p)}})

    def _fast_run_mm_openai(self, memory_dict_list, query_prompt, query_img=None):
        if not memory_dict_list:
            content = [{"type":"text","text":query_prompt}]
            self._append_query_img_openai(content, query_img)
            return self._run_openai([{"role":"system","content":SystemPrompt},{"role":"user","content":content}])
        flow = self._build_mm_flow_openai(memory_dict_list)
        content = [{"type":"text","text":MsgStartPromptWOMemory}, *flow, {"type":"text","text":query_prompt}]
        self._append_query_img_openai(content, query_img)
        return self._run_openai([{"role":"system","content":SystemPrompt},{"role":"user","content":content}])

    def _fast_run_mm_gemini(self, memory_dict_list, query_prompt, query_img=None):
        parts = [{"text": f"{SystemPrompt}\n\n"}]
        if not memory_dict_list:
            parts.append({"text": query_prompt})
            return self._run_gemini(parts)
        parts.append({"text": MsgStartPromptWOMemory})
        parts.extend(self._build_mm_flow_gemini(memory_dict_list))
        parts.append({"text": query_prompt})
        return self._run_gemini(parts)

    def fast_run_with_textual_memory(self, text_memory_prompt, query_prompt, query_img=None):
        if self.is_gemini:
            parts = [{"text":f"{SystemPrompt}\n\n"}, {"text":text_memory_prompt}, {"text":query_prompt}]
            return self._run_gemini(parts)
        content = [{"type":"text","text":text_memory_prompt}, {"type":"text","text":query_prompt}]
        self._append_query_img_openai(content, query_img)
        return self._run_openai([{"role":"system","content":SystemPrompt},{"role":"user","content":content}])


# ── Dialogue Agent ───────────────────────────────────────────────────────────

class DialogueAgent:
    def __init__(self, memory_name, config, model_name=None, seed=None):
        self.vlm = VLMAgent(model_name=model_name, seed=seed)
        self.memory_name = memory_name
        self.memory = eval(memory_name)(MemoryConfig(config))
        self.is_multimodal = config.get('is_multimodal', False)

    def reset(self):
        self.memory.reset()

    def memory_store(self, message_dict):
        if self.is_multimodal:
            self.memory.store(message_dict)
            return
        text_str = message_dict.get('text', '') if isinstance(message_dict, dict) else message_dict
        for modality, id_key, cap_key in [('image','img_id','caption'), ('audio','audio_id','caption'), ('video','video_id','caption')]:
            info = message_dict.get(modality)
            if info and isinstance(info, dict):
                mid = info.get(id_key, '')
                mcap = info.get(cap_key, '')
                if mid or mcap:
                    text_str += f'\n{modality}:\n{modality}_id: {mid}\n{modality}_caption: {mcap}'
        if isinstance(message_dict, dict):
            message_dict['text'] = text_str
        self.memory.store(message_dict)

    def memory_recall(self, observation, observation_image=None):
        if self.is_multimodal:
            return self.memory.recall({'text': observation, 'image': observation_image})
        if observation_image and observation_image.get('caption'):
            observation += "\nquestion's image:\nimage_caption: " + observation_image['caption']
        return self.memory.recall(observation)

    def response(self, memory_result, observation, speaker_a, speaker_b, observation_image=None, format_constraint=None):
        fc_str = f"\n\n{format_constraint}" if format_constraint else ""
        query_prompt = PromptTemplate(
            input_variables=['observation','speaker_a','speaker_b','format_constraint'],
            template=DialogueAgentPrompt).format(
            observation=observation, speaker_a=speaker_a, speaker_b=speaker_b, format_constraint=fc_str)
        if self.is_multimodal:
            return self.vlm.fast_run_with_mm_memory(memory_result, query_prompt, observation_image)
        text_memory_prompt = PromptTemplate(
            input_variables=['memory_context'], template=TextMsgPrompt).format(memory_context=memory_result)
        return self.vlm.fast_run_with_textual_memory(text_memory_prompt, query_prompt, observation_image)


# ── Data processing ──────────────────────────────────────────────────────────

def _extract_media_from_mm(dialog):
    mm = dialog.get("mm")
    if not mm or not isinstance(mm, list):
        return [], [], [], None, None
    img_list, img_caption, img_id_list = [], [], []
    audio_info, video_info = None, None
    for item in mm:
        mod = (item.get("type") or "").strip().lower()
        url = (item.get("url") or "").strip()
        caption = (item.get("caption") or item.get("description") or "").strip()
        mm_id = (item.get("mm_id") or item.get("media_id") or "").strip()
        if mod == "image" and url and not img_list:
            img_list.append(url); img_caption.append(caption); img_id_list.append(mm_id)
        elif mod == "audio" and url and audio_info is None:
            audio_info = {'url': url, 'caption': caption, 'audio_id': mm_id}
        elif mod == "video" and url and video_info is None:
            video_info = {'url': url, 'caption': caption, 'video_id': mm_id}
    return img_list, img_caption, img_id_list, audio_info, video_info


def process_conversation(conversation_data, data_dir=None, character_profile=None):
    if data_dir is None:
        data_dir = DATA_DIR
    processed = []
    speaker_a = f"user ({character_profile.get('name')})" if character_profile and character_profile.get("name") else "user"

    for session_data in conversation_data:
        session_id = session_data.get("session_id", "")
        session_date = session_data.get("date", "")

        for dialog in session_data.get("dialogues", []):
            user_text = dialog.get('user', '')
            assistant_text = dialog.get('assistant', '')
            if not user_text and not assistant_text:
                continue

            parts = []
            if user_text: parts.append(f'{speaker_a}: {user_text}')
            if assistant_text: parts.append(f'assistant: {assistant_text}')
            combined_text = '\n'.join(parts)

            # Extract media
            img_list, img_caption, img_id_list = [], [], []
            mm_audio, mm_video = None, None
            if dialog.get("input_image"):
                img_list = dialog["input_image"]
                img_caption = dialog.get("image_caption", [])
                img_id_list = dialog.get("image_id", [])
            elif dialog.get("mm"):
                img_list, img_caption, img_id_list, mm_audio, mm_video = _extract_media_from_mm(dialog)

            observation = {
                'text': combined_text,
                'image': None,
                'timestamp': session_date,
                'dialogue_id': dialog.get('round', ''),
                'session_id': session_id,
            }

            if img_list:
                try:
                    observation['image'] = {
                        'path': resolve_media_path(img_list[0]) if not img_list[0].startswith(("http://","https://")) else img_list[0],
                        'caption': img_caption[0] if img_caption else "",
                        'img_id': img_id_list[0] if img_id_list else "",
                    }
                except (IndexError, TypeError) as e:
                    print(f"Warning: Failed to process image in {dialog.get('round','unknown')}: {e}")

            if dialog.get("input_audio"):
                observation['audio'] = {'url': dialog["input_audio"][0],
                    'caption': dialog.get("audio_caption",[""])[0], 'audio_id': dialog.get("audio_id",[""])[0]}
            elif mm_audio:
                observation['audio'] = mm_audio
            if dialog.get("input_video"):
                observation['video'] = {'url': dialog["input_video"][0],
                    'caption': dialog.get("video_caption",[""])[0], 'video_id': dialog.get("video_id",[""])[0]}
            elif mm_video:
                observation['video'] = mm_video

            processed.append(observation)
    return processed


# ── Main benchmark runner ────────────────────────────────────────────────────

FORMAT_CONSTRAINT_MAP = {"AR": "ar_prompt.txt", "CD": "cd_prompt.txt", "VS": "vs_prompt.txt"}

def run_mm_bench(llm_name, memory_name, config, data_name, save_results, save_efficiency,
                 eval_retrieval_metrics=False, eval_topk=5, model_name=None, seed=None,
                 num_workers=1, emb_mode='raw', reuse_agent=None,
                 phase='both', store_state_path=None, store_llm_name=None):
    from omnimembench.engine.function.LLM import APILLM

    if memory_name == 'MemGPT':
        num_workers = 1

    is_multimodal = config.get('is_multimodal', False)
    _store_llm_arg = store_llm_name
    _result_mid = os.path.join('store', f"{memory_name}_{emb_mode}") if _store_llm_arg else f"{memory_name}_{emb_mode}"
    if _DATA_ROOT and _TIER:
        output_file = os.path.join(RESULT_DIR, llm_name, _result_mid, _TIER, data_name, f"{data_name}_results.json")
    else:
        output_file = os.path.join(RESULT_DIR, llm_name, _result_mid, data_name, f"{data_name}_results.json")
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    if _DATA_ROOT and _TIER:
        json_path = os.path.join(_DATA_ROOT, data_name, f"{_TIER}.json")
    else:
        json_path = os.path.join(DIALOG_DIR, f"{data_name}.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}"); return

    _llm_safe = llm_name.replace('/', '_').replace(' ', '_')
    if _DATA_ROOT and _TIER:
        _dir_tag = _TIER
    else:
        _dir_tag = os.path.basename(DIALOG_DIR)
    _isolation_key = f"{_llm_safe}__{_dir_tag}__{data_name}"
    TMP_ROOT = os.path.join(PROJECT_ROOT, 'tmp')
    _method_tmp = os.path.join(TMP_ROOT, memory_name)

    if memory_name == 'Mem0' and 'mem0' in config:
        _qdrant_path = os.path.join(_method_tmp, _isolation_key)
        config['mem0']['user_id'] = _isolation_key
        config['mem0']['qdrant_path'] = _qdrant_path
        print(f"[Mem0] Isolated qdrant_path: {_qdrant_path}")

    if memory_name == 'HippoRAG' and 'hipporag' in config:
        _hipporag_save_dir = os.path.join(_method_tmp, _isolation_key)
        config['hipporag']['save_dir'] = _hipporag_save_dir
        print(f"[HippoRAG] Isolated save_dir: {_hipporag_save_dir}")

    APILLM.reset_token_counter()
    VLMAgent.reset_token_counter()

    if reuse_agent is not None:
        memory_agent = reuse_agent
    else:
        memory_agent = DialogueAgent(memory_name, config, model_name=model_name, seed=seed)
    memory_agent.reset()

    character_profile = dataset.get("character_profile", {})
    sample_id = character_profile.get("name", data_name)
    conversation_data = dataset.get("multi_session_dialogues", [])
    qa_pairs = dataset.get("QAs") or dataset.get("human-annotated QAs", [])

    processed_dialogs = process_conversation(conversation_data, data_dir=DATA_DIR, character_profile=character_profile)
    if not processed_dialogs:
        print(f"No valid conversation data in {sample_id}"); return

    round_text_map = {}
    for obs in processed_dialogs:
        did = obs.get('dialogue_id', '')
        if did:
            round_text_map[did] = obs.get('text', '')

    speaker_a = f"user ({character_profile.get('name')})" if character_profile and character_profile.get("name") else "user"
    speaker_b = "assistant"

    # ── Store phase ──────────────────────────────────────────────────────────
    memory_duration = 0.0
    memory_engine_tokens = {'input': 0, 'output': 0, 'calls': 0}
    memory_vlm_tokens = {'input': 0, 'output': 0, 'calls': 0}

    if phase in ('both', 'store'):
        if store_state_path and os.path.exists(store_state_path) and phase == 'both':
            print(f"Store state already exists at {store_state_path}, skipping store phase")
            memory_agent.memory.load_state(store_state_path)
            print(f"Memory state loaded from {store_state_path}")
        else:
            APILLM.reset_token_counter()
            VLMAgent.reset_token_counter()
            memory_start = time.time()
            for i, dialog in enumerate(tqdm(processed_dialogs, desc="Storing dialogs")):
                try:
                    mm_info = {m: bool(dialog.get(m)) for m in ('image','audio','video')}
                    # print(f"[Store] {i}/{len(processed_dialogs)} id={dialog.get('dialogue_id')} media={mm_info}", flush=True)
                    memory_agent.memory_store(dialog)
                except Exception as e:
                    print(f"Warning: store failed {dialog.get('dialogue_id','unknown')}: {e}")
            memory_duration = time.time() - memory_start
            memory_engine_tokens = APILLM.get_token_counter()
            memory_vlm_tokens = VLMAgent.get_token_counter()

            if store_state_path:
                os.makedirs(os.path.dirname(store_state_path), exist_ok=True)
                memory_agent.memory.save_state(store_state_path)
                print(f"Memory state saved to {store_state_path}")

    if phase == 'store':
        if save_efficiency:
            eff = {
                "sample_id": sample_id, "llm_name": llm_name, "memory_name": memory_name,
                "is_multimodal": is_multimodal, "conversation_turns": len(processed_dialogs),
                "phase": "store",
                "memory_phase": {
                    "time_seconds": memory_duration,
                    "engine_tokens": memory_engine_tokens,
                    "vlm_tokens": memory_vlm_tokens,
                },
                "recorded_at": get_timestamp(),
            }
            eff_dir = os.path.dirname(store_state_path) if store_state_path else output_dir
            eff_file = os.path.join(eff_dir, f"{data_name}_store_efficiency.json")
            os.makedirs(os.path.dirname(eff_file), exist_ok=True)
            with open(eff_file, "w", encoding="utf-8") as f:
                json.dump(eff, f, ensure_ascii=False, indent=2)
            print(f"Store efficiency saved to {eff_file}")
        return

    # ── Load state for QA-only phase ─────────────────────────────────────────
    if phase == 'qa':
        if not store_state_path or not os.path.exists(store_state_path):
            print(f"Error: store state not found at {store_state_path}")
            return
        memory_agent.memory.load_state(store_state_path)
        print(f"Memory state loaded from {store_state_path}")

    # ── QA phase ─────────────────────────────────────────────────────────────
    APILLM.reset_token_counter()
    VLMAgent.reset_token_counter()
    qa_start = time.time()

    existing_results = {}
    _prev_qa_time = 0.0
    _prev_qa_engine_tokens = {'input': 0, 'output': 0, 'calls': 0}
    _prev_qa_vlm_tokens = {'input': 0, 'output': 0, 'calls': 0}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for r in json.load(f):
                    q = r.get("question", "")
                    if q and r.get("status") != "error" and r.get("system_answer"):
                        existing_results[q] = r
            if existing_results:
                print(f"Resume: loaded {len(existing_results)} completed QAs from {output_file}")
                eff_file_prev = os.path.join(os.path.dirname(output_file), f"{data_name}_efficiency.json")
                if os.path.exists(eff_file_prev):
                    try:
                        with open(eff_file_prev, "r", encoding="utf-8") as ef:
                            prev_eff = json.load(ef)
                        prev_qa = prev_eff.get("qa_phase", {})
                        _prev_qa_time = float(prev_qa.get("time_seconds", 0.0))
                        for k in ('input', 'output', 'calls'):
                            _prev_qa_engine_tokens[k] = int((prev_qa.get("engine_tokens") or {}).get(k, 0))
                            _prev_qa_vlm_tokens[k] = int((prev_qa.get("vlm_tokens") or {}).get(k, 0))
                        print(f"Resume: loaded previous efficiency from {eff_file_prev}")
                    except Exception as e:
                        print(f"Resume: failed to load previous efficiency ({e}), will only record incremental stats")
        except Exception:
            existing_results = {}

    _results_lock = threading.Lock()
    _all_results = list(existing_results.values())
    _new_count = 0
    _SAVE_INTERVAL = 20

    def _save_results():
        with _results_lock:
            snapshot = list(_all_results)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def process_one_qa(qa):
        nonlocal _new_count
        question = qa.get("question", "")
        if not question: return None

        if question in existing_results:
            return None

        question_image = None
        question_image_caption = None
        if qa.get("question_image"):
            question_image = resolve_media_path(qa["question_image"])
            question_image_caption = qa.get("image_caption")

        category = qa.get("type", "") or qa.get("point", "")
        format_constraint = None
        if category and category.upper() in FORMAT_CONSTRAINT_MAP:
            format_constraint = load_prompt_file(FORMAT_CONSTRAINT_MAP[category.upper()])

        qa_session_id = qa.get("session_id", "")
        if isinstance(qa_session_id, list) and qa_session_id:
            qa_session_id = qa_session_id[0]

        obs_img = None
        if question_image:
            obs_img = {'path': question_image}
            if question_image_caption: obs_img['caption'] = question_image_caption

        status = "ok"
        try:
            memory_context = memory_agent.memory_recall(question, obs_img) or []
        except Exception as e:
            print(f"Recall error: {e}"); memory_context = []; status = "error"

        retrieved_ids = []
        retrieved_texts = []
        try:
            local_data = None
            if hasattr(memory_agent.memory, 'recall_op'):
                local_data = getattr(memory_agent.memory.recall_op, '_local', None)
            if local_data is None:
                local_data = getattr(memory_agent.memory, '_local', None)
            retrieved_ids = [str(x) for x in (getattr(local_data, 'last_retrieved_ids', []) or [])]
            seen = set(); retrieved_ids = [x for x in retrieved_ids if not (x in seen or seen.add(x))]
            retrieved_texts = list(getattr(local_data, 'last_retrieved_texts', []) or [])
        except Exception: pass

        resp_img = {'path': question_image} if question_image else None
        if resp_img and question_image_caption: resp_img['caption'] = question_image_caption
        try:
            system_answer = memory_agent.response(memory_context, question, speaker_a, speaker_b, resp_img, format_constraint) or ""
        except Exception as e:
            print(f"Response error: {e}"); system_answer = ""; status = "error"

        result = {
            "sample_id": sample_id, "session_id": qa_session_id,
            "question_id": qa.get("question_id", ""),
            "speaker_a": speaker_a, "speaker_b": speaker_b,
            "question": question, "system_answer": system_answer,
            "original_answer": qa.get("answer", ""),
            "full_answer": qa.get("full_answer", ""),
            "category": category,
            "timestamp": get_timestamp(),
            "status": status,
            "has_retrieval": memory_name not in NO_RETRIEVAL_MEMORIES,
            "retrieved_ids": retrieved_ids,
            "retrieved_texts": retrieved_texts,
            "clue": qa.get("clue", []),
            "clue_texts": [round_text_map.get(cid, '') for cid in qa.get("clue", [])],
            "nuggets": qa.get("nuggets"),
        }

        with _results_lock:
            _all_results.append(result)
            _new_count += 1
        if save_results and _new_count % _SAVE_INTERVAL == 0:
            _save_results()

        return result

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(tqdm(ex.map(process_one_qa, qa_pairs), total=len(qa_pairs), desc="QA"))
    else:
        for qa in tqdm(qa_pairs, desc="QA"):
            process_one_qa(qa)

    qa_duration = time.time() - qa_start
    qa_engine_tokens = APILLM.get_token_counter()
    qa_vlm_tokens = VLMAgent.get_token_counter()

    results = _all_results
    n_error = sum(1 for r in results if r.get("status") == "error")
    if n_error:
        print(f"Warning: {n_error}/{len(results)} QAs have status=error (re-run to retry)")

    if save_results:
        _save_results()
        print(f"Results saved to {output_file} ({len(results)} total, {_new_count} new)")

    if save_efficiency:
        def _merge_tokens(prev, curr):
            return {k: prev.get(k, 0) + curr.get(k, 0) for k in ('input', 'output', 'calls')}
        merged_qa_time = _prev_qa_time + qa_duration
        merged_qa_engine = _merge_tokens(_prev_qa_engine_tokens, qa_engine_tokens)
        merged_qa_vlm = _merge_tokens(_prev_qa_vlm_tokens, qa_vlm_tokens)
        eff = {
            "sample_id": sample_id, "llm_name": llm_name, "memory_name": memory_name,
            "is_multimodal": is_multimodal, "conversation_turns": len(processed_dialogs),
            "qa_count": len(qa_pairs),
            "metadata": {
                "emb_mode": emb_mode,
                "retrieval_topk": config.get('_retrieval_topk'),
                "context_topk": config.get('_context_topk'),
                "eval_retrieval_metrics": eval_retrieval_metrics,
            },
            "memory_phase": {
                "time_seconds": memory_duration,
                "engine_tokens": memory_engine_tokens,
                "vlm_tokens": memory_vlm_tokens,
            },
            "qa_phase": {
                "time_seconds": merged_qa_time,
                "engine_tokens": merged_qa_engine,
                "vlm_tokens": merged_qa_vlm,
                "resumed_count": len(existing_results),
                "new_count": _new_count,
            },
            "total_time_seconds": memory_duration + merged_qa_time,
            "recorded_at": get_timestamp(),
        }
        eff_file = os.path.join(os.path.dirname(output_file), f"{data_name}_efficiency.json")
        with open(eff_file, "w", encoding="utf-8") as f:
            json.dump(eff, f, ensure_ascii=False, indent=2)
        print(f"Efficiency saved to {eff_file}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    load_local_env_files()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', default='')
    parser.add_argument('--memory_name', default='NaiveRAG', choices=list(MEMORY_CONFIG_MAP.keys()))
    parser.add_argument('--data_name', default=None)
    parser.add_argument('--all_datasets', action="store_true")
    parser.add_argument('--data_root', default=None,
                        help='Root dir of new-style per-sample data layout')
    parser.add_argument('--tier', default=None,
                        help='Benchmark tier file name without .json extension')
    parser.add_argument('--save_results', action="store_true")
    parser.add_argument('--save_efficiency', action="store_true")
    parser.add_argument('--eval_retrieval_metrics', action="store_true")
    parser.add_argument('--eval_topk', type=int, default=10)
    parser.add_argument('--retrieval_topk', type=int, default=None)
    parser.add_argument('--context_topk', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gemini_native', action='store_true',
                        help='Use Gemini native protocol instead of OpenAI-compatible')
    parser.add_argument('--emb_mode', default='raw', choices=['caption', 'raw'])
    parser.add_argument('--proxy_port', type=int, default=9090)
    parser.add_argument('--phase', default='both', choices=['both', 'store', 'qa'],
                        help='Run phase: both (store+QA), store (store only + serialize), qa (load state + QA only)')
    parser.add_argument('--store_llm', default=None,
                        help='Store-phase LLM name for local vLLM')
    parser.add_argument('--store_llm_base', default='http://localhost:8000/v1',
                        help='Store-phase LLM base URL')
    parser.add_argument('--store_llm_key', default='EMPTY',
                        help='Store-phase API key')
    parser.add_argument('--store_state_dir', default=None,
                        help='Directory for serialized store states')
    parser.add_argument('--result_dir', default=None,
                        help='Directory for results')
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.result_dir:
        RESULT_DIR = os.path.abspath(args.result_dir)
        print(f"[ResultDir] Overridden to: {RESULT_DIR}")

    if args.data_root:
        if not args.tier:
            print("Error: --tier is required when --data_root is specified")
            sys.exit(1)
        _DATA_ROOT = os.path.abspath(args.data_root)
        _TIER      = args.tier
        print(f"[DataLayout] new-style: data_root={_DATA_ROOT}  tier={_TIER}")
    else:
        _DATA_ROOT = None
        _TIER      = None

    # ── Resolve QA LLM config ──────────────────────────────────────────────────
    QA_APIKEY = os.getenv("LLM_API_KEY", "EMPTY")
    QA_APIBASE = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
    QA_MODEL = args.llm_name or os.getenv("LLM_MODEL", "")
    if not QA_MODEL:
        raise ValueError("Must specify --llm_name or set LLM_MODEL env var")

    if args.gemini_native:
        USE_GEMINI_NATIVE = True

    # ── Resolve Store LLM config ─────────────────────────────────────────────
    if args.store_llm:
        STORE_APIKEY = args.store_llm_key
        STORE_APIBASE = args.store_llm_base
        STORE_MODEL = args.store_llm
    else:
        STORE_APIKEY = QA_APIKEY
        STORE_APIBASE = QA_APIBASE
        STORE_MODEL = QA_MODEL

    OPENAI_APIKEY = QA_APIKEY
    OPENAI_APIBASE = QA_APIBASE
    OPENAI_MODEL = QA_MODEL

    import omnimembench.config.defaults as _defaults
    _defaults.OMNI_ENCODER['emb_mode'] = args.emb_mode

    # Caption mode: let MultiModalUtilization fold captions into text instead of passing raw media
    from omnimembench.config.memory_configs import _MM_UTILIZATION
    _MM_UTILIZATION['use_caption'] = (args.emb_mode == 'caption')

    # ── Store state directory ────────────────────────────────────────────────
    _store_state_base = args.store_state_dir or STORE_STATE_DIR
    _store_llm_tag = (args.store_llm or args.llm_name).replace('/', '_')

    def _get_store_state_path(data_name):
        if _DATA_ROOT and _TIER:
            return os.path.join(_store_state_base, _store_llm_tag,
                                f"{args.memory_name}_{args.emb_mode}", _TIER,
                                data_name, "state.pkl")
        return os.path.join(_store_state_base, _store_llm_tag,
                            f"{args.memory_name}_{args.emb_mode}",
                            data_name, "state.pkl")

    # ── External baseline config ─────────────────────────────────────────────
    EXTERNAL_MEMORY_METHODS = {'AMem', 'Mem0', 'LangMem', 'HippoRAG'}
    PROXY_BASE_URL = None

    # ── Datasets ─────────────────────────────────────────────────────────────
    import copy

    def _override_topk(d, k):
        for key in list(d.keys()):
            if key in ('topk', 'top_k', 'max_nodes') and isinstance(d[key], int):
                d[key] = k
            elif isinstance(d[key], dict):
                _override_topk(d[key], k)

    if args.all_datasets:
        datasets = get_available_datasets()
        if not datasets: print("No datasets found!"); sys.exit(1)
    elif args.data_name:
        datasets = [args.data_name]
    else:
        print("Error: --data_name or --all_datasets required"); sys.exit(1)

    def _build_config_and_run(phase_label, api_key, api_base, model_name):
        inject_llm_config(api_key, api_base, model_name)
        if args.memory_name in EXTERNAL_MEMORY_METHODS:
            inject_external_baseline_config(
                args.memory_name, api_key, api_base, model_name,
                proxy_base_url=PROXY_BASE_URL)

        cfg = copy.deepcopy(MEMORY_CONFIG_MAP[args.memory_name])
        if args.retrieval_topk is not None:
            _override_topk(cfg, args.retrieval_topk)
            print(f"Retrieval top-k overridden to {args.retrieval_topk}")
        if args.context_topk is not None and 'recall' in cfg:
            cfg['recall']['context_topk'] = args.context_topk
            print(f"Context top-k set to {args.context_topk}")
        cfg['_retrieval_topk'] = args.retrieval_topk
        cfg['_context_topk'] = args.context_topk

        IN_PROCESS_MODEL_METHODS = {'MemGPT'}
        shared_agent = None
        if args.memory_name in IN_PROCESS_MODEL_METHODS and len(datasets) > 1:
            print(f"[Optimization] Pre-creating shared DialogueAgent for {args.memory_name}")
            shared_agent = DialogueAgent(args.memory_name, cfg, model_name=model_name, seed=args.seed)

        for data_name in datasets:
            print(f"\n{'='*60}\n[{phase_label}] Processing: {data_name}\n{'='*60}")
            try:
                run_mm_bench(args.llm_name, args.memory_name, cfg,
                             data_name, args.save_results, args.save_efficiency,
                             args.eval_retrieval_metrics, args.eval_topk,
                             model_name=model_name, seed=args.seed,
                             num_workers=args.num_workers, emb_mode=args.emb_mode,
                             reuse_agent=shared_agent,
                             phase=phase_label,
                             store_state_path=_get_store_state_path(data_name),
                             store_llm_name=args.store_llm if args.store_llm else None)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"Error processing {data_name}: {e}")

    # ── Execute phases ───────────────────────────────────────────────────────
    if args.phase in ('both', 'store'):
        print(f"\n{'#'*60}\n# STORE PHASE (LLM: {STORE_MODEL})\n{'#'*60}")
        _build_config_and_run('store', STORE_APIKEY, STORE_APIBASE, STORE_MODEL)

    if args.phase in ('both', 'qa'):
        print(f"\n{'#'*60}\n# QA PHASE (LLM: {QA_MODEL})\n{'#'*60}")
        _build_config_and_run('qa', QA_APIKEY, QA_APIBASE, QA_MODEL)