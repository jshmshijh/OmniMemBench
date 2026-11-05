#!/usr/bin/env python3
"""
"""
import json
import time
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


API_KEY: Optional[str] = None

BASE_URL = "https://localhost:8000/v1"

MODEL = os.environ.get("LLM_MODEL", "your-model-name")

BASE_DIR = Path(__file__).resolve().parent
GA_ROOT = BASE_DIR.parent
DEFAULT_QA_ROOT = BASE_DIR / "qa"
DEFAULT_RUN_ID = "00001"
PROMPT_FILE = BASE_DIR / "prompts/caption_prompt.txt"
DEFAULT_API_KEY_FILE = BASE_DIR / ".llm_api_api_key"
DEFAULT_ENV_FILE = GA_ROOT / ".env"

MAX_WORKERS = 20

PROMPT_VERSION = "v1"
CAPTION_TEMPERATURE = 0.3


def load_prompt_template() -> str:
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()


PROMPT_TEMPLATE: Optional[str] = None


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
    url_lower = url.lower()

    if modality_type == "audio":
        if url_lower.endswith('.mp3'):
            return "audio/mp3"
        elif url_lower.endswith('.wav'):
            return "audio/wav"
        elif url_lower.endswith('.m4a'):
            return "audio/mp4"
        elif url_lower.endswith('.ogg'):
            return "audio/ogg"
        else:
            return "audio/wav"

    elif modality_type == "image":
        if url_lower.endswith('.jpg') or url_lower.endswith('.jpeg'):
            return "image/jpeg"
        elif url_lower.endswith('.png'):
            return "image/png"
        elif url_lower.endswith('.webp'):
            return "image/webp"
        elif url_lower.endswith('.gif'):
            return "image/gif"
        else:
            return "image/jpeg"

    elif modality_type == "video":
        if url_lower.endswith('.mp4'):
            return "video/mp4"
        elif url_lower.endswith('.webm'):
            return "video/webm"
        elif url_lower.endswith('.avi'):
            return "video/x-msvideo"
        elif url_lower.endswith('.mkv'):
            return "video/x-matroska"
        else:
            return "video/mp4"

    return "application/octet-stream"


def build_media_part(url: str, modality_type: str) -> Dict:
    mime_type = get_mime_type(url, modality_type)

    return {
        "fileData": {
            "mimeType": mime_type,
            "fileUri": url
        }
    }


def _is_rate_limit_error(status_code: Optional[int], response_obj: Optional[Dict[str, Any]]) -> bool:
    if status_code == 429:
        return True
    text = ""
    if response_obj is not None:
        text = json.dumps(response_obj, ensure_ascii=False).lower()
    return (
        "resource exhausted" in text
        or "limit_requests" in text
        or "http code: 429" in text
        or '"code":429' in text
        or "'code': 429" in text
    )


def _backoff_sleep(attempt: int, rate_limited: bool) -> float:
    if rate_limited:
        base = 15
        max_wait = 30
    else:
        base = 3
        max_wait = 30

    wait_time = min(base * (2 ** min(attempt, 6)), max_wait)
    jitter = random.uniform(0, min(10.0, wait_time * 0.1))
    total_wait = wait_time + jitter
    time.sleep(total_wait)
    return total_wait


def call_gemini_for_caption(
    url: str,
    modality_type: str,
    model: str = MODEL,
    temperature: float = CAPTION_TEMPERATURE,
    max_retries: int = 5
) -> Tuple[bool, str]:
    if not API_KEY:
        return False, "Missing API key. Provide --api_key or set LLM_API_KEY."
    if not PROMPT_TEMPLATE:
        return False, "Missing prompt template. Provide --prompt_file or check PROMPT_FILE."

    prompt = PROMPT_TEMPLATE.replace("{audio/image/video}", modality_type)
    prompt = prompt.replace("{multimodal_content}", f"[{modality_type.upper()} from URL]")

    max_retries = max(1, int(max_retries))
    attempt = 0
    last_error = "Unknown error"
    while attempt < max_retries:
        try:
            attempt += 1
            media_part = build_media_part(url, modality_type)

            payload = {
                "model": model,
                "contents": [
                    {
                        "parts": [
                            media_part,
                            {
                                "text": prompt
                            }
                        ],
                        "role": "user"
                    }
                ]
            }

            curl_command = [
                "curl",
                "--location",
                f"{BASE_URL}/chat/completions",
                "--header",
                "Content-Type: application/json",
                "--header",
                f"Authorization: Bearer {API_KEY}",
                "--data",
                json.dumps(payload),
                "--silent",
                "--show-error",
                "--max-time",
                "120",
                "--write-out",
                "\nHTTPSTATUS:%{http_code}",
            ]

            run_result = subprocess.run(
                curl_command,
                capture_output=True,
                text=True,
                timeout=130,
            )

            if run_result.returncode != 0:
                error_msg = f"curl failed: {run_result.stderr}"
                last_error = error_msg
                if attempt < max_retries:
                    waited = _backoff_sleep(attempt, rate_limited=False)
                    if attempt % 10 == 0:
                        print(f"[retry][{modality_type}] attempt={attempt} wait={waited:.1f}s url={url} reason={error_msg[:200]}", flush=True)
                    continue
                return False, error_msg

            stdout = run_result.stdout
            if "\nHTTPSTATUS:" not in stdout:
                error_msg = f"Unexpected curl output (missing HTTPSTATUS): {stdout[:2000]}"
                last_error = error_msg
                if attempt < max_retries:
                    waited = _backoff_sleep(attempt, rate_limited=False)
                    if attempt % 10 == 0:
                        print(f"[retry][{modality_type}] attempt={attempt} wait={waited:.1f}s url={url} reason={error_msg[:200]}", flush=True)
                    continue
                return False, error_msg

            body, status_line = stdout.rsplit("\nHTTPSTATUS:", 1)
            status_code = int(status_line.strip()) if status_line.strip().isdigit() else None

            try:
                result = json.loads(body)
            except json.JSONDecodeError as e:
                error_msg = f"JSON parse error: {str(e)}; body head: {body[:1000]}"
                last_error = error_msg
                if attempt < max_retries:
                    waited = _backoff_sleep(attempt, rate_limited=False)
                    if attempt % 10 == 0:
                        print(f"[retry][{modality_type}] attempt={attempt} wait={waited:.1f}s url={url} reason={error_msg[:200]}", flush=True)
                    continue
                return False, error_msg

            if status_code and status_code >= 400:
                error_msg = f"HTTP {status_code}: {result}"
                last_error = error_msg
                if attempt < max_retries:
                    waited = _backoff_sleep(attempt, rate_limited=_is_rate_limit_error(status_code, result))
                    if attempt % 10 == 0:
                        print(f"[retry][{modality_type}] attempt={attempt} wait={waited:.1f}s url={url} reason={error_msg[:200]}", flush=True)
                    continue
                return False, error_msg

            if "candidates" in result and len(result["candidates"]) > 0:
                caption = result["candidates"][0]["content"]["parts"][0]["text"]

                if not caption:
                    error_msg = "Empty caption from API"
                    last_error = error_msg
                    if attempt < max_retries:
                        waited = _backoff_sleep(attempt, rate_limited=False)
                        if attempt % 10 == 0:
                            print(f"[retry][{modality_type}] attempt={attempt} wait={waited:.1f}s url={url} reason={error_msg}", flush=True)
                        continue
                    return False, error_msg

                caption = caption.strip()
                if not caption:
                    error_msg = "Blank caption after strip"
                    last_error = error_msg
                    if attempt < max_retries:
                        waited = _backoff_sleep(attempt, rate_limited=False)
                        if attempt % 10 == 0:
                            print(f"[retry][{modality_type}] attempt={attempt} wait={waited:.1f}s url={url} reason={error_msg}", flush=True)
                        continue
                    return False, error_msg

                return True, caption
            else:
                error_msg = f"Unexpected response format: {result}"
                last_error = error_msg
                if attempt < max_retries:
                    waited = _backoff_sleep(attempt, rate_limited=False)
                    if attempt % 10 == 0:
                        print(f"[retry][{modality_type}] attempt={attempt} wait={waited:.1f}s url={url} reason={error_msg[:200]}", flush=True)
                    continue
                return False, error_msg

        except subprocess.TimeoutExpired:
            error_msg = "Request timeout"
            last_error = error_msg
            if attempt < max_retries:
                waited = _backoff_sleep(attempt, rate_limited=False)
                if attempt % 10 == 0:
                    print(f"[retry][{modality_type}] attempt={attempt} wait={waited:.1f}s url={url} reason={error_msg}", flush=True)
                continue
            return False, error_msg

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            last_error = error_msg
            if attempt < max_retries:
                waited = _backoff_sleep(attempt, rate_limited=False)
                if attempt % 10 == 0:
                    print(f"[retry][{modality_type}] attempt={attempt} wait={waited:.1f}s url={url} reason={error_msg[:200]}", flush=True)
                continue
            return False, error_msg

    return False, last_error


def caption_task(url: str, modality_type: str, model: str) -> Tuple[str, str, bool, str]:
    success, caption = call_gemini_for_caption(url, modality_type, model=model)
    return (url, modality_type, success, caption)


def convert_modalities_format(
    old_modalities: Dict[str, List[str]],
    model: str = MODEL
) -> Tuple[Dict[str, List[Dict]], List[Tuple[str, str]]]:
    new_modalities = {}
    tasks = []

    for modality_type, urls in old_modalities.items():
        new_modalities[modality_type] = []

        for url in urls:
            new_modalities[modality_type].append({
                "url": url,
                "captions": []
            })
            tasks.append((url, modality_type))

    return new_modalities, tasks


def _has_target_caption(captions: Any, model: str) -> bool:
    if not isinstance(captions, list):
        return False
    for cap in captions:
        if not isinstance(cap, dict):
            continue
        if cap.get("model") == model and cap.get("prompt_version") == PROMPT_VERSION and cap.get("caption"):
            return True
    return False


def normalize_and_merge_modalities(
    input_modalities: Any,
    existing_modalities: Optional[Any],
    model: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Tuple[str, str]]]:
    merged: Dict[str, List[Dict[str, Any]]] = {}
    tasks: List[Tuple[str, str]] = []

    existing_index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if isinstance(existing_modalities, dict):
        for modality_type, items in existing_modalities.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and "url" in item:
                    existing_index.setdefault(modality_type, {})[str(item["url"])] = item

    def add_url(modality_type: str, url: str) -> None:
        existing_item = existing_index.get(modality_type, {}).get(url)
        captions = []
        if isinstance(existing_item, dict):
            maybe_caps = existing_item.get("captions", [])
            if isinstance(maybe_caps, list):
                captions = maybe_caps
        merged.setdefault(modality_type, []).append({"url": url, "captions": captions})

        if not _has_target_caption(captions, model):
            tasks.append((url, modality_type))

    seen: Dict[str, Set[str]] = {}

    if isinstance(input_modalities, dict):
        for modality_type, items in input_modalities.items():
            if isinstance(items, list) and items and isinstance(items[0], dict):
                for item in items:
                    if not isinstance(item, dict) or "url" not in item:
                        continue
                    url = str(item["url"])
                    if url in seen.setdefault(modality_type, set()):
                        continue
                    seen[modality_type].add(url)
                    add_url(modality_type, url)
            elif isinstance(items, list):
                for url in items:
                    if not url:
                        continue
                    url_str = str(url)
                    if url_str in seen.setdefault(modality_type, set()):
                        continue
                    seen[modality_type].add(url_str)
                    add_url(modality_type, url_str)

    if not merged and isinstance(existing_modalities, dict):
        for modality_type, items in existing_modalities.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "url" not in item:
                    continue
                url = str(item["url"])
                if url in seen.setdefault(modality_type, set()):
                    continue
                seen[modality_type].add(url)
                add_url(modality_type, url)

    return merged, tasks


def update_caption_in_modalities(
    modalities: Dict[str, List[Dict]],
    url: str,
    modality_type: str,
    caption: str,
    model: str = MODEL
):
    if modality_type in modalities:
        for item in modalities[modality_type]:
            if item["url"] == url:
                existing_caption = None
                for cap in item["captions"]:
                    if cap["model"] == model and cap["prompt_version"] == PROMPT_VERSION:
                        existing_caption = cap
                        break

                if existing_caption:
                    existing_caption["caption"] = caption
                else:
                    item["captions"].append({
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                        "caption": caption
                    })
                break


def _atomic_write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def process_dataset(
    input_path: Path,
    output_path: Path,
    model: str = MODEL,
    resume: bool = True,
    save_every: int = 50,
) -> Tuple[int, int]:
    print(f"\n{'='*60}")
    print(f"{'='*60}")

    all_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line.strip()))

    existing_data = {}
    if resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                existing_data[item['id']] = item
    else:
        pass

    all_tasks = []  # [(data_id, url, modality_type)]
    processed_data = []

    for data in all_data:
        data_id = data['id']
        existing_item = existing_data.get(data_id)
        existing_modalities = existing_item.get("modalities") if isinstance(existing_item, dict) else None

        new_modalities, tasks = normalize_and_merge_modalities(
            input_modalities=data.get("modalities"),
            existing_modalities=existing_modalities,
            model=model,
        )
        data["modalities"] = new_modalities
        processed_data.append(data)

        for url, modality_type in tasks:
            all_tasks.append((data_id, url, modality_type))


    if len(all_tasks) == 0:
        _atomic_write_jsonl(output_path, processed_data)
        return 0, 0

    failed_tasks = []
    processed_by_id = {d["id"]: d for d in processed_data if isinstance(d, dict) and "id" in d}
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(caption_task, url, modality_type, model): (data_id, url, modality_type)
            for data_id, url, modality_type in all_tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating captions"):
            data_id, url, modality_type = futures[future]
            returned_url, returned_modality, success, result = future.result()

            if success:
                data_obj = processed_by_id.get(data_id)
                if data_obj is not None:
                    update_caption_in_modalities(
                        data_obj["modalities"],
                        returned_url,
                        returned_modality,
                        result,
                        model,
                    )
            else:
                failed_tasks.append((data_id, url, modality_type, result))
                print(f"\n  Warning: Failed to generate caption for {url}: {result}")

            completed += 1
            if save_every > 0 and (completed % save_every == 0):
                _atomic_write_jsonl(output_path, processed_data)

    succeeded = len(all_tasks) - len(failed_tasks)
    if failed_tasks:
        pass
    _atomic_write_jsonl(output_path, processed_data)


    failed_log = output_path.parent / f"{output_path.stem}_failed.jsonl"
    if failed_tasks:
        with open(failed_log, 'w', encoding='utf-8') as f:
            for data_id, url, modality_type, error in failed_tasks:
                f.write(json.dumps({
                    "data_id": data_id,
                    "url": url,
                    "modality_type": modality_type,
                    "error": error
                }, ensure_ascii=False) + '\n')
    elif failed_log.exists():
        failed_log.unlink()

    return succeeded, len(failed_tasks)


def main():
    global MAX_WORKERS
    global API_KEY
    global PROMPT_TEMPLATE
    global PROMPT_FILE

    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(
        "--input",
        type=str,
        help=""
    )
    parser.add_argument(
        "--output",
        type=str,
        help=""
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_QA_ROOT,
        help=f"...",
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=None,
        help="",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=DEFAULT_RUN_ID,
        help=f"...",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--api_key_file",
        type=Path,
        default=DEFAULT_API_KEY_FILE,
        help=f"...",
    )
    parser.add_argument(
        "--env_file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f".env ...",
    )
    parser.add_argument(
        "--prompt_file",
        type=Path,
        default=PROMPT_FILE,
        help=f"caption prompt...",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=""
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"..."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=""
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"..."
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="",
    )

    args = parser.parse_args()

    MAX_WORKERS = args.workers
    load_env_file(args.env_file)
    API_KEY = args.api_key or os.environ.get("LLM_API_KEY")
    if not API_KEY and args.api_key_file and args.api_key_file.exists():
        key_text = args.api_key_file.read_text(encoding="utf-8").strip()
        if key_text:
            API_KEY = key_text

    PROMPT_FILE = args.prompt_file
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read()

    input_root = args.input_root or args.output_root
    input_dir = input_root / args.run_id
    output_dir = args.output_root / args.run_id

    print("=" * 60)
    print("=" * 60)

    if not API_KEY:
        return

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            return

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_captioned.jsonl"


        dataset_files = [(input_path, output_path)]

    elif args.datasets:
        dataset_files = []
        for name in args.datasets:
            input_file = input_dir / f"{name}.jsonl"
            if input_file.exists():
                output_file = output_dir / input_file.name.replace("_processed.jsonl", "_captioned.jsonl")
                dataset_files.append((input_file, output_file))
            else:
                pass

        if not dataset_files:
            return
    else:
        default_input = input_dir / "sampled_data.jsonl"
        if default_input.exists():
            dataset_files = [(default_input, output_dir / f"{default_input.stem}_captioned.jsonl")]
        else:
            input_files = sorted(input_dir.glob("*_processed.jsonl"))
            dataset_files = [
                (input_file, output_dir / input_file.name.replace("_processed.jsonl", "_captioned.jsonl"))
                for input_file in input_files
            ]

    for input_file, output_file in dataset_files:
        print(f"  - {input_file.name} -> {output_file.name}")

    total_start = time.time()

    total_failed_tasks = 0
    dataset_errors = 0
    for input_file, output_file in dataset_files:
        try:
            _, failed_tasks = process_dataset(
                input_file,
                output_file,
                model=args.model,
                resume=not args.no_resume,
                save_every=args.save_every,
            )
            total_failed_tasks += failed_tasks
        except Exception as e:
            import traceback
            traceback.print_exc()
            dataset_errors += 1
            continue

    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"{'='*60}")

    if total_failed_tasks > 0 or dataset_errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
