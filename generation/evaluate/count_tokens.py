#!/usr/bin/env python3

import json
import re
import os
import argparse
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse
import tempfile
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoProcessor
    import torch
except ImportError:
    exit(1)

GA_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", "")
DEFAULT_BENCHMARK_ROOT = GA_ROOT / "session/output/05_benchmark_assembly"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "output/token_stats"
TMP_DIR = Path(tempfile.gettempdir()) / "benchmark_media"

def download_file(url: str, save_path: str) -> bool:
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        tmp_path = save_path + '.tmp'
        with open(tmp_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=os.path.basename(save_path)
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        os.rename(tmp_path, save_path)
        return True
    except Exception as e:
        tmp_path = save_path + '.tmp'
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

def extract_urls_from_json(data: dict) -> Dict[str, List[str]]:
    urls = {
        'image': [],
        'video': [],
        'audio': []
    }

    def find_urls(obj, parent_key=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                find_urls(v, k)
        elif isinstance(obj, list):
            for item in obj:
                find_urls(item, parent_key)
        elif isinstance(obj, str):
            if obj.startswith('http'):
                parsed = urlparse(obj)
                path_lower = parsed.path.lower()
                if path_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                    urls['image'].append(obj)
                elif path_lower.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    urls['video'].append(obj)
                elif path_lower.endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg')):
                    urls['audio'].append(obj)
                elif '/image/' in path_lower or '/images/' in path_lower:
                    urls['image'].append(obj)
                elif '/video/' in path_lower or '/videos/' in path_lower:
                    urls['video'].append(obj)
                elif '/audio/' in path_lower or '/audios/' in path_lower:
                    urls['audio'].append(obj)

    find_urls(data)

    for key in urls:
        urls[key] = list(set(urls[key]))

    return urls

def get_media_token_count(file_path: str, media_type: str, processor) -> int:
    if processor is None:
        return {'image': 256, 'video': 2048, 'audio': 512}.get(media_type, 0)

    try:
        from qwen_omni_utils import process_mm_info

        content = [{"type": "text", "text": "hi"}]
        if media_type == 'image':
            content.insert(0, {"type": "image", "image": f"file://{file_path}"})
        elif media_type == 'video':
            content.insert(0, {"type": "video", "video": f"file://{file_path}"})
        elif media_type == 'audio':
            content.insert(0, {"type": "audio", "audio": f"file://{file_path}"})

        conversation = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=[text],
            audio=audios if audios else None,
            images=images if images else None,
            videos=videos if videos else None,
            return_tensors="pt",
            padding=True,
        )

        pad_token_map = {
            'image': "<|image_pad|>",
            'video': "<|video_pad|>",
            'audio': "<|audio_pad|>",
        }
        pad_id = processor.tokenizer.convert_tokens_to_ids(pad_token_map[media_type])
        return (inputs['input_ids'] == pad_id).sum().item()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 0

def count_text_tokens(text: str, tokenizer) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def analyze_benchmark(benchmark_data: dict, tokenizer, processor) -> Dict:
    stats = {
        'total_tokens': 0,
        'text_tokens': 0,
        'image_tokens': 0,
        'video_tokens': 0,
        'audio_tokens': 0,
        'sessions': {},
        'qas': {
            'total': 0,
            'by_type': {}
        }
    }

    profile_text = json.dumps(benchmark_data.get('character_profile', {}))
    profile_tokens = count_text_tokens(profile_text, tokenizer)
    stats['text_tokens'] += profile_tokens
    print(f"Character Profile: {profile_tokens} tokens")

    sessions = benchmark_data.get('multi_session_dialogues', [])

    for session in tqdm(sessions, desc="Sessions"):
        session_id = session.get('session_id', 'unknown')
        session_stats = {
            'text_tokens': 0,
            'rounds': len(session.get('dialogues', []))
        }

        for dialogue in session.get('dialogues', []):
            user_msg = dialogue.get('user', '')
            assistant_msg = dialogue.get('assistant', '')

            user_tokens = count_text_tokens(user_msg, tokenizer)
            assistant_tokens = count_text_tokens(assistant_msg, tokenizer)

            session_stats['text_tokens'] += user_tokens + assistant_tokens

        stats['sessions'][session_id] = session_stats
        stats['text_tokens'] += session_stats['text_tokens']

    qas = benchmark_data.get('QAs', [])

    for qa in qas:
        qa_type = qa.get('type', 'unknown')
        question = qa.get('question', '')
        answer = qa.get('answer', '')

        qa_tokens = count_text_tokens(question + answer, tokenizer)
        stats['text_tokens'] += qa_tokens
        stats['qas']['total'] += qa_tokens

        if qa_type not in stats['qas']['by_type']:
            stats['qas']['by_type'][qa_type] = 0
        stats['qas']['by_type'][qa_type] += qa_tokens

    urls = extract_urls_from_json(benchmark_data)


    for media_type, url_list in urls.items():
        if not url_list:
            continue

        count = 0
        for i, url in enumerate(tqdm(url_list, desc=f"{media_type}")):
            parsed = urlparse(url)
            ext = os.path.splitext(parsed.path)[1] or f'.{media_type}'
            url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
            local_path = os.path.join(TMP_DIR, f"{media_type}_{url_hash}{ext}")

            if not os.path.exists(local_path):
                if not download_file(url, local_path):
                    continue

            tokens = get_media_token_count(local_path, media_type, processor)
            if tokens > 0:
                count += 1
            stats[f'{media_type}_tokens'] += tokens
        stats[f'{media_type}_count'] = count

    stats['total_tokens'] = (
        stats['text_tokens'] +
        stats['image_tokens'] +
        stats['video_tokens'] +
        stats['audio_tokens']
    )

    return stats

def _list_run_ids(benchmark_root: Path) -> List[str]:
    run_ids = []
    if not benchmark_root.exists():
        return run_ids
    for p in benchmark_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "benchmark.json").exists():
            run_ids.append(p.name)
    return sorted(run_ids)


def _count_one_run(run_id: str, benchmark_file: Path, out_file: Path, tokenizer, processor) -> bool:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and out_file.stat().st_size > 0:
        return True

    print("\n" + "=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Benchmark: {benchmark_file}")
    print("=" * 60)

    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    stats = analyze_benchmark(benchmark_data, tokenizer, processor)
    stats["run_id"] = run_id
    stats["benchmark_file"] = str(benchmark_file)

    tmp_path = out_file.with_suffix(out_file.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_file)
    return True


def main():
    print("=" * 60)
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Count tokens for benchmark.json across run_ids (resume supported).")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tmp_dir", type=Path, default=TMP_DIR)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        processor = None

    if args.run_id:
        run_ids = [args.run_id]
    elif args.start is not None or args.end is not None:
        if args.start is None or args.end is None:
            return
        a, b = sorted([args.start, args.end])
        run_ids = [f"{i:05d}" for i in range(a, b + 1)]
    else:
        run_ids = _list_run_ids(args.benchmark_root)

    if not run_ids:
        return


    ok = 0
    failed = 0
    for rid in run_ids:
        benchmark_file = args.benchmark_root / rid / "benchmark.json"
        if not benchmark_file.exists():
            failed += 1
            continue
        out_file = args.out_dir / rid / "token_stats.json"
        try:
            _count_one_run(rid, benchmark_file, out_file, tokenizer, processor)
            ok += 1
        except Exception as e:
            failed += 1

    summary = {
        "benchmark_root": str(args.benchmark_root),
        "out_dir": str(args.out_dir),
        "total_runs": len(run_ids),
        "ok": ok,
        "failed": failed,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
