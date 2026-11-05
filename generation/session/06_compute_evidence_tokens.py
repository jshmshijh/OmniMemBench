#!/usr/bin/env python3

import json, os, argparse, hashlib, tempfile, requests
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse

try:
    from transformers import AutoTokenizer, AutoProcessor
except ImportError:
    raise SystemExit("pip install transformers torch")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "")
TMP_DIR = Path(tempfile.gettempdir()) / "distractor_media"


def _download(url: str, path: str) -> bool:
    if os.path.exists(path):
        return True
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        os.rename(tmp, path)
        return True
    except Exception as e:
        if os.path.exists(path + ".tmp"):
            os.remove(path + ".tmp")
        return False


def _detect_type(url: str) -> str:
    path = urlparse(url).path.lower()
    if any(path.endswith(e) for e in (".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return "image"
    if any(path.endswith(e) for e in (".mp4", ".avi", ".mov", ".mkv", ".webm")):
        return "video"
    if any(path.endswith(e) for e in (".mp3", ".wav", ".flac", ".m4a", ".ogg")):
        return "audio"
    for kw, t in [("/image", "image"), ("/video", "video"), ("/audio", "audio")]:
        if kw in path:
            return t
    return ""


def _local_path(url: str, tmp_dir: Path = None) -> str:
    base = tmp_dir if tmp_dir is not None else TMP_DIR
    ext = os.path.splitext(urlparse(url).path)[1] or ".bin"
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return str(base / f"{h}{ext}")


def count_text_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_mm_tokens(file_path: str, media_type: str, processor) -> int:
    if processor is None:
        return 0
    try:
        from qwen_omni_utils import process_mm_info
        content = [{"type": "text", "text": "hi"}]
        content.insert(0, {"type": media_type, media_type: f"file://{file_path}"})
        conv = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = processor(text=[text],
                           audio=audios or None, images=images or None, videos=videos or None,
                           return_tensors="pt", padding=True)
        pad_map = {"image": "<|image_pad|>", "video": "<|video_pad|>", "audio": "<|audio_pad|>"}
        pad_id = processor.tokenizer.convert_tokens_to_ids(pad_map[media_type])
        return (inputs["input_ids"] == pad_id).sum().item()
    except Exception as e:
        return 0


def count_session_tokens(session: dict, tokenizer, processor, tmp_dir: Path = None) -> dict:
    text_tok = 0
    mm_tok = 0
    seen_urls = set()
    for rnd in session.get("dialogues", []):
        text_tok += count_text_tokens(rnd.get("user", ""), tokenizer)
        text_tok += count_text_tokens(rnd.get("assistant", ""), tokenizer)
        for mm in rnd.get("mm", []):
            url = mm.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            mtype = mm.get("type", "") or _detect_type(url)
            if not mtype:
                continue
            lp = _local_path(url, tmp_dir)
            if _download(url, lp):
                mm_tok += count_mm_tokens(lp, mtype, processor)
    return {"text_tokens": text_tok, "mm_tokens": mm_tok, "total_tokens": text_tok + mm_tok}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--benchmark_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tmp_dir", type=Path, default=TMP_DIR)
    args = parser.parse_args()

    benchmark_file = args.benchmark_file or (
        BASE_DIR / "output" / "05_benchmark_assembly" / args.run_id / "benchmark.json")
    output_dir = args.output_dir or (BASE_DIR / "output" / "06_evidence_tokens" / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / "evidence_token_count.json"
    if out_file.exists():
        return

    benchmark = json.loads(benchmark_file.read_text("utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        processor = None

    # profile tokens
    profile_tok = count_text_tokens(json.dumps(benchmark.get("character_profile", {})), tokenizer)

    # QA tokens
    qa_tok = 0
    for qa in benchmark.get("QAs", []):
        qa_tok += count_text_tokens(qa.get("question", "") + qa.get("answer", ""), tokenizer)

    # session tokens
    sessions = benchmark.get("multi_session_dialogues", [])
    details = []
    for s in tqdm(sessions, desc="Sessions"):
        info = count_session_tokens(s, tokenizer, processor, args.tmp_dir)
        info["session_id"] = s.get("session_id", "")
        details.append(info)

    session_total = sum(d["total_tokens"] for d in details)
    result = {
        "run_id": args.run_id,
        "profile_tokens": profile_tok,
        "qa_tokens": qa_tok,
        "session_tokens": session_total,
        "total_tokens": session_total,
        "session_details": details,
    }
    tmp = out_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
    os.replace(tmp, out_file)


if __name__ == "__main__":
    main()