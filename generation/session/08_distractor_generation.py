#!/usr/bin/env python3

import os
import json, os, sys, re, time, argparse, random, hashlib, tempfile, requests
from pathlib import Path
from typing import Dict, List, Tuple, Set
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent / "preprocess"))
from call_gemini import call_gemini, call_gemini_multimodal

BASE_DIR = Path(__file__).resolve().parent
GA_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "")
DEFAULT_INPUT_DIR = GA_ROOT / "preprocess" / "datasets" / "scenario_tagged"
TMP_DIR = Path(tempfile.gettempdir()) / "distractor_media"

# left room for QA and output token
TIER_TOKENS = {
    "128k": 108_000,
    "256k": 230_000,
    "512k": 460_000,
    "1m":   950_000,
    "10m": 9_500_000,
}
TIER_ORDER = ["128k", "256k", "512k", "1m", "10m"]

MIME_MAP = {"image": "image/jpeg", "audio": "audio/mpeg", "video": "video/mp4"}

CAPTION_PROMPT = (
    "Describe this {media_type} in detail. Focus on what is visually/audibly present. "
    "Be specific about objects, actions, colors, sounds, text, layout, and any other notable details. "
    "Output ONLY the description, no preamble."
)


def _get_mm_url(item) -> str:
    return item if isinstance(item, str) else item.get("url", "")


def _normalize_mm_item(item) -> dict:
    if isinstance(item, str):
        return {"url": item, "captions": []}
    return item


# Token counting

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
    except Exception:
        if os.path.exists(path + ".tmp"):
            os.remove(path + ".tmp")
        return False


def _local_path(url: str, tmp_dir: Path) -> str:
    ext = os.path.splitext(urlparse(url).path)[1] or ".bin"
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return str(tmp_dir / f"{h}{ext}")


def _count_text_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _count_audio_tokens_direct(file_path: str) -> int:
    try:
        import soundfile as sf
        info = sf.info(file_path)
        duration_sec = info.frames / info.samplerate
        return max(int(duration_sec * 25 + 0.5), 1)
    except Exception as e:
        return 100


def _count_mm_tokens(file_path: str, media_type: str, processor) -> int:
    if media_type == "audio":
        return _count_audio_tokens_direct(file_path)
    if processor is None:
        return 0
    try:
        from qwen_omni_utils import process_mm_info
        content = [{"type": media_type, media_type: f"file://{file_path}"}, {"type": "text", "text": "hi"}]
        conv = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = processor(text=[text], audio=audios or None, images=images or None,
                           videos=videos or None, return_tensors="pt", padding=True)
        pad_map = {"image": "<|image_pad|>", "video": "<|video_pad|>", "audio": "<|audio_pad|>"}
        pad_id = processor.tokenizer.convert_tokens_to_ids(pad_map[media_type])
        return (inputs["input_ids"] == pad_id).sum().item()
    except Exception as e:
        return 0


def count_session_tokens(session: dict, registry: dict, tokenizer, processor, tmp_dir: Path) -> int:
    total = 0
    seen_urls = set()
    for rnd in session.get("dialogues", []):
        total += _count_text_tokens(rnd.get("user", ""), tokenizer)
        total += _count_text_tokens(rnd.get("assistant", ""), tokenizer)
        for mm in rnd.get("mm", []):
            mm_id = mm.get("mm_id", "")
            entry = registry.get(mm_id, {})
            url = entry.get("url", "")
            mtype = mm.get("type", "") or entry.get("type", "")
            if not url or not mtype or url in seen_urls:
                continue
            seen_urls.add(url)
            lp = _local_path(url, tmp_dir)
            if _download(url, lp):
                total += _count_mm_tokens(lp, mtype, processor)
    return total


def load_evidence_ids(run_id: str) -> Set[str]:
    sampled = GA_ROOT / "filter" / "qa" / run_id / "sampled_data.jsonl"
    ids = set()
    if sampled.exists():
        for line in sampled.read_text("utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def load_pool(input_dir: Path, exclude_ids: Set[str], seed: int) -> List[dict]:
    by_category: Dict[str, List[dict]] = {}
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        for line in jsonl_file.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj["id"] in exclude_ids:
                continue
            mods = obj.get("modalities", {})
            if not any(
                any(_get_mm_url(item) for item in mods.get(t, []))
                for t in ("image", "audio", "video")
            ):
                continue
            cats = obj.get("categories", ["other"])
            cat = cats[0] if cats else "other"
            by_category.setdefault(cat, []).append(obj)

    rng = random.Random(seed)
    for v in by_category.values():
        rng.shuffle(v)

    result = []
    cats = sorted(by_category.keys())
    idx = {c: 0 for c in cats}
    while True:
        added = False
        for c in cats:
            if idx[c] < len(by_category[c]):
                result.append(by_category[c][idx[c]])
                idx[c] += 1
                added = True
        if not added:
            break
    return result


def extract_mm_from_records(records: List[dict]) -> List[dict]:
    items = []
    for rec in records:
        mods = rec.get("modalities", {})
        for mtype in ("image", "audio", "video"):
            for raw_item in mods.get(mtype, []):
                item = _normalize_mm_item(raw_item)
                url = item.get("url", "")
                if url:
                    items.append({
                        "url": url,
                        "type": mtype,
                        "captions": item.get("captions", []),
                        "source_id": rec["id"],
                    })
    return items


# Caption

def generate_caption(url: str, media_type: str, model: str) -> str:
    prompt = CAPTION_PROMPT.format(media_type=media_type)
    mime = MIME_MAP.get(media_type, "application/octet-stream")
    media = [{"url": url, "mime_type": mime, "modality_type": media_type}]
    success, response = call_gemini_multimodal(
        prompt=prompt, media=media, model=model,
        temperature=0.2, max_tokens=1024, max_retries=5, max_time=120,
    )
    return response.strip() if success else ""


def batch_caption(mm_items: List[dict], model: str, max_workers: int) -> None:
    need = [(i, item) for i, item in enumerate(mm_items) if not item.get("captions")]
    if not need:
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(generate_caption, item["url"], item["type"], model): i
            for i, item in need
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Caption", leave=False):
            idx = future_map[future]
            try:
                caption = future.result()
            except Exception as e:
                caption = ""
            mm_items[idx]["captions"] = [{"model": model, "prompt_version": "v1", "caption": caption}]


def extract_json_from_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{.*\}', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError("...")


def format_mm_assets(mm_items: List[dict], registry: dict) -> str:
    lines = []
    for m in mm_items:
        caption = registry.get(m["mm_id"], {}).get("caption", "")
        lines.append(f"- mm_id: {m['mm_id']}, type: {m['type']}, caption: {caption}")
    return "\n".join(lines)


def format_forbidden_questions(benchmark: dict) -> str:
    questions = []
    for qa in benchmark.get("QAs", []):
        q = qa.get("question", "").strip()
        if q:
            questions.append(f"- {q}")
    return "\n".join(questions) if questions else "(none)"


def assign_distractor_dates(evidence_sessions: List[dict], count: int, seed: int) -> List[str]:
    from datetime import datetime, timedelta
    dates = sorted(s.get("date", "") for s in evidence_sessions if s.get("date"))
    if len(dates) < 2:
        base = datetime(2023, 1, 1)
        return [(base + timedelta(days=i * 3)).strftime("%Y-%m-%d") for i in range(count)]

    gaps = []
    for i in range(len(dates) - 1):
        d1 = datetime.strptime(dates[i], "%Y-%m-%d")
        d2 = datetime.strptime(dates[i + 1], "%Y-%m-%d")
        gap_days = (d2 - d1).days
        if gap_days > 1:
            for offset in range(1, gap_days):
                gaps.append((d1 + timedelta(days=offset)).strftime("%Y-%m-%d"))

    first = datetime.strptime(dates[0], "%Y-%m-%d")
    last = datetime.strptime(dates[-1], "%Y-%m-%d")
    for offset in range(1, 31):
        gaps.insert(0, (first - timedelta(days=offset)).strftime("%Y-%m-%d"))
        gaps.append((last + timedelta(days=offset)).strftime("%Y-%m-%d"))

    rng = random.Random(seed)
    rng.shuffle(gaps)
    return [gaps[i % len(gaps)] for i in range(count)]


def generate_distractor_session(
    session_id: str, date: str, mm_items: List[dict],
    persona: dict, event_graph: dict, prompt_template: str,
    forbidden_questions_str: str,
    registry: dict = None,
    max_retries: int = 5,
) -> Tuple[bool, dict]:
    mm_assets_str = format_mm_assets(mm_items, registry or {})
    event_graph_str = json.dumps(event_graph.get("graph", []), indent=2, ensure_ascii=False)

    prompt = prompt_template.replace("{persona}",
        f"Name & Role: {persona.get('persona_summary', '')}\n"
        f"Conversation Style: {persona.get('conversation_style', '')}")
    prompt = prompt.replace("{session_id}", session_id)
    prompt = prompt.replace("{date}", date)
    prompt = prompt.replace("{mm_assets}", mm_assets_str)
    prompt = prompt.replace("{n_assets}", str(len(mm_items)))
    prompt = prompt.replace("{event_graph}", event_graph_str)
    prompt = prompt.replace("{forbidden_questions}", forbidden_questions_str)

    for attempt in range(1, max_retries + 1):
        try:
            success, response = call_gemini(
                prompt=prompt, model=os.environ.get("LLM_MODEL", "your-model-name"),
                temperature=0.8, max_tokens=32768, max_retries=5)
            if not success:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return False, {"error": f"API ..."}

            result = extract_json_from_response(response)

            expected_ids = {m["mm_id"] for m in mm_items}
            got_ids = set()
            for rnd in result.get("dialogues", []):
                for mm in rnd.get("mm", []):
                    got_ids.add(mm.get("mm_id", ""))
            missing = expected_ids - got_ids
            if missing:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue

            result["session_id"] = session_id
            result["date"] = date
            result["is_distractor"] = True
            return True, result

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return False, {"error": str(e)}

    return False, {"error": "max retries exceeded"}


# IO helpers

def _atomic_json_write(data: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
    os.replace(tmp, path)


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--target_tier", default="10m", choices=TIER_ORDER)
    parser.add_argument("--mm_per_session_min", type=int, default=10)
    parser.add_argument("--mm_per_session_max", type=int, default=20)
    parser.add_argument("--batch_sessions", type=int, default=10,
                        help="")
    parser.add_argument("--max_workers", type=int, default=50)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--caption_model", default=os.environ.get("LLM_MODEL", "your-model-name"))
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--template_file", type=Path,
                        default=BASE_DIR / "prompt" / "08_distractor_prompt_template.txt")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--tmp_dir", type=Path, default=TMP_DIR)
    parser.add_argument("--max_total_videos", type=int, default=None,
                        help="")
    args = parser.parse_args()

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    target_tokens = TIER_TOKENS[args.target_tier]

    output_dir = BASE_DIR / "output" / "08_distractor_gen" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_dir = BASE_DIR / "output" / "07_distractor_pool" / args.run_id
    registry_dir.mkdir(parents=True, exist_ok=True)

    evidence_token_file = BASE_DIR / "output" / "06_evidence_tokens" / args.run_id / "evidence_token_count.json"
    persona_file = BASE_DIR.parent / "framework" / "output" / "01_persona_gen_caption" / args.run_id / "generated_persona_caption.json"
    events_file = BASE_DIR.parent / "framework" / "output" / "02_event_gen_caption" / args.run_id / "generated_events.json"
    benchmark_file = BASE_DIR / "output" / "05_benchmark_assembly" / args.run_id / "benchmark.json"

    sessions_file = output_dir / "distractor_sessions.jsonl"
    checkpoint_file = output_dir / "distractor_checkpoints.json"
    registry_file = registry_dir / "distractor_mm_registry.json"

    evidence_tokens = json.loads(evidence_token_file.read_text("utf-8"))
    base_tokens = evidence_tokens["total_tokens"]
    persona = json.loads(persona_file.read_text("utf-8"))
    event_graph = json.loads(events_file.read_text("utf-8"))
    prompt_template = args.template_file.read_text("utf-8")
    benchmark = json.loads(benchmark_file.read_text("utf-8"))
    evidence_sessions = benchmark.get("multi_session_dialogues", [])

    forbidden_questions_str = format_forbidden_questions(benchmark)

    evidence_video_count = 0
    if args.max_total_videos is not None:
        for sess in evidence_sessions:
            for rnd in sess.get("dialogues", []):
                for mm in rnd.get("mm", []):
                    if mm.get("type") == "video":
                        evidence_video_count += 1

    # ── Resume: registry ──
    registry = {}
    if registry_file.exists():
        registry = json.loads(registry_file.read_text("utf-8"))

    # ── Resume: sessions ──
    existing_sessions = []
    used_source_ids = set()
    if sessions_file.exists():
        for line in sessions_file.read_text("utf-8").splitlines():
            if line.strip():
                existing_sessions.append(json.loads(line))

    for entry in registry.values():
        sid = entry.get("source_id", "")
        if sid:
            used_source_ids.add(sid)

    # ── Resume: checkpoints ──
    checkpoints = {}
    if checkpoint_file.exists():
        checkpoints = json.loads(checkpoint_file.read_text("utf-8"))

    for tier in TIER_ORDER:
        if tier not in checkpoints and base_tokens >= TIER_TOKENS[tier]:
            checkpoints[tier] = {
                "distractor_count": 0,
                "total_tokens": base_tokens,
                "note": "evidence already exceeds tier",
            }
            print(f"  ★ Tier {tier}: evidence ({base_tokens}) >= {TIER_TOKENS[tier]}, distractor_count=0")
    _atomic_json_write(checkpoints, checkpoint_file)

    # ── Tokenizer ──
    from transformers import AutoTokenizer, AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        processor = None

    distractor_tokens = sum(s.get("_tokens", 0) for s in existing_sessions)
    current_total = base_tokens + distractor_tokens

    if current_total >= target_tokens:
        return

    exclude_ids = load_evidence_ids(args.run_id) | used_source_ids
    seed = json.loads((BASE_DIR.parent / "_pipeline_logs" / args.run_id / "state.json").read_text("utf-8")).get("seed", 42)
    pool_records = load_pool(args.input_dir, exclude_ids, seed)
    pool_mm = extract_mm_from_records(pool_records)

    distractor_video_used = 0
    if args.max_total_videos is not None:
        for sess in existing_sessions:
            for rnd in sess.get("dialogues", []):
                for mm in rnd.get("mm", []):
                    if mm.get("type") == "video":
                        distractor_video_used += 1
        video_budget = args.max_total_videos - evidence_video_count

    max_possible = len(pool_mm) // args.mm_per_session_min
    dates = assign_distractor_dates(evidence_sessions, max_possible, seed)

    rng = random.Random(seed + len(existing_sessions))
    mm_counter = len(registry)
    session_counter = len(existing_sessions)
    pool_cursor = 0

    with open(sessions_file, "a", encoding="utf-8") as f_out:
        while current_total < target_tokens:
            batch_mm_counts = []
            total_in_batch = 0
            for _ in range(args.batch_sessions):
                n = rng.randint(args.mm_per_session_min, args.mm_per_session_max)
                if pool_cursor + total_in_batch + n > len(pool_mm):
                    break
                batch_mm_counts.append(n)
                total_in_batch += n

            if not batch_mm_counts:
                remaining = len(pool_mm) - pool_cursor
                if remaining >= args.mm_per_session_min:
                    batch_mm_counts = [remaining]
                    total_in_batch = remaining
                else:
                    break

            batch_mm = pool_mm[pool_cursor:pool_cursor + total_in_batch]
            pool_cursor += total_in_batch

            for item in batch_mm:
                item["mm_id"] = f"D{mm_counter:04d}_{item['type']}_0"
                mm_counter += 1

            batch_caption(batch_mm, args.caption_model, args.max_workers)

            for item in batch_mm:
                caption = item["captions"][0].get("caption", "") if item.get("captions") else ""
                registry[item["mm_id"]] = {
                    "url": item["url"],
                    "type": item["type"],
                    "caption": caption,
                    "source_id": item["source_id"],
                }
            _atomic_json_write(registry, registry_file)

            mm_offset = 0
            for n_mm in batch_mm_counts:
                if current_total >= target_tokens:
                    break

                raw_session_mm = batch_mm[mm_offset:mm_offset + n_mm]
                mm_offset += n_mm

                if args.max_total_videos is not None:
                    session_mm = []
                    for item in raw_session_mm:
                        if item["type"] == "video":
                            if distractor_video_used >= video_budget:
                                continue
                            distractor_video_used += 1
                        session_mm.append(item)
                else:
                    session_mm = raw_session_mm

                if not session_mm:
                    continue

                session_id = f"D{session_counter + 1:03d}"
                date = dates[session_counter] if session_counter < len(dates) else "2024-06-01"


                ok, result = generate_distractor_session(
                    session_id, date, session_mm, persona, event_graph,
                    prompt_template, forbidden_questions_str,
                    registry=registry, max_retries=args.max_retries)

                if not ok:
                    print(f"  ✗ {result.get('error', '?')}")
                    continue

                tok = count_session_tokens(result, registry, tokenizer, processor, args.tmp_dir)
                result["_tokens"] = tok
                current_total += tok
                session_counter += 1

                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

                print(f"  ✓ {len(result.get('dialogues', []))} rounds, {tok} tokens, ...")

                for tier in TIER_ORDER:
                    if tier not in checkpoints and current_total >= TIER_TOKENS[tier]:
                        checkpoints[tier] = {
                            "distractor_count": session_counter,
                            "total_tokens": current_total,
                        }
                        _atomic_json_write(checkpoints, checkpoint_file)
                        print(f"  ★ Tier {tier}: {session_counter} sessions, {current_total} tokens")

    _atomic_json_write(checkpoints, checkpoint_file)

    if current_total < target_tokens:
        pass
    else:
        pass


if __name__ == "__main__":
    main()