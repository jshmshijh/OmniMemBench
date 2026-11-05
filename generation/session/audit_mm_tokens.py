#!/usr/bin/env python3

import json, os, sys, re, hashlib, argparse, tempfile, requests
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR   = Path(__file__).resolve().parent
GA_ROOT    = BASE_DIR.parent
STEP4_OUT  = BASE_DIR / "output"
LOG_BASE   = GA_ROOT / "_pipeline_logs"
TMP_DIR    = Path(tempfile.gettempdir()) / "distractor_media"
TMP_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = os.environ.get("MODEL_PATH", "")

TIER_TOKENS = {"128k": 108_000, "256k": 230_000, "512k": 460_000, "1m": 950_000}
TIER_ORDER  = ["128k", "256k", "512k", "1m"]


def _local_path(url: str) -> str:
    ext = os.path.splitext(urlparse(url).path)[1] or ".bin"
    h   = hashlib.sha256(url.encode()).hexdigest()[:24]
    return str(TMP_DIR / f"{h}{ext}")


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


def _count_text_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _count_audio_tokens_direct(file_path: str) -> int:
    try:
        import soundfile as sf
        info = sf.info(file_path)
        duration_sec = info.frames / info.samplerate
        frames = int(duration_sec * 25 + 0.5)
        return max(frames, 1)
    except Exception as e:
        return -1


def _count_mm_tokens(file_path: str, media_type: str, processor) -> int:
    if processor is None:
        return 0
    if media_type == "audio":
        return _count_audio_tokens_direct(file_path)
    try:
        from qwen_omni_utils import process_mm_info
        content = [{"type": media_type, media_type: f"file://{file_path}"}, {"type": "text", "text": "hi"}]
        conv    = [{"role": "user", "content": content}]
        text    = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs  = processor(text=[text], audio=audios or None, images=images or None,
                            videos=videos or None, return_tensors="pt", padding=True)
        pad_map = {"image": "<|image_pad|>", "video": "<|video_pad|>", "audio": "<|audio_pad|>"}
        pad_id  = processor.tokenizer.convert_tokens_to_ids(pad_map[media_type])
        return int((inputs["input_ids"] == pad_id).sum().item())
    except Exception as e:
        return -1


def count_session_tokens_real(session: dict, registry: dict, tokenizer, processor):
    text_tok  = 0
    mm_tok    = 0
    mm_fails  = 0
    seen_urls = set()
    for rnd in session.get("dialogues", []):
        text_tok += _count_text_tokens(rnd.get("user",      ""), tokenizer)
        text_tok += _count_text_tokens(rnd.get("assistant", ""), tokenizer)
        for mm in rnd.get("mm", []):
            mm_id  = mm.get("mm_id", "")
            entry  = registry.get(mm_id, {})
            url    = entry.get("url", "")
            mtype  = mm.get("type", "") or entry.get("type", "")
            if not url or not mtype or url in seen_urls:
                continue
            seen_urls.add(url)
            lp = _local_path(url)
            if _download(url, lp):
                t = _count_mm_tokens(lp, mtype, processor)
                if t == -1:
                    mm_fails += 1
                else:
                    mm_tok += t
    return text_tok, mm_tok, mm_fails


def audit_run(run_id: str, tokenizer, processor):
    sessions_file  = STEP4_OUT / "08_distractor_gen"  / run_id / "distractor_sessions.jsonl"
    registry_file  = STEP4_OUT / "07_distractor_pool" / run_id / "distractor_mm_registry.json"
    checkpoint_file= STEP4_OUT / "08_distractor_gen"  / run_id / "distractor_checkpoints.json"
    evidence_file  = STEP4_OUT / "06_evidence_tokens" / run_id / "evidence_token_count.json"

    if not sessions_file.exists():
        return None

    registry   = json.loads(registry_file.read_text("utf-8")) if registry_file.exists() else {}
    base_tokens= json.loads(evidence_file.read_text("utf-8")).get("total_tokens", 0) if evidence_file.exists() else 0
    cp_data    = json.loads(checkpoint_file.read_text("utf-8")) if checkpoint_file.exists() else {}

    sessions   = [json.loads(l) for l in sessions_file.read_text("utf-8").splitlines() if l.strip()]

    recorded_total = sum(s.get("_tokens", 0) for s in sessions)
    real_total     = 0
    session_deltas = []
    total_mm_fails = 0

    print(f"\n[{run_id}] {len(sessions)} sessions, base={base_tokens:,}, recorded_dist={recorded_total:,}", flush=True)

    for i, sess in enumerate(sessions):
        sid      = sess.get("session_id", f"#{i}")
        recorded = sess.get("_tokens", 0)
        text_tok, mm_tok, fails = count_session_tokens_real(sess, registry, tokenizer, processor)
        real     = text_tok + mm_tok
        delta    = real - recorded
        real_total    += real
        total_mm_fails+= fails
        session_deltas.append({
            "session_id": sid,
            "recorded": recorded,
            "real":     real,
            "text":     text_tok,
            "mm":       mm_tok,
            "delta":    delta,
            "mm_fails": fails,
        })
        print(f"  {sid}: recorded={recorded:>7,}  real={real:>7,}  (text={text_tok:>6,} + mm={mm_tok:>6,})  delta={delta:>+7,}  mm_fail={fails}", flush=True)

    grand_recorded = base_tokens + recorded_total
    grand_real     = base_tokens + real_total
    grand_delta    = grand_real - grand_recorded

    reached_tiers  = [t for t in TIER_ORDER if t in cp_data]
    real_tier = "-"
    for t in reversed(TIER_ORDER):
        if grand_real >= TIER_TOKENS[t]:
            real_tier = t
            break
    recorded_tier = "-"
    for t in reversed(TIER_ORDER):
        if grand_recorded >= TIER_TOKENS[t]:
            recorded_tier = t
            break

    result = {
        "run_id":          run_id,
        "n_sessions":      len(sessions),
        "base_tokens":     base_tokens,
        "recorded_dist":   recorded_total,
        "real_dist":       real_total,
        "dist_delta":      real_total - recorded_total,
        "grand_recorded":  grand_recorded,
        "grand_real":      grand_real,
        "grand_delta":     grand_delta,
        "mm_fail_count":   total_mm_fails,
        "checkpoint_tiers":reached_tiers,
        "recorded_tier":   recorded_tier,
        "real_tier":       real_tier,
        "sessions":        session_deltas,
    }

    print(f"  distractor: recorded={recorded_total:>10,}  real={real_total:>10,}  delta={real_total-recorded_total:>+10,}")
    print(f"  grand:      recorded={grand_recorded:>10,}  real={grand_real:>10,}  delta={grand_delta:>+10,}")
    print(f"  tier: checkpoint={reached_tiers}  recorded_tier={recorded_tier}  real_tier={real_tier}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_ids", nargs="+",
        default=["00001","00006","00008","00012","00016","00018","00019","00022",
                 "00024","00027","00029","00030","00031","00036","00037","00039","00040","00046"])
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="audit_result.json", help="")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        processor = None

    all_results = []
    for run_id in args.run_ids:
        res = audit_run(run_id, tokenizer, processor)
        if res:
            all_results.append(res)

    out_path = BASE_DIR / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*100)
    print(f"{'run':<8} {'sessions':>8} {'dist_delta':>12} {'grand_recorded':>15} {'grand_real':>12} {'delta':>10}  {'cp_tiers':<20} {'rec_tier':<8} {'real_tier':<8} {'mm_fails':>8}")
    print("-"*100)
    for r in all_results:
        tier_change = f"{r['recorded_tier']} -> {r['real_tier']}" if r["recorded_tier"] != r["real_tier"] else r["real_tier"]
        print(f"{r['run_id']:<8} {r['n_sessions']:>8} {r['dist_delta']:>+12,} {r['grand_recorded']:>15,} {r['grand_real']:>12,} {r['grand_delta']:>+10,}  {str(r['checkpoint_tiers']):<20} {r['recorded_tier']:<8} {r['real_tier']:<8} {r['mm_fail_count']:>8}")


if __name__ == "__main__":
    main()
