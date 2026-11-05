#!/usr/bin/env python3

import json, os, sys, hashlib, argparse, tempfile, requests, shutil
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR  = Path(__file__).resolve().parent
GA_ROOT   = BASE_DIR.parent
STEP4_OUT = BASE_DIR / "output"

DEFAULT_MODEL = os.environ.get("MODEL_PATH", "")

TIER_TOKENS = {"128k": 108_000, "256k": 230_000, "512k": 460_000, "1m": 950_000}
TIER_ORDER  = ["128k", "256k", "512k", "1m"]

MM_FALLBACK_TOKENS = {
    "image": 256,
    "video": 1024,
    "audio": 100,
}

TMP_DIR = Path(tempfile.gettempdir()) / "distractor_media"
TMP_DIR.mkdir(exist_ok=True)


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
        return max(int(duration_sec * 25 + 0.5), 1)
    except Exception as e:
        return -1


def _count_mm_tokens(file_path: str, media_type: str, processor) -> int:
    """
    """
    if media_type == "audio":
        return _count_audio_tokens_direct(file_path)

    if processor is None:
        return -1

    try:
        from qwen_omni_utils import process_mm_info
        content = [{"type": media_type, media_type: f"file://{file_path}"},
                   {"type": "text", "text": "hi"}]
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
    text_tok   = 0
    mm_tok     = 0
    mm_fails   = 0
    mm_est     = 0
    seen_urls  = set()

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
            if not _download(url, lp):
                mm_fails += 1
                continue
            t = _count_mm_tokens(lp, mtype, processor)
            if t >= 0:
                mm_tok += t
            else:
                fallback = MM_FALLBACK_TOKENS.get(mtype, 256)
                mm_tok  += fallback
                mm_est  += 1

    return text_tok, mm_tok, mm_fails, mm_est


def _atomic_write_jsonl(sessions: list, path: Path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp, str(path))


def _atomic_write_json(obj: dict, path: Path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, str(path))


def patch_run(run_id: str, tokenizer, processor, dry_run: bool) -> dict:
    sessions_file   = STEP4_OUT / "08_distractor_gen"  / run_id / "distractor_sessions.jsonl"
    registry_file   = STEP4_OUT / "07_distractor_pool" / run_id / "distractor_mm_registry.json"
    checkpoint_file = STEP4_OUT / "08_distractor_gen"  / run_id / "distractor_checkpoints.json"
    evidence_file   = STEP4_OUT / "06_evidence_tokens" / run_id / "evidence_token_count.json"

    if not sessions_file.exists():
        return {}

    registry    = json.loads(registry_file.read_text("utf-8")) if registry_file.exists() else {}
    base_tokens = json.loads(evidence_file.read_text("utf-8")).get("total_tokens", 0) if evidence_file.exists() else 0
    old_ckpt    = json.loads(checkpoint_file.read_text("utf-8")) if checkpoint_file.exists() else {}

    sessions_raw = [json.loads(l) for l in sessions_file.read_text("utf-8").splitlines() if l.strip()]

    print(f"\n[{run_id}] {len(sessions_raw)} sessions, base={base_tokens:,}", flush=True)

    patched_sessions = []
    total_old_tok    = 0
    total_new_tok    = 0
    changed_count    = 0
    total_mm_est     = 0
    total_mm_fails   = 0

    for i, sess in enumerate(sessions_raw):
        sid      = sess.get("session_id", f"#{i}")
        old_tok  = sess.get("_tokens", 0)
        total_old_tok += old_tok

        text_tok, mm_tok, mm_fails, mm_est = count_session_tokens_real(sess, registry, tokenizer, processor)
        new_tok = text_tok + mm_tok
        total_new_tok  += new_tok
        total_mm_fails += mm_fails
        total_mm_est   += mm_est

        if new_tok != old_tok:
            changed_count += 1
            print(f"  {sid}: {old_tok:>7,} → {new_tok:>7,}  "
                  f"(text={text_tok:>6,} mm={mm_tok:>6,}  "
                  f"delta={new_tok-old_tok:>+7,}  est={mm_est} fail={mm_fails})", flush=True)
        else:
            print(f"  {sid}: {old_tok:>7,} (unchanged)  est={mm_est} fail={mm_fails}", flush=True)

        new_sess = dict(sess)
        new_sess["_tokens"] = new_tok
        if mm_est > 0:
            new_sess["_mm_estimated"] = True
        patched_sessions.append(new_sess)

    new_ckpt     = {}
    running_total = base_tokens
    for sess in patched_sessions:
        running_total += sess["_tokens"]
        for tier in TIER_ORDER:
            if tier not in new_ckpt and running_total >= TIER_TOKENS[tier]:
                idx = patched_sessions.index(sess)
                new_ckpt[tier] = {
                    "distractor_count": idx + 1,
                    "total_tokens":     running_total,
                }

    grand_old = base_tokens + total_old_tok
    grand_new = base_tokens + total_new_tok

    old_tier = "-"
    for t in reversed(TIER_ORDER):
        if grand_old >= TIER_TOKENS[t]:
            old_tier = t
            break
    new_tier = "-"
    for t in reversed(TIER_ORDER):
        if grand_new >= TIER_TOKENS[t]:
            new_tier = t
            break

    tier_change = f"{old_tier} → {new_tier}" if old_tier != new_tier else f"{old_tier} (unchanged)"

    old_ckpt_tiers = list(old_ckpt.keys())
    new_ckpt_tiers = list(new_ckpt.keys())

    print(f"    distractor tokens: {total_old_tok:>10,} → {total_new_tok:>10,}  delta={total_new_tok-total_old_tok:>+10,}")
    print(f"    grand total:       {grand_old:>10,} → {grand_new:>10,}  delta={grand_new-grand_old:>+10,}")
    print(f"    tier:              {tier_change}")
    print(f"    checkpoint tiers:  {old_ckpt_tiers} → {new_ckpt_tiers}")
    print(f"    changed sessions:  {changed_count}/{len(sessions_raw)}")
    print(f"    mm_estimated:      {total_mm_est}   mm_download_fail: {total_mm_fails}")

    result = {
        "run_id":        run_id,
        "n_sessions":    len(sessions_raw),
        "base_tokens":   base_tokens,
        "old_dist_tok":  total_old_tok,
        "new_dist_tok":  total_new_tok,
        "dist_delta":    total_new_tok - total_old_tok,
        "grand_old":     grand_old,
        "grand_new":     grand_new,
        "grand_delta":   grand_new - grand_old,
        "old_tier":      old_tier,
        "new_tier":      new_tier,
        "old_ckpt_tiers": old_ckpt_tiers,
        "new_ckpt_tiers": new_ckpt_tiers,
        "changed_sessions": changed_count,
        "mm_estimated":  total_mm_est,
        "mm_dl_fail":    total_mm_fails,
    }

    if dry_run:
        return result

    bak = sessions_file.with_suffix(".jsonl.patch_bak")
    if not bak.exists():
        shutil.copy2(str(sessions_file), str(bak))

    ckpt_bak = checkpoint_file.with_suffix(".json.patch_bak")
    if not ckpt_bak.exists() and checkpoint_file.exists():
        shutil.copy2(str(checkpoint_file), str(ckpt_bak))

    _atomic_write_jsonl(patched_sessions, sessions_file)

    _atomic_write_json(new_ckpt, checkpoint_file)

    return result


DEFAULT_AFFECTED = [
    "00001", "00006", "00016", "00018", "00019",
    "00022", "00036", "00037", "00040", "00046",
]


def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--run_ids", nargs="+", default=DEFAULT_AFFECTED,
                        help="")
    parser.add_argument("--model_path", default=DEFAULT_MODEL,
                        help="")
    parser.add_argument("--dry_run", action="store_true",
                        help="")
    parser.add_argument("--output", default="patch_result.json",
                        help="")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        processor = None

    if args.dry_run:
        pass

    all_results = []
    for run_id in args.run_ids:
        res = patch_run(run_id, tokenizer, processor, dry_run=args.dry_run)
        if res:
            all_results.append(res)

    print("\n" + "=" * 110)
    print(f"{'run':<8} {'sessions':>8} {'dist_delta':>12} {'grand_old':>12} {'grand_new':>12} "
          f"{'delta':>10}  {'tier_change':<18} {'ckpt_new':<22} {'est':>5} {'dl_fail':>7}")
    print("-" * 110)
    for r in all_results:
        tier_str = f"{r['old_tier']} → {r['new_tier']}" if r["old_tier"] != r["new_tier"] else r["new_tier"]
        print(f"{r['run_id']:<8} {r['n_sessions']:>8} {r['dist_delta']:>+12,} {r['grand_old']:>12,} "
              f"{r['grand_new']:>12,} {r['grand_delta']:>+10,}  {tier_str:<18} "
              f"{str(r['new_ckpt_tiers']):<22} {r['mm_estimated']:>5} {r['mm_dl_fail']:>7}")

    need_reassemble = [r["run_id"] for r in all_results if r.get("grand_delta", 0) != 0]
    if need_reassemble:
        for rid in need_reassemble:
            print(f"  python 09_tier_benchmark_assembly.py --run_id {rid}")

    if not args.dry_run:
        out_path = BASE_DIR / args.output
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
