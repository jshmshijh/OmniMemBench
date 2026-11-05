#!/usr/bin/env python3
"""
OmniMemBench Long-Context Evaluation Runner.
Scans step6 output for benchmark_{tier}.json, evaluates all run_id × tier combos,
outputs per-run JSONL + aggregated summary.  Supports --type omni|vl.
"""
from __future__ import annotations

import os
import argparse, json, os, time, traceback
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from utils import (
    load_json, load_text, load_jsonl_rows, load_done_indices, append_jsonl,
    parse_input_form, input_form_tag, summarize_rows,
    build_contents, build_raw_parts, build_caption_block,
    extract_qa_media, llm_chat,
    exact_match, token_f1,
    format_judge_prompt, format_nugget_judge_prompt,
    parse_judge_json, compute_nugget_metrics, qa_score,
    _is_gemini_model,
)

# ── Config ────────────────────────────────────────────────────────────────────

TIER_ORDER = ["128k", "256k", "512k", "1m"]
DEFAULT_JUDGE_MODEL = "glm-5.2"

INPUT_FORM_BY_TYPE = {
    "omni": "image=raw,audio=raw,video=raw",
    "vl":   "image=raw,audio=caption,video=raw",
    "text": "image=caption,audio=caption,video=caption",
}

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = Path("./data")
DEFAULT_OUTPUT_BASE = SCRIPT_DIR


# ── Job discovery ─────────────────────────────────────────────────────────────

def count_videos_in_dataset(data_path: Path) -> int:
    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    count = 0
    for session in data.get("multi_session_dialogues") or []:
        for dialog in session.get("dialogues") or []:
            for mm in dialog.get("mm") or []:
                if mm.get("type") == "video":
                    count += 1
    return count


def discover_jobs(input_dir: Path, tiers: List[str], max_videos: Optional[int] = None) -> List[Dict]:
    jobs = []
    skipped = []
    for run_dir in sorted(input_dir.iterdir()):
        if not run_dir.is_dir(): continue
        for tier in tiers:
            fp = run_dir / f"benchmark_{tier}.json"
            if not fp.exists(): continue
            if max_videos is not None:
                n = count_videos_in_dataset(fp)
                if n > max_videos:
                    skipped.append((run_dir.name, tier, n))
                    continue
            jobs.append({"run_id": run_dir.name, "tier": tier, "data_path": fp})
    if skipped:
        print(f"[Filter] max_videos={max_videos}: skipping {len(skipped)} (run_id, tier) pairs (>{max_videos} videos)")
        for rid, tier, n in skipped:
            print(f"  skip {rid}/{tier}: {n} videos")
    if max_videos is not None:
        print(f"[Filter] Eligible (run_id, tier) pairs: {len(jobs)}")
    return jobs


# ── Single QA processing ─────────────────────────────────────────────────────

def process_one(
    *, qa_index: int, qa: Dict, base_contents: List[Dict],
    input_form: Dict[str, str], run_id: str,
    api_key: str, base_url: str, model: str,
    judge_model: Optional[str], enable_judge: bool,
    judge_template: str, nugget_judge_template: str,
    temperature: float, max_tokens: int,
    timeout_seconds: int, max_retries: int,
    judge_api_key: Optional[str] = None, judge_base_url: Optional[str] = None,
    eval_type: str = "omni",
    media_root: Optional[Path] = None,
) -> Dict:
    question = str(qa.get("question", "") or "")
    short_gt = str(qa.get("answer", "") or "")
    full_gt  = str(qa.get("full_answer", "") or "")
    category = str(qa.get("type", qa.get("point", "")) or "")
    nuggets  = qa.get("nuggets") or []
    is_sum   = category.upper() == "SUM" and nuggets

    qa_media = extract_qa_media(qa)
    qa_parts = build_raw_parts(qa_media, input_form, media_root)
    merged_q = question.strip()
    caption_blk = build_caption_block(qa_media, input_form)
    if caption_blk: merged_q = f"{caption_blk}\n{merged_q}"
    if merged_q: qa_parts.append({"text": merged_q})

    contents = base_contents + [{"role": "user", "parts": qa_parts}]

    t0 = time.perf_counter()
    pred, raw_resp = llm_chat(
        api_key=api_key, base_url=base_url, model=model,
        contents=contents, temperature=temperature,
        max_tokens=max_tokens, timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    latency = time.perf_counter() - t0

    item = {
        "sample_id": run_id, "run_id": run_id, "qa_index": qa_index,
        "question_id": qa.get("question_id", ""),
        "question": question, "system_answer": pred,
        "original_answer": short_gt, "full_answer": full_gt,
        "category": category,
        "nuggets": nuggets or None,
        "clue": qa.get("clue", []),
        "metrics": {"EM": exact_match(pred, short_gt), "F1": token_f1(pred, short_gt)},
        "latency_seconds": latency,
        "usage": (raw_resp.get("usage") or raw_resp.get("usageMetadata")) if isinstance(raw_resp, dict) else None,
        "model": model, "input_form": {k: v for k, v in input_form.items()},
        "eval_type": eval_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ok",
    }

    if enable_judge:
        item = _run_judge_for_item(
            item, qa=qa,
            api_key=judge_api_key or api_key, base_url=judge_base_url or base_url,
            judge_model=judge_model or DEFAULT_JUDGE_MODEL,
            judge_template=judge_template,
            nugget_judge_template=nugget_judge_template,
            timeout_seconds=timeout_seconds, max_retries=max_retries,
        )
    return item


def _run_judge_for_item(
    item: Dict, *, qa: Dict,
    api_key: str, base_url: str, judge_model: str,
    judge_template: str, nugget_judge_template: str,
    timeout_seconds: int, max_retries: int,
) -> Dict:
    question = item.get("question", "")
    pred = item.get("system_answer", "")
    ground_truth = item.get("original_answer", "")
    nuggets = qa.get("nuggets") or []
    is_sum = str(item.get("category", "")).upper() == "SUM"

    if is_sum:
        jp = format_nugget_judge_prompt(nugget_judge_template, question=question, model_output=pred, nuggets=nuggets)
    else:
        jp = format_judge_prompt(judge_template, question=question, ground_truth=ground_truth, model_output=pred)

    jt0 = time.perf_counter()
    judge_raw = {}
    judge_result = {"error": "No judge attempts"}
    for attempt in range(max_retries):
        try:
            judge_text, judge_raw = llm_chat(
                api_key=api_key, base_url=base_url, model=judge_model,
                contents=[{"role": "USER", "parts": [{"text": jp}]}],
                temperature=0.0, timeout_seconds=timeout_seconds, max_retries=1,
            )
            parsed = parse_judge_json(judge_text)
            if is_sum:
                judge_result = compute_nugget_metrics(parsed, nuggets)
            else:
                judge_result = {**parsed, "score": qa_score(parsed)}
            break
        except Exception as e:
            judge_result = {"error": str(e)}
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 8))

    item["judge"] = {
        "model": judge_model, "temperature": 0.0,
        "latency_seconds": time.perf_counter() - jt0,
        "usage": judge_raw.get("usage") if isinstance(judge_raw, dict) else None,
        "result": judge_result,
    }
    return item


def judge_one(
    *, row: Dict, qa: Dict,
    api_key: str, base_url: str, judge_model: str,
    judge_template: str, nugget_judge_template: str,
    timeout_seconds: int, max_retries: int,
) -> Dict:
    updated = dict(row)  # shallow copy
    updated = _run_judge_for_item(
        updated, qa=qa,
        api_key=api_key, base_url=base_url, judge_model=judge_model,
        judge_template=judge_template, nugget_judge_template=nugget_judge_template,
        timeout_seconds=timeout_seconds, max_retries=max_retries,
    )
    updated["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return updated


def backfill_judge(
    *, jsonl_path: Path, qas: List[Dict], run_id: str,
    api_key: str, base_url: str, judge_model: str,
    judge_template: str, nugget_judge_template: str,
    timeout_seconds: int, max_retries: int,
    max_workers: int, print_every: int = 1,
    eval_type: str = "omni",
):
    rows = load_jsonl_rows(jsonl_path, dedup=True)
    qa_map = {i: qa for i, qa in enumerate(qas) if isinstance(qa, dict)}

    pending = []
    for r in rows:
        qi = r.get("qa_index")
        if r.get('status') != 'ok' or qi not in qa_map:
            continue
        result = (r.get('judge') or {}).get('result') or {}
        try:
            if str(r.get('category', '')).upper() == 'SUM':
                compute_nugget_metrics(result.get('nugget_coverage'), qa_map[qi].get('nuggets') or [])
            else:
                qa_score(result)
            valid_judge = True
        except (ValueError, KeyError, TypeError, AttributeError):
            valid_judge = False
        if r.get("status") == "ok" and (
            "judge" not in r
            or r["judge"].get("model") != judge_model
            or r.get("eval_type") != eval_type
            or not valid_judge
        ):
            qi = r.get("qa_index")
            if qi is not None and qi in qa_map:
                pending.append((qi, dict(r, eval_type=eval_type), qa_map[qi]))

    if not pending:
        return 0

    print(f"  [Backfill] {len(pending)} QAs missing judge, backfilling...")
    write_lock = Lock()
    completed, ok_cnt, err_cnt = 0, 0, 0

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        fut_map = {}
        for qi, row, qa in pending:
            fut = ex.submit(
                judge_one, row=row, qa=qa,
                api_key=api_key, base_url=base_url, judge_model=judge_model,
                judge_template=judge_template, nugget_judge_template=nugget_judge_template,
                timeout_seconds=timeout_seconds, max_retries=max_retries,
            )
            fut_map[fut] = (qi, row)
        for fut in fut_map:
            qi, row = fut_map[fut]
            try:
                updated = fut.result()
                append_jsonl(jsonl_path, updated, write_lock)
                if 'error' in updated['judge']['result']:
                    err_cnt += 1
                else:
                    ok_cnt += 1
            except Exception as e:
                err_item = dict(row, judge={'model': judge_model, 'result': {'error': str(e)}},
                                recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                append_jsonl(jsonl_path, err_item, write_lock)
                err_cnt += 1
            completed += 1
            if print_every and completed % print_every == 0:
                print(f"  [Backfill] [{completed}/{len(pending)}] QA {qi} {'ok' if ok_cnt > err_cnt else 'err'}")

    print(f"  [Backfill] Done: {ok_cnt} ok, {err_cnt} errors")
    return len(pending)
# ── END NEW ──────────────────────────────────────────────────────────────────


# ── Evaluate one (run_id, tier) ───────────────────────────────────────────────

def eval_single_run(
    *, data_path: Path, run_output_dir: Path, run_id: str, tier: str, eval_type: str,
    api_key: str, base_url: str, model: str, judge_model: Optional[str],
    sys_prompt: str, judge_template: str, nugget_judge_template: str,
    temperature: float, max_tokens: int, timeout_seconds: int, max_retries: int,
    max_workers: int, enable_judge: bool, resume: bool, include_persona: bool,
    judge_only: bool = False,                                       # ── NEW
    heartbeat_seconds: int = 30, print_every: int = 1,
    judge_api_key: Optional[str] = None, judge_base_url: Optional[str] = None,
    media_root: Optional[Path] = None,
):
    judge_model = judge_model or os.environ.get("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
    judge_api_key = judge_api_key or os.environ.get("JUDGE_API_KEY") or api_key
    judge_base_url = judge_base_url or os.environ.get("JUDGE_API_BASE") or base_url
    run_output_dir.mkdir(parents=True, exist_ok=True)
    input_form = parse_input_form(INPUT_FORM_BY_TYPE[eval_type])

    dataset = load_json(data_path)
    qas = dataset.get("QAs") or dataset.get("human-annotated QAs") or []

    jsonl_path   = run_output_dir / "results.jsonl"
    summary_path = run_output_dir / "summary.json"
    meta_path    = run_output_dir / "meta.json"

    if judge_only:
        if not jsonl_path.exists():
            print(f"  [judge_only] No results.jsonl found, skip")
            return
        backfill_judge(
            jsonl_path=jsonl_path, qas=qas, run_id=run_id,
            api_key=judge_api_key, base_url=judge_base_url,
            judge_model=judge_model,
            judge_template=judge_template,
            nugget_judge_template=nugget_judge_template,
            timeout_seconds=timeout_seconds, max_retries=max_retries,
            max_workers=max_workers, print_every=print_every,
            eval_type=eval_type,
        )
        # rewrite summary
        rows = load_jsonl_rows(jsonl_path)
        summary = summarize_rows(rows)
        _warn_judge_missing(summary, run_id, tier)
        meta = load_json(meta_path) if meta_path.exists() else {}
        meta.update(judge_model=judge_model, judge_temperature=0, enable_judge=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "summary": summary},
                      f, ensure_ascii=False, indent=2)
        return
    # ── END NEW ──────────────────────────────────────────────────────────────

    if enable_judge and resume:
        backfill_judge(
            jsonl_path=jsonl_path, qas=qas, run_id=run_id,
            api_key=judge_api_key, base_url=judge_base_url, judge_model=judge_model,
            judge_template=judge_template, nugget_judge_template=nugget_judge_template,
            timeout_seconds=timeout_seconds, max_retries=max_retries,
            max_workers=max_workers, print_every=print_every, eval_type=eval_type,
        )
    base_contents = build_contents(dataset, input_form, sys_prompt, include_persona, media_root)

    done_indices = load_done_indices(jsonl_path) if resume else set()
    if done_indices:
        print(f"  Resume: {len(done_indices)} completed QAs found")

    fmt = "Gemini" if _is_gemini_model(model) else "OpenAI"
    tag = f"{tier}_{eval_type}"
    meta = {
        "run_id": run_id, "tag": tag, "tier": tier, "eval_type": eval_type,
        "data_path": str(data_path), "model": model, "judge_model": judge_model if enable_judge else None,
        "api_format": fmt, "input_form": INPUT_FORM_BY_TYPE[eval_type],
        "enable_judge": enable_judge,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    write_lock = Lock()
    jobs = [(i, qa) for i, qa in enumerate(qas) if i not in done_indices and isinstance(qa, dict)]
    if not jobs:
        print("  No pending QAs.")

    print(f"  Running {len(jobs)} QAs, workers={max_workers}, format={fmt}")

    completed, ok_cnt, err_cnt = 0, 0, 0
    t_start = time.perf_counter()
    last_hb = t_start
    hb = max(0, heartbeat_seconds)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        fut_map = {}
        for qi, qa in jobs:
            fut = ex.submit(
                process_one,
                qa_index=qi, qa=qa, base_contents=base_contents,
                input_form=input_form, run_id=run_id,
                api_key=api_key, base_url=base_url, model=model,
                judge_model=judge_model, enable_judge=enable_judge,
                judge_api_key=judge_api_key, judge_base_url=judge_base_url,
                eval_type=eval_type,
                media_root=media_root,
                judge_template=judge_template, nugget_judge_template=nugget_judge_template,
                temperature=temperature, max_tokens=max_tokens,
                timeout_seconds=timeout_seconds, max_retries=max_retries,
            )
            fut_map[fut] = qi
        pending = set(fut_map)

        while pending:
            done, pending = wait(pending, timeout=hb or None, return_when=FIRST_COMPLETED)
            now = time.perf_counter()
            if not done:
                if hb and now - last_hb >= hb:
                    print(f"  [{completed}/{len(jobs)}] pending={len(pending)} ok={ok_cnt} err={err_cnt} {now-t_start:.0f}s")
                    last_hb = now
                continue

            for fut in done:
                qi = fut_map[fut]
                try:
                    item = fut.result()
                except Exception as e:
                    item = {"run_id": run_id, "qa_index": qi, "status": "error",
                            "error": str(e), "traceback": traceback.format_exc(limit=10),
                            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")}

                append_jsonl(jsonl_path, item, write_lock)
                completed += 1
                if item.get("status") == "ok": ok_cnt += 1
                else: err_cnt += 1

                if print_every and completed % print_every == 0:
                    msg = f"  [{completed}/{len(jobs)}] QA {qi}"
                    if item.get("status") == "ok":
                        m = item.get("metrics", {})
                        msg += f" EM={m.get('EM')} F1={m.get('F1',0):.3f} lat={item.get('latency_seconds',0):.1f}s"
                        jr = (item.get("judge") or {}).get("result") or {}
                        if "nugget_recall" in jr: msg += f" nugget_recall={jr['nugget_recall']:.3f}"
                        elif "score" in jr: msg += f" judge={jr['score']}"
                    else:
                        msg += f" ERR={item.get('error','')}"
                    print(msg)

            if hb and now - last_hb >= hb:
                print(f"  [{completed}/{len(jobs)}] pending={len(pending)} ok={ok_cnt} err={err_cnt}")
                last_hb = now

    # write per-run summary
    rows = load_jsonl_rows(jsonl_path)
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["qa_count"] = len(rows)
    meta["summary"] = summarize_rows(rows)

    _warn_judge_missing(meta["summary"], run_id, tier)
    # ── END NEW ──────────────────────────────────────────────────────────────

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": meta["summary"]}, f, ensure_ascii=False, indent=2)


def _warn_judge_missing(summary: Dict, run_id: str, tier: str):
    n = summary.get("judge_missing", 0)
    if n > 0:
        indices = summary.get("judge_missing_indices", [])
        print(f"\n  {run_id}/{tier}: {n} QAs have missing or invalid judgments.")
        if indices:
            print(f"     qa_indices: {indices[:20]}{'...' if len(indices) > 20 else ''}")
        print(f"     Run with --judge_only to backfill.\n")
# ── END NEW ──────────────────────────────────────────────────────────────────


# ── Aggregated summary ────────────────────────────────────────────────────────

def build_model_summary(model_output_dir: Path, tiers: List[str]) -> Dict:
    by_tier: Dict[str, Any] = {}
    for tier in tiers:
        tier_dir = model_output_dir / tier
        if not tier_dir.is_dir(): continue
        tier_rows: List[Dict] = []
        run_rows: Dict[str, List[Dict]] = {}
        for run_subdir in sorted(tier_dir.iterdir()):
            if not run_subdir.is_dir(): continue
            f = run_subdir / "results.jsonl"
            if not f.exists(): continue
            rows = load_jsonl_rows(f)
            tier_rows.extend(rows)
            run_rows[run_subdir.name] = rows
        if not tier_rows: continue
        ts = summarize_rows(tier_rows)
        ts["run_count"] = len(run_rows)
        ts["by_run"] = {rid: summarize_rows(rr) for rid, rr in sorted(run_rows.items())}
        jm = ts.get("judge_missing", 0)
        if jm > 0:
            print(f"  ⚠️  WARNING: tier={tier} aggregated: {jm} QAs with missing judge")
        by_tier[tier] = ts
    return {"by_tier": by_tier}


def build_top_summary(eval_type_dir: Path, tiers: List[str], eval_type: str) -> Dict:
    by_model: Dict[str, Any] = {}
    for model_dir in sorted(eval_type_dir.iterdir()):
        if not model_dir.is_dir(): continue
        model_summary = build_model_summary(model_dir, tiers)
        if model_summary["by_tier"]:
            by_model[model_dir.name] = model_summary
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_type": eval_type,
        "by_model": by_model,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="OmniMemBench longctx evaluation runner.")
    # data
    p.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--media_root", type=Path, default=SCRIPT_DIR.parents[1])
    p.add_argument("--tiers", default="128k,256k,512k,1m")
    p.add_argument("--run_ids", default=None)
    p.add_argument("--max_videos", type=int, default=None)
    # eval type
    p.add_argument("--type", choices=["omni", "vl", "text", "all"], default="all")
    # output
    p.add_argument("--output_base", default=str(DEFAULT_OUTPUT_BASE))
    # api
    p.add_argument("--api_key", default=None)
    p.add_argument("--base_url", default=os.environ.get("LLM_API_BASE") or "http://localhost:8000/v1")
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", "your-model-name"))
    p.add_argument("--judge_model", default=os.environ.get("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge_api_key", default=None)
    p.add_argument("--judge_api_base", default=None, help="Judge endpoint; falls back to JUDGE_API_BASE, then --base_url")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=32000)
    p.add_argument("--timeout_seconds", type=int, default=1200)
    p.add_argument("--max_retries", type=int, default=20)
    p.add_argument("--max_workers", type=int, default=50)
    # prompt
    p.add_argument("--sys_prompt_file", default=None)
    p.add_argument("--judge_prompt_file", default=None)
    p.add_argument("--nugget_judge_prompt_file", default=None)
    p.add_argument("--no_persona", action="store_true")
    # flags
    p.add_argument("--enable_judge", dest="enable_judge", action="store_true", default=True)
    p.add_argument("--disable_judge", dest="enable_judge", action="store_false")
    p.add_argument("--judge_only", action="store_true", default=False)  # ── NEW
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--heartbeat_seconds", type=int, default=30)
    p.add_argument("--print_every", type=int, default=1)
    args = p.parse_args()

    if args.judge_only:
        args.enable_judge = True
    # ── END NEW ──────────────────────────────────────────────────────────────

    # resolve API key
    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        env_file = SCRIPT_DIR.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip().startswith("LLM_API_KEY=") and not line.strip().startswith("#"):
                    api_key = line.split("=", 1)[1].strip(); break
    judge_api_key = args.judge_api_key or os.environ.get("JUDGE_API_KEY") or api_key
    judge_base_url = args.judge_api_base or os.environ.get("JUDGE_API_BASE") or args.base_url
    if not args.dry_run and not args.summary_only:
        if not args.judge_only and not api_key:
            raise SystemExit("Missing LLM_API_KEY")
        if args.enable_judge and not judge_api_key:
            raise SystemExit("Missing JUDGE_API_KEY (or LLM_API_KEY for a shared gateway)")

    # resolve prompt files
    prompt_dir = SCRIPT_DIR.parent / "prompt"
    eval_dir = Path(__file__).resolve().parents[2] / "omnimembench" / "engine" / "evaluate"
    sys_prompt = load_text(Path(args.sys_prompt_file)) if args.sys_prompt_file else load_text(prompt_dir / "sys_prompt.txt")
    judge_template = load_text(Path(args.judge_prompt_file)) if args.judge_prompt_file else load_text(eval_dir / "llm_judge.txt")
    nugget_judge_template = load_text(Path(args.nugget_judge_prompt_file)) if args.nugget_judge_prompt_file else load_text(eval_dir / "llm_judge_nugget.txt")

    input_dir = Path(args.input_dir).resolve()
    output_base = Path(args.output_base).resolve()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    eval_types = ["omni", "vl", "text"] if args.type == "all" else [args.type]

    filter_run_ids = set(x.strip() for x in args.run_ids.split(",") if x.strip()) if args.run_ids else None
    all_jobs = discover_jobs(input_dir, tiers, max_videos=args.max_videos)
    if filter_run_ids:
        all_jobs = [j for j in all_jobs if j["run_id"] in filter_run_ids]
    print(f"Discovered {len(all_jobs)} (run_id, tier) pairs across tiers: {tiers}")
    if args.dry_run:
        for job in all_jobs:
            print(f"  {job['run_id']}/{job['tier']}: {job['data_path']}")
        print(f"Judge: {args.judge_model}; temperature=0; endpoint={judge_base_url}")
        return

    for eval_type in eval_types:
        model_dir = output_base / eval_type / args.model
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#'*60}\n# Eval type: {eval_type}  ->  {model_dir}\n{'#'*60}")

        if not args.summary_only:
            for idx, job in enumerate(all_jobs):
                run_output_dir = model_dir / job["tier"] / job["run_id"]
                print(f"\n[{idx+1}/{len(all_jobs)}] run_id={job['run_id']}  tier={job['tier']}")
                eval_single_run(
                    data_path=job["data_path"],
                    run_output_dir=run_output_dir,
                    run_id=job["run_id"], tier=job["tier"], eval_type=eval_type,
                    api_key=api_key, base_url=args.base_url, model=args.model,
                    judge_model=args.judge_model,
                    judge_api_key=judge_api_key, judge_base_url=judge_base_url,
                    sys_prompt=sys_prompt, judge_template=judge_template,
                    nugget_judge_template=nugget_judge_template if args.enable_judge else "",
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    timeout_seconds=args.timeout_seconds, max_retries=args.max_retries,
                    max_workers=args.max_workers, enable_judge=args.enable_judge,
                    resume=args.resume, include_persona=not args.no_persona,
                    judge_only=args.judge_only,                         # ── NEW
                    media_root=args.media_root,
                    heartbeat_seconds=args.heartbeat_seconds, print_every=args.print_every,
                )

        # per-model summary
        print(f"\n[Summary] Building model summary for type={eval_type} model={args.model} ...")
        model_agg = build_model_summary(model_dir, tiers)
        model_summary_path = model_dir / "aggregated_summary.json"
        with open(model_summary_path, "w", encoding="utf-8") as f:
            json.dump(model_agg, f, ensure_ascii=False, indent=2)
        print(f"[Summary] Written: {model_summary_path}")

        for tier in tiers:
            ts = model_agg["by_tier"].get(tier)
            if not ts: continue
            parts = [f"  tier={tier}: ok={ts['counts']['ok']}/{ts['counts']['total']}"]
            for k, label in [("EM_mean","EM"), ("F1_mean","F1"), ("judge_score_mean","judge"), ("nugget_recall_mean","nugget_recall")]:
                v = ts["metrics"].get(k)
                if v is not None: parts.append(f"{label}={v:.4f}")
            print("  ".join(parts))

        # top-level cross-model summary
        eval_type_dir = output_base / eval_type
        print(f"\n[Summary] Building top-level summary for type={eval_type} ...")
        top_agg = build_top_summary(eval_type_dir, tiers, eval_type)
        top_summary_path = eval_type_dir / "aggregated_summary.json"
        with open(top_summary_path, "w", encoding="utf-8") as f:
            json.dump(top_agg, f, ensure_ascii=False, indent=2)
        print(f"[Summary] Written: {top_summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
