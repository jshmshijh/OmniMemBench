#!/usr/bin/env python3
import os
import argparse
import json
import secrets
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

BASE_DIR = Path(__file__).resolve().parent

TIER_TOKENS = {
    "128k": 108_000,
    "256k": 230_000,
    "512k": 460_000,
    "1m":   950_000,
    "10m": 9_500_000,
}

# Return type alias for check functions
CheckResult = tuple[bool, str]


# ── JSON / JSONL helpers ──────────────────────────────────────────────

def _jload(p: Path) -> dict:
    return json.loads(p.read_text("utf-8"))


def _jlines(p: Path):
    if not p.exists():
        return
    for line in p.read_text("utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _jcount(p: Path) -> int:
    return sum(1 for _ in _jlines(p)) if p.exists() else 0


def _jsonl_ok(
    path: Path,
    expected: Optional[int] = None,
    *,
    unique_key: Optional[str] = None,
    required_fields: tuple[str, ...] = (),
    pred: Optional[Callable[[dict], bool]] = None,
) -> CheckResult:
    if not path.exists():
        return False, f"file not found: {path}"
    ids, total = set(), 0
    try:
        for obj in _jlines(path):
            total += 1
            if unique_key:
                v = str(obj.get(unique_key, ""))
                if not v:
                    return False, f"{path.name}:line {total}: empty {unique_key}"
                if v in ids:
                    return False, f"{path.name}:line {total}: duplicate {unique_key}={v!r}"
                ids.add(v)
            for f in required_fields:
                if not str(obj.get(f, "")).strip():
                    return False, f"{path.name}:line {total}: missing/empty field {f!r}"
            if pred and not pred(obj):
                return False, f"{path.name}:line {total}: pred failed, keys={list(obj.keys())}"
    except Exception as e:
        return False, f"{path.name}: parse error: {e}"
    if total == 0:
        return False, f"{path.name}: empty file"
    if expected is not None and total != expected:
        return False, f"{path.name}: count mismatch: got {total}, expected {expected}"
    return True, ""


def _json_list_ok(
    path: Path,
    list_key: str,
    expected: Optional[int] = None,
    *,
    unique_key: str = "",
    required_fields: tuple[str, ...] = (),
    pred: Optional[Callable[[dict], bool]] = None,
) -> CheckResult:
    if not path.exists():
        return False, f"file not found: {path}"
    try:
        items = _jload(path).get(list_key, [])
    except Exception as e:
        return False, f"{path.name}: parse error: {e}"
    if not isinstance(items, list) or len(items) == 0:
        return False, f"{path.name}: {list_key!r} is empty or not a list"
    ids = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return False, f"{path.name}: {list_key}[{i}] is not a dict"
        if unique_key:
            v = str(item.get(unique_key, "")).strip()
            if not v:
                return False, f"{path.name}: {list_key}[{i}]: empty {unique_key}"
            if v in ids:
                return False, f"{path.name}: {list_key}[{i}]: duplicate {unique_key}={v!r}"
            ids.add(v)
        for f in required_fields:
            if not str(item.get(f, "")).strip():
                return False, f"{path.name}: {list_key}[{i}]: missing/empty field {f!r}"
        if pred and not pred(item):
            return False, f"{path.name}: {list_key}[{i}]: pred failed, keys={list(item.keys())}"
    if expected is not None and len(items) != expected:
        return False, f"{path.name}: {list_key} count mismatch: got {len(items)}, expected {expected}"
    return True, ""


# ── Domain-specific checks ───────────────────────────────────────────

def _sampled_ok(p: Path) -> CheckResult:
    return _jsonl_ok(p, unique_key="id",
                     pred=lambda o: isinstance(o.get("modalities", {}), dict))


def _registry_ok(reg: Path, data: Path, expected: int) -> CheckResult:
    if not reg.exists():
        return False, f"registry not found: {reg}"
    ok, reason = _jsonl_ok(data, expected)
    if not ok:
        return False, f"data check failed: {reason}"
    try:
        r = _jload(reg)
    except Exception as e:
        return False, f"registry parse error: {e}"
    if not all(isinstance(e, dict) and e.get("url") for e in r.values()):
        return False, "registry has entries without url"
    for i, obj in enumerate(_jlines(data)):
        for mod in ("audio", "video", "image"):
            for item in obj.get("modalities", {}).get(mod, []):
                if isinstance(item, dict) and "url" in item and not item.get("mm_id"):
                    return False, f"{data.name}:line {i+1}: {mod} item has url but no mm_id"
    return True, ""


def _qa_assign_ok(p: Path, expected: Optional[int] = None) -> CheckResult:
    return _jsonl_ok(p, expected,
                     pred=lambda o: o.get("api_status") == "success" and bool(o.get("top_types")))


def _qa_typed_ok(p: Path, expected: Optional[int] = None) -> CheckResult:
    return _jsonl_ok(p, expected, pred=lambda o: o.get("assigned_type") is not None)


def _load_type_map(p: Path) -> tuple[dict[str, str], str]:
    out: dict[str, str] = {}
    try:
        for obj in _jlines(p):
            qid, at = str(obj["question_id"]), str(obj["assigned_type"])
            if qid in out and out[qid] != at:
                return {}, f"conflicting type for question_id={qid}"
            out[qid] = at
    except Exception as e:
        return {}, f"load error: {e}"
    if not out:
        return {}, "empty type map"
    return out, ""


def _type_split_ok(directory: Path, qa_with_types: Path, suffix: str, id_key: str = "original_question_id",
                   extra_pred: Optional[Callable[[dict], bool]] = None) -> CheckResult:
    assigned, err = _load_type_map(qa_with_types)
    if not assigned:
        return False, f"type map load failed: {err}"
    for t in ("MH", "CR", "ADV", "REF", "TR", "UPD"):
        expected_ids = {qid for qid, at in assigned.items() if at == t}
        if not expected_ids:
            continue
        f = directory / f"{t.lower()}_{suffix}.jsonl"
        if not f.exists():
            return False, f"missing split file: {f.name}"
        got = set()
        for obj in _jlines(f):
            qid = obj.get(id_key) or obj.get("question_id")
            if extra_pred and not extra_pred(obj):
                continue
            if qid:
                got.add(str(qid))
        missing = expected_ids - got
        if missing:
            return False, f"{f.name}: missing {len(missing)} ids, e.g. {list(missing)[:3]}"
    return True, ""


def _persona_ok(p: Path) -> CheckResult:
    if not p.exists():
        return False, f"file not found: {p}"
    try:
        o = _jload(p)
    except Exception as e:
        return False, f"parse error: {e}"
    for f in ("persona_summary", "conversation_style", "primary_domain", "reason"):
        if not str(o.get(f, "")).strip():
            return False, f"missing/empty field {f!r}"
    return True, ""


def _events_ok(p: Path) -> CheckResult:
    return _json_list_ok(p, "graph", unique_key="id", required_fields=("sub-event", "date"),
                         pred=lambda o: isinstance(o.get("attached_question", []), list))


def _session_total(p: Path) -> int:
    if not p.exists():
        return 0
    try:
        s = _jload(p).get("sessions", [])
        return len(s) if isinstance(s, list) else 0
    except Exception:
        return 0


def _sessions1_ok(p: Path) -> CheckResult:
    n = _session_total(p)
    if n <= 0:
        return False, f"no sessions in {p}"
    return _json_list_ok(p, "sessions", n, unique_key="session_id",
                         required_fields=("session_id", "date", "session_summary"),
                         pred=lambda o: isinstance(o.get("questions_involved", []), list))


def _plans_ok(p: Path, s1: Path) -> CheckResult:
    n = _session_total(s1)
    if n <= 0:
        return False, f"no sessions in {s1}"
    if not p.exists():
        return False, f"file not found: {p}"
    try:
        o = _jload(p)
    except Exception as e:
        return False, f"parse error: {e}"
    total_s = int(o.get("total_sessions", 0))
    completed = int(o.get("completed", 0))
    if total_s != n:
        return False, f"total_sessions={total_s}, expected {n}"
    if completed != n:
        return False, f"completed={completed}, expected {n}"
    return _json_list_ok(p, "session_plans", n, unique_key="session_id",
                         pred=lambda x: isinstance(x.get("dialogue_flow", {}), dict))


def _dialogues_ok(dp: Path, cp: Path, s1: Path) -> CheckResult:
    n = _session_total(s1)
    if n <= 0:
        return False, f"no sessions in {s1}"
    if not dp.exists():
        return False, f"dialogues not found: {dp}"
    if not cp.exists():
        return False, f"coverage not found: {cp}"
    try:
        _jload(cp)
    except Exception as e:
        return False, f"coverage parse error: {e}"

    def _round_ok(s):
        ds = s.get("dialogues", [])
        return isinstance(ds, list) and len(ds) > 0 and all(
            isinstance(r, dict) and str(r.get("user", "")).strip() and str(r.get("assistant", "")).strip()
            for r in ds)

    return _json_list_ok(dp, "multi_session_dialogues", n, unique_key="session_id", pred=_round_ok)


def _summary_ok(p: Path, s1: Path) -> CheckResult:
    n = _session_total(s1)
    if n <= 0:
        return False, f"no sessions in {s1}"
    return _json_list_ok(p, "sessions", n, required_fields=("session_id", "session_summary"))


def _final_qa_ok(p: Path, expected: int) -> CheckResult:
    if expected <= 0:
        return False, f"expected count is {expected}"
    return _jsonl_ok(p, expected, unique_key="question_id")


def _es_ok(p: Path, min_q: int = 1) -> CheckResult:
    if not p.exists():
        return False, f"file not found: {p}"
    n = 0
    for i, obj in enumerate(_jlines(p)):
        if not obj.get("question_id"):
            return False, f"{p.name}:line {i+1}: missing question_id"
        if not obj.get("nuggets"):
            return False, f"{p.name}:line {i+1}: missing nuggets"
        n += 1
    if n < min_q:
        return False, f"{p.name}: only {n} questions, need >= {min_q}"
    return True, ""


def _bench_ok(p: Path, exp_q: int, exp_s: int) -> CheckResult:
    if exp_q <= 0:
        return False, f"expected questions={exp_q}"
    if exp_s <= 0:
        return False, f"expected sessions={exp_s}"
    if not p.exists():
        return False, f"file not found: {p}"
    try:
        o = _jload(p)
    except Exception as e:
        return False, f"parse error: {e}"
    if not isinstance(o.get("character_profile", {}), dict):
        return False, "missing/invalid character_profile"
    ok_q, reason_q = _json_list_ok(p, "QAs", exp_q, unique_key="question_id", required_fields=("question",))
    if not ok_q:
        return False, f"QAs check: {reason_q}"
    ok_s, reason_s = _json_list_ok(p, "multi_session_dialogues", exp_s, unique_key="session_id")
    if not ok_s:
        return False, f"sessions check: {reason_s}"
    return True, ""


def _audit_summary_ok(p: Path, expected_total: int) -> CheckResult:
    if expected_total <= 0:
        return False, f"expected_total={expected_total}"
    if not p.exists():
        return False, f"file not found: {p}"
    try:
        o = _jload(p)
    except Exception as e:
        return False, f"parse error: {e}"
    if not isinstance(o, dict):
        return False, "not a dict"
    got = o.get("total", 0)
    if got != expected_total:
        return False, f"total={got}, expected {expected_total}"
    return True, ""


# ── Distractor checks ────────────────────────────────────────────────

def _evidence_tokens_ok(p: Path) -> CheckResult:
    if not p.exists():
        return False, f"file not found: {p}"
    try:
        o = _jload(p)
        t = o.get("total_tokens")
        if not isinstance(t, (int, float)) or t <= 0:
            return False, f"invalid total_tokens={t}"
    except Exception as e:
        return False, f"parse error: {e}"
    return True, ""


def _distractor_gen_ok(checkpoints_path: Path, target_tier: str) -> CheckResult:
    if not checkpoints_path.exists():
        return False, f"file not found: {checkpoints_path}"
    try:
        ckpt = _jload(checkpoints_path)
        if target_tier not in ckpt:
            return False, f"tier {target_tier!r} not in checkpoints (keys={list(ckpt.keys())})"
    except Exception as e:
        return False, f"parse error: {e}"
    return True, ""


def _tier_assembly_ok(tier_dir: Path, target_tier: str) -> CheckResult:
    f = tier_dir / f"benchmark_{target_tier}.json"
    if not f.exists():
        return False, f"file not found: {f}"
    return True, ""


# ── Uniqueness / filter checks ───────────────────────────────────────

def _uniqueness_summary_ok(p: Path, expected_total: int) -> CheckResult:
    if expected_total <= 0:
        return False, f"expected_total={expected_total}"
    if not p.exists():
        return False, f"file not found: {p}"
    try:
        o = _jload(p)
    except Exception as e:
        return False, f"parse error: {e}"
    if not isinstance(o, dict):
        return False, "not a dict"
    got = o.get("total", 0)
    if got != expected_total:
        return False, f"total={got}, expected {expected_total}"
    return True, ""


def _filtered_ok(filtered_dir: Path, tier_dir: Path, target_tier: str) -> CheckResult:
    f = filtered_dir / "benchmark.json"
    if not f.exists():
        return False, f"file not found: {f}"
    tier_src = tier_dir / f"benchmark_{target_tier}.json"
    if tier_src.exists():
        tier_dst = filtered_dir / f"benchmark_{target_tier}.json"
        if not tier_dst.exists():
            return False, f"tier file missing: {tier_dst}"
    return True, ""


# ── Paths container ──────────────────────────────────────────────────

class Paths:
    def __init__(self, run_id: str):
        b = BASE_DIR
        q = b / "filter" / "qa" / run_id
        s2 = b / "framework" / "output"
        s3 = b / "rewrite" / "output"
        s4 = b / "session" / "output"
        a = s2 / "00_qa_type_assign" / run_id

        self.sampled = q / "sampled_data.jsonl"
        self.captioned = q / "sampled_data_captioned.jsonl"
        self.caption_failed = q / "sampled_data_captioned_failed.jsonl"
        self.mm_registry = q / "mm_registry.json"
        self.sampled_mmid = q / "sampled_data_with_mmid.jsonl"
        self.generated_assign = a / "generated_qa_assignments.jsonl"
        self.qa_with_types = a / "qa_with_assigned_types.jsonl"
        self.persona = s2 / "01_persona_gen_caption" / run_id / "generated_persona_caption.json"
        self.events = s2 / "02_event_gen_caption" / run_id / "generated_events.json"
        self.sessions1 = s2 / "03_session_gen_stage1" / run_id / "generated_sessions_stage1.json"
        self.rewrites_dir = s3 / "02_qa_rewrite" / run_id
        self.validated_dir = s3 / "03_qa_validated" / run_id
        self.final_qa = s3 / "04_final_qa" / run_id / "final_qa_aggregated.jsonl"
        self.plans = s4 / "01_session_dialogue_planning" / run_id / "all_session_plans.json"
        self.dialogues = s4 / "02_dialogue_generation" / run_id / "all_dialogues.json"
        self.evidence_cov = s4 / "02_dialogue_generation" / run_id / "evidence_coverage.json"
        self.summaries = s4 / "03_post_hoc_summary" / run_id / "post_hoc_summaries.json"
        self.es = s4 / "04_es_generation" / run_id / "es_questions.jsonl"
        self.benchmark = s4 / "05_benchmark_assembly" / run_id / "benchmark.json"
        self.evidence_tokens = s4 / "06_evidence_tokens" / run_id / "evidence_token_count.json"
        self.distractor_sessions = s4 / "08_distractor_gen" / run_id / "distractor_sessions.jsonl"
        self.distractor_checkpoints = s4 / "08_distractor_gen" / run_id / "distractor_checkpoints.json"
        self.tier_assembly_dir = s4 / "09_tier_assembly" / run_id
        self._q = q
        s5 = b / "evaluate" / "output"
        self.qa_audit_summary = s5 / "01_qa_audit" / run_id / "qa_audit_summary.json"
        self.embed_audit_summary = s5 / "02_embed_audit" / run_id / "embed_audit_summary.json"
        self.uniqueness_audit_summary = s5 / "03_uniqueness_audit" / run_id / "uniqueness_audit_summary.json"
        self.filtered_dir = s5 / "04_filtered" / run_id


# ── Runner ───────────────────────────────────────────────────────────

def _run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] CMD: {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=log, text=True)
        log.write(f"[exit={p.returncode}] elapsed={time.time()-start:.2f}s\n")
        return p.returncode


def _make_seed(state_file: Path) -> int:
    if state_file.exists():
        try:
            return int(_jload(state_file)["seed"])
        except Exception:
            pass
    seed = secrets.randbelow(2**31 - 1)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"seed": seed}), "utf-8")
    return seed


def _log(rid: str, msg: str):
    print(f"[{time.strftime('%H:%M:%S')}][{rid}] {msg}", flush=True)


def _log_path(logs_dir: Path, step_name: str) -> Path:
    cat, _, rest = step_name.partition("_")
    return logs_dir / cat / f"{rest or cat}.log"


@dataclass(frozen=True)
class Step:
    name: str
    cmd: list[str]
    ok: Callable[[], CheckResult]


def _build_steps(rid: str, P: Paths, seed: int, target_tier: str = "10m") -> list[Step]:
    py = sys.executable
    b = BASE_DIR

    def s(script: str, *args: str) -> list[str]:
        return [py, str(b / script), *args]

    return [
        # ── Step 1: Data Filter ──
        Step("step1_sample",
             s("filter/02_sample_data.py",
               "--output_root", str(P._q.parent), "--run_id", rid, "--seed", str(seed)),
             lambda: _sampled_ok(P.sampled)),
        Step("step1_caption",
             s("filter/03_generate_captions.py",
               "--output_root", str(P._q.parent), "--run_id", rid),
             lambda: (lambda r: r if not r[0] else
                      ((False, f"caption_failed exists: {P.caption_failed}") if P.caption_failed.exists() else r))
                     (_jsonl_ok(P.captioned, _jcount(P.sampled)))),
        Step("step1_registry",
             s("filter/04_build_mm_registry.py",
               "--input", str(P.captioned), "--output_dir", str(P._q)),
             lambda: _registry_ok(P.mm_registry, P.sampled_mmid, _jcount(P.captioned))),

        # ── Step 2: Framework Gen ──
        Step("step2_assign",
             s("framework/00_qa_type_assign.py",
               "--run_id", rid, "--batch_size", "5", "--api_retries", "20",
               "--model", os.environ.get("LLM_MODEL", "your-model-name")),
             lambda: _qa_assign_ok(P.generated_assign, _jcount(P.captioned))),
        Step("step2_reassign",
             s("framework/00_qa_type_reassign.py", "--run_id", rid),
             lambda: _qa_typed_ok(P.qa_with_types, _jcount(P.captioned))),
        Step("step2_persona",
             s("framework/01_persona_gen_caption.py", "--run_id", rid, "--max_retries", "20"),
             lambda: _persona_ok(P.persona)),
        Step("step2_event",
             s("framework/02_event_gen_caption.py", "--run_id", rid, "--max_retries", "20"),
             lambda: _events_ok(P.events)),
        Step("step2_sessions",
             s("framework/03_session_gen_stage1.py", "--run_id", rid, "--max_retries", "20"),
             lambda: _sessions1_ok(P.sessions1)),

        # ── Step 3: QA Rewrite ──
        Step("step3_rewrite",
             s("rewrite/01_qa_rewrite.py", "--run_id", rid,
             "--multimodal",
               "--question_retries", "20", "--api_retries", "20", "--retry_wait_cap_s", "30"),
             lambda: _type_split_ok(P.rewrites_dir, P.qa_with_types, "rewrites",
                                    extra_pred=lambda o: o.get("api_status") == "success")),
        Step("step3_select",
             s("rewrite/02_qa_select.py", "--run_id", rid,
               "--max_retries", "20", "--api_retries", "20", "--retry_wait_cap_s", "30"),
             lambda: _type_split_ok(P.validated_dir, P.qa_with_types, "validated")),
        Step("step3_aggregate",
             s("rewrite/03_aggregate_final_qa.py", "--run_id", rid),
             lambda: _final_qa_ok(P.final_qa, _jcount(P.captioned))),

        # ── Step 4: Session Gen ──
        Step("step4_plan",
             s("session/01_session_dialogue_planning.py", "--run_id", rid),
             lambda: _plans_ok(P.plans, P.sessions1)),
        Step("step4_dialogue",
             s("session/02_dialogue_generation.py", "--run_id", rid),
             lambda: _dialogues_ok(P.dialogues, P.evidence_cov, P.sessions1)),
        Step("step4_summary",
             s("session/03_post_hoc_summary.py", "--run_id", rid,
               "--dialogues_file", str(P.dialogues)),
             lambda: _summary_ok(P.summaries, P.sessions1)),
        Step("step4_es",
             s("session/04_es_generation.py", "--run_id", rid,
               "--dialogues_file", str(P.dialogues)),
             lambda: _es_ok(P.es)),
        Step("step4_assemble",
             s("session/05_benchmark_assembly.py", "--run_id", rid,
               "--dialogues_file", str(P.dialogues),
               "--evidence_coverage_file", str(P.evidence_cov)),
             lambda: _bench_ok(P.benchmark,
                               _jcount(P.final_qa) + _jcount(P.es),
                               _session_total(P.sessions1))),

        # ── Step 4 (cont.): Distractor ──
        Step("step4_evidence-tokens",
             s("session/06_compute_evidence_tokens.py", "--run_id", rid),
             lambda: _evidence_tokens_ok(P.evidence_tokens)),
        Step("step4_distractor-gen",
             s("session/08_distractor_generation.py",
               "--run_id", rid, "--target_tier", target_tier),
             lambda: _distractor_gen_ok(P.distractor_checkpoints, target_tier)),
        Step("step4_tier-assembly",
             s("session/09_tier_benchmark_assembly.py", "--run_id", rid),
             lambda: _tier_assembly_ok(P.tier_assembly_dir, target_tier)),

        # ── Step 5: Evaluation ──
        Step("step5_qa-audit",
             s("evaluate/01_qa_audit.py", "--run_id", rid,
               "--input", str(P.final_qa)),
             lambda: _audit_summary_ok(P.qa_audit_summary, _jcount(P.final_qa))),
        Step("step5_embed-audit",
             s("evaluate/02_embed_audit.py", "--run_id", rid,
               "--qa_file", str(P.final_qa),
               "--dialogues_file", str(P.dialogues)),
             lambda: _audit_summary_ok(P.embed_audit_summary, _jcount(P.final_qa))),
        Step("step5_uniqueness-audit",
             s("evaluate/03_uniqueness_audit.py", "--run_id", rid,
               "--benchmark", str(P.benchmark)),
             lambda: _uniqueness_summary_ok(P.uniqueness_audit_summary, _jcount(P.final_qa) + _jcount(P.es))),
        Step("step5_filter",
             s("evaluate/04_filter.py", "--run_id", rid),
             lambda: _filtered_ok(P.filtered_dir, P.tier_assembly_dir, target_tier)),
    ]


def run_one(
    run_id: str,
    target_tier: str = "10m",
    approval_note: str = "",
    max_step_attempts: int = 20,
    max_retry_wait_s: int = 30,
) -> None:
    logs_dir = BASE_DIR / "_pipeline_logs" / run_id
    seed = _make_seed(logs_dir / "state.json")
    P = Paths(run_id)

    _n_qa = _jcount(P.captioned)
    _n_total_qa = _n_qa + _jcount(P.es)
    if (all(ok for ok, _ in [
        _bench_ok(P.benchmark, _n_total_qa, _session_total(P.sessions1)),
        _tier_assembly_ok(P.tier_assembly_dir, target_tier),
        _audit_summary_ok(P.qa_audit_summary, _n_qa),
        _audit_summary_ok(P.embed_audit_summary, _n_qa),
        _uniqueness_summary_ok(P.uniqueness_audit_summary, _n_total_qa),
        _filtered_ok(P.filtered_dir, P.tier_assembly_dir, target_tier),
    ])):
        _log(run_id, "already complete, skipping")
        return

    _log(run_id, f"starting  seed={seed}  target_tier={target_tier}")
    t0 = time.time()

    for step in _build_steps(run_id, P, seed, target_tier):
        attempt = 0
        while True:
            ok, reason = step.ok()
            if ok:
                _log(run_id, f"{step.name} {'done after '+str(attempt)+' attempt(s)' if attempt else 'already satisfied, skipping'}")
                break
            if 0 < max_step_attempts <= attempt:
                raise RuntimeError(f"{step.name} exceeded max attempts ({max_step_attempts}), last reason: {reason}")
            attempt += 1
            if attempt > 1:
                _log(run_id, f"{step.name} check failed: {reason}")
            _log(run_id, f"{step.name} attempt={attempt}")
            rc = _run_cmd(step.cmd, _log_path(logs_dir, step.name))
            if rc in {-signal.SIGINT, -signal.SIGTERM, 130, 143}:
                _log(run_id, f"{step.name} interrupted (exit={rc}), aborting")
                raise KeyboardInterrupt
            ok2, reason2 = step.ok()
            if rc == 0 and ok2:
                _log(run_id, f"{step.name} done")
                break
            wait = min(5 * (2 ** min(attempt, 6)), max(1, max_retry_wait_s))
            _log(run_id, f"{step.name} not ready (exit={rc}): {reason2}  retry in {wait}s")
            time.sleep(wait)

    _log(run_id, f"pipeline complete elapsed={time.time()-t0:.2f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=1)
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--target_tier", default="256k", choices=list(TIER_TOKENS.keys()))
    p.add_argument("--max_step_attempts", type=int, default=20)
    p.add_argument("--max_retry_wait_s", type=int, default=30)
    args = p.parse_args()

    lo, hi = min(args.start, args.end), max(args.start, args.end)
    run_ids = [f"{i:05d}" for i in range(lo, hi + 1)]
    jobs = max(1, args.jobs)

    if jobs == 1:
        try:
            for rid in run_ids:
                run_one(rid, target_tier=args.target_tier,
                        max_step_attempts=args.max_step_attempts,
                        max_retry_wait_s=args.max_retry_wait_s)
        except KeyboardInterrupt:
            print("[INTERRUPTED]", flush=True)
            raise SystemExit(130)
        return

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_one, rid, args.target_tier, "",
                          args.max_step_attempts, args.max_retry_wait_s): rid
                for rid in run_ids}
        try:
            for fut in as_completed(futs):
                fut.result()
                print(f"[DONE] {futs[fut]}", flush=True)
        except KeyboardInterrupt:
            print("[INTERRUPTED] terminating...", flush=True)
            for fut in futs:
                fut.cancel()
            for proc in getattr(ex, "_processes", {}).values():
                if proc.is_alive():
                    proc.terminate()
            raise SystemExit(130)


if __name__ == "__main__":
    main()
