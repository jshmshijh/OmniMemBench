#!/usr/bin/env python3

import json, os, argparse
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent

TIER_ORDER = ["128k", "256k", "512k", "1m", "10m"]
TIER_TOKENS = {"128k": 128_000, "256k": 210_000, "512k": 470_000, "1m": 960_000, "10m": 10_000_000}


def resolve_distractor_mm_ids(sessions: List[dict], registry: dict) -> List[str]:
    unresolved = []
    for session in sessions:
        for rnd in session.get("dialogues", []):
            for mm in rnd.get("mm", []):
                mm_id = mm.get("mm_id", "")
                if mm_id in registry:
                    mm["url"] = registry[mm_id]["url"]
                    mm["caption"] = registry[mm_id].get("caption", "")
                elif mm_id:
                    unresolved.append(mm_id)
    return unresolved


def expand_tiers(tiers: List[str]) -> List[str]:
    if not tiers:
        return []
    max_idx = max(TIER_ORDER.index(t) for t in tiers if t in TIER_ORDER)
    return TIER_ORDER[:max_idx + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--tiers", nargs="+", default=None,
                        help="")
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (BASE_DIR / "output" / "09_tier_assembly" / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_file = BASE_DIR / "output" / "05_benchmark_assembly" / args.run_id / "benchmark.json"
    benchmark = json.loads(evidence_file.read_text("utf-8"))
    evidence_sessions = benchmark["multi_session_dialogues"]
    qas = benchmark["QAs"]
    profile = benchmark["character_profile"]

    evidence_token_file = BASE_DIR / "output" / "06_evidence_tokens" / args.run_id / "evidence_token_count.json"
    evidence_token_info = {}
    if evidence_token_file.exists():
        evidence_token_info = json.loads(evidence_token_file.read_text("utf-8"))
    actual_evidence_tokens = evidence_token_info.get("total_tokens", 0)

    dist_file = BASE_DIR / "output" / "08_distractor_gen" / args.run_id / "distractor_sessions.jsonl"
    distractor_sessions = []
    if dist_file.exists():
        for line in dist_file.read_text("utf-8").splitlines():
            if line.strip():
                s = json.loads(line)
                distractor_sessions.append(s)

    dist_registry_file = BASE_DIR / "output" / "07_distractor_pool" / args.run_id / "distractor_mm_registry.json"
    dist_registry = {}
    if dist_registry_file.exists():
        dist_registry = json.loads(dist_registry_file.read_text("utf-8"))

    ckpt_file = BASE_DIR / "output" / "08_distractor_gen" / args.run_id / "distractor_checkpoints.json"
    checkpoints = {}
    if ckpt_file.exists():
        checkpoints = json.loads(ckpt_file.read_text("utf-8"))

    if args.tiers:
        tiers = expand_tiers(args.tiers)
    else:
        available = [t for t in TIER_ORDER if t in checkpoints]
        tiers = expand_tiers(available) if available else ["128k"]

    print(f"Evidence sessions: {len(evidence_sessions)}")
    print(f"Evidence tokens: {actual_evidence_tokens}")
    print(f"Distractor sessions: {len(distractor_sessions)}")
    print(f"Checkpoints: {checkpoints}")

    for tier in tiers:
        target = TIER_TOKENS[tier]
        running = actual_evidence_tokens
        n_distractor = 0
        for sess in distractor_sessions:
            if running >= target:
                break
            running += sess.get("_tokens", 0)
            n_distractor += 1

        import copy
        tier_distractor = copy.deepcopy(distractor_sessions[:n_distractor])

        if tier_distractor and dist_registry:
            unresolved = resolve_distractor_mm_ids(tier_distractor, dist_registry)
            if unresolved:
                print(f"  ⚠ {tier}: {len(unresolved)} unresolved distractor mm_ids")

        distractor_tokens = sum(s.get("_tokens", 0) for s in tier_distractor)
        total_tokens = actual_evidence_tokens + distractor_tokens
        for s in tier_distractor:
            s.pop("_tokens", None)

        all_sessions = list(evidence_sessions) + tier_distractor
        all_sessions.sort(key=lambda s: s.get("date", ""))

        tier_benchmark = {
            "character_profile": profile,
            "multi_session_dialogues": all_sessions,
            "QAs": qas,
            "tier": tier,
            "total_tokens": total_tokens,
            "actual_evidence_tokens": actual_evidence_tokens,
            "evidence_sessions": len(evidence_sessions),
            "distractor_sessions": len(tier_distractor),
        }

        if actual_evidence_tokens > TIER_TOKENS[tier]:
            tier_benchmark["note"] = (
                f"evidence tokens ({actual_evidence_tokens}) exceed tier target ({TIER_TOKENS[tier]})"
            )

        out_file = output_dir / f"benchmark_{tier}.json"
        tmp = out_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(tier_benchmark, indent=2, ensure_ascii=False), "utf-8")
        os.replace(tmp, out_file)
        print(f"  ✓ {tier}: {len(all_sessions)} sessions ({len(tier_distractor)} distractor), {total_tokens} tokens → {out_file}")


if __name__ == "__main__":
    main()