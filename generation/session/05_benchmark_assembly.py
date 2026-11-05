#!/usr/bin/env python3

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_mm_ids(benchmark: dict, registry: dict) -> List[str]:
    unresolved = []

    for session in benchmark.get("multi_session_dialogues", []):
        for rnd in session.get("dialogues", []):
            for mm in rnd.get("mm", []):
                mm_id = mm.get("mm_id")
                if not mm_id:
                    continue
                if mm_id in registry:
                    mm["url"] = registry[mm_id]["url"]
                    mm["caption"] = registry[mm_id].get("caption", "")
                else:
                    unresolved.append(mm_id)

    for qa in benchmark.get("QAs", []):
        for ev in qa.get("evidence_chain", []):
            mm_id = ev.get("mm_id")
            if not mm_id:
                continue
            if mm_id in registry:
                ev["url"] = registry[mm_id]["url"]
                ev["caption"] = registry[mm_id].get("caption", "")
            else:
                unresolved.append(mm_id)

    return unresolved


def extract_sessions_from_evidence_chain(evidence_chain: List[Dict]) -> Set[str]:
    sessions = set()
    for evidence in evidence_chain:
        sid = evidence.get("session_id")
        if sid:
            sessions.add(sid)
    return sessions


def convert_dialogue_format(dialogue_session: Dict) -> Dict:
    session_output = {
        "session_id": dialogue_session["session_id"],
        "date": dialogue_session["date"],
        "dialogues": []
    }

    for dialogue_round in dialogue_session.get("dialogues", []):
        round_output = {
            "round": dialogue_round["round"],
            "user": dialogue_round["user"],
            "assistant": dialogue_round["assistant"]
        }

        mm_items = dialogue_round.get("mm", [])
        if mm_items:
            image_ids, input_images, image_captions = [], [], []
            video_ids, input_videos, video_captions = [], [], []
            audio_ids, input_audios, audio_captions = [], [], []

            processed_mm = []
            for mm_item in mm_items:
                processed_mm.append({
                    "mm_id": mm_item.get("mm_id", ""),
                    "type": mm_item.get("type", ""),
                    "description": mm_item.get("description", "") or mm_item.get("caption", ""),
                    "url": mm_item.get("url", ""),
                    "caption": mm_item.get("caption", ""),
                    "evidence_id": mm_item.get("evidence_id", ""),
                })

            round_output["mm"] = processed_mm

        session_output["dialogues"].append(round_output)

    return session_output


def convert_qa_format(qa: Dict, coverage_map: Dict) -> Dict:
    question_id = qa.get("question_id") or qa.get("id")

    answer_fallback = qa.get("answer", "")
    short_answer = qa.get("short_answer") or answer_fallback
    full_answer = qa.get("full_answer") or answer_fallback

    qa_output = {
        "question_id": question_id,
        "type": qa.get("task_type", ""),
        "question": qa.get("question", ""),
        "answer": short_answer,
        "full_answer": full_answer,
    }

    evidence_chain = qa.get("evidence_chain", [])
    sessions = sorted(list(extract_sessions_from_evidence_chain(evidence_chain)))
    qa_output["session_id"] = sessions
    qa_output["evidence_chain"] = evidence_chain

    clues = []
    for idx, evidence in enumerate(evidence_chain, start=1):
        evidence_id = f"{question_id}_hop{idx}"
        if evidence_id in coverage_map:
            round_id = coverage_map[evidence_id].get("round")
            if round_id and round_id not in clues:
                clues.append(round_id)

    qa_output["clue"] = sorted(clues)
    return qa_output


def convert_es_format(es_q: Dict) -> Dict:
    clues = []
    for nugget in es_q.get("nuggets", []):
        for src in nugget.get("source_rounds", []):
            sid = src.get("session_id", "")
            rnd = src.get("round")
            if sid and rnd is not None:
                clue = f"{sid}:{rnd}"
                if clue not in clues:
                    clues.append(clue)

    return {
        "question_id": es_q.get("question_id", ""),
        "type": "SUM",
        "subtype": es_q.get("subtype", ""),
        "question": es_q.get("question", ""),
        "answer": es_q.get("ideal_summary", ""),
        "nuggets": es_q.get("nuggets", []),
        "session_id": sorted(es_q.get("session_ids", [])),
        "clue": sorted(clues),
    }


def main():
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Benchmark Assembly")
    parser.add_argument("--run_id", type=str, default="all")
    parser.add_argument("--persona_file", type=Path, default=None)
    parser.add_argument("--dialogues_file", type=Path, default=None)
    parser.add_argument("--evidence_coverage_file", type=Path, default=None)
    parser.add_argument("--qa_file", type=Path, default=None)
    parser.add_argument("--es_file", type=Path, default=None)
    parser.add_argument("--registry_file", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=base_dir / "output" / "05_benchmark_assembly")
    args = parser.parse_args()

    run_id_arg = (args.run_id or "").strip()
    if run_id_arg.lower() == "all":
        if any([args.persona_file, args.dialogues_file, args.evidence_coverage_file, args.qa_file, args.es_file]):
            raise SystemExit("When --run_id is 'all', do not pass *_file overrides.")
        dialogues_root = base_dir / "output" / "02_dialogue_generation"
        if not dialogues_root.exists():
            raise SystemExit(f"dialogues root not found: {dialogues_root}")
        run_ids = sorted([p.name for p in dialogues_root.iterdir() if p.is_dir()])
        if not run_ids:
            raise SystemExit(f"no run_ids found under: {dialogues_root}")
    else:
        run_ids = [run_id_arg]

    print("=" * 60)
    print("Benchmark Assembly")
    print("=" * 60)
    print(f"run_ids: {', '.join(run_ids)}")

    collection_dir = args.output_root / "collection"
    collection_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0
    for run_id in run_ids:
        persona_file = args.persona_file or (
            base_dir.parent / "framework" / "output" / "01_persona_gen_caption" / run_id / "generated_persona_caption.json"
        )
        dialogues_file = args.dialogues_file or (base_dir / "output" / "02_dialogue_generation" / run_id / "all_dialogues.json")
        evidence_coverage_file = args.evidence_coverage_file or (
            base_dir / "output" / "02_dialogue_generation" / run_id / "evidence_coverage.json"
        )
        qa_file = args.qa_file or (
            base_dir.parent / "rewrite" / "output" / "04_final_qa" / run_id / "final_qa_aggregated.jsonl"
        )
        es_file = args.es_file or (
            base_dir / "output" / "04_es_generation" / run_id / "es_questions.jsonl"
        )
        registry_file = args.registry_file or (
            base_dir.parent / "filter" / "qa" / run_id / "mm_registry.json"
        )

        output_dir = args.output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        final_output = output_dir / "benchmark.json"
        collection_output = collection_dir / f"{run_id}.json"

        print("-" * 60)
        print(f"Processing run_id={run_id}")

        missing = []
        for name, path in [("persona_file", persona_file), ("dialogues_file", dialogues_file),
                           ("evidence_coverage_file", evidence_coverage_file), ("qa_file", qa_file)]:
            if not path.exists():
                missing.append(f"{name}: {path}")
        if missing:
            failed += 1
            for item in missing:
                print(f"  - {item}")
            continue

        persona = load_json(str(persona_file))
        dialogues_data = load_json(str(dialogues_file))
        evidence_coverage = load_json(str(evidence_coverage_file))
        qa_data = load_jsonl(str(qa_file))

        registry = {}
        if registry_file.exists():
            registry = load_json(str(registry_file))
            print(f"  Registry loaded: {len(registry)} entries")
        else:
            print(f"  ⚠ Registry not found: {registry_file}")

        es_data = []
        if es_file.exists():
            es_data = load_jsonl(str(es_file))
            print(f"  ES questions loaded: {len(es_data)}")


        coverage_map = evidence_coverage

        character_profile = {
            "persona_summary": persona.get("persona_summary", ""),
            "conversation_style": persona.get("conversation_style", ""),
        }

        multi_session_dialogues = []
        for session_data in dialogues_data.get("multi_session_dialogues", []):
            converted_session = convert_dialogue_format(session_data)
            multi_session_dialogues.append(converted_session)

        qas = []
        for qa in qa_data:
            converted_qa = convert_qa_format(qa, coverage_map)
            qas.append(converted_qa)

        for es_q in es_data:
            converted_es = convert_es_format(es_q)
            qas.append(converted_es)

        benchmark = {
            "character_profile": character_profile,
            "multi_session_dialogues": multi_session_dialogues,
            "QAs": qas,
        }

        if registry:
            unresolved = resolve_mm_ids(benchmark, registry)
            if unresolved:
                print(f"  ⚠ {len(unresolved)} unresolved mm_ids: {unresolved[:10]}")
            else:
                print(f"  ✓ All mm_ids resolved successfully")

        final_output.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
        collection_output.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
        ok += 1

        print(f"  Sessions: {len(multi_session_dialogues)}")
        print(f"  Questions: {len(qas)} (including {len(es_data)} ES)")

    print("-" * 60)
    print(f"Done. ok={ok}, skipped={failed}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()