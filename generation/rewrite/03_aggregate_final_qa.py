#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def build_subtype_lookup(qa_types_file: Path) -> Dict[str, str]:
    if not qa_types_file.exists():
        return {}
    lookup: Dict[str, str] = {}
    with open(qa_types_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("question_id")
            if not qid:
                continue
            subtype = obj.get("assigned_subtype")
            if subtype:
                lookup[str(qid)] = str(subtype)
    return lookup


def extract_from_validated(
    validated_data: List[Dict],
    task_type: str,
    subtype_lookup: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    results = []
    subtype_lookup = subtype_lookup or {}

    for item in validated_data:
        question_id = item.get('original_question_id', '')
        best = item.get('best_rewrite', {})

        task_subtype = best.get('task_subtype', '') or ""
        if task_type == "SH" and not task_subtype:
            task_subtype = subtype_lookup.get(str(question_id), "")

        short_answer = best.get("short_answer", "") or ""
        full_answer = best.get("full_answer", "") or ""

        output = {
            "question_id": question_id,
            "question": best.get('rewritten_question', ''),
            "answer": full_answer,
            "short_answer": short_answer,
            "full_answer": full_answer,
            "task_type": task_type,
            "task_subtype": task_subtype,
            "evidence_chain": best.get('evidence_chain', [])
        }
        results.append(output)

    return results


def extract_from_sh(qa_with_types: List[Dict]) -> List[Dict]:
    results = []

    for item in qa_with_types:
        if item.get('assigned_type') != 'SH':
            continue

        qid = item.get("question_id", "")

        mm_items = item.get("mm_items", [])
        evidence_chain: List[Dict[str, Any]] = []
        for mm_item in mm_items:
            evidence_chain.append({
                "type": mm_item.get("type", ""),
                "mm_id": mm_item.get("mm_id", ""),
                "evidence": mm_item.get("caption", ""),
            })

        original_answer = item.get("original_answer", "")
        output = {
            "question_id": qid,
            "question": item.get('question_text', ''),
            "answer": original_answer,
            "short_answer": original_answer,
            "full_answer": original_answer,
            "task_type": "SH",
            "task_subtype": item.get('assigned_subtype', ''),
            "evidence_chain": evidence_chain
        }
        results.append(output)

    return results


def main():
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Aggregate final QA data")
    parser.add_argument("--run_id", type=str, default="00001")
    args = parser.parse_args()

    validated_dir = base_dir / "output" / "03_qa_validated" / args.run_id
    qa_types_file = base_dir.parent / "framework" / "output" / "00_qa_type_assign" / args.run_id / "qa_with_assigned_types.jsonl"
    subtype_lookup = build_subtype_lookup(qa_types_file)

    output_dir = base_dir / "output" / "04_final_qa" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "final_qa_aggregated.jsonl"

    print("=" * 80)
    print("=" * 80)

    all_results = []

    type_configs = {
        'MH': 'mh_validated.jsonl',
        'CR': 'cr_validated.jsonl',
        'ADV': 'adv_validated.jsonl',
        'REF': 'ref_validated.jsonl',
        'TR': 'tr_validated.jsonl',
        'UPD': 'upd_validated.jsonl',
        'SH': 'sh_validated.jsonl',
    }

    sh_validated_loaded = False
    for task_type, filename in type_configs.items():
        filepath = validated_dir / filename
        if not filepath.exists():
            continue
        data = load_jsonl(str(filepath))
        results = extract_from_validated(data, task_type, subtype_lookup=subtype_lookup)
        all_results.extend(results)
        print(f"  ✓ {task_type}: {len(results)} ...")
        if task_type == "SH":
            sh_validated_loaded = True

    if not sh_validated_loaded:
        qa_with_types = load_jsonl(str(qa_types_file))
        sh_results = extract_from_sh(qa_with_types)
        all_results.extend(sh_results)
        print(f"  ✓ SH: {len(sh_results)} ...")
    else:
        pass

    all_results.sort(key=lambda x: int(x['question_id'][1:]) if x['question_id'].startswith('Q') else 0)

    seen: dict[str, int] = {}
    for idx, item in enumerate(all_results):
        seen[item['question_id']] = idx
    deduped = [all_results[i] for i in sorted(seen.values())]
    if len(deduped) < len(all_results):
        pass
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


    print("\n" + "=" * 80)
    type_counts = {}
    for item in all_results:
        t = item['task_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    for t in sorted(type_counts.keys()):
        pass


if __name__ == "__main__":
    main()