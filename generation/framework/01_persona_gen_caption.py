#!/usr/bin/env python3

import os
import json
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "preprocess"))
from call_gemini import call_gemini


def load_content_groups(jsonl_file: str) -> tuple:
    formatted_groups = []
    category_counts = {}

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())

            for cat in data.get('categories', []):
                category_counts[cat] = category_counts.get(cat, 0) + 1

            parts = []
            mm_items = data.get('mm_items', [])
            if mm_items:
                parts.append("Multimodal Items:")
                for item in mm_items:
                    mm_id = item.get('mm_id', 'unknown')
                    mm_type = item.get('type', 'unknown')
                    caption = item.get('caption', '')
                    parts.append(f"  - {mm_id} [{mm_type}]: {caption}")
            else:
                ctx = data.get('multimodal_context', [])
                if ctx:
                    parts.append("Multimodal Context:")
                    for c in ctx:
                        parts.append(f"  - {c}")

            hint = data.get('hint', '')
            if hint:
                parts.append(f"Hint: {hint}")

            formatted_groups.append("\n".join(parts) if parts else "(no content)")

    return formatted_groups, category_counts


def build_domain_context(category_counts: dict) -> str:
    if not category_counts:
        return "No domain information available."
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    return "\n".join([f"- {cat} ({count} groups)" for cat, count in sorted_cats])


def load_prompt_template(template_file: str) -> str:
    with open(template_file, 'r', encoding='utf-8') as f:
        return f.read()


def format_prompt(template: str, groups: list, domain_context: str) -> str:
    groups_str = "\n\n".join([f"{i+1}. {g}" for i, g in enumerate(groups)])
    prompt = template.replace("{N}", str(len(groups)))
    prompt = prompt.replace("{questions_list}", groups_str)
    prompt = prompt.replace("{domain_context}", domain_context)
    return prompt


def generate_persona(prompt: str, model: str, temperature: float, max_tokens: int, max_retries: int) -> dict:
    success, response = call_gemini(
        prompt=prompt, model=model, temperature=temperature,
        max_tokens=max_tokens, max_retries=max_retries,
    )
    if not success:
        return None
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        return None


def save_persona(persona: dict, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)


def main():
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Persona Generation using Captioned Multi-Modal Data")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--output_root", type=Path, default=base_dir / "output")
    parser.add_argument("--qa_type_output_root", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--template_file", type=Path, default=base_dir / "prompt" / "01_persona_generation_prompt.txt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "your-model-name"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--max_retries", type=int, default=20)
    parser.add_argument("--preview", type=int, default=2)
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--print_prompt", action="store_true")
    args = parser.parse_args()

    qa_type_output_root = args.qa_type_output_root or (args.output_root / "00_qa_type_assign")
    input_file = args.input or (qa_type_output_root / args.run_id / "qa_with_assigned_types.jsonl")
    template_file = args.template_file

    output_dir = args.output_root / "01_persona_gen_caption" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output or (output_dir / "generated_persona_caption.json")

    print("=" * 80)
    print("Persona Generation using Captioned Multi-Modal Data")
    print("=" * 80)

    if not input_file.exists():
        sys.exit(1)
    if not template_file.exists():
        sys.exit(1)

    groups, category_counts = load_content_groups(str(input_file))
    if args.max_questions and args.max_questions > 0:
        groups = groups[:args.max_questions]
    if category_counts:
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    print()

    if args.preview and args.preview > 0:
        for i, g in enumerate(groups[:args.preview]):
            print(f"\n{i+1}. {g}")
        if len(groups) > args.preview:
            pass
        print()

    template = load_prompt_template(str(template_file))

    domain_context = build_domain_context(category_counts)
    prompt = format_prompt(template, groups, domain_context)

    if args.print_prompt:
        print("-" * 80)
        print(prompt)
        print("-" * 80)
        print()

    persona = generate_persona(prompt, args.model, args.temperature, args.max_tokens, args.max_retries)

    if persona:
        print(json.dumps(persona, indent=2, ensure_ascii=False))
        save_persona(persona, str(output_file))
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
