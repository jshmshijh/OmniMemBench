#!/usr/bin/env python3
"""
Event Generation using Captioned Multi-Modal Data and Persona
"""

import os
import json
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "preprocess"))
from call_gemini import call_gemini


def load_persona(persona_file: str) -> dict:
    with open(persona_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_content_groups(jsonl_file: str) -> list:
    groups = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            data = json.loads(line.strip())
            question_id = data.get('question_id', f'Q{idx}')

            group = {
                'id': question_id,
                'captions': [],
                'assigned_type': data.get('assigned_type', 'N/A'),
                'assigned_subtype': data.get('assigned_subtype', 'N/A'),
            }

            mm_items = data.get('mm_items', [])
            if mm_items:
                for item in mm_items:
                    mm_id = item.get('mm_id', 'unknown')
                    mm_type = item.get('type', 'unknown')
                    caption = item.get('caption', '')
                    group['captions'].append(f"{mm_id} [{mm_type}]: {caption}")
            else:
                group['captions'] = data.get('multimodal_context', [])

            groups.append(group)
    return groups


def format_persona_info(persona: dict) -> str:
    lines = []
    if persona.get('primary_domain'):
        lines.append("- Primary Domain: " + persona['primary_domain'])
    lines.append("- Persona Summary: " + persona['persona_summary'])
    lines.append("- Conversation Style: " + persona['conversation_style'])
    return "\n".join(lines)


def format_questions_list(groups: list) -> str:
    formatted = []
    for g in groups:
        task_type = g.get('assigned_type', 'N/A')
        task_subtype = g.get('assigned_subtype', 'N/A')
        q_str = f"- {g['id']} [{task_type}/{task_subtype}]"
        for caption in g['captions']:
            q_str += f"\n  - {caption}"
        formatted.append(q_str)
    return "\n\n".join(formatted)


def load_prompt_template(template_file: str) -> str:
    with open(template_file, 'r', encoding='utf-8') as f:
        return f.read()


def format_prompt(template: str, persona: dict, groups: list) -> str:
    persona_info = format_persona_info(persona)
    questions_list = format_questions_list(groups)
    prompt = template.replace("{N}", str(len(groups)))
    prompt = prompt.replace("{persona_info}", persona_info)
    prompt = prompt.replace("{questions_list}", questions_list)
    return prompt


def generate_events(prompt: str, raw_output_file: str, model: str, temperature: float, max_tokens: int, max_retries: int) -> dict:
    success, response = call_gemini(
        prompt=prompt, model=model, temperature=temperature,
        max_tokens=max_tokens, max_retries=max_retries,
    )
    if not success:
        return None

    with open(raw_output_file, 'w', encoding='utf-8') as f:
        f.write(response)

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        return None


def save_events(events: dict, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)


def main():
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Event Generation using Captioned Multi-Modal Data and Persona")
    parser.add_argument("--run_id", type=str, default="00001")
    parser.add_argument("--output_root", type=Path, default=base_dir / "output")
    parser.add_argument("--persona_file", type=Path, default=None)
    parser.add_argument("--questions_file", type=Path, default=None)
    parser.add_argument("--template_file", type=Path, default=base_dir / "prompt" / "02_event_generation_prompt.txt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--raw_output", type=Path, default=None)
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "your-model-name"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--max_retries", type=int, default=20)
    parser.add_argument("--print_prompt", action="store_true")
    args = parser.parse_args()

    persona_file = args.persona_file or (args.output_root / "01_persona_gen_caption" / args.run_id / "generated_persona_caption.json")
    questions_file = args.questions_file or (args.output_root / "00_qa_type_assign" / args.run_id / "qa_with_assigned_types.jsonl")
    template_file = args.template_file

    output_dir = args.output_root / "02_event_gen_caption" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output or (output_dir / "generated_events.json")
    raw_output_file = args.raw_output or (output_dir / "generated_events_raw.txt")

    print("=" * 80)
    print("Event Generation using Captioned Multi-Modal Data and Persona")
    print("=" * 80)

    if not persona_file.exists():
        sys.exit(1)
    persona = load_persona(str(persona_file))
    print(f"✓ Persona: {persona.get('persona_summary', 'N/A')[:80]}...\n")

    if not questions_file.exists():
        sys.exit(1)
    groups = load_content_groups(str(questions_file))
    print(f"✓ {len(groups)} ...")

    for g in groups[:2]:
        print(f"\n{g['id']}:")
        for c in g['captions']:
            print(f"  - {c}")
    if len(groups) > 2:
        pass
    print()

    if not template_file.exists():
        sys.exit(1)
    template = load_prompt_template(str(template_file))
    print("✓\n")

    prompt = format_prompt(template, persona, groups)
    print(f"✓ {len(prompt)} ...")

    if args.print_prompt:
        print("-" * 80)
        print(prompt)
        print("-" * 80)
        print()

    events = generate_events(prompt, str(raw_output_file), args.model, args.temperature, args.max_tokens, args.max_retries)

    if events:
        print(json.dumps(events, indent=2, ensure_ascii=False))
        save_events(events, str(output_file))
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
