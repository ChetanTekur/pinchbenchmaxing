#!/usr/bin/env python3
"""
Generate a single training example via real-time API for debugging.

Usage:
  python scripts/test_single_gen.py task_19_spreadsheet_summary
  python scripts/test_single_gen.py task_13_image_gen
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from utils.config import load_config
from datagen.task_loader import load_tasks
from datagen.dynamic_gen import build_dynamic_meta_prompt, VARIATION_CONFIGS, HARD_TASKS
from datagen.topup import parse_example, extract_json_array

cfg = load_config()

if len(sys.argv) < 2:
    print("Usage: python scripts/test_single_gen.py <task_id>")
    sys.exit(1)

task_id = sys.argv[1]
tasks = load_tasks()

if task_id not in tasks:
    print(f"Task '{task_id}' not found. Available: {sorted(tasks.keys())}")
    sys.exit(1)

task_def = tasks[task_id]
variation = VARIATION_CONFIGS[0]  # happy_formal
epc = 1  # just 1 example

print(f"Task: {task_id} — {task_def.get('name', '?')}")
print(f"Variation: {variation['id']}")

prompt = build_dynamic_meta_prompt(task_id, task_def, variation, epc)
print(f"Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")

# Show first/last bits of prompt for debugging
print(f"\n--- PROMPT START (first 500 chars) ---")
print(prompt[:500])
print(f"\n--- PROMPT END (last 300 chars) ---")
print(prompt[-300:])

print(f"\n--- Calling Claude (real-time API) ---")
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
max_tok = 16000 if task_id in HARD_TASKS else 8192

try:
    resp = client.messages.create(
        model=cfg.claude.generation,
        max_tokens=max_tok,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text
    print(f"Response: {len(raw)} chars, {resp.usage.input_tokens} in / {resp.usage.output_tokens} out")

    # Try to parse
    examples = extract_json_array(raw)
    if examples:
        print(f"\nParsed {len(examples)} examples")
        for i, ex in enumerate(examples):
            parsed = parse_example(ex, task_id)
            if parsed:
                msgs = parsed.get("messages", [])
                tools = [m for m in msgs if m["role"] == "assistant" and "<tool_call>" in m["content"]]
                print(f"\n  Example {i+1}:")
                print(f"    Messages: {len(msgs)}")
                print(f"    Tool calls: {len(tools)}")
                user = next((m["content"][:100] for m in msgs if m["role"] == "user"), "?")
                print(f"    User: {user}")
            else:
                print(f"\n  Example {i+1}: PARSE FAILED")
    else:
        print(f"\nFailed to parse JSON. Raw output (first 1000 chars):")
        print(raw[:1000])

except Exception as e:
    print(f"API error: {e}")
