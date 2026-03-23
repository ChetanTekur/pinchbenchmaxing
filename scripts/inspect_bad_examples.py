#!/usr/bin/env python3
"""
Show actual tool calls from bad examples to understand what's wrong.

Usage:
  python scripts/inspect_bad_examples.py                    # top 5 worst tasks, 2 examples each
  python scripts/inspect_bad_examples.py task_01_calendar 5  # specific task, 5 examples
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config

cfg = load_config()
report_file = cfg.data_dir / "validation_report.json"
train_file = cfg.train_file


def extract_tool_calls(content):
    """Extract tool calls from assistant message content."""
    calls = []
    for block in re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
        try:
            obj = json.loads(block.strip())
            calls.append(obj)
        except json.JSONDecodeError:
            calls.append({"_parse_error": block.strip()[:100]})
    return calls


def main():
    task_filter = sys.argv[1] if len(sys.argv) > 1 else None
    n_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    if not report_file.exists():
        print("No validation_report.json found. Run: python -m datagen.validate_data")
        sys.exit(1)

    report = json.loads(report_file.read_text())
    issues_by_example = report.get("issues", [])

    # Load train data for lookup
    examples_by_key = {}
    for line in train_file.read_text().splitlines():
        if line.strip():
            ex = json.loads(line)
            task_id = ex.get("task_id", "?")
            msgs = ex.get("messages", [])
            user_msgs = [m["content"][:80] for m in msgs if m.get("role") == "user"]
            key = f"{task_id}|{user_msgs[0] if user_msgs else '?'}"
            examples_by_key[key] = ex

    # Group issues by task
    by_task = defaultdict(list)
    for item in issues_by_example:
        tid = item.get("task_id", "?")
        by_task[tid].append(item)

    # Pick tasks to show
    if task_filter:
        tasks_to_show = [task_filter]
    else:
        # Top 5 worst by issue count
        ranked = sorted(by_task.items(), key=lambda x: -len(x[1]))
        tasks_to_show = [t for t, _ in ranked[:5]]

    for task_id in tasks_to_show:
        items = by_task.get(task_id, [])
        if not items:
            print(f"\n{task_id}: no issues found")
            continue

        print(f"\n{'='*70}")
        print(f"  {task_id} — {len(items)} examples with issues")
        print(f"{'='*70}")

        # Summarize issue types
        issue_types = defaultdict(int)
        for item in items:
            for issue in item.get("issues", []):
                issue_types[issue.get("check", "?")] += 1
        print(f"  Issue types: {dict(issue_types)}")

        # Show N examples
        for i, item in enumerate(items[:n_examples]):
            print(f"\n  --- Example {i+1} ---")
            for issue in item.get("issues", []):
                print(f"  ⚠ [{issue.get('severity')}] {issue.get('check')}: {issue.get('detail')}")

            # Find the actual example and show tool calls
            ex_key = f"{task_id}|{item.get('user_prefix', '?')}"
            # Try to find by task_id match in the training data
            matching = [ex for ex in examples_by_key.values()
                        if ex.get("task_id") == task_id]

            if matching:
                # Find one that matches this issue's user message
                user_prefix = item.get("user_prefix", "")
                match = None
                for ex in matching:
                    msgs = ex.get("messages", [])
                    user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
                    if user_msgs and user_msgs[0][:60] == user_prefix[:60]:
                        match = ex
                        break

                if match:
                    msgs = match.get("messages", [])
                    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "?")
                    print(f"\n  User: {user_msg[:120]}...")

                    # Show all tool calls
                    for msg in msgs:
                        if msg["role"] == "assistant":
                            calls = extract_tool_calls(msg["content"])
                            for call in calls:
                                name = call.get("name", "?")
                                args = call.get("arguments", {})
                                arg_keys = list(args.keys())
                                print(f"  → Tool: {name}({', '.join(arg_keys)})")

        print()


if __name__ == "__main__":
    main()
