#!/usr/bin/env python3
"""
Compare structure of two datasets — avg turns, chars, tool calls per task.
Shows if the data "shape" is different between old and new.

Usage:
  python scripts/compare_structure.py OLD_TRAIN NEW_TRAIN
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

def extract_tool_count(messages):
    count = 0
    for m in messages:
        if m.get("role") == "assistant":
            count += len(re.findall(r'<tool_call>', m.get("content", "")))
    return count

def analyze_dataset(path):
    by_task = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
            tid = ex.get("task_id", "?")
            msgs = ex.get("messages", [])
            by_task[tid].append({
                "turns": len(msgs),
                "chars": sum(len(m.get("content", "")) for m in msgs),
                "tools": extract_tool_count(msgs),
                "has_system": any(m.get("role") == "system" for m in msgs),
            })
        except json.JSONDecodeError:
            continue
    return by_task

def main():
    old_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v8_backup/data/train.jsonl"
    new_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/synthbench/data/train.v12.bak"

    old = analyze_dataset(old_path)
    new = analyze_dataset(new_path)

    all_tasks = sorted(set(list(old.keys()) + list(new.keys())))

    print(f"\nOld: {old_path}")
    print(f"New: {new_path}\n")
    print(f"  {'Task':<35} {'old_n':>5} {'new_n':>5} | {'old_turns':>9} {'new_turns':>9} | {'old_tools':>9} {'new_tools':>9} | {'old_chars':>9} {'new_chars':>9}")
    print(f"  {'-'*35} {'-'*5} {'-'*5}   {'-'*9} {'-'*9}   {'-'*9} {'-'*9}   {'-'*9} {'-'*9}")

    for tid in all_tasks:
        o = old.get(tid, [])
        n = new.get(tid, [])

        o_n = len(o)
        n_n = len(n)
        o_turns = f"{sum(e['turns'] for e in o)/max(len(o),1):.1f}" if o else "-"
        n_turns = f"{sum(e['turns'] for e in n)/max(len(n),1):.1f}" if n else "-"
        o_tools = f"{sum(e['tools'] for e in o)/max(len(o),1):.1f}" if o else "-"
        n_tools = f"{sum(e['tools'] for e in n)/max(len(n),1):.1f}" if n else "-"
        o_chars = f"{sum(e['chars'] for e in o)/max(len(o),1):.0f}" if o else "-"
        n_chars = f"{sum(e['chars'] for e in n)/max(len(n),1):.0f}" if n else "-"

        print(f"  {tid:<35} {o_n:>5} {n_n:>5} | {o_turns:>9} {n_turns:>9} | {o_tools:>9} {n_tools:>9} | {o_chars:>9} {n_chars:>9}")

    # Check system prompts
    print(f"\n  System prompt check:")
    for label, data in [("old", old), ("new", new)]:
        has_sys = sum(1 for tid in data for e in data[tid] if e["has_system"])
        total = sum(len(v) for v in data.values())
        print(f"    {label}: {has_sys}/{total} examples have system message")

if __name__ == "__main__":
    main()
