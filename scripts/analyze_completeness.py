#!/usr/bin/env python3
"""
Analyze training data completeness — does each example show full task execution?

The v10 model did one tool call and stopped. This checks if the training
data teaches complete multi-step chains or truncated single-step responses.

For each task, reports:
- Avg tool calls per example
- % of examples that end with write_file (task completion)
- % with only 1 tool call (might teach "do one thing and stop")
- % with error recovery (tool error followed by retry)
- Avg assistant turns (more = more reasoning shown)
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config
from agents.base import TASK_IDS

cfg = load_config()


def extract_tool_calls(content):
    calls = []
    for block in re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
        try:
            calls.append(json.loads(block.strip()))
        except json.JSONDecodeError:
            calls.append({"name": "PARSE_ERROR"})
    return calls


def analyze_example(ex):
    msgs = ex.get("messages", [])
    assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]

    all_calls = []
    for m in assistant_msgs:
        all_calls.extend(extract_tool_calls(m["content"]))

    tool_names = [c.get("name", "?") for c in all_calls]

    # Check if last assistant message has content (not just a tool call)
    has_final_summary = False
    if assistant_msgs:
        last = assistant_msgs[-1]["content"]
        has_final_summary = "<tool_call>" not in last and len(last) > 20

    # Check for error recovery
    has_error_recovery = False
    for m in tool_msgs:
        content = m.get("content", "").lower()
        if "error" in content or "failed" in content or "not found" in content:
            has_error_recovery = True
            break

    # Check if example ends with a write (task completion)
    ends_with_write = any(n in ("write_file", "draft_email", "create_calendar_event",
                                "generate_image", "write_memory")
                          for n in tool_names[-2:]) if tool_names else False

    return {
        "n_tool_calls": len(all_calls),
        "n_assistant_turns": len(assistant_msgs),
        "tool_names": tool_names,
        "has_final_summary": has_final_summary,
        "has_error_recovery": has_error_recovery,
        "ends_with_write": ends_with_write,
        "single_tool": len(all_calls) <= 1,
        "total_chars": sum(len(m.get("content", "")) for m in msgs),
    }


def main():
    task_filter = sys.argv[1] if len(sys.argv) > 1 else None

    by_task = defaultdict(list)
    for line in cfg.train_file.read_text().splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        tid = ex.get("task_id", "?")
        if task_filter and tid != task_filter:
            continue
        by_task[tid].append(analyze_example(ex))

    print(f"\n{'Task':<35} {'count':>5} {'avg_tools':>9} {'1-tool%':>7} {'write%':>7} {'summary%':>8} {'errRecov%':>9} {'avg_chars':>9}")
    print("-" * 100)

    for task_id in TASK_IDS:
        analyses = by_task.get(task_id, [])
        if not analyses:
            print(f"  {task_id:<33} {'0':>5} {'—':>9} {'—':>7} {'—':>7} {'—':>8} {'—':>9} {'—':>9}")
            continue

        n = len(analyses)
        avg_tools = sum(a["n_tool_calls"] for a in analyses) / n
        pct_single = 100 * sum(1 for a in analyses if a["single_tool"]) / n
        pct_write = 100 * sum(1 for a in analyses if a["ends_with_write"]) / n
        pct_summary = 100 * sum(1 for a in analyses if a["has_final_summary"]) / n
        pct_error = 100 * sum(1 for a in analyses if a["has_error_recovery"]) / n
        avg_chars = sum(a["total_chars"] for a in analyses) / n

        flags = []
        if pct_single > 30:
            flags.append("HIGH-1TOOL")
        if pct_write < 50 and task_id != "task_00_sanity":
            flags.append("LOW-WRITE")
        if pct_summary < 50:
            flags.append("NO-SUMMARY")

        flag_str = "  ⚠ " + ", ".join(flags) if flags else ""

        print(f"  {task_id:<33} {n:>5} {avg_tools:>9.1f} {pct_single:>6.0f}% {pct_write:>6.0f}% "
              f"{pct_summary:>7.0f}% {pct_error:>8.0f}% {avg_chars:>9.0f}{flag_str}")

    print()


if __name__ == "__main__":
    main()
