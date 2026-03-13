#!/usr/bin/env python3
"""
Dataset inspector and validator.

Usage:
  python inspect_data.py stats              # summary stats across the dataset
  python inspect_data.py sample [N]         # print N random examples (default 3)
  python inspect_data.py task <task_id>     # show all examples for one task
  python inspect_data.py validate           # run quality checks, flag bad examples
  python inspect_data.py validate --clean   # same + remove flagged examples
"""

import json, random, re, argparse, sys
from pathlib import Path
from collections import defaultdict

TRAIN_FILE = Path("/workspace/data/train.jsonl")
VAL_FILE   = Path("/workspace/data/val.jsonl")

VALID_TOOLS = {
    "read_file", "write_file", "create_directory", "list_files",
    "run_bash", "run_python", "web_search", "fetch_url",
    "create_calendar_event", "draft_email", "search_emails", "read_email",
    "generate_image", "read_memory", "write_memory",
    "search_skills", "install_skill",
}

# Tasks that MUST contain a tool call (not pure text responses)
TOOL_REQUIRED_TASKS = {
    "task_01_calendar", "task_02_stock", "task_04_weather",
    "task_05_summary", "task_06_events", "task_07_email",
    "task_08_memory", "task_09_files", "task_10_workflow",
    "task_11_config_update", "task_12_skill_search", "task_13_image_gen",
    "task_14_humanizer", "task_15_daily_summary", "task_16_email_triage",
    "task_17_email_search", "task_18_market_research", "task_19_spreadsheet_summary",
    "task_20_eli5_pdf", "task_21_openclaw_comprehension", "task_22_second_brain",
}

# Tasks with specific expected values in the output
EXPECTED_VALUES = {
    "task_08_memory":             ["June 1, 2024"],
    "task_21_openclaw_comprehension": ["5,705", "2,999", "SKILL.md", "February 7, 2026"],
    "task_19_spreadsheet_summary": ["119,900", "47,960", "Alice Chen", "Widget B"],
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_records(include_val=True) -> list[dict]:
    records = []
    for path in ([TRAIN_FILE, VAL_FILE] if include_val else [TRAIN_FILE]):
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def display_example(rec: dict, index: int = 0):
    task_id  = rec.get("task_id", "unknown")
    messages = rec.get("messages", [])

    print(f"\n{'═'*60}")
    print(f"  Example #{index}  |  Task: {task_id}")
    print(f"{'═'*60}")

    for msg in messages:
        role    = msg["role"]
        content = msg["content"]

        if role == "system":
            # Just show first 120 chars of system prompt
            print(f"\n[SYSTEM] {content[:120]}...")
            continue

        label = {
            "user":      "👤 USER",
            "assistant": "🤖 ASSISTANT",
            "tool":      "🔧 TOOL RESULT",
        }.get(role, f"[{role.upper()}]")

        print(f"\n{label}")
        print("─" * 40)

        # Highlight tool calls
        if "<tool_call>" in content:
            parts = content.split("<tool_call>")
            print(parts[0].strip())
            for part in parts[1:]:
                call, *rest = part.split("</tool_call>", 1)
                print(f"\033[93m<tool_call>{call}</tool_call>\033[0m")  # yellow
                if rest:
                    print(rest[0].strip())
        else:
            # Truncate very long content
            if len(content) > 800:
                print(content[:800] + f"\n  ... [{len(content)-800} chars truncated]")
            else:
                print(content)

    print(f"\n{'─'*60}")
    print(f"  Turns: {len(messages)}  |  "
          f"Total chars: {sum(len(m['content']) for m in messages):,}")


# ─────────────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────────────
def cmd_stats():
    records = load_records()
    by_task = defaultdict(list)
    for r in records:
        by_task[r.get("task_id", "unknown")].append(r)

    total_chars = 0
    total_turns = 0
    for r in records:
        msgs = r.get("messages", [])
        total_turns += len(msgs)
        total_chars += sum(len(m["content"]) for m in msgs)

    print(f"\n{'═'*50}")
    print(f"  DATASET STATS")
    print(f"{'═'*50}")
    print(f"  Train file:   {TRAIN_FILE}")
    print(f"  Val file:     {VAL_FILE}")
    train_count = sum(1 for _ in TRAIN_FILE.read_text().splitlines() if _.strip())
    val_count   = sum(1 for _ in VAL_FILE.read_text().splitlines()   if _.strip())
    print(f"  Train:        {train_count} examples")
    print(f"  Val:          {val_count} examples")
    print(f"  Total:        {len(records)} examples")
    print(f"  Avg turns:    {total_turns/len(records):.1f} per example")
    print(f"  Avg chars:    {total_chars/len(records):,.0f} per example")
    print(f"  Total tokens: ~{total_chars//4:,} (rough estimate)")
    print()
    print(f"  {'Task':<40} {'Count':>6}")
    print(f"  {'─'*40} {'─'*6}")
    for task_id in sorted(by_task):
        count = len(by_task[task_id])
        bar   = "█" * (count // 3)
        print(f"  {task_id:<40} {count:>5}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE
# ─────────────────────────────────────────────────────────────────────────────
def cmd_sample(n: int = 3):
    records = load_records()
    chosen  = random.sample(records, min(n, len(records)))
    for i, rec in enumerate(chosen):
        display_example(rec, index=i + 1)


# ─────────────────────────────────────────────────────────────────────────────
# TASK VIEW
# ─────────────────────────────────────────────────────────────────────────────
def cmd_task(task_id: str):
    records = load_records()
    matches = [r for r in records if r.get("task_id") == task_id]
    if not matches:
        # fuzzy: show available task IDs
        all_ids = sorted({r.get("task_id") for r in records})
        print(f"Task '{task_id}' not found. Available tasks:")
        for t in all_ids:
            print(f"  {t}")
        return
    print(f"\nFound {len(matches)} examples for {task_id}")
    for i, rec in enumerate(matches):
        display_example(rec, index=i + 1)
        if i < len(matches) - 1:
            inp = input("\n  [Enter] next  |  [q] quit  > ").strip().lower()
            if inp == "q":
                break


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
def extract_tool_names(content: str) -> list[str]:
    """Extract tool names from <tool_call> blocks in an assistant message."""
    names = []
    for block in re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
        try:
            obj = json.loads(block.strip())
            if "name" in obj:
                names.append(obj["name"])
        except json.JSONDecodeError:
            names.append("__INVALID_JSON__")
    return names


def validate_record(rec: dict) -> list[str]:
    """Return a list of issue strings. Empty list = clean."""
    issues = []
    task_id  = rec.get("task_id", "unknown")
    messages = rec.get("messages", [])

    # 1. Must have system + user + at least one assistant message
    roles = [m["role"] for m in messages]
    if roles[0] != "system":
        issues.append("missing system message")
    if "user" not in roles:
        issues.append("missing user message")
    if "assistant" not in roles:
        issues.append("missing assistant message")
        return issues  # can't do further checks

    # 2. Last message must be assistant
    if messages[-1]["role"] != "assistant":
        issues.append(f"last message is '{messages[-1]['role']}', not assistant")

    # 3. User message must not be empty
    user_msgs = [m for m in messages if m["role"] == "user"]
    if any(not m["content"].strip() for m in user_msgs):
        issues.append("empty user message")

    # 4. Check for tool calls in tasks that require them
    assistant_contents = " ".join(
        m["content"] for m in messages if m["role"] == "assistant"
    )
    if task_id in TOOL_REQUIRED_TASKS and "<tool_call>" not in assistant_contents:
        issues.append("no tool calls found (required for this task)")

    # 5. Validate tool names in tool calls
    all_tool_names = []
    for msg in messages:
        if msg["role"] == "assistant":
            all_tool_names.extend(extract_tool_names(msg["content"]))

    invalid_tools = [t for t in all_tool_names if t not in VALID_TOOLS and t != "__INVALID_JSON__"]
    bad_json_tools = all_tool_names.count("__INVALID_JSON__")
    if invalid_tools:
        issues.append(f"unknown tool names: {set(invalid_tools)}")
    if bad_json_tools:
        issues.append(f"{bad_json_tools} tool call(s) with invalid JSON")

    # 6. Check for expected values in tasks with known correct answers
    full_text = " ".join(m["content"] for m in messages)
    if task_id in EXPECTED_VALUES:
        for expected in EXPECTED_VALUES[task_id]:
            if expected not in full_text:
                issues.append(f"missing expected value: '{expected}'")

    # 7. Sanity check: no absurdly short examples
    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars < 200:
        issues.append(f"suspiciously short ({total_chars} chars total)")

    # 8. Final assistant message should not end mid-sentence (truncation check)
    final = messages[-1]["content"].strip()
    if len(final) > 0 and final[-1] not in ".!?\")}'`":
        if len(final) > 100:  # ignore very short confirmations
            issues.append("final message may be truncated (doesn't end with punctuation)")

    return issues


def cmd_validate(clean: bool = False):
    records      = load_records(include_val=True)
    issue_counts = defaultdict(int)
    bad_records  = []
    clean_records = []

    print(f"\nValidating {len(records)} examples...\n")

    for rec in records:
        issues = validate_record(rec)
        if issues:
            bad_records.append((rec, issues))
            for issue in issues:
                issue_counts[issue] += 1
        else:
            clean_records.append(rec)

    print(f"{'═'*55}")
    print(f"  VALIDATION RESULTS")
    print(f"{'═'*55}")
    print(f"  Clean:   {len(clean_records)} / {len(records)} "
          f"({100*len(clean_records)/len(records):.1f}%)")
    print(f"  Issues:  {len(bad_records)}")
    print()

    if issue_counts:
        print(f"  Issue breakdown:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"    {count:>4}×  {issue}")
        print()

    # Show a sample of bad examples
    if bad_records:
        print(f"  Sample of flagged examples (first 5):")
        for rec, issues in bad_records[:5]:
            print(f"\n  Task: {rec.get('task_id')}")
            for issue in issues:
                print(f"    ✗ {issue}")
            # Show the user message for context
            user_msgs = [m for m in rec.get("messages", []) if m["role"] == "user"]
            if user_msgs:
                print(f"    User: {user_msgs[0]['content'][:100]}...")

    if clean:
        # Re-split train/val from clean records only
        by_task = defaultdict(list)
        for r in clean_records:
            by_task[r["task_id"]].append(r)

        train_out, val_out = [], []
        VAL_PER_TASK = 2
        for task_id, exs in by_task.items():
            random.shuffle(exs)
            val_cut = min(VAL_PER_TASK, len(exs))
            val_out.extend(exs[:val_cut])
            train_out.extend(exs[val_cut:])

        with open(TRAIN_FILE, "w") as f:
            for r in train_out:
                f.write(json.dumps(r) + "\n")
        with open(VAL_FILE, "w") as f:
            for r in val_out:
                f.write(json.dumps(r) + "\n")

        print(f"\n  ✓ Cleaned dataset written:")
        print(f"    Train: {len(train_out)}  →  {TRAIN_FILE}")
        print(f"    Val:   {len(val_out)}   →  {VAL_FILE}")
    else:
        print(f"\n  Dry run. Use --clean to remove flagged examples.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("stats")

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("n", nargs="?", type=int, default=3)

    p_task = sub.add_parser("task")
    p_task.add_argument("task_id")

    p_val = sub.add_parser("validate")
    p_val.add_argument("--clean", action="store_true",
                       help="Remove flagged examples and rewrite train/val files")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "stats":
        cmd_stats()
    elif args.command == "sample":
        cmd_sample(args.n)
    elif args.command == "task":
        cmd_task(args.task_id)
    elif args.command == "validate":
        cmd_validate(clean=args.clean)
