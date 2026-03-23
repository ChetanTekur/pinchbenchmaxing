#!/usr/bin/env python3
"""
Comprehensive training data validator.

Checks tool call schemas, semantic correctness, structural integrity,
and quality signals. Catches issues that cause benchmark failures:
- Wrong tool names (e.g. 'image' instead of 'generate_image')
- Wrong argument names (e.g. 'description' instead of 'prompt')
- Repetitive tool calls (model loops)
- Truncated responses
- Missing required tool calls
- Invalid JSON in tool calls
- Orphaned tool results

Usage:
  python -m datagen.validate_data                    # full report
  python -m datagen.validate_data --fix              # remove bad examples
  python -m datagen.validate_data --task task_07     # check one task
  python -m datagen.validate_data --verbose          # show each issue
"""

import json
import re
import argparse
import sys
from pathlib import Path
from collections import defaultdict

from utils.config import load_config
from utils.prompts import VALID_TOOLS

_cfg = load_config()
TRAIN_FILE = _cfg.train_file
VAL_FILE = _cfg.val_file

# ─────────────────────────────────────────────────────────────────────────────
# TOOL SCHEMAS — expected argument names for each tool
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SIGNATURES = {
    "read_file":              {"required": ["path"], "optional": []},
    "write_file":             {"required": ["path", "content"], "optional": []},
    "create_directory":       {"required": ["path"], "optional": []},
    "list_files":             {"required": [], "optional": ["directory", "path"]},
    "run_bash":               {"required": ["command"], "optional": []},
    "run_python":             {"required": ["code"], "optional": []},
    "web_search":             {"required": ["query"], "optional": ["num_results"]},
    "fetch_url":              {"required": ["url"], "optional": []},
    "create_calendar_event":  {"required": ["title", "date", "time"], "optional": ["attendees", "description", "filename"]},
    "draft_email":            {"required": ["to", "subject", "body"], "optional": ["cc"]},
    "search_emails":          {"required": ["query"], "optional": ["folder"]},
    "read_email":             {"required": ["email_id"], "optional": []},
    "generate_image":         {"required": ["prompt", "filename"], "optional": []},
    "read_memory":            {"required": [], "optional": ["key"]},
    "write_memory":           {"required": ["key", "value"], "optional": []},
    "search_skills":          {"required": ["query"], "optional": []},
    "install_skill":          {"required": ["name"], "optional": []},
}

# Common wrong tool names → what they should be
TOOL_NAME_TYPOS = {
    "image": "generate_image",
    "create_image": "generate_image",
    "gen_image": "generate_image",
    "image_gen": "generate_image",
    "execute_python": "run_python",
    "exec_python": "run_python",
    "python": "run_python",
    "execute_bash": "run_bash",
    "exec_bash": "run_bash",
    "bash": "run_bash",
    "shell": "run_bash",
    "file_read": "read_file",
    "file_write": "write_file",
    "read": "read_file",
    "write": "write_file",
    "list_dir": "list_files",
    "list_directory": "list_files",
    "ls": "list_files",
    "mkdir": "create_directory",
    "search": "web_search",
    "google": "web_search",
    "email": "draft_email",
    "send_email": "draft_email",
    "find_email": "search_emails",
    "email_search": "search_emails",
    "email_lookup": "search_emails",
    "calendar": "create_calendar_event",
    "memory_read": "read_memory",
    "memory_write": "write_memory",
    "get_memory": "read_memory",
    "set_memory": "write_memory",
    "store_memory": "write_memory",
}

# Tasks that MUST use specific tools
# Required tools per task — must match what the benchmark actually expects.
# These are validated against training data: examples missing required tools
# get flagged as critical issues and removed by --fix.
#
# IMPORTANT: These must match reality (what tools the agent needs to pass the
# benchmark), NOT theoretical ideal. v8 scored 79-95% on task_16/17 using
# list_files + read_file (file I/O), not search_emails.
REQUIRED_TOOLS = {
    "task_01_calendar": ["create_calendar_event"],
    "task_02_stock": ["web_search"],
    "task_03_blog": ["write_file"],
    "task_04_weather": ["write_file"],
    "task_05_summary": ["read_file", "write_file"],
    "task_06_events": ["web_search"],
    "task_07_email": ["write_file"],  # saves email_draft.txt; draft_email also acceptable
    "task_08_memory": ["read_file", "write_file"],
    "task_09_files": ["create_directory", "write_file"],
    "task_10_workflow": ["write_file"],
    "task_11_config_update": ["read_file", "write_file"],
    "task_12_skill_search": ["search_skills"],
    "task_13_image_gen": ["generate_image"],
    "task_14_humanizer": ["read_file", "write_file"],
    "task_15_daily_summary": ["read_file", "write_file"],
    "task_16_email_triage": ["list_files", "read_file", "write_file"],  # reads email files from directory
    "task_17_email_search": ["list_files", "read_file", "write_file"],  # reads email files from directory
    "task_18_market_research": ["web_search", "write_file"],
    "task_19_spreadsheet_summary": ["read_file", "write_file"],
    "task_20_eli5_pdf": ["read_file", "write_file"],
    "task_21_openclaw_comprehension": ["read_file"],
    "task_22_second_brain": ["write_memory"],
}


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_tool_calls(content: str) -> list[dict]:
    """Extract all tool calls from an assistant message."""
    calls = []
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL):
        raw = m.group(1).strip()
        try:
            parsed = json.loads(raw)
            calls.append(parsed)
        except json.JSONDecodeError:
            calls.append({"_parse_error": True, "_raw": raw[:200]})
    return calls


def extract_all_tools_used(messages: list[dict]) -> list[dict]:
    """Extract all tool calls across all assistant messages."""
    all_calls = []
    for msg in messages:
        if msg.get("role") == "assistant":
            all_calls.extend(extract_tool_calls(msg.get("content", "")))
    return all_calls


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def validate_example(example: dict, verbose: bool = False) -> list[dict]:
    """Run all validators on a single example. Returns list of issues."""
    issues = []
    task_id = example.get("task_id", "unknown")
    messages = example.get("messages", [])

    if not messages:
        issues.append({"severity": "critical", "check": "empty_messages",
                       "detail": "Example has no messages"})
        return issues

    # ── Structural checks ──────────────────────────────────────────────────

    # Message roles
    roles = [m.get("role") for m in messages]

    if roles[0] != "system":
        issues.append({"severity": "warning", "check": "missing_system",
                       "detail": "First message is not system role"})

    if roles[-1] != "assistant":
        issues.append({"severity": "high", "check": "no_final_assistant",
                       "detail": f"Last message role is '{roles[-1]}', not assistant"})

    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        issues.append({"severity": "critical", "check": "no_user_message",
                       "detail": "No user message found"})
    elif not user_msgs[0].get("content", "").strip():
        issues.append({"severity": "critical", "check": "empty_user_message",
                       "detail": "User message is empty"})

    # ── Tool call schema checks ────────────────────────────────────────────

    all_calls = extract_all_tools_used(messages)
    tool_names_used = set()

    for i, call in enumerate(all_calls):
        if call.get("_parse_error"):
            issues.append({"severity": "high", "check": "invalid_tool_json",
                           "detail": f"Tool call {i+1}: invalid JSON — {call.get('_raw', '')[:100]}"})
            continue

        name = call.get("name", "")
        args = call.get("arguments", {})
        tool_names_used.add(name)

        # Wrong tool name?
        if name not in VALID_TOOLS:
            suggestion = TOOL_NAME_TYPOS.get(name, "unknown")
            issues.append({"severity": "critical", "check": "invalid_tool_name",
                           "detail": f"Tool '{name}' does not exist. Did you mean '{suggestion}'?"})
            continue

        # Check argument names
        sig = TOOL_SIGNATURES.get(name)
        if sig:
            all_valid = set(sig["required"] + sig["optional"])
            for req in sig["required"]:
                if req not in args:
                    issues.append({"severity": "high", "check": "missing_required_arg",
                                   "detail": f"Tool '{name}': missing required arg '{req}'"})
            for arg_name in args:
                if arg_name not in all_valid:
                    issues.append({"severity": "medium", "check": "unknown_arg",
                                   "detail": f"Tool '{name}': unknown arg '{arg_name}' "
                                             f"(valid: {', '.join(all_valid)})"})

        # Empty arguments?
        if isinstance(args, dict):
            for arg_name, val in args.items():
                if val == "" or val is None:
                    issues.append({"severity": "medium", "check": "empty_arg",
                                   "detail": f"Tool '{name}': arg '{arg_name}' is empty"})

    # ── Required tools check ───────────────────────────────────────────────

    required = REQUIRED_TOOLS.get(task_id, [])
    for req_tool in required:
        if req_tool not in tool_names_used:
            issues.append({"severity": "high", "check": "missing_required_tool",
                           "detail": f"Task requires '{req_tool}' but it was never called"})

    # ── Repetition check ───────────────────────────────────────────────────

    if all_calls:
        call_strings = []
        for c in all_calls:
            if not c.get("_parse_error"):
                call_strings.append(json.dumps({"name": c.get("name"), "args_keys": sorted(c.get("arguments", {}).keys())}))

        # Same tool call signature > 5 times = looping
        from collections import Counter
        call_counts = Counter(call_strings)
        for call_sig, count in call_counts.items():
            if count > 5:
                issues.append({"severity": "high", "check": "repetitive_tool_calls",
                               "detail": f"Same tool call repeated {count} times (likely looping): {call_sig[:100]}"})

        # Total tool calls > 15 = suspicious
        if len(all_calls) > 15:
            issues.append({"severity": "medium", "check": "excessive_tool_calls",
                           "detail": f"{len(all_calls)} tool calls (>15 is suspicious)"})

    # ── Truncation check ───────────────────────────────────────────────────

    last_assistant = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break

    if last_assistant and len(last_assistant) > 100:
        last_char = last_assistant.rstrip()[-1:] if last_assistant.rstrip() else ""
        if last_char not in '.!?"\')}>':
            issues.append({"severity": "medium", "check": "truncated_response",
                           "detail": f"Final assistant message may be truncated (ends with '{last_char}')"})

    # ── Empty assistant check ──────────────────────────────────────────────

    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if assistant_msgs:
        empty_count = sum(1 for m in assistant_msgs if not m.get("content", "").strip())
        if empty_count > 0:
            issues.append({"severity": "high", "check": "empty_assistant",
                           "detail": f"{empty_count} assistant message(s) have empty content"})

    # ── Tool result checks ─────────────────────────────────────────────────

    tool_results = [m for m in messages if m.get("role") in ("tool", "tool_result")]
    for tr in tool_results:
        content = tr.get("content", "")
        if not content.strip():
            issues.append({"severity": "medium", "check": "empty_tool_result",
                           "detail": "Tool result is empty"})
        # Placeholder detection
        if content.strip() in ("...", "TODO", "placeholder", "result"):
            issues.append({"severity": "high", "check": "placeholder_tool_result",
                           "detail": f"Tool result looks like a placeholder: '{content.strip()[:50]}'"})

    # ── Conversation length check ──────────────────────────────────────────

    if task_id != "task_00_sanity" and len(messages) < 4:
        issues.append({"severity": "medium", "check": "too_short",
                       "detail": f"Only {len(messages)} messages (expected at least 4 for tool-using tasks)"})

    # ── Total content length check ─────────────────────────────────────────

    total_chars = sum(len(m.get("content", "")) for m in messages)
    if total_chars < 200 and task_id != "task_00_sanity":
        issues.append({"severity": "medium", "check": "very_short_content",
                       "detail": f"Total content only {total_chars} chars"})

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def load_examples(task_filter: str | None = None) -> list[dict]:
    examples = []
    for path in [TRAIN_FILE, VAL_FILE]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                if task_filter and ex.get("task_id") != task_filter:
                    continue
                examples.append(ex)
            except json.JSONDecodeError:
                continue
    return examples


def run_validation(task_filter: str | None = None, verbose: bool = False,
                   fix: bool = False):
    examples = load_examples(task_filter)

    print(f"\n{'='*70}")
    print(f"  DATA QUALITY VALIDATION")
    print(f"  Examples: {len(examples)}")
    if task_filter:
        print(f"  Filter: {task_filter}")
    print(f"{'='*70}")

    # Validate all examples
    by_severity = defaultdict(int)
    by_check = defaultdict(int)
    by_task = defaultdict(lambda: defaultdict(int))
    bad_indices = set()
    all_issues = []

    for i, ex in enumerate(examples):
        issues = validate_example(ex, verbose)
        task_id = ex.get("task_id", "unknown")

        if issues:
            all_issues.append((i, task_id, issues))
            has_critical = any(iss["severity"] in ("critical", "high") for iss in issues)
            if has_critical:
                bad_indices.add(i)

        for iss in issues:
            by_severity[iss["severity"]] += 1
            by_check[iss["check"]] += 1
            by_task[task_id][iss["check"]] += 1

    # ── Summary ────────────────────────────────────────────────────────────

    total_issues = sum(by_severity.values())
    clean = len(examples) - len(all_issues)

    print(f"\n  SUMMARY")
    print(f"  Clean examples:  {clean}/{len(examples)} ({100*clean/max(len(examples),1):.0f}%)")
    print(f"  With issues:     {len(all_issues)}")
    print(f"  Critical/High:   {len(bad_indices)} (removable)")
    print(f"  Total issues:    {total_issues}")

    # ── By severity ────────────────────────────────────────────────────────

    print(f"\n  BY SEVERITY")
    for sev in ["critical", "high", "medium", "warning"]:
        count = by_severity.get(sev, 0)
        if count:
            print(f"    {sev:<12} {count:>5}")

    # ── By check type ──────────────────────────────────────────────────────

    print(f"\n  BY CHECK TYPE")
    for check, count in sorted(by_check.items(), key=lambda x: -x[1]):
        print(f"    {check:<30} {count:>5}")

    # ── By task ────────────────────────────────────────────────────────────

    print(f"\n  WORST TASKS")
    task_totals = {t: sum(checks.values()) for t, checks in by_task.items()}
    for task, total in sorted(task_totals.items(), key=lambda x: -x[1])[:10]:
        checks = by_task[task]
        top_issues = ", ".join(f"{c}({n})" for c, n in sorted(checks.items(), key=lambda x: -x[1])[:3])
        print(f"    {task:<35} {total:>4} issues  ({top_issues})")

    # ── Verbose: show each issue ───────────────────────────────────────────

    if verbose:
        print(f"\n  ALL ISSUES")
        for idx, task_id, issues in all_issues[:50]:  # cap at 50
            print(f"\n    Example {idx} ({task_id}):")
            for iss in issues:
                sev = iss["severity"].upper()
                print(f"      [{sev}] {iss['check']}: {iss['detail']}")

    # ── Fix mode: remove bad examples ──────────────────────────────────────

    if fix and bad_indices:
        print(f"\n  FIXING: removing {len(bad_indices)} examples with critical/high issues...")

        for path in [TRAIN_FILE, VAL_FILE]:
            if not path.exists():
                continue
            lines = path.read_text().splitlines()
            # We need to map back — this is simpler if we just filter
            clean_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    issues = validate_example(ex)
                    has_bad = any(iss["severity"] in ("critical", "high") for iss in issues)
                    if task_filter and ex.get("task_id") != task_filter:
                        clean_lines.append(line)  # keep examples outside filter
                    elif not has_bad:
                        clean_lines.append(line)
                except json.JSONDecodeError:
                    continue
            path.write_text("\n".join(clean_lines) + "\n" if clean_lines else "")

        remaining = sum(1 for _ in load_examples())
        print(f"  Done. {remaining} examples remaining.")

    # ── Save report ────────────────────────────────────────────────────────

    report = {
        "total_examples": len(examples),
        "clean": clean,
        "with_issues": len(all_issues),
        "critical_high": len(bad_indices),
        "by_severity": dict(by_severity),
        "by_check": dict(by_check),
        "worst_tasks": dict(task_totals),
    }
    report_file = _cfg.data_dir / "validation_report.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {report_file}")

    # ── Save bad examples detail report ───────────────────────────────────
    # Shows actual tool calls for each bad example so you can see what's wrong

    bad_detail = []
    for idx, task_id, issues in all_issues:
        has_critical = any(iss["severity"] in ("critical", "high") for iss in issues)
        if not has_critical:
            continue

        ex = examples[idx]
        msgs = ex.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "?")

        # Extract tool calls
        tool_calls = []
        for msg in msgs:
            if msg.get("role") == "assistant":
                for block in re.findall(r'<tool_call>(.*?)</tool_call>', msg["content"], re.DOTALL):
                    try:
                        obj = json.loads(block.strip())
                        tool_calls.append({
                            "name": obj.get("name", "?"),
                            "args": list(obj.get("arguments", {}).keys()),
                        })
                    except json.JSONDecodeError:
                        tool_calls.append({"name": "PARSE_ERROR", "raw": block[:100]})

        bad_detail.append({
            "task_id": task_id,
            "user_message": user_msg[:200],
            "issues": [{"severity": i["severity"], "check": i["check"], "detail": i["detail"]} for i in issues],
            "tool_calls": tool_calls,
        })

    bad_report_file = _cfg.data_dir / "bad_examples_report.json"
    bad_report_file.write_text(json.dumps(bad_detail, indent=2))
    print(f"  Bad examples: {bad_report_file} ({len(bad_detail)} examples)")
    print(f"{'='*70}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Comprehensive training data validator")
    parser.add_argument("--task", type=str, default=None,
                        help="Validate only this task ID")
    parser.add_argument("--verbose", action="store_true",
                        help="Show each issue in detail")
    parser.add_argument("--fix", action="store_true",
                        help="Remove examples with critical/high issues")
    args = parser.parse_args()
    run_validation(task_filter=args.task, verbose=args.verbose, fix=args.fix)


if __name__ == "__main__":
    main()
