#!/usr/bin/env python3
"""
Adversarial data generation — creates training examples from benchmark failures.

Parses benchmark logs to find what the model actually did wrong on each task,
then generates training examples showing the correct approach to that exact
scenario. This is the most targeted fix possible: the model sees its own
failure and learns the correct behavior.

Usage:
  python adversarial_gen.py run --tasks task_09_files,task_20_eli5_pdf
  python adversarial_gen.py run --log-dir /workspace/synthbench/logs
  python adversarial_gen.py analyze --tasks task_09_files  # show failures without generating
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import anthropic

from utils.config import load_config
from utils.prompts import OPENCLAW_SYSTEM
from datagen.topup import TASKS, parse_example, extract_json_array

_cfg       = load_config()
DATA_DIR   = _cfg.data_dir
TRAIN_FILE = _cfg.train_file
VAL_FILE   = _cfg.val_file
MODEL      = _cfg.claude.generation
EXAMPLES_PER_TASK = 10  # default; each API call generates max 3 in batches


# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_log(log_dir: Path) -> Path | None:
    """Find the most recent benchmark log file."""
    logs = sorted(log_dir.glob("bench_*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def extract_task_section(log_text: str, task_id: str) -> str | None:
    """Extract the section of a benchmark log for a specific task."""
    # PinchBench logs have sections like:
    # 🤖 Agent [...] starting task: task_09_files
    # ... task output ...
    # ⚠️ Task task_09_files: 0.4/1.0 (40%) - ...
    # or
    # ❌ Task task_09_files: 0.0/1.0 (0%) - ...
    pattern = rf'starting task: {re.escape(task_id)}.*?(?=starting task:|$)'
    m = re.search(pattern, log_text, re.DOTALL)
    return m.group() if m else None


def extract_failure_pattern(section: str, task_id: str) -> dict:
    """Analyze a task section to identify what the model did wrong."""
    result = {
        "task_id": task_id,
        "score": None,
        "score_type": None,
        "judge_notes": "",
        "tool_calls": [],
        "errors": [],
        "patterns": [],
    }

    # Extract score
    score_match = re.search(
        rf'Task {re.escape(task_id)}:\s*([01](?:\.\d+)?)/1\.0\s*\((\d+)%\)\s*-\s*(\w+)',
        section
    )
    if score_match:
        result["score"] = float(score_match.group(1))
        result["score_type"] = score_match.group(3)

    # Extract judge notes
    notes_match = re.search(r'Notes:\s*(.+?)(?:\n={5,}|\Z)', section, re.DOTALL)
    if notes_match:
        result["judge_notes"] = notes_match.group(1).strip()[:500]

    # Look for common failure patterns
    section_lower = section.lower()

    if section_lower.count("tool_call") > 10:
        result["patterns"].append("excessive_tool_calls")
    if "loop" in section_lower or section_lower.count("read_file") > 5:
        result["patterns"].append("looping")
    if "not found" in section_lower or "no such file" in section_lower:
        result["patterns"].append("wrong_filename")
    if "pk" in section_lower and "bytes" in section_lower:
        result["patterns"].append("binary_as_text")
    if "truncat" in section_lower or "cut off" in section_lower:
        result["patterns"].append("truncation")

    # Extract error lines
    for line in section.splitlines():
        if re.search(r'\bERROR\b', line):
            result["errors"].append(line.strip()[:200])

    # Extract transcript warnings
    for line in section.splitlines():
        if "Failed to parse transcript" in line:
            result["patterns"].append("malformed_output")
            break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_adversarial_prompt(
    task_id: str, task: dict, failure: dict, n_examples: int = 3,
) -> str:
    """Build a prompt that shows the model's failure and asks for correct examples."""
    grading_list = "\n".join(f"  - {g}" for g in task["grading"])

    # Build failure description
    failure_lines = []
    if failure.get("judge_notes"):
        failure_lines.append(f"Judge notes: {failure['judge_notes']}")
    if failure.get("patterns"):
        pattern_descriptions = {
            "looping": "The model got stuck in a loop, calling the same tool repeatedly",
            "wrong_filename": "The model tried to read a nonexistent file instead of the correct one",
            "binary_as_text": "The model read a binary file (xlsx/pdf) as raw text, getting garbage bytes",
            "excessive_tool_calls": "The model made too many tool calls without progress",
            "truncation": "The model's response was truncated mid-sentence",
            "malformed_output": "The model produced malformed output that couldn't be parsed",
        }
        for p in failure["patterns"]:
            if p in pattern_descriptions:
                failure_lines.append(f"Pattern: {pattern_descriptions[p]}")
    if failure.get("errors"):
        failure_lines.append(f"Errors from log: {failure['errors'][:2]}")

    failure_text = "\n".join(failure_lines) if failure_lines else "Task failed with low score."
    score_text = f"{failure['score']:.1f}/1.0" if failure["score"] is not None else "unknown"

    return f"""\
You are generating TARGETED training data for an LLM agent called Clawd.

The model FAILED this task during benchmarking. Your job is to generate
{n_examples} training examples showing the CORRECT approach.

## Task
Name: {task["name"]}
ID: {task_id}
Complexity: {task["complexity"]}
Task prompt: \"\"\"{task["prompt"]}\"\"\"

## Grading Criteria (ALL must be satisfied)
{grading_list}

## What the Model Did WRONG (score: {score_text})
{failure_text}

## Your Job
Generate {n_examples} training examples that demonstrate the CORRECT approach.
Each example must:
1. AVOID the failure patterns described above
2. Complete the task cleanly and efficiently
3. Satisfy ALL grading criteria
4. Use the correct tools with the correct arguments
5. If the task involves reading binary files (.xlsx, .pdf), use run_python
   with appropriate libraries (pandas, pdfplumber) instead of read_file

Each example MUST follow this JSON structure:
{{
  "user_message": "<user request>",
  "turns": [
    {{"role": "assistant", "content": "<action with <tool_call> tags>"}},
    {{"role": "tool_result", "content": "<realistic tool output>"}},
    ... more turns ...
    {{"role": "assistant", "content": "<final confirmation>"}}
  ]
}}

Tool call format: <tool_call>{{"name": "tool_name", "arguments": {{"arg": "value"}}}}</tool_call>

ONLY USE REAL CLAWD TOOLS: read_file, write_file, create_directory, list_files,
run_bash, run_python, web_search, fetch_url, create_calendar_event, draft_email,
search_emails, read_email, generate_image, read_memory, write_memory,
search_skills, install_skill.

Return ONLY a valid JSON array of {n_examples} objects. No markdown, no preamble.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_analyze(log_dir: Path, tasks: list[str]):
    """Show failure analysis without generating data."""
    log_file = find_latest_log(log_dir)
    if not log_file:
        print(f"No benchmark logs found in {log_dir}")
        return

    log_text = log_file.read_text(errors="replace")
    print(f"Analyzing: {log_file.name}")

    for task_id in tasks:
        section = extract_task_section(log_text, task_id)
        if not section:
            print(f"\n  {task_id}: not found in log")
            continue

        failure = extract_failure_pattern(section, task_id)
        print(f"\n  {task_id}: score={failure['score']}")
        if failure["patterns"]:
            print(f"    Patterns: {failure['patterns']}")
        if failure["judge_notes"]:
            print(f"    Notes: {failure['judge_notes'][:200]}")
        if failure["errors"]:
            print(f"    Errors: {failure['errors'][:2]}")


def cmd_run(log_dir: Path, tasks: list[str], n_per_task: int = EXAMPLES_PER_TASK, log_file: Path | None = None):
    """Generate adversarial examples from benchmark failures."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    if not log_file:
        log_file = find_latest_log(log_dir)
    if not log_file:
        print(f"No benchmark logs found in {log_dir}")
        sys.exit(1)

    log_text = log_file.read_text(errors="replace")
    print(f"Source log: {log_file.name}")

    client = anthropic.Anthropic(api_key=api_key)
    results = {"total_generated": 0, "per_task": {}, "errors": []}
    all_new = []

    for task_id in tasks:
        if task_id not in TASKS:
            print(f"  {task_id}: unknown task, skipping")
            continue

        section = extract_task_section(log_text, task_id)
        if not section:
            print(f"  {task_id}: not found in log, skipping")
            continue

        failure = extract_failure_pattern(section, task_id)
        if failure["score"] is not None and failure["score"] >= 0.8:
            print(f"  {task_id}: score={failure['score']:.1f}, not a failure — skipping")
            continue

        print(f"  {task_id}: score={failure.get('score', '?')}, "
              f"patterns={failure.get('patterns', [])}")

        # Generate in batches of 3 (Claude reliably produces 3 per call)
        BATCH_SIZE = 3
        n_batches = (n_per_task + BATCH_SIZE - 1) // BATCH_SIZE
        parsed = []

        for batch_idx in range(n_batches):
            batch_n = min(BATCH_SIZE, n_per_task - len(parsed))
            if batch_n <= 0:
                break

            prompt = build_adversarial_prompt(
                task_id, TASKS[task_id], failure, batch_n
            )

            try:
                resp = client.messages.create(
                    model=MODEL, max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_text = resp.content[0].text.strip()
                examples = extract_json_array(raw_text)

                if not examples:
                    print(f"    ✗ Batch {batch_idx+1}: could not parse response")
                    results["errors"].append({"task": task_id, "error": f"parse_failure_batch_{batch_idx}"})
                    continue

                from datagen.validate_data import validate_example
                batch_kept = 0
                for ex in examples:
                    p = parse_example(ex, task_id)
                    if p:
                        # Validate before accepting — reject structurally bad examples
                        issues = validate_example(p)
                        blocking = [i for i in issues if i["severity"] in ("critical", "high")
                                    or i["check"] == "missing_required_tool"]
                        if blocking:
                            check_types = [i["check"] for i in blocking]
                            print(f"      ✗ Rejected: {', '.join(check_types)}")
                            continue
                        p["source"] = "adversarial"
                        parsed.append(p)
                        batch_kept += 1

                print(f"    Batch {batch_idx+1}: {len(examples)} generated, {batch_kept} kept")

            except Exception as e:
                print(f"    ✗ Batch {batch_idx+1} API error: {e}")
                results["errors"].append({"task": task_id, "error": str(e)})

        all_new.extend(parsed)
        results["per_task"][task_id] = len(parsed)
        results["total_generated"] += len(parsed)
        print(f"    ✓ Generated {len(parsed)} adversarial examples total")

    # Append to train file (adversarial examples go to train only, not val)
    if all_new:
        with open(TRAIN_FILE, "a") as f:
            for ex in all_new:
                f.write(json.dumps(ex) + "\n")
        print(f"\n✓ Appended {len(all_new)} adversarial examples to {TRAIN_FILE}")

    # Write results for gate checking
    results_file = DATA_DIR / "adversarial_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"  Results: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Adversarial data generation from benchmark failures")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Generate adversarial examples")
    run_p.add_argument("--log-dir", type=str, default=None)
    run_p.add_argument("--log-file", type=str, default=None,
                       help="Specific benchmark log file (overrides --log-dir)")
    run_p.add_argument("--tasks", type=str, required=True,
                       help="Comma-separated task IDs")
    run_p.add_argument("--n-per-task", type=int, default=EXAMPLES_PER_TASK)

    analyze_p = sub.add_parser("analyze", help="Show failures without generating")
    analyze_p.add_argument("--log-dir", type=str, default=None)
    analyze_p.add_argument("--log-file", type=str, default=None)
    analyze_p.add_argument("--tasks", type=str, required=True)

    args = parser.parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else (_cfg.data_dir.parent / "logs")
    log_file = Path(args.log_file) if getattr(args, 'log_file', None) else None
    tasks = args.tasks.split(",")

    if args.command == "run":
        cmd_run(log_dir, tasks, args.n_per_task, log_file=log_file)
    elif args.command == "analyze":
        cmd_analyze(log_dir, tasks)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
