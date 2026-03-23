#!/usr/bin/env python3
"""
Dynamic data generation — reads task definitions from PinchBench .md files
instead of hardcoded TASKS dict.

The key innovation: the meta-prompt includes the FULL benchmark task definition
so Claude knows exactly what the benchmark tests for, plus the OPENCLAW_SYSTEM
prompt so it knows what tools are available.

Usage:
  python -m datagen.dynamic_gen run --tasks task_11_config_update,task_12_skill_search
  python -m datagen.dynamic_gen run --tasks task_08_memory --min-per-task 40
  python -m datagen.dynamic_gen run --all-below 40
  python -m datagen.dynamic_gen count
  python -m datagen.dynamic_gen submit --tasks task_11_config_update
  python -m datagen.dynamic_gen status
  python -m datagen.dynamic_gen collect
"""

import json
import os
import random
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import anthropic

from utils.config import load_config
from utils.prompts import OPENCLAW_SYSTEM, VALID_TOOLS
from datagen.task_loader import load_tasks, load_task
from datagen.topup import (
    VARIATION_CONFIGS,
    parse_example, extract_json_array,
    count_existing,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_cfg       = load_config()
DATA_DIR   = _cfg.data_dir
RAW_DIR    = DATA_DIR / "raw_dynamic"
BATCH_FILE = DATA_DIR / "dynamic_batch_id.txt"
TRAIN_FILE = _cfg.train_file
VAL_FILE   = _cfg.val_file
MODEL      = _cfg.claude.generation
TARGET_PER_TASK = _cfg.data.examples_per_task
VAL_SPLIT  = _cfg.data.val_split

# Tasks that produce long examples — generate 1 per call to avoid truncation
HARD_TASKS = {
    "task_09_files",
    "task_10_workflow",
    "task_15_daily_summary",
    "task_16_email_triage",
    "task_17_email_search",
    "task_18_market_research",
    "task_21_openclaw_comprehension",
}


def _epc(task_id: str) -> int:
    """Examples per Claude call for this task."""
    return 1 if task_id in HARD_TASKS else 3


# ─────────────────────────────────────────────────────────────────────────────
# META-PROMPT BUILDER (the key innovation)
# ─────────────────────────────────────────────────────────────────────────────

def build_dynamic_meta_prompt(
    task_id: str,
    task_def: dict,
    variation: dict,
    epc: int,
    diagnosis: dict | None = None,
) -> str:
    """Build a meta-prompt using the actual PinchBench .md file content.

    Unlike the old approach that used hardcoded task prompts and grading
    criteria, this reads the full benchmark definition so Claude knows
    EXACTLY what the benchmark tests for.
    """
    # Full benchmark task content from the .md file
    raw_content = task_def["raw_content"]

    # Build expected files / values hints if available
    files_hint = ""
    if task_def.get("expected_files"):
        files_list = ", ".join(f"`{f}`" for f in task_def["expected_files"])
        files_hint = f"\n**Expected output files**: {files_list}"

    values_hint = ""
    if task_def.get("expected_values"):
        vals = task_def["expected_values"][:10]  # cap at 10 to avoid bloat
        values_list = "\n".join(f"  - `{v}`" for v in vals)
        values_hint = f"\n**Expected values that MUST appear in output**:\n{values_list}"

    # Error recovery note
    error_note = ""
    if variation["has_error"]:
        error_note = (
            "\nERROR RECOVERY REQUIRED: Include exactly one tool call that returns "
            "an error. The agent must read the error and try a different approach.\n"
        )

    # Diagnosis context (optional — for targeted fixes)
    diag_section = ""
    if diagnosis:
        root_cause = diagnosis.get("root_cause", "")
        fix = diagnosis.get("fix", "")
        reason = diagnosis.get("reason", "")

        parts = []
        if root_cause:
            parts.append(f"Root cause of failure: {root_cause}")
        if fix:
            parts.append(f"Recommended fix: {fix}")
        if reason:
            parts.append(f"Context: {reason}")

        if parts:
            diag_text = "\n".join(parts)
            diag_section = f"""
## Diagnosis Context (from benchmark failure analysis)
{diag_text}

CRITICAL: The generated examples MUST demonstrate the CORRECT behavior that
avoids the failure pattern described above. Show clean, direct task completion
without the described failure mode.
"""

    # Valid tool names as a formatted list
    tools_list = ", ".join(sorted(VALID_TOOLS))

    return f"""\
You are generating synthetic fine-tuning data for an AI agent called Clawd.

## Agent System Prompt (this is what the agent sees at inference time)
{OPENCLAW_SYSTEM}

## Benchmark Task Definition (this is the ACTUAL benchmark the agent must pass)
The following is the full task definition from PinchBench. The agent must
demonstrate behavior that would PASS this benchmark task.

---
{raw_content}
---
{files_hint}
{values_hint}

## Variation Type: {variation["id"]}
User style: {variation["user_style"]}
Scenario: {variation["scenario"]}
{error_note}
{diag_section}

## Your Job
Generate {epc} diverse training examples that would teach a model to PASS this benchmark task.

Each example MUST follow this exact JSON structure:
{{
  "user_message": "<user request that triggers this task, phrased per the variation style>",
  "turns": [
    {{
      "role": "assistant",
      "content": "<thinking + tool_call tags when calling tools>"
    }},
    {{
      "role": "tool_result",
      "content": "<realistic JSON result from the tool>"
    }},
    ... more turns as needed ...
    {{
      "role": "assistant",
      "content": "<final confirmation summarising what was done>"
    }}
  ]
}}

Tool call format (inside assistant content):
<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>

## CRITICAL RULES
1. Each example must show COMPLETE task execution — from user request to final confirmation.
2. The ACTUAL FILE CONTENT must be visible in the write_file tool calls — NOT truncated.
   If the task requires writing a 500-word blog post, the full 500-word post must appear
   in the write_file arguments. NEVER use "[content truncated]" or "..." placeholders.
3. Use ONLY these tool names: {tools_list}
   NEVER invent tool names. Tools like execute_python, list_directory, get_current_date,
   execute_skill, move_file, file_exists DO NOT EXIST.
4. The examples must demonstrate the EXACT behavior the benchmark tests for.
   Read the benchmark definition above carefully — every grading criterion must be satisfied.
5. Output files must use the EXACT filenames the benchmark expects.
6. user_message must vary meaningfully across the {epc} examples.
7. tool_result content must be plausible — use real-looking paths, timestamps, data.
8. For tasks with specific expected values (exact figures, filenames, dates), those
   exact values MUST appear in the tool results and final response.
9. The final assistant turn must confirm completion and reference what was done.
10. Do NOT skip required tool calls — if the task needs a file created, call write_file.
11. For multi-document tasks, show ALL required files being read individually.
12. JSON ESCAPING IS MANDATORY: Inside any JSON string value, you MUST escape:
    - Double quotes as \\\"
    - Backslashes as \\\\
    - Newlines as \\n
13. The final assistant message MUST end with a complete sentence (period, !, or ?).
    Never produce a response that cuts off mid-sentence.
14. task_00_sanity is a SIMPLE hello/greeting task. No tool calls. Keep it trivial.

Return ONLY a valid JSON array of {epc} objects. No markdown fences, no preamble.
"""


# ─────────────────────────────────────────────────────────────────────────────
# TASK RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def resolve_tasks(
    only_tasks: list[str] | None = None,
    all_below: int | None = None,
) -> dict[str, dict]:
    """Load and filter PinchBench tasks.

    Returns {task_id: task_def} for tasks that need generation.
    """
    all_tasks = load_tasks()

    if not all_tasks:
        print("ERROR: No PinchBench tasks found. Ensure the benchmark .md files")
        print("       are at {workspace}/skill/tasks/")
        sys.exit(1)

    if only_tasks:
        # Filter to requested tasks
        resolved = {}
        for tid in only_tasks:
            # Try exact match first, then partial
            if tid in all_tasks:
                resolved[tid] = all_tasks[tid]
            else:
                # Partial match: "task_08" matches "task_08_memory"
                matches = {k: v for k, v in all_tasks.items() if tid in k}
                if len(matches) == 1:
                    key = list(matches.keys())[0]
                    resolved[key] = matches[key]
                elif len(matches) > 1:
                    print(f"  WARNING: '{tid}' matches multiple tasks: {list(matches.keys())}")
                    # Take the first match
                    key = list(matches.keys())[0]
                    resolved[key] = matches[key]
                else:
                    print(f"  WARNING: No task found matching '{tid}'")
        return resolved

    if all_below is not None:
        # Filter to tasks with fewer than N existing examples
        counts = count_existing()
        resolved = {}
        for tid, tdef in all_tasks.items():
            current = counts.get(tid, 0)
            if current < all_below:
                resolved[tid] = tdef
        return resolved

    # Default: all tasks
    return all_tasks


def compute_dynamic_deficits(
    tasks: dict[str, dict],
    min_per_task: int = 0,
    target: int = TARGET_PER_TASK,
) -> dict[str, int]:
    """Compute how many examples each task needs.

    Returns {task_id: n_needed} for tasks with deficits.
    """
    counts = count_existing()
    deficits = {}

    for task_id in tasks:
        current = counts.get(task_id, 0)
        needed = max(0, target - current)

        # If min_per_task is set, ensure at least that many new examples
        if min_per_task > 0:
            needed = max(needed, min_per_task)

        if needed > 0:
            deficits[task_id] = needed

    return deficits


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_count(only_tasks: list[str] | None = None, all_below: int | None = None):
    """Show current example counts for PinchBench tasks."""
    tasks = resolve_tasks(only_tasks=only_tasks, all_below=all_below)
    counts = count_existing()

    print(f"\n  Data paths:")
    for path in [TRAIN_FILE, VAL_FILE]:
        size = f"{path.stat().st_size:,} bytes" if path.exists() else "NOT FOUND"
        print(f"    {path}  [{size}]")

    print(f"\n{'='*65}")
    print(f"  DYNAMIC GEN — TASK COUNTS vs TARGET ({TARGET_PER_TASK})")
    print(f"{'='*65}")
    print(f"  {'Task':<40} {'Have':>5}  {'Need':>5}  {'Gap':>5}")
    print(f"  {'-'*40} {'-'*5}  {'-'*5}  {'-'*5}")

    total_gap = 0
    for task_id in sorted(tasks.keys()):
        current = counts.get(task_id, 0)
        gap = max(0, TARGET_PER_TASK - current)
        total_gap += gap
        flag = "  !" if gap > 0 else "  ok"
        print(f"  {task_id:<40} {current:>5}  {TARGET_PER_TASK:>5}  {gap:>5}{flag}")

    print(f"\n  Tasks loaded from PinchBench .md files: {len(tasks)}")
    print(f"  Total new examples needed: ~{total_gap}")


def cmd_submit(
    only_tasks: list[str] | None = None,
    all_below: int | None = None,
    min_per_task: int = 0,
    diagnosis_file: str | None = None,
):
    """Submit a Claude Batch API request for dynamic generation."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load tasks from PinchBench .md files
    tasks = resolve_tasks(only_tasks=only_tasks, all_below=all_below)
    if not tasks:
        print("No tasks resolved. Check task IDs or --all-below threshold.")
        sys.exit(1)

    print(f"  Loaded {len(tasks)} tasks from PinchBench .md files")

    # Load optional diagnosis
    diagnosis = {}
    if diagnosis_file and Path(diagnosis_file).exists():
        diagnosis = json.loads(Path(diagnosis_file).read_text())
        print(f"  Loaded diagnosis for {len(diagnosis)} tasks")

    # Compute deficits
    deficits = compute_dynamic_deficits(tasks, min_per_task=min_per_task)
    if not deficits:
        print("All tasks are at or above target. Nothing to do.")
        sys.exit(0)

    print(f"\n  Tasks to generate for: {list(deficits.keys())}")

    # Build batch requests
    requests = []
    for task_id, needed in deficits.items():
        task_def = tasks[task_id]
        epc = _epc(task_id)
        n_calls = (needed + epc - 1) // epc
        max_tok = 16000 if task_id in HARD_TASKS else 8192

        # Get diagnosis for this task (may be empty)
        task_diag = diagnosis.get(task_id)

        for i in range(n_calls):
            variation = VARIATION_CONFIGS[i % len(VARIATION_CONFIGS)]
            custom_id = f"dynamic__{task_id}__{variation['id']}__{i:03d}"
            prompt = build_dynamic_meta_prompt(
                task_id, task_def, variation, epc, diagnosis=task_diag
            )
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": max_tok,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })

    # Submit batch
    client = anthropic.Anthropic(api_key=api_key)
    batch = client.messages.batches.create(requests=requests)
    BATCH_FILE.write_text(batch.id)

    total_examples = sum(
        _epc(r["custom_id"].split("__")[1]) for r in requests
    )
    print(f"\n  Submitted {len(requests)} requests -> ~{total_examples} new examples")
    print(f"  Batch ID: {batch.id}  (saved to {BATCH_FILE})")
    print(f"\n  python -m datagen.dynamic_gen status   # check progress")
    print(f"  python -m datagen.dynamic_gen collect   # when done")


def cmd_status():
    """Check batch status."""
    if not BATCH_FILE.exists():
        print("No dynamic batch found. Run: python -m datagen.dynamic_gen submit")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(BATCH_FILE.read_text().strip())
    counts = batch.request_counts

    print(f"Status: {batch.processing_status}")
    print(f"  Processing: {counts.processing}")
    print(f"  Succeeded:  {counts.succeeded}")
    print(f"  Errored:    {counts.errored}")
    if batch.processing_status == "ended":
        print("\n  Ready. Run: python -m datagen.dynamic_gen collect")


def cmd_collect():
    """Collect results from completed batch and append to train/val."""
    if not BATCH_FILE.exists():
        print("No dynamic batch found. Run: python -m datagen.dynamic_gen submit")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    batch_id = BATCH_FILE.read_text().strip()

    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"Not ready yet: {batch.processing_status}")
        sys.exit(1)

    from tqdm import tqdm

    new_by_task = defaultdict(list)
    api_errors = 0
    parse_fails = 0

    for result in tqdm(client.messages.batches.results(batch_id), desc="Collecting"):
        cid = result.custom_id  # dynamic__task_XX__var__NNN
        parts = cid.split("__")
        task_id = parts[1] if len(parts) >= 2 else "unknown"

        if result.result.type == "errored":
            api_errors += 1
            continue

        msg = result.result.message
        raw_text = msg.content[0].text if msg.content else ""

        # Save raw response for debugging
        raw_file = RAW_DIR / f"{cid}.json"
        raw_file.write_text(raw_text)

        examples = extract_json_array(raw_text)
        if not examples:
            parse_fails += 1
            continue

        for ex in examples:
            parsed = parse_example(ex, task_id)
            if parsed:
                parsed["source"] = "dynamic"
                new_by_task[task_id].append(parsed)

    # Append to train/val (append mode — does not rewrite existing data)
    total_train = 0
    total_val = 0
    per_task_report = {}

    for task_id, examples in sorted(new_by_task.items()):
        random.shuffle(examples)
        val_target = max(2, round(TARGET_PER_TASK * VAL_SPLIT))
        val_count = min(val_target, max(1, len(examples) // 5))
        val_set = examples[:val_count]
        train_set = examples[val_count:]

        with open(TRAIN_FILE, "a") as f:
            for ex in train_set:
                f.write(json.dumps(ex) + "\n")
        with open(VAL_FILE, "a") as f:
            for ex in val_set:
                f.write(json.dumps(ex) + "\n")

        total_train += len(train_set)
        total_val += len(val_set)
        per_task_report[task_id] = len(examples)

    print(f"\n{'-'*50}")
    print(f"  Dynamic generation complete")
    print(f"  API errors:     {api_errors}")
    print(f"  Parse failures: {parse_fails}")
    print(f"\n  Per-task additions:")
    for tid, count in sorted(per_task_report.items()):
        print(f"    {tid:<45} + {count:>3}")
    print(f"\n  New train: {total_train}")
    print(f"  New val:   {total_val}")

    return total_train + total_val


def cmd_run(
    only_tasks: list[str] | None = None,
    all_below: int | None = None,
    min_per_task: int = 0,
    diagnosis_file: str | None = None,
):
    """Submit -> poll -> collect in one shot."""
    cmd_submit(
        only_tasks=only_tasks,
        all_below=all_below,
        min_per_task=min_per_task,
        diagnosis_file=diagnosis_file,
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    batch_id = BATCH_FILE.read_text().strip()

    print("\nPolling every 2 minutes...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        ts = time.strftime("%H:%M")
        print(f"  [{ts}] processing={counts.processing} "
              f"succeeded={counts.succeeded} errored={counts.errored}")
        if batch.processing_status == "ended":
            break
        time.sleep(120)

    return cmd_collect()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dynamic data generation from PinchBench .md files"
    )
    sub = parser.add_subparsers(dest="command")

    # run: submit -> poll -> collect
    run_p = sub.add_parser("run", help="Submit -> poll -> collect in one shot")
    run_p.add_argument("--tasks", type=str, default=None,
                       help="Comma-separated task IDs (e.g. task_11_config_update,task_08_memory)")
    run_p.add_argument("--all-below", type=int, default=None,
                       help="Generate for all tasks with fewer than N examples")
    run_p.add_argument("--min-per-task", type=int, default=0,
                       help="Minimum new examples per task (overrides deficit)")
    run_p.add_argument("--diagnosis-file", type=str, default=None,
                       help="JSON file with per-task diagnosis from eval analysis")

    # count
    count_p = sub.add_parser("count", help="Show current counts vs target")
    count_p.add_argument("--tasks", type=str, default=None)
    count_p.add_argument("--all-below", type=int, default=None)

    # submit
    submit_p = sub.add_parser("submit", help="Submit batch only")
    submit_p.add_argument("--tasks", type=str, default=None)
    submit_p.add_argument("--all-below", type=int, default=None)
    submit_p.add_argument("--min-per-task", type=int, default=0)
    submit_p.add_argument("--diagnosis-file", type=str, default=None)

    # status / collect
    sub.add_parser("status", help="Check batch status")
    sub.add_parser("collect", help="Collect results from completed batch")

    args = parser.parse_args()

    only_tasks = None
    if hasattr(args, "tasks") and args.tasks:
        only_tasks = [t.strip() for t in args.tasks.split(",")]

    all_below = getattr(args, "all_below", None)
    min_per_task = getattr(args, "min_per_task", 0)
    diagnosis_file = getattr(args, "diagnosis_file", None)

    if args.command == "run":
        cmd_run(only_tasks=only_tasks, all_below=all_below,
                min_per_task=min_per_task, diagnosis_file=diagnosis_file)
    elif args.command == "count":
        cmd_count(only_tasks=only_tasks, all_below=all_below)
    elif args.command == "submit":
        cmd_submit(only_tasks=only_tasks, all_below=all_below,
                   min_per_task=min_per_task, diagnosis_file=diagnosis_file)
    elif args.command == "status":
        cmd_status()
    elif args.command == "collect":
        cmd_collect()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
