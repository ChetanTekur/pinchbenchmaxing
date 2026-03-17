#!/usr/bin/env python3
"""
Diagnosis-aware data generation — replaces blind round-robin with targeted generation.

Uses the EvalAnalysisAgent's diagnosis to:
1. Inject root cause context into the meta-prompt
2. Weight variation types based on failure patterns
3. Generate examples that specifically avoid known failure modes

Usage:
  python targeted_topup.py run --diagnosis-file data/current_diagnosis.json
  python targeted_topup.py run --diagnosis-file data/current_diagnosis.json --tasks task_09,task_13
  python targeted_topup.py count
"""

import json
import os
import random
import sys
import argparse
from pathlib import Path

import anthropic

from utils.config import load_config
from utils.prompts import OPENCLAW_SYSTEM
from topup import (
    TASKS, VARIATION_CONFIGS, HARD_TASKS,
    _epc, count_existing, compute_deficits,
    parse_example, extract_json_array,
    cmd_status, cmd_count,
)

_cfg       = load_config()
DATA_DIR   = _cfg.data_dir
RAW_DIR    = DATA_DIR / "raw_targeted"
BATCH_FILE = DATA_DIR / "targeted_batch_id.txt"
TRAIN_FILE = _cfg.train_file
VAL_FILE   = _cfg.val_file
MODEL      = _cfg.claude.generation
TARGET_PER_TASK = _cfg.data.examples_per_task
VAL_SPLIT  = _cfg.data.val_split

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSIS-AWARE VARIATION SELECTION
# ─────────────────────────────────────────────────────────────────────────────

# Maps failure pattern keywords → variation types that address them
DIAGNOSIS_VARIATION_WEIGHTS = {
    "loop":            {"error_recovery": 3.0, "self_correction": 3.0},
    "repeat":          {"error_recovery": 3.0, "self_correction": 3.0},
    "stuck":           {"error_recovery": 3.0, "self_correction": 2.0},
    "wrong tool":      {"multi_tool_chain": 3.0, "detailed_user": 2.0},
    "missing tool":    {"multi_tool_chain": 3.0, "detailed_user": 2.0},
    "no tool":         {"multi_tool_chain": 3.0, "happy_formal": 2.0},
    "truncat":         {"happy_formal": 2.0, "terse": 2.0},
    "hallucin":        {"detailed_user": 3.0, "happy_formal": 2.0},
    "wrong value":     {"detailed_user": 3.0},
    "wrong file":      {"detailed_user": 3.0, "self_correction": 2.0},
    "binary":          {"multi_tool_chain": 3.0, "error_recovery": 2.0},
    "raw bytes":       {"multi_tool_chain": 3.0, "error_recovery": 2.0},
    "xlsx":            {"multi_tool_chain": 3.0},
    "pdf":             {"multi_tool_chain": 3.0, "error_recovery": 2.0},
    "confused":        {"detailed_user": 3.0, "happy_formal": 2.0},
    "incomplete":      {"self_correction": 3.0, "detailed_user": 2.0},
}


def select_variations(task_id: str, diagnosis: dict, n_calls: int) -> list[dict]:
    """Select n_calls variation configs, weighted by diagnosis keywords."""
    root_cause = diagnosis.get("root_cause", "").lower()
    fix = diagnosis.get("fix", "").lower()
    combined = root_cause + " " + fix

    # Build weight dict (default 1.0 for each variation)
    weights = {v["id"]: 1.0 for v in VARIATION_CONFIGS}

    for keyword, boosts in DIAGNOSIS_VARIATION_WEIGHTS.items():
        if keyword in combined:
            for var_id, multiplier in boosts.items():
                if var_id in weights:
                    weights[var_id] *= multiplier

    # Weighted sampling
    ids = [v["id"] for v in VARIATION_CONFIGS]
    w   = [weights[vid] for vid in ids]
    total = sum(w)
    probs = [x / total for x in w]

    chosen_ids = random.choices(ids, weights=probs, k=n_calls)
    id_to_var  = {v["id"]: v for v in VARIATION_CONFIGS}
    return [id_to_var[vid] for vid in chosen_ids]


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSIS-AWARE META-PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_targeted_meta_prompt(
    task_id: str, task: dict, variation: dict,
    epc: int, diagnosis: dict,
) -> str:
    """Build a meta-prompt with diagnosis context injected."""
    tools_hint = (
        f"Key tools for this task: {', '.join(task['tools_needed'])}"
        if task["tools_needed"] else "This task may not need tool calls."
    )
    grading_list = "\n".join(f"  - {g}" for g in task["grading"])
    error_note = (
        "\n⚠️  ERROR RECOVERY REQUIRED: Include exactly one tool call that returns "
        "an error. Agent must read the error and try a different approach.\n"
        if variation["has_error"] else ""
    )

    # Build diagnosis context section
    diag_section = ""
    if diagnosis:
        root_cause = diagnosis.get("root_cause", "")
        fix = diagnosis.get("fix", "")
        reason = diagnosis.get("reason", "")

        parts = []
        if root_cause:
            parts.append(f"Root cause: {root_cause}")
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
avoids the failure pattern described above. Specifically:
- {fix or root_cause}
- Show clean, direct task completion without the described failure mode
- If the model was looping or repeating actions, show examples that complete
  in a single pass with no redundant steps
"""

    return f"""\
You are generating synthetic fine-tuning data for training an LLM to act as \
an OpenClaw AI agent called Clawd.

## Task Being Tested
Name: {task["name"]}
ID: {task_id}
Complexity: {task["complexity"]}

Original task prompt:
\"\"\"{task["prompt"]}\"\"\"

## Grading Criteria (ALL must be satisfied)
{grading_list}
{diag_section}
## Variation Type: {variation["id"]}
User style: {variation["user_style"]}
Scenario: {variation["scenario"]}
{error_note}

## Your Job
Generate {epc} diverse, complete agent conversation examples.

Each example MUST follow this exact JSON structure:
{{
  "user_message": "<user request, phrased per the variation style>",
  "turns": [
    {{
      "role": "assistant",
      "content": "<action or thinking — include <tool_call> tags when calling tools>"
    }},
    {{
      "role": "tool_result",
      "content": "<realistic JSON result from the tool>"
    }},
    ... more turns ...
    {{
      "role": "assistant",
      "content": "<final confirmation summarising what was done>"
    }}
  ]
}}

{tools_hint}

Tool call format (inside assistant content):
<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>

## Critical Rules
1. user_message must vary meaningfully across the {epc} examples.
2. Tool arguments must be realistic and specific to the task.
3. tool_result content must be plausible — use real-looking paths, timestamps, data.
4. For tasks with specific expected values (exact figures, filenames, dates), those \
exact values MUST appear in the tool results and final response.
5. The final assistant turn must confirm completion and satisfy ALL grading criteria.
6. Do NOT skip required tool calls — if the task needs a file created, call write_file.
7. For multi-document tasks, show ALL required files being read individually.
8. JSON ESCAPING IS MANDATORY: Inside any JSON string value, you MUST escape:
   - Double quotes as \\\"  (e.g. content with "quotes" → \\"quotes\\")
   - Backslashes as \\\\
   - Newlines as \\n
9. ONLY USE REAL CLAWD TOOLS. The ONLY valid tool names are:
   read_file, write_file, create_directory, list_files, run_bash, run_python,
   web_search, fetch_url, create_calendar_event, draft_email, search_emails,
   read_email, generate_image, read_memory, write_memory, search_skills, install_skill.
   NEVER invent tool names.
10. task_00_sanity is a SIMPLE hello/greeting task. No tool calls. Keep it trivial.
11. The final assistant message MUST end with a complete sentence (period, !, or ?).

Return ONLY a valid JSON array of {epc} objects. No markdown, no preamble.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_submit(diagnosis_file: str | None = None, only_tasks: list | None = None,
               min_per_task: int = 0):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load diagnosis
    diagnosis = {}
    if diagnosis_file and Path(diagnosis_file).exists():
        diagnosis = json.loads(Path(diagnosis_file).read_text())
        print(f"  Loaded diagnosis for {len(diagnosis)} tasks")

    deficits = compute_deficits(only_tasks=only_tasks)

    # For weak tasks, ensure at least min_per_task new examples even if
    # the task is already at the count target. A task can have 70 examples
    # and still score 0% — it needs BETTER data, not just more.
    if min_per_task > 0 and only_tasks:
        for task_id in only_tasks:
            if task_id in TASKS:
                deficits[task_id] = max(deficits.get(task_id, 0), min_per_task)

    if not deficits:
        print("All tasks are at or above target. Nothing to do.")
        sys.exit(0)

    print(f"Deficit tasks: {list(deficits.keys())}")

    requests = []
    for task_id, needed in deficits.items():
        task = TASKS[task_id]
        epc  = _epc(task_id)
        n_calls = (needed + epc - 1) // epc
        max_tok = 16000 if task_id in HARD_TASKS else 8192

        # Get diagnosis for this task (may be empty)
        task_diag = diagnosis.get(task_id, {})

        # Select variations weighted by diagnosis
        if task_diag:
            variations = select_variations(task_id, task_diag, n_calls)
            print(f"  {task_id}: {n_calls} calls (diagnosis-weighted variations)")
        else:
            # Fallback to round-robin if no diagnosis
            variations = [VARIATION_CONFIGS[i % len(VARIATION_CONFIGS)]
                          for i in range(n_calls)]
            print(f"  {task_id}: {n_calls} calls (round-robin, no diagnosis)")

        for i, variation in enumerate(variations):
            custom_id = f"targeted__{task_id}__{variation['id']}__{i:03d}"
            prompt = build_targeted_meta_prompt(
                task_id, task, variation, epc, task_diag
            )
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": max_tok,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })

    client = anthropic.Anthropic(api_key=api_key)
    batch  = client.messages.batches.create(requests=requests)
    BATCH_FILE.write_text(batch.id)

    total_examples = sum(
        _epc(r["custom_id"].split("__")[1]) for r in requests
    )
    print(f"\n  Submitted {len(requests)} requests → ~{total_examples} new examples")
    print(f"  Batch ID: {batch.id}  (saved to {BATCH_FILE})")


def cmd_collect():
    """Collect results and append to train/val — same logic as topup.py."""
    if not BATCH_FILE.exists():
        print("No targeted batch found. Run: python targeted_topup.py submit")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)
    batch_id = BATCH_FILE.read_text().strip()

    from tqdm import tqdm
    from collections import defaultdict

    new_by_task = defaultdict(list)
    api_errors  = 0
    parse_fails = 0

    for result in tqdm(client.messages.batches.results(batch_id), desc="Collecting"):
        cid = result.custom_id                     # targeted__task_XX__var__NNN
        parts = cid.split("__")
        task_id = parts[1] if len(parts) >= 2 else "unknown"

        if result.result.type == "errored":
            api_errors += 1
            continue

        msg = result.result.message
        raw_text = msg.content[0].text if msg.content else ""

        examples = extract_json_array(raw_text)
        if not examples:
            parse_fails += 1
            # Save raw for debugging
            raw_file = RAW_DIR / f"{cid}.json"
            raw_file.write_text(raw_text)
            continue

        for ex in examples:
            parsed = parse_example(ex, task_id)
            if parsed:
                parsed["source"] = "targeted"
                new_by_task[task_id].append(parsed)

    # Append to train/val
    total_train = 0
    total_val   = 0
    per_task_report = {}

    for task_id, examples in sorted(new_by_task.items()):
        random.shuffle(examples)
        val_target = max(2, round(TARGET_PER_TASK * VAL_SPLIT))
        val_count  = min(val_target, max(1, len(examples) // 5))
        val_set    = examples[:val_count]
        train_set  = examples[val_count:]

        with open(TRAIN_FILE, "a") as f:
            for ex in train_set:
                f.write(json.dumps(ex) + "\n")
        with open(VAL_FILE, "a") as f:
            for ex in val_set:
                f.write(json.dumps(ex) + "\n")

        total_train += len(train_set)
        total_val   += len(val_set)
        per_task_report[task_id] = len(examples)

    print(f"\n{'─'*50}")
    print(f"✓ Targeted top-up complete")
    print(f"  API errors:     {api_errors}")
    print(f"  Parse failures: {parse_fails}")
    print(f"\n  Per-task additions:")
    for tid, count in sorted(per_task_report.items()):
        print(f"    {tid:<45} + {count:>3}")
    print(f"\n  New train: {total_train}")
    print(f"  New val:   {total_val}")


def cmd_run(diagnosis_file: str | None = None, only_tasks: list | None = None,
            min_per_task: int = 0):
    """Submit → poll → collect in one shot."""
    cmd_submit(diagnosis_file=diagnosis_file, only_tasks=only_tasks,
               min_per_task=min_per_task)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)
    batch_id = BATCH_FILE.read_text().strip()

    import time
    print("\nPolling every 2 minutes...")
    while True:
        batch  = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        ts = time.strftime("%H:%M")
        print(f"  [{ts}] processing={counts.processing} "
              f"succeeded={counts.succeeded} errored={counts.errored}")
        if batch.processing_status == "ended":
            break
        time.sleep(120)

    cmd_collect()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diagnosis-aware data generation")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Submit → poll → collect")
    run_p.add_argument("--diagnosis-file", type=str, default=None)
    run_p.add_argument("--tasks", type=str, default=None,
                       help="Comma-separated task IDs to target")
    run_p.add_argument("--min-per-task", type=int, default=0,
                       help="Minimum new examples per task (overrides deficit calculation)")

    sub.add_parser("count", help="Show current vs target per task")

    submit_p = sub.add_parser("submit", help="Submit batch")
    submit_p.add_argument("--diagnosis-file", type=str, default=None)
    submit_p.add_argument("--tasks", type=str, default=None)
    submit_p.add_argument("--min-per-task", type=int, default=0)

    sub.add_parser("status", help="Check batch status")
    sub.add_parser("collect", help="Collect results")

    args = parser.parse_args()
    only_tasks = args.tasks.split(",") if hasattr(args, "tasks") and args.tasks else None
    min_per_task = getattr(args, "min_per_task", 0)

    if args.command == "run":
        cmd_run(diagnosis_file=args.diagnosis_file, only_tasks=only_tasks,
                min_per_task=min_per_task)
    elif args.command == "count":
        cmd_count()
    elif args.command == "submit":
        cmd_submit(diagnosis_file=args.diagnosis_file, only_tasks=only_tasks)
    elif args.command == "status":
        cmd_status()
    elif args.command == "collect":
        cmd_collect()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
