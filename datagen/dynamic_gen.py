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
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import anthropic

from utils.config import load_config
from utils.prompts import OPENCLAW_SYSTEM, VALID_TOOLS
from datagen.task_loader import load_tasks, load_task
from datagen.gen_utils import (
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

# Default generation parameters -- self-healing pilot adjusts dynamically
DEFAULT_EPC = 3        # examples per call (reduced to 1 on truncation)
DEFAULT_MAX_TOK = 8192  # increased up to 32K on truncation


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
    # Benchmark task content — cap at 4000 chars to avoid hitting input token limits
    # (some tasks like task_19 have 1300-line .md files with full grading code)
    raw_content = task_def["raw_content"]
    if len(raw_content) > 4000:
        raw_content = raw_content[:4000] + "\n\n[... task definition truncated for token budget ...]"

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
        epc = DEFAULT_EPC
        n_calls = (needed + epc - 1) // epc
        max_tok = DEFAULT_MAX_TOK

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
        DEFAULT_EPC for r in requests
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
                parsed["source"] = f"dynamic_{datetime.now().strftime('%Y%m%d')}"
                new_by_task[task_id].append(parsed)

    # Execution filter: run traces for real, reject broken ones
    from datagen.trace_executor import execute_trace, passes_quality_filter, _find_task_fixtures
    exec_rejected = 0
    for task_id in list(new_by_task.keys()):
        fixtures = _find_task_fixtures(task_id)
        approved = []
        for ex in new_by_task[task_id]:
            exec_result = execute_trace(ex, task_id, fixtures_dir=fixtures)
            if passes_quality_filter(exec_result):
                approved.append(exec_result["example"])
            else:
                exec_rejected += 1
        new_by_task[task_id] = approved

    if exec_rejected > 0:
        print(f"  Execution filter: rejected {exec_rejected} examples")

    # Append to train/val (append mode -- does not rewrite existing data)
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


def _pilot_batch(task_id: str, task_def: dict, client, prompt: str,
                  max_tok: int | None = None) -> tuple[list, dict]:
    """Call the direct Messages API for pilot generation. Returns (parsed_examples, metadata).

    Uses the synchronous Messages API (not batch) for reliability. The batch API
    has shown repeated 500 errors on single-request batches.
    """
    if max_tok is None:
        max_tok = DEFAULT_MAX_TOK

    meta = {"stop_reason": None, "was_truncated": False, "max_tokens_used": max_tok,
            "input_tokens": 0, "output_tokens": 0}

    # Retry on transient API errors
    resp = None
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=max_tok,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            if attempt < 2:
                print(f"    API error (attempt {attempt+1}/3): {e}")
                time.sleep(10 * (attempt + 1))
            else:
                print(f"    API error (attempt 3/3): {e}")
                return [], meta

    if resp is None:
        return [], meta

    meta["stop_reason"] = resp.stop_reason
    meta["was_truncated"] = resp.stop_reason == "max_tokens"
    if hasattr(resp, "usage"):
        meta["input_tokens"] = getattr(resp.usage, "input_tokens", 0)
        meta["output_tokens"] = getattr(resp.usage, "output_tokens", 0)

    raw_text = resp.content[0].text if resp.content else ""
    parsed = []
    examples = extract_json_array(raw_text)
    if examples:
        for ex in examples:
            p = parse_example(ex, task_id)
            if p:
                p["source"] = "dynamic_pilot"
                parsed.append(p)
    return parsed, meta


def _pilot_generate(task_id: str, task_def: dict, client, diagnosis: dict | None = None,
                     failure: dict | None = None, max_attempts: int = 3) -> tuple[str, list, dict]:
    """Generate pilot examples, validate, self-heal, retry.

    Returns (verdict, examples, learned_params) where learned_params contains
    the epc and max_tok that worked, so bulk generation can reuse them.
    """
    from datagen.deep_validate import semantic_check

    variation = VARIATION_CONFIGS[0]
    epc = DEFAULT_EPC
    max_tok = DEFAULT_MAX_TOK
    refinement_feedback = ""
    verdict = "FAILED"
    parsed = []
    failure_log = []  # track what went wrong for final diagnostics

    for attempt in range(1, max_attempts + 1):
        print(f"\n    Pilot attempt {attempt}/{max_attempts} for {task_id} (epc={epc}, max_tok={max_tok})...")

        if failure:
            prompt = build_adversarial_prompt(task_id, task_def, failure, epc)
        else:
            prompt = build_dynamic_meta_prompt(task_id, task_def, variation, epc, diagnosis=diagnosis)
        if refinement_feedback:
            prompt += f"""

## IMPORTANT: Previous attempt had issues. Fix these:
{refinement_feedback}

Generate improved examples that address ALL the issues above.
"""

        parsed, meta = _pilot_batch(task_id, task_def, client, prompt, max_tok=max_tok)

        # ── Self-heal: truncation ──
        if meta["was_truncated"]:
            failure_log.append(f"attempt {attempt}: truncated (max_tokens={max_tok}, output={meta['output_tokens']})")
            print(f"    Output truncated (stop_reason=max_tokens, used {meta['output_tokens']} tokens)")

            if epc > 1:
                # Try fewer examples per call first
                epc = 1
                print(f"    Self-heal: reduced epc to 1")
                refinement_feedback = (
                    "Output was truncated. Generate only 1 example. "
                    "Keep the example complete -- do not cut off mid-response."
                )
                continue
            elif max_tok < 32000:
                # Already at epc=1, increase token budget
                max_tok = min(max_tok * 2, 32000)
                print(f"    Self-heal: increased max_tokens to {max_tok}")
                refinement_feedback = (
                    "Output was truncated even with 1 example. "
                    "The example must be complete. Prioritize completeness over detail."
                )
                continue
            else:
                # Can't fix truncation -- max budget reached
                failure_log.append(f"attempt {attempt}: truncation unfixable (max_tok=32000, epc=1)")
                print(f"    Cannot fix truncation: already at epc=1 and max_tokens=32000")
                verdict = "BAD"
                break

        # ── Self-heal: no valid examples parsed ──
        if not parsed:
            failure_log.append(f"attempt {attempt}: no valid examples parsed")
            print(f"    No valid examples from pilot batch")
            if epc > 1:
                epc = 1
                print(f"    Self-heal: reduced epc to 1 for simpler output")
            refinement_feedback = (
                "No valid examples were produced. Output must be a valid JSON array "
                "with user_message and turns fields. Generate exactly 1 example."
            )
            continue

        print(f"    Generated {len(parsed)} examples, validating...")

        # ── Semantic validation ──
        result = semantic_check(task_id, parsed, task_def)

        if result.get("skipped"):
            print(f"    Semantic check skipped: {result.get('reason')}")
            return "UNVERIFIED", parsed, {"epc": epc, "max_tok": max_tok}

        verdict = result.get("verdict", "UNKNOWN")
        reasoning = result.get("reasoning", "")
        issues = result.get("issues", [])
        missing = result.get("missing_behaviors", [])
        recommendation = result.get("recommendation", "")

        print(f"    Verdict: {verdict}")
        if reasoning:
            print(f"    Reasoning: {reasoning[:200]}")

        if verdict == "GOOD":
            # Execute traces for real before accepting
            from datagen.trace_executor import execute_trace, passes_quality_filter, _find_task_fixtures
            fixtures = _find_task_fixtures(task_id)
            exec_passed = []
            exec_failed = 0
            for ex in parsed:
                exec_result = execute_trace(ex, task_id, fixtures_dir=fixtures)
                if passes_quality_filter(exec_result):
                    exec_passed.append(exec_result["example"])
                else:
                    exec_failed += 1
                    failed_tools = [e for e in exec_result["execution_log"]
                                    if e.get("status") == "error"]
                    if failed_tools:
                        tool_names = [e["tool"] for e in failed_tools]
                        print(f"      Execution rejected: {', '.join(tool_names)} failed")

            if exec_failed > 0:
                print(f"    Execution filter: {len(exec_passed)} passed, {exec_failed} rejected")

            if exec_passed:
                parsed = exec_passed
                print(f"    Pilot passed on attempt {attempt}")
                return verdict, parsed, {"epc": epc, "max_tok": max_tok}

            # All examples failed execution -- retry with feedback
            failure_log.append(f"attempt {attempt}: semantic GOOD but execution failed")
            refinement_feedback = (
                "Examples passed semantic validation but failed real execution. "
                "Ensure Python code is syntactically correct and runs without errors. "
                "Ensure write_file produces non-empty content. "
                "Ensure the trace completes with a final assistant message."
            )
            continue

        # Build refinement feedback for next attempt
        failure_log.append(f"attempt {attempt}: {verdict} -- {'; '.join(issues[:3])}")
        feedback_parts = []
        if issues:
            feedback_parts.append("Issues found:\n" + "\n".join(f"  - {i}" for i in issues))
        if missing:
            feedback_parts.append("Missing behaviors:\n" + "\n".join(f"  - {m}" for m in missing))
        if recommendation:
            feedback_parts.append(f"Recommendation: {recommendation}")
        refinement_feedback = "\n".join(feedback_parts)

        print(f"    Issues: {len(issues)}, Missing: {len(missing)}")
        if attempt < max_attempts:
            print(f"    Refining prompt and retrying...")

    # All attempts exhausted -- diagnose why
    print(f"\n    PILOT FAILED for {task_id} after {max_attempts} attempts (last: {verdict})")
    print(f"    Failure log:")
    for entry in failure_log:
        print(f"      - {entry}")

    return "BAD", [], {"epc": epc, "max_tok": max_tok}  # return empty -- do not save bad data


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL MODE -- generates from benchmark failure transcripts
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_log(log_dir: Path) -> Path | None:
    """Find the most recent benchmark log file."""
    logs = sorted(log_dir.glob("bench_*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def extract_task_section(log_text: str, task_id: str) -> str | None:
    """Extract the section of a benchmark log for a specific task."""
    pattern = rf'starting task: {re.escape(task_id)}.*?(?=starting task:|$)'
    m = re.search(pattern, log_text, re.DOTALL)
    return m.group() if m else None


def extract_failure_pattern(section: str, task_id: str) -> dict:
    """Analyze a task section to identify what the model did wrong."""
    result = {
        "task_id": task_id,
        "score": None,
        "judge_notes": "",
        "patterns": [],
        "errors": [],
    }

    score_match = re.search(
        rf'Task {re.escape(task_id)}:\s*([01](?:\.\d+)?)/1\.0\s*\((\d+)%\)\s*-\s*(\w+)',
        section
    )
    if score_match:
        result["score"] = float(score_match.group(1))

    notes_match = re.search(r'Notes:\s*(.+?)(?:\n={5,}|\Z)', section, re.DOTALL)
    if notes_match:
        result["judge_notes"] = notes_match.group(1).strip()[:500]

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
    for line in section.splitlines():
        if re.search(r'\bERROR\b', line):
            result["errors"].append(line.strip()[:200])
        if "Failed to parse transcript" in line:
            result["patterns"].append("malformed_output")
            break

    return result


def build_adversarial_prompt(
    task_id: str, task_def: dict, failure: dict, n_examples: int = 3,
) -> str:
    """Build a prompt showing the model's failure and asking for correct examples."""
    raw_content = task_def.get("raw_content", "")
    if len(raw_content) > 4000:
        raw_content = raw_content[:4000] + "\n\n[... truncated ...]"

    failure_lines = []
    if failure.get("judge_notes"):
        failure_lines.append(f"Judge notes: {failure['judge_notes']}")
    pattern_descriptions = {
        "looping": "The model got stuck in a loop, calling the same tool repeatedly",
        "wrong_filename": "The model tried to read a nonexistent file instead of the correct one",
        "binary_as_text": "The model read a binary file (xlsx/pdf) as raw text, getting garbage bytes",
        "excessive_tool_calls": "The model made too many tool calls without progress",
        "truncation": "The model's response was truncated mid-sentence",
        "malformed_output": "The model produced malformed output that couldn't be parsed",
    }
    for p in failure.get("patterns", []):
        if p in pattern_descriptions:
            failure_lines.append(f"Pattern: {pattern_descriptions[p]}")
    if failure.get("errors"):
        failure_lines.append(f"Errors from log: {failure['errors'][:2]}")

    failure_text = "\n".join(failure_lines) if failure_lines else "Task failed with low score."
    score_text = f"{failure['score']:.1f}/1.0" if failure.get("score") is not None else "unknown"
    tools_list = ", ".join(sorted(VALID_TOOLS))

    return f"""\
You are generating TARGETED training data for an LLM agent called Clawd.

The model FAILED this task during benchmarking. Your job is to generate
{n_examples} training examples showing the CORRECT approach.

## Agent System Prompt
{OPENCLAW_SYSTEM}

## Benchmark Task Definition
{raw_content}

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

ONLY USE REAL CLAWD TOOLS: {tools_list}

Return ONLY a valid JSON array of {n_examples} objects. No markdown, no preamble.
"""


def cmd_run(
    only_tasks: list[str] | None = None,
    all_below: int | None = None,
    min_per_task: int = 0,
    diagnosis_file: str | None = None,
    benchmark_log: str | None = None,
):
    """Pilot-validate-refine then bulk generate for each task.

    Modes:
    - Standard: generates from task definitions + optional diagnosis context
    - Adversarial (benchmark_log provided): generates from failure transcripts

    Self-healing pilot adapts epc and max_tokens dynamically -- no hardcoded
    HARD_TASKS list needed.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tasks = resolve_tasks(only_tasks=only_tasks, all_below=all_below)
    if not tasks:
        print("No tasks to generate for.")
        return

    # Load diagnosis context
    diagnosis = {}
    if diagnosis_file and Path(diagnosis_file).exists():
        diagnosis = json.loads(Path(diagnosis_file).read_text())

    # Load benchmark log for adversarial mode
    adversarial_failures = {}
    if benchmark_log:
        log_path = Path(benchmark_log)
        if log_path.exists():
            log_text = log_path.read_text(errors="replace")
            print(f"  Adversarial mode: {log_path.name}")
            for task_id in tasks:
                section = extract_task_section(log_text, task_id)
                if section:
                    failure = extract_failure_pattern(section, task_id)
                    if failure["score"] is None or failure["score"] < 0.8:
                        adversarial_failures[task_id] = failure
        else:
            print(f"  WARNING: benchmark log not found: {benchmark_log}")

    deficits = compute_dynamic_deficits(tasks, min_per_task=min_per_task)
    if not deficits:
        print("All tasks at or above target.")
        return

    print(f"\n{'='*60}")
    print(f"  DYNAMIC GENERATION — PILOT → VALIDATE → BULK")
    print(f"{'='*60}")
    print(f"  Tasks: {len(deficits)}")
    print(f"  Target: {TARGET_PER_TASK}")

    # Phase 1: Pilot each task
    pilot_results = {}
    bulk_tasks = {}

    for task_id, needed in deficits.items():
        task_def = tasks[task_id]
        task_diag = diagnosis.get(task_id)
        failure = adversarial_failures.get(task_id)

        print(f"\n{'─'*60}")
        print(f"  {task_id} — need {needed} examples")
        print(f"  Ground truth: {task_def.get('name', '?')}")
        if failure:
            print(f"  Mode: adversarial (score={failure.get('score', '?')}, patterns={failure.get('patterns', [])})")

        verdict, pilot_examples, learned = _pilot_generate(
            task_id, task_def, client, diagnosis=task_diag, failure=failure
        )
        pilot_results[task_id] = {"verdict": verdict, "n_pilot": len(pilot_examples)}

        if verdict in ("GOOD", "UNVERIFIED") and pilot_examples:
            for ex in pilot_examples:
                with open(TRAIN_FILE, "a") as f:
                    f.write(json.dumps(ex) + "\n")
            print(f"    Saved {len(pilot_examples)} pilot examples")

            remaining = needed - len(pilot_examples)
            if remaining > 0:
                bulk_tasks[task_id] = {"task_def": task_def, "needed": remaining,
                                       "diagnosis": task_diag, "failure": failure,
                                       "learned_epc": learned.get("epc", DEFAULT_EPC),
                                       "learned_max_tok": learned.get("max_tok", DEFAULT_MAX_TOK)}
        else:
            print(f"    SKIPPING {task_id} — pilot verdict: {verdict}")

    # Phase 2: Bulk generate (single batch for all passing tasks)
    if bulk_tasks:
        print(f"\n{'='*60}")
        print(f"  BULK GENERATION — {len(bulk_tasks)} tasks")
        print(f"{'='*60}")

        requests = []
        for task_id, info in bulk_tasks.items():
            task_def = info["task_def"]
            needed = info["needed"]
            task_diag = info["diagnosis"]
            task_failure = info.get("failure")
            # Use pilot's learned parameters -- avoids truncation in bulk
            epc = info.get("learned_epc", DEFAULT_EPC)
            max_tok = info.get("learned_max_tok", DEFAULT_MAX_TOK)
            n_calls = (needed + epc - 1) // epc

            for i in range(n_calls):
                variation = VARIATION_CONFIGS[i % len(VARIATION_CONFIGS)]
                custom_id = f"dynamic__{task_id}__{variation['id']}__{i:03d}"
                if task_failure:
                    prompt = build_adversarial_prompt(task_id, task_def, task_failure, epc)
                else:
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

        if requests:
            batch = client.messages.batches.create(requests=requests)
            BATCH_FILE.write_text(batch.id)

            total_est = sum(DEFAULT_EPC for r in requests)
            print(f"  Submitted {len(requests)} requests → ~{total_est} examples")
            print(f"  Batch ID: {batch.id}")

            print("\n  Polling every 2 minutes...")
            while True:
                batch = client.messages.batches.retrieve(batch.id)
                counts = batch.request_counts
                ts = time.strftime("%H:%M")
                print(f"  [{ts}] processing={counts.processing} "
                      f"succeeded={counts.succeeded} errored={counts.errored}")
                if batch.processing_status == "ended":
                    break
                time.sleep(120)

            cmd_collect()

    # Summary
    print(f"\n{'='*60}")
    print(f"  GENERATION SUMMARY")
    print(f"{'='*60}")
    for task_id, info in pilot_results.items():
        status = "pilot+bulk" if task_id in bulk_tasks else "pilot only"
        if info["verdict"] == "BAD":
            status = "SKIPPED"
        print(f"  {task_id:<35} {info['verdict']:<12} {status}")
    print()


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
    run_p.add_argument("--benchmark-log", type=str, default=None,
                       help="Benchmark log file for adversarial generation (failure-aware)")

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

    benchmark_log = getattr(args, "benchmark_log", None)

    if args.command == "run":
        cmd_run(only_tasks=only_tasks, all_below=all_below,
                min_per_task=min_per_task, diagnosis_file=diagnosis_file,
                benchmark_log=benchmark_log)
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
