#!/usr/bin/env python3
"""
Deep validation: paranoid, diligent analysis of training data quality.

Three levels:
1. STRUCTURAL — tool names, args, format (cheap, runs on every example)
2. STATISTICAL — diversity, completeness patterns (cheap, runs on all examples)
3. SEMANTIC — does the data teach correct behavior? (uses Claude to reason
   about a SAMPLE of examples against the ground truth task definition)

The semantic check is the key differentiator. It sends Claude:
- The actual benchmark task definition (ground truth)
- A sample of training examples for that task
- And asks: "Would a model trained on these examples pass this benchmark?"

Usage:
    python -m datagen.deep_validate                    # all tasks
    python -m datagen.deep_validate --task task_12     # single task
    python -m datagen.deep_validate --no-llm           # skip semantic (LLM) checks
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config
from agents.base import TASK_IDS, log_print

_cfg = load_config()
TRAIN_FILE = _cfg.train_file


def load_examples(task_filter=None):
    examples = []
    if not TRAIN_FILE.exists():
        return examples
    for line in TRAIN_FILE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
            if task_filter and ex.get("task_id") != task_filter:
                continue
            examples.append(ex)
        except json.JSONDecodeError:
            continue
    return examples


def extract_tool_calls(messages):
    calls = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for block in re.findall(r'<tool_call>(.*?)</tool_call>', msg["content"], re.DOTALL):
            try:
                calls.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                calls.append({"name": "PARSE_ERROR"})
    return calls


def extract_written_files(messages):
    files = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for block in re.findall(r'<tool_call>(.*?)</tool_call>', msg["content"], re.DOTALL):
            try:
                obj = json.loads(block.strip())
                if obj.get("name") in ("write_file", "create_directory"):
                    path = obj.get("arguments", {}).get("path", "")
                    if path:
                        files.append(path)
            except json.JSONDecodeError:
                continue
    return files


def load_ground_truth(task_id):
    """Load ground truth — load_tasks() already maps PinchBench IDs to our internal IDs."""
    from datagen.task_loader import load_tasks
    try:
        all_tasks = load_tasks()
        if task_id in all_tasks:
            return all_tasks[task_id]
        print(f"  WARNING: No ground truth found for {task_id}")
        return None
    except Exception as e:
        print(f"  WARNING: Failed to load ground truth for {task_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 1: STRUCTURAL (per example)
# ─────────────────────────────────────────────────────────────────────────────

def structural_check(examples):
    """Run existing validate_data checks. Returns summary."""
    try:
        from datagen.validate_data import validate_example
        clean = 0
        critical = 0
        for ex in examples:
            issues = validate_example(ex)
            has_bad = any(i["severity"] in ("critical", "high") for i in issues)
            if has_bad:
                critical += 1
            elif not issues:
                clean += 1
        return {"clean": clean, "critical": critical, "total": len(examples)}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2: STATISTICAL (across all examples for a task)
# ─────────────────────────────────────────────────────────────────────────────

def statistical_check(task_id, examples):
    """Analyze patterns across all examples for a task."""
    issues = []
    n = len(examples)
    if n == 0:
        return {"issues": ["NO DATA"], "stats": {}}

    tool_calls_counts = []
    single_tool = 0
    has_write = 0
    has_summary = 0
    has_error_recovery = 0
    user_messages = []
    tool_combos = set()

    for ex in examples:
        msgs = ex.get("messages", [])
        calls = extract_tool_calls(msgs)
        written = extract_written_files(msgs)

        tool_calls_counts.append(len(calls))
        if len(calls) <= 1:
            single_tool += 1
        if written:
            has_write += 1

        # Final summary
        last_asst = ""
        for msg in reversed(msgs):
            if msg.get("role") == "assistant":
                last_asst = msg["content"]
                break
        if last_asst and "<tool_call>" not in last_asst and len(last_asst) > 20:
            has_summary += 1

        # Error recovery
        for msg in msgs:
            if msg.get("role") == "tool":
                c = msg.get("content", "").lower()
                if "error" in c or "failed" in c or "not found" in c:
                    has_error_recovery += 1
                    break

        # User message
        for msg in msgs:
            if msg.get("role") == "user":
                user_messages.append(msg["content"][:80])
                break

        # Tool combo
        combo = tuple(sorted(set(c.get("name", "") for c in calls)))
        tool_combos.add(combo)

    pct_single = 100 * single_tool / n
    pct_write = 100 * has_write / n
    pct_summary = 100 * has_summary / n
    unique_prompts = len(set(user_messages))

    if pct_single > 40 and task_id != "task_00_sanity":
        issues.append(f"{pct_single:.0f}% of examples use only 1 tool — model may learn to stop early")
    if pct_write < 50 and task_id not in ("task_00_sanity",):
        issues.append(f"Only {pct_write:.0f}% write an output file — task may require file output")
    if pct_summary < 70:
        issues.append(f"Only {pct_summary:.0f}% have a final summary message")
    if len(tool_combos) < 3 and n > 10:
        issues.append(f"Only {len(tool_combos)} unique tool combinations — low diversity")
    if unique_prompts < n * 0.7 and n > 10:
        issues.append(f"Only {unique_prompts}/{n} unique user messages ({100*unique_prompts/n:.0f}%)")

    stats = {
        "avg_tools": round(sum(tool_calls_counts) / n, 1),
        "pct_single_tool": round(pct_single),
        "pct_write": round(pct_write),
        "pct_summary": round(pct_summary),
        "pct_error_recovery": round(100 * has_error_recovery / n),
        "unique_tool_combos": len(tool_combos),
        "unique_prompts": unique_prompts,
    }
    return {"issues": issues, "stats": stats}


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3: SEMANTIC (Claude reasons about sample vs ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def semantic_check(task_id, examples, ground_truth):
    """Ask Claude to reason about whether training data teaches correct behavior."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"skipped": True, "reason": "ANTHROPIC_API_KEY not set"}
    if not ground_truth:
        return {"skipped": True, "reason": "ground truth not loaded"}

    import anthropic

    # Sample 3 examples to keep cost low
    sample = random.sample(examples, min(3, len(examples)))

    # Format examples for Claude
    example_summaries = []
    for i, ex in enumerate(sample):
        msgs = ex.get("messages", [])
        calls = extract_tool_calls(msgs)
        written = extract_written_files(msgs)
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "?")
        final = ""
        for msg in reversed(msgs):
            if msg.get("role") == "assistant":
                final = msg["content"][:300]
                break

        tool_summary = [f"{c.get('name', '?')}({', '.join(c.get('arguments', {}).keys())})"
                        for c in calls]

        example_summaries.append(
            f"Example {i+1}:\n"
            f"  User: {user_msg[:200]}\n"
            f"  Tool calls ({len(calls)}): {' → '.join(tool_summary)}\n"
            f"  Files written: {written}\n"
            f"  Final response: {final[:200]}"
        )

    examples_text = "\n\n".join(example_summaries)

    prompt = f"""You are a data quality analyst reviewing training examples for an AI agent.

## Benchmark Task Definition (ground truth — this is what the model must learn to do):

{ground_truth['raw_content'][:3000]}

## Sample Training Examples (3 of {len(examples)} total):

{examples_text}

## Your Analysis

Analyze whether these training examples would teach a model to PASS the benchmark task above. Be specific and critical. Consider:

1. **Approach alignment**: Do the examples demonstrate the correct workflow to complete this task? Are the right tools being used in the right order?
2. **Output completeness**: Do the examples produce all expected outputs (files, content, format)?
3. **Quality of demonstration**: Would a model that imitates these examples score well on the grading criteria?
4. **Missing patterns**: What behavior does the benchmark test for that these examples DON'T demonstrate?
5. **Harmful patterns**: Do any examples teach behavior that would HURT benchmark performance?

Return JSON only:
{{
    "verdict": "GOOD | NEEDS_WORK | BAD",
    "reasoning": "2-3 sentence explanation",
    "issues": ["specific issue 1", "specific issue 2"],
    "missing_behaviors": ["what the examples should teach but don't"],
    "recommendation": "what to change in data generation"
}}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=_cfg.claude.analysis,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()

        # Parse JSON
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"verdict": "UNKNOWN", "reasoning": raw[:500], "issues": [], "missing_behaviors": []}

        cost = (getattr(resp.usage, 'input_tokens', 0) * 0.003 +
                getattr(resp.usage, 'output_tokens', 0) * 0.015) / 1000
        result["cost_usd"] = round(cost, 4)
        return result

    except Exception as e:
        return {"skipped": True, "reason": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def deep_validate_task(task_id, examples, ground_truth=None, use_llm=True):
    """Full deep validation for a single task."""
    structural = structural_check(examples)
    statistical = statistical_check(task_id, examples)
    semantic = {}

    if use_llm and examples and ground_truth:
        semantic = semantic_check(task_id, examples, ground_truth)
        if semantic.get("skipped"):
            print(f"  LLM SKIPPED for {task_id}: {semantic.get('reason', '?')}")
    elif use_llm and not ground_truth and examples:
        print(f"  LLM SKIPPED for {task_id}: no ground truth loaded")
    elif use_llm and not examples:
        pass  # no data, nothing to check

    all_issues = statistical["issues"][:]
    if semantic.get("issues"):
        all_issues.extend([f"SEMANTIC: {i}" for i in semantic["issues"]])
    if semantic.get("missing_behaviors"):
        all_issues.extend([f"MISSING: {b}" for b in semantic["missing_behaviors"]])

    return {
        "task_id": task_id,
        "ground_truth_name": ground_truth.get("name", "?") if ground_truth else "(not loaded)",
        "count": len(examples),
        "structural": structural,
        "statistical": statistical.get("stats", {}),
        "semantic": semantic,
        "issues": all_issues,
        "verdict": semantic.get("verdict", "NO_LLM_CHECK"),
    }


def main():
    parser = argparse.ArgumentParser(description="Deep training data validation")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--no-llm", action="store_true", help="Skip semantic (LLM) checks")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  DEEP DATA QUALITY ANALYSIS")
    print(f"{'='*70}")

    all_examples = load_examples(task_filter=args.task)
    by_task = defaultdict(list)
    for ex in all_examples:
        by_task[ex.get("task_id", "?")].append(ex)

    print(f"  Total examples: {len(all_examples)}")
    print(f"  LLM semantic check: {'OFF' if args.no_llm else 'ON'}\n")

    task_ids = [args.task] if args.task else TASK_IDS
    results = []
    total_cost = 0

    print(f"  {'Task':<35} {'n':>4} {'struct':>7} {'verdict':>10} {'issues':>6}")
    print(f"  {'-'*35} {'-'*4} {'-'*7} {'-'*10} {'-'*6}")

    for task_id in task_ids:
        examples = by_task.get(task_id, [])
        ground_truth = load_ground_truth(task_id)
        result = deep_validate_task(task_id, examples, ground_truth, use_llm=not args.no_llm)
        results.append(result)

        struct = result.get("structural", {})
        struct_str = f"{struct.get('clean', 0)}/{struct.get('total', 0)}"
        verdict = result.get("verdict", "?")
        cost = result.get("semantic", {}).get("cost_usd", 0)
        total_cost += cost

        marker = " ⚠" if result["issues"] else ""
        print(f"  {task_id:<35} {result['count']:>4} {struct_str:>7} {verdict:>10} "
              f"{len(result['issues']):>5}{marker}")

    # Show issues
    flagged = [r for r in results if r["issues"]]
    if flagged:
        print(f"\n{'='*70}")
        print(f"  ISSUES ({len(flagged)} tasks)")
        print(f"{'='*70}")
        for result in flagged:
            print(f"\n  {result['task_id']} ({result['count']} examples) — {result['verdict']}")
            if result.get("semantic", {}).get("reasoning"):
                print(f"  Claude says: {result['semantic']['reasoning']}")
            for issue in result["issues"]:
                print(f"    ⚠ {issue}")
            if result.get("semantic", {}).get("recommendation"):
                print(f"    → Recommendation: {result['semantic']['recommendation']}")

    if total_cost > 0:
        print(f"\n  LLM cost: ${total_cost:.4f}")

    # Save report
    report_file = _cfg.data_dir / "deep_validation_report.json"
    report_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Report saved: {report_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
