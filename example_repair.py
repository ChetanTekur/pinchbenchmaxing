#!/usr/bin/env python3
"""
Repair borderline training examples instead of deleting them.

Examples scoring 2-3/5 often have the right structure but fixable issues
(wrong value, truncated ending, missing criterion). This script sends them
to Claude with the judge's feedback, gets a repair, and re-scores.

Usage:
  python example_repair.py run                    # repair score 2-3 examples
  python example_repair.py run --min-score 2 --max-score 3
  python example_repair.py report                 # show repair stats
"""

import json
import os
import sys
import argparse
from pathlib import Path

import anthropic

from utils.config import load_config

_cfg        = load_config()
TRAIN_FILE  = _cfg.train_file
VAL_FILE    = _cfg.val_file
SCORES_FILE = _cfg.data_dir / "scores.json"
REPAIR_REPORT = _cfg.data_dir / "repair_report.json"
MODEL       = _cfg.claude.judge  # use same model as judge for consistency
MAX_BATCH   = 30  # cost control: don't repair more than this per pass


# ─────────────────────────────────────────────────────────────────────────────
# REPAIR LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def build_repair_prompt(example: dict, score_data: dict) -> str:
    """Build a prompt asking Claude to fix a borderline example."""
    score   = score_data.get("score", "?")
    issues  = score_data.get("issues", [])
    reasoning = score_data.get("reasoning", "")
    task_id = example.get("task_id", "unknown")

    issues_text = "\n".join(f"  - {issue}" for issue in issues) if issues else "  (no specific issues recorded)"

    # Serialize the example compactly
    example_json = json.dumps(example, indent=2, ensure_ascii=False)
    if len(example_json) > 8000:
        # Truncate very long examples to avoid token overflow
        example_json = example_json[:8000] + "\n... (truncated)"

    return f"""\
Fix this training example for task "{task_id}". It scored {score}/5.

## Issues Found by Quality Judge
{issues_text}

## Judge Reasoning
{reasoning}

## Original Example
```json
{example_json}
```

## Instructions
1. Fix ONLY the specific issues listed above
2. Do NOT change the overall structure, tool sequence, or conversation flow
3. If an issue is "missing expected value", add that value to the relevant tool_result
4. If an issue is "truncated", complete the final assistant message properly
5. If an issue is "wrong tool", replace with the correct tool name and arguments
6. Keep the user_message and general approach the same
7. Ensure all tool_call JSON is properly escaped

Return ONLY the repaired example as a single JSON object (not an array).
Same format as the original: {{"task_id": "...", "messages": [...]}}.
No markdown, no explanation.
"""


def repair_example(
    client: anthropic.Anthropic, example: dict, score_data: dict,
) -> dict | None:
    """Send example + feedback to Claude, return repaired version or None."""
    prompt = build_repair_prompt(example, score_data)
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()

        # Strip markdown fences
        import re
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()

        repaired = json.loads(raw)

        # Validate basic structure
        if "messages" not in repaired or not isinstance(repaired["messages"], list):
            return None
        if len(repaired["messages"]) < 3:  # at least system + user + assistant
            return None

        # Preserve task_id
        repaired["task_id"] = example.get("task_id", "unknown")
        if "source" not in repaired:
            repaired["source"] = "repaired"

        return repaired

    except (json.JSONDecodeError, anthropic.APIError, KeyError):
        return None


def score_example(client: anthropic.Anthropic, example: dict) -> dict | None:
    """Quick-score a single example using the same approach as llm_judge."""
    try:
        from llm_judge import build_judge_prompt
    except ImportError:
        return None

    prompt = build_judge_prompt(example)
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()

        import re
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)

        return json.loads(raw.strip())
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(min_score: int = 2, max_score: int = 3):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    if not SCORES_FILE.exists():
        print("No scores.json found. Run llm_judge.py first.")
        sys.exit(1)

    scores = json.loads(SCORES_FILE.read_text())
    client = anthropic.Anthropic(api_key=api_key)

    # Load all examples
    all_examples = {}
    for path in [TRAIN_FILE, VAL_FILE]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Build the same key llm_judge uses
                task_id = rec.get("task_id", "unknown")
                msgs = rec.get("messages", [])
                user_msgs = [m for m in msgs if m.get("role") == "user"]
                user_text = user_msgs[0]["content"][:80] if user_msgs else ""
                key = f"{task_id}::{user_text}"
                all_examples[key] = rec
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    # Find borderline examples
    borderline = []
    for key, score_data in scores.items():
        s = score_data.get("score", 0)
        if min_score <= s <= max_score and key in all_examples:
            borderline.append((key, all_examples[key], score_data))

    if not borderline:
        print(f"No examples scoring {min_score}-{max_score}. Nothing to repair.")
        return

    # Cap at MAX_BATCH for cost control
    if len(borderline) > MAX_BATCH:
        print(f"  {len(borderline)} borderline examples found, capping at {MAX_BATCH}")
        borderline = borderline[:MAX_BATCH]

    print(f"Repairing {len(borderline)} examples (score {min_score}-{max_score})...")

    stats = {"attempted": 0, "improved": 0, "unchanged": 0, "degraded": 0, "failed": 0}
    repairs = {}

    for key, example, score_data in borderline:
        stats["attempted"] += 1
        old_score = score_data.get("score", 0)
        task_id = example.get("task_id", "?")

        repaired = repair_example(client, example, score_data)
        if repaired is None:
            stats["failed"] += 1
            continue

        # Re-score the repair
        new_score_data = score_example(client, repaired)
        if new_score_data is None:
            stats["failed"] += 1
            continue

        new_score = new_score_data.get("score", 0)

        if new_score > old_score:
            stats["improved"] += 1
            repairs[key] = {"example": repaired, "score_data": new_score_data}
            print(f"  ✓ {task_id}: {old_score} → {new_score}")
        elif new_score < old_score:
            stats["degraded"] += 1
            print(f"  ✗ {task_id}: {old_score} → {new_score} (keeping original)")
        else:
            stats["unchanged"] += 1

    # Apply repairs: rewrite train/val files with repaired examples
    if repairs:
        for path in [TRAIN_FILE, VAL_FILE]:
            if not path.exists():
                continue
            lines = path.read_text().splitlines()
            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    task_id = rec.get("task_id", "unknown")
                    msgs = rec.get("messages", [])
                    user_msgs = [m for m in msgs if m.get("role") == "user"]
                    user_text = user_msgs[0]["content"][:80] if user_msgs else ""
                    key = f"{task_id}::{user_text}"

                    if key in repairs:
                        new_lines.append(json.dumps(repairs[key]["example"]))
                        # Update scores
                        scores[key] = repairs[key]["score_data"]
                    else:
                        new_lines.append(line)
                except (json.JSONDecodeError, KeyError, IndexError):
                    new_lines.append(line)

            path.write_text("\n".join(new_lines) + "\n")

        # Save updated scores
        SCORES_FILE.write_text(json.dumps(scores, indent=2))

    # Write report
    report = {**stats, "success_rate": round(stats["improved"] / max(stats["attempted"], 1) * 100, 1)}
    REPAIR_REPORT.write_text(json.dumps(report, indent=2))

    print(f"\n{'─'*50}")
    print(f"Repair complete:")
    print(f"  Attempted:  {stats['attempted']}")
    print(f"  Improved:   {stats['improved']}")
    print(f"  Unchanged:  {stats['unchanged']}")
    print(f"  Degraded:   {stats['degraded']} (kept originals)")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Success rate: {report['success_rate']}%")


def main():
    parser = argparse.ArgumentParser(description="Repair borderline training examples")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Repair borderline examples")
    run_p.add_argument("--min-score", type=int, default=2)
    run_p.add_argument("--max-score", type=int, default=3)

    sub.add_parser("report", help="Show last repair stats")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(min_score=args.min_score, max_score=args.max_score)
    elif args.command == "report":
        if REPAIR_REPORT.exists():
            print(json.dumps(json.loads(REPAIR_REPORT.read_text()), indent=2))
        else:
            print("No repair report found. Run: python example_repair.py run")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
