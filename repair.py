#!/usr/bin/env python3
"""
Repair script — re-parses raw batch responses with more robust extraction.
Run this after generate.py collect reports parse failures.

Usage:
  python repair.py          # dry run: shows recovery stats
  python repair.py --apply  # writes recovered examples to train/val JSONL
"""

import json, re, random, argparse
from pathlib import Path

DATA_DIR   = Path("/workspace/data")
RAW_DIR    = DATA_DIR / "raw"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE   = DATA_DIR / "val.jsonl"
VAL_PER_TASK = 2

# Paste OPENCLAW_SYSTEM here (same as generate.py) so we can rebuild messages
OPENCLAW_SYSTEM = """\
You are Clawd, an autonomous AI agent powered by OpenClaw. You help users \
accomplish real-world tasks by using tools. Be direct and competent — \
start with action, not explanation. Get things done.

## Available Tools

read_file(path: str) -> str
  Read the contents of a file.

write_file(path: str, content: str) -> dict
  Write content to a file (creates parent dirs automatically).
  Returns: {"status": "success", "path": "..."}

create_directory(path: str) -> dict
  Create a directory and any needed parents.

list_files(directory: str = ".") -> list
  List files in a directory.

run_bash(command: str) -> dict
  Execute a shell command.
  Returns: {"stdout": "...", "stderr": "...", "exit_code": 0}

run_python(code: str) -> dict
  Execute Python code.
  Returns: {"output": "...", "error": null}

web_search(query: str, num_results: int = 5) -> list
  Search the web. Returns [{title, url, snippet}, ...]

fetch_url(url: str) -> str
  Fetch the text content of a URL.

create_calendar_event(title: str, date: str, time: str,
                      attendees: list = [], description: str = "") -> dict
  Create a calendar event. Date: YYYY-MM-DD, Time: HH:MM.

draft_email(to: str, subject: str, body: str, cc: str = "") -> dict
  Draft an email.

search_emails(query: str, folder: str = "inbox") -> list
  Search emails.

read_email(email_id: str) -> dict
  Read a full email by ID.

generate_image(prompt: str, filename: str) -> dict
  Generate an image and save to workspace.

read_memory(key: str = None) -> str
  Read from persistent memory.

write_memory(key: str, value: str) -> dict
  Write a key-value pair to persistent memory.

search_skills(query: str) -> list
  Search ClawHub for installable skills.

install_skill(name: str) -> dict
  Install a skill from ClawHub.

## Format

Use tool calls like this:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

## Rules
- Working directory for file tasks: /workspace/tasks/
- When a tool fails, adapt — try an alternative approach
- Confirm task completion with a brief summary at the end
"""


def extract_json_array(text: str) -> list | None:
    """
    Try multiple strategies to extract a JSON array from raw LLM output.
    Returns the parsed list, or None if all strategies fail.
    """
    text = text.strip()

    # Strategy 1: strip ```json ... ``` or ``` ... ``` fences (any variant)
    fence_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 2: find the outermost [...] array in the text
    # Handles preamble/postamble text around the JSON
    bracket_match = re.search(r'(\[[\s\S]*\])', text)
    if bracket_match:
        candidate = bracket_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 3: try the whole text as-is
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]   # single object instead of array
    except json.JSONDecodeError:
        pass

    # Strategy 4: aggressive — find the first '[' and last ']' and try that span
    start = text.find('[')
    end   = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 5: try to parse individual {...} objects between the array brackets
    # Useful when the array itself is valid but some elements have bad escaping
    if start != -1 and end != -1:
        inner = text[start + 1:end]
        objects = []
        depth = 0
        obj_start = None
        for i, ch in enumerate(inner):
            if ch == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    snippet = inner[obj_start:i + 1]
                    try:
                        objects.append(json.loads(snippet))
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
        if objects:
            return objects

    return None


def parse_example(raw: dict, task_id: str) -> dict | None:
    """Convert a raw generated example dict into a training record."""
    try:
        user_msg = raw.get("user_message") or raw.get("user") or raw.get("prompt")
        turns    = raw.get("turns") or raw.get("conversation") or raw.get("messages", [])
        if not user_msg or not turns:
            return None

        messages = [{"role": "system", "content": OPENCLAW_SYSTEM}]
        messages.append({"role": "user", "content": str(user_msg)})

        for turn in turns:
            role    = turn.get("role", "")
            content = turn.get("content", "")
            if role == "assistant":
                messages.append({"role": "assistant", "content": str(content)})
            elif role in ("tool_result", "tool", "function"):
                messages.append({"role": "tool", "content": str(content)})

        # Must have at least one assistant turn beyond the system/user opening
        if len(messages) < 3:
            return None

        return {"task_id": task_id, "messages": messages}
    except Exception:
        return None


def run(apply: bool):
    raw_files = sorted(RAW_DIR.glob("*.json"))
    print(f"Found {len(raw_files)} raw files in {RAW_DIR}\n")

    total_attempted  = 0
    total_recovered  = 0
    total_failed     = 0
    failed_files     = []

    all_examples: dict[str, list] = {}

    for raw_file in raw_files:
        task_id = raw_file.stem.split("__")[0]
        text    = raw_file.read_text(encoding="utf-8", errors="replace")

        total_attempted += 1
        examples = extract_json_array(text)

        if examples is None:
            total_failed += 1
            failed_files.append(raw_file.name)
            continue

        recovered = 0
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            parsed = parse_example(ex, task_id)
            if parsed:
                if task_id not in all_examples:
                    all_examples[task_id] = []
                all_examples[task_id].append(parsed)
                recovered += 1

        if recovered > 0:
            total_recovered += 1
        else:
            total_failed += 1
            failed_files.append(raw_file.name)

    # Count total training records
    train_records = []
    val_records   = []
    for task_id, examples in all_examples.items():
        random.shuffle(examples)
        val_cut = min(VAL_PER_TASK, len(examples))
        val_records.extend(examples[:val_cut])
        train_records.extend(examples[val_cut:])

    print(f"{'─'*50}")
    print(f"Files parsed successfully:  {total_recovered} / {total_attempted}")
    print(f"Files still failing:        {total_failed}")
    print(f"Train examples recovered:   {len(train_records)}")
    print(f"Val examples recovered:     {len(val_records)}")
    print(f"Total:                      {len(train_records) + len(val_records)}")

    if failed_files:
        print(f"\nStill-failing files ({len(failed_files)}):")
        for f in failed_files[:10]:
            print(f"  {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    if apply:
        with open(TRAIN_FILE, "w") as f:
            for rec in train_records:
                f.write(json.dumps(rec) + "\n")
        with open(VAL_FILE, "w") as f:
            for rec in val_records:
                f.write(json.dumps(rec) + "\n")
        print(f"\n✓ Written to {TRAIN_FILE} and {VAL_FILE}")
    else:
        print(f"\nDry run — use --apply to overwrite train/val files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Write recovered examples to train/val JSONL")
    args = parser.parse_args()
    run(apply=args.apply)
