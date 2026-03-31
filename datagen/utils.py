"""
Shared utilities for data generation scripts.

Extracted from topup.py to avoid circular imports and reduce coupling.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from utils.config import load_config
from utils.prompts import OPENCLAW_SYSTEM

_cfg = load_config()
TRAIN_FILE = _cfg.train_file
VAL_FILE = _cfg.val_file


# ── Variation types for data generation ──────────────────────────────────────

VARIATION_CONFIGS = [
    {
        "id": "happy_formal",
        "user_style": "formal and precise",
        "scenario": "Straightforward task. User provides all needed information. Agent completes in the most direct way. No errors.",
        "has_error": False,
    },
    {
        "id": "happy_casual",
        "user_style": "casual and conversational",
        "scenario": "Straightforward task. User is casual. Agent adapts and completes cleanly.",
        "has_error": False,
    },
    {
        "id": "vague_input",
        "user_style": "vague, leaving out one or two details",
        "scenario": "User omits a detail. Agent makes a smart assumption, mentions it briefly, then completes the task.",
        "has_error": False,
    },
    {
        "id": "error_recovery",
        "user_style": "normal",
        "scenario": "One tool call fails with a realistic error. Agent reads the error, tries an alternative, successfully completes the task.",
        "has_error": True,
    },
    {
        "id": "multi_tool_chain",
        "user_style": "detailed with multiple requirements",
        "scenario": "Task requires chaining 3+ tools. Agent plans the sequence, uses output from each step in the next.",
        "has_error": False,
    },
    {
        "id": "terse",
        "user_style": "extremely brief, one sentence",
        "scenario": "User gives a minimal request. Agent infers intent correctly and completes the task.",
        "has_error": False,
    },
    {
        "id": "detailed_user",
        "user_style": "verbose with very specific instructions",
        "scenario": "User over-specifies exactly how they want it done. Agent follows every specification precisely.",
        "has_error": False,
    },
    {
        "id": "one_clarification",
        "user_style": "ambiguous on exactly one critical detail",
        "scenario": "User's request has one genuinely ambiguous detail. Agent asks ONE targeted question, gets an answer, then completes the task.",
        "has_error": False,
    },
    {
        "id": "self_correction",
        "user_style": "normal",
        "scenario": "Agent makes a first attempt, inspects its own output, notices it is incomplete, corrects it, delivers final correct output.",
        "has_error": False,
    },
]


# ── Example parsing ──────────────────────────────────────────────────────────

def parse_example(raw: dict, task_id: str) -> dict | None:
    """Parse a raw Claude-generated example into training format."""
    try:
        user_msg = raw.get("user_message") or raw.get("user") or raw.get("prompt")
        turns    = raw.get("turns") or raw.get("conversation") or []
        if not user_msg or not turns:
            return None

        messages = [{"role": "system", "content": OPENCLAW_SYSTEM}]
        messages.append({"role": "user", "content": str(user_msg)})
        for turn in turns:
            role = turn.get("role", "")
            if role == "assistant":
                messages.append({"role": "assistant", "content": str(turn["content"])})
            elif role in ("tool_result", "tool", "function"):
                messages.append({"role": "tool", "content": str(turn["content"])})

        return {"task_id": task_id, "messages": messages} if len(messages) >= 3 else None
    except Exception:
        return None


# ── JSON extraction ──────────────────────────────────────────────────────────

def extract_json_array(text: str):
    """Extract and merge ALL JSON arrays from a Claude response.

    Claude sometimes returns multiple separate JSON arrays (one per example)
    when examples are long. This merges all of them into a single list.
    """
    text = text.strip()
    merged = []

    # Find all ```json ... ``` or ``` ... ``` code blocks
    blocks = re.findall(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
    for block in blocks:
        try:
            result = json.loads(block.strip())
            if isinstance(result, list):
                merged.extend(result)
            elif isinstance(result, dict):
                merged.append(result)
        except json.JSONDecodeError:
            pass

    if merged:
        return merged

    # No code blocks -- try finding all top-level JSON arrays in the text
    for m in re.finditer(r'\[[\s\S]*?\](?=\s*(?:\[|$|```))', text):
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                merged.extend(result)
        except json.JSONDecodeError:
            pass

    if merged:
        return merged

    # Last resort -- parse the whole thing
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    return None


# ── Dataset counting ─────────────────────────────────────────────────────────

def count_existing() -> dict[str, int]:
    """Count examples per task in train + val files."""
    counts = defaultdict(int)
    for path in [TRAIN_FILE, VAL_FILE]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                counts[rec.get("task_id", "unknown")] += 1
            except json.JSONDecodeError:
                counts["__corrupt__"] += 1
    return counts
