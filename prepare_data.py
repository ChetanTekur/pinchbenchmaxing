#!/usr/bin/env python3
"""
Data preparation for Qwen3-8B SFT fine-tuning.

Converts train.jsonl / val.jsonl to SFT-ready format:
  - Converts 'tool' role messages → 'user' role with <tool_result> wrapper
  - Keeps <tool_call> tags as plain text inside assistant messages
  - Drops the task_id field, keeps only 'messages'

Output: train_sft.jsonl, val_sft.jsonl (same directory)

Usage:
  python prepare_data.py
  python prepare_data.py --stats   # show token length distribution
"""

import json, argparse
from pathlib import Path

DATA_DIR   = Path("/workspace/synthbench/data")
TRAIN_IN   = DATA_DIR / "train.jsonl"
VAL_IN     = DATA_DIR / "val.jsonl"
TRAIN_OUT  = DATA_DIR / "train_sft.jsonl"
VAL_OUT    = DATA_DIR / "val_sft.jsonl"


def convert_messages(messages: list[dict]) -> list[dict]:
    """
    Convert a message list to SFT format:
      - 'tool' role  →  'user' role wrapped in <tool_result> tags
      - All other roles pass through unchanged
    Consecutive user messages (e.g. tool_result after tool_result) are merged.
    """
    converted = []
    for msg in messages:
        role    = msg["role"]
        content = msg["content"].strip()

        if role == "tool":
            # Wrap tool result so the model knows it's a tool response
            content = f"<tool_result>\n{content}\n</tool_result>"
            role    = "user"

        # Merge consecutive same-role messages (avoids invalid chat sequences)
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"] += "\n\n" + content
        else:
            converted.append({"role": role, "content": content})

    return converted


def process_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    """Process one JSONL file. Returns (total, skipped)."""
    total = skipped = 0

    with open(out_path, "w") as out_f:
        for line in in_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)

            messages = rec.get("messages", [])
            if not messages:
                skipped += 1
                continue

            converted = convert_messages(messages)

            # Sanity: must start with system, have at least one user + assistant
            roles = [m["role"] for m in converted]
            if roles[0] != "system" or "user" not in roles or "assistant" not in roles:
                skipped += 1
                continue

            out_f.write(json.dumps({"messages": converted}) + "\n")

    return total, skipped


def cmd_stats():
    """Print token-length distribution (rough char/4 estimate)."""
    from collections import Counter

    for path in [TRAIN_OUT, VAL_OUT]:
        if not path.exists():
            print(f"{path.name} not found — run prepare_data.py first")
            continue

        lengths = []
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                chars = sum(len(m["content"]) for m in rec["messages"])
                lengths.append(chars // 4)  # rough token estimate

        if not lengths:
            continue

        lengths.sort()
        n = len(lengths)
        print(f"\n{path.name}  ({n} examples)")
        print(f"  min:    {lengths[0]:,} tokens")
        print(f"  median: {lengths[n//2]:,} tokens")
        print(f"  p90:    {lengths[int(n*0.9)]:,} tokens")
        print(f"  p99:    {lengths[int(n*0.99)]:,} tokens")
        print(f"  max:    {lengths[-1]:,} tokens")

        # Bucket distribution
        buckets = Counter()
        for l in lengths:
            if   l <  512: buckets["<512"]    += 1
            elif l < 1024: buckets["512-1k"]  += 1
            elif l < 2048: buckets["1k-2k"]   += 1
            elif l < 4096: buckets["2k-4k"]   += 1
            else:          buckets["4k+"]      += 1

        for label in ["<512", "512-1k", "1k-2k", "2k-4k", "4k+"]:
            bar = "█" * (buckets[label] // 2)
            print(f"  {label:<10} {buckets[label]:>4}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true",
                        help="Show token length distribution of output files")
    args = parser.parse_args()

    if args.stats:
        cmd_stats()
    else:
        for in_path, out_path, label in [
            (TRAIN_IN, TRAIN_OUT, "train"),
            (VAL_IN,   VAL_OUT,   "val"),
        ]:
            if not in_path.exists():
                print(f"  {in_path} not found, skipping")
                continue
            total, skipped = process_file(in_path, out_path)
            kept = total - skipped
            print(f"  {label}: {total} in → {kept} out  ({skipped} skipped)  →  {out_path}")

        print("\nDone. Run with --stats to check token lengths.")
        print("Next: python finetune.py")
