#!/usr/bin/env python3
"""
Convert train.jsonl / val.jsonl to SFT-ready format.

- Converts 'tool' role messages → 'user' role with <tool_result> wrapper
- Drops task_id, keeps only 'messages'
- Merges consecutive same-role messages

Usage:
  python stages/prepare.py
  python stages/prepare.py --config /path/to/config.yaml
  python stages/prepare.py --stats
"""

import argparse
import json
from pathlib import Path

from utils.config import load_config


def convert_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for msg in messages:
        role    = msg["role"]
        content = msg["content"].strip()
        if role == "tool":
            content = f"<tool_result>\n{content}\n</tool_result>"
            role    = "user"
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"] += "\n\n" + content
        else:
            converted.append({"role": role, "content": content})
    return converted


def process_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    total = skipped = 0
    with open(out_path, "w") as out_f:
        for line in in_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            total += 1
            rec      = json.loads(line)
            messages = rec.get("messages", [])
            if not messages:
                skipped += 1
                continue
            converted = convert_messages(messages)
            roles = [m["role"] for m in converted]
            if roles[0] != "system" or "user" not in roles or "assistant" not in roles:
                skipped += 1
                continue
            out_f.write(json.dumps({"messages": converted}) + "\n")
    return total, skipped


def cmd_stats(cfg):
    from collections import Counter
    for path in [cfg.train_sft_file, cfg.val_sft_file]:
        if not path.exists():
            print(f"{path.name} not found — run prepare.py first")
            continue
        lengths = []
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                chars = sum(len(m["content"]) for m in rec["messages"])
                lengths.append(chars // 4)
        if not lengths:
            continue
        lengths.sort()
        n = len(lengths)
        print(f"\n{path.name}  ({n} examples)")
        print(f"  min:    {lengths[0]:,}")
        print(f"  median: {lengths[n//2]:,}")
        print(f"  p90:    {lengths[int(n*0.9)]:,}")
        print(f"  max:    {lengths[-1]:,}")
        buckets = Counter()
        for l in lengths:
            if   l <  512: buckets["<512"]   += 1
            elif l < 1024: buckets["512-1k"] += 1
            elif l < 2048: buckets["1k-2k"]  += 1
            elif l < 4096: buckets["2k-4k"]  += 1
            else:          buckets["4k+"]    += 1
        for label in ["<512", "512-1k", "1k-2k", "2k-4k", "4k+"]:
            bar = "█" * (buckets[label] // 2)
            print(f"  {label:<10} {buckets[label]:>4}  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--stats",  action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.stats:
        cmd_stats(cfg)
        return

    for in_path, out_path, label in [
        (cfg.train_file, cfg.train_sft_file, "train"),
        (cfg.val_file,   cfg.val_sft_file,   "val"),
    ]:
        if not in_path.exists():
            print(f"  {in_path} not found, skipping")
            continue
        total, skipped = process_file(in_path, out_path)
        kept = total - skipped
        print(f"  {label}: {total} in → {kept} out ({skipped} skipped) → {out_path}")

    print("\nDone. Next: python stages/finetune.py")


if __name__ == "__main__":
    main()
