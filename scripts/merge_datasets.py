#!/usr/bin/env python3
"""
Merge two datasets, deduplicate, and validate.

Combines old (v8 backup) and new data, keeping the best of both.
Deduplicates by task_id + user message similarity.
Validates the merged result.

Usage:
  python scripts/merge_datasets.py OLD_TRAIN NEW_TRAIN [--output OUTPUT_DIR]

Example:
  python scripts/merge_datasets.py /tmp/v8_backup/data/train.jsonl /workspace/synthbench/data/snapshots/pre-validate-fix-v11_20260323_045131/train.jsonl
"""

import json
import sys
import hashlib
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config


def load_jsonl(path):
    examples = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return examples


def example_key(ex):
    """Stable key for deduplication: task_id + hash of user message."""
    task_id = ex.get("task_id", "unknown")
    msgs = ex.get("messages", [])
    user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
    user_text = user_msgs[0] if user_msgs else ""
    h = hashlib.sha256(user_text.encode()).hexdigest()[:16]
    return f"{task_id}|{h}"


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/merge_datasets.py OLD_TRAIN NEW_TRAIN [--output DIR]")
        sys.exit(1)

    old_path = sys.argv[1]
    new_path = sys.argv[2]

    cfg = load_config()
    output_dir = cfg.data_dir
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        output_dir = Path(sys.argv[idx + 1])

    print(f"Old dataset: {old_path}")
    print(f"New dataset: {new_path}")
    print(f"Output:      {output_dir}")
    print()

    old = load_jsonl(old_path)
    new = load_jsonl(new_path)

    print(f"Old: {len(old)} examples")
    print(f"New: {len(new)} examples")

    # Merge: new takes priority over old for duplicates
    seen = {}
    for ex in old:
        k = example_key(ex)
        seen[k] = ("old", ex)

    for ex in new:
        k = example_key(ex)
        seen[k] = ("new", ex)  # new overwrites old

    merged = [ex for _, ex in seen.values()]
    print(f"Merged (after dedup): {len(merged)} examples")

    # Count per task
    by_task = defaultdict(list)
    for ex in merged:
        by_task[ex.get("task_id", "unknown")].append(ex)

    from_old = sum(1 for src, _ in seen.values() if src == "old")
    from_new = sum(1 for src, _ in seen.values() if src == "new")

    print(f"\nSources: {from_old} from old, {from_new} from new")
    print(f"\n{'Task':<40} {'count':>5}")
    print("-" * 48)
    from agents.base import TASK_IDS
    for task_id in TASK_IDS:
        count = len(by_task.get(task_id, []))
        marker = "  ⚠ LOW" if count < 40 else ("  ⚠ MISSING" if count == 0 else "")
        print(f"  {task_id:<38} {count:>5}{marker}")

    total = sum(len(v) for v in by_task.values())
    print(f"\n  TOTAL: {total}")

    # Split train/val
    import random
    random.seed(42)
    val_split = cfg._data.get("data", {}).get("val_split", 0.1)

    train_out = []
    val_out = []
    for task_id, exs in by_task.items():
        random.shuffle(exs)
        val_count = max(1, round(len(exs) * val_split))
        val_out.extend(exs[:val_count])
        train_out.extend(exs[val_count:])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"

    with open(train_file, "w") as f:
        for ex in train_out:
            f.write(json.dumps(ex) + "\n")
    with open(val_file, "w") as f:
        for ex in val_out:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWritten:")
    print(f"  Train: {len(train_out)} → {train_file}")
    print(f"  Val:   {len(val_out)} → {val_file}")
    print(f"\nNext: python -m datagen.validate_data")


if __name__ == "__main__":
    main()
