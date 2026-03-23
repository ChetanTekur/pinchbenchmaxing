#!/usr/bin/env python3
"""
Remove all training data for specific tasks.

Use when task definitions were wrong and all data needs regeneration.

Usage:
  python scripts/remove_task_data.py task_09_files task_11_config_update task_18_market_research
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config

cfg = load_config()

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/remove_task_data.py task_id1 task_id2 ...")
        sys.exit(1)

    tasks_to_remove = set(sys.argv[1:])
    print(f"Removing data for: {tasks_to_remove}")

    for path in [cfg.train_file, cfg.val_file]:
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        before = len(lines)
        kept = []
        removed_counts = {}
        for line in lines:
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                tid = ex.get("task_id", "")
                if tid in tasks_to_remove:
                    removed_counts[tid] = removed_counts.get(tid, 0) + 1
                else:
                    kept.append(line)
            except json.JSONDecodeError:
                kept.append(line)

        path.write_text("\n".join(kept) + "\n" if kept else "")
        after = len(kept)
        print(f"  {path.name}: {before} → {after} (removed {before - after})")
        for tid, count in sorted(removed_counts.items()):
            print(f"    {tid}: -{count}")

    print("\nDone. Run inspect_data stats to verify.")


if __name__ == "__main__":
    main()
