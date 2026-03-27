#!/usr/bin/env python3
"""
Trim a task to N examples (keeps first N, removes rest).

Usage:
  python scripts/trim_task.py task_06_events 45
  python scripts/trim_task.py task_13_image_gen 45
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config

cfg = load_config()

if len(sys.argv) < 3:
    print("Usage: python scripts/trim_task.py <task_id> <max_count>")
    sys.exit(1)

task_id = sys.argv[1]
max_count = int(sys.argv[2])

for path in [cfg.train_file, cfg.val_file]:
    if not path.exists():
        continue
    lines = path.read_text().splitlines()
    kept = []
    task_count = 0
    removed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
            if ex.get("task_id") == task_id:
                task_count += 1
                if task_count <= max_count:
                    kept.append(line)
                else:
                    removed += 1
            else:
                kept.append(line)
        except json.JSONDecodeError:
            kept.append(line)
    path.write_text("\n".join(kept) + "\n" if kept else "")
    if removed:
        print(f"  {path.name}: {task_id} trimmed from {task_count} to {max_count} (removed {removed})")
    else:
        print(f"  {path.name}: {task_id} has {task_count} (already ≤{max_count})")
