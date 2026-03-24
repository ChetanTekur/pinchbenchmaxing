#!/usr/bin/env python3
"""
Cap training data to N examples per task.

Keeps the first N examples per task (by order in file).
Creates a backup before modifying.

Usage:
  python scripts/cap_data.py 50          # cap at 50 per task
  python scripts/cap_data.py 40          # cap at 40 per task
"""

import json
import shutil
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config

cfg = load_config()

if len(sys.argv) < 2:
    print("Usage: python scripts/cap_data.py <max_per_task>")
    sys.exit(1)

cap = int(sys.argv[1])
print(f"Capping to {cap} examples per task\n")

for path in [cfg.train_file, cfg.val_file]:
    if not path.exists():
        continue

    # Backup
    backup = path.with_suffix(f".pre_cap_{cap}.bak")
    shutil.copy2(str(path), str(backup))

    by_task = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
            by_task[ex.get("task_id", "?")].append(line)
        except json.JSONDecodeError:
            continue

    kept = []
    removed = 0
    for task_id in sorted(by_task.keys()):
        examples = by_task[task_id]
        if len(examples) > cap:
            removed += len(examples) - cap
            kept.extend(examples[:cap])
            print(f"  {task_id:<40} {len(examples):>4} → {cap} (removed {len(examples) - cap})")
        else:
            kept.extend(examples)
            print(f"  {task_id:<40} {len(examples):>4} (ok)")

    path.write_text("\n".join(kept) + "\n" if kept else "")
    print(f"\n  {path.name}: {sum(len(v) for v in by_task.values())} → {len(kept)} (removed {removed})")
    print(f"  Backup: {backup}\n")
