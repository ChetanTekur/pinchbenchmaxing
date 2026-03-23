#!/usr/bin/env python3
"""Show summary of bad examples from validation report."""

import json
from collections import Counter
from pathlib import Path

bad = json.load(open("/workspace/synthbench/data/bad_examples_report.json"))
tasks = Counter(e["task_id"] for e in bad)

print(f"\n{len(bad)} bad examples across {len(tasks)} tasks\n")

for task, count in tasks.most_common(5):
    print(f"=== {task} ({count} bad) ===")
    for ex in [e for e in bad if e["task_id"] == task][:3]:
        print(f"  Issues: {[i['check'] for i in ex['issues']]}")
        print(f"  Tools:  {[(c['name'], c['args']) for c in ex['tool_calls']]}")
        print()
