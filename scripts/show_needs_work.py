#!/usr/bin/env python3
"""Show tasks with NEEDS_WORK verdict from deep validation."""
import json
r = json.load(open("/workspace/synthbench/data/deep_validation_report.json"))
for t in r:
    if t.get("verdict") != "GOOD":
        print(f"{t['task_id']} — {t['verdict']}")
        sem = t.get("semantic", {})
        if sem.get("reasoning"):
            print(f"  Reason: {sem['reasoning'][:300]}")
        for issue in sem.get("issues", []):
            print(f"  - {issue}")
        if sem.get("recommendation"):
            print(f"  Rec: {sem['recommendation'][:200]}")
