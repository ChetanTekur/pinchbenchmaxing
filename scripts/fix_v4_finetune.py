#!/usr/bin/env python3
"""Fix the v4 era finetune.py type casting issue."""
import re
from pathlib import Path

f = Path("/workspace/synthbench/pbm_v4/stages/finetune.py")
code = f.read_text()

replacements = {
    '=t["epochs"]': '=int(t["epochs"])',
    '=t["batch_size"]': '=int(t["batch_size"])',
    '=t["grad_accum"]': '=int(t["grad_accum"])',
    '=t["learning_rate"]': '=float(t["learning_rate"])',
    '=t["warmup_ratio"]': '=float(t["warmup_ratio"])',
    '=t["weight_decay"]': '=float(t["weight_decay"])',
    '=t["max_grad_norm"]': '=float(t["max_grad_norm"])',
}

for old, new in replacements.items():
    code = code.replace(old, new)

f.write_text(code)
print("Fixed type casting in v4 finetune.py")
