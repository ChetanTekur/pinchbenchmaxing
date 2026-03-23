#!/usr/bin/env python3
"""Validate the v8 backup data without modifying current data."""

import shutil
import sys
sys.path.insert(0, ".")

from utils.config import load_config
from datagen.validate_data import run_validation

cfg = load_config()
train = str(cfg.train_file)
backup = "/tmp/v8_backup/data/train.jsonl"

# Save current, swap in old, validate, restore
shutil.copy(train, "/tmp/current_train.jsonl")
shutil.copy(backup, train)

try:
    r = run_validation(fix=False)
    print(f"\nv8 data: {r['total_examples']} examples, {r['clean']} clean, {r['critical_high']} critical/high")
finally:
    shutil.copy("/tmp/current_train.jsonl", train)
    print("(restored current train.jsonl)")
