#!/usr/bin/env python3
"""Show error details from the last batch."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import anthropic
from utils.config import load_config

cfg = load_config()
batch_file = cfg.data_dir / "dynamic_batch_id.txt"

if not batch_file.exists():
    print("No batch ID found")
    sys.exit(1)

batch_id = batch_file.read_text().strip()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

print(f"Batch: {batch_id}\n")

for i, result in enumerate(client.messages.batches.results(batch_id)):
    print(f"Request {i+1}: {result.custom_id}")
    print(f"  Type: {result.result.type}")
    if result.result.type == "errored":
        err = result.result.error
        print(f"  Error: {err}")
        # Try different attribute patterns
        for attr in ["message", "type", "error", "detail"]:
            if hasattr(err, attr):
                print(f"  .{attr}: {getattr(err, attr)}")
    elif result.result.type == "succeeded":
        msg = result.result.message
        print(f"  Tokens: in={msg.usage.input_tokens} out={msg.usage.output_tokens}")
    print()
    if i >= 4:
        print(f"... ({i+1} of {17} shown)")
        break
