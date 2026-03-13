#!/usr/bin/env python3
"""
Remove invalid 'image' key from ~/.openclaw/openclaw.json.
OpenClaw doesn't support this key — image generation is handled automatically via OpenRouter.
"""
import json
from pathlib import Path

path = Path.home() / ".openclaw" / "openclaw.json"

if not path.exists():
    print(f"ERROR: {path} not found.")
    exit(1)

cfg = json.loads(path.read_text())

removed = False
if "image" in cfg.get("tools", {}):
    del cfg["tools"]["image"]
    removed = True

path.write_text(json.dumps(cfg, indent=2))

if removed:
    print("Removed invalid 'image' key from tools section.")
else:
    print("No 'image' key found — nothing to do.")

print("openclaw.json is clean.")
