#!/usr/bin/env python3
"""
Patch /root/.openclaw/openclaw.json to add image generation config.
Run this once after openclaw is installed on a new pod.
"""
import json
from pathlib import Path

path = Path("/root/.openclaw/openclaw.json")

if not path.exists():
    print(f"ERROR: {path} not found. Run startup.sh first.")
    exit(1)

cfg = json.loads(path.read_text())

cfg.setdefault("tools", {}).setdefault("image", {})["generation"] = {
    "enabled": True,
    "provider": "openrouter",
    "model": "black-forest-labs/flux-schnell"
}

path.write_text(json.dumps(cfg, indent=2))
print(f"Done. Image generation configured: black-forest-labs/flux-schnell via openrouter")
