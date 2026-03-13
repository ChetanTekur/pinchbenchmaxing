#!/usr/bin/env bash
# One-time pod setup: install OpenClaw + all deps needed for PinchBench
# Run this once after spinning up a new RunPod instance.
# Usage: bash setup_pod.sh

set -euo pipefail

echo "=== [1/4] Installing system deps ==="
apt-get update && apt-get install -y --no-install-recommends \
    jq \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

echo "=== [2/4] Installing Python deps ==="
pip install --no-cache-dir \
    pandas \
    openpyxl \
    pdfplumber \
    PyPDF2 \
    requests

echo "=== [3/4] Installing Node.js 22 + OpenClaw ==="
# Remove old Node if present
apt-get remove -y nodejs npm 2>/dev/null || true

# Install Node 22 LTS via NodeSource
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
node --version

# Install real OpenClaw via official installer
curl -fsSL https://openclaw.ai/install.sh | bash

echo "=== [4/4] Verifying OpenClaw ==="
export PATH="$HOME/.local/bin:$HOME/.openclaw/bin:/usr/local/bin:$PATH"
openclaw --version || echo "WARNING: openclaw not in PATH — check ~/.openclaw/bin or ~/.local/bin"

echo ""
echo "=== Setup complete ==="
echo "Next: set env vars and run startup.sh"
echo "  export OPENROUTER_API_KEY=..."
echo "  export BRAVE_API_KEY=..."
echo "  export OPENCLAW_GATEWAY_TOKEN=..."
echo "  bash /root/scripts/startup.sh"
