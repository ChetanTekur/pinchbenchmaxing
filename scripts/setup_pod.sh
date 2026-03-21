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

# Install OpenClaw via npm (more reliable than curl pipe)
npm install -g openclaw@latest

echo "=== [4/5] Verifying OpenClaw ==="
export PATH="$HOME/.local/bin:$HOME/.openclaw/bin:/usr/local/bin:$PATH"
openclaw --version || echo "WARNING: openclaw not in PATH — check ~/.openclaw/bin or ~/.local/bin"

echo "=== [5/5] Patching PinchBench lib_agent.py for Ollama support ==="
WORKSPACE="${PBM_WORKSPACE:-/workspace/synthbench}"
LIB_AGENT="$WORKSPACE/skill/scripts/lib_agent.py"
if [ -f "$LIB_AGENT" ]; then
    if ! grep -q 'ollama' "$LIB_AGENT"; then
        echo "  Patching validate_openrouter_model to skip ollama models..."
        sed -i '/# Skip validation for non-OpenRouter models/i\
    # Skip validation for Ollama (local) models\
    if bare_model_id.startswith("ollama/"):\
        logger.info("Skipping model validation for Ollama model: %s", model_id)\
        return True\
' "$LIB_AGENT"
        echo "  Done."
    else
        echo "  [OK] ollama patch already applied"
    fi
else
    echo "  WARNING: $LIB_AGENT not found — clone pinchbench/skill first"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Task dependency coverage:"
echo "  [OK] jq              — task_02 stock research"
echo "  [OK] pdfplumber      — task_20, task_21 PDF reading"
echo "  [OK] pandas+openpyxl — task_19 spreadsheet"
echo "  [OK] image gen       — task_13 (via OpenRouter flux-schnell, configured in openclaw.json)"
echo "  [OK] web search      — task_02, task_06, task_18 (via Brave)"
echo ""
echo "Next: set env vars and run startup.sh"
echo "  export OPENROUTER_API_KEY=..."
echo "  export BRAVE_API_KEY=..."
echo "  export OPENCLAW_GATEWAY_TOKEN=\$(openssl rand -hex 24)"
echo "  bash scripts/startup.sh"
