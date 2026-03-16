#!/usr/bin/env bash
# Startup sequence for PinchBench / OpenClaw benchmarking
#
# Run this every time you start a new session before running benchmarks.
# Kills stale processes, generates OpenClaw config from env vars, starts
# Ollama, starts the OpenClaw gateway, and verifies both are healthy.
#
# Required env vars:
#   OPENROUTER_API_KEY   - OpenRouter key (for LLM judge: claude-opus-4.5)
#   BRAVE_API_KEY        - Brave search key (for web search tasks)
#   OPENCLAW_GATEWAY_TOKEN - Gateway auth token (any random string)
#
# Optional:
#   ANTHROPIC_API_KEY    - Anthropic key (for data generation scripts only)
#   PBM_WORKSPACE  - Root workspace dir (default: ./workspace)
#
# Usage: bash scripts/startup.sh

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.openclaw/bin:/usr/local/bin:$PATH"

WORKSPACE="${PBM_WORKSPACE:-./workspace}"
REPO_DIR="/root/pbm"
REPO_URL="https://github.com/ChetanTekur/pinchbenchmaxing.git"
ENV_FILE="/workspace/synthbench/set_env.sh"

# ── 0a. Load persistent env vars (API keys) ──────────────────────────────────
if [ -f "$ENV_FILE" ]; then
    echo "=== [0/5] Loading env vars from $ENV_FILE ==="
    set +u  # allow unset vars during source
    source "$ENV_FILE"
    set -u
    echo "  ANTHROPIC_API_KEY:      ${ANTHROPIC_API_KEY:+set}"
    echo "  OPENROUTER_API_KEY:     ${OPENROUTER_API_KEY:+set}"
    echo "  BRAVE_API_KEY:          ${BRAVE_API_KEY:+set}"
    echo "  OPENCLAW_GATEWAY_TOKEN: ${OPENCLAW_GATEWAY_TOKEN:+set}"
else
    echo "=== [0/5] No env file at $ENV_FILE — using existing env vars ==="
    echo "  To persist API keys across restarts:"
    echo "    cp /root/scripts/set_env.sh $ENV_FILE"
    echo "    vim $ENV_FILE  # fill in keys"
fi

# ── 0b. Clone or update project repo ─────────────────────────────────────────
echo ""
if [ -d "$REPO_DIR/.git" ]; then
    echo "=== Updating repo: git pull ==="
    git -C "$REPO_DIR" pull --ff-only 2>&1 | sed 's/^/  /'
else
    echo "=== Cloning repo to $REPO_DIR ==="
    git clone "$REPO_URL" "$REPO_DIR" 2>&1 | sed 's/^/  /'
fi
export PYTHONPATH="$REPO_DIR"
echo "  PYTHONPATH=$PYTHONPATH"

# ── 1. Kill stale processes ───────────────────────────────────────────────────
echo "=== [1/5] Killing stale openclaw / ollama processes ==="
PIDS=$(ps aux | grep -E "openclaw|ollama" | grep -v grep | awk '{print $2}' || true)
if [ -n "$PIDS" ]; then
    echo "  Killing PIDs: $PIDS"
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    sleep 2
else
    echo "  No stale processes found."
fi

# ── 2. Generate OpenClaw config from env vars ─────────────────────────────────
echo ""
echo "=== [2/5] Configuring OpenClaw ==="
mkdir -p "$HOME/.openclaw"

# Template is in the cloned repo
TEMPLATE="$REPO_DIR/openclaw_template.json"
DEST="$HOME/.openclaw/openclaw.json"

if [ ! -f "$TEMPLATE" ]; then
    echo "  ERROR: openclaw_template.json not found at $TEMPLATE"
    exit 1
fi

OPENROUTER_KEY="${OPENROUTER_API_KEY:-}"
BRAVE_KEY="${BRAVE_API_KEY:-}"

if [ -z "$OPENROUTER_KEY" ]; then
    echo "  WARNING: OPENROUTER_API_KEY not set — LLM judge will fail"
    OPENROUTER_KEY="NOT_SET"
fi

if [ -z "$BRAVE_KEY" ]; then
    echo "  WARNING: BRAVE_API_KEY not set — web search tasks will fail"
    BRAVE_KEY="NOT_SET"
fi

GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-$(openssl rand -hex 24)}"
MODEL_NAME=$(cd "$REPO_DIR" && python3 -c "from utils.config import load_config; print(load_config().ollama_model_name)" 2>/dev/null || echo "qwen35-9b-clawd")

sed \
    -e "s|__OPENROUTER_API_KEY__|$OPENROUTER_KEY|g" \
    -e "s|__BRAVE_API_KEY__|$BRAVE_KEY|g" \
    -e "s|__OPENCLAW_GATEWAY_TOKEN__|$GATEWAY_TOKEN|g" \
    -e "s|__MODEL_NAME__|$MODEL_NAME|g" \
    "$TEMPLATE" > "$DEST"
echo "  Default model: ollama/$MODEL_NAME:latest"

echo "  OpenClaw config written to $DEST"

# Set up judge agent auth
mkdir -p "$HOME/.openclaw/agents/bench-judge-anthropic-claude-opus-4-5/agent"
ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-}"

if [ -n "$ANTHROPIC_KEY" ]; then
    cat > "$HOME/.openclaw/agents/bench-judge-anthropic-claude-opus-4-5/agent/auth-profiles.json" << AUTHEOF
{
  "version": 1,
  "profiles": {
    "anthropic:default": {
      "type": "api_key",
      "provider": "anthropic",
      "key": "$ANTHROPIC_KEY"
    }
  }
}
AUTHEOF
    echo "  Judge auth-profiles.json written"
    export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
else
    echo "  WARNING: ANTHROPIC_API_KEY not set — data generation scripts will fail"
fi

# ── 3. Start Ollama ───────────────────────────────────────────────────────────
echo ""
echo "=== [3/5] Starting Ollama ==="
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "  Ollama PID: $OLLAMA_PID"

echo "  Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:11434/ > /dev/null 2>&1; then
        echo "  Ollama is up (after ${i}s)."
        break
    fi
    sleep 1
    if [ "$i" -eq 30 ]; then
        echo "  ERROR: Ollama did not start in 30s. Check /tmp/ollama.log"
        exit 1
    fi
done

# ── 4. Start OpenClaw gateway ─────────────────────────────────────────────────
echo ""
echo "=== [4/5] Starting OpenClaw gateway ==="
openclaw gateway --port 18789 > /tmp/openclaw-gateway.log 2>&1 &
OPENCLAW_PID=$!
echo "  OpenClaw gateway PID: $OPENCLAW_PID"

echo "  Waiting for OpenClaw gateway to be ready..."
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:18789/health > /dev/null 2>&1; then
        echo "  OpenClaw gateway is up (after ${i}s)."
        break
    fi
    sleep 1
    if [ "$i" -eq 60 ]; then
        echo "  WARNING: OpenClaw gateway health check timed out — check /tmp/openclaw-gateway.log"
    fi
done

# ── 5. Auto-register latest fine-tuned model ─────────────────────────────────
# Reads current_ollama_model from loop_state.json and re-registers the GGUF.
# Ollama's model registry (~/.ollama) is ephemeral; the GGUF is on the network volume.
echo ""
echo "=== [5/6] Auto-registering fine-tuned model ==="
STATE_FILE="$WORKSPACE/data/loop_state.json"
if [ -f "$STATE_FILE" ]; then
    CURRENT_MODEL=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('current_ollama_model',''))" 2>/dev/null || true)
    if [ -n "$CURRENT_MODEL" ]; then
        echo "  Found model in loop state: $CURRENT_MODEL"
        OLLAMA_MODEL="$CURRENT_MODEL" bash "$REPO_DIR/scripts/fix_modelfile.sh" \
            && echo "  [OK] $CURRENT_MODEL registered" \
            || echo "  [WARN] Registration failed — GGUF may not exist yet (first run?)"
    else
        echo "  No model in loop state yet — skipping (normal on first run)"
    fi
else
    echo "  No loop_state.json found — skipping (normal on first run)"
fi

# ── 6. Health summary ─────────────────────────────────────────────────────────
echo ""
echo "=== [6/6] Health summary ==="

if curl -sf http://127.0.0.1:11434/ > /dev/null 2>&1; then
    echo "  [OK] Ollama        http://127.0.0.1:11434"
else
    echo "  [FAIL] Ollama not responding — check /tmp/ollama.log"
fi

echo "  Ollama models available:"
ollama list 2>/dev/null | sed 's/^/    /' || echo "    (none or ollama not ready)"

if curl -sf http://127.0.0.1:18789/health > /dev/null 2>&1; then
    echo "  [OK] OpenClaw gw   http://127.0.0.1:18789"
else
    echo "  [WARN] OpenClaw gateway not yet responding"
fi

echo ""
echo "=== Startup complete ==="
echo "  Logs: /tmp/ollama.log  |  /tmp/openclaw-gateway.log"
echo ""
echo "  Next steps:"
echo "    cd /root/pbm"
echo "    PYTHONPATH=. python3 loop.py run --model <ollama-model-name>"
