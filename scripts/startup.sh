#!/usr/bin/env bash
# RunPod startup sequence for PinchBench / OpenClaw benchmarking
#
# Run this every time you start or restart a RunPod pod before running benchmarks.
# It kills stale processes, generates OpenClaw config from env vars, starts Ollama,
# starts the OpenClaw gateway, and verifies both are healthy before exiting.
#
# Required env vars (set in RunPod UI):
#   OPENROUTER_API_KEY   - OpenRouter key (for LLM judge: claude-opus-4.5)
#   ANTHROPIC_API_KEY    - Anthropic key (for data generation scripts)
#   BRAVE_API_KEY        - Brave search key (for web search tasks)
#
# Usage: bash startup.sh

set -euo pipefail

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

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
mkdir -p /root/.openclaw

TEMPLATE="/root/openclaw_template.json"
DEST="/root/.openclaw/openclaw.json"

if [ ! -f "$TEMPLATE" ]; then
    echo "  ERROR: openclaw_template.json not found at $TEMPLATE"
    exit 1
fi

# Substitute placeholders with real keys
OPENROUTER_KEY="${OPENROUTER_API_KEY:-}"
BRAVE_KEY="${BRAVE_API_KEY:-}"
if [ -z "$BRAVE_KEY" ] && [ -f /workspace/synthbench/brave_key ]; then
    BRAVE_KEY=$(cat /workspace/synthbench/brave_key)
fi

if [ -z "$OPENROUTER_KEY" ]; then
    # Fall back to checking /workspace
    if [ -f /workspace/synthbench/openrouter_key ]; then
        OPENROUTER_KEY=$(cat /workspace/synthbench/openrouter_key)
    else
        echo "  WARNING: OPENROUTER_API_KEY not set — LLM judge will fail"
        OPENROUTER_KEY="NOT_SET"
    fi
fi

GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-$(openssl rand -hex 24)}"

sed \
    -e "s|__OPENROUTER_API_KEY__|$OPENROUTER_KEY|g" \
    -e "s|__BRAVE_API_KEY__|$BRAVE_KEY|g" \
    -e "s|__OPENCLAW_GATEWAY_TOKEN__|$GATEWAY_TOKEN|g" \
    "$TEMPLATE" > "$DEST"

echo "  OpenClaw config written to $DEST"

# Set up judge agent auth
mkdir -p /root/.openclaw/agents/bench-judge-anthropic-claude-opus-4-5/agent
ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-}"
if [ -z "$ANTHROPIC_KEY" ] && [ -f /workspace/synthbench/anthropic_key ]; then
    ANTHROPIC_KEY=$(cat /workspace/synthbench/anthropic_key)
fi

if [ -n "$ANTHROPIC_KEY" ]; then
    cat > /root/.openclaw/agents/bench-judge-anthropic-claude-opus-4-5/agent/auth-profiles.json << AUTHEOF
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
    # Also export for scripts
    export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
    echo "export ANTHROPIC_API_KEY=$ANTHROPIC_KEY" >> ~/.bashrc
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
for i in $(seq 1 20); do
    if curl -sf http://127.0.0.1:18789/health > /dev/null 2>&1; then
        echo "  OpenClaw gateway is up (after ${i}s)."
        break
    fi
    sleep 1
    if [ "$i" -eq 20 ]; then
        echo "  WARNING: OpenClaw gateway health check timed out — check /tmp/openclaw-gateway.log"
    fi
done

# ── 5. Health summary ─────────────────────────────────────────────────────────
echo ""
echo "=== [5/5] Health summary ==="

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
echo "  Ollama log:       /tmp/ollama.log"
echo "  OpenClaw gw log:  /tmp/openclaw-gateway.log"
echo ""
echo "  Next steps:"
echo "    bash /root/fix_modelfile.sh                          # rebuild fine-tuned GGUF model"
echo "    bash /root/benchmark_run.sh ollama/qwen3.5:9b             # run base model benchmark"
echo "    bash /root/benchmark_run.sh ollama/qwen35-9b-gguf-claw    # run fine-tuned benchmark"
