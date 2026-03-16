#!/usr/bin/env bash
# Run a PinchBench benchmark against a given model via OpenClaw / Ollama.
#
# Usage:
#   bash benchmark_run.sh <model> [--no-upload]
#
# Examples:
#   bash benchmark_run.sh ollama/qwen35-9b-gguf-claw           # run + upload to leaderboard
#   bash benchmark_run.sh ollama/qwen35-9b-gguf-claw --no-upload  # dry run, no upload
#
# The script:
#   1. Validates that Ollama and the OpenClaw gateway are running.
#   2. Derives a safe log-file name from the model argument.
#   3. Runs the benchmark, logging output to /tmp/bench_<safe_name>.log.
#   4. Prints a summary when done.
#
# Prerequisites:
#   - startup.sh has already been run (Ollama + OpenClaw gateway are up).
#   - /workspace/synthbench/skill/scripts/run.sh exists on the pod.

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.openclaw/bin:/usr/local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-$SCRIPT_DIR/..}"

WORKSPACE="${PBM_WORKSPACE:-./workspace}"
BENCH_DIR="$WORKSPACE/skill"
SCRIPTS_DIR="$BENCH_DIR/scripts"

# ── Argument validation ───────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: bash benchmark_run.sh <model>"
    echo ""
    echo "Examples:"
    echo "  bash benchmark_run.sh ollama/qwen3:8b"
    echo "  bash benchmark_run.sh ollama/qwen3-8b-gguf-claw"
    exit 1
fi

MODEL="$1"
NO_UPLOAD=""
[ "${2:-}" = "--no-upload" ] && NO_UPLOAD="--no-upload"

# Derive a filesystem-safe name for log files (replace / : with _)
SAFE_NAME=$(echo "$MODEL" | tr '/: ' '___')
LOG_DIR="${PBM_WORKSPACE:-/workspace/synthbench}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/bench_${SAFE_NAME}.log"

echo "=== PinchBench Run ==="
echo "  Model:    $MODEL"
echo "  Log file: $LOG_FILE"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "--- Pre-flight checks ---"

# Ollama
if ! curl -sf http://127.0.0.1:11434/ > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running. Run startup.sh first."
    exit 1
fi
echo "  [OK] Ollama is up"

# OpenClaw gateway
if ! curl -sf http://127.0.0.1:18789/health > /dev/null 2>&1; then
    echo "WARNING: OpenClaw gateway health check failed — it may still work, proceeding..."
else
    echo "  [OK] OpenClaw gateway is up"
fi

# Benchmark scripts directory
if [ ! -d "$SCRIPTS_DIR" ]; then
    echo "ERROR: Benchmark scripts not found at $SCRIPTS_DIR"
    echo "  Set PBM_WORKSPACE to your workspace directory."
    exit 1
fi
echo "  [OK] Benchmark scripts found at $SCRIPTS_DIR"

# If model is an ollama model, verify it's available
if [[ "$MODEL" == ollama/* ]]; then
    OLLAMA_MODEL="${MODEL#ollama/}"
    if ! ollama list 2>/dev/null | grep -q "^${OLLAMA_MODEL}"; then
        echo ""
        echo "WARNING: Model '$OLLAMA_MODEL' not found in 'ollama list'."
        echo "  Available models:"
        ollama list 2>/dev/null | sed 's/^/    /' || echo "    (could not list)"
        echo ""
        echo "  If this is the fine-tuned GGUF model, run fix_modelfile.sh first."
        echo "  Proceeding anyway — ollama will error if the model is truly missing."
    else
        echo "  [OK] Ollama model '$OLLAMA_MODEL' is registered"
    fi
fi

echo ""
echo "--- Starting benchmark ---"
echo "  $(date '+%Y-%m-%d %H:%M:%S') — running $MODEL"
echo ""

# ── Run benchmark ─────────────────────────────────────────────────────────────
cd "$BENCH_DIR"
./scripts/run.sh --model "$MODEL" $NO_UPLOAD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=== Benchmark complete ==="
echo "  Exit code: $EXIT_CODE"
echo "  Full log:  $LOG_FILE"
echo ""

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "  Benchmark exited with errors. Last 20 lines of log:"
    tail -20 "$LOG_FILE"
    exit "$EXIT_CODE"
fi

# ── Quick score summary ───────────────────────────────────────────────────────
echo "--- Score summary (from log) ---"
grep -E "(Score|score|SCORE|%|passed|failed|task_)" "$LOG_FILE" | tail -30 || true

echo ""
echo "  To view full log:  cat $LOG_FILE"
