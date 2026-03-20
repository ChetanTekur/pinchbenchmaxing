#!/usr/bin/env bash
# Register a fine-tuned GGUF model in Ollama with correct chat template and tool support
#
# Dynamically extracts the Modelfile template from the base Ollama model
# (e.g. gemma3:1b, qwen3:8b) so it works with any model family.
#
# Override values with env vars: GGUF_PATH, OLLAMA_MODEL, BASE_OLLAMA_MODEL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# ── Derive paths from config ──────────────────────────────────────────────────
GGUF_PATH="${GGUF_PATH:-$(python3 -c "from utils.config import load_config; print(load_config().gguf_file)")}"
MODEL_NAME="${OLLAMA_MODEL:-$(python3 -c "from utils.config import load_config; print(load_config().ollama_model_name)")}"

# Map HuggingFace model names to Ollama base model names for template extraction
# Override with BASE_OLLAMA_MODEL env var for custom mappings
if [ -z "${BASE_OLLAMA_MODEL:-}" ]; then
    HF_BASE=$(python3 -c "from utils.config import load_config; print(load_config().base_model)")
    case "$HF_BASE" in
        *gemma-3-1b*)   BASE_OLLAMA_MODEL="gemma3:1b" ;;
        *gemma-3-4b*)   BASE_OLLAMA_MODEL="gemma3:4b" ;;
        *gemma-3-12b*)  BASE_OLLAMA_MODEL="gemma3:12b" ;;
        *gemma-3-27b*)  BASE_OLLAMA_MODEL="gemma3:27b" ;;
        *Qwen3.5-9B*)   BASE_OLLAMA_MODEL="qwen3:8b" ;;
        *Qwen3-8B*)     BASE_OLLAMA_MODEL="qwen3:8b" ;;
        *Mistral-7B*)   BASE_OLLAMA_MODEL="mistral:7b" ;;
        *Llama-3*)      BASE_OLLAMA_MODEL="llama3:8b" ;;
        *Phi-3*)        BASE_OLLAMA_MODEL="phi3:mini" ;;
        *)
            echo "WARNING: Unknown base model '$HF_BASE' — defaulting to generic Modelfile"
            BASE_OLLAMA_MODEL=""
            ;;
    esac
fi

MODELFILE="/tmp/Modelfile-clawd"

if [ ! -f "$GGUF_PATH" ]; then
    echo "ERROR: GGUF not found at $GGUF_PATH"
    echo "  Run 'python -m stages.convert' first, or set GGUF_PATH to override."
    exit 1
fi

echo "GGUF:       $GGUF_PATH"
echo "Model:      $MODEL_NAME"
echo "Base model: ${BASE_OLLAMA_MODEL:-generic}"

# ── Build Modelfile ───────────────────────────────────────────────────────────

if [ -n "$BASE_OLLAMA_MODEL" ]; then
    # Pull base model if not already present (needed for template extraction)
    if ! ollama list 2>/dev/null | grep -q "^${BASE_OLLAMA_MODEL}"; then
        echo "Pulling base model for template extraction: $BASE_OLLAMA_MODEL"
        ollama pull "$BASE_OLLAMA_MODEL"
    fi

    # Extract template from base model's Modelfile
    echo "Extracting Modelfile template from $BASE_OLLAMA_MODEL..."
    BASE_MODELFILE=$(ollama show "$BASE_OLLAMA_MODEL" --modelfile 2>/dev/null || true)

    if [ -z "$BASE_MODELFILE" ]; then
        echo "ERROR: Could not extract Modelfile from $BASE_OLLAMA_MODEL"
        exit 1
    fi

    # Build new Modelfile: replace FROM line, keep everything else
    echo "FROM $GGUF_PATH" > "$MODELFILE"

    # Extract everything after the FROM line from the base modelfile
    echo "$BASE_MODELFILE" | sed '1,/^FROM /d' >> "$MODELFILE"

    # Append our system prompt (overrides the base model's)
    cat >> "$MODELFILE" << 'SYSTEM_EOF'
SYSTEM """You are Clawd, an autonomous AI agent powered by OpenClaw. You help users accomplish real-world tasks by using tools. Be direct and competent — start with action, not explanation. Get things done."""
SYSTEM_EOF

else
    # Generic Modelfile — no template extraction available
    echo "FROM $GGUF_PATH" > "$MODELFILE"
    cat >> "$MODELFILE" << 'GENERIC_EOF'
PARAMETER temperature 0.6
PARAMETER top_k 20
PARAMETER top_p 0.95
SYSTEM """You are Clawd, an autonomous AI agent powered by OpenClaw. You help users accomplish real-world tasks by using tools. Be direct and competent — start with action, not explanation. Get things done."""
GENERIC_EOF
fi

echo ""
echo "Removing old model..."
ollama rm "$MODEL_NAME" 2>/dev/null || true

echo "Creating model..."
ollama create "$MODEL_NAME" -f "$MODELFILE"

echo ""
echo "Testing..."
ollama run "$MODEL_NAME" "Say hello and stop."
