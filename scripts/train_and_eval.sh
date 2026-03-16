#!/usr/bin/env bash
# Full pipeline: prepare data → fine-tune → convert to GGUF → register → benchmark
#
# Usage:
#   bash scripts/train_and_eval.sh                          # uses model name from config.yaml
#   OLLAMA_MODEL=qwen3-8b-gguf-claw-v3 bash scripts/train_and_eval.sh
#
# Set NO_UPLOAD=1 to skip leaderboard upload:
#   NO_UPLOAD=1 bash scripts/train_and_eval.sh

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

UPLOAD_FLAG=""
[ "${NO_UPLOAD:-0}" = "1" ] && UPLOAD_FLAG="--no-upload"

MODEL_NAME="${OLLAMA_MODEL:-$(python3 -c "from utils.config import load_config; print(load_config().ollama_model_name)")}"

echo "============================================"
echo "  train_and_eval.sh"
echo "  Ollama model: $MODEL_NAME"
echo "============================================"
echo ""

# ── Step 1: Prepare SFT data ─────────────────────────────────────────────────
echo "--- Step 1/5: Prepare SFT data ---"
python3 stages/prepare.py
echo ""

# ── Step 2: Fine-tune ─────────────────────────────────────────────────────────
echo "--- Step 2/5: Fine-tune ---"
python3 stages/finetune.py
echo ""

# ── Step 3: Convert to GGUF ──────────────────────────────────────────────────
echo "--- Step 3/5: Convert to GGUF ---"
python3 stages/convert.py
echo ""

# ── Step 4: Register with Ollama ─────────────────────────────────────────────
echo "--- Step 4/5: Register with Ollama (model: $MODEL_NAME) ---"
OLLAMA_MODEL="$MODEL_NAME" bash scripts/register_model.sh
echo ""

# ── Step 5: Run benchmark ────────────────────────────────────────────────────
echo "--- Step 5/5: Run PinchBench ---"
bash scripts/benchmark_run.sh "ollama/$MODEL_NAME" $UPLOAD_FLAG
