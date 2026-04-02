#!/usr/bin/env bash
# Manual convert + register + benchmark for v28
# Usage: bash scripts/manual_v28.sh
set -euo pipefail

cd /root/pbm
export PYTHONPATH=/root/pbm
export PBM_MODEL_NAME=qwen35-9b-clawd-v28

echo "=== Step 1: Clean disk (remove old Ollama models) ==="
ollama list
echo ""
echo "Removing old models to free disk..."
for model in $(ollama list | tail -n +2 | awk '{print $1}' | grep -v 'v28'); do
    echo "  Removing $model"
    ollama rm "$model" 2>/dev/null || true
done
echo ""
df -h / | head -2
echo ""

echo "=== Step 2: Convert to GGUF ==="
python -m stages.convert
echo ""

echo "=== Step 3: Register in Ollama ==="
bash scripts/register_model.sh
echo ""

echo "=== Step 4: Verify model loaded ==="
ollama list
echo ""

echo "=== Step 5: Benchmark ==="
bash scripts/benchmark_run.sh ollama/qwen35-9b-clawd-v28
