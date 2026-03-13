#!/usr/bin/env bash
# RunPod environment setup for synthbench/PinchBench project
# Requires 50GB+ container disk
# Usage: bash setup.sh

set -e

echo "=== Installing system packages ==="
# jq: needed by benchmark scripts (lib_grading.py shell calls, result parsing)
apt-get update -qq && apt-get install -y vim jq

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo "=== Installing Python packages ==="
pip install --upgrade pip
pip install \
    anthropic \
    trl \
    transformers \
    peft \
    datasets \
    accelerate \
    huggingface_hub \
    safetensors \
    tqdm \
    vllm \
    pandas \
    openpyxl \
    pdfplumber \
    PyPDF2

echo "=== Installing Unsloth ==="
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"

echo "=== Setting up ANTHROPIC_API_KEY ==="
if [ -f /workspace/synthbench/anthropic_key ]; then
    export ANTHROPIC_API_KEY=$(cat /workspace/synthbench/anthropic_key)
    echo "export ANTHROPIC_API_KEY=$(cat /workspace/synthbench/anthropic_key)" >> ~/.bashrc
    echo "  API key loaded from /workspace/synthbench/anthropic_key"
else
    echo "  WARNING: /workspace/synthbench/anthropic_key not found — set ANTHROPIC_API_KEY manually"
fi

echo ""
echo "=== Setup complete! ==="
echo "  Run 'source ~/.bashrc' or start a new shell to apply PATH changes."
