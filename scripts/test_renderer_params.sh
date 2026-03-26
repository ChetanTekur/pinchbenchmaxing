#!/usr/bin/env bash
# Test: does RENDERER qwen3.5 work with fixed parameters?
# If task_02 passes → issue was parameters, not template format
# If task_02 fails → explicit template is truly needed

set -euo pipefail

GGUF_PATH="${GGUF_PATH:-/workspace/synthbench/models/qwen35-9b-clawd_gguf/qwen35-9b-clawd.Q4_K_M.gguf}"

cat > /tmp/Modelfile-test << EOF
FROM $GGUF_PATH
TEMPLATE {{ .Prompt }}
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER temperature 0.6
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
SYSTEM """You are Clawd, an autonomous AI agent powered by OpenClaw. You help users accomplish real-world tasks by using tools. Be direct and competent. Get things done."""
EOF

ollama rm qwen35-9b-clawd-test 2>/dev/null || true
ollama create qwen35-9b-clawd-test -f /tmp/Modelfile-test

echo ""
echo "Running task_02_stock benchmark..."
cd /workspace/synthbench/skill
./scripts/run.sh --model ollama/qwen35-9b-clawd-test --suite task_02_stock --no-upload
