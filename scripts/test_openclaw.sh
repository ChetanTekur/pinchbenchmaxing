#!/usr/bin/env bash
# Test model through OpenClaw gateway (how benchmark actually runs it)
set -euo pipefail

MODEL="${1:-ollama/qwen35-9b-clawd-v22}"

echo "Testing health..."
curl -s http://127.0.0.1:18789/health && echo ""

echo ""
echo "Testing $MODEL through OpenClaw with tools..."
echo "Raw response:"
curl -v http://127.0.0.1:18789/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"List the files in the current directory\"}],
    \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"list_files\", \"description\": \"List files in a directory\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}}}}]
  }" 2>&1
