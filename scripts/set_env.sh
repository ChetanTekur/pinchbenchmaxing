#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Environment Configuration
#
# Fill in your values below, then save this file to the network volume so it
# persists across pod restarts:
#
#   cp /root/scripts/set_env.sh /workspace/synthbench/set_env.sh
#   vim /workspace/synthbench/set_env.sh   # fill in your values
#
# startup.sh automatically sources /workspace/synthbench/set_env.sh on boot.
# You can also source it manually: source /workspace/synthbench/set_env.sh
# ─────────────────────────────────────────────────────────────────────────────

# Workspace root (network volume — persists across pod restarts)
export PBM_WORKSPACE="/workspace/synthbench"

# Required for data generation and LLM judge
export ANTHROPIC_API_KEY=""

# Required for PinchBench judge (claude-opus-4-5 via OpenRouter)
export OPENROUTER_API_KEY=""

# Required for web search tasks in PinchBench
export BRAVE_API_KEY=""

# OpenClaw gateway auth token (any random string — generated automatically if unset)
export OPENCLAW_GATEWAY_TOKEN=""
