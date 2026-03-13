#!/usr/bin/env bash
# Pre-benchmark setup checker
# Verifies all dependencies, services, and API keys are working before running PinchBench
# Usage: bash check_setup.sh

PASS=0
FAIL=0

green() { echo "  [OK]  $1"; PASS=$((PASS+1)); }
red()   { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
warn()  { echo "  [WARN] $1"; }
header(){ echo ""; echo "=== $1 ==="; }

# ── 1. System tools ───────────────────────────────────────────────────────────
header "System Tools"

command -v jq     &>/dev/null && green "jq installed"           || red "jq not installed (apt-get install -y jq)"
command -v curl   &>/dev/null && green "curl installed"         || red "curl not installed"
command -v python3 &>/dev/null && green "python3 installed"     || red "python3 not installed"
command -v uv     &>/dev/null && green "uv installed"           || red "uv not installed"

# ── 2. Python packages ────────────────────────────────────────────────────────
header "Python Packages"

python3 -c "import pandas"     &>/dev/null && green "pandas"     || red "pandas missing (pip install pandas)"
python3 -c "import openpyxl"   &>/dev/null && green "openpyxl"   || red "openpyxl missing (pip install openpyxl)"
python3 -c "import pdfplumber" &>/dev/null && green "pdfplumber" || red "pdfplumber missing (pip install pdfplumber)"
python3 -c "import PyPDF2"     &>/dev/null && green "PyPDF2"     || red "PyPDF2 missing (pip install PyPDF2)"
python3 -c "import yaml"       &>/dev/null && green "pyyaml"     || red "pyyaml missing (pip install pyyaml)"

# ── 3. OpenClaw ───────────────────────────────────────────────────────────────
header "OpenClaw"

export PATH="$HOME/.local/bin:$HOME/.openclaw/bin:/usr/local/bin:$PATH"

if command -v openclaw &>/dev/null; then
    VER=$(openclaw --version 2>/dev/null | head -1)
    green "openclaw installed ($VER)"
else
    red "openclaw not installed — run: curl -fsSL https://openclaw.ai/install.sh | bash"
fi

OPENCLAW_CFG="$HOME/.openclaw/openclaw.json"
if [ -f "$OPENCLAW_CFG" ]; then
    green "openclaw.json exists"
    if python3 -c "import json; json.load(open('$OPENCLAW_CFG'))" &>/dev/null; then
        green "openclaw.json is valid JSON"
    else
        red "openclaw.json is invalid JSON"
    fi
else
    red "openclaw.json missing — run startup.sh"
fi

# Check gateway is running
if curl -sf http://127.0.0.1:18789/health &>/dev/null; then
    green "OpenClaw gateway running on :18789"
else
    red "OpenClaw gateway not running — run startup.sh"
fi

# Check gateway token matches between openclaw.json and env var
if [ -f "$OPENCLAW_CFG" ]; then
    CFG_TOKEN=$(python3 -c "import json; c=json.load(open('$OPENCLAW_CFG')); print(c.get('gateway',{}).get('auth',{}).get('token',''))" 2>/dev/null)
    ENV_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-}"

    if [ -z "$CFG_TOKEN" ]; then
        red "No gateway token in openclaw.json — re-run startup.sh"
    elif [ -z "$ENV_TOKEN" ]; then
        warn "OPENCLAW_GATEWAY_TOKEN env var not set — CLI will use token from openclaw.json (should be fine)"
    elif [ "$CFG_TOKEN" = "$ENV_TOKEN" ]; then
        green "Gateway token matches env var and openclaw.json"
    else
        red "Gateway token MISMATCH — re-run startup.sh with the same OPENCLAW_GATEWAY_TOKEN, or run: openclaw config set gateway.remote.token $CFG_TOKEN"
    fi

    REMOTE_TOKEN=$(python3 -c "import json; c=json.load(open('$OPENCLAW_CFG')); print(c.get('gateway',{}).get('remote',{}).get('token','NOT_SET'))" 2>/dev/null)
    if [ "$REMOTE_TOKEN" = "NOT_SET" ] || [ -z "$REMOTE_TOKEN" ]; then
        warn "gateway.remote.token not set in openclaw.json (may cause token_mismatch if openclaw CLI was run separately)"
    elif [ "$REMOTE_TOKEN" = "$CFG_TOKEN" ]; then
        green "gateway.remote.token matches gateway.auth.token"
    else
        red "gateway.remote.token does not match gateway.auth.token — run: openclaw config set gateway.remote.token $CFG_TOKEN"
    fi
fi

# ── 4. Ollama ─────────────────────────────────────────────────────────────────
header "Ollama"

if curl -sf http://127.0.0.1:11434/ &>/dev/null; then
    green "Ollama running on :11434"
else
    red "Ollama not running — run: ollama serve &"
fi

if ollama list 2>/dev/null | grep -q "qwen35-9b-gguf-claw"; then
    green "qwen35-9b-gguf-claw registered in Ollama"
else
    red "qwen35-9b-gguf-claw not found in Ollama — run fix_modelfile.sh"
fi

# ── 5. API Keys ───────────────────────────────────────────────────────────────
header "API Keys"

if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    # Test with a real API call
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        https://openrouter.ai/api/v1/models)
    [ "$HTTP" = "200" ] && green "OPENROUTER_API_KEY valid (HTTP $HTTP)" || red "OPENROUTER_API_KEY set but API returned HTTP $HTTP"
else
    red "OPENROUTER_API_KEY not set"
fi

if [ -n "${BRAVE_API_KEY:-}" ]; then
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "X-Subscription-Token: $BRAVE_API_KEY" \
        "https://api.search.brave.com/res/v1/web/search?q=test&count=1")
    [ "$HTTP" = "200" ] && green "BRAVE_API_KEY valid (HTTP $HTTP)" || red "BRAVE_API_KEY set but API returned HTTP $HTTP"
else
    red "BRAVE_API_KEY not set"
fi

if [ -n "${OPENCLAW_GATEWAY_TOKEN:-}" ]; then
    green "OPENCLAW_GATEWAY_TOKEN set"
else
    warn "OPENCLAW_GATEWAY_TOKEN not set (startup.sh will generate one)"
fi

# ── 6. PinchBench ─────────────────────────────────────────────────────────────
header "PinchBench"

WORKSPACE="${PBM_WORKSPACE:-./workspace}"
BENCH_DIR="$WORKSPACE/skill"
if [ -d "$BENCH_DIR" ]; then
    green "PinchBench skill dir exists ($BENCH_DIR)"
else
    red "PinchBench skill dir missing at $BENCH_DIR"
fi

if [ -f "$BENCH_DIR/scripts/run.sh" ]; then
    green "run.sh found"
else
    red "run.sh not found"
fi

LIB_AGENT="$BENCH_DIR/scripts/lib_agent.py"
if [ -f "$LIB_AGENT" ]; then
    if grep -q '"ollama/"' "$LIB_AGENT" || grep -q "'ollama/'" "$LIB_AGENT"; then
        green "lib_agent.py has ollama/ in KNOWN_PROVIDERS"
    else
        red "lib_agent.py missing ollama/ in KNOWN_PROVIDERS — run: sed -i 's/KNOWN_PROVIDERS = (/KNOWN_PROVIDERS = (\"ollama\/\", /' $LIB_AGENT"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════"
echo "  PASSED: $PASS"
echo "  FAILED: $FAIL"
echo "════════════════════════════════════"

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "All checks passed! Ready to benchmark:"
    echo "  bash /root/scripts/benchmark_run.sh ollama/qwen35-9b-gguf-claw"
else
    echo ""
    echo "Fix the $FAIL failing checks above before running the benchmark."
fi
