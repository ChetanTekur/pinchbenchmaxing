# Session Report — PinchBench Benchmarking Setup & v1 Results

**Date:** 2026-03-12
**Goal:** Get PinchBench running end-to-end on RunPod (A100 80GB) using Ollama-hosted models, validate base Qwen3 8B, and measure the fine-tuned v1 model.

---

## Timeline

### Phase 1 — Environment bootstrap

1. Confirmed RunPod pod had A100 80GB (CA-2 region), network storage at `/workspace/`.
2. Ran `setup.sh` to install Python packages, uv, and Unsloth.
3. Installed `openclaw` CLI via pip into `~/.local/bin`.
4. Cloned / confirmed `/workspace/synthbench` existed with PinchBench skill scripts.

### Phase 2 — Configuring OpenClaw for Ollama

OpenClaw defaults to Anthropic's API. We needed it to call a local Ollama server instead.

**Steps:**
- Started Ollama: `ollama serve &`
- Pulled base model: `ollama pull qwen3:8b`
- Edited `~/.openclaw/openclaw.json` to add the `ollama` provider with `baseUrl` and a dummy `apiKey`:
  ```json
  "ollama": {
    "baseUrl": "http://127.0.0.1:11434",
    "apiKey": "ollama-local",
    "api": "ollama",
    "models": []
  }
  ```
- Started OpenClaw gateway: `openclaw gateway --port 18789 &`

### Phase 3 — Configuring the LLM judge (OpenRouter)

PinchBench uses a separate LLM judge agent (`bench-judge-anthropic-claude-opus-4-5`) to score open-ended answers. OpenClaw does not support the native Anthropic API directly; it must route through OpenRouter.

**Steps:**
- Added `openrouter` provider to `openclaw.json`:
  ```json
  "openrouter": {
    "baseUrl": "https://openrouter.ai/api/v1",
    "apiKey": "sk-or-v1-...",
    "models": []
  }
  ```
- Set the judge agent's model to `openrouter/anthropic/claude-opus-4.5` in both `openclaw.json` (agents list) and the agent's `auth-profiles.json`.
- Fixed `lib_agent.py` KNOWN_PROVIDERS to include `"anthropic/"` and `"openrouter/"` so the routing logic didn't reject these prefixes.
- Fixed `lib_grading.py` to use `anthropic/claude-opus-4.5` (routed via openrouter provider) as the judge model.

### Phase 4 — Baseline benchmark (base qwen3:8b)

Ran:
```bash
cd /workspace/synthbench/skill
./scripts/run.sh --model ollama/qwen3:8b --no-upload 2>&1 | tee /tmp/bench_base.log
```

**Result:** 13% (3.065 / 23 tasks)
**Submission:** https://pinchbench.com/submission/e150a1ec-dbf6-460a-867b-afc625fdf105

The base model had no agent-specific training — poor tool use and task completion.

### Phase 5 — Registering the fine-tuned GGUF model

The fine-tuned v1 model was converted to GGUF (`qwen3-8b.Q4_K_M.gguf`) and stored at `/workspace/synthbench/qwen3-8b-clawd_gguf_gguf/`.

**Problem:** The initial Modelfile created for the GGUF did not include the tool-calling template block. Without this, Ollama never passes tools to the model, and the model cannot invoke any OpenClaw tools — resulting in near-zero task scores.

**Fix (`fix_modelfile.sh`):** Copied the full chat template from `ollama show qwen3:8b --modelfile` (which includes `<tools>...</tools>` XML handling and `<tool_call>` response parsing), replaced the `FROM` line with the GGUF path, and re-created the model:
```bash
ollama rm qwen3-8b-gguf-claw
ollama create qwen3-8b-gguf-claw -f /tmp/Modelfile-clawd
```

### Phase 6 — Fine-tuned v1 benchmark

Ran:
```bash
cd /workspace/synthbench/skill
./scripts/run.sh --model ollama/qwen3-8b-gguf-claw --no-upload 2>&1 | tee /tmp/bench_finetune.log
```

**Result:** 43% (9.9 / 23 tasks)
**Submission:** https://pinchbench.com/submission/a7fc17f7-b5a6-41d5-be65-a93be14253c2

---

## Issues Encountered & Fixes

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `openclaw-agent` processes left over from previous run | RunPod pod reuse | Kill with `ps aux \| grep -E "openclaw\|ollama" \| grep -v grep \| awk '{print $2}' \| xargs kill -9` |
| 2 | "fetch failed" when running benchmark | Ollama not started | Run `ollama serve > /tmp/ollama.log 2>&1 &` before anything |
| 3 | "No API key for ollama provider" | openclaw.json missing ollama `apiKey` | Set dummy key: `openclaw config set models.providers.ollama.apiKey "ollama-local"` (also set `baseUrl`) |
| 4 | Judge model prefix rejected | `KNOWN_PROVIDERS` in `lib_agent.py` didn't include `"anthropic/"` | Added `"anthropic/"` and `"openrouter/"` to the tuple |
| 5 | "No API provider for anthropic" | OpenClaw has no native Anthropic API support | Route through OpenRouter by using `openrouter/anthropic/claude-opus-4.5` as model ID with the openrouter provider configured |
| 6 | `openrouter/` prefix double-applied | `openclaw.json` agent list had `openrouter/anthropic/claude-opus-4.5`, but lib code added prefix again | Standardized: agent list uses `anthropic/claude-opus-4.5`, the openrouter provider routes it |
| 7 | GGUF model never calls tools | Default Modelfile template missing tool-call block | Ran `fix_modelfile.sh` to regenerate with full qwen3:8b template |
| 8 | Fine-tuned model loops on datautils tasks | Over-fit to `task_11_clawhub` (datautils call pattern in training data) | Identified for v2 fix: clean training data, reduce datautils examples, adjust hyperparameters |

---

## Benchmark Results — Fine-tuned v1 Category Breakdown

| Category | Score | Max | % |
|----------|-------|-----|---|
| Basic | 1.0 | 1.0 | 100% |
| Writing | 1.9 | 2.0 | 93% |
| Coding | 1.0 | 1.0 | 100% |
| Calendar | 0.8 | 1.0 | 83% |
| Context | 0.8 | 1.0 | 80% |
| Research | 0.8 | 3.0 | 26% |
| Comprehension | 0.8 | 4.0 | 21% |
| Organization | 0.4 | 1.0 | 36% |
| Data Analysis | 0.4 | 1.0 | 43% |
| File Operations | 1.1 | 3.0 | 38% |
| Complex Tasks | 0.5 | 1.0 | 47% |
| Memory | 0.3 | 1.0 | 28% |
| Creative | 0.0 | 1.0 | 0% |
| Content Transformation | 0.0 | 1.0 | 0% |
| Synthesis | 0.1 | 1.0 | 8% |
| **Total** | **9.9** | **23.0** | **43%** |

### Analysis

**Strong (>75%):** Basic, Writing, Coding, Calendar, Context — these map closely to training data coverage and straightforward tool-use patterns.

**Weak (<30%):** Research, Comprehension, Memory, Creative, Content Transformation, Synthesis — several patterns:
- **Looping bug**: model sometimes repeats the same tool call (datautils overfitting). Affects Research and Comprehension scores.
- **Missing tool calls**: Creative and Content Transformation likely need specific tool combinations the model wasn't trained on.
- **Memory**: model doesn't maintain state across long task chains (no memory tool training examples).
- **Synthesis**: requires combining multiple sources; training data may not cover this well.

---

## What's Needed for v2

### Data Fixes
1. **Remove/cap datautils examples** — likely over-represented in task_11 training data, causing looping.
2. **Add more examples for weak categories**: Research (multi-step web lookups), Comprehension (long-doc reading), Memory (state persistence), Creative (generation + formatting), Content Transformation (convert + reformat), Synthesis (multi-source aggregation).
3. **Diversity**: ensure no single tool-call pattern appears more than ~5% of total examples.

### Training Fixes
1. **Reduce epochs** or add early stopping — v1 likely over-trained on the datautils pattern.
2. **Increase learning rate warmup** to prevent memorization of early batches.
3. **Add response-length regularization** to penalize looping completions.

### Infrastructure
1. Write `finetune.py` (currently TODO) using Unsloth + QLoRA for Qwen3-8B.
2. Add `jq`, `pandas`, `openpyxl`, `pdfplumber`, `PyPDF2` to `setup.sh` (done this session).
3. Keep `startup.sh` as the canonical startup sequence for every new pod session.

### Target
60%+ on PinchBench leaderboard with v2 fine-tuned model.

---

## Working Configuration Reference

### openclaw.json (key sections)
```json
{
  "models": {
    "providers": {
      "ollama": {
        "baseUrl": "http://127.0.0.1:11434",
        "apiKey": "ollama-local",
        "api": "ollama",
        "models": []
      },
      "openrouter": {
        "baseUrl": "https://openrouter.ai/api/v1",
        "apiKey": "sk-or-v1-...",
        "models": []
      }
    }
  },
  "agents": {
    "list": [
      {
        "id": "bench-judge-anthropic-claude-opus-4-5",
        "model": "openrouter/anthropic/claude-opus-4.5"
      }
    ]
  }
}
```

### lib_agent.py KNOWN_PROVIDERS
```python
KNOWN_PROVIDERS = ("anthropic/","openrouter/", "vllm/", "ollama/", "nvidia-api/", "nvidia-nemotron/")
```

### Startup sequence
```bash
# Kill stale processes
ps aux | grep -E "openclaw|ollama" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null

# Start Ollama
ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

# Start OpenClaw gateway
export PATH="$HOME/.local/bin:$PATH"
openclaw gateway --port 18789 > /tmp/openclaw-gateway.log 2>&1 &
sleep 3

# Run benchmark
cd /workspace/synthbench/skill
./scripts/run.sh --model ollama/qwen3:8b --no-upload 2>&1 | tee /tmp/bench_base.log
# or:
./scripts/run.sh --model ollama/qwen3-8b-gguf-claw --no-upload 2>&1 | tee /tmp/bench_finetune.log
```

### Key file locations on RunPod
| File | Path |
|------|------|
| OpenClaw config | `/root/.openclaw/openclaw.json` |
| Judge auth profiles | `/root/.openclaw/agents/bench-judge-anthropic-claude-opus-4-5/agent/auth-profiles.json` |
| PinchBench lib_agent | `/workspace/synthbench/skill/scripts/lib_agent.py` |
| PinchBench lib_grading | `/workspace/synthbench/skill/scripts/lib_grading.py` |
| Fine-tuned GGUF | `/workspace/synthbench/qwen3-8b-clawd_gguf_gguf/qwen3-8b.Q4_K_M.gguf` |
| Training data | `/workspace/synthbench/data/train.jsonl` + `val.jsonl` |
| Ollama log | `/tmp/ollama.log` |
| OpenClaw gateway log | `/tmp/openclaw-gateway.log` |
