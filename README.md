# PinchBench Maxing

Fine-tune open-source LLMs to compete on [PinchBench](https://pinchbench.com) — a 23-task benchmark for AI agents running on the [OpenClaw](https://openclaw.ai) framework.

## Results

| Model | Score | % |
|-------|-------|---|
| qwen3:8b (base) | 3.065 / 23 | 13% |
| qwen3-8b-gguf-claw (v1 fine-tune) | 9.9 / 23 | 43% |
| qwen35-9b-gguf-claw (v2 fine-tune) | TBD | TBD |

---

## Project Structure

```
config.yaml              # single source of truth — model, paths, training params
stages/
  prepare.py             # convert train.jsonl → SFT format
  finetune.py            # LoRA fine-tuning with Unsloth
  convert.py             # export merged model → GGUF for Ollama
  probe.py               # interactive model testing (no GGUF needed)
scripts/
  setup_pod.sh           # one-time pod setup (OpenClaw, Node 22, deps)
  check_setup.sh         # pre-benchmark checker (APIs, deps, services)
  startup.sh             # every-session startup (Ollama + OpenClaw gateway)
  benchmark_run.sh       # run PinchBench for a given model
  fix_modelfile.sh       # register GGUF with Ollama with correct tool-call template
  patch_openclaw_image_gen.py  # remove invalid keys from openclaw.json
utils/
  config.py              # loads config.yaml, computes all paths
generate.py              # generate synthetic training data via Claude Batch API
llm_judge.py             # score examples 1-5, filter at --min 3
topup.py                 # fill gaps to target examples/task
repair.py                # fix JSON parse failures in training data
inspect_data.py          # stats and validation
openclaw_template.json   # OpenClaw config template (secrets injected at runtime)
Dockerfile               # full image: PyTorch + Unsloth + Ollama (for RunPod)
Dockerfile.bench         # lightweight image: Ollama only (no PyTorch)
```

---

## Config

All paths, model names, and hyperparameters live in `config.yaml`. Override the workspace with:

```bash
export SYNTHDATA_WORKSPACE=/your/path   # default: ./workspace
```

Key settings:

```yaml
model:
  base: Qwen/Qwen3.5-9B      # HuggingFace model to fine-tune
  name: qwen35-9b-clawd       # used for output dirs and Ollama model name

paths:
  workspace: ${SYNTHDATA_WORKSPACE:-./workspace}
```

---

## Required API Keys

| Key | Purpose | Required For |
|-----|---------|--------------|
| `OPENROUTER_API_KEY` | LLM judge (claude-opus-4.5) + web search | Benchmarking |
| `BRAVE_API_KEY` | Web search tasks (task_02, task_06, task_18) | Benchmarking |
| `OPENCLAW_GATEWAY_TOKEN` | OpenClaw gateway auth (any random string) | Benchmarking |
| `ANTHROPIC_API_KEY` | Synthetic data generation | Data generation only |

---

## Useful Paths

| Path | Purpose |
|------|---------|
| `/tmp/openclaw-gateway.log` | OpenClaw gateway logs |
| `/tmp/ollama.log` | Ollama logs |
| `/root/.openclaw/openclaw.json` | Generated OpenClaw config (from template + env vars) |
| `/workspace/synthbench/skill/` | PinchBench benchmark scripts |
| `/workspace/synthbench/data/` | Training data |
| `/workspace/synthbench/qwen35-9b-clawd_merged_gguf/` | Fine-tuned GGUF |

---

## RunPod Setup (New Pod)

### Step 1 — One-time setup

In Jupyter:
```python
import urllib.request, subprocess
urllib.request.urlretrieve("https://raw.githubusercontent.com/ChetanTekur/pinchbenchmaxing/main/scripts/setup_pod.sh", "/root/setup_pod.sh")
subprocess.run(["bash", "/root/setup_pod.sh"])
```

Installs: Node 22, OpenClaw, jq, pandas, pdfplumber, openpyxl. Also patches `lib_agent.py` to add `ollama/` to `KNOWN_PROVIDERS`.

### Step 2 — Every session startup

Set env vars, then run startup:
```bash
export OPENROUTER_API_KEY="sk-or-..."
export BRAVE_API_KEY="BSA..."
export OPENCLAW_GATEWAY_TOKEN=$(openssl rand -hex 24)
bash /root/scripts/startup.sh
```

`startup.sh` handles: kill stale processes, generate `openclaw.json` from template, start Ollama, start OpenClaw gateway, health check.

### Step 3 — Check everything is ready

In Jupyter:
```python
import urllib.request, subprocess
urllib.request.urlretrieve("https://raw.githubusercontent.com/ChetanTekur/pinchbenchmaxing/main/scripts/check_setup.sh", "/root/check_setup.sh")
subprocess.run(["bash", "/root/check_setup.sh"])
```

Fix anything flagged before running the benchmark.

### Step 4 — Run benchmark

```bash
bash /root/scripts/benchmark_run.sh ollama/qwen35-9b-gguf-claw
```

---

## Fine-Tuning Pipeline

### 1. Generate synthetic data
```bash
python generate.py
```
Uses Claude Batch API to generate ~40 agent traces per PinchBench task.

### 2. Score and filter
```bash
python llm_judge.py --min 3
```

### 3. Top up weak tasks
```bash
python topup.py
```

### 4. Prepare for SFT
```bash
python stages/prepare.py
```

### 5. Fine-tune
```bash
python stages/finetune.py --dry-run   # sanity check first
python stages/finetune.py
```

### 6. Convert to GGUF
```bash
python stages/convert.py
```

### 7. Register with Ollama
```bash
bash scripts/fix_modelfile.sh
```

### 8. Test the model
```bash
bash scripts/test_tool_call.sh
```

### 9. Benchmark
```bash
bash scripts/benchmark_run.sh ollama/qwen35-9b-gguf-claw
```

---

## Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| `openclaw` not found | `curl -fsSL https://openclaw.ai/install.sh \| bash` — NOT `npm install -g openclaw` (that's a placeholder) |
| `ollama/` rejected by PinchBench | Add `"ollama/"` to `KNOWN_PROVIDERS` in `skill/scripts/lib_agent.py` — `setup_pod.sh` does this automatically |
| GGUF model ignores tool calls | Run `scripts/fix_modelfile.sh` to recreate with correct Modelfile template |
| Unsloth saves GGUF to `{merged}_gguf/` not output dir | Expected — `stages/convert.py` handles this |
| OpenClaw config invalid key | Run `python3 scripts/patch_openclaw_image_gen.py` to clean up |
| Gateway log | `cat /tmp/openclaw-gateway.log` |
| Ollama log | `cat /tmp/ollama.log` |

---

## PinchBench Task Dependencies

| Tasks | Dependency |
|-------|-----------|
| task_00–01, task_03, task_05, task_07–12, task_14–17, task_22 | None |
| task_02, task_06, task_18 | Brave web search (`BRAVE_API_KEY`) |
| task_04 | wttr.in (free public API, no key needed) |
| task_13 | Image generation (via OpenRouter, uses `OPENROUTER_API_KEY`) |
| task_19 | pandas + openpyxl |
| task_20, task_21 | pdfplumber + PyPDF2 |
