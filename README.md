# PinchBench Maxing

Fine-tune open-source LLMs to compete on [PinchBench](https://pinchbench.com) — a 23-task benchmark for AI agents running on the [OpenClaw](https://openclaw.ai) framework.

---

## Results

| Model | Score | % |
|-------|-------|---|
| qwen3:8b (base, no fine-tune) | 3.065 / 23 | 13% |
| qwen3-8b-gguf-claw (v1 fine-tune) | 9.9 / 23 | 43% |
| qwen35-9b-clawd-v3 (v3 fine-tune) | 16.8 / 23 | 73% |
| qwen35-9b-clawd-v4 (in progress) | TBD | TBD |

---

## Quick Guide

**What you need:** RunPod account, Anthropic API key (data generation), OpenRouter API key (benchmark judge + image gen), Brave API key (web search tasks).

```bash
# 1. Start a RunPod pod using the Docker image (see RunPod Setup below)
#    Mount a network volume at /workspace

# 2. SSH in, fill in your API keys
cp /root/scripts/set_env.sh /workspace/synthbench/set_env.sh
vim /workspace/synthbench/set_env.sh   # fill in all keys

# 3. Run startup (every session)
bash /root/scripts/startup.sh

# 4. Register with PinchBench (one time only)
cd $PBM_WORKSPACE/skill && bash scripts/run.sh --register

# 5. Run the agentic loop
cd /root/pbm
PYTHONPATH=. python3 loop.py run --model qwen35-9b-clawd-v3
```

The loop runs: **Eval → Analysis → Data → Curator → Trainer → repeat**.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [RunPod Setup](#runpod-setup)
- [API Keys](#api-keys)
- [Configuration](#configuration)
- [Data Generation Pipeline](#data-generation-pipeline)
- [Fine-Tuning Pipeline](#fine-tuning-pipeline)
- [Running the Benchmark](#running-the-benchmark)
- [Agentic Loop](#agentic-loop)
- [Scripts Reference](#scripts-reference)
- [Key Insights and Gotchas](#key-insights-and-gotchas)
- [PinchBench Task Reference](#pinchbench-task-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Agentic Loop                      │
│                                                     │
│  EvalAgent → EvalAnalysisAgent → DataAgent          │
│      ↑          CuratorAgent → TrainerAgent         │
│      └──────────────────────────────────────────────┘
│                                                     │
│  State persisted to: $PBM_WORKSPACE/data/loop_state.json
└─────────────────────────────────────────────────────┘

Data pipeline:  generate.py → llm_judge.py → topup.py → train.jsonl
Train pipeline: stages/prepare.py → stages/finetune.py → stages/convert.py
Serving:        fix_modelfile.sh → Ollama → OpenClaw gateway → PinchBench
```

**Key components:**
- **OpenClaw** — agent framework that routes tool calls (web search, file ops, image gen) to providers. Runs as a local gateway on port 18789.
- **Ollama** — serves the fine-tuned GGUF model locally on port 11434.
- **PinchBench** — sends tasks to the model via OpenClaw, scores responses with automated checks + an LLM judge (Claude via OpenRouter).
- **Unsloth** — fast LoRA fine-tuning on GPU, exports merged 16-bit → GGUF.

---

## Project Structure

```
config.yaml                  # single source of truth — model, paths, training params
loop.py                      # agentic loop orchestrator
agents/
  base.py                    # AgentState dataclass + Agent base class
  eval_agent.py              # runs PinchBench, parses scores
  eval_analysis_agent.py     # probes all model versions, generates hypothesis/diagnosis
  data_agent.py              # calls topup.py for weak tasks
  curator_agent.py           # runs llm_judge, filters low-quality examples
  trainer_agent.py           # prepare → finetune → convert → register in Ollama
stages/
  prepare.py                 # convert train.jsonl → SFT chat format
  finetune.py                # LoRA fine-tuning with Unsloth
  convert.py                 # export merged model → GGUF for Ollama
  probe.py                   # interactive model testing (no GGUF needed)
scripts/
  startup.sh                 # every-session startup (kill stale, start Ollama + OpenClaw)
  benchmark_run.sh           # run PinchBench for a given model
  fix_modelfile.sh           # register GGUF in Ollama with correct tool-call Modelfile
  set_env.sh                 # template for API keys (copy to network volume)
utils/
  config.py                  # loads config.yaml, computes all derived paths
generate.py                  # generate synthetic training data via Claude Batch API
llm_judge.py                 # score examples 1–5 using Claude, filter at --min 3
topup.py                     # fill gaps to target examples/task
repair.py                    # fix JSON parse failures in raw generated data
inspect_data.py              # dataset stats and validation
openclaw_template.json       # OpenClaw config template (secrets injected at runtime)
Dockerfile                   # full image: PyTorch + Unsloth + Ollama + OpenClaw
Dockerfile.bench             # lightweight image: Ollama + OpenClaw only (no training)
```

---

## RunPod Setup

### Why RunPod?

Fine-tuning Qwen3.5-9B requires a GPU with ~20 GB VRAM. RunPod's network volumes persist across pod restarts, so your training data, model weights, and loop state are never lost when you stop a pod.

### Network Volume (create once)

1. Go to RunPod → **Storage** → **+ Network Volume**
2. Name: `synthbench` (or anything you like)
3. Region: **CA-2** (important — pods must be in the same datacenter)
4. Size: **200 GB** minimum (GGUF alone is ~5 GB; multiple model versions accumulate)
5. Click **Create**

The volume mounts at `/workspace` inside the pod.

### Pod Configuration

1. Go to RunPod → **Pods** → **+ Deploy**
2. Select a GPU with ≥24 GB VRAM:
   - **RTX 4090** (24 GB) — cheapest option, works
   - **A100 PCIe 40 GB** — faster, more headroom
   - **A100 SXM 80 GB** — fastest, needed for very long sequences
3. **Docker image:** `ghcr.io/chetantekur/pinchbenchmaxing:latest`
   - Or build your own: `docker build -t your-tag .`
4. **Volume:** attach the network volume, mount at `/workspace`
5. **Region:** must match your volume (CA-2)
6. **Container disk:** 50 GB minimum (Unsloth, CUDA, base model weights cache)
7. **Expose ports:** 18789 (OpenClaw gateway), 11434 (Ollama) — optional, not required
8. Click **Deploy**

### What the Docker Image Contains

The `Dockerfile` bakes in everything needed for training and benchmarking:

- `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` as base
- System: `curl wget git vim jq build-essential libssl-dev`
- Python: `pyyaml anthropic trl transformers peft datasets accelerate huggingface_hub safetensors tqdm pandas openpyxl pdfplumber PyPDF2`
- Unsloth (CUDA 12.4 + Torch 2.6 build) — for fast LoRA fine-tuning
- llama.cpp — pre-compiled inside the image so `convert.py` doesn't wait 3 minutes at runtime
- Ollama — for serving GGUF models
- Node.js 22 + OpenClaw — **must be Node 22**, Node 20 will fail (openclaw requires ≥22.12.0)

The image does **not** contain the project code. Instead, `startup.sh` clones `https://github.com/ChetanTekur/pinchbenchmaxing.git` to `/root/pbm` on every pod start, and runs `git pull` on subsequent restarts. This means you can update the code by pushing to GitHub and restarting the pod — no need to rebuild the Docker image.

### First-Time Pod Setup: PinchBench Workspace

The benchmark scripts live on the PinchBench server side, not in this repo. You need to download them once to your network volume:

```bash
# SSH into the pod, then:
bash /root/scripts/startup.sh        # starts Ollama + OpenClaw, clones repo

# Download the PinchBench benchmark scripts to the network volume
# (exact command provided when you register at pinchbench.com)
# They install to: /workspace/synthbench/skill/
```

The benchmark scripts directory (`$PBM_WORKSPACE/skill/`) must exist before the agentic loop can run the eval stage.

---

## API Keys

### The full picture

| Key | Used For | Where |
|-----|----------|-------|
| `ANTHROPIC_API_KEY` | Generating synthetic training data (`generate.py`, `llm_judge.py`, `topup.py`) | Local machine or pod |
| `OPENROUTER_API_KEY` | **LLM judge inside PinchBench** (claude-opus-4.5), image generation task | Pod only |
| `BRAVE_API_KEY` | Web search tasks in PinchBench (task_02, task_06, task_18) | Pod only |
| `OPENCLAW_GATEWAY_TOKEN` | Auth token for OpenClaw gateway (any random string) | Pod only |

### Critical insight: OpenRouter vs Anthropic for the judge

PinchBench's LLM judge — which grades your model's responses — calls Claude via **OpenRouter**, not directly via the Anthropic API. This is because:

1. OpenRouter is configured in `openclaw.json` as the provider for `claude-opus-4.5`
2. The judge agent lives inside OpenClaw's configuration, not in this codebase
3. The `ANTHROPIC_API_KEY` is **only** used by this repo's data generation scripts on your local machine (or directly in the pod for topup/judge)

So you need **both** keys for different things. If you only have an Anthropic key, your generated data will be fine but the PinchBench scoring will fail.

### Setting up keys on the pod

The pod persists keys across restarts via a file on the network volume:

```bash
# Copy the template from the cloned repo
cp /root/pbm/scripts/set_env.sh /workspace/synthbench/set_env.sh

# Edit and fill in all values
vim /workspace/synthbench/set_env.sh
```

The file looks like:
```bash
export PBM_WORKSPACE="/workspace/synthbench"
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
export BRAVE_API_KEY="BSA..."
export OPENCLAW_GATEWAY_TOKEN="any-random-string-here"
```

`startup.sh` sources this file automatically on every pod start. **Do not run `sh set_env.sh`** — use `source set_env.sh` or let `startup.sh` do it. Running with `sh` creates a subshell that exits immediately, exporting nothing to your current session.

---

## Configuration

Everything configurable lives in `config.yaml`. Never hardcode paths in scripts — always read from config via `utils/config.py`.

```yaml
model:
  base: Qwen/Qwen3.5-9B       # HuggingFace base model
  name: qwen35-9b-clawd        # used for output dirs + Ollama model name

paths:
  workspace: ${PBM_WORKSPACE:-./workspace}  # override via env var

data:
  examples_per_task: 70        # target training examples per PinchBench task
  val_split: 0.1               # fraction held out as validation
  min_judge_score: 3           # discard examples below this quality (1–5 scale)

training:
  epochs: 3
  batch_size: 2
  grad_accum: 4                # effective batch size = batch_size × grad_accum
  learning_rate: 2e-4
  lora_r: 16
  lora_alpha: 32
  max_seq_len: 4096

convert:
  quantization: q4_k_m         # q4_k_m = good default; q8_0 = higher quality

loop:
  max_iterations: 2            # increase after verifying end-to-end works
  target_score: 0.85           # loop stops when benchmark avg exceeds this
  weak_task_threshold: 0.50    # tasks below this trigger data generation
  examples_per_weak_task: 20   # new examples to generate per weak task per iter
```

### Derived paths (from `utils/config.py`)

All paths are computed from `workspace` — never hardcoded:

| Property | Path |
|----------|------|
| `cfg.data_dir` | `{workspace}/data/` |
| `cfg.adapter_dir` | `{workspace}/{model_name}/` |
| `cfg.merged_dir` | `{workspace}/{model_name}_merged/` |
| `cfg.gguf_dir` | `{workspace}/{model_name}_merged_gguf/` |
| `cfg.gguf_file` | `{workspace}/{model_name}_merged_gguf/{model_name}_merged.Q4_K_M.gguf` |
| `cfg.train_file` | `{workspace}/data/train.jsonl` |
| `cfg.val_file` | `{workspace}/data/val.jsonl` |

---

## Data Generation Pipeline

This section covers generating the synthetic training dataset. Run these locally (cheaper) or on the pod.

### Step 1: Generate initial data

```bash
cd /root/pbm
export ANTHROPIC_API_KEY="sk-ant-..."
export PYTHONPATH=/root/pbm

python generate.py submit    # submits batch to Claude Batch API
python generate.py status    # poll until done (check every few minutes)
python generate.py collect   # downloads results → train.jsonl + val.jsonl
```

`generate.py` uses Claude's Batch API (50% cheaper than real-time) to generate ~40 synthetic agent traces per PinchBench task (23 tasks × 40 = ~920 examples). Each trace is a realistic multi-turn conversation where Clawd uses tools to complete a task.

The output is two JSONL files:
- `$PBM_WORKSPACE/data/train.jsonl` — training examples
- `$PBM_WORKSPACE/data/val.jsonl` — validation examples (held out during fine-tuning)

### Step 2: Inspect and validate

```bash
python inspect_data.py stats     # examples per task, format breakdown
python inspect_data.py validate  # check for structural issues
python inspect_data.py sample    # preview random examples
```

Look for tasks with very few examples or high validation error rates — these need special attention.

### Step 3: Score with LLM judge

```bash
python llm_judge.py run           # scores all examples 1–5 (resumes if interrupted)
python llm_judge.py report        # summary stats per task
python llm_judge.py filter --min 3  # discard low-quality examples in-place
```

The judge calls Claude (via `ANTHROPIC_API_KEY`) to rate each example on:
- Task completion
- Correct tool usage
- Realistic agent behavior
- Response quality

Scores are saved to `$PBM_WORKSPACE/data/scores.json`. The `filter` step rewrites `train.jsonl` and `val.jsonl` keeping only examples that scored ≥3.

### Step 4: Top up weak tasks

If some tasks have fewer than `examples_per_task` examples after filtering:

```bash
python topup.py count   # show current vs target per task
python topup.py run     # generate more examples for tasks below target

# Check progress and collect when done:
python topup.py status
python topup.py collect
```

For hard tasks that often get truncated (complex tool chains, long outputs):
```bash
EXAMPLES_PER_CALL=1 python topup.py run   # generate one at a time (safer)
```

### Step 5: Verify final counts

```bash
python topup.py count    # all tasks should be at or above target
python inspect_data.py stats
```

---

## Fine-Tuning Pipeline

Run on the pod (requires GPU). The agentic loop runs these automatically, but you can also run them manually.

### Step 1: Prepare SFT data

```bash
cd /root/pbm
PYTHONPATH=/root/pbm python -m stages.prepare
```

Converts `train.jsonl` and `val.jsonl` from the raw agent-trace format into the ChatML SFT format that Unsloth expects. Output: `train_sft.jsonl` and `val_sft.jsonl`.

### Step 2: Fine-tune

```bash
python -m stages.finetune --dry-run   # sanity check: loads model, prints config, exits
python -m stages.finetune             # actual fine-tuning (takes 30–90 min on A100)
```

Uses Unsloth's LoRA implementation. Key behavior:
- Downloads `Qwen/Qwen3.5-9B` from HuggingFace on first run (~18 GB). Subsequent runs use the local cache.
- Saves LoRA adapter to `{workspace}/{model_name}/`
- Saves merged 16-bit model to `{workspace}/{model_name}_merged/`
- Prints loss every few steps — final loss around 0.8–1.2 is typical

### Step 3: Convert to GGUF

```bash
python -m stages.convert
```

Uses llama.cpp (bundled with Unsloth) to quantize the merged 16-bit model to GGUF. Output: `{workspace}/{model_name}_merged_gguf/{model_name}_merged.Q4_K_M.gguf`

**Important:** Unsloth saves the GGUF in a directory called `{model_name}_merged_gguf`, not in the main output directory. `config.py` accounts for this — do not change the path logic.

### Step 4: Register with Ollama

```bash
bash scripts/fix_modelfile.sh
```

This is a critical step that's easy to get wrong. When you run `ollama create` with a plain `FROM <gguf>` Modelfile, the tool-calling template is not included — and without it, the model silently ignores all tool calls and scores near 0% on PinchBench.

`fix_modelfile.sh` solves this by:
1. Reading the full Modelfile from `qwen3:8b` (the base model with correct tool-call support)
2. Replacing the `FROM` line to point at your fine-tuned GGUF
3. Re-creating the Ollama model with `ollama create`

You must re-run this script after every new GGUF. The agentic loop's TrainerAgent does this automatically.

To register a specific versioned model name:
```bash
OLLAMA_MODEL=qwen35-9b-clawd-v4 bash scripts/fix_modelfile.sh
```

### Step 5: Verify

```bash
ollama list                                    # should show your model
ollama run qwen35-9b-clawd-v4 "Hello"         # basic sanity check
```

---

## Running the Benchmark

### Prerequisites

- Ollama running (`bash scripts/startup.sh`)
- OpenClaw gateway running (also started by `startup.sh`)
- PinchBench scripts at `$PBM_WORKSPACE/skill/`
- Model registered in Ollama

### Run benchmark

```bash
bash scripts/benchmark_run.sh ollama/qwen35-9b-clawd-v4
```

This:
1. Validates Ollama and OpenClaw are running
2. Checks the model is registered
3. Runs the benchmark via `$PBM_WORKSPACE/skill/scripts/run.sh`
4. Saves full log to `$PBM_WORKSPACE/logs/bench_ollama_qwen35-9b-clawd-v4.log`
5. Prints a score summary

To run without uploading to the leaderboard:
```bash
bash scripts/benchmark_run.sh ollama/qwen35-9b-clawd-v4 --no-upload
```

### Monitoring

```bash
# Tail benchmark output live (it runs for ~20–30 min)
tail -f $PBM_WORKSPACE/logs/bench_ollama_qwen35-9b-clawd-v4.log

# OpenClaw gateway logs (tool call routing, errors)
tail -f /tmp/openclaw-gateway.log

# Ollama inference logs
tail -f /tmp/ollama.log
```

---

## Agentic Loop

The loop automates the full iteration cycle: evaluate current model → analyze failures → generate more data → curate → train → repeat.

### Architecture

```
loop.py
  │
  ├── EvalAgent          — runs PinchBench, parses scores per task
  ├── EvalAnalysisAgent  — queries all model versions via Ollama, asks Claude to
  │                        generate hypotheses about regressions, writes diagnosis
  ├── DataAgent          — calls topup.py for tasks below weak_task_threshold
  ├── CuratorAgent       — runs llm_judge + filter to maintain quality gate
  └── TrainerAgent       — prepare → finetune → convert → register versioned model
```

State is persisted to `$PBM_WORKSPACE/data/loop_state.json` after every stage. If the pod crashes mid-run, restart and the loop resumes from where it left off (eval is skipped if already done for the current model version).

### Starting the loop

**From scratch (no model yet):**
```bash
cd /root/pbm
PYTHONPATH=. python3 loop.py run
```
The first iteration will benchmark the base model, then train v1.

**Resuming with an existing fine-tuned model:**
```bash
PYTHONPATH=. python3 loop.py run --model qwen35-9b-clawd-v3
```
This seeds the state with v3 as the current model. The loop will benchmark v3 (if not already done), then train v4.

**Seeding scores from an existing benchmark log (skips re-running eval):**
```bash
PYTHONPATH=. python3 loop.py run \
  --model qwen35-9b-clawd-v3 \
  --log $PBM_WORKSPACE/logs/bench_ollama_qwen35-9b-clawd-v3.log
```

**Check status:**
```bash
PYTHONPATH=. python3 loop.py status
```

### Loop control flow

The loop pauses (exit code 3) and requires human review when:
- Score drops >5% below the best version ever achieved (regression detected)
- No improvement for 3 consecutive iterations
- **EvalAnalysisAgent fails or produces a deficient diagnosis** — see gate details below
- DataAgent generates 0 new examples (all tasks already at target)
- `train.jsonl` is empty after curation
- TrainerAgent fails to produce a verified Ollama model

Hard failures (eval parse error, training crash, GGUF not found) exit with code 1 and print the error. Fix the issue and re-run — state is saved so you don't lose progress.

**Training is gated on a valid analysis.** Fine-tuning is expensive — the loop will not reach DataAgent or TrainerAgent unless `state.last_analysis` passes all of these checks:
1. Non-empty (agent produced output)
2. `summary` does not start with `"Error:"` (diagnosis itself did not error)
3. At least one of `root_causes` or `data_fixes` is non-empty (actionable content exists)

If any check fails, the loop pauses with a clear message. Fix the underlying issue (check `eval_analysis_*.json` in the data dir), then resume normally.

### Iteration flow in detail

**Iteration N:**
1. **EvalAgent** — runs `benchmark_run.sh` against `current_ollama_model`, saves log to `$PBM_WORKSPACE/logs/`, parses task scores. Updates `state.weak_tasks` (tasks below `weak_task_threshold`). Skipped if `eval_version == model_version`.

2. **Regression gate** — if score dropped >5% below best ever, pause for review.

3. **Target gate** — if avg score ≥ `target_score`, loop exits successfully.

4. **EvalAnalysisAgent** — discovers all registered model versions in Ollama, sends targeted prompts to each to observe behavioral differences, asks Claude to hypothesize root causes. Writes a diagnosis JSON to `$PBM_WORKSPACE/data/eval_analysis_*.json`. Populates `state.failure_analysis` for DataAgent.

4b. **Analysis gate** — pauses if: agent threw an exception, `last_analysis` is empty, summary starts with `"Error:"`, or both `root_causes` and `data_fixes` are empty. Training does not proceed without a real diagnosis.

5. **DataAgent** — calls `topup.py` for each task in `state.weak_tasks`, targeting `examples_per_weak_task` new examples per task. Waits for Claude Batch API, collects results.

6. **Gate** — if no new examples were added, pause (all tasks may already be at target).

7. **CuratorAgent** — runs `llm_judge.py run` then `llm_judge.py filter --min 3` to score all examples and discard low-quality ones.

8. **Gate** — if `train.jsonl` is empty after curation, pause.

9. **TrainerAgent** — increments `model_version` to N+1, runs `stages.prepare` → `stages.finetune` → `stages.convert` → `fix_modelfile.sh`, verifies the new model is in `ollama list`. Previous versions remain registered so EvalAnalysisAgent can compare them.

10. Next iteration begins with eval of the new model.

### Recommended iteration settings

Start conservative:
```yaml
loop:
  max_iterations: 2    # verify one full cycle works before increasing
  target_score: 0.85
```

After verifying end-to-end works, increase to 5–10 iterations for a longer run.

---

## Scripts Reference

### `startup.sh` — run every session

```bash
bash /root/scripts/startup.sh
```

Does everything needed to start a working session:
1. Loads API keys from `/workspace/synthbench/set_env.sh`
2. Clones repo to `/root/pbm` (or `git pull` if already cloned)
3. Kills any stale `ollama` or `openclaw` processes
4. Generates `~/.openclaw/openclaw.json` from `openclaw_template.json` + env vars
5. Starts Ollama (`ollama serve`) — waits up to 30s for it to be ready
6. Starts OpenClaw gateway (`openclaw gateway --port 18789`) — waits up to 60s
7. Auto-registers the fine-tuned model from `loop_state.json` (so you don't need to manually re-run `fix_modelfile.sh` after a pod restart)
8. Prints a health summary

**Important:** Ollama's model registry (`~/.ollama`) is on the container's ephemeral disk, not the network volume. GGUF files are on the network volume. After a pod restart, Ollama has no models registered — `startup.sh` handles re-registration automatically using `loop_state.json` to find the current model.

### `benchmark_run.sh` — run a benchmark

```bash
bash scripts/benchmark_run.sh ollama/qwen35-9b-clawd-v4 [--no-upload]
```

Saves logs to `$PBM_WORKSPACE/logs/bench_*.log` (persistent, on network volume).

### `fix_modelfile.sh` — register GGUF with correct Modelfile

```bash
bash scripts/fix_modelfile.sh                              # uses config.yaml defaults
OLLAMA_MODEL=qwen35-9b-clawd-v4 bash scripts/fix_modelfile.sh  # specific name
GGUF_PATH=/path/to/custom.gguf bash scripts/fix_modelfile.sh   # custom path
```

Must be run after every new GGUF. The script:
1. Reads `GGUF_PATH` and `MODEL_NAME` from config (or env var overrides)
2. Copies the full Qwen3 chat template from `qwen3:8b` base model
3. Creates a new Modelfile pointing `FROM` at the GGUF
4. Runs `ollama rm` + `ollama create` + `ollama run` test

### `set_env.sh` — API key template

```bash
# Copy to network volume, fill in keys:
cp /root/pbm/scripts/set_env.sh /workspace/synthbench/set_env.sh
vim /workspace/synthbench/set_env.sh

# Load manually:
source /workspace/synthbench/set_env.sh
```

---

## Key Insights and Gotchas

### 1. OpenRouter is required for benchmarking — Anthropic API alone is not enough

The PinchBench judge uses Claude to grade model responses. This judge is configured inside OpenClaw's `openclaw.json` and calls Claude via **OpenRouter** (`OPENROUTER_API_KEY`). Your `ANTHROPIC_API_KEY` is only used by the data generation scripts in this repo (`generate.py`, `llm_judge.py`, `topup.py`).

Get an OpenRouter key at [openrouter.ai](https://openrouter.ai). Add credits — each benchmark run costs roughly $1–3 in judge calls.

### 2. GGUF models need a custom Modelfile for tool calling

When you quantize a fine-tuned model to GGUF and do a plain `ollama create FROM <gguf>`, Ollama generates a minimal Modelfile that does not include the `<tools>...</tools>` block in the chat template. Without this block, the model receives no tool definitions at inference time — it will chat normally but never call any tools. On PinchBench this results in near-zero scores.

Fix: always use `fix_modelfile.sh` instead of a plain `ollama create`. The script copies the complete chat template from `qwen3:8b` (which Ollama ships with correct tool support).

### 3. Node.js 22 is required for OpenClaw

OpenClaw requires Node.js ≥22.12.0. If you install Node 20 (the default in many Docker images), `npm install -g openclaw@latest` will succeed but OpenClaw will fail at runtime with a cryptic error. The `Dockerfile` explicitly installs Node 22 via nodesource.

If you're on an old pod with Node 20, upgrade manually:
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
npm install -g openclaw@latest
openclaw --version   # verify: should print a version ≥22.12.0-compatible
```

### 4. `~/.ollama` is ephemeral — models must be re-registered after pod restart

The pod's container disk (where `~/.ollama` lives) is wiped when the pod stops. Your GGUF file is safe on the network volume. But after a restart, `ollama list` will be empty. `startup.sh` handles this automatically by reading the current model from `loop_state.json` and calling `fix_modelfile.sh`. This is why you should always run `startup.sh` at the start of every session.

### 5. Always use `source`, never `sh`, for env files

```bash
source /workspace/synthbench/set_env.sh    # correct — exports to current shell
sh /workspace/synthbench/set_env.sh        # WRONG — exports into a subshell that exits
```

### 6. PYTHONPATH must point to `/root/pbm`

All imports like `from utils.config import load_config` assume the project root is on the Python path. Always run:
```bash
cd /root/pbm
PYTHONPATH=. python3 loop.py run ...
```
Or set it persistently: the `startup.sh` writes `export PYTHONPATH=/root/pbm` to `~/.bashrc`.

### 7. Model version numbers are tracked and must be seeded correctly

TrainerAgent increments `state.model_version` by 1 each run. If you start the loop with `--model qwen35-9b-clawd-v3`, the loop parses `-v3` and sets `state.model_version = 3` so the next fine-tune becomes v4. If you don't pass `--model`, the loop starts from version 0 and will create v1 — even if v3 already exists.

### 8. Training is gated on a valid analysis — analysis failures pause the loop

EvalAnalysisAgent failures (exception, empty output, error summary, or no actionable findings) trigger a pause before DataAgent runs. The loop will not proceed to data generation or training without a real diagnosis.

This is intentional cost control: fine-tuning on the wrong data wastes ~$10–30 in GPU time. Fix the analysis error, then resume with `PYTHONPATH=. python3 loop.py run --model <current-model>`. The loop picks up from where it left off.

### 9. Benchmark logs are saved to the network volume

Logs go to `$PBM_WORKSPACE/logs/bench_*.log` (not `/tmp`). They persist across pod restarts and can be used to seed the loop with `--log`. Never delete them — EvalAnalysisAgent reads them for signal collection.

### 10. The datacenter region matters

RunPod network volumes are region-specific. If you create a volume in CA-2 and then deploy a pod in EU-RO-1, the volume won't be available. Always create pods in the same region as your volume.

---

## PinchBench Task Reference

| Task | Name | External Dependency |
|------|------|---------------------|
| task_00 | sanity | none |
| task_01 | calendar | none (creates .ics file) |
| task_02 | stock | Brave web search |
| task_03 | blog | none |
| task_04 | weather | wttr.in (free, no key) |
| task_05 | summary | none |
| task_06 | events | Brave web search |
| task_07 | email | none |
| task_08 | memory | none |
| task_09 | files | none |
| task_10 | workflow | none |
| task_11 | config_update | none |
| task_12 | skill_search | none |
| task_13 | image_gen | OpenRouter (OPENROUTER_API_KEY) |
| task_14 | humanizer | none |
| task_15 | daily_summary | none |
| task_16 | email_triage | none |
| task_17 | email_search | none |
| task_18 | market_research | Brave web search |
| task_19 | spreadsheet_summary | pandas + openpyxl |
| task_20 | eli5_pdf | pdfplumber + PyPDF2 |
| task_21 | openclaw_comprehension | none |
| task_22 | second_brain | none |

Tasks 02, 06, 18 require `BRAVE_API_KEY`. Task 13 requires `OPENROUTER_API_KEY`. Tasks 19–21 require Python packages (included in the Dockerfile).

---

## Troubleshooting

### OpenClaw gateway not starting

```bash
cat /tmp/openclaw-gateway.log
```

Common causes:
- `OPENCLAW_GATEWAY_TOKEN` not set → gateway starts but auth fails
- `OPENROUTER_API_KEY` missing → gateway starts but LLM judge fails at runtime
- Port 18789 already in use → `startup.sh` kills stale processes first; if still stuck, `kill -9 $(lsof -t -i:18789)`

### "Model not found" in benchmark

```bash
ollama list   # is the model registered?
bash scripts/fix_modelfile.sh   # re-register if missing
```

Remember: `~/.ollama` is wiped on pod restart. Always run `startup.sh`.

### Benchmark scores all 0.0 (tools not being called)

The model is loading but ignoring tool definitions. Almost certainly a Modelfile issue:
```bash
OLLAMA_MODEL=qwen35-9b-clawd-v4 bash scripts/fix_modelfile.sh
```

Verify with a manual tool-call test:
```bash
ollama run qwen35-9b-clawd-v4 "Search for the latest news on AI."
# Should produce a <tool_call> block, not a plain text response
```

### `ModuleNotFoundError: No module named 'utils'`

PYTHONPATH not set. Run from `/root/pbm`:
```bash
cd /root/pbm && PYTHONPATH=. python3 loop.py run
```

### `config.yaml not found`

Scripts must be run from the project root (`/root/pbm`). `config.py` walks up the directory tree looking for `config.yaml`. If you're in `/root/` or somewhere else, it won't find it.

### `GGUF not found` during fix_modelfile.sh or TrainerAgent

The conversion step didn't complete successfully, or `config.yaml` has the wrong path. Verify:
```bash
python3 -c "from utils.config import load_config; print(load_config().gguf_file)"
ls -lh $(python3 -c "from utils.config import load_config; print(load_config().gguf_dir)")
```

### Loop seeded wrong version → TrainerAgent creates v1 instead of vN+1

Always pass `--model` when resuming:
```bash
PYTHONPATH=. python3 loop.py run --model qwen35-9b-clawd-v3
```

Check current state:
```bash
PYTHONPATH=. python3 loop.py status
```

### Analysis JSON parse error

Fixed in the codebase. If you see `Expecting ',' delimiter` from EvalAnalysisAgent, pull the latest code:
```bash
git -C /root/pbm pull
```

### `sh: openclaw: not found` after startup

OpenClaw binary is at `~/.openclaw/bin/openclaw` or `/root/.openclaw/bin/openclaw`. Startup.sh adds this to PATH. If it's missing entirely, reinstall:
```bash
npm install -g openclaw@latest
openclaw --version
```

---

## Useful Paths

| Path | Purpose |
|------|---------|
| `/workspace/synthbench/data/train.jsonl` | Training data |
| `/workspace/synthbench/data/val.jsonl` | Validation data |
| `/workspace/synthbench/data/scores.json` | LLM judge scores |
| `/workspace/synthbench/data/loop_state.json` | Agentic loop state (persisted) |
| `/workspace/synthbench/logs/bench_*.log` | Benchmark logs (persisted) |
| `/workspace/synthbench/{model_name}_merged_gguf/` | GGUF model files (persisted) |
| `/workspace/synthbench/skill/` | PinchBench benchmark scripts |
| `~/.openclaw/openclaw.json` | Generated OpenClaw config |
| `/tmp/openclaw-gateway.log` | OpenClaw gateway logs (ephemeral) |
| `/tmp/ollama.log` | Ollama logs (ephemeral) |
| `/root/pbm/` | Cloned project code |
