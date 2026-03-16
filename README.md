# PinchBench Maxing

A multi-agent system that autonomously fine-tunes an open-source LLM to compete on [PinchBench](https://pinchbench.com) — a 23-task benchmark for AI agents.

Five specialized agents work together in a loop: one benchmarks the model, one diagnoses failures using Claude, one generates targeted training data, one curates quality, one trains. Each agent has a single, well-defined responsibility. They communicate through a shared `AgentState` that persists to disk — so the system survives pod restarts and can be resumed at any stage. No agent needs to know what the others do internally; they only read from and write to the shared state.

The loop runs overnight without human intervention. This is the same idea as [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): give an AI system a metric to optimize and let it drive the improvement cycle while you sleep. The difference here is that instead of modifying training code, the agents modify the *training data* — using Claude to understand what the model gets wrong and synthesize examples that target exactly those gaps.

---

## Results

| Model | Score | Notes |
|-------|-------|-------|
| qwen3:8b (base) | 3.1 / 23 (13%) | No fine-tuning |
| qwen35-9b-clawd-v1 | 9.9 / 23 (43%) | First fine-tune, ~900 examples |
| qwen35-9b-clawd-v3 | 16.8 / 23 (73%) | ~1,400 examples, targeted topup |
| qwen35-9b-clawd-v4 | in progress | Agentic loop iteration |

---

## How the Agentic Loop Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        loop.py                                  │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│   │  Eval    │───▶│ Analysis │───▶│   Data   │                 │
│   │  Agent   │    │  Agent   │    │  Agent   │                 │
│   └──────────┘    └──────────┘    └────┬─────┘                 │
│        ▲                               │                        │
│        │          ┌──────────┐    ┌────▼─────┐                 │
│        └──────────│ Trainer  │◀───│ Curator  │                 │
│                   │  Agent   │    │  Agent   │                 │
│                   └──────────┘    └──────────┘                 │
└─────────────────────────────────────────────────────────────────┘
         State: loop_state.json (persisted across restarts)
```

Each agent has one job. They hand off through `AgentState`:

**1. EvalAgent** — benchmarks the current model against all 23 PinchBench tasks. Writes per-task scores and a list of `weak_tasks` (tasks below `weak_task_threshold`) to state. Skips automatically if eval has already been run for the current model version — no redundant benchmarking.

**2. EvalAnalysisAgent** — this is where Claude becomes the brain. It reads the scores from state, discovers all registered model versions in Ollama, and sends targeted probe prompts to each version to observe behavioral differences. It then asks Claude to form hypotheses, test them, and produce a structured diagnosis: root causes, affected tasks, and concrete data fixes. The diagnosis is written back to `state.last_analysis` and `state.failure_analysis` for the next agent to consume. This is not a script deciding "score < threshold → add data" — it is Claude reasoning about *why* the model fails.

**3. DataAgent** — reads `state.weak_tasks` and `state.failure_analysis` to decide what to generate *and how*. For each weak task, it selects one of three strategies:
- **Targeted topup** (default) — writes the diagnosis to `current_diagnosis.json`, passes it to `targeted_topup.py`. The meta-prompt includes the root cause and fix recommendation. Variation types (error_recovery, multi_tool_chain, etc.) are weighted based on failure patterns — not round-robin.
- **Adversarial generation** — for tasks that scored 0.0 (complete failure). Parses the model's actual benchmark transcript to see what it did wrong, then generates examples showing the correct approach to that exact scenario. Uses real-time Claude API since volume is low (~3 examples per failed task).
- **Plain topup** (fallback) — original round-robin when no diagnosis is available.

**4. CuratorAgent** — five-stage quality pipeline:
1. **Score** all examples via LLM judge (1–5 scale)
2. **Repair** borderline examples (score 2–3) — sends them to Claude with the judge's feedback to fix specific issues while preserving structure, then re-scores. Only keeps repairs that actually improved.
3. **Filter** below `min_judge_score` — removes the rest
4. **Deduplicate** — TF-IDF cosine similarity on user messages + Jaccard similarity on tool calls. Removes near-identical examples, keeping the highest-scored from each cluster.
5. **Verify** train.jsonl is non-empty

**5. TrainerAgent** — reads the curated dataset and runs the full training pipeline: prepare → finetune → convert → `register_model.sh`. Registers the result in Ollama as a versioned model (`qwen35-9b-clawd-v4`, etc.) and records it in `state.model_history`. Previous versions stay registered so EvalAnalysisAgent can probe and compare them in future iterations.

**Gates block the loop before expensive stages.** Agents cannot proceed to training on bad inputs:
- EvalAnalysisAgent failure or empty diagnosis → **PAUSE** *(training never starts without a real diagnosis)*
- Score regression >5% below best ever → **PAUSE**
- No improvement for 3 consecutive iterations → **PAUSE**
- DataAgent generates 0 new examples → **PAUSE**
- `train.jsonl` empty after curation → **PAUSE**

State is saved after every stage. Pod crashes mid-run? Resume with the same command — the loop picks up where it left off. Each agent is also independently runnable (`python -m agents.eval_agent`, etc.) for debugging.

### Parallel to Karpathy's AutoResearch

| Karpathy autoresearch | PinchBench Maxing |
|---|---|
| Single agent modifies `train.py` | Five specialized agents, each with one responsibility |
| Agent modifies training code | Agents modify training *data* |
| Fixed 5-min experiment budget | Full fine-tune per iteration |
| Metric: val bits-per-byte | Metric: PinchBench score (0–23 tasks) |
| Runs overnight on a single GPU | Runs overnight on a single GPU |
| Agent reads eval output, iterates | EvalAnalysisAgent uses Claude to *reason* about why tasks fail |
| `program.md` guides the agent | Shared `AgentState` coordinates agents; Claude provides diagnosis |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ChetanTekur/pinchbenchmaxing && cd pinchbenchmaxing

# 2. Fill in API keys — copy template to persistent storage and edit
cp scripts/set_env.sh /workspace/synthbench/set_env.sh
vim /workspace/synthbench/set_env.sh

# 3. Start services (sources set_env.sh, starts Ollama + OpenClaw)
bash scripts/startup.sh

# 4. Register with PinchBench (one time only)
cd $PBM_WORKSPACE/skill && bash scripts/run.sh --register && cd -

# 5. Generate initial training data (cheapest to run locally)
python generate.py submit && python generate.py status && python generate.py collect
python llm_judge.py run && python llm_judge.py filter --min 3

# 6. Run the agentic loop — this is the main entry point
python3 loop.py run
```

That's it. The loop evaluates, diagnoses, generates data, curates, and trains — uploading results to PinchBench automatically on each iteration.

---

## GPU Options

This project requires a GPU for fine-tuning. You can use any of the following:

### Cloud GPU providers

| Provider | Notes |
|----------|-------|
| **[RunPod](https://runpod.io)** | Network volumes persist data across pod restarts. Recommended. |
| **[Vast.ai](https://vast.ai)** | Often cheapest for spot instances. Set `PBM_WORKSPACE` to a mounted volume. |
| **[Lambda Labs](https://lambdalabs.com)** | Reliable, fixed pricing. Persistent filesystems available. |
| **[Google Colab Pro+](https://colab.research.google.com)** | A100 available; mount Google Drive as workspace. Short session limits. |

A GPU with **≥20 GB VRAM** is recommended (RTX 4090 / A100). Qwen3.5-9B with LoRA fits in 20 GB.

**Local CUDA:** Works directly. Set `PBM_WORKSPACE` to a local directory and install dependencies from the Dockerfile or via `scripts/setup_pod.sh`.

**Apple Silicon / Mac:** This is a future goal and has not been tested yet. Unsloth has experimental MPS support and the training stages use standard HuggingFace/TRL primitives, so it may work. If you try it, we'd love to hear how it goes — open an issue or PR.

### RunPod-specific setup

If using RunPod:

1. **Network Volume** — create one in your target region (e.g. CA-2), 200 GB, mounted at `/workspace`
2. **Pod** — deploy with Docker image `ghcr.io/chetantekur/pinchbenchmaxing:latest`, attach the volume, ≥24 GB VRAM GPU
3. **Region** — pod and volume must be in the same datacenter
4. **Container disk** — 50 GB minimum (model cache + GGUF)

The Docker image (`Dockerfile`) contains PyTorch 2.6 + CUDA 12.4, Unsloth, llama.cpp, Ollama, Node.js 22, and OpenClaw. It does **not** contain the project code — `startup.sh` clones this repo to `/root/pbm` on first start and `git pull`s on subsequent restarts, so code updates only require a pod restart.

---

## API Keys

| Key | Purpose |
|-----|---------|
| `ANTHROPIC_API_KEY` | Generating synthetic training data (`generate.py`, `llm_judge.py`, `topup.py`, `eval_analysis_agent.py`) |
| `OPENROUTER_API_KEY` | **PinchBench LLM judge** (claude-opus-4.5 via OpenRouter), image generation task |
| `BRAVE_API_KEY` | Web search tasks (task_02, task_06, task_18) |
| `OPENCLAW_GATEWAY_TOKEN` | OpenClaw gateway auth (any random string) |

**Critical:** The PinchBench judge calls Claude through **OpenRouter**, not the Anthropic API directly. `OPENROUTER_API_KEY` is required for benchmarking — `ANTHROPIC_API_KEY` alone is not sufficient. See [Key Insights](#key-insights--gotchas) for details.

### Persisting keys across restarts

```bash
# Copy the template to your persistent storage, fill in values
cp scripts/set_env.sh /workspace/synthbench/set_env.sh
vim /workspace/synthbench/set_env.sh
```

`startup.sh` sources this file automatically on every pod start. Always use `source`, not `sh`:
```bash
source /workspace/synthbench/set_env.sh    # correct
sh /workspace/synthbench/set_env.sh        # WRONG — exports into a subshell that exits
```

---

## Configuration

All settings live in `config.yaml`. `utils/config.py` is a thin Python loader that reads `config.yaml` and computes derived paths — it is not a separate config file.

```yaml
model:
  base: Qwen/Qwen3.5-9B       # HuggingFace model to fine-tune
  name: qwen35-9b-clawd        # used for output dirs + Ollama model name

paths:
  workspace: ${PBM_WORKSPACE:-./workspace}  # override with env var

data:
  examples_per_task: 70        # target training examples per task
  val_split: 0.1               # held out for validation
  min_judge_score: 3           # discard examples below this (1–5 scale)

training:
  epochs: 3
  batch_size: 2
  grad_accum: 4                # effective batch = batch_size × grad_accum
  learning_rate: 2e-4
  lora_r: 16
  lora_alpha: 32
  max_seq_len: 4096

convert:
  quantization: q4_k_m         # q4_k_m (default), q8_0, f16

claude:
  generation: claude-sonnet-4-5   # model for synthetic data generation
  judge: claude-sonnet-4-5        # model for quality scoring
  analysis: claude-sonnet-4-6     # model for failure diagnosis

loop:
  max_iterations: 2            # start low, verify end-to-end, then increase
  target_score: 0.85           # loop exits when benchmark avg exceeds this
  weak_task_threshold: 0.50    # tasks below this trigger data generation
  examples_per_weak_task: 20   # new examples per weak task per iteration
```

**Derived paths** (all relative to `workspace`):

| Config property | Resolves to |
|----------------|-------------|
| `data_dir` | `{workspace}/data/` |
| `adapter_dir` | `{workspace}/{model_name}/` |
| `merged_dir` | `{workspace}/{model_name}_merged/` |
| `gguf_dir` | `{workspace}/{model_name}_merged_gguf/` |
| `gguf_file` | `{workspace}/{model_name}_merged_gguf/{model_name}_merged.Q4_K_M.gguf` |
| `train_file` | `{workspace}/data/train.jsonl` |
| `val_file` | `{workspace}/data/val.jsonl` |

Override workspace: `export PBM_WORKSPACE=/your/path`

---

## Project Structure

```
config.yaml                  # all settings — single source of truth
loop.py                      # agentic loop orchestrator

agents/
  base.py                    # AgentState (shared state between all agents), Agent base class, TASK_IDS registry
  eval_agent.py              # benchmarks current model, parses scores, identifies weak tasks
  eval_analysis_agent.py     # probes model versions via Ollama, diagnoses regressions with Claude
  data_agent.py              # calls topup.py for weak tasks, drives Claude Batch API
  curator_agent.py           # llm_judge quality gate — scores + filters training data
  trainer_agent.py           # prepare → finetune → convert → register versioned Ollama model

stages/
  prepare.py                 # convert train.jsonl → SFT chat format for Unsloth
  finetune.py                # LoRA fine-tuning with Unsloth (supports --dry-run)
  convert.py                 # export merged model → GGUF (llama.cpp quantization)
  probe.py                   # interactive model testing from merged weights (no GGUF needed)

scripts/
  startup.sh                 # every-session: kill stale procs, start Ollama + OpenClaw, health check
  benchmark_run.sh           # run PinchBench for a given model, save log to network volume
  register_model.sh          # register GGUF in Ollama with correct tool-calling Modelfile
  set_env.sh                 # API key template — copy to persistent storage and fill in
  setup_pod.sh               # one-time pod setup (if not using the Docker image)
  check_setup.sh             # pre-benchmark checker: APIs, deps, services
  train_and_eval.sh          # manual pipeline: prepare → finetune → convert → benchmark
  patch_openclaw_image_gen.py  # remove invalid 'image' key from openclaw.json

utils/
  config.py                  # loads config.yaml, computes all derived paths
  prompts.py                 # shared constants: OPENCLAW_SYSTEM prompt, VALID_TOOLS

generate.py                  # generate initial synthetic training data via Claude Batch API
llm_judge.py                 # score examples 1–5 with Claude, filter at --min 3
topup.py                     # fill per-task gaps (plain round-robin, used as fallback)
targeted_topup.py            # diagnosis-aware generation — weighted variations, injected context
adversarial_gen.py           # generate from benchmark failure transcripts
example_repair.py            # repair borderline examples (score 2-3) instead of discarding
dedup.py                     # semantic deduplication (TF-IDF + tool Jaccard similarity)
inspect_data.py              # dataset stats, validation, sampling
test_tool_call.sh            # verify fine-tuned model emits proper <tool_call> blocks
openclaw_template.json       # OpenClaw config template (secrets injected by startup.sh)
Dockerfile                   # full image: PyTorch + Unsloth + Ollama + OpenClaw
Dockerfile.bench             # lightweight image: Ollama + OpenClaw only (no training)
```

---

## Data Generation Pipeline

Run locally (Anthropic Batch API calls are ~50% cheaper and there's no GPU requirement here).

### 1. Generate initial data

Make sure `set_env.sh` is sourced (it sets `ANTHROPIC_API_KEY` and `PBM_WORKSPACE`), then from the repo root:

```bash
python generate.py submit    # submit ~920 examples to Claude Batch API
python generate.py status    # poll until complete
python generate.py collect   # write train.jsonl + val.jsonl
```

Generates ~40 synthetic agent traces per PinchBench task (23 tasks). Each trace is a multi-turn conversation where Clawd uses tools to complete a real task.

### 2. Score and filter

```bash
python llm_judge.py run             # score all examples 1–5 (resumes if interrupted)
python llm_judge.py report          # per-task score summary
python llm_judge.py filter --min 3  # remove low-quality examples in-place
```

### 3. Inspect and validate

```bash
python inspect_data.py stats     # examples per task, format breakdown
python inspect_data.py validate  # structural issues
python inspect_data.py sample    # preview random examples
```

### 4. Top up weak tasks

```bash
python topup.py count   # show current vs target per task
python topup.py run     # generate more for tasks below target
python topup.py status  # check batch progress
python topup.py collect # download results when done
```

For tasks that tend to produce truncated examples:
```bash
EXAMPLES_PER_CALL=1 python topup.py run   # one example at a time (avoids truncation)
```

---

## Fine-Tuning Pipeline (Advanced / Manual)

> **Normal path:** `python3 loop.py run` — TrainerAgent handles all of this automatically.
>
> This section is for debugging individual stages or running a one-off fine-tune outside the loop.

### 1. Prepare SFT data

```bash
python -m stages.prepare
```

Converts `train.jsonl` / `val.jsonl` from agent-trace format to ChatML SFT format for Unsloth.

### 2. Fine-tune

```bash
python -m stages.finetune --dry-run   # load model, print config, exit
python -m stages.finetune             # 30–90 min on A100
```

Downloads `Qwen/Qwen3.5-9B` from HuggingFace on first run (~18 GB). Saves LoRA adapter and merged 16-bit weights to the workspace.

### 3. Convert to GGUF

```bash
python -m stages.convert
```

Quantizes the merged model to GGUF using llama.cpp (pre-installed in the Docker image). Output: `{workspace}/{model_name}_merged_gguf/{model_name}_merged.Q4_K_M.gguf`.

### 4. Register with Ollama

```bash
bash scripts/register_model.sh

# Or with explicit version name:
OLLAMA_MODEL=qwen35-9b-clawd-v4 bash scripts/register_model.sh
```

**This step is critical** — see [Key Insights](#key-insights--gotchas) for why a plain `ollama create` will silently break tool calling.

### 5. Verify

```bash
ollama list
bash test_tool_call.sh   # verify <tool_call> blocks are being emitted
```

### Manual full pipeline (non-agentic)

```bash
# Train and benchmark in one shot:
OLLAMA_MODEL=qwen35-9b-clawd-v4 bash scripts/train_and_eval.sh
```

---

## Running the Benchmark

```bash
# Run startup first (every session)
bash scripts/startup.sh

# Check everything is ready
bash scripts/check_setup.sh

# Run benchmark — uploads to PinchBench leaderboard by default
bash scripts/benchmark_run.sh ollama/qwen35-9b-clawd-v4

# Skip leaderboard upload (local testing only)
bash scripts/benchmark_run.sh ollama/qwen35-9b-clawd-v4 --no-upload
```

Logs saved to `$PBM_WORKSPACE/logs/bench_*.log` (persistent).

```bash
# Monitor live
tail -f $PBM_WORKSPACE/logs/bench_ollama_qwen35-9b-clawd-v4.log

# Service logs
tail -f /tmp/openclaw-gateway.log
tail -f /tmp/ollama.log
```

---

## Agentic Loop

### Starting

```bash
cd /root/pbm   # or wherever you cloned the repo
python3 loop.py run
```

**Resuming with an existing model** (seeds version number so TrainerAgent creates the right next version):
```bash
python3 loop.py run --model qwen35-9b-clawd-v3
```

**Seeding scores from an existing log** (skips re-running the benchmark):
```bash
python3 loop.py run \
  --model qwen35-9b-clawd-v3 \
  --log $PBM_WORKSPACE/logs/bench_ollama_qwen35-9b-clawd-v3.log
```

**Check status:**
```bash
python3 loop.py status
```

### Iteration flow

```
Iteration N
├── EvalAgent          — benchmark current model, parse scores
│   └── Gate: regression > 5% below best ever → PAUSE
│   └── Gate: score ≥ target → EXIT (done)
│   └── Gate: no weak tasks → EXIT
├── EvalAnalysisAgent  — probe model versions, diagnose with Claude
│   └── Gate: exception / empty / error summary / no findings → PAUSE
│       (training never proceeds without a real diagnosis)
├── DataAgent          — topup weak tasks via Claude Batch API
│   └── Gate: 0 new examples → PAUSE
├── CuratorAgent       — llm_judge score + filter
│   └── Gate: train.jsonl empty after curation → PAUSE
└── TrainerAgent       — prepare → finetune → convert → register vN+1
    └── Gate: GGUF not found / model not in ollama list → FAIL (exit 1)
```

All pause exits are code 3 (distinct from error = 1). State is saved at every gate — resume safely after fixing the issue.

### Version tracking

Each TrainerAgent run increments `model_version` and registers `{model_name}-v{N}` in Ollama. Previous versions remain registered so EvalAnalysisAgent can probe and compare behavior across versions.

When starting with `--model qwen35-9b-clawd-v3`, the loop parses `-v3` and sets `model_version = 3` internally, so the next fine-tune becomes v4. Without `--model`, the loop starts from v0 and will create v1.

---

## Scripts Reference

### `startup.sh` — run every session

```bash
bash /root/scripts/startup.sh   # if on RunPod (scripts are at /root/scripts)
bash scripts/startup.sh         # if running locally from repo root
```

1. Sources API keys from `$PBM_WORKSPACE/set_env.sh`
2. Clones repo to `/root/pbm` (or `git pull` if already present)
3. Kills stale `ollama` and `openclaw` processes
4. Generates `~/.openclaw/openclaw.json` from `openclaw_template.json` + env vars
5. Starts Ollama, waits up to 30s
6. Starts OpenClaw gateway on port 18789, waits up to 60s
7. Auto-registers the fine-tuned model from `loop_state.json` (so you don't need to manually re-run `register_model.sh` after a pod restart)
8. Prints health summary

### `benchmark_run.sh`

```bash
bash scripts/benchmark_run.sh ollama/qwen35-9b-clawd-v4              # run + upload to leaderboard
bash scripts/benchmark_run.sh ollama/qwen35-9b-clawd-v4 --no-upload  # local test, no upload
```

### `register_model.sh`

```bash
bash scripts/register_model.sh
OLLAMA_MODEL=qwen35-9b-clawd-v4 bash scripts/register_model.sh
GGUF_PATH=/custom/path.gguf bash scripts/register_model.sh
```

### `check_setup.sh`

```bash
bash scripts/check_setup.sh
```

Validates Ollama is running, OpenClaw gateway is running, all API keys are set, and Python packages are present. Run before every benchmark.

### `train_and_eval.sh`

```bash
OLLAMA_MODEL=qwen35-9b-clawd-v4 bash scripts/train_and_eval.sh
NO_UPLOAD=1 bash scripts/train_and_eval.sh   # skip leaderboard upload
```

Runs the full manual pipeline: prepare → finetune → convert → fix_modelfile → benchmark.

---

## Key Insights & Gotchas

### 1. OpenRouter is required for benchmarking — Anthropic API alone is not enough

The PinchBench judge uses Claude via **OpenRouter** (`OPENROUTER_API_KEY`). This is because the judge agent lives inside OpenClaw's configuration, not in this codebase. `ANTHROPIC_API_KEY` is only used by the data generation scripts in this repo.

You need both keys for different things. Get an OpenRouter key at [openrouter.ai](https://openrouter.ai) and add credits (~$1–3 per benchmark run for judge calls).

### 2. GGUF models need a custom Modelfile for tool calling

A plain `ollama create FROM <gguf>` generates a minimal Modelfile without the `<tools>...</tools>` chat template block. Without it, the model silently ignores all tool calls and scores near 0% on PinchBench.

`register_model.sh` fixes this by copying the full chat template from `qwen3:8b` (which ships with proper tool-call support) and pointing `FROM` at your fine-tuned GGUF. Always use `register_model.sh`, never plain `ollama create`.

### 3. Node.js 22 is required for OpenClaw

OpenClaw requires Node.js ≥22.12.0. Node 20 installs without error but OpenClaw fails at runtime. The `Dockerfile` explicitly installs Node 22 via nodesource. On an existing machine:
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs && npm install -g openclaw@latest
openclaw --version
```

### 4. `~/.ollama` is ephemeral on cloud pods — models must re-register after restart

Ollama's model registry (`~/.ollama`) lives on the container's ephemeral disk, not the network volume. The GGUF file is safe on the network volume, but `ollama list` will be empty after a restart. `startup.sh` handles re-registration automatically from `loop_state.json`.

### 5. `utils/config.py` is a loader for `config.yaml` — not a second config file

`config.yaml` is where you change settings. `utils/config.py` is Python code that reads `config.yaml` and exposes it as a typed object with computed properties (e.g. `cfg.gguf_file`). Never edit `utils/config.py` to change settings — edit `config.yaml`.

### 6. PYTHONPATH is set automatically — you don't need to prefix commands

On cloud pods, `PYTHONPATH=/root/pbm` is set as a Docker `ENV` variable and also written to `~/.bashrc` by `startup.sh`. After running `startup.sh` (or in any new SSH session), all commands work without a prefix:

```bash
python3 loop.py run          # correct — PYTHONPATH already set
PYTHONPATH=. python3 loop.py run  # unnecessary on the pod
```

If running locally without the Docker image, set it once:
```bash
cd /path/to/repo && export PYTHONPATH=$(pwd)
```

### 7. Model version numbers must be seeded when resuming

```bash
python3 loop.py run --model qwen35-9b-clawd-v3
```

Without `--model`, the loop starts from version 0 and TrainerAgent will create v1 even if v3 exists. The loop parses `-v3` from the model name and sets `model_version = 3` so the next fine-tune becomes v4.

### 8. Training is gated on a valid analysis

Fine-tuning is expensive (~$10–30 in GPU time per run). The loop will not proceed to DataAgent or TrainerAgent unless `state.last_analysis` passes:
- Non-empty output
- `summary` does not start with `"Error:"`
- At least one of `root_causes` or `data_fixes` is non-empty

If analysis fails, the loop pauses with a clear message. Fix the issue (check `eval_analysis_*.json` in the data dir), then resume.

### 9. Benchmark logs are saved to the network volume — never `/tmp`

Logs go to `$PBM_WORKSPACE/logs/bench_*.log`. They persist across pod restarts and can be used to seed the next loop run with `--log`. EvalAnalysisAgent also reads them for signal collection.

### 10. Source env files — never `sh`

```bash
source /workspace/synthbench/set_env.sh    # exports to current shell
sh /workspace/synthbench/set_env.sh        # WRONG — subshell exits, nothing exported
```

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
| task_13 | image_gen | OpenRouter (`OPENROUTER_API_KEY`) |
| task_14 | humanizer | none |
| task_15 | daily_summary | none |
| task_16 | email_triage | none |
| task_17 | email_search | none |
| task_18 | market_research | Brave web search |
| task_19 | spreadsheet_summary | pandas + openpyxl |
| task_20 | eli5_pdf | pdfplumber + PyPDF2 |
| task_21 | openclaw_comprehension | none |
| task_22 | second_brain | none |

---

## Troubleshooting

### OpenClaw gateway not starting
```bash
cat /tmp/openclaw-gateway.log
```
Common causes: `OPENCLAW_GATEWAY_TOKEN` not set, port 18789 already in use, Node 20 instead of Node 22.

### Benchmark scores all 0.0 (tools not being called)
The model is ignoring tool definitions. Almost always a Modelfile issue:
```bash
OLLAMA_MODEL=qwen35-9b-clawd-v4 bash scripts/register_model.sh
bash test_tool_call.sh   # verify tool calls are emitted
```

### `ModuleNotFoundError: No module named 'utils'`
```bash
cd /root/pbm && python3 loop.py run
```

### `config.yaml not found`
Scripts must run from the project root. `utils/config.py` walks up the directory tree for `config.yaml`.

### `GGUF not found` during convert or register_model
```bash
python3 -c "from utils.config import load_config; print(load_config().gguf_file)"
```
Verify the path matches where Unsloth actually saved the GGUF. Unsloth saves to `{model_name}_merged_gguf/`, not the main output dir.

### Loop created v1 instead of v4
Always pass `--model` when resuming with an existing fine-tuned model:
```bash
python3 loop.py run --model qwen35-9b-clawd-v3
```

### Analysis JSON parse error
```bash
git -C /root/pbm pull   # fix was shipped in a recent commit
```

---

## Useful Paths

| Path | Contents |
|------|---------|
| `$PBM_WORKSPACE/data/train.jsonl` | Training data |
| `$PBM_WORKSPACE/data/val.jsonl` | Validation data |
| `$PBM_WORKSPACE/data/scores.json` | LLM judge scores |
| `$PBM_WORKSPACE/data/loop_state.json` | Agentic loop state (persisted) |
| `$PBM_WORKSPACE/data/eval_analysis_*.json` | Per-iteration diagnosis reports |
| `$PBM_WORKSPACE/logs/bench_*.log` | Benchmark logs (persisted) |
| `$PBM_WORKSPACE/{model_name}_merged_gguf/` | GGUF files |
| `$PBM_WORKSPACE/skill/` | PinchBench benchmark scripts |
| `~/.openclaw/openclaw.json` | Generated OpenClaw config |
| `/tmp/openclaw-gateway.log` | OpenClaw gateway logs (ephemeral) |
| `/tmp/ollama.log` | Ollama logs (ephemeral) |

---

## Roadmap

- **Apple Silicon support** — test and validate the full pipeline on Mac (M1/M2/M3/M4). Unsloth has experimental MPS support; the training stages use standard HuggingFace/TRL primitives.
- **Multi-model support** — the pipeline currently targets Qwen3.5-9B. We want to make it easy to swap in other base models (Llama, Mistral, Phi, Gemma, etc.).
- **Smarter data generation** — use EvalAnalysisAgent's diagnosis to not just topup weak tasks, but rewrite bad examples and generate adversarial variants.
- **Parallel benchmarking** — run eval against multiple model versions simultaneously to speed up the loop.

---

## Contributing: Adding Support for Other Models

The pipeline is designed around Qwen3.5-9B, but adding a new base model is straightforward. Here's what to change:

### 1. `config.yaml` — point to the new model

```yaml
model:
  base: mistralai/Mistral-7B-v0.3   # HuggingFace model ID
  name: mistral-7b-clawd             # used for output dirs + Ollama model name
```

All derived paths (`adapter_dir`, `merged_dir`, `gguf_dir`, `gguf_file`) update automatically.

### 2. `stages/finetune.py` — verify compatibility

Unsloth supports a wide range of models (Llama, Mistral, Phi, Gemma, Qwen, etc.). The fine-tuning stage uses `FastLanguageModel.from_pretrained()` which handles most architectures automatically. You may need to adjust:

- `max_seq_len` in `config.yaml` — some models support longer contexts
- LoRA `target_modules` — Unsloth auto-detects these, but verify for your architecture
- Chat template — Unsloth applies the model's native template; verify it produces correct `<tool_call>` formatting

### 3. `scripts/register_model.sh` — update the Modelfile template

This is the critical step. The current Modelfile template is Qwen3-specific (uses `<|im_start|>`/`<|im_end|>` tokens and a Qwen-specific `<tools>` block). For a different model family:

1. Pull the base model in Ollama: `ollama pull mistral:7b`
2. Export its Modelfile: `ollama show mistral:7b --modelfile`
3. Verify it includes tool-call support (the `{{- if .Tools }}` template block)
4. Update `register_model.sh` to use the new model's chat template instead of copying from `qwen3:8b`

Without the correct Modelfile template, tool calling will silently fail and benchmark scores will be near zero.

### 4. `openclaw_template.json` — update the default model reference

The template uses `ollama/__MODEL_NAME__:latest` which is injected by `startup.sh`. No changes needed unless your model requires different inference parameters (temperature, top_k, etc.).

### 5. Training data — the dataset is model-agnostic

The synthetic training data (`train.jsonl`) uses a generic chat format with tool calls. `stages/prepare.py` converts it to the target model's SFT format using HuggingFace's chat template. The same dataset should work across model families — the SFT conversion handles the format differences.

### What to validate

After making these changes, run through the pipeline manually once before using the agentic loop:

```bash
python -m stages.finetune --dry-run   # verify model loads and config is valid
python -m stages.finetune             # full training run
python -m stages.convert              # GGUF conversion
bash scripts/register_model.sh        # Ollama registration with correct template
bash test_tool_call.sh                # verify tool calls are emitted
bash scripts/benchmark_run.sh ollama/mistral-7b-clawd --no-upload  # sanity benchmark
```

If you get it working with a new model, please open a PR. We'd love to expand the set of supported architectures.
| `/root/pbm/` | Project code (cloned by startup.sh) |
