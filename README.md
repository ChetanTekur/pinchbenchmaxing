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
| qwen35-9b-clawd-v3 | 16.8 / 23 (73%) | ~1,400 examples, manual topup |
| qwen35-9b-clawd-v4 | 13.6 / 23 (59%) | Regression — data pipeline bugs |
| qwen35-9b-clawd-v5 | 15.0 / 23 (65%) | New agentic pipeline, recovering |
| qwen35-9b-clawd-v6 | in progress | Score-proportional data gen |

Dataset published at [huggingface.co/datasets/ChetanTekur/pinchbench-clawd](https://huggingface.co/datasets/ChetanTekur/pinchbench-clawd) (CC BY 4.0).

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

**1. EvalAgent** — benchmarks the current model against all 23 PinchBench tasks. Writes per-task scores and a list of `weak_tasks` (tasks below `weak_task_threshold`) to state. Skips automatically if eval has already been run for the current model version — no redundant benchmarking. Each benchmark log is archived with version + timestamp so no iteration overwrites another.

**2. EvalAnalysisAgent** — this is where Claude becomes the brain. It reads the scores from state, discovers all registered model versions in Ollama, and sends targeted probe prompts to each version to observe behavioral differences. It then asks Claude to form hypotheses, test them, and produce a structured diagnosis: root causes, affected tasks, and concrete data fixes. The diagnosis is written back to `state.last_analysis` and `state.failure_analysis` for the next agent to consume. Raw Claude responses are saved to `data/debug/` for post-mortem analysis when JSON parsing fails.

**3. DataAgent** — reads the diagnosis and computes a **score-proportional generation plan**. Instead of generating a flat number of examples per task, it scales with the gap to target:

```
n = max_examples_per_task × (gap / target_score)

Score 0.00 → 100 examples (full allocation)
Score 0.18 → 79 examples
Score 0.50 → 41 examples
Score 0.70 → 18 examples
Score 0.85 → 0 (at target)
```

Total capped at `total_new_examples_cap` per iteration. Two generation strategies run for each weak task:
- **Targeted topup** — diagnosis-aware meta-prompts with weighted variation types (error_recovery, multi_tool_chain, etc. biased by failure patterns)
- **Adversarial generation** (for score-0 tasks) — parses the model's benchmark transcript to see what it actually did wrong, generates examples showing the correct approach

**4. CuratorAgent** — six-stage quality pipeline:
1. **Score** all examples via LLM judge (1–5 scale)
2. **Repair** borderline examples (score 2–3) — sends to Claude with judge feedback, re-scores, keeps only improvements
3. **Filter** below `min_judge_score`
4. **Deduplicate** — TF-IDF cosine similarity + tool-call Jaccard similarity, keeps highest-scored per cluster
5. **Verify** train.jsonl is non-empty
6. **Snapshot** to HuggingFace — versioned push of train.jsonl, val.jsonl, scores.json for public access and rollback

**5. TrainerAgent** — validates the base model (HuggingFace existence, Unsloth support, architecture check — cached after first run), then runs the full training pipeline: prepare → finetune → convert → `register_model.sh`. Registers the result in Ollama as a versioned model (`qwen35-9b-clawd-v6`, etc.).

**Gates block the loop before expensive stages:**
- Model validation fails (wrong architecture, not an LLM, etc.) → **EXIT**
- EvalAnalysisAgent failure or empty diagnosis → **PAUSE**
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

### One-time setup (on the pod)

```bash
# 1. Start a pod with ghcr.io/chetantekur/pinchbenchmaxing:latest (see GPU Options)
#    The container starts idle. SSH in, then:

# 2. Run startup — clones repo, starts Ollama + OpenClaw
bash /root/scripts/startup.sh

# 3. Set up API keys (one time — persists on network volume)
cp /root/pbm/scripts/set_env.sh /workspace/synthbench/set_env.sh
vim /workspace/synthbench/set_env.sh   # fill in all keys
source /workspace/synthbench/set_env.sh

# 4. Register with PinchBench (one time)
cd $PBM_WORKSPACE/skill && bash scripts/run.sh --register && cd -
```

### Run the agentic loop

```bash
# Always run in tmux so it survives SSH disconnects
cd /root/pbm
tmux new -s loop
python3 loop.py run
# Detach: Ctrl+B, D — reattach later: tmux attach -t loop
```

### On pod restart

```bash
bash /root/scripts/startup.sh    # re-starts Ollama + OpenClaw, re-registers model
cd /root/pbm
tmux new -s loop
python3 loop.py run              # resumes from saved state automatically
```

That's it. The loop evaluates, diagnoses, generates targeted data, curates, trains, and repeats.

---

## GPU Options

This project requires a GPU for fine-tuning. Any provider works:

| Provider | Notes |
|----------|-------|
| **[RunPod](https://runpod.io)** | Network volumes persist data across pod restarts. |
| **[Vast.ai](https://vast.ai)** | Often cheapest for spot instances. |
| **[Lambda Labs](https://lambdalabs.com)** | Reliable, fixed pricing. |
| **[Google Colab Pro+](https://colab.research.google.com)** | A100 available; mount Google Drive as workspace. |

A GPU with **≥20 GB VRAM** is recommended (RTX 4090 / A100). Qwen3.5-9B with LoRA fits in 20 GB.

**Local CUDA:** Works directly. Set `PBM_WORKSPACE` to a local directory and install dependencies from the Dockerfile or via `scripts/setup_pod.sh`.

**Apple Silicon / Mac:** Not tested yet. Unsloth has experimental MPS support and the training stages use standard HuggingFace/TRL primitives, so it may work. If you try it, we'd love to hear how it goes — open an issue or PR.

### RunPod-specific setup

1. **Network Volume** — create one in your target region (e.g. CA-2), 200 GB, mounted at `/workspace`
2. **Pod** — deploy with Docker image `ghcr.io/chetantekur/pinchbenchmaxing:latest`, attach the volume, ≥24 GB VRAM GPU
3. **Region** — pod and volume must be in the same datacenter
4. **Container disk** — 50 GB minimum (model cache + GGUF)

The Docker image contains PyTorch 2.6 + CUDA 12.4, Unsloth, Ollama, Node.js 22, and OpenClaw. It does **not** contain the project code — `startup.sh` clones this repo to `/root/pbm` on first start and `git pull`s on subsequent restarts.

The container starts idle (`tail -f /dev/null`). After SSH, run `bash /root/scripts/startup.sh` to start services.

---

## API Keys

| Key | Purpose |
|-----|---------|
| `ANTHROPIC_API_KEY` | Generating synthetic training data, LLM judge, failure analysis |
| `OPENROUTER_API_KEY` | **PinchBench LLM judge** (claude-opus-4.5 via OpenRouter), image generation |
| `BRAVE_API_KEY` | Web search tasks (task_02, task_06, task_18) |
| `OPENCLAW_GATEWAY_TOKEN` | OpenClaw gateway auth (any random string) |
| `HF_TOKEN` | HuggingFace dataset push (optional — CuratorAgent auto-pushes after curation) |

**Critical:** The PinchBench judge calls Claude through **OpenRouter**, not the Anthropic API directly. You need both keys.

### Persisting keys across restarts

```bash
cp scripts/set_env.sh /workspace/synthbench/set_env.sh
vim /workspace/synthbench/set_env.sh
```

`startup.sh` sources this file automatically. Always use `source`, not `sh`:
```bash
source /workspace/synthbench/set_env.sh    # correct
sh /workspace/synthbench/set_env.sh        # WRONG — subshell exits, nothing exported
```

---

## Configuration

All settings live in `config.yaml`. `utils/config.py` is a thin Python loader — not a separate config file.

```yaml
model:
  base: Qwen/Qwen3.5-9B       # HuggingFace model to fine-tune
  name: qwen35-9b-clawd        # used for output dirs + Ollama model name

paths:
  workspace: ${PBM_WORKSPACE:-./workspace}

data:
  examples_per_task: 70        # target training examples per task
  val_split: 0.1
  min_judge_score: 3           # discard examples below this (1–5 scale)

training:
  epochs: 3
  batch_size: 2
  grad_accum: 4
  learning_rate: 2e-4
  lora_r: 16
  lora_alpha: 32
  max_seq_len: 4096

convert:
  quantization: q4_k_m

claude:
  generation: claude-sonnet-4-5   # synthetic data generation
  judge: claude-sonnet-4-5        # quality scoring
  analysis: claude-sonnet-4-6     # failure diagnosis

loop:
  max_iterations: 5
  target_score: 0.85
  weak_task_threshold: 0.50
  max_examples_per_task: 100  # ceiling for score-0 tasks (decays proportionally)
  total_new_examples_cap: 500 # max new examples per iteration

huggingface:
  dataset_repo: ChetanTekur/pinchbench-clawd  # auto-push after curation
```

---

## Project Structure

```
config.yaml                  # all settings — single source of truth
loop.py                      # agentic loop orchestrator
LICENSE                      # MIT (code), CC BY 4.0 (dataset)
dataset_card.md              # HuggingFace dataset card (pushed as README.md to HF)

agents/
  base.py                    # AgentState (shared state), Agent base class, file logger
  eval_agent.py              # benchmarks model, parses scores, archives logs per iteration
  eval_analysis_agent.py     # probes model versions, diagnoses regressions with Claude
  data_agent.py              # score-proportional generation: targeted topup + adversarial
  curator_agent.py           # score → repair → filter → dedup → verify → HF push
  trainer_agent.py           # prepare → finetune → convert → register versioned model

stages/
  prepare.py                 # convert train.jsonl → SFT chat format for Unsloth
  finetune.py                # LoRA fine-tuning with Unsloth (supports --dry-run)
  convert.py                 # export merged model → GGUF (llama.cpp quantization)
  validate_model.py          # check HF existence, architecture, Unsloth support, VRAM
  probe.py                   # interactive model testing from merged weights

scripts/
  startup.sh                 # every-session: start Ollama + OpenClaw (DNS retry built in)
  benchmark_run.sh           # run PinchBench, save log to network volume
  register_model.sh          # register GGUF in Ollama with correct tool-calling Modelfile
  set_env.sh                 # API key template — copy to persistent storage
  setup_pod.sh               # one-time pod setup (if not using Docker image)
  check_setup.sh             # pre-benchmark checker
  train_and_eval.sh          # manual pipeline: prepare → finetune → convert → benchmark
  patch_openclaw_image_gen.py

utils/
  config.py                  # loads config.yaml, computes all derived paths
  prompts.py                 # shared constants: OPENCLAW_SYSTEM prompt, VALID_TOOLS

generate.py                  # generate initial training data via Claude Batch API
llm_judge.py                 # score examples 1–5 with Claude, filter at --min 3
topup.py                     # fill per-task gaps (plain round-robin fallback)
targeted_topup.py            # diagnosis-aware generation — weighted variations, injected context
adversarial_gen.py           # generate from benchmark failure transcripts
example_repair.py            # repair borderline examples (score 2-3) instead of discarding
dedup.py                     # semantic deduplication (TF-IDF + tool Jaccard similarity)
inspect_data.py              # dataset stats, validation, sampling
test_tool_call.sh            # verify fine-tuned model emits <tool_call> blocks
openclaw_template.json       # OpenClaw config template (secrets injected by startup.sh)
Dockerfile                   # full image: PyTorch + Unsloth + Ollama + OpenClaw
Dockerfile.bench             # lightweight image: Ollama + OpenClaw only
```

---

## Agentic Loop

### Starting

**Always run in tmux** so the loop survives SSH disconnects:
```bash
tmux new -s loop
python3 loop.py run
# Detach: Ctrl+B, D
# Reattach: tmux attach -t loop
```

**Resuming with an existing model:**
```bash
python3 loop.py run --model qwen35-9b-clawd-v5
```

**Seeding scores directly** (skips re-running the benchmark):
```bash
python3 loop.py run --model qwen35-9b-clawd-v5 --scores '{"task_00_sanity":1.0,...}'
```

**Seeding from a benchmark log:**
```bash
python3 loop.py run --model qwen35-9b-clawd-v5 \
  --log $PBM_WORKSPACE/logs/bench_ollama_qwen35-9b-clawd-v5.log
```

**Check status:**
```bash
python3 loop.py status
```

### Iteration flow

```
Startup
└── Model validation (cached after first run)

Iteration N
├── EvalAgent          — benchmark current model, parse + archive scores
│   └── Gate: regression > 5% below best ever → PAUSE
│   └── Gate: score ≥ target → EXIT (done)
│   └── Gate: no weak tasks → EXIT
├── EvalAnalysisAgent  — probe model versions, diagnose with Claude
│   └── Gate: exception / empty / error → PAUSE
│       (training never proceeds without a real diagnosis)
├── DataAgent          — score-proportional topup + adversarial generation
│   └── Gate: 0 new examples → PAUSE
├── CuratorAgent       — score → repair → filter → dedup → verify → HF push
│   └── Gate: train.jsonl empty after curation → PAUSE
└── TrainerAgent       — validate model → prepare → finetune → convert → register
    └── Gate: GGUF not found / model not in ollama list → FAIL
```

### Logs and debugging

```bash
# Live loop output (if running in tmux)
tmux attach -t loop

# Loop log file (always written, survives disconnects)
tail -f $PBM_WORKSPACE/logs/loop.log

# Benchmark logs (archived per iteration, never overwritten)
ls $PBM_WORKSPACE/logs/bench_v*_iter*.log

# Raw Claude responses (for debugging JSON parse failures)
ls $PBM_WORKSPACE/data/debug/

# EvalAnalysis reports
ls $PBM_WORKSPACE/data/eval_analysis_*.json

# Service logs
tail -f /tmp/ollama.log
tail -f /tmp/openclaw-gateway.log
```

---

## Data Generation Pipeline

Run locally (Anthropic Batch API is ~50% cheaper, no GPU needed).

### 1. Generate initial data

```bash
source /workspace/synthbench/set_env.sh
python generate.py submit    # submit ~920 examples to Claude Batch API
python generate.py status    # poll until complete
python generate.py collect   # write train.jsonl + val.jsonl
```

### 2. Score and filter

```bash
python llm_judge.py run             # score all examples 1–5 (resumes if interrupted)
python llm_judge.py report          # per-task score summary
python llm_judge.py filter --min 3  # remove low-quality examples
```

### 3. Inspect and validate

```bash
python inspect_data.py stats     # examples per task
python inspect_data.py validate  # structural issues
python inspect_data.py sample    # preview random examples
```

### 4. Top up weak tasks

```bash
python topup.py count   # show current vs target per task
python topup.py run     # generate more for tasks below target
```

---

## Fine-Tuning Pipeline (Advanced / Manual)

> **Normal path:** `python3 loop.py run` — TrainerAgent handles all of this automatically.

```bash
python -m stages.validate_model              # check base model is valid
python -m stages.prepare                     # convert to SFT format
python -m stages.finetune --dry-run          # sanity check
python -m stages.finetune                    # 30–90 min on A100
python -m stages.convert                     # merge + quantize to GGUF
OLLAMA_MODEL=qwen35-9b-clawd-v6 bash scripts/register_model.sh
bash scripts/benchmark_run.sh ollama/qwen35-9b-clawd-v6
```

---

## Key Insights & Gotchas

### 1. Always run the loop in tmux

The loop takes hours (fine-tuning + benchmarking per iteration). If your SSH disconnects, the process dies. Always use tmux:
```bash
tmux new -s loop
python3 loop.py run
# Ctrl+B, D to detach — tmux attach -t loop to reattach
```

### 2. OpenRouter is required for benchmarking

The PinchBench judge calls Claude via **OpenRouter** (`OPENROUTER_API_KEY`), not the Anthropic API. You need both keys for different things.

### 3. GGUF models need `register_model.sh` for tool calling

A plain `ollama create FROM <gguf>` silently breaks tool calling. Always use `register_model.sh`.

### 4. `~/.ollama` is ephemeral — models re-register after restart

`startup.sh` handles this automatically from `loop_state.json`.

### 5. Source env files — never `sh`

```bash
source /workspace/synthbench/set_env.sh    # correct
sh /workspace/synthbench/set_env.sh        # WRONG
```

### 6. PYTHONPATH is set automatically after `startup.sh`

No need to prefix commands. If running locally: `export PYTHONPATH=$(pwd)`

### 7. Model version numbers must be seeded when resuming

```bash
python3 loop.py run --model qwen35-9b-clawd-v5
```

Without `--model`, the loop starts from v0.

### 8. Training is gated on a valid analysis

The loop pauses if EvalAnalysisAgent produces no root causes or data fixes. Check `data/debug/` for raw Claude responses.

### 9. Benchmark logs are archived per iteration

Each eval creates `bench_v{N}_iter{M}_{timestamp}.log` so no run overwrites another.

### 10. Base model is validated at loop startup

`stages/validate_model.py` checks: HuggingFace existence, text-generation architecture, Unsloth support, tokenizer, VRAM estimate. Result is cached in state — only runs once per model.

---

## PinchBench Task Reference

| Task | Name | External Dependency |
|------|------|---------------------|
| task_00 | sanity | none |
| task_01 | calendar | none |
| task_02 | stock | Brave web search |
| task_03 | blog | none |
| task_04 | weather | wttr.in (free) |
| task_05 | summary | none |
| task_06 | events | Brave web search |
| task_07 | email | none |
| task_08 | memory | none |
| task_09 | files | none |
| task_10 | workflow | none |
| task_11 | config_update | none |
| task_12 | skill_search | none |
| task_13 | image_gen | OpenRouter |
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

| Problem | Fix |
|---------|-----|
| Training dies on SSH disconnect | Run in `tmux new -s loop` |
| OpenClaw gateway not starting | `cat /tmp/openclaw-gateway.log` — check Node.js version (needs 22) |
| Benchmark scores all 0.0 | `OLLAMA_MODEL=... bash scripts/register_model.sh` |
| `ModuleNotFoundError: No module named 'utils'` | `cd /root/pbm` (PYTHONPATH must be project root) |
| `config.yaml not found` | Run from project root (`/root/pbm`) |
| `GGUF not found` | `python3 -c "from utils.config import load_config; print(load_config().gguf_file)"` |
| Loop created v1 instead of v6 | Pass `--model qwen35-9b-clawd-v5` when resuming |
| Analysis JSON parse error | Check `$PBM_WORKSPACE/data/debug/` for raw response |
| Pod crash-loops on start | Container starts idle; run `bash /root/scripts/startup.sh` manually |
| `source` vs `sh` for env files | Always `source`, never `sh` |

---

## Useful Paths

| Path | Contents |
|------|---------|
| `$PBM_WORKSPACE/data/train.jsonl` | Training data |
| `$PBM_WORKSPACE/data/val.jsonl` | Validation data |
| `$PBM_WORKSPACE/data/scores.json` | LLM judge scores |
| `$PBM_WORKSPACE/data/loop_state.json` | Agentic loop state |
| `$PBM_WORKSPACE/data/current_diagnosis.json` | Per-task diagnosis from latest analysis |
| `$PBM_WORKSPACE/data/eval_analysis_*.json` | Per-iteration diagnosis reports |
| `$PBM_WORKSPACE/data/debug/` | Raw Claude responses for debugging |
| `$PBM_WORKSPACE/data/dedup_report.json` | Deduplication stats |
| `$PBM_WORKSPACE/data/repair_report.json` | Example repair stats |
| `$PBM_WORKSPACE/data/adversarial_results.json` | Adversarial generation stats |
| `$PBM_WORKSPACE/logs/loop.log` | Unified agent log (timestamped) |
| `$PBM_WORKSPACE/logs/bench_v*_iter*.log` | Archived benchmark logs (per iteration) |
| `$PBM_WORKSPACE/logs/bench_*.log` | Latest benchmark log (per model) |
| `~/.openclaw/openclaw.json` | Generated OpenClaw config |
| `/tmp/ollama.log` | Ollama logs (ephemeral) |
| `/tmp/openclaw-gateway.log` | OpenClaw gateway logs (ephemeral) |
| `/root/pbm/` | Project code (cloned by startup.sh) |

---

## Roadmap

- **Apple Silicon support** — test and validate the full pipeline on Mac (M1/M2/M3/M4)
- **Multi-model support** — easy swap to Llama, Mistral, Phi, Gemma (see Contributing below)
- **Parallel benchmarking** — eval multiple model versions simultaneously
- **Curriculum learning** — weight harder tasks more during training
- **Cross-iteration dedup** — track example hashes across all iterations to prevent drift

---

## Contributing: Adding Support for Other Models

The pipeline is designed around Qwen3.5-9B, but adding a new base model is straightforward:

### 1. `config.yaml` — point to the new model

```yaml
model:
  base: mistralai/Mistral-7B-v0.3
  name: mistral-7b-clawd
```

### 2. Validate the model

```bash
python -m stages.validate_model --model mistralai/Mistral-7B-v0.3
```

This checks HuggingFace existence, architecture, Unsloth support, tokenizer, and VRAM estimate.

### 3. `scripts/register_model.sh` — update the Modelfile template

The current template is Qwen3-specific. For a different model family:
1. Pull the base model: `ollama pull mistral:7b`
2. Export its Modelfile: `ollama show mistral:7b --modelfile`
3. Verify tool-call support (`{{- if .Tools }}` template block)
4. Update `register_model.sh` to use the new template

### 4. Test end-to-end

```bash
python -m stages.finetune --dry-run
python -m stages.finetune
python -m stages.convert
bash scripts/register_model.sh
bash test_tool_call.sh
bash scripts/benchmark_run.sh ollama/mistral-7b-clawd --no-upload
```

If you get it working, please open a PR.
