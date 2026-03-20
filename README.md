# PinchBench Maxing

An autonomous system that fine-tunes an open-source LLM to compete on [PinchBench](https://pinchbench.com) — a 23-task benchmark for AI agents.

A Claude-powered **orchestrator** examines the current state, decides what to do next, and calls tools to execute: benchmark, diagnose failures, generate targeted training data, curate quality, and train. Each decision is a single Claude API call — no fixed pipeline, no hardcoded stage order. The orchestrator adapts to whatever situation it finds.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): give an AI system a metric to optimize and let it drive the improvement cycle.

---

## Results

| Model | Score | Notes |
|-------|-------|-------|
| qwen3:8b (base) | 3.1 / 23 (13%) | No fine-tuning |
| qwen35-9b-clawd-v1 | 9.9 / 23 (43%) | First fine-tune, ~900 examples |
| qwen35-9b-clawd-v3 | 16.8 / 23 (73%) | Manual topup |
| qwen35-9b-clawd-v5 | 15.0 / 23 (65%) | First agentic pipeline run |
| qwen35-9b-clawd-v6 | 16.6 / 23 (72%) | Score-proportional data gen |
| qwen35-9b-clawd-v7 | 14.8 / 23 (64%) | Regression from dataset imbalance |

Dataset: [huggingface.co/datasets/cptekur/pinchbench-clawd](https://huggingface.co/datasets/cptekur/pinchbench-clawd) (CC BY 4.0)

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│              Claude Orchestrator                        │
│                                                         │
│   State → Claude decides → Tool call → State updates    │
│                    ↑                         │           │
│                    └─────────────────────────┘           │
│                   (repeat until DONE)                    │
└─────────────────────────────────────────────────────────┘
```

The orchestrator is a loop:
1. Claude receives the current state (scores, dataset stats, budget, action history)
2. Claude decides what to do and calls **one tool**
3. The tool executes, state updates
4. Repeat until target score is reached, budget is exhausted, or Claude says DONE

Each turn is a **single Claude API call** (~$0.03). No conversation history accumulates — "memory" is the action history in `loop_state.json`. Zero risk of context overflow.

### Two Layers of Reasoning

**Orchestrator** (decides WHAT to do):
- Lightweight: picks the next action based on state
- Example: "dataset is imbalanced → call rebalance before training"

**Specialist reasoning tools** (deeper analysis):
- `diagnose` — Claude analyzes benchmark results, forms hypotheses about failures, produces structured root causes
- `plan_strategy` — Claude takes the diagnosis + dataset stats and produces a specific data generation plan

The orchestrator delegates deep thinking to these tools and acts on their output.

### Available Tools

| Tool | Type | What it does |
|------|------|-------------|
| `inspect_data` | Data | Dataset stats: counts, balance, quality |
| `generate_data` | Data | Generate targeted training examples |
| `generate_adversarial` | Data | Generate from benchmark failure transcripts |
| `score_data` | Data | Score all examples 1-5 via LLM judge |
| `filter_data` | Data | Remove examples below score threshold |
| `repair_data` | Data | Fix borderline examples (score 2-3) |
| `dedup_data` | Data | Remove semantically similar examples |
| `rebalance_data` | Data | Trim overweight tasks to target |
| `snapshot` | Data | Save dataset before destructive operations |
| `push_hf` | Data | Push to HuggingFace |
| `diagnose` | Reasoning | Deep failure analysis with Claude |
| `plan_strategy` | Reasoning | Data generation planning with Claude |
| `benchmark` | Eval | Run PinchBench (23 tasks) |
| `check_disk` | Eval | Check free disk space |
| `validate_model` | Training | Verify base model on HuggingFace |
| `train` | Training | Fine-tune with Unsloth LoRA |
| `convert` | Training | Merge + quantize to GGUF |
| `register` | Training | Register GGUF in Ollama |
| `get_state` | Control | Return full state |
| `request_approval` | Control | Pause for human input |

### Guardrails

The orchestrator auto-stops if:
- Budget drops below $5
- Score regresses >10% from best ever
- Dataset drops below 500 examples
- A tool fails 3 times consecutively

All prompts are in `prompts/*.md` as templates — editable without touching code. Variables come from `config.yaml`.

---

## Quick Start

### One-time setup (on the pod)

```bash
# 1. Start a pod with ghcr.io/chetantekur/pinchbenchmaxing:latest
#    Container starts idle. SSH in, then:

# 2. Run startup — clones repo, starts Ollama + OpenClaw
bash /root/scripts/startup.sh

# 3. Set up API keys (one time — persists on network volume)
cp /root/pbm/scripts/set_env.sh /workspace/synthbench/set_env.sh
vim /workspace/synthbench/set_env.sh   # fill in all keys
source /workspace/synthbench/set_env.sh

# 4. Register with PinchBench (one time)
cd $PBM_WORKSPACE/skill && bash scripts/run.sh --register && cd -
```

### Run the orchestrator

```bash
cd /root/pbm
tmux new -s loop
python3 orchestrator.py run
# Detach: Ctrl+B, D — reattach: tmux attach -t loop
```

**Resuming with an existing model:**
```bash
python3 orchestrator.py run --model qwen35-9b-clawd-v7
```

**Seeding scores from a benchmark log:**
```bash
python3 orchestrator.py run --model qwen35-9b-clawd-v7 \
  --log $PBM_WORKSPACE/logs/bench_ollama_qwen35-9b-clawd-v7.log
```

**Dry run (Claude decides but tools don't execute):**
```bash
python3 orchestrator.py run --model qwen35-9b-clawd-v7 --dry-run
```

**Legacy fixed pipeline (fallback):**
```bash
python3 loop.py run --model qwen35-9b-clawd-v7 --mode pipeline
```

**Check status:**
```bash
python3 orchestrator.py status
```

### On pod restart

```bash
bash /root/scripts/startup.sh
cd /root/pbm
tmux new -s loop
python3 orchestrator.py run    # resumes from saved state
```

---

## GPU Options

Any provider with ≥20 GB VRAM works. Suggestions: [RunPod](https://runpod.io), [Vast.ai](https://vast.ai), [Lambda Labs](https://lambdalabs.com).

**Local CUDA:** Set `PBM_WORKSPACE` and install deps from the Dockerfile.

**Apple Silicon:** Not tested. Contributions welcome.

### RunPod setup

1. **Network Volume** — 300 GB, your target region (e.g. CA-2)
2. **Pod** — `ghcr.io/chetantekur/pinchbenchmaxing:latest`, attach volume, ≥24 GB VRAM
3. **Container starts idle** — run `bash /root/scripts/startup.sh` after SSH

---

## API Keys

| Key | Purpose |
|-----|---------|
| `ANTHROPIC_API_KEY` | Orchestrator decisions, data generation, LLM judge, failure analysis |
| `OPENROUTER_API_KEY` | PinchBench LLM judge (claude-opus-4.5 via OpenRouter) |
| `BRAVE_API_KEY` | Web search tasks (task_02, task_06, task_18) |
| `OPENCLAW_GATEWAY_TOKEN` | OpenClaw gateway auth (any random string) |
| `HF_TOKEN` | HuggingFace dataset push (optional) |

**Critical:** PinchBench's judge uses **OpenRouter**, not the Anthropic API. You need both keys.

Persist keys:
```bash
cp scripts/set_env.sh /workspace/synthbench/set_env.sh
vim /workspace/synthbench/set_env.sh
# Always: source, never sh
```

---

## Configuration

All settings in `config.yaml`. Prompts in `prompts/*.md`. No hardcoded values.

```yaml
model:
  base: Qwen/Qwen3.5-9B
  name: qwen35-9b-clawd

paths:
  workspace: ${PBM_WORKSPACE:-./workspace}

data:
  examples_per_task: 70
  val_split: 0.1
  min_judge_score: 3

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
  generation: claude-sonnet-4-5
  judge: claude-sonnet-4-5
  analysis: claude-sonnet-4-6

orchestrator:
  model: claude-sonnet-4-6
  max_actions: 20
  budget_usd: 25
  gpu_rate_per_hour: 3.60
  auto_pause:
    score_regression_pct: 10
    min_dataset_size: 500
    min_disk_gb: 10
    max_consecutive_failures: 3

loop:
  max_iterations: 5
  target_score: 0.85
  weak_task_threshold: 0.50
  max_new_per_task: 50
  max_total_per_task: 120
  total_new_examples_cap: 300

huggingface:
  dataset_repo: cptekur/pinchbench-clawd
```

---

## Project Structure

```
orchestrator.py              # Claude-powered orchestrator (main entry point)
loop.py                      # --mode agentic (default) | pipeline (fallback)
config.yaml                  # all settings
LICENSE                      # MIT (code), CC BY 4.0 (dataset)

prompts/                     # agent prompts as markdown templates
  orchestrator.md            # orchestrator system prompt
  diagnose.md                # failure analysis reasoning
  plan_strategy.md           # data generation planning

tools/                       # tool implementations
  registry.py                # 19 tool schemas + dispatch
  data_tools.py              # inspect, generate, score, filter, repair, dedup, rebalance, snapshot
  training_tools.py          # validate, train, convert, register, benchmark, check_disk
  reasoning_tools.py         # diagnose, plan_strategy (Claude-calls-Claude)
  eval_tools.py              # get_state, request_approval

agents/                      # pipeline agents (used by --mode pipeline)
  base.py                    # AgentState, Agent base class, file logger
  eval_agent.py              # benchmarks model, parses scores
  eval_analysis_agent.py     # probes model versions, diagnoses regressions
  data_agent.py              # score-proportional generation
  curator_agent.py           # score → repair → filter → dedup → rebalance → HF push
  trainer_agent.py           # prepare → finetune → convert → register

stages/                      # training pipeline stages
  prepare.py                 # convert to SFT format
  finetune.py                # LoRA fine-tuning with Unsloth
  convert.py                 # merge + quantize to GGUF
  validate_model.py          # check HF existence, architecture, Unsloth support
  probe.py                   # interactive model testing

datagen/                     # data generation and curation scripts
  generate.py                # initial data via Claude Batch API
  topup.py                   # plain topup (round-robin fallback)
  targeted_topup.py          # diagnosis-aware topup
  adversarial_gen.py         # generate from benchmark failure transcripts
  llm_judge.py               # score examples 1-5 with Claude
  example_repair.py          # repair borderline examples
  dedup.py                   # semantic deduplication
  rebalance.py               # trim overweight tasks
  inspect_data.py            # dataset stats and validation
  analyze_dataset.py         # dataset health analysis

scripts/                     # shell scripts
  startup.sh                 # every-session: start Ollama + OpenClaw
  benchmark_run.sh           # run PinchBench
  register_model.sh          # register GGUF in Ollama
  push_to_hf.sh              # push dataset to HuggingFace
  set_env.sh                 # API key template
  check_setup.sh             # pre-benchmark checker
  train_and_eval.sh          # manual pipeline
  test_tool_call.sh          # verify tool calling works

utils/                       # shared utilities
  config.py                  # loads config.yaml, computes derived paths
  prompts.py                 # shared constants: OPENCLAW_SYSTEM, VALID_TOOLS

docs/                        # documentation
  architecture.md            # system design
```

---

## Logs and Debugging

```bash
# Live orchestrator output
tmux attach -t loop

# Full log (all tool output captured)
tail -f $PBM_WORKSPACE/logs/loop.log

# Benchmark logs (archived per iteration)
ls $PBM_WORKSPACE/logs/bench_v*_iter*.log

# Raw Claude responses (for JSON parse debugging)
ls $PBM_WORKSPACE/data/debug/

# Dataset health
python3 -m datagen.analyze_dataset

# Orchestrator status
python3 orchestrator.py status
```

---

## Key Insights

1. **Always run in tmux** — training takes hours, SSH disconnects kill the process
2. **OpenRouter is required for benchmarking** — PinchBench judge uses OpenRouter, not Anthropic API
3. **GGUF models need `register_model.sh`** — plain `ollama create` breaks tool calling
4. **`~/.ollama` is ephemeral** — `startup.sh` re-registers models from `loop_state.json`
5. **`source` env files, never `sh`** — `sh` runs in a subshell, exports nothing
6. **Dataset balance matters** — overweight tasks cause catastrophic forgetting (v7 lesson)
7. **Snapshot before destructive ops** — filter/dedup/rebalance delete examples
8. **Model checkpoints are versioned** — `qwen35-9b-clawd-v8_merged/` etc.
9. **All prompts are in `prompts/*.md`** — edit behavior without touching code
10. **`--dry-run` tests orchestrator decisions** — Claude decides but tools don't execute

---

## PinchBench Task Reference

| Task | Name | Dependency |
|------|------|-----------|
| task_00 | sanity | none |
| task_01 | calendar | none |
| task_02 | stock | Brave search |
| task_03 | blog | none |
| task_04 | weather | wttr.in |
| task_05 | summary | none |
| task_06 | events | Brave search |
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
| task_18 | market_research | Brave search |
| task_19 | spreadsheet_summary | pandas + openpyxl |
| task_20 | eli5_pdf | pdfplumber + PyPDF2 |
| task_21 | openclaw_comprehension | none |
| task_22 | second_brain | none |

---

## Roadmap

- **Apple Silicon support** — test on Mac M1/M2/M3/M4
- **Multi-model support** — easy swap to Llama, Mistral, Phi, Gemma
- **Curriculum learning** — weight harder tasks more during training
- **Parallel benchmarking** — eval multiple versions simultaneously

---

## Contributing

### Adding a new base model

1. Update `config.yaml`: `model.base` and `model.name`
2. Validate: `python -m stages.validate_model --model your/model-id`
3. Update `scripts/register_model.sh` Modelfile template for your model family
4. Test: `python -m stages.finetune --dry-run`

### Adding a new tool

1. Add the implementation to `tools/data_tools.py` (or appropriate module)
2. Add the schema to `tools/registry.py` (TOOL_SCHEMAS + _DISPATCH)
3. Document in `prompts/orchestrator.md` so the orchestrator knows about it
4. Test standalone, then with `orchestrator.py --dry-run`

### Editing orchestrator behavior

All prompts are in `prompts/*.md`. Edit the markdown, restart the orchestrator. No code changes needed for:
- Adding new guardrails
- Changing decision priorities
- Adjusting scenario guidance
- Modifying reasoning tool instructions
