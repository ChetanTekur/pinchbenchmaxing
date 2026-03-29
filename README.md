# PinchBench Maxing

An agentic system that autonomously fine-tunes an open-source LLM to compete on [PinchBench](https://pinchbench.com) — a 23-task benchmark for AI agents.

## The Idea

Most ML training loops are rigid pipelines: generate data → train → eval → repeat. When something goes wrong — the dataset is imbalanced, a task regresses, disk fills up — the pipeline breaks and a human has to debug.

PinchBench Maxing replaces the pipeline with a **Claude-powered orchestrator** that *reasons* about what to do next. It sees the current state (benchmark scores, dataset stats, budget, disk space, action history) and makes a decision: should I generate more data? Rebalance the dataset first? Diagnose why email scoring dropped? Clean up disk before training?

Each decision is a single Claude API call. The orchestrator has 19 tools at its disposal and a set of guardrails defined in plain markdown (`prompts/orchestrator.md`). No hardcoded stage order. No brittle `if/else` logic. The orchestrator adapts.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): give an AI system a metric to optimize and let it drive the improvement cycle while you sleep.

| Karpathy autoresearch | PinchBench Maxing |
|---|---|
| Single agent modifies `train.py` | Claude orchestrator calls 19 tools |
| Agent modifies training code | Orchestrator modifies training *data* |
| Fixed 5-min experiment budget | Budget-capped sessions ($25 default) |
| Metric: val bits-per-byte | Metric: PinchBench score (0–23 tasks) |
| Runs overnight on a single GPU | Runs overnight on a single GPU |
| Agent reads eval output, iterates | `diagnose` tool uses Claude to *reason* about failures |
| `program.md` guides the agent | `prompts/orchestrator.md` guides the orchestrator |

---

## Results

| Model | Score | Notes |
|-------|-------|-------|
| qwen3:8b (base) | 3.1 / 23 (13%) | No fine-tuning |
| qwen35-9b-clawd-v1 | 9.9 / 23 (43%) | First fine-tune, ~900 examples |
| qwen35-9b-clawd-v3 | 16.8 / 23 (73%) | Manual topup |
| qwen35-9b-clawd-v5 | 15.0 / 23 (65%) | First agentic pipeline run |
| qwen35-9b-clawd-v7 | 14.8 / 23 (64%) | Regression — dataset imbalance |
| qwen35-9b-clawd-v9 | 5.6 / 23 (24%) | Regression — wrong Modelfile template |
| qwen35-9b-clawd-v14 | 11.7 / 23 (51%) | Fixed Modelfile template (explicit Qwen3 chat template) |
| qwen35-9b-clawd-v15 | 16.1 / 23 (70%) | Fixed seq_len 4096→8192 (long tasks were truncated) |
| qwen35-9b-clawd-v16 | 16.2 / 23 (70%) | Adversarial data for task_12 (0→100%) |
| qwen35-9b-clawd-v17 | 15.8 / 23 (69%) | Balanced data, smart analyzer |
| qwen35-9b-clawd-v19 | 12.8 / 23 (56%) | Regression — blind data generation without diagnosis |
| qwen35-9b-clawd-v20 | 15.5 / 23 (67%) | Partial recovery |
| **qwen35-9b-clawd-v21** | **17.8 / 23 (77%)** | **New best — adversarial fixes for 6 zero tasks** |
| qwen35-9b-clawd-v22 | ~14 / 23 (~61%) | Regression — old orchestrator overwrote v21 gold data |

### Key Lessons

1. **The Modelfile template matters more than the data.** Switching from Ollama's `RENDERER qwen3.5` shortcut to the explicit Qwen3 chat template (with `<tools>/<tool_call>` XML blocks) doubled scores from 25% to 50%. Same model weights, same data.

2. **Sequence length truncation silently kills multi-step tasks.** Training at `max_seq_len=4096` truncated long tasks (email triage at ~6K tokens, market research at ~8K tokens). The model learned "read files and stop" because it never saw the write step. Increasing to 8192 fixed tasks 15, 16, 17, 22 from 0% to 88-98%.

3. **Data imbalance causes catastrophic forgetting.** When some tasks had 118 examples and others had 27, the model over-learned the bloated tasks and forgot the small ones. Balanced data (40-50 per task) consistently scores better.

4. **Quality over quantity.** 50 high-quality examples per task (judge score ≥4.5) outperforms 120 mixed-quality examples. The smart data analyzer uses benchmark scores + judge quality + example counts to decide when to stop generating.

5. **Diagnose before generating.** The v19-v22 regression cycle proved that blind adversarial data generation destroys capabilities. The orchestrator now has a hard gate: `generate_data` and `generate_adversarial` are blocked until `diagnose` runs after each benchmark. Understanding *why* tasks fail is mandatory before changing data.

6. **Gold data must be recoverable.** v22 overwrote v21's gold dataset with worse data. The orchestrator now auto-restores the best-scoring dataset from HuggingFace on regression detection.

Dataset: [huggingface.co/datasets/cptekur/pinchbench-clawd](https://huggingface.co/datasets/cptekur/pinchbench-clawd) (CC BY 4.0)

---

## Architecture

### The Orchestrator

The orchestrator is a **stateful multi-turn agent loop**. Each turn:

1. **State is loaded** from `loop_state.json` — scores, dataset stats, budget
2. **A system prompt** is assembled from `prompts/orchestrator.md` with variables from `config.yaml`
3. **Claude receives** the full conversation history (previous tool calls + results) and returns **one tool call**
4. **The tool executes** and the result is appended to the conversation as a `tool_result` message
5. **State is saved** — crash-safe, resumable from any point

Claude maintains full context of what it diagnosed, hypothesized, and decided — no scratchpad workarounds needed.

```
                     prompts/orchestrator.md
                              │
                    ┌─────────▼──────────┐
                    │                    │
 config.yaml ──────▶  Claude Sonnet 4.6  ◀────── loop_state.json
                    │                    │           (state + action history)
                    └─────────┬──────────┘
                              │
                         tool call
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      ┌───────▼──────┐ ┌─────▼─────┐ ┌───────▼──────┐
      │  Data Tools  │ │ Reasoning │ │  Training    │
      │              │ │   Tools   │ │    Tools     │
      │ inspect      │ │           │ │              │
      │ generate     │ │ diagnose  │ │ train        │
      │ score        │ │ plan      │ │ convert      │
      │ filter       │ │ strategy  │ │ register     │
      │ repair       │ │           │ │ benchmark    │
      │ dedup        │ │ (Claude   │ │              │
      │ rebalance    │ │  calls    │ │              │
      │ snapshot     │ │  Claude)  │ │              │
      └──────────────┘ └───────────┘ └──────────────┘
```

**Stateful conversation** — Claude keeps the full conversation history, so it remembers diagnoses and hypotheses across turns. Context is automatically compressed when it exceeds 40 messages (keeps first + last 20). Cost is tracked from actual API token usage.

### Prompt Architecture

All agent behavior is defined in **markdown templates**, not Python code:

```
prompts/
  orchestrator.md     ← orchestrator system prompt (guardrails, decision framework, scenarios)
  diagnose.md         ← failure analysis reasoning (root causes, data fixes)
  plan_strategy.md    ← data generation planning (score-proportional allocation)
```

Each template uses `{variable}` placeholders filled from `config.yaml` at runtime:
- `{target_score}` → `0.85`
- `{max_total_per_task}` → `120`
- `{budget_usd}` → `25`

**To change the orchestrator's behavior, edit the markdown.** Add a guardrail, change a priority, adjust a scenario — no code changes, no redeployment. Just edit `prompts/orchestrator.md` and restart.

### Three Layers of Reasoning

**Layer 1: Orchestrator decisions** — lightweight ("dataset is imbalanced → rebalance before training"). Guided by `prompts/orchestrator.md` which includes a post-benchmark decision framework: when to LEAVE data alone, when to FIX, when to REGENERATE, when to just TRAIN AND SEE.

**Layer 2: Specialist analysis** — the orchestrator delegates to reasoning tools that call Claude:

- **`diagnose`** — receives benchmark scores, dataset stats, **bad examples report** (actual tool calls vs expected), and **validator expectations** (TOOL_SIGNATURES, REQUIRED_TOOLS). Claude performs a **three-way comparison**: what the benchmark expects vs what the training data teaches vs what the validator enforces. Can identify when the validator is wrong, not just the data.
- **`plan_strategy`** — receives the diagnosis + dataset state. Produces a targeted data generation plan.

**Layer 3: Deep validation** — Claude reasons about whether training examples would actually teach the model to pass the benchmark (`datagen/deep_validate.py`). Checks structural quality (tool names, args), statistical patterns (diversity, completeness), and semantic alignment (does this example teach correct behavior?).

### Dynamic Data Generation

Data generation reads task definitions directly from PinchBench `.md` files (`datagen/task_loader.py`) — no hardcoded copies that can drift. The meta-prompt includes:
- The full benchmark task definition (ground truth)
- The OPENCLAW_SYSTEM prompt (available tools)
- Variation configs for diversity
- Optional diagnosis context for targeted fixes

**Pilot-validate-refine flow** (`datagen/dynamic_gen.py`):
1. Generate 3 pilot examples via batch API
2. Deep validate against ground truth
3. If NEEDS_WORK: feed issues back, refine prompt, retry (up to 3 attempts)
4. If GOOD: bulk generate remaining examples
5. If BAD after 3 attempts: skip task and report

### Smart Data Analyzer

The data analyzer (`datagen/data_analyzer.py`) replaces hard-coded count caps with signal-based decisions. For each task it considers:
- **Benchmark score** — how the model actually performs
- **Judge score** — average LLM quality rating of training examples
- **Example count** — with a hard ceiling of 80

Recommendations per task:
| Action | When |
|--------|------|
| LEAVE_ALONE | Benchmark ≥80%, or at hard ceiling |
| GENERATE | Low benchmark + few examples |
| ADVERSARIAL | Low benchmark despite good data — needs failure-targeted examples |
| REGENERATE | Low benchmark + low judge score — data quality is bad |
| TRIM | Too many examples (>100) causing forgetting |
| INVESTIGATE | Task regressed ≥20% from previous version |
| INFRASTRUCTURE | 0% benchmark on infra-dependent task (image gen, web search) |

### Training Data Version Control

Training data is versioned on HuggingFace. The orchestrator pushes to HF before every training run with commit messages like `"Pre-v21 training data: ..."`. On regression, the gold restore mechanism downloads the best version's data from HF history automatically.

### Training Gates

The `train` tool has 3 gates:
1. **Coverage**: all 23 tasks must have ≥30 examples
2. **Quality**: ≥90% of examples must be clean (no critical issues)
3. **Disk**: root filesystem must have ≥15 GB free

### Example Session

```
Turn 1:  inspect_data     → "email=500 (4x mean), skill_search=39 — severely imbalanced"
Turn 2:  rebalance_data   → "trimmed 1200 examples, now 1703 total"
Turn 3:  diagnose         → "5 root causes: email schema wrong, second_brain never worked..."
Turn 4:  plan_strategy    → "email: 20 targeted, second_brain: 30 adversarial, ..."
Turn 5:  snapshot         → "saved pre-v8-training snapshot"
Turn 6:  generate_data    → "generated 150 new examples across 5 tasks"
Turn 7:  score_data       → "scored 1853 examples"
Turn 8:  filter_data      → "kept 1820, removed 33"
Turn 9:  dedup_data       → "removed 25 duplicates"
Turn 10: inspect_data     → "balanced: all tasks 60-120 examples"
Turn 11: check_disk       → "142 GB free"
Turn 12: push_hf          → "pushed to huggingface.co/datasets/cptekur/pinchbench-clawd"
Turn 13: train            → "loss 0.35, 45 minutes on H200"
Turn 14: convert          → "GGUF at .../v8_merged.Q4_K_M.gguf (5.1 GB)"
Turn 15: register         → "qwen35-9b-clawd-v8 registered in Ollama"
Turn 16: benchmark        → "74% (17.0/23) — up from 64%"
Turn 17: DONE             → "improved 64% → 74%. Budget: $18 remaining. Continue next session."
```

Notice how the orchestrator inspected the data *first*, saw the imbalance, and rebalanced *before* generating anything. A fixed pipeline would have generated more data on top of an already-broken dataset.

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

The orchestrator has hard-coded safety mechanisms:

**Auto-stops:**
- Budget drops below $5
- Score regresses >10% from best ever
- Dataset drops below 500 examples
- A tool fails 3 times consecutively
- Generation loop detected (5+ generate calls without training)

**Hard gates:**
- **Diagnose gate** — `generate_data` and `generate_adversarial` are blocked after benchmark until `diagnose` runs at least once. Prevents blind data generation. Capped at 2 diagnose calls per cycle to avoid analysis paralysis.
- **Gold restore** — on regression (model_version > best_version), auto-downloads the best-scoring dataset from HuggingFace and restores it before starting. Checks local cache (`data/gold_v{N}/`) first.

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
  examples_per_task: 50
  min_per_task: 30          # hard gate: train blocked below this
  val_split: 0.1
  min_judge_score: 3

training:
  epochs: 3
  batch_size: 2             # auto-tuned based on GPU VRAM after model load
  learning_rate: 2e-4
  lora_r: 16
  lora_alpha: 32
  max_seq_len: 4096

orchestrator:
  model: claude-sonnet-4-6
  max_actions: 200          # budget is the real limit
  budget_usd: 75

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
  dynamic_gen.py             # dynamic generation from PinchBench .md files (primary)
  task_loader.py             # loads task definitions from benchmark .md files
  deep_validate.py           # 3-level validation: structural + statistical + semantic (Claude)
  validate_data.py           # structural validation (tool names, args, schemas)
  generate.py                # initial data via Claude Batch API (legacy)
  topup.py                   # task definitions + variation configs
  targeted_topup.py          # diagnosis-aware topup (legacy — use dynamic_gen)
  adversarial_gen.py         # generate from benchmark failure transcripts
  llm_judge.py               # score examples 1-5 with Claude
  example_repair.py          # repair borderline examples
  dedup.py                   # semantic deduplication
  rebalance.py               # trim overweight tasks
  inspect_data.py            # dataset stats, diversity analysis, validation

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

1. **Bad data is worse than no data** — wrong tool names, wrong filenames, or wrong task definitions actively destroy model capabilities
2. **Always deep validate before training** — `python -m datagen.deep_validate` catches semantic issues that structural validation misses
3. **Task definitions must match the benchmark** — use `task_loader.py` to read from PinchBench `.md` files, never hardcode
4. **Pilot before bulk generating** — generate 3 examples, validate, refine prompt, then bulk
5. **Always run in tmux** — training takes hours, SSH disconnects kill the process
6. **OpenRouter is required for benchmarking** — PinchBench judge uses OpenRouter, not Anthropic API
7. **Symlink `~/.cache/huggingface/hub` to network volume** — root disk is only 50GB, model weights eat 18GB
8. **`source` env files, never `sh`** — `sh` runs in a subshell, exports nothing
9. **Snapshot before destructive ops** — filter/dedup/rebalance delete examples
10. **All prompts are in `prompts/*.md`** — edit behavior without touching code

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
| task_11 | create project structure (datautils) | none |
| task_12 | search and replace in files | none |
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
