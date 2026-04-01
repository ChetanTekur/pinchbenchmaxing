# PinchBench Maxing

Autonomous fine-tuning system that trains an open-source LLM to compete on [PinchBench](https://pinchbench.com) -- a 23-task benchmark for AI agents. A Claude-powered orchestrator diagnoses failures, generates targeted training data, fine-tunes, benchmarks, and iterates. You start it before bed and wake up to a better model.

13% -> 81% so far. Target: 85%.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## How it works

The orchestrator is a stateful Claude agent with 22 tools. Each turn it sees benchmark scores, dataset stats, and budget, then makes one decision: diagnose a failure, generate data, clean bad examples, train, benchmark, or stop.

```
orchestrator.py          prompts/orchestrator.md        config.yaml
      |                         |                           |
      v                         v                           v
  Claude Sonnet 4.6  <-- system prompt + state --> loop_state.json
      |
      v
  one tool call per turn
      |
      +-- Data tools:     generate, score, filter, validate, snapshot, ...
      +-- Analysis tools:  read_benchmark_transcript, diagnose
      +-- Training tools:  train, convert, register, benchmark
```

No hardcoded pipeline order. The orchestrator decides what to do based on what it sees.

**Key behaviors are defined in markdown** (`prompts/orchestrator.md`), not Python. To change how the orchestrator thinks, edit the markdown and restart.

## Quick start

```bash
# 1. Start a RunPod with ghcr.io/chetantekur/pinchbenchmaxing:latest (24GB+ VRAM)
# 2. SSH in, then:
bash /root/scripts/startup.sh

# 3. Set API keys (one time, persists on network volume)
cp /root/pbm/scripts/set_env.sh /workspace/synthbench/set_env.sh
vim /workspace/synthbench/set_env.sh
source /workspace/synthbench/set_env.sh

# 4. Run
cd /root/pbm && tmux new -s loop
python3 orchestrator.py run
```

Resume from a previous run:
```bash
python3 orchestrator.py run --model qwen35-9b-clawd-v25 \
  --log /workspace/synthbench/logs/bench_ollama_qwen35-9b-clawd-v25.log
```

## Results

| Version | Score | What happened |
|---------|-------|---------------|
| base qwen3:8b | 13% | No fine-tuning |
| v1 | 43% | First fine-tune, ~900 examples |
| v3 | 73% | Manual data topup |
| v9 | 24% | Regression: wrong Modelfile template |
| v15 | 70% | Fixed seq_len 4096->8192 |
| **v21** | **81%** | **Best: adversarial fixes for 6 zero-score tasks** |
| v25 | 70% | Regression: data corruption from blind generation |

Dataset: [huggingface.co/datasets/cptekur/pinchbench-clawd](https://huggingface.co/datasets/cptekur/pinchbench-clawd) (CC BY 4.0)

## What a run looks like

```
Turn 1:  diagnose              -> "5 root causes: task_21 uses wrong tools, task_01 schema broke..."
Turn 2:  read_benchmark_transcript -> reads raw model output for failing tasks
Turn 3:  validate_data fix=true -> removes 32 bad examples (missing required tools)
Turn 4:  generate_data         -> pilot validates 3 examples, bulk generates 50 more
Turn 5:  score_data            -> LLM judge scores all examples 1-5
Turn 6:  filter_data           -> removes examples scoring below 4
Turn 7:  push_hf               -> backs up dataset to HuggingFace
Turn 8:  train                 -> LoRA fine-tune on RTX 4090, ~45 minutes
Turn 9:  convert               -> merge + quantize to GGUF (5.6 GB)
Turn 10: register              -> register in Ollama
Turn 11: benchmark             -> 78% (up from 70%)
Turn 12: diagnose              -> analyzes what improved, what still fails
...continues until budget runs out or target reached
```

The orchestrator adapts. If data is imbalanced, it rebalances first. If a task regressed, it reads the transcript to understand why. If disk is low, it cleans old Ollama models. No fixed stage order.

## Design choices

**Data, not code.** The orchestrator modifies training data, not training code. Autoresearch modifies `train.py`. We modify `train.jsonl`. The training pipeline (LoRA, GGUF conversion, Ollama registration) is fixed. What changes is what the model sees during training.

**Pilot before bulk.** Every generation run starts with 3 pilot examples, validated by Claude against the ground truth task definition. If the pilot is bad, the prompt is refined and retried. Only after the pilot passes does bulk generation proceed. This caught wrong filenames, missing tools, and truncation before they entered the dataset.

**Self-healing generation.** No hardcoded list of "hard tasks." Every task starts with the same defaults (3 examples per call, 8192 tokens). If output is truncated, the system reduces to 1 example per call and increases the token budget. Bulk generation inherits whatever the pilot learned.

**Diagnosis flows into generation.** The `diagnose` tool saves root causes per task. The `generate_data` tool auto-reads them and injects them into the generation prompt. The generated examples specifically avoid the diagnosed failure patterns.

**Gold data protection.** Training data is versioned on HuggingFace. On regression, the orchestrator can selectively restore specific tasks from the best-ever version while keeping improvements on other tasks. Tasks scoring <30% are exempt from the protection gate -- removing bad data from failing tasks is expected.

## API keys

| Key | Purpose |
|-----|---------|
| `ANTHROPIC_API_KEY` | Orchestrator, data generation, LLM judge |
| `OPENROUTER_API_KEY` | PinchBench judge (uses OpenRouter, not Anthropic) |
| `BRAVE_API_KEY` | Web search tasks |
| `HF_TOKEN` | Dataset backup to HuggingFace |

## Project structure

```
orchestrator.py              # main entry point
config.yaml                  # all settings
prompts/orchestrator.md      # orchestrator behavior (edit this to change decisions)
prompts/diagnose.md          # failure analysis prompt

tools/                       # 22 tools the orchestrator can call
  registry.py                # schemas + dispatch
  data_tools.py              # generate, score, filter, validate, snapshot, ...
  training_tools.py          # train, convert, register, benchmark
  reasoning_tools.py         # diagnose, plan_strategy

datagen/                     # data generation
  dynamic_gen.py             # pilot-validate-refine + self-healing + adversarial mode
  task_loader.py             # reads task definitions from PinchBench .md files
  deep_validate.py           # semantic validation (Claude checks if data teaches correct behavior)
  validate_data.py           # structural validation (tool names, args, schemas)

stages/                      # training pipeline
  finetune.py                # LoRA fine-tuning with Unsloth
  convert.py                 # merge + quantize to GGUF
  prepare.py                 # convert to SFT format
```

## Lessons learned

1. **Modelfile template > data.** Explicit Qwen3 chat template doubled scores from 25% to 50%. Same weights, same data.
2. **Sequence length truncation is silent.** Training at 4096 tokens cut off multi-step tasks. The model learned "read files and stop." Increasing to 8192 fixed 4 tasks from 0% to 88-98%.
3. **Bad data is worse than no data.** Wrong tool names or filenames actively destroy capabilities. 50 validated examples beat 120 unvalidated ones.
4. **Diagnose before generating.** Blind data generation caused the v19-v22 regression cycle. The orchestrator now blocks generation until it reads the benchmark transcript.
5. **Data imbalance causes forgetting.** 118 examples on one task + 27 on another = the model forgets the small tasks.

## License

MIT (code), CC BY 4.0 (dataset)
