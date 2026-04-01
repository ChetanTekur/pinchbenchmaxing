# PinchBench Maxing

Autonomous fine-tuning system that trains an open-source LLM to compete on [PinchBench](https://pinchbench.com) -- a 23-task benchmark for AI agents. A Claude-powered orchestrator diagnoses failures, generates targeted training data, fine-tunes, benchmarks, and iterates. You start it before bed and wake up to a better model.

13% -> 81% so far. Target: 85%.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## How it works

The orchestrator is a stateful Claude agent in a multi-turn conversation. Each turn it sees benchmark scores, dataset stats, and budget, then calls one of 22 tools. No hardcoded pipeline. It decides what to do based on what it sees.

```
                        ┌─────────────────────┐
                        │  Claude Sonnet 4.6  │
                        │                     │
                        │  Sees: scores,      │
                        │  data stats, budget, │
                        │  action history     │
                        └────────┬────────────┘
                                 │
                            one tool call
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   ┌────▼─────┐           ┌──────▼──────┐          ┌─────▼──────┐
   │ Analyze  │           │  Fix Data   │          │   Train    │
   │          │           │             │          │            │
   │ read     │           │ generate    │          │ train      │
   │ transcript│──diag──▶│ validate    │──ready──▶│ convert    │
   │ diagnose │  flows   │ score       │  to      │ register   │
   │          │  into    │ filter      │  train   │ benchmark  │──▶ repeat
   └──────────┘  gen     │ snapshot    │          └────────────┘
                         └─────────────┘
```

The orchestrator manages the full loop autonomously. It reads benchmark transcripts to understand *why* tasks fail, generates targeted data to fix them, validates the data before training, and benchmarks the result. If a regression happens, it restores gold data and tries a different approach.

**Behavior is defined in markdown** (`prompts/orchestrator.md`), not Python. To change how it thinks, edit the prompt and restart.

### Guardrails

The orchestrator has hard gates to prevent the failure modes we hit during development:

- **Diagnosis gate** -- generation is blocked until the orchestrator reads benchmark transcripts or runs diagnosis. This prevents blind data generation, which caused the v19-v22 regression cycle.
- **Gold integrity gate** -- training is blocked if working tasks (>=30% score) lost significant data vs the best-ever version. Failing tasks (<30%) are exempt because removing bad data is expected.
- **Data quality gate** -- training requires >=90% clean examples. Examples with missing required tools are auto-removed.
- **Self-healing generation** -- if output is truncated, the system reduces examples-per-call and increases the token budget automatically. Bulk generation inherits whatever parameters the pilot learned.
- **Auto disk cleanup** -- training auto-removes old Ollama models (~5.6GB each) when disk is low.

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

**Diagnosis drives generation.** Early versions generated data blindly -- "task_21 scores 20%, generate more examples." This made things worse because the existing data taught wrong behavior (wrong tool names, missing required steps). Adding more bad data on top of bad data destroyed capabilities across v19-v22.

Now diagnosis is mandatory. The orchestrator reads the raw benchmark transcript to see exactly what the model did wrong, then the `diagnose` tool saves per-task root causes to a JSON file. The `generate_data` tool auto-reads this and injects it into the generation prompt: "Root cause: model uses run_python instead of read_file. Generate examples that use read_file." The generated examples specifically avoid the diagnosed failure patterns.

**Pilot-validate-refine.** Getting data generation right was the hardest part. The first approach (batch-generate hundreds of examples, hope they're good) produced data with wrong filenames, missing tools, and truncated responses that silently poisoned the model. We added a three-step loop:

1. Generate 3 pilot examples via the direct API
2. Claude validates them against the actual benchmark task definition (semantic check)
3. If bad: feed the issues back into the prompt, retry up to 3 times. If good: bulk generate the rest.

This catches problems before they enter the dataset. The pilot also self-heals: if output is truncated, it reduces examples-per-call from 3 to 1 and increases the token budget. Bulk generation inherits whatever parameters the pilot settled on, so it doesn't re-discover the same truncation issues.

**Gold data protection.** Training data is versioned on HuggingFace. On regression, the orchestrator selectively restores specific tasks from the best-ever version while keeping improvements on other tasks. This prevents the "scorched earth" failure mode where reverting one regression throws away gains on five other tasks.

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
