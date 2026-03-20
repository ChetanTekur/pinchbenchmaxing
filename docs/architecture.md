# Architecture

PinchBench Maxing uses a **Claude-powered orchestrator** to autonomously improve an LLM's benchmark score. The orchestrator examines the current state, decides what to do next, and calls tools to execute actions.

## Two Layers

### Layer 1: Orchestrator (decides WHAT to do)

The orchestrator is a Claude agent that receives:
- Current state (model version, scores, dataset stats, budget)
- Action history (what was already tried this session)
- Available tools

It takes **one action per turn**, then gets called again with the updated state. This continues until it returns `DONE` or hits a guardrail (budget, max actions, catastrophic regression).

Each turn is a **single Claude API call** (~$0.03). No conversation history accumulates — the "memory" is the action history persisted in `loop_state.json`. This means zero risk of context compaction errors.

### Layer 2: Reasoning Tools (deeper analysis)

Two tools use Claude internally for deeper reasoning:
- **diagnose** — analyzes benchmark results, forms hypotheses about failures, produces structured root causes
- **plan_strategy** — takes the diagnosis and dataset stats, produces a specific data generation plan

These are more expensive (~$0.15-0.20 each) but provide the analysis quality that makes the orchestrator's decisions good.

## Tool Categories

### Data Tools
| Tool | What | Cost |
|------|------|------|
| inspect_data | Dataset stats (counts, balance, quality) | Free |
| generate_data | Generate targeted training examples | $5-15 |
| generate_adversarial | Generate from benchmark failure transcripts | $1-3 |
| score_data | Score all examples 1-5 via LLM judge | $2-5 |
| filter_data | Remove examples below score threshold | Free |
| repair_data | Fix borderline examples (score 2-3) | $0.50 |
| dedup_data | Remove semantically similar examples | Free |
| rebalance_data | Trim overweight tasks | Free |
| snapshot | Save dataset state before destructive ops | Free |
| push_hf | Push to HuggingFace | Free |

### Training Tools
| Tool | What | Cost |
|------|------|------|
| validate_model | Check base model on HuggingFace | Free |
| train | Fine-tune with Unsloth LoRA | $10-30 (GPU) |
| convert | Merge + quantize to GGUF | Free |
| register | Register GGUF in Ollama | Free |

### Eval Tools
| Tool | What | Cost |
|------|------|------|
| benchmark | Run PinchBench (23 tasks) | $1-3 |
| check_disk | Return disk space | Free |

### Reasoning Tools
| Tool | What | Cost |
|------|------|------|
| diagnose | Deep failure analysis with Claude | $0.15 |
| plan_strategy | Data generation planning with Claude | $0.10 |

### Control Tools
| Tool | What |
|------|------|
| get_state | Return full state |
| request_approval | Pause for human input |

## Guardrails

- **Budget cap**: stops when remaining budget < $5
- **Max actions**: 20 per session
- **Regression detection**: pauses if score drops >10% from best ever
- **Dataset protection**: pauses if dataset drops below 500 examples
- **Disk safety**: checks disk before training or GGUF conversion
- **Snapshot before destruction**: always snapshots before filter/dedup/rebalance
- **Balance enforcement**: never lets any task exceed max_total_per_task

## State

All state persists in `loop_state.json`:
- Model version and history
- Per-task benchmark scores
- Dataset generation version tracking
- Action history (compressed after 10 actions)
- Budget spent

The orchestrator rebuilds full context from this file each turn. Crashes are safe — resume with the same command.

## Fallback

`python loop.py run --mode pipeline` runs the original fixed-sequence pipeline for predictable behavior when agentic mode isn't appropriate.
