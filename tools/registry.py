"""
Tool registry for PinchBench Maxing agent.

Defines all tool schemas in Anthropic tool_use format and dispatches
execution to the appropriate implementation module.
"""

from .data_tools import (
    inspect_data, generate_data, generate_adversarial,
    score_data, filter_data, repair_data,
    dedup_data, rebalance_data, snapshot, push_hf,
    validate_data,
)
from .training_tools import (
    train, convert, register, validate_model, benchmark, check_disk,
)
from .reasoning_tools import diagnose, plan_strategy
from .eval_tools import get_state, request_approval, write_note


# ── Tool Schemas (Anthropic tool_use format) ─────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "inspect_data",
        "description": (
            "Inspect the training dataset. Returns total example count, "
            "per-task counts, balance ratio, and lists of overweight/underweight tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "diagnose",
        "description": (
            "Analyze benchmark results and training data to diagnose why "
            "the model is underperforming. Calls Claude to produce a structured "
            "diagnosis with root causes and recommended data fixes. "
            "Optionally reads a benchmark log file for error context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "benchmark_log_path": {
                    "type": "string",
                    "description": "Path to a benchmark log file. If null, uses the most recent log.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "plan_strategy",
        "description": (
            "Given a diagnosis dict, produce a concrete data-generation plan "
            "with per-task example counts, strategies (topup/adversarial/both), "
            "and variation weights. Calls Claude for intelligent allocation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "object",
                    "description": "Diagnosis dict from the diagnose tool.",
                },
            },
            "required": ["diagnosis"],
        },
    },
    {
        "name": "generate_data",
        "description": (
            "Generate targeted training data for specified tasks using "
            "the targeted topup script. Supports diagnosis-aware generation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of task IDs to generate data for.",
                },
                "min_per_task": {
                    "type": "integer",
                    "description": "Minimum examples to generate per task.",
                    "default": 10,
                },
                "diagnosis_file": {
                    "type": "string",
                    "description": "Path to diagnosis JSON file for context-aware generation. Optional.",
                },
            },
            "required": ["tasks"],
        },
    },
    {
        "name": "generate_adversarial",
        "description": (
            "Generate adversarial training examples derived from benchmark "
            "failure transcripts. Best for score-0 tasks and recurring failure patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of task IDs to generate adversarial data for.",
                },
                "n_per_task": {
                    "type": "integer",
                    "description": "Number of adversarial examples per task.",
                    "default": 10,
                },
            },
            "required": ["tasks"],
        },
    },
    {
        "name": "score_data",
        "description": (
            "Run the LLM judge on all unscored examples in train.jsonl. "
            "Scores each example 1-5 and writes to scores.json."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "filter_data",
        "description": (
            "Filter training data, removing examples below the minimum judge score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_score": {
                    "type": "integer",
                    "description": "Minimum score threshold (1-5). Examples below this are removed.",
                    "default": 3,
                },
            },
            "required": [],
        },
    },
    {
        "name": "repair_data",
        "description": (
            "Attempt to repair borderline examples (those between min_score and max_score) "
            "by re-generating them with targeted prompts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_score": {
                    "type": "integer",
                    "description": "Lower bound of score range to repair.",
                    "default": 2,
                },
                "max_score": {
                    "type": "integer",
                    "description": "Upper bound of score range to repair.",
                    "default": 3,
                },
            },
            "required": [],
        },
    },
    {
        "name": "dedup_data",
        "description": (
            "Deduplicate semantically similar examples in the training set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold (0.0-1.0). Higher = more aggressive dedup.",
                    "default": 0.85,
                },
            },
            "required": [],
        },
    },
    {
        "name": "rebalance_data",
        "description": (
            "Rebalance the dataset by trimming overweight tasks to a target count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "integer",
                    "description": "Maximum examples per task after rebalancing.",
                    "default": 120,
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "snapshot",
        "description": (
            "Create a timestamped snapshot of train.jsonl and val.jsonl "
            "in data/snapshots/ for rollback safety."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Human-readable label for this snapshot (e.g. 'pre-filter', 'v3-ready').",
                },
            },
            "required": ["label"],
        },
    },
    {
        "name": "benchmark",
        "description": (
            "Run PinchBench benchmark against a model via OpenClaw/Ollama. "
            "Returns per-task scores and the log file path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Ollama model name (e.g. 'qwen35-9b-clawd-v3'). Will be prefixed with 'ollama/'.",
                },
            },
            "required": ["model_name"],
        },
    },
    {
        "name": "validate_data",
        "description": (
            "Run comprehensive data quality validation: check tool call schemas, "
            "argument names, required tools per task, repetition patterns, truncation. "
            "Returns counts of clean/issues/critical. MUST be called before train — "
            "never train on data with critical or high severity issues."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fix": {
                    "type": "boolean",
                    "description": "If true, remove examples with critical/high issues.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "train",
        "description": (
            "Fine-tune the model: prepare SFT data, run Unsloth LoRA training. "
            "Uses the current train.jsonl/val.jsonl. "
            "PRE-REQUISITES: validate_data must pass, all tasks must have ≥20 examples, check_disk must show enough space."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "version": {
                    "type": "integer",
                    "description": "Version number for this training run (e.g. 3 for v3).",
                },
            },
            "required": ["version"],
        },
    },
    {
        "name": "convert",
        "description": (
            "Convert a fine-tuned model to GGUF format for Ollama serving."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "version": {
                    "type": "integer",
                    "description": "Model version number to convert.",
                },
            },
            "required": ["version"],
        },
    },
    {
        "name": "register",
        "description": (
            "Register a GGUF model in Ollama with the correct chat template "
            "and tool-calling support."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "version": {
                    "type": "integer",
                    "description": "Model version number.",
                },
                "model_name": {
                    "type": "string",
                    "description": "Ollama model name to register as.",
                },
            },
            "required": ["version", "model_name"],
        },
    },
    {
        "name": "validate_model",
        "description": (
            "Validate that the base model is suitable for fine-tuning. "
            "Checks HuggingFace existence, architecture, Unsloth support, tokenizer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "check_disk",
        "description": (
            "Check available disk space on workspace and root partitions. "
            "Returns free space in GB and a warning flag if low."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "push_hf",
        "description": (
            "Push the current dataset (train.jsonl, val.jsonl, scores.json) "
            "to HuggingFace for versioning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message for the HuggingFace push.",
                },
            },
            "required": ["message"],
        },
    },
    {
        "name": "get_state",
        "description": (
            "Return the full current agent state dict including scores, "
            "model version, history, weak tasks, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "write_note",
        "description": (
            "Write a note to your scratchpad. Notes persist across turns and are "
            "shown at the start of every turn. Use this to record learnings, "
            "track what failed and why, and remind yourself what to do next. "
            "Costs nothing — use freely."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to save. Be specific — include error names, versions, task IDs.",
                },
            },
            "required": ["note"],
        },
    },
    {
        "name": "request_approval",
        "description": (
            "Pause execution and request human approval before proceeding "
            "with a potentially expensive or destructive operation. "
            "Prints the reason to stdout and waits for yes/no input."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Human-readable explanation of what needs approval and why.",
                },
            },
            "required": ["reason"],
        },
    },
]

# ── Dispatch table ───────────────────────────────────────────────────────────

_DISPATCH = {
    "inspect_data":         inspect_data,
    "diagnose":             diagnose,
    "plan_strategy":        plan_strategy,
    "generate_data":        generate_data,
    "generate_adversarial": generate_adversarial,
    "score_data":           score_data,
    "filter_data":          filter_data,
    "repair_data":          repair_data,
    "dedup_data":           dedup_data,
    "rebalance_data":       rebalance_data,
    "validate_data":        validate_data,
    "snapshot":             snapshot,
    "benchmark":            benchmark,
    "train":                train,
    "convert":              convert,
    "register":             register,
    "check_disk":           check_disk,
    "validate_model":       validate_model,
    "push_hf":              push_hf,
    "get_state":            get_state,
    "request_approval":     request_approval,
    "write_note":           write_note,
}


def execute_tool(name: str, args: dict, cfg, state) -> dict:
    """
    Dispatch a tool call to the appropriate implementation.

    Returns a dict with at minimum:
      {"status": "success"|"error", "result": {...}}
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"status": "error", "error": f"Unknown tool: {name}"}
    try:
        return fn(args, cfg, state)
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
