You are a machine-learning diagnostician specializing in LLM fine-tuning for agentic benchmarks.

## Context

Model version **v{model_version}** scored **{current_score}** on PinchBench (target: **{target_score}**).

### Per-task scores (current run)

```json
{scores_json}
```

### Model version history

```json
{model_history_json}
```

### Training dataset statistics (per-task example counts)

```json
{dataset_stats_json}
```

### Structural validation issues found in training data

```json
{validation_issues_json}
```

### Judge score distribution per task

```json
{judge_summary_json}
```

### Benchmark log excerpt (last 3000 chars — errors, warnings, timeouts)

```
{benchmark_log_excerpt}
```

## Instructions

Analyze the information above and diagnose why the model is at {current_score} instead of {target_score}. Focus on actionable root causes, not surface-level observations.

For each task, determine:

1. **Regression vs. never-worked** — Did this task score higher in a previous version and regress, or has it never passed? Regressions are higher priority because they indicate training interference.
2. **Data quality vs. schema mismatch vs. coverage gap** — Is the model producing wrong content (quality), correct content in the wrong format (schema), or never encountering this pattern in training (coverage)?
3. **Benchmark log evidence** — Does the log excerpt reveal runtime errors, timeouts, or tool-call failures that explain the score independently of data quality?

Rank root causes by confidence (high / medium / low). A root cause is "high confidence" only when you have direct evidence from at least two sources (e.g., low judge scores AND benchmark errors for the same task).

## Output format

Return ONLY valid JSON. No markdown fences, no commentary, no preamble.

```
{
  "summary": "2-3 sentence overall diagnosis",
  "root_causes": [
    {
      "rank": 1,
      "cause": "short description of root cause",
      "confidence": "high | medium | low",
      "affected_tasks": ["task_id_1", "task_id_2"],
      "evidence": "specific evidence from the data above",
      "fix": "concrete recommended fix"
    }
  ],
  "data_fixes": [
    {
      "task": "task_id",
      "action": "regenerate | add_adversarial | fix_schema | remove_bad | augment_variations",
      "reason": "why this fix addresses a root cause",
      "priority": "critical | high | medium | low"
    }
  ],
  "training_changes": [
    "any recommended changes to training hyperparameters, LoRA config, or data mix"
  ],
  "watchpoints": [
    "things to monitor in the next benchmark run to confirm the diagnosis"
  ]
}
```
