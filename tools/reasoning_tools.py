"""
Reasoning tool implementations for PinchBench Maxing agent.

Tools: diagnose, plan_strategy

These call the Claude API directly (real-time, not batch) using prompt
templates from prompts/*.md with {var} template substitution.
"""

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import anthropic

from agents.base import log_print, _write_log

_PROJECT_ROOT = Path(__file__).parent.parent


# ── JSON extraction (robust, from eval_analysis_agent.py) ────────────────────

def _extract_json_object(text: str) -> dict | None:
    """Robustly extract the outermost JSON object from a Claude response."""
    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError as e:
        log_print(f"  [json] Direct parse failed: {e}")

    # Try brace-depth tracking — find ALL possible {} matches, not just first
    start = 0
    while True:
        idx = text.find('{', start)
        if idx == -1:
            break
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[idx:], idx):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[idx:i + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        pass
                    break
        start = idx + 1

    # Last resort: try to fix common issues (trailing comma, truncation)
    idx = text.find('{')
    if idx != -1:
        # Find the last } and try everything between
        ridx = text.rfind('}')
        if ridx > idx:
            candidate = text[idx:ridx + 1]
            # Remove trailing commas before } or ]
            candidate = re.sub(r',\s*}', '}', candidate)
            candidate = re.sub(r',\s*]', ']', candidate)
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    return None
    return None


def _load_prompt(name: str) -> str:
    """Load a prompt template from prompts/{name}.md."""
    path = _PROJECT_ROOT / "prompts" / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text()


def _collect_dataset_stats(cfg) -> dict:
    """Collect dataset statistics by running inspect_data."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "datagen.inspect_data", "stats"],
            stderr=subprocess.DEVNULL, timeout=30, text=True,
        )
        by_task: dict[str, int] = {}
        for line in out.splitlines():
            m = re.match(r'\s+(task_\w+)\s+(\d+)', line)
            if m:
                by_task[m.group(1)] = int(m.group(2))
        return by_task
    except Exception:
        return {}


def _collect_validation_issues(cfg) -> dict:
    """Collect structural validation issues."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "datagen.inspect_data", "validate"],
            stderr=subprocess.DEVNULL, timeout=90, text=True,
        )
        by_type: dict[str, int] = {}
        for line in out.splitlines():
            m = re.match(r'\s+(\d+).\s+(.+)', line)
            if m:
                by_type[m.group(2).strip()] = int(m.group(1))
        return by_type
    except Exception:
        return {}


def _collect_judge_summary(cfg) -> dict:
    """Collect LLM judge score summary."""
    scores_file = cfg.data_dir / "scores.json"
    if not scores_file.exists():
        return {}
    try:
        scores = json.loads(scores_file.read_text())
        by_task: dict[str, list] = defaultdict(list)
        for v in scores.values():
            s = v.get("score", 0)
            if s > 0:
                by_task[v.get("task_id", "unknown")].append(s)
        return {t: round(sum(s) / len(s), 2) for t, s in by_task.items() if s}
    except Exception:
        return {}


def _read_benchmark_log(cfg, benchmark_log_path: str | None = None) -> str:
    """Read the most recent benchmark log excerpt."""
    if benchmark_log_path:
        p = Path(benchmark_log_path)
        if p.exists():
            text = p.read_text(errors="replace")
            return text[-3000:]

    logs_dir = cfg.data_dir.parent / "logs"
    if not logs_dir.exists():
        return ""

    log_files = sorted(
        logs_dir.glob("bench_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if log_files:
        text = log_files[0].read_text(errors="replace")
        return text[-3000:]
    return ""


# ── diagnose ─────────────────────────────────────────────────────────────────

def diagnose(args: dict, cfg, state) -> dict:
    """Analyze benchmark results and training data to diagnose underperformance."""
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {"status": "error", "error": "ANTHROPIC_API_KEY not set"}

        benchmark_log_path = args.get("benchmark_log_path")

        # Collect all signals
        log_print("  [diagnose] Collecting signals...")
        dataset_stats = _collect_dataset_stats(cfg)
        validation_issues = _collect_validation_issues(cfg)
        judge_summary = _collect_judge_summary(cfg)
        benchmark_excerpt = _read_benchmark_log(cfg, benchmark_log_path)

        # Collect bad examples detail (actual tool calls vs expectations)
        bad_examples_summary = ""
        bad_report_file = cfg.data_dir / "bad_examples_report.json"
        if bad_report_file.exists():
            try:
                bad_examples = json.loads(bad_report_file.read_text())
                # Group by task and summarize
                from collections import Counter
                bad_by_task = defaultdict(list)
                for ex in bad_examples:
                    bad_by_task[ex["task_id"]].append(ex)
                parts = []
                for tid, exs in sorted(bad_by_task.items(), key=lambda x: -len(x[1])):
                    issue_types = Counter(i["check"] for ex in exs for i in ex["issues"])
                    sample_tools = exs[0]["tool_calls"][:5] if exs else []
                    parts.append(f"  {tid} ({len(exs)} bad): issues={dict(issue_types)}, "
                                 f"sample_tools={json.dumps(sample_tools)}")
                bad_examples_summary = "\n".join(parts[:10])
            except Exception:
                pass

        # Collect validator expectations for comparison
        from datagen.validate_data import TOOL_SIGNATURES, REQUIRED_TOOLS
        validator_context = (
            f"TOOL_SIGNATURES (what validator expects):\n"
            f"{json.dumps({k: v for k, v in TOOL_SIGNATURES.items()}, indent=2)}\n\n"
            f"REQUIRED_TOOLS (must appear in examples):\n"
            f"{json.dumps(REQUIRED_TOOLS, indent=2)}"
        )

        # Load prompt template
        template = _load_prompt("diagnose")

        # Fill template variables
        try:
            target_score = cfg.loop.target_score
        except AttributeError:
            target_score = 0.85

        current_score = round(state.avg_score, 3) if state.scores else "N/A"

        # Safe variable substitution (doesn't break on literal {} in markdown)
        variables = {
            "model_version": str(state.model_version),
            "current_score": str(current_score),
            "target_score": str(target_score),
            "scores_json": json.dumps(state.scores, indent=2),
            "model_history_json": json.dumps(state.model_history, indent=2),
            "dataset_stats_json": json.dumps(dataset_stats, indent=2),
            "validation_issues_json": json.dumps(validation_issues, indent=2),
            "judge_summary_json": json.dumps(judge_summary, indent=2),
            "benchmark_log_excerpt": benchmark_excerpt,
            "bad_examples_summary": bad_examples_summary or "(no bad examples report found)",
            "validator_context": validator_context,
        }
        prompt = template
        for key, value in variables.items():
            prompt = prompt.replace("{" + key + "}", value)

        # Call Claude
        log_print("  [diagnose] Calling Claude for diagnosis...")
        analysis_model = cfg.claude.analysis
        client = anthropic.Anthropic(api_key=api_key)

        resp = client.messages.create(
            model=analysis_model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = resp.content[0].text.strip()
        diagnosis = _extract_json_object(raw)

        if diagnosis is None:
            log_print(f"  [diagnose] WARNING: Could not parse JSON ({len(raw)} chars)")
            # Save raw for debugging
            debug_dir = cfg.data_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / f"diagnose_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            debug_file.write_text(raw)
            log_print(f"  [diagnose] Raw saved to {debug_file}")
            diagnosis = {
                "summary": raw[:500],
                "root_causes": [],
                "data_fixes": [],
                "training_changes": [],
                "watchpoints": [],
            }

        # Estimate cost from usage
        input_tokens = getattr(resp.usage, 'input_tokens', 0)
        output_tokens = getattr(resp.usage, 'output_tokens', 0)
        cost_usd = round(input_tokens * 0.003 / 1000 + output_tokens * 0.015 / 1000, 4)

        log_print(f"  [diagnose] Summary: {diagnosis.get('summary', '')[:100]}")
        log_print(f"  [diagnose] Root causes: {len(diagnosis.get('root_causes', []))}")

        return {
            "status": "success",
            "result": {
                "summary": diagnosis.get("summary", ""),
                "root_causes": diagnosis.get("root_causes", []),
                "data_fixes": diagnosis.get("data_fixes", []),
                "training_changes": diagnosis.get("training_changes", []),
                "watchpoints": diagnosis.get("watchpoints", []),
            },
            "cost_usd": cost_usd,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── plan_strategy ────────────────────────────────────────────────────────────

def plan_strategy(args: dict, cfg, state) -> dict:
    """Produce a concrete data-generation plan based on a diagnosis."""
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {"status": "error", "error": "ANTHROPIC_API_KEY not set"}

        diagnosis = args.get("diagnosis", {})

        # Collect dataset stats
        dataset_stats = _collect_dataset_stats(cfg)

        # Load prompt template
        template = _load_prompt("plan_strategy")

        # Resolve config values
        try:
            target_score = cfg.loop.target_score
        except AttributeError:
            target_score = 0.85
        try:
            max_new_per_task = cfg.loop.max_new_per_task
        except AttributeError:
            max_new_per_task = 50
        try:
            max_total_per_task = cfg.loop.max_total_per_task
        except AttributeError:
            max_total_per_task = 120
        try:
            total_cap = cfg.loop.total_new_examples_cap
        except AttributeError:
            total_cap = 300

        # Safe variable substitution
        variables = {
            "target_score": str(target_score),
            "diagnosis_json": json.dumps(diagnosis, indent=2),
            "dataset_stats_json": json.dumps(dataset_stats, indent=2),
            "scores_json": json.dumps(state.scores, indent=2),
            "max_new_per_task": str(max_new_per_task),
            "max_total_per_task": str(max_total_per_task),
            "total_cap": str(total_cap),
        }
        prompt = template
        for key, value in variables.items():
            prompt = prompt.replace("{" + key + "}", value)

        # Call Claude
        log_print("  [plan_strategy] Calling Claude for strategy plan...")
        analysis_model = cfg.claude.analysis
        client = anthropic.Anthropic(api_key=api_key)

        resp = client.messages.create(
            model=analysis_model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = resp.content[0].text.strip()
        plan = _extract_json_object(raw)

        if plan is None:
            log_print(f"  [plan_strategy] WARNING: Could not parse JSON ({len(raw)} chars)")
            plan = {"plan": [], "total_examples": 0, "estimated_cost_usd": 0.0}

        # Estimate cost
        input_tokens = getattr(resp.usage, 'input_tokens', 0)
        output_tokens = getattr(resp.usage, 'output_tokens', 0)
        cost_usd = round(input_tokens * 0.003 / 1000 + output_tokens * 0.015 / 1000, 4)

        log_print(f"  [plan_strategy] Plan: {len(plan.get('plan', []))} tasks, "
                  f"{plan.get('total_examples', 0)} total examples")

        return {
            "status": "success",
            "result": {
                "plan": plan.get("plan", []),
                "total_examples": plan.get("total_examples", 0),
                "estimated_cost_usd": plan.get("estimated_cost_usd", 0.0),
            },
            "cost_usd": cost_usd,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
