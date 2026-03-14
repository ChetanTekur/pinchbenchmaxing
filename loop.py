#!/usr/bin/env python3
"""
AutoResearch loop: benchmark → analyze → generate data → finetune → repeat

Usage:
  python loop.py run --scores '{"task_09_files": 0.0, "task_22_second_brain": 0.1}'
  python loop.py run --log /tmp/bench_finetuned.log
  python loop.py status
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils.config import load_config

# ─────────────────────────────────────────────────────────────────────────────
# TASK ID LIST  (must match generate.py)
# ─────────────────────────────────────────────────────────────────────────────
TASK_IDS = [
    "task_00_sanity",
    "task_01_calendar",
    "task_02_stock",
    "task_03_blog",
    "task_04_weather",
    "task_05_summary",
    "task_06_events",
    "task_07_email",
    "task_08_memory",
    "task_09_files",
    "task_10_workflow",
    "task_11_config_update",
    "task_12_skill_search",
    "task_13_image_gen",
    "task_14_humanizer",
    "task_15_daily_summary",
    "task_16_email_triage",
    "task_17_email_search",
    "task_18_market_research",
    "task_19_spreadsheet_summary",
    "task_20_eli5_pdf",
    "task_21_openclaw_comprehension",
    "task_22_second_brain",
]

# ─────────────────────────────────────────────────────────────────────────────
# HARD TASKS  (use EXAMPLES_PER_CALL=1 to avoid truncation)
# ─────────────────────────────────────────────────────────────────────────────
HARD_TASKS = {
    "task_10_workflow",
    "task_12_skill_search",
    "task_15_daily_summary",
    "task_16_email_triage",
    "task_17_email_search",
    "task_18_market_research",
    "task_21_openclaw_comprehension",
}


# ─────────────────────────────────────────────────────────────────────────────
# SCORE PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_scores_from_log(log_path: str) -> dict[str, float]:
    """
    Parse per-task scores from a benchmark log file.

    Looks for patterns like:
      task_09_files: 0.0/1.0
      task_09_files  0.85
      "task_09_files": 0.85
    Returns a dict {task_id: score_0_to_1}.
    """
    scores: dict[str, float] = {}
    text = Path(log_path).read_text(errors="replace")

    # Pattern: task_XX_name followed by a float (optionally /1.0)
    pattern = re.compile(
        r'(task_\d{2}_\w+)["\s:]*\s+([01](?:\.\d+)?)\s*(?:/\s*1\.0)?'
    )
    for m in pattern.finditer(text):
        task_id = m.group(1)
        score = float(m.group(2))
        if task_id in TASK_IDS:
            scores[task_id] = score

    # Also try JSON objects embedded in the log
    json_pattern = re.compile(r'\{[^{}]*"task_\d{2}_\w+"[^{}]*\}')
    for blob in json_pattern.findall(text):
        try:
            obj = json.loads(blob)
            for k, v in obj.items():
                if k in TASK_IDS and isinstance(v, (int, float)):
                    scores[k] = float(v)
        except json.JSONDecodeError:
            pass

    return scores


def parse_scores_from_json_str(json_str: str) -> dict[str, float]:
    """Parse task scores from a JSON string argument."""
    raw = json.loads(json_str)
    scores: dict[str, float] = {}
    for k, v in raw.items():
        # Accept both full task IDs and short names
        if k in TASK_IDS:
            scores[k] = float(v)
        else:
            # Try prefix match: "task_09" → "task_09_files"
            matched = [t for t in TASK_IDS if t.startswith(k)]
            if len(matched) == 1:
                scores[matched[0]] = float(v)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGES
# ─────────────────────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], env: dict | None = None, check: bool = True) -> int:
    """Run a subprocess command, streaming output to stdout."""
    merged_env = {**os.environ, **(env or {})}
    print(f"\n[loop] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=merged_env)
    if check and result.returncode != 0:
        print(f"[loop] ERROR: command exited with code {result.returncode}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def topup_weak_tasks(weak_tasks: list[str], cfg) -> None:
    """Run topup.py submit/collect for each weak task."""
    # Split into hard vs normal to set EXAMPLES_PER_CALL correctly
    normal_tasks = [t for t in weak_tasks if t not in HARD_TASKS]
    hard_tasks   = [t for t in weak_tasks if t in HARD_TASKS]

    for task_group, epc in [(normal_tasks, "3"), (hard_tasks, "1")]:
        if not task_group:
            continue
        task_str = ",".join(task_group)
        env = {"TOPUP_TASKS": task_str, "EXAMPLES_PER_CALL": epc}
        run_cmd([sys.executable, "topup.py", "run", "--tasks", task_str], env=env)


def run_llm_judge(cfg) -> None:
    min_score = cfg.data.min_judge_score
    run_cmd([sys.executable, "llm_judge.py", "filter", "--min", str(min_score)])


def run_prepare(cfg) -> None:
    run_cmd([sys.executable, "-m", "stages.prepare"])


def run_finetune(cfg) -> None:
    run_cmd([sys.executable, "-m", "stages.finetune"])


def run_convert(cfg) -> None:
    run_cmd([sys.executable, "-m", "stages.convert"])


def run_fix_modelfile(cfg) -> None:
    script = Path(__file__).parent / "scripts" / "fix_modelfile.sh"
    run_cmd(["bash", str(script)])


def run_benchmark(cfg) -> Path:
    """Run benchmark and return the log file path."""
    script = Path(__file__).parent / "scripts" / "benchmark_run.sh"
    model = f"ollama/{cfg.model_name}"
    run_cmd(["bash", str(script), model])
    safe_name = model.replace("/", "_").replace(":", "_")
    return Path(f"/tmp/bench_{safe_name}.log")


# ─────────────────────────────────────────────────────────────────────────────
# LOOP STATE
# ─────────────────────────────────────────────────────────────────────────────

def load_state(state_file: Path) -> dict:
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {"iteration": 0, "history": []}


def save_state(state: dict, state_file: Path) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))
    print(f"[loop] State saved to {state_file}")


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_status(args, cfg) -> None:
    state_file = cfg.data_dir / "loop_state.json"
    if not state_file.exists():
        print("No loop state found. Run 'python loop.py run' to start.")
        return

    state = load_state(state_file)
    print(f"Current iteration: {state['iteration']}")
    for entry in state.get("history", []):
        ts = entry.get("timestamp", "?")
        n_weak = entry.get("n_weak_tasks", "?")
        avg = entry.get("avg_score", "?")
        print(f"  [{ts}] iter={entry['iteration']}  avg_score={avg}  weak_tasks={n_weak}")


def cmd_run(args, cfg) -> None:
    # ── 1. Load / init state ─────────────────────────────────────────────────
    state_file = cfg.data_dir / "loop_state.json"
    state = load_state(state_file)

    # ── 2. Parse scores ──────────────────────────────────────────────────────
    if args.scores:
        scores = parse_scores_from_json_str(args.scores)
        print(f"[loop] Loaded {len(scores)} task scores from --scores argument.")
    elif args.log:
        scores = parse_scores_from_log(args.log)
        print(f"[loop] Parsed {len(scores)} task scores from log: {args.log}")
    elif state.get("history"):
        # Resume from last iteration's scores
        last = state["history"][-1]
        scores = last.get("scores", {})
        print(f"[loop] Resumed scores from iteration {last['iteration']} ({len(scores)} tasks).")
    else:
        # No history and no scores — run benchmark first to get baseline
        print("[loop] No prior scores found. Running benchmark to get baseline scores...")
        log_file = run_benchmark(cfg)
        scores = parse_scores_from_log(str(log_file))
        if scores:
            print(f"[loop] Parsed {len(scores)} task scores from benchmark log.")
        else:
            print("[loop] WARNING: Could not parse scores from log. Treating all tasks as weak.")
            scores = {t: 0.0 for t in TASK_IDS}

    max_iter   = cfg.loop.max_iterations
    target     = cfg.loop.target_score
    threshold  = cfg.loop.weak_task_threshold

    # ── 3. Main loop ─────────────────────────────────────────────────────────
    for _ in range(max_iter):
        state["iteration"] += 1
        iteration = state["iteration"]
        print(f"\n{'='*60}")
        print(f"[loop] === ITERATION {iteration}/{max_iter} ===")
        print(f"{'='*60}")

        # Identify weak tasks — only among tasks that have been explicitly scored
        weak_tasks = [t for t in TASK_IDS if t in scores and scores[t] < threshold]
        avg_score  = sum(scores.values()) / len(scores) if scores else 0.0

        print(f"[loop] Average score: {avg_score:.3f}  |  Target: {target}")
        print(f"[loop] Weak tasks ({len(weak_tasks)}): {weak_tasks}")

        # Record entry
        entry = {
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "scores": dict(scores),
            "avg_score": round(avg_score, 4),
            "weak_tasks": weak_tasks,
            "n_weak_tasks": len(weak_tasks),
        }

        if avg_score >= target:
            print(f"[loop] Target score {target} reached! Stopping.")
            entry["status"] = "target_reached"
            state["history"].append(entry)
            save_state(state, state_file)
            break

        if not weak_tasks:
            print("[loop] No weak tasks found. All tasks above threshold.")
            entry["status"] = "no_weak_tasks"
            state["history"].append(entry)
            save_state(state, state_file)
            break

        entry["status"] = "running"
        state["history"].append(entry)
        save_state(state, state_file)

        # Pipeline
        try:
            topup_weak_tasks(weak_tasks, cfg)
            run_llm_judge(cfg)
            run_prepare(cfg)
            run_finetune(cfg)
            run_convert(cfg)
            run_fix_modelfile(cfg)
            run_benchmark(cfg)

            # After benchmark, prompt user to provide new scores for next iteration
            print(
                "\n[loop] Benchmark complete. To continue the loop, re-run with "
                "new scores:\n"
                "  python loop.py run --scores '{\"task_XX_name\": 0.5, ...}'"
            )
            entry["status"] = "awaiting_scores"
            save_state(state, state_file)
            break

        except subprocess.CalledProcessError as exc:
            print(f"[loop] Pipeline failed at step: {exc.cmd}", file=sys.stderr)
            entry["status"] = "failed"
            entry["error"] = str(exc)
            save_state(state, state_file)
            sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="AutoResearch loop: benchmark → analyze → generate → finetune → repeat"
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ──────────────────────────────────────────────────────────────────
    run_parser = sub.add_parser("run", help="Run one loop iteration")
    score_group = run_parser.add_mutually_exclusive_group()
    score_group.add_argument(
        "--scores",
        type=str,
        help=(
            'JSON dict of task_id→score, e.g. '
            '\'{"task_09_files": 0.0, "task_22_second_brain": 0.1}\''
        ),
    )
    score_group.add_argument(
        "--log",
        type=str,
        help="Path to benchmark log file to parse scores from",
    )

    # ── status ────────────────────────────────────────────────────────────────
    sub.add_parser("status", help="Show current loop state")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args, cfg)
    elif args.command == "status":
        cmd_status(args, cfg)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
