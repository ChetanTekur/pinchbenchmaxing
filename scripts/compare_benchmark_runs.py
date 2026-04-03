#!/usr/bin/env python3
"""Compare per-task scores from variance test against original leaderboard submissions.

Usage:
  python scripts/compare_benchmark_runs.py
"""
import json
import os
from pathlib import Path

WS = os.environ.get("PBM_WORKSPACE", "/workspace/synthbench")
VARIANCE_DIR = Path(WS) / "logs" / "variance_test"

# Original leaderboard scores (from PinchBench submissions)
LEADERBOARD = {
    "v21": {  # 77% - submission b688f5d5
        "task_00_sanity": 1.00, "task_01_calendar": 0.67, "task_02_stock": 1.00,
        "task_03_blog": 0.93, "task_04_weather": 1.00, "task_05_summary": 0.94,
        "task_06_events": 0.45, "task_07_email": 0.96, "task_08_memory": 0.80,
        "task_09_files": 0.86, "task_10_workflow": 0.71, "task_11_config_update": 1.00,
        "task_12_skill_search": 1.00, "task_13_image_gen": 0.46, "task_14_humanizer": 0.55,
        "task_15_daily_summary": 0.88, "task_16_email_triage": 0.92,
        "task_17_email_search": 1.00, "task_18_market_research": 0.92,
        "task_19_spreadsheet_summary": 0.53, "task_20_eli5_pdf": 0.04,
        "task_21_openclaw_comprehension": 0.22, "task_22_second_brain": 0.95,
    },
}


def load_variance_scores(model_name):
    """Load per-task scores from variance test runs."""
    runs = {}
    for f in sorted(VARIANCE_DIR.glob(f"{model_name}_run*.scores.json")):
        run_num = f.stem.split("_run")[1].split(".")[0]
        try:
            runs[f"run{run_num}"] = json.loads(f.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return runs


def main():
    print(f"\n{'='*80}")
    print(f"  BENCHMARK SCORE COMPARISON: Original Leaderboard vs Current Runs")
    print(f"{'='*80}\n")

    for version, original in LEADERBOARD.items():
        model_name = f"qwen35-9b-clawd-{version}"
        runs = load_variance_scores(model_name)

        if not runs:
            print(f"  No variance test data for {model_name}\n")
            continue

        # Header
        orig_avg = sum(original.values()) / len(original)
        print(f"  {model_name} (original leaderboard: {orig_avg*100:.1f}%)")
        print()

        header = f"  {'Task':<40} {'Orig':>6}"
        for run_name in sorted(runs.keys()):
            avg = sum(runs[run_name].values()) / len(runs[run_name]) if runs[run_name] else 0
            header += f" {run_name + f' ({avg*100:.0f}%)':>14}"
        header += f" {'Dropped?':>10}"
        print(header)
        print(f"  {'-'*40} {'-'*6}" + f" {'-'*14}" * len(runs) + f" {'-'*10}")

        all_tasks = sorted(set(list(original.keys()) + [t for r in runs.values() for t in r]))
        dropped = []

        for task in all_tasks:
            orig = original.get(task, 0)
            row = f"  {task:<40} {orig:>6.2f}"

            for run_name in sorted(runs.keys()):
                curr = runs[run_name].get(task, 0)
                row += f" {curr:>14.2f}"

            # Check if consistently dropped
            run_scores = [runs[r].get(task, 0) for r in runs]
            avg_now = sum(run_scores) / len(run_scores) if run_scores else 0
            delta = avg_now - orig

            if delta < -0.3:
                row += f" {'YES (' + f'{delta:+.2f}' + ')':>10}"
                dropped.append((task, orig, avg_now, delta))
            elif delta < -0.1:
                row += f" {'maybe':>10}"
            else:
                row += f" {'':>10}"

            print(row)

        print()
        if dropped:
            total_lost = sum(d[3] for d in dropped)
            print(f"  MAJOR DROPS (>{0.3:.0%} decline):")
            for task, orig, now, delta in sorted(dropped, key=lambda x: x[3]):
                print(f"    {task:<40} {orig:.2f} -> {now:.2f} ({delta:+.2f})")
            print(f"\n  Total score lost from major drops: {total_lost:+.1f} points")
            print(f"  This explains {abs(total_lost)/orig_avg*100:.0f}% of the regression")
        print()


if __name__ == "__main__":
    main()
